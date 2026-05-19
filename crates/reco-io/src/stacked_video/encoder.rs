//! Stacked-video encoder. Wraps
//! [`crate::ffmpeg::encoder::VideoEncoder`] with a default
//! configuration suited to the replay-recording use case:
//! fragmented MP4 (readable while being written), YUV420P planar
//! pixel format (skips the RGBA scaler), and a software encoder
//! so it doesn't contend with the GPU encoder already running
//! the live stitch output.
//!
//! Consumers can override any of these via the [`StackedEncoderConfig`]
//! builder. Hardware-encoded stacked output would need an NV12
//! pack variant; not implemented in this cut (see the vault note
//! `architecture/stacked-video-replay-2026-04-19.md`).
use super::{GridLayout, StackError, pack_yuv420p};
use crate::ffmpeg::encoder::{
    Container, EncodeError, EncoderConfig, Quality, VideoCodec, VideoEncoder,
};
use reco_core::source::YuvFrame;
use std::path::Path;
use thiserror::Error;

/// Errors from stacked-video encoding. Flattens both layers
/// (pack and ffmpeg encode) so consumers pattern-match on one
/// type.
#[derive(Debug, Error)]
pub enum StackedEncodeError {
    /// Pack-layer error (tile dimensions, count, etc.).
    #[error("pack: {0}")]
    Pack(#[from] StackError),
    /// FFmpeg-layer error (codec open, mux, write).
    #[error("encode: {0}")]
    Encode(#[from] EncodeError),
}

/// Configuration for [`StackedEncoder`]. Derived from
/// [`EncoderConfig`] but with replay-friendly defaults:
/// Matroska container, auto-detect encoder, no audio passthrough.
/// Override any field as needed.
#[derive(Debug, Clone)]
pub struct StackedEncoderConfig {
    /// Inner encoder config. Container defaults to
    /// [`Container::Matroska`]; codec defaults to
    /// [`VideoCodec::H264`]; encoder auto-detects hardware
    /// (NVENC/AMF/VT) with software fallback. The encoder
    /// handles YUV420P-to-NV12 interleave automatically when
    /// a hardware encoder is selected.
    pub inner: EncoderConfig,
    /// Output frames-per-second (numerator, denominator).
    pub fps: (i32, i32),
}

impl Default for StackedEncoderConfig {
    fn default() -> Self {
        Self {
            inner: EncoderConfig {
                // Matroska is the replay-recording default:
                // streamable (readers can open mid-write),
                // crash-safe (partial files are always
                // recoverable), and OBS's own default. fMP4 is
                // also supported via `Container::Mp4Fragmented`
                // but currently trips a muxer/libx264 finalize
                // bug in ffmpeg 7 / ffmpeg-next 8 that we
                // haven't root-caused (write_trailer returns
                // AVERROR -162 even with the canonical
                // movflags recipe; plain MP4 and Matroska
                // both finalize cleanly with identical stream
                // setup). Flagged for follow-up; consumers
                // that explicitly opt in to fMP4 today will
                // hit it.
                container: Container::Matroska,
                codec: VideoCodec::H264,
                quality_preset: Quality::Fast,
                encoder_name: None,
                quality: None,
                preset: None,
                audio_source: None,
                // Short GOP so replay readers see recent
                // frames within ~1 second. For Matroska the
                // GOP controls cluster cadence; for fMP4 it
                // would control fragment cadence. 30 frames
                // at 30fps costs ~5-10% bitrate vs the libx264
                // default of 250.
                gop_size: Some(30),
                stream_url: None,
            },
            fps: (30, 1),
        }
    }
}

/// Stacked-video encoder. One instance per output file.
///
/// Not [`Sync`] - the underlying `ffmpeg` encoder owns raw
/// pointers and must stay on one thread.
pub struct StackedEncoder {
    layout: GridLayout,
    encoder: VideoEncoder,
}

impl StackedEncoder {
    /// Open a new stacked-video file for the given grid layout.
    ///
    /// The file's dimensions are
    /// `layout.packed_width() x layout.packed_height()`. All
    /// tile dimensions must be even (enforced by [`GridLayout`]
    /// construction).
    pub fn new(
        layout: GridLayout,
        output_path: &Path,
        config: StackedEncoderConfig,
    ) -> Result<Self, StackedEncodeError> {
        let fps = ffmpeg_next::Rational(config.fps.0, config.fps.1);
        let encoder = VideoEncoder::new(
            output_path,
            layout.packed_width(),
            layout.packed_height(),
            fps,
            &config.inner,
        )?;
        log::info!(
            "StackedEncoder: {}x{} grid ({} rows x {} cols, tile {}x{}) -> {} ({}, {})",
            layout.packed_width(),
            layout.packed_height(),
            layout.rows(),
            layout.cols(),
            layout.tile_width(),
            layout.tile_height(),
            output_path.display(),
            encoder.encoder_name(),
            match config.inner.container {
                Container::Mp4 => "mp4",
                Container::Mp4Fragmented => "fmp4",
                Container::Matroska => "mkv",
                Container::Flv => "flv",
            },
        );
        Ok(Self { layout, encoder })
    }

    /// Push one row-major tuple of tiles. `tiles[i]` is either
    /// `Some(&YuvFrame)` or `None` for an empty tile (zero-filled
    /// neutral grey). Callers driving the stacked encoder with
    /// pre-synced input typically pass all `Some`; a partially
    /// filled grid is supported for N-up layouts with fewer
    /// cameras than capacity.
    pub fn push(&mut self, tiles: &[Option<&YuvFrame>]) -> Result<(), StackedEncodeError> {
        let packed = pack_yuv420p(&self.layout, tiles)?;
        self.encoder
            .write_yuv420p_planes(&packed.y, &packed.u, &packed.v)?;
        Ok(())
    }

    /// Convenience: push a slice of owned frames (all
    /// populated). Equivalent to building an `Option` slice with
    /// `Some` on every entry.
    pub fn push_all(&mut self, tiles: &[&YuvFrame]) -> Result<(), StackedEncodeError> {
        let opts: Vec<Option<&YuvFrame>> = tiles.iter().copied().map(Some).collect();
        self.push(&opts)
    }

    /// Push buffered bytes to disk without finalizing the
    /// container.
    ///
    /// Required for write-while-read replay: a concurrent reader
    /// only sees bytes once the AVIO layer has written them to
    /// the file descriptor, and fMP4 flushes fragments only on
    /// this call (or when the next keyframe forces one). Call
    /// periodically from the replay path, typically once per
    /// keyframe or every few seconds.
    ///
    /// Does not finalize the file; [`Self::finish`] is still
    /// required when the recording session ends.
    pub fn flush(&mut self) -> Result<(), StackedEncodeError> {
        self.encoder.flush_to_disk()?;
        Ok(())
    }

    /// Write pre-packed YUV420P atlas bytes directly, bypassing
    /// the per-tile pack step. Used by the M7 GPU-pack path
    /// where the shader produces the final atlas and the
    /// encoder just needs to receive the bytes.
    ///
    /// `y` / `u` / `v` must match the layout's
    /// `packed_width() x packed_height()` Y plane and
    /// `(packed_width() / 2) x (packed_height() / 2)` chroma
    /// planes respectively.
    pub fn push_prepacked_yuv420p(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<(), StackedEncodeError> {
        self.encoder.write_yuv420p_planes(y, u, v)?;
        Ok(())
    }

    /// Flush remaining packets and finalize the container.
    /// Required for plain MP4 (writes the `moov` atom); optional
    /// but recommended for fMP4 (writes a trailing `mfra` index
    /// so seek performance on the finished file is better).
    pub fn finish(&mut self) -> Result<(), StackedEncodeError> {
        self.encoder.finish()?;
        Ok(())
    }

    /// The grid layout this encoder was configured with.
    pub fn layout(&self) -> &GridLayout {
        &self.layout
    }
}
