//! Encode packed camera planes for replay, without applying rendered-output colour conversion.
//!
//! Uses Matroska and automatic encoder selection by default. Planar input is interleaved when the encoder requires NV12. Consumers configure encoding through [`StackedEncoderConfig`].
use super::{GridLayout, StackError, pack_yuv420p};
use crate::ffmpeg::encoder::{EncodeError, EncoderConfig, VideoEncoder};
use crate::output::{Codec, Container, Quality, QualityPreset};
use reco_core::source::Yuv420pFrame;
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
    /// [`Container::Mkv`]; codec defaults to
    /// [`Codec::H264`]; encoder auto-detects hardware
    /// (NVENC/AMF/VT) with software fallback. The encoder
    /// handles YUV420P-to-NV12 interleave automatically when
    /// a hardware encoder is selected.
    pub inner: EncoderConfig,
    /// Output frames-per-second (numerator, denominator). `None` means
    /// "follow the source": [`crate::StitchJob`] resolves it from the
    /// probed source rate before opening the encoder. A bare
    /// [`StackedEncoder`] opened with `None` falls back to 30fps with a
    /// warning - a 60fps session recorded at a 30fps time base plays
    /// half-speed.
    pub fps: Option<(i32, i32)>,
    /// Range of the supplied camera planes. Recording wrappers derive this from their source.
    pub full_range: bool,
}

impl Default for StackedEncoderConfig {
    fn default() -> Self {
        Self {
            inner: EncoderConfig {
                // Matroska does not require a finalized central index to read completed clusters.
                container: Container::Mkv,
                codec: Some(Codec::H264),
                quality: Quality::Preset(QualityPreset::Fast),
                encoder_name: None,
                preset: None,
                audio_source: None,
                audio_start_time: 0.0,
                // Short GOP so replay readers see recent
                // frames within ~1 second. For Matroska the
                // GOP controls cluster cadence; for fMP4 it
                // would control fragment cadence. 30 frames
                // at 30fps costs ~5-10% bitrate vs the libx264
                // default of 250.
                gop_size: Some(30),
                stream_url: None,
                ..Default::default()
            },
            fps: None,
            full_range: false,
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
        let (fps_num, fps_den) = config.fps.unwrap_or_else(|| {
            log::warn!(
                "StackedEncoder opened without an fps; defaulting to 30fps - \
                 pass the source rate via StackedEncoderConfig::fps"
            );
            (30, 1)
        });
        let fps = ffmpeg_next::Rational(fps_num, fps_den);
        let encoder = VideoEncoder::new_source(
            output_path,
            layout.packed_width(),
            layout.packed_height(),
            fps,
            &config.inner,
            config.full_range,
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
                Container::Mov => "mov",
                Container::Mkv => "mkv",
                Container::Flv => "flv",
            },
        );
        Ok(Self { layout, encoder })
    }

    /// Push one row-major tuple of tiles. `tiles[i]` is either
    /// `Some(&Yuv420pFrame)` or `None` for an empty tile (zero-filled
    /// neutral grey). Callers driving the stacked encoder with
    /// pre-synced input typically pass all `Some`; a partially
    /// filled grid is supported for N-up layouts with fewer
    /// cameras than capacity.
    pub fn push(&mut self, tiles: &[Option<&Yuv420pFrame>]) -> Result<(), StackedEncodeError> {
        let packed = pack_yuv420p(&self.layout, tiles)?;
        self.encoder
            .write_yuv420p_planes(&packed.y, &packed.u, &packed.v)?;
        Ok(())
    }

    /// Convenience: push a slice of owned frames (all
    /// populated). Equivalent to building an `Option` slice with
    /// `Some` on every entry.
    pub fn push_all(&mut self, tiles: &[&Yuv420pFrame]) -> Result<(), StackedEncodeError> {
        let opts: Vec<Option<&Yuv420pFrame>> = tiles.iter().copied().map(Some).collect();
        self.push(&opts)
    }

    /// Publish encoded packets and container buffers for concurrent replay readers.
    ///
    /// Frames buffered inside the encoder may still be pending. No storage synchronization is performed.
    ///
    /// Does not finalize the file; [`Self::finish`] is still
    /// required when the recording session ends.
    pub fn flush(&mut self) -> Result<(), StackedEncodeError> {
        self.encoder.flush_to_disk()?;
        Ok(())
    }

    /// Write pre-packed YUV420P atlas bytes directly, bypassing
    /// the per-tile pack step. Used by the GPU-pack path
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
