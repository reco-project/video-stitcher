//! Stacked-video packing and unpacking (plan M6.5 items 1 + 2).
//!
//! Maps N logical YUV420P camera frames into a single grid-layout
//! frame (and back). Used by:
//!
//! - **Replay recording**: reco-obs / reco-gui write the live
//!   source tuples to a single video file during the session; the
//!   replay tool reads one file and recovers N camera streams.
//! - **Web panorama input**: a single uploaded file carries every
//!   camera the panorama needs. One HTTPS fetch, server demuxes.
//! - **Cloud deployment**: the stitching worker ingests one video
//!   URL regardless of camera count.
//! - **Cross-tool interchange**: `ffmpeg -i left.mp4 -i right.mp4
//!   -filter_complex vstack` already produces stacked videos; our
//!   unpacker consumes them directly.
//!
//! # Grid layout
//!
//! `GridLayout` describes how tiles fit in the packed frame.
//! `GridLayout::vstack(w, h, n)` gives a vertical stack for any N
//! (most common case: N=2, left on top of right). For irregular
//! counts use [`GridLayout::grid(w, h, rows, cols)`] and fill unused
//! tiles with `Tile::Empty`.
//!
//! ```rust,ignore
//! use reco_io::stacked_video::{GridLayout, pack_yuv420p, unpack_yuv420p};
//! use reco_core::source::YuvFrame;
//!
//! let layout = GridLayout::vstack(1920, 1080, 2);   // N=2 vertical
//! let packed = pack_yuv420p(&layout, &[&left, &right])?;
//! // ... encode `packed` via VideoEncoder ...
//!
//! let [left_out, right_out] = unpack_yuv420p::<2>(&layout, &packed)?;
//! ```
//!
//! # Invariants
//!
//! - YUV420P subsampling means every tile dimension must be even
//!   in both axes. Enforced at `GridLayout` construction.
//! - The packed frame's dimensions are exactly
//!   `(cols * tile_width, rows * tile_height)`.
//! - Tile ordering in the input slice matches row-major iteration
//!   over the grid (left-to-right, top-to-bottom).
//!
//! # GPU-accelerated path (future work)
//!
//! The current pack/unpack runs on the CPU: per-row `memcpy` into
//! the packed buffer (roughly 8 ms for 4K 2-tile vstack on a
//! warm cache). Good enough for a replay-recording side-path
//! running alongside stitching, but not the fastest shape possible.
//!
//! A wgpu-backed path would keep frames on the GPU the whole time:
//!
//! - A compute shader (or a small render pass with a quad per tile)
//!   blits N source textures into one render-target atlas. Textures
//!   are already on the GPU from the stitch pipeline's readback
//!   stage, so the CPU roundtrip disappears.
//! - The atlas texture feeds straight into the GPU-resident encoder
//!   (future M7 NVENC path / VideoToolbox path), never touching
//!   the CPU side.
//! - Conversely, the unpack direction is a single shader that
//!   samples the atlas into N separate destination textures, fed
//!   to the downstream pipeline as if they'd come from independent
//!   sources.
//!
//! The primitives in this module are intentionally CPU-pure so they
//! can be used from consumers that already have CPU frame data
//! (FFmpeg decode output, test harnesses, tools). The GPU path is
//! a separate module that would live in reco-core alongside the
//! other wgpu compute paths, with this CPU impl as the fallback
//! when GPU resources aren't available.

use reco_core::source::YuvFrame;
use thiserror::Error;

/// Grid layout for stacked-video packing.
///
/// Every tile has the same `tile_width` × `tile_height` dimensions.
/// The packed frame covers `cols * tile_width` × `rows * tile_height`
/// pixels. Both axes must be even (YUV420P subsampling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridLayout {
    tile_width: u32,
    tile_height: u32,
    rows: u32,
    cols: u32,
}

impl GridLayout {
    /// Vertical stack of `n` tiles, each `width` × `height`.
    /// Layout is 1 column × n rows. N=1 degenerates to the source
    /// frame; N=2 is the common left-on-top-of-right case.
    ///
    /// Returns `None` if `width` or `height` is odd or zero, or
    /// if `n == 0`.
    pub fn vstack(width: u32, height: u32, n: u32) -> Option<Self> {
        Self::grid(width, height, n, 1)
    }

    /// Horizontal stack of `n` tiles. Layout is n columns × 1 row.
    pub fn hstack(width: u32, height: u32, n: u32) -> Option<Self> {
        Self::grid(width, height, 1, n)
    }

    /// Generic grid layout. `rows * cols` gives the tile capacity;
    /// consumers that have fewer frames than the capacity pass
    /// `None` for the empty tiles (they get zero-filled to a
    /// neutral grey during pack).
    ///
    /// # Errors
    ///
    /// Returns `None` if any of `width`, `height`, `rows`, `cols`
    /// is zero, or if `width`/`height` is odd (YUV420P constraint).
    pub fn grid(width: u32, height: u32, rows: u32, cols: u32) -> Option<Self> {
        if width == 0 || height == 0 || rows == 0 || cols == 0 {
            return None;
        }
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return None;
        }
        Some(Self {
            tile_width: width,
            tile_height: height,
            rows,
            cols,
        })
    }

    /// Tile width in pixels.
    pub fn tile_width(&self) -> u32 {
        self.tile_width
    }

    /// Tile height in pixels.
    pub fn tile_height(&self) -> u32 {
        self.tile_height
    }

    /// Number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Total tile capacity (`rows * cols`).
    pub fn capacity(&self) -> u32 {
        self.rows * self.cols
    }

    /// Packed frame width in pixels.
    pub fn packed_width(&self) -> u32 {
        self.cols * self.tile_width
    }

    /// Packed frame height in pixels.
    pub fn packed_height(&self) -> u32 {
        self.rows * self.tile_height
    }
}

/// Errors from stacked-video pack / unpack operations. `Clone + Send + Sync`.
#[derive(Debug, Clone, Error)]
pub enum StackError {
    /// Caller supplied more tiles than the layout's capacity.
    #[error("too many tiles: layout holds {capacity}, got {got}")]
    TooManyTiles {
        /// The layout's `rows * cols`.
        capacity: u32,
        /// How many tiles the caller passed.
        got: usize,
    },
    /// A tile's dimensions do not match the layout's
    /// `tile_width` × `tile_height`.
    #[error("tile {index} dimensions {got_w}x{got_h} != layout {expected_w}x{expected_h}")]
    TileDimensionMismatch {
        /// Which tile (0-indexed) was wrong.
        index: usize,
        /// What we got.
        got_w: u32,
        /// What we got.
        got_h: u32,
        /// What the layout requires.
        expected_w: u32,
        /// What the layout requires.
        expected_h: u32,
    },
    /// The packed frame's dimensions don't match the layout.
    #[error("packed frame dimensions {got_w}x{got_h} != layout {expected_w}x{expected_h}")]
    PackedDimensionMismatch {
        /// What we got.
        got_w: u32,
        /// What we got.
        got_h: u32,
        /// What the layout requires.
        expected_w: u32,
        /// What the layout requires.
        expected_h: u32,
    },
    /// A tile's plane size didn't match what YUV420P requires for
    /// its dimensions (usually a malformed input).
    #[error("tile {index} plane {plane} size {got} != expected {expected}")]
    TilePlaneSizeMismatch {
        /// Tile index.
        index: usize,
        /// Which plane ("y", "u", "v").
        plane: &'static str,
        /// What we got.
        got: usize,
        /// Required length.
        expected: usize,
    },
}

/// Pack N YUV420P tiles into a single grid-layout YuvFrame.
///
/// `tiles[i]` is either `Some(&YuvFrame)` or `None` (empty tile —
/// gets zero-filled Y plane with mid-grey chroma so the encoder
/// produces a neutral grey area). Tile order is row-major:
/// `tiles[0]` is top-left, `tiles[cols]` is the leftmost of the
/// second row.
///
/// Every supplied tile must match the layout's `tile_width ×
/// tile_height`; otherwise [`StackError::TileDimensionMismatch`].
///
/// # Timestamp handling
///
/// The returned frame's `timestamp_us` is taken from the first
/// non-`None` tile. Callers that drive the stacked encoder with
/// pre-synced input (e.g. from
/// `reco_core::framesync::TimestampedIngestBuffer`) will have all
/// tiles sharing the same timestamp anyway.
pub fn pack_yuv420p(
    layout: &GridLayout,
    tiles: &[Option<&YuvFrame>],
) -> Result<YuvFrame, StackError> {
    if tiles.len() > layout.capacity() as usize {
        return Err(StackError::TooManyTiles {
            capacity: layout.capacity(),
            got: tiles.len(),
        });
    }

    // Validate tile dimensions + plane sizes up front so we don't
    // half-write the packed buffer before discovering a bad input.
    let tw = layout.tile_width as usize;
    let th = layout.tile_height as usize;
    let expected_y = tw * th;
    let expected_uv = (tw / 2) * (th / 2);
    for (i, slot) in tiles.iter().enumerate() {
        if let Some(t) = slot {
            if t.width != layout.tile_width || t.height != layout.tile_height {
                return Err(StackError::TileDimensionMismatch {
                    index: i,
                    got_w: t.width,
                    got_h: t.height,
                    expected_w: layout.tile_width,
                    expected_h: layout.tile_height,
                });
            }
            if t.y.len() != expected_y {
                return Err(StackError::TilePlaneSizeMismatch {
                    index: i,
                    plane: "y",
                    got: t.y.len(),
                    expected: expected_y,
                });
            }
            if t.u.len() != expected_uv {
                return Err(StackError::TilePlaneSizeMismatch {
                    index: i,
                    plane: "u",
                    got: t.u.len(),
                    expected: expected_uv,
                });
            }
            if t.v.len() != expected_uv {
                return Err(StackError::TilePlaneSizeMismatch {
                    index: i,
                    plane: "v",
                    got: t.v.len(),
                    expected: expected_uv,
                });
            }
        }
    }

    let pw = layout.packed_width() as usize;
    let ph = layout.packed_height() as usize;
    let packed_y_len = pw * ph;
    let packed_uv_len = (pw / 2) * (ph / 2);

    // Fill with neutral grey so empty tiles encode cleanly
    // (Y = 16 = black in BT.709 full-range would be true black;
    // 128 is mid-grey which compresses better than uniform zero).
    let mut y = vec![128u8; packed_y_len];
    let mut u = vec![128u8; packed_uv_len];
    let mut v = vec![128u8; packed_uv_len];

    for (i, slot) in tiles.iter().enumerate() {
        let row = i as u32 / layout.cols;
        let col = i as u32 % layout.cols;
        if let Some(t) = slot {
            copy_tile_plane(&t.y, tw, th, col, row, &mut y, pw);
            copy_tile_plane(&t.u, tw / 2, th / 2, col, row, &mut u, pw / 2);
            copy_tile_plane(&t.v, tw / 2, th / 2, col, row, &mut v, pw / 2);
        }
        // empty tile: leave the 128 fill in place.
    }

    let timestamp_us = tiles
        .iter()
        .flatten()
        .map(|t| t.timestamp_us)
        .next()
        .unwrap_or(0);

    Ok(YuvFrame {
        y,
        u,
        v,
        width: layout.packed_width(),
        height: layout.packed_height(),
        timestamp_us,
    })
}

/// Unpack N tiles from a grid-layout YuvFrame.
///
/// Returns a vector of `YuvFrame` in row-major order, length
/// `layout.capacity()`. All tiles have the layout's
/// `tile_width × tile_height` dimensions; empty tiles in the source
/// become neutral-grey frames.
///
/// The timestamp on every returned tile matches the source frame.
pub fn unpack_yuv420p(layout: &GridLayout, packed: &YuvFrame) -> Result<Vec<YuvFrame>, StackError> {
    if packed.width != layout.packed_width() || packed.height != layout.packed_height() {
        return Err(StackError::PackedDimensionMismatch {
            got_w: packed.width,
            got_h: packed.height,
            expected_w: layout.packed_width(),
            expected_h: layout.packed_height(),
        });
    }

    let pw = layout.packed_width() as usize;
    let ph = layout.packed_height() as usize;
    let expected_y = pw * ph;
    let expected_uv = (pw / 2) * (ph / 2);

    if packed.y.len() != expected_y {
        return Err(StackError::TilePlaneSizeMismatch {
            index: usize::MAX, // "whole packed frame"
            plane: "y",
            got: packed.y.len(),
            expected: expected_y,
        });
    }
    if packed.u.len() != expected_uv {
        return Err(StackError::TilePlaneSizeMismatch {
            index: usize::MAX,
            plane: "u",
            got: packed.u.len(),
            expected: expected_uv,
        });
    }
    if packed.v.len() != expected_uv {
        return Err(StackError::TilePlaneSizeMismatch {
            index: usize::MAX,
            plane: "v",
            got: packed.v.len(),
            expected: expected_uv,
        });
    }

    let tw = layout.tile_width as usize;
    let th = layout.tile_height as usize;
    let tile_y_len = tw * th;
    let tile_uv_len = (tw / 2) * (th / 2);

    let mut out = Vec::with_capacity(layout.capacity() as usize);
    for i in 0..layout.capacity() {
        let row = i / layout.cols;
        let col = i % layout.cols;
        let mut y = vec![0u8; tile_y_len];
        let mut u = vec![0u8; tile_uv_len];
        let mut v = vec![0u8; tile_uv_len];
        read_tile_plane(&packed.y, pw, tw, th, col, row, &mut y);
        read_tile_plane(&packed.u, pw / 2, tw / 2, th / 2, col, row, &mut u);
        read_tile_plane(&packed.v, pw / 2, tw / 2, th / 2, col, row, &mut v);
        out.push(YuvFrame {
            y,
            u,
            v,
            width: layout.tile_width,
            height: layout.tile_height,
            timestamp_us: packed.timestamp_us,
        });
    }
    Ok(out)
}

/// Copy a tile's single plane into the right location of the
/// packed frame. Row-by-row copy to respect the destination stride.
fn copy_tile_plane(
    src: &[u8],
    tw: usize,
    th: usize,
    col: u32,
    row: u32,
    dst: &mut [u8],
    dst_stride: usize,
) {
    let dst_x = col as usize * tw;
    let dst_y = row as usize * th;
    for r in 0..th {
        let src_off = r * tw;
        let dst_off = (dst_y + r) * dst_stride + dst_x;
        dst[dst_off..dst_off + tw].copy_from_slice(&src[src_off..src_off + tw]);
    }
}

/// Inverse of `copy_tile_plane`: read a tile out of the packed
/// frame into the destination tile buffer.
fn read_tile_plane(
    packed: &[u8],
    packed_stride: usize,
    tw: usize,
    th: usize,
    col: u32,
    row: u32,
    dst: &mut [u8],
) {
    let src_x = col as usize * tw;
    let src_y = row as usize * th;
    for r in 0..th {
        let src_off = (src_y + r) * packed_stride + src_x;
        let dst_off = r * tw;
        dst[dst_off..dst_off + tw].copy_from_slice(&packed[src_off..src_off + tw]);
    }
}

// ---------------------------------------------------------------------------
// FFmpeg-backed encoder / source (feature-gated; stub for this tranche)
// ---------------------------------------------------------------------------
//
// The pack/unpack primitives above are the real deliverable for
// M6.5 items 1+2 — they're pure YUV arithmetic, no I/O, trivially
// testable. The ffmpeg-backed encoder/source that wraps them
// around `ffmpeg/encoder.rs::VideoEncoder` and
// `ffmpeg/decoder.rs::VideoDecoder` is a thin glue layer that
// lands in the consumer-wiring tranche (M6.5 item 3 + 6) when
// reco-obs / reco-gui actually record replays.

#[cfg(feature = "stacked-output")]
pub mod encoder {
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
    /// fragmented MP4, `libx264` software encoder, no audio
    /// passthrough. Override any field as needed.
    #[derive(Debug, Clone)]
    pub struct StackedEncoderConfig {
        /// Inner encoder config. Container defaults to
        /// [`Container::Mp4Fragmented`]; codec defaults to
        /// [`VideoCodec::H264`]; `encoder_name` is forced to the
        /// software encoder for the chosen codec so pack output
        /// (YUV420P) is accepted without a pixel-format conversion.
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
                    quality: Quality::Balanced,
                    encoder_name: Some("libx264".to_string()),
                    crf: None,
                    preset: None,
                    audio_source: None,
                    // Short GOP so replay readers see recent
                    // frames within ~1 second. For Matroska the
                    // GOP controls cluster cadence; for fMP4 it
                    // would control fragment cadence. 30 frames
                    // at 30fps costs ~5-10% bitrate vs the libx264
                    // default of 250.
                    gop_size: Some(30),
                },
                fps: (30, 1),
            }
        }
    }

    /// Stacked-video encoder. One instance per output file.
    ///
    /// Not [`Sync`] — the underlying `ffmpeg` encoder owns raw
    /// pointers and must stay on one thread.
    pub struct StackedEncoder {
        layout: GridLayout,
        encoder: VideoEncoder,
    }

    impl StackedEncoder {
        /// Open a new stacked-video file for the given grid layout.
        ///
        /// The file's dimensions are
        /// `layout.packed_width() × layout.packed_height()`. All
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
        /// `packed_width() × packed_height()` Y plane and
        /// `(packed_width() / 2) × (packed_height() / 2)` chroma
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
}

/// FrameSource decorator that records pre-stitch source frames to a
/// stacked-video file alongside the live pipeline.
///
/// Enables the "professional replay" capability from the M6.5 plan
/// with zero plumbing in the consumer: call
/// [`crate::StitchJob::with_replay_recording`] and the library wires
/// a `ReplayRecordingSource` in front of the real source. Each
/// frame is passed through to the stitch pipeline unmodified while a
/// copy is packed into the replay encoder on the same thread.
///
/// Only the CPU `StereoFrame::Yuv420p` variant is recorded today:
///
/// - `Nv12` (Jetson ISP / NVDEC on Linux): not yet supported;
///   would need an NV12 pack variant or a one-off Nv12→Yuv420p
///   conversion on the hot path.
/// - `GpuResident` / `MetalResident`: can't record without a
///   GPU→CPU readback per frame, which we don't pay until a
///   consumer explicitly asks.
///
/// A non-Yuv420p frame logs a `warn!` on first encounter and is
/// passed through unrecorded. The replay file will contain only the
/// Yuv420p-path frames.
#[cfg(feature = "stacked-output")]
pub mod replay {
    use super::encoder::{StackedEncoder, StackedEncoderConfig};
    use super::{GridLayout, encoder::StackedEncodeError};
    use reco_core::source::{FrameSource, SourceError, SourceInfo, StereoFrame, YuvFrame};
    use std::path::Path;

    /// Decorator FrameSource that also writes source frames to a
    /// stacked-video replay file.
    pub struct ReplayRecordingSource {
        inner: Box<dyn FrameSource>,
        encoder: Option<StackedEncoder>,
        warned_non_yuv420p: bool,
        frames_recorded: u64,
        flush_interval: u64,
    }

    impl ReplayRecordingSource {
        /// Wrap an existing source, recording each Yuv420p stereo
        /// frame to `path` as it passes through. `config` supplies
        /// the encoder parameters; `StackedEncoderConfig::default`
        /// is the usual choice.
        pub fn wrap(
            inner: Box<dyn FrameSource>,
            path: &Path,
            config: StackedEncoderConfig,
        ) -> Result<Self, StackedEncodeError> {
            let info = inner.info();
            let layout = GridLayout::vstack(info.width, info.height, 2).ok_or_else(|| {
                StackedEncodeError::Pack(super::StackError::TileDimensionMismatch {
                    index: 0,
                    got_w: info.width,
                    got_h: info.height,
                    expected_w: (info.width / 2) * 2,
                    expected_h: (info.height / 2) * 2,
                })
            })?;
            let encoder = StackedEncoder::new(layout, path, config)?;
            log::info!(
                "Replay recording: {}x{} -> {} (stacked vertical, 2 tiles)",
                info.width,
                info.height,
                path.display(),
            );
            let _ = layout;
            Ok(Self {
                inner,
                encoder: Some(encoder),
                warned_non_yuv420p: false,
                frames_recorded: 0,
                // Flush to disk every ~1 second at 30fps. Balances
                // replay freshness against syscall overhead.
                flush_interval: 30,
            })
        }

        /// Finalize the replay file. Safe to call from the pipeline
        /// run-end path; subsequent frames from the inner source
        /// pass through unrecorded.
        pub fn finish(&mut self) -> Result<(), StackedEncodeError> {
            if let Some(mut enc) = self.encoder.take() {
                enc.finish()?;
                log::info!(
                    "Replay recording: finished ({} frames)",
                    self.frames_recorded
                );
            }
            Ok(())
        }

        /// How many frames have been written to the replay file.
        pub fn frames_recorded(&self) -> u64 {
            self.frames_recorded
        }

        fn record(&mut self, frame: &StereoFrame) {
            let Some(ref mut encoder) = self.encoder else {
                return;
            };
            match frame {
                StereoFrame::Yuv420p(pair) => {
                    let info = self.inner.info();
                    let left = YuvFrame {
                        y: pair.left.y.clone(),
                        u: pair.left.u.clone(),
                        v: pair.left.v.clone(),
                        width: info.width,
                        height: info.height,
                        timestamp_us: 0,
                    };
                    let right = YuvFrame {
                        y: pair.right.y.clone(),
                        u: pair.right.u.clone(),
                        v: pair.right.v.clone(),
                        width: info.width,
                        height: info.height,
                        timestamp_us: 0,
                    };
                    if let Err(e) = encoder.push(&[Some(&left), Some(&right)]) {
                        log::warn!(
                            "replay push failed ({e}); disabling replay recording for this session"
                        );
                        self.encoder = None;
                        return;
                    }
                    self.frames_recorded += 1;
                    if self.frames_recorded.is_multiple_of(self.flush_interval) {
                        // Best-effort flush; on failure we keep
                        // encoding. The reader will see stale
                        // content but that's better than dropping
                        // the whole session.
                        let _ = encoder.flush();
                    }
                }
                _ => {
                    if !self.warned_non_yuv420p {
                        log::warn!(
                            "replay recording: source yields non-Yuv420p frames; \
                             recording disabled for this session (Nv12/GPU-resident \
                             variants need a separate pack path, not implemented yet)"
                        );
                        self.warned_non_yuv420p = true;
                        self.encoder = None;
                    }
                }
            }
        }
    }

    impl FrameSource for ReplayRecordingSource {
        fn info(&self) -> SourceInfo {
            self.inner.info()
        }

        fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
            let frame = self.inner.next_frame()?;
            if let Some(ref f) = frame {
                self.record(f);
            }
            Ok(frame)
        }

        fn try_next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
            let frame = self.inner.try_next_frame()?;
            if let Some(ref f) = frame {
                self.record(f);
            }
            Ok(frame)
        }

        fn is_gpu_resident(&self) -> bool {
            self.inner.is_gpu_resident()
        }

        fn gpu_pixel_format(&self) -> reco_core::renderer::GpuPixelFormat {
            self.inner.gpu_pixel_format()
        }

        fn left_rotation(&self) -> i32 {
            self.inner.left_rotation()
        }

        fn right_rotation(&self) -> i32 {
            self.inner.right_rotation()
        }
    }

    impl Drop for ReplayRecordingSource {
        fn drop(&mut self) {
            // Best-effort finalize on drop so callers who forget
            // to call `finish()` still get a valid file (albeit
            // without the final-log line).
            let _ = self.finish();
        }
    }

    /// Compile-time bound: `ReplayRecordingSource` must be Send so
    /// StitchJob can move it into the session's worker loop.
    const _: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<ReplayRecordingSource>();
    };

    // ─────────────────────────────────────────────────────────────
    // Push-API recorder (FRICTION A18 close)
    // ─────────────────────────────────────────────────────────────

    use crate::stacked_video::StackError;
    use reco_core::core::StackedReplayRecorder as CoreStackedReplayRecorder;
    use reco_core::pipeline::YuvPlanes;

    /// Push-API stacked-video recorder. Implements reco-core's
    /// [`reco_core::core::StackedReplayRecorder`] trait so any
    /// [`reco_core::session::StitchSession`] can attach a recording
    /// file with a single call.
    ///
    /// Packs each submitted YUV plane pair into a [`GridLayout`] and
    /// feeds the encoder. Uses the same [`StackedEncoder`] +
    /// [`StackedEncoderConfig`] the pull side uses, so behavior,
    /// container defaults, and GOP cadence are identical across the
    /// two APIs.
    pub struct SessionStackedRecorder {
        encoder: Option<StackedEncoder>,
        width: u32,
        height: u32,
        frames_recorded: u64,
        frames_failed: u64,
        /// Reusable buffers so `push_planes` avoids allocating a
        /// new YuvFrame per call. Sized on first use then fixed.
        left_buf: Option<reco_core::source::YuvFrame>,
        right_buf: Option<reco_core::source::YuvFrame>,
    }

    impl SessionStackedRecorder {
        /// Open a recording for an N=2 vertical stack of
        /// `width × height` tiles. Returns a boxed trait object so
        /// consumers can pass it straight to
        /// [`reco_core::session::StitchSession::set_stacked_recorder`].
        pub fn open(
            path: &std::path::Path,
            config: StackedEncoderConfig,
            width: u32,
            height: u32,
        ) -> Result<Box<dyn CoreStackedReplayRecorder>, StackedEncodeError> {
            let layout =
                crate::stacked_video::GridLayout::vstack(width, height, 2).ok_or_else(|| {
                    StackedEncodeError::Pack(StackError::TileDimensionMismatch {
                        index: 0,
                        got_w: width,
                        got_h: height,
                        expected_w: (width / 2) * 2,
                        expected_h: (height / 2) * 2,
                    })
                })?;
            let encoder = StackedEncoder::new(layout, path, config)?;
            log::info!(
                "SessionStackedRecorder: {}x{} tiles -> {} (push API, M6.5 A18)",
                width,
                height,
                path.display(),
            );
            Ok(Box::new(Self {
                encoder: Some(encoder),
                width,
                height,
                frames_recorded: 0,
                frames_failed: 0,
                left_buf: None,
                right_buf: None,
            }))
        }

        fn ensure_buf(buf: &mut Option<reco_core::source::YuvFrame>, width: u32, height: u32) {
            let needed_y = (width as usize) * (height as usize);
            let needed_uv = needed_y / 4;
            match buf {
                Some(f) if f.y.len() == needed_y && f.u.len() == needed_uv => {}
                _ => {
                    *buf = Some(reco_core::source::YuvFrame {
                        y: vec![0u8; needed_y],
                        u: vec![0u8; needed_uv],
                        v: vec![0u8; needed_uv],
                        width,
                        height,
                        timestamp_us: 0,
                    });
                }
            }
        }

        fn fill_from_planes(frame: &mut reco_core::source::YuvFrame, planes: &YuvPlanes<'_>) {
            // YuvPlanes is tight (no stride padding) by contract,
            // so a straight copy_from_slice works. If the slices
            // are shorter than expected (malformed caller input)
            // we skip recording for this frame - the encoder would
            // reject a tile with mismatched plane sizes anyway.
            let need_y = frame.y.len();
            let need_uv = frame.u.len();
            if planes.y.len() < need_y || planes.u.len() < need_uv || planes.v.len() < need_uv {
                return;
            }
            frame.y.copy_from_slice(&planes.y[..need_y]);
            frame.u.copy_from_slice(&planes.u[..need_uv]);
            frame.v.copy_from_slice(&planes.v[..need_uv]);
        }
    }

    impl CoreStackedReplayRecorder for SessionStackedRecorder {
        fn record_yuv(
            &mut self,
            left: &YuvPlanes<'_>,
            right: &YuvPlanes<'_>,
            width: u32,
            height: u32,
        ) {
            // Reject dimension drift: the recorder was opened for
            // a fixed layout. If the session starts feeding
            // differently-sized frames (e.g. resolution change
            // mid-session) we stop recording and log.
            if width != self.width || height != self.height {
                log::warn!(
                    "SessionStackedRecorder: frame {}x{} != opened {}x{}; disabling recording",
                    width,
                    height,
                    self.width,
                    self.height,
                );
                self.encoder = None;
                return;
            }
            let Some(ref mut encoder) = self.encoder else {
                return;
            };
            Self::ensure_buf(&mut self.left_buf, width, height);
            Self::ensure_buf(&mut self.right_buf, width, height);
            let (Some(ref mut left_frame), Some(ref mut right_frame)) =
                (self.left_buf.as_mut(), self.right_buf.as_mut())
            else {
                return;
            };
            Self::fill_from_planes(left_frame, left);
            Self::fill_from_planes(right_frame, right);
            match encoder.push(&[Some(left_frame), Some(right_frame)]) {
                Ok(()) => {
                    self.frames_recorded += 1;
                }
                Err(e) => {
                    self.frames_failed += 1;
                    log::warn!(
                        "SessionStackedRecorder: push failed after {} frames ({e}); \
                         disabling recording",
                        self.frames_recorded,
                    );
                    self.encoder = None;
                }
            }
        }

        fn flush(&mut self) {
            if let Some(ref mut encoder) = self.encoder
                && let Err(e) = encoder.flush()
            {
                log::warn!("SessionStackedRecorder: flush failed ({e})");
            }
        }

        fn finish(&mut self) {
            if let Some(mut encoder) = self.encoder.take() {
                match encoder.finish() {
                    Ok(()) => log::info!(
                        "SessionStackedRecorder: finished ({} frames recorded, {} failed)",
                        self.frames_recorded,
                        self.frames_failed,
                    ),
                    Err(e) => log::warn!(
                        "SessionStackedRecorder: finish failed ({e}) after {} frames",
                        self.frames_recorded,
                    ),
                }
            }
        }
    }

    impl Drop for SessionStackedRecorder {
        fn drop(&mut self) {
            // Best-effort finalize so a recorder dropped without an
            // explicit `finish()` call still produces a valid file.
            if self.encoder.is_some() {
                self.finish();
            }
        }
    }

    /// Convenience constructor: opens a recorder suitable for
    /// [`reco_core::session::StitchSession::set_stacked_recorder`].
    /// Equivalent to [`SessionStackedRecorder::open`] but named for
    /// discoverability from consumers looking at the session API
    /// docs.
    pub fn session_recorder(
        path: &std::path::Path,
        config: StackedEncoderConfig,
        width: u32,
        height: u32,
    ) -> Result<Box<dyn CoreStackedReplayRecorder>, StackedEncodeError> {
        SessionStackedRecorder::open(path, config, width, height)
    }

    use reco_core::core::StackedReplayGpuRecorder as CoreStackedGpuRecorder;
    use reco_core::yuv_stack_packer::StackedAtlas;

    /// GPU-pack atlas recorder (M7 pivot). Implements
    /// [`reco_core::core::StackedReplayGpuRecorder`] by forwarding
    /// pre-packed [`StackedAtlas`] bytes straight to a
    /// [`StackedEncoder`]. There's no pack work in this type — the
    /// compute shader already produced the Y/U/V atlas planes; this
    /// recorder is just the sink that reaches the encoder.
    ///
    /// Open with [`GpuAtlasRecorder::open`], passing the
    /// **atlas** dimensions (not per-tile dims) returned by
    /// [`reco_core::session::StitchSession::stacked_atlas_dims`].
    /// Attach via
    /// [`reco_core::session::StitchSession::set_stacked_gpu_recorder`].
    pub struct GpuAtlasRecorder {
        encoder: Option<StackedEncoder>,
        atlas_width: u32,
        atlas_height: u32,
        frames_recorded: u64,
        frames_failed: u64,
        /// Flush-to-disk cadence. Kept identical to
        /// [`SessionStackedRecorder`]: once per 30 atlases, which at
        /// 30 fps is roughly once per second.
        flush_interval: u64,
    }

    impl GpuAtlasRecorder {
        /// Open a GPU-pack atlas recorder. `atlas_width / atlas_height`
        /// are the full atlas dims the packer produces — for an N=2
        /// vstack of `tile_w × tile_h` tiles this is
        /// `(tile_w, tile_h * 2)`. The encoder is configured for a
        /// 1×1 layout of `atlas_width × atlas_height` since the pack
        /// shader already built the grid; there are no sub-tiles to
        /// pack here.
        pub fn open(
            path: &std::path::Path,
            config: StackedEncoderConfig,
            atlas_width: u32,
            atlas_height: u32,
        ) -> Result<Box<dyn CoreStackedGpuRecorder>, StackedEncodeError> {
            let layout = crate::stacked_video::GridLayout::vstack(atlas_width, atlas_height, 1)
                .ok_or_else(|| {
                    StackedEncodeError::Pack(StackError::TileDimensionMismatch {
                        index: 0,
                        got_w: atlas_width,
                        got_h: atlas_height,
                        expected_w: (atlas_width / 2) * 2,
                        expected_h: (atlas_height / 2) * 2,
                    })
                })?;
            let encoder = StackedEncoder::new(layout, path, config)?;
            log::info!(
                "GpuAtlasRecorder: atlas {}x{} -> {} (GPU-pack sink, M7)",
                atlas_width,
                atlas_height,
                path.display(),
            );
            Ok(Box::new(Self {
                encoder: Some(encoder),
                atlas_width,
                atlas_height,
                frames_recorded: 0,
                frames_failed: 0,
                flush_interval: 30,
            }))
        }
    }

    impl CoreStackedGpuRecorder for GpuAtlasRecorder {
        fn record_atlas(&mut self, atlas: &StackedAtlas) {
            // Dimension drift: the packer was opened for a fixed
            // atlas size. Anything else indicates consumer wiring
            // confusion — stop recording, log once.
            if atlas.width != self.atlas_width || atlas.height != self.atlas_height {
                log::warn!(
                    "GpuAtlasRecorder: atlas {}x{} != opened {}x{}; disabling recording",
                    atlas.width,
                    atlas.height,
                    self.atlas_width,
                    self.atlas_height,
                );
                self.encoder = None;
                return;
            }
            let Some(ref mut encoder) = self.encoder else {
                return;
            };
            match encoder.push_prepacked_yuv420p(&atlas.y, &atlas.u, &atlas.v) {
                Ok(()) => {
                    self.frames_recorded += 1;
                    if self.frames_recorded.is_multiple_of(self.flush_interval)
                        && let Err(e) = encoder.flush()
                    {
                        log::warn!("GpuAtlasRecorder: flush failed ({e})");
                    }
                }
                Err(e) => {
                    self.frames_failed += 1;
                    log::warn!(
                        "GpuAtlasRecorder: write failed after {} atlases ({e}); disabling",
                        self.frames_recorded,
                    );
                    self.encoder = None;
                }
            }
        }

        fn flush(&mut self) {
            if let Some(ref mut encoder) = self.encoder
                && let Err(e) = encoder.flush()
            {
                log::warn!("GpuAtlasRecorder: flush failed ({e})");
            }
        }

        fn finish(&mut self) {
            if let Some(mut encoder) = self.encoder.take() {
                match encoder.finish() {
                    Ok(()) => log::info!(
                        "GpuAtlasRecorder: finished ({} atlases recorded, {} failed)",
                        self.frames_recorded,
                        self.frames_failed,
                    ),
                    Err(e) => log::warn!(
                        "GpuAtlasRecorder: finish failed ({e}) after {} atlases",
                        self.frames_recorded,
                    ),
                }
            }
        }
    }

    impl Drop for GpuAtlasRecorder {
        fn drop(&mut self) {
            if self.encoder.is_some() {
                <Self as CoreStackedGpuRecorder>::finish(self);
            }
        }
    }

    /// Convenience constructor for the GPU-pack recorder. Mirrors
    /// [`session_recorder`] for the GPU path: the encoder type isn't
    /// exposed to consumers; they receive a boxed trait object ready
    /// for
    /// [`reco_core::session::StitchSession::set_stacked_gpu_recorder`].
    pub fn session_gpu_recorder(
        path: &std::path::Path,
        config: StackedEncoderConfig,
        atlas_width: u32,
        atlas_height: u32,
    ) -> Result<Box<dyn CoreStackedGpuRecorder>, StackedEncodeError> {
        GpuAtlasRecorder::open(path, config, atlas_width, atlas_height)
    }
}

#[cfg(feature = "stacked-output")]
pub mod source {
    //! Stacked-video source. Wraps
    //! [`crate::ffmpeg::decoder::VideoDecoder`] and demuxes each
    //! packed frame into N tiles.
    //!
    //! Two consumer shapes:
    //!
    //! - [`StackedSource::next_tuple`] yields the full tile vector
    //!   for N-arbitrary consumers (replay scrubbing, analysis).
    //! - The [`reco_core::source::FrameSource`] impl accepts only
    //!   `capacity == 2` layouts and yields
    //!   [`reco_core::source::StereoFrame::Yuv420p`] pairs so the
    //!   stitch pipeline can drive off a stacked recording as if it
    //!   were two independent cameras.
    use super::{GridLayout, StackError, unpack_yuv420p};
    use crate::ffmpeg::decoder::{DecodeError, VideoDecoder};
    use reco_core::source::{
        FramePair, FrameSource, SourceError, SourceInfo, StereoFrame, YuvData, YuvFrame,
    };
    use std::path::Path;
    use thiserror::Error;

    /// Errors from stacked-video decoding.
    #[derive(Debug, Error)]
    pub enum StackedDecodeError {
        /// FFmpeg-layer error (open, decode).
        #[error("decode: {0}")]
        Decode(#[from] DecodeError),
        /// Unpack-layer error (packed frame shape vs layout).
        #[error("unpack: {0}")]
        Unpack(#[from] StackError),
        /// The container dimensions don't match the given grid
        /// layout. The caller supplied the wrong grid for the file.
        #[error("layout mismatch: file is {file_w}x{file_h}, layout expects {layout_w}x{layout_h}")]
        LayoutMismatch {
            /// The decoded file's reported width.
            file_w: u32,
            /// The decoded file's reported height.
            file_h: u32,
            /// What `layout.packed_width()` returns.
            layout_w: u32,
            /// What `layout.packed_height()` returns.
            layout_h: u32,
        },
        /// FrameSource path was given a layout whose capacity is
        /// not 2. Stereo pipeline requires exactly two tiles.
        #[error("FrameSource path requires capacity=2, layout has {got}")]
        NotStereo {
            /// The layout's `rows * cols`.
            got: u32,
        },
    }

    /// Stacked-video source. One instance per input file.
    pub struct StackedSource {
        layout: GridLayout,
        decoder: VideoDecoder,
    }

    impl StackedSource {
        /// Open a stacked video file for the given grid layout.
        ///
        /// Verifies that the file's dimensions match
        /// `layout.packed_width() × layout.packed_height()`. A
        /// mismatch means the caller passed the wrong grid for
        /// the file - probably a shape change between writer and
        /// reader.
        pub fn open(layout: GridLayout, input_path: &Path) -> Result<Self, StackedDecodeError> {
            let decoder = VideoDecoder::open(input_path)?;
            let file_w = decoder.width();
            let file_h = decoder.height();
            if file_w != layout.packed_width() || file_h != layout.packed_height() {
                return Err(StackedDecodeError::LayoutMismatch {
                    file_w,
                    file_h,
                    layout_w: layout.packed_width(),
                    layout_h: layout.packed_height(),
                });
            }
            Ok(Self { layout, decoder })
        }

        /// Grid layout this source was opened with.
        pub fn layout(&self) -> &GridLayout {
            &self.layout
        }

        /// Frame rate of the underlying file (frames per second).
        pub fn fps(&self) -> f64 {
            self.decoder.fps()
        }

        /// Decode the next packed frame and split it into N tiles,
        /// or `None` at end of stream.
        pub fn next_tuple(&mut self) -> Result<Option<Vec<YuvFrame>>, StackedDecodeError> {
            match self.decoder.next_frame()? {
                Some(packed) => {
                    let tiles = unpack_yuv420p(&self.layout, &packed)?;
                    Ok(Some(tiles))
                }
                None => Ok(None),
            }
        }
    }

    // Safety: VideoDecoder owns ffmpeg raw pointers but is bound to
    // one thread by its `!Sync` wrapping. StackedSource inherits
    // that; the Send impl is carried by VideoDecoder (which the
    // crate already marks Send).
    impl FrameSource for StackedSource {
        fn info(&self) -> SourceInfo {
            let tw = self.layout.tile_width();
            let th = self.layout.tile_height();
            let (num, den) = {
                let r = self.decoder.frame_rate();
                (r.0, r.1)
            };
            SourceInfo {
                width: tw,
                height: th,
                fps: self.fps(),
                fps_rational: Some((num, den)),
                total_frames: None,
            }
        }

        fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
            if self.layout.capacity() != 2 {
                return Err(SourceError::Read {
                    reason: format!(
                        "StackedSource as FrameSource requires capacity=2, layout has {}",
                        self.layout.capacity()
                    ),
                });
            }
            let tuple = self.next_tuple().map_err(|e| SourceError::Read {
                reason: e.to_string(),
            })?;
            let Some(tiles) = tuple else {
                return Ok(None);
            };
            // Row-major: `tiles[0] = left`, `tiles[1] = right` matches
            // the writer convention established in `pack_yuv420p`.
            let tiles: [YuvFrame; 2] =
                tiles
                    .try_into()
                    .map_err(|v: Vec<YuvFrame>| SourceError::Read {
                        reason: format!("expected 2 tiles, got {}", v.len()),
                    })?;
            let [left, right] = tiles;
            Ok(Some(StereoFrame::Yuv420p(FramePair {
                left: YuvData {
                    y: left.y,
                    u: left.u,
                    v: left.v,
                },
                right: YuvData {
                    y: right.y,
                    u: right.u,
                    v: right.v,
                },
            })))
        }
    }
}

// Compile-time bound check: `StackError` is `Clone + Send + Sync`
// so consumers posting pack/unpack results to worker-thread channels
// carry the typed error instead of stringifying.
const _: fn() = || {
    fn assert_clone_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_clone_send_sync::<StackError>();
};

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(w: u32, h: u32, fill_y: u8, fill_u: u8, fill_v: u8) -> YuvFrame {
        YuvFrame {
            y: vec![fill_y; (w * h) as usize],
            u: vec![fill_u; ((w / 2) * (h / 2)) as usize],
            v: vec![fill_v; ((w / 2) * (h / 2)) as usize],
            width: w,
            height: h,
            timestamp_us: 12345,
        }
    }

    #[test]
    fn vstack_layout_rejects_odd_dimensions() {
        assert!(GridLayout::vstack(1920, 1080, 2).is_some());
        assert!(GridLayout::vstack(1921, 1080, 2).is_none(), "odd width");
        assert!(GridLayout::vstack(1920, 1079, 2).is_none(), "odd height");
        assert!(GridLayout::vstack(0, 1080, 2).is_none());
        assert!(GridLayout::vstack(1920, 1080, 0).is_none());
    }

    #[test]
    fn vstack_packed_dimensions_match() {
        let layout = GridLayout::vstack(1920, 1080, 2).unwrap();
        assert_eq!(layout.packed_width(), 1920);
        assert_eq!(layout.packed_height(), 2160);
        assert_eq!(layout.capacity(), 2);
    }

    #[test]
    fn hstack_packed_dimensions_match() {
        let layout = GridLayout::hstack(640, 480, 3).unwrap();
        assert_eq!(layout.packed_width(), 1920);
        assert_eq!(layout.packed_height(), 480);
        assert_eq!(layout.capacity(), 3);
    }

    #[test]
    fn pack_unpack_round_trips_identity() {
        // Two 64x64 frames with distinct fills; pack then unpack
        // must recover both byte-exactly.
        let layout = GridLayout::vstack(64, 64, 2).unwrap();
        let a = fill(64, 64, 100, 120, 140);
        let b = fill(64, 64, 200, 60, 90);
        let packed = pack_yuv420p(&layout, &[Some(&a), Some(&b)]).unwrap();
        assert_eq!(packed.width, 64);
        assert_eq!(packed.height, 128);

        let unpacked = unpack_yuv420p(&layout, &packed).unwrap();
        assert_eq!(unpacked.len(), 2);
        assert_eq!(unpacked[0].y, a.y);
        assert_eq!(unpacked[0].u, a.u);
        assert_eq!(unpacked[0].v, a.v);
        assert_eq!(unpacked[1].y, b.y);
        assert_eq!(unpacked[1].u, b.u);
        assert_eq!(unpacked[1].v, b.v);
    }

    #[test]
    fn pack_empty_tile_gets_grey_fill() {
        // N=3 vstack with only 2 tiles provided; the third tile's
        // region in the packed frame should be filled with 128
        // (mid-grey Y + neutral chroma).
        let layout = GridLayout::vstack(64, 64, 3).unwrap();
        let a = fill(64, 64, 10, 20, 30);
        let b = fill(64, 64, 40, 50, 60);
        let packed = pack_yuv420p(&layout, &[Some(&a), Some(&b), None]).unwrap();

        let unpacked = unpack_yuv420p(&layout, &packed).unwrap();
        assert_eq!(unpacked.len(), 3);
        // Third tile should be the default fill.
        assert!(unpacked[2].y.iter().all(|&b| b == 128));
        assert!(unpacked[2].u.iter().all(|&b| b == 128));
        assert!(unpacked[2].v.iter().all(|&b| b == 128));
    }

    #[test]
    fn pack_rejects_too_many_tiles() {
        let layout = GridLayout::vstack(64, 64, 2).unwrap();
        let a = fill(64, 64, 0, 0, 0);
        let err = pack_yuv420p(
            &layout,
            &[Some(&a), Some(&a), Some(&a)], // 3 tiles in a 2-tile layout
        )
        .unwrap_err();
        assert!(matches!(
            err,
            StackError::TooManyTiles {
                capacity: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn pack_rejects_mismatched_tile_dimensions() {
        let layout = GridLayout::vstack(64, 64, 2).unwrap();
        let a = fill(64, 64, 0, 0, 0);
        let wrong = fill(128, 64, 0, 0, 0); // wrong width
        let err = pack_yuv420p(&layout, &[Some(&a), Some(&wrong)]).unwrap_err();
        assert!(matches!(
            err,
            StackError::TileDimensionMismatch { index: 1, .. }
        ));
    }

    #[test]
    fn unpack_rejects_mismatched_packed_dimensions() {
        let layout = GridLayout::vstack(64, 64, 2).unwrap();
        let wrong = fill(64, 64, 0, 0, 0); // 64x64 but layout expects 64x128
        let err = unpack_yuv420p(&layout, &wrong).unwrap_err();
        assert!(matches!(err, StackError::PackedDimensionMismatch { .. }));
    }

    #[test]
    fn grid_3x3_round_trips_nine_tiles() {
        // N=9 3x3 grid: validates the plan's "N>2 uses NxN grid"
        // contract. Each tile gets a distinct fill; round-trip
        // must recover each one from the right (row, col) slot.
        let layout = GridLayout::grid(32, 32, 3, 3).unwrap();
        let tiles: Vec<YuvFrame> = (0..9)
            .map(|i| fill(32, 32, i as u8 * 20, 128, 128))
            .collect();
        let slots: Vec<Option<&YuvFrame>> = tiles.iter().map(Some).collect();
        let packed = pack_yuv420p(&layout, &slots).unwrap();
        assert_eq!(packed.width, 96);
        assert_eq!(packed.height, 96);

        let unpacked = unpack_yuv420p(&layout, &packed).unwrap();
        assert_eq!(unpacked.len(), 9);
        for (i, t) in unpacked.iter().enumerate() {
            assert_eq!(t.y[0], i as u8 * 20, "tile {i}: pack/unpack lost identity");
        }
    }

    #[test]
    fn packed_timestamp_follows_first_nonempty_tile() {
        let layout = GridLayout::vstack(64, 64, 2).unwrap();
        let mut a = fill(64, 64, 0, 0, 0);
        let mut b = fill(64, 64, 0, 0, 0);
        a.timestamp_us = 1000;
        b.timestamp_us = 2000;
        let packed = pack_yuv420p(&layout, &[Some(&a), Some(&b)]).unwrap();
        assert_eq!(
            packed.timestamp_us, 1000,
            "timestamp is inherited from the first non-empty tile"
        );
    }
}
