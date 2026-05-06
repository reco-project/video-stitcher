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

#[cfg(feature = "stacked-output")]
pub mod encoder;
#[cfg(feature = "stacked-output")]
pub mod replay;
#[cfg(feature = "stacked-output")]
pub mod source;

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
/// `tiles[i]` is either `Some(&YuvFrame)` or `None` (empty tile -
/// gets zero-filled Y plane with mid-grey chroma so the encoder
/// produces a neutral grey area). Tile order is row-major:
/// `tiles[0]` is top-left, `tiles[cols]` is the leftmost of the
/// second row.
///
/// Every supplied tile must match the layout's `tile_width x
/// tile_height`; otherwise [`StackError::TileDimensionMismatch`].
///
/// # Timestamp handling
///
/// The returned frame's `timestamp_us` is taken from the first
/// non-`None` tile. Callers that drive the stacked encoder with
/// pre-synced input will have all tiles sharing the same timestamp
/// anyway.
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
/// `tile_width x tile_height` dimensions; empty tiles in the source
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
