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
    /// `layout.packed_width() x layout.packed_height()`. A
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
