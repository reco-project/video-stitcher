//! Encoder trait for consuming stitched frames.
//!
//! The encoder receives rendered frames after GPU readback and produces
//! the output video file or stream. Implementations live in `reco-io`.

use thiserror::Error;

/// Errors that can occur during encoding. `Clone + Send + Sync` so a
/// background encoder thread can send the result through an mpsc
/// channel without forcing the consumer to stringify.
#[derive(Debug, Clone, Error)]
pub enum EncodeError {
    /// The encoder failed to initialize.
    #[error("encoder initialization failed: {reason}")]
    Init {
        /// Human-readable explanation of the failure.
        reason: String,
    },

    /// Failed to encode a frame.
    #[error("frame encoding failed (frame {frame_index:?}): {reason}")]
    Frame {
        /// Index of the frame that failed, if known.
        frame_index: Option<u64>,
        /// Human-readable explanation of the failure.
        reason: String,
    },

    /// Failed to finalize the output.
    #[error("finalization failed: {reason}")]
    Finalize {
        /// Human-readable explanation of the failure.
        reason: String,
    },
}

/// Pixel format of the frame data passed to the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// RGBA 8-bit per channel, 4 bytes per pixel.
    Rgba8,
    /// NV12: Y plane followed by interleaved UV plane.
    Nv12,
}

/// A rendered frame ready for encoding.
///
/// Borrows pixel data from the GPU readback buffer. The data is valid
/// until the next frame is rendered (the readback buffer is reused).
#[derive(Debug)]
pub struct OutputFrame<'a> {
    /// Raw pixel data (borrowed from readback buffer).
    pub data: &'a [u8],
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format of the data.
    pub format: PixelFormat,
    /// Presentation timestamp in microseconds.
    pub pts_us: i64,
}

/// Trait for video encoders.
///
/// The pipeline calls [`Encoder::submit`] for each rendered frame, then
/// [`Encoder::finish`] when all frames have been processed.
pub trait Encoder: Send {
    /// Submit a rendered frame for encoding.
    ///
    /// Frames are submitted in presentation order. The encoder is
    /// responsible for buffering and reordering if needed (e.g. B-frames).
    fn submit(&mut self, frame: OutputFrame<'_>) -> Result<(), EncodeError>;

    /// Signal that all frames have been submitted.
    ///
    /// The encoder should flush any buffered frames and finalize the
    /// output file or stream.
    fn finish(&mut self) -> Result<(), EncodeError>;
}

