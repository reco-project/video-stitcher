//! Encoder trait for consuming stitched GPU frames.
//!
//! The encoder is the final stage of the pipeline. It receives rendered
//! frames from the GPU and produces the output video file or stream.
//!
//! ## Implementations (in `reco-io`)
//!
//! - FFmpeg backend: encoding with support for software (libx264/libx265)
//!   and hardware encoders (NVENC, VideoToolbox, VAAPI)
//! - Future: direct NVENC for Jetson, WebRTC for livestreaming

use thiserror::Error;

/// Errors that can occur during encoding.
#[derive(Debug, Error)]
pub enum EncodeError {
    /// The encoder failed to initialize.
    #[error("encoder initialization failed: {0}")]
    Init(String),

    /// Failed to encode a frame.
    #[error("frame encoding failed: {0}")]
    Frame(String),

    /// Failed to finalize the output.
    #[error("finalization failed: {0}")]
    Finalize(String),
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
/// Contains the pixel data read back from the GPU. In phase 1, this
/// is always a CPU buffer (GPU readback). In the future, this may
/// carry a GPU surface handle for zero-copy hardware encoding.
pub struct OutputFrame {
    /// Raw pixel data.
    pub data: Vec<u8>,
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
    fn submit(&mut self, frame: OutputFrame) -> Result<(), EncodeError>;

    /// Signal that all frames have been submitted.
    ///
    /// The encoder should flush any buffered frames and finalize the
    /// output file or stream.
    fn finish(&mut self) -> Result<(), EncodeError>;
}
