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

/// Trait for GPU-resident video encoders.
///
/// Unlike [`Encoder`], which receives CPU-side pixel data, a `GpuEncoder`
/// receives a [`wgpu::Texture`] reference directly. This enables zero-copy
/// encode paths where the GPU render output is encoded without ever reading
/// back to CPU memory (e.g. NVENC encoding from a Vulkan texture, or
/// VideoToolbox encoding from a Metal texture).
///
/// No implementations exist yet - this trait defines the API surface so
/// that future GPU encoder backends can be wired into [`StitchSession`]
/// without breaking changes.
///
/// [`StitchSession`]: crate::session::StitchSession
pub trait GpuEncoder: Send {
    /// Submit a GPU texture for encoding.
    ///
    /// The texture contains the rendered panoramic frame in RGBA8 format.
    /// The encoder is responsible for any format conversion (e.g. RGBA to NV12)
    /// on the GPU side. `pts_us` is the presentation timestamp in microseconds.
    ///
    /// The texture is only valid for the duration of this call - the encoder
    /// must either consume it immediately or copy it to an internal buffer.
    fn submit_texture(&mut self, texture: &wgpu::Texture, pts_us: i64) -> Result<(), EncodeError>;

    /// Signal that all frames have been submitted.
    ///
    /// The encoder should flush any buffered frames and finalize the
    /// output file or stream.
    fn finish(&mut self) -> Result<(), EncodeError>;
}
