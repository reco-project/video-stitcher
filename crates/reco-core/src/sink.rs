//! Output sink trait for consuming stitched frames.
//!
//! A sink receives rendered frames after GPU readback and delivers
//! them somewhere: a video encoder, a periodic JPEG snapshot, a
//! network stream. Implementations live in `reco-io` and consumers;
//! the session fans each frame out to every attached sink.

use thiserror::Error;

/// Errors that can occur in a sink. `Clone + Send + Sync` so a
/// background sink thread can send the result through an mpsc
/// channel without forcing the consumer to stringify.
#[derive(Debug, Clone, Error)]
pub enum SinkError {
    /// The sink failed to initialize.
    #[error("sink initialization failed: {reason}")]
    Init {
        /// Human-readable explanation of the failure.
        reason: String,
    },

    /// Failed to consume a frame.
    #[error("sink rejected frame {frame_index:?}: {reason}")]
    Consume {
        /// Index of the frame that failed, if known.
        frame_index: Option<u64>,
        /// Human-readable explanation of the failure.
        reason: String,
    },

    /// Failed to finalize the output.
    #[error("sink finalization failed: {reason}")]
    Finish {
        /// Human-readable explanation of the failure.
        reason: String,
    },
}

/// Pixel format of the frame data passed to a sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// RGBA 8-bit per channel, 4 bytes per pixel.
    Rgba8,
    /// NV12: Y plane followed by interleaved UV plane.
    Nv12,
}

/// What a sink consumes, declared up front so the delivery loop can
/// provide it (or reject the sink at attach time with a typed error).
///
/// Residency is part of the declaration: GPU-resident variants (a
/// sink consuming the rendered texture without readback) extend this
/// enum later without breaking the [`OutputSink`] trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SinkInput {
    /// CPU bytes in the given pixel format, borrowed per frame.
    CpuBytes(PixelFormat),
}

/// A rendered frame ready for a sink.
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

/// Trait for stitched-output consumers (encoders, snapshots, streams).
///
/// The delivery loop calls [`OutputSink::consume`] for each rendered
/// frame in presentation order, then [`OutputSink::finish`] when all
/// frames have been processed.
pub trait OutputSink: Send {
    /// A short human-readable identity for logs and errors
    /// (e.g. the active encoder name, `"snapshot"`).
    fn name(&self) -> &str;

    /// What this sink consumes. The session delivers frames in the
    /// declared format, or rejects the sink at attach time if it
    /// cannot provide it.
    fn wants(&self) -> SinkInput;

    /// Consume a rendered frame.
    ///
    /// Frames arrive in presentation order. The sink is responsible
    /// for buffering and reordering if needed (e.g. B-frames).
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError>;

    /// Signal that all frames have been submitted.
    ///
    /// The sink should flush any buffered frames and finalize its
    /// output file or stream.
    fn finish(&mut self) -> Result<(), SinkError>;
}
