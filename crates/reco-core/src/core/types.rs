//! Pure types and traits extracted from the `core` module.
//!
//! Houses error types, configuration, render outcome, replay frame, and
//! the recorder traits that decouple reco-core from I/O implementations.

use std::time::Duration;

use thiserror::Error;

use crate::geometry::Pose;
#[cfg(feature = "gpu")]
use crate::gpu::rgba_readback::RgbaReadbackError;
#[cfg(feature = "gpu")]
use crate::gpu::yuv_stack_packer::{PackerError, StackedAtlas};
#[cfg(feature = "gpu")]
use crate::render::pipeline::PipelineError;
use crate::render::planes::YuvPlanes;
use crate::stitch::StitchError;

/// Errors from [`super::StitchCore`]. `Clone + Send + Sync` so consumers
/// posting render results to worker-thread channels carry the typed
/// error instead of stringifying at the boundary.
#[derive(Debug, Clone, Error)]
pub enum StitchCoreError {
    /// Executor construction or stitch error. The `From` impl lets
    /// consumers `?` a [`GpuExecutor`](crate::stitch::GpuExecutor)
    /// build straight into the engine's error type.
    #[error("executor: {0}")]
    Executor(#[from] StitchError),
    /// GPU pipeline error (upload, render, or state mismatch).
    #[cfg(feature = "gpu")]
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Readback staging / mapping error.
    #[cfg(feature = "gpu")]
    #[error("readback: {0}")]
    Readback(#[from] RgbaReadbackError),
    /// Caller-facing configuration error (e.g. unsupported combination).
    #[error("config: {0}")]
    Config(String),
    /// The operation needs the GPU executor (streaming render,
    /// zero-copy import, GPU readback) but the engine is running the
    /// CPU executor. Route through the byte submit paths instead.
    #[error("operation requires the GPU executor; this engine runs the CPU executor")]
    RequiresGpu,
    /// GPU stacked-replay packer error (shader pipeline build, dim check).
    #[cfg(feature = "gpu")]
    #[error("stacked packer: {0}")]
    StackedPacker(#[from] PackerError),
}

/// Returned from every [`super::StitchCore::submit_frame`] /
/// `submit_frame_*_at_pose` call.
///
/// On the GPU executor, readback is triple-buffered: the first two
/// calls produce [`RenderOutcome::Warmup`] while the staging ring
/// fills; from the third call onward every submit produces
/// [`RenderOutcome::Rgba`] holding the tight RGBA bytes of the frame
/// submitted two frames ago. On the CPU executor the stitch is
/// synchronous: every submit returns [`RenderOutcome::Rgba`] for the
/// frame just submitted - `Warmup` never occurs.
pub enum RenderOutcome<'a> {
    /// Pipeline warm-up - submit more frames before expecting output.
    /// Only returned on the first two submit calls after construction.
    Warmup,
    /// A rendered panorama frame, tightly packed as RGBA
    /// (`output_width * output_height * 4` bytes). Borrowed from the
    /// core's internal staging; valid until the next submit call.
    Rgba(&'a [u8]),
}

/// A snapshot of one rendered panorama frame for the replay buffer.
///
/// The bytes are owned (not a borrow into the readback ring) because
/// the replay buffer outlives any single frame's staging slot.
#[derive(Debug, Clone)]
pub struct ReplayFrame {
    /// Tight RGBA bytes: `output_width * output_height * 4`.
    pub rgba: Vec<u8>,
    /// Monotonic timestamp captured at submit time (from the
    /// `StitchCore`'s session-start anchor).
    pub captured_at: Duration,
    /// Viewport pose the frame was rendered with. Useful for replay
    /// overlays that want to annotate where the camera pointed.
    pub pose: Pose,
}

/// Recorder hook for the push-API replay backend.
///
/// reco-core doesn't know about ffmpeg or the stacked-video file
/// format - this trait is the abstraction boundary so a concrete
/// implementation in reco-io (under the `stacked-output` feature)
/// can be plugged into [`super::StitchCore`] without pulling I/O types
/// into core. Consumers who only care about the pull API and go
/// through [`crate::session::StitchSession`] plus a
/// `reco_io::StitchJob::with_replay_recording(...)` builder never
/// touch this trait directly.
///
/// # Semantics
///
/// - `record_yuv` fires after every successful YUV submit via
///   [`super::StitchCore::submit_frame`] and
///   [`super::StitchCore::submit_frame_yuv_at_pose`]. It sees the tight
///   (no-stride) YUV420P planes the render consumed, so the
///   recorded replay exactly matches what the stitch pipeline saw.
/// - BGRA submit paths are not recorded today: the stacked
///   encoder is YUV-native, so recording BGRA frames would force
///   a BGRA-to-YUV420P conversion on the hot path. Skipped with a
///   one-shot `warn!`.
/// - `flush` and `finish` are best-effort; errors are logged by
///   the implementation and never propagated back to the submit
///   path so a failing recorder cannot break the stitch output.
///
/// # Thread safety
///
/// The recorder is owned by `StitchCore` (single-thread consumer
/// of the push API) so `Send` is sufficient; no `Sync`.
pub trait StackedReplayRecorder: Send {
    /// Record a stereo YUV420P tile pair. `width` / `height` are
    /// the tile dimensions for both cameras (identical).
    fn record_yuv(&mut self, left: &YuvPlanes<'_>, right: &YuvPlanes<'_>, width: u32, height: u32);
    /// Best-effort push buffered bytes to disk. Called on demand
    /// by the session (e.g. once per second) so a concurrent
    /// reader sees recent frames.
    fn flush(&mut self) {}
    /// Finalize the recording. Called when the session ends.
    /// After this call the recorder stops recording; subsequent
    /// `record_yuv` calls are no-ops.
    fn finish(&mut self) {}
}

/// Recorder hook for the GPU-pack replay path.
///
/// The GPU-pack path is chosen by
/// [`super::StitchCore::enable_gpu_stacked_replay`] when the source frames
/// are already on the GPU: the pack shader reads the renderer's
/// YUV textures into a tiled atlas and reads back a single
/// YUV420P buffer via a triple-buffered staging ring. Consumers
/// receive that buffer here, two frames after the submit that
/// produced it - mirroring the RGBA readback's lag.
///
/// Unlike [`StackedReplayRecorder`], this trait does NOT fire on
/// every submit; it fires when `YuvStackPacker::poll_ready`
/// returns a complete atlas. Early submits during the warm-up
/// (first two frames) produce no `record_atlas` call at all.
/// Path-choice is decided once per session at
/// [`super::StitchCore::enable_gpu_stacked_replay`] and logged explicitly
/// so CPU vs GPU packing is never a silent decision.
///
/// # Thread safety
///
/// Owned by `StitchCore` on the render thread; `Send` is enough.
#[cfg(feature = "gpu")]
pub trait StackedReplayGpuRecorder: Send {
    /// Receive a packed YUV420P atlas. The bytes live in
    /// `atlas.y / u / v`; dimensions are `atlas.width x atlas.height`
    /// (Y-plane). Called at most once per `submit_frame_*` call,
    /// and only when the triple-buffer produces a ready slot.
    fn record_atlas(&mut self, atlas: &StackedAtlas);
    /// Best-effort push buffered bytes to disk.
    fn flush(&mut self) {}
    /// Finalize the recording. Called when the session ends.
    fn finish(&mut self) {}
}
