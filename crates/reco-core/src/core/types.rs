//! Pure types and traits extracted from the `core` module.
//!
//! Houses error types, configuration, render outcome, replay frame, and
//! the recorder traits that decouple reco-core from I/O implementations.

use std::time::Duration;

use thiserror::Error;

use crate::calibration::MatchCalibration;
use crate::detect::director::ViewportPosition;
use crate::gpu::rgba_readback::RgbaReadbackError;
use crate::gpu::yuv_stack_packer::{PackerError, StackedAtlas};
use crate::projection::Projection;
use crate::render::pipeline::PipelineError;
use crate::render::planes::YuvPlanes;
use crate::render::renderer::InputFormat;
use crate::render::viewport::ViewportConfig;
use crate::source::CameraInput;

/// Errors from [`super::StitchCore`]. `Clone + Send + Sync` so consumers
/// posting render results to worker-thread channels carry the typed
/// error instead of stringifying at the boundary.
#[derive(Debug, Clone, Error)]
pub enum StitchCoreError {
    /// GPU pipeline error (upload, render, or state mismatch).
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Readback staging / mapping error.
    #[error("readback: {0}")]
    Readback(#[from] RgbaReadbackError),
    /// Caller-facing configuration error (e.g. unsupported combination).
    #[error("config: {0}")]
    Config(String),
    /// GPU stacked-replay packer error (shader pipeline build, dim check).
    #[error("stacked packer: {0}")]
    StackedPacker(#[from] PackerError),
}

/// Returned from every [`super::StitchCore::submit_frame_yuv`] /
/// [`super::StitchCore::submit_frame_bgra`] call.
///
/// The pipeline triple-buffers readback, so the first two calls produce
/// [`RenderOutcome::Warmup`] while the GPU fills the staging ring; from
/// the third call onward every submit produces
/// [`RenderOutcome::Rgba`] holding the tight RGBA bytes of the frame
/// submitted two frames ago.
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
    pub pose: ViewportPosition,
}

/// Configuration for building a [`super::StitchCore`].
///
/// Required fields: `calibration`, `input_width`, `input_height`,
/// `input_format`. Everything else has sensible defaults.
pub struct StitchCoreConfig {
    /// Camera calibration data.
    pub calibration: MatchCalibration,
    /// Output viewport (dimensions, blend width, FOV).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels (per camera).
    pub input_width: u32,
    /// Input frame height in pixels (per camera).
    pub input_height: u32,
    /// GPU render target format. `Rgba8Unorm` is the default and is
    /// what every compositor consumer needs; `Bgra8Unorm` matches
    /// native Windows DirectX surfaces for consumers that prefer to
    /// swizzle on upload instead of on readback.
    pub output_format: wgpu::TextureFormat,
    /// Input pixel format.
    pub input_format: InputFormat,
    /// Optional custom projection. Defaults to
    /// [`LShapeProjection`](crate::projection::LShapeProjection) - the
    /// 2-plane L-shape that matches today's geometric model.
    pub projection: Option<Box<dyn Projection>>,
    /// Optional camera-input marker. Defaults to
    /// [`StereoCameraInput`](crate::source::StereoCameraInput); future
    /// mono / N-input builds pick a different impl here.
    pub camera_input: Option<Box<dyn CameraInput>>,
    /// Opt-in replay ring buffer duration. `None` (default) keeps no
    /// history and allocates nothing for replay.
    pub replay_buffer_duration: Option<Duration>,
}

impl StitchCoreConfig {
    /// New config with required fields only; defaults everywhere else.
    pub fn new(
        calibration: MatchCalibration,
        input_width: u32,
        input_height: u32,
        input_format: InputFormat,
    ) -> Self {
        Self {
            calibration,
            viewport: ViewportConfig {
                width: 1920,
                height: 1080,
                blend_width: 0.15,
                ..Default::default()
            },
            input_width,
            input_height,
            output_format: wgpu::TextureFormat::Rgba8Unorm,
            input_format,
            projection: None,
            camera_input: None,
            replay_buffer_duration: None,
        }
    }
}

/// Recorder hook for the push-API replay backend (FRICTION A18 /
/// plan M6.5 item 3 on the push side).
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
///   [`super::StitchCore::submit_frame_yuv`] and
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

/// Recorder hook for the GPU-pack replay path (M7 pivot item 1).
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
