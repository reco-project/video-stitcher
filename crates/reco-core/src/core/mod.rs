//! `StitchCore` — push-first canonical entry point for the stitching engine.
//!
//! `StitchCore` is the M3 unification of what used to be two parallel
//! session APIs: `StitchSession` (pull, batch-oriented) and the former
//! `LiveStitchSession` (push, live-oriented, since deleted). Live sports
//! production is the primary use case, so the canonical API is push-based:
//! consumers call `StitchCore::submit_frame_yuv` / `submit_frame_bgra`
//! whenever a new frame pair is ready, and the core owns the pipeline,
//! readback, director, detection, coverage, and replay ring buffer.
//!
//! Batch file processing layers a thin pull-adapter on top
//! ([`StitchSession`](crate::session::StitchSession)::run).
//!
//! ## Foundation traits
//!
//! `StitchCore` composes the four M3 foundation traits that landed as
//! separate commits immediately before this one:
//!
//! - [`crate::projection::Projection`] — camera-geometry contract; today's
//!   L-shape is [`LShapeProjection`](crate::projection::LShapeProjection).
//! - [`crate::source::CameraInput`] — input-camera-count contract;
//!   [`StereoCameraInput`](crate::source::StereoCameraInput) is the
//!   current impl.
//! - [`crate::stage::PipelineStage`] — pluggable mid-pipeline transforms
//!   (color correction, exposure, remote compute shims).
//! - [`crate::detector::UnifiedDetector`] — collapsed CPU/CUDA/Metal
//!   detector contract with `DetectorError` for remote-inference futures.
//!
//! The first two are consumed at construction (see `StitchCoreConfig`).
//! `UnifiedDetector` is wired via `StitchCore::set_detector`; detection
//! runs on every `submit_frame_*` whose frame count is a multiple of
//! `StitchCore::detection_interval`, and raw detections are mapped to
//! panorama coordinates before reaching the director. `PipelineStage`
//! slots in via `StitchCore::push_pipeline_stage` but has no registered
//! stages yet.
//!
//! ## Usage (push, live)
//!
//! ```rust,ignore
//! use reco_core::core::{StitchCore, StitchCoreConfig, RenderOutcome};
//!
//! let mut core = StitchCore::new(gpu, StitchCoreConfig {
//!     calibration,
//!     viewport,
//!     input_width, input_height,
//!     output_format: wgpu::TextureFormat::Rgba8Unorm,
//!     input_format: InputFormat::Yuv420p,
//!     ..Default::default()
//! })?;
//!
//! core.set_director(Box::new(MyDirector::new()));
//!
//! // Per-frame, from the consumer's callback thread:
//! match core.submit_frame_yuv(&left, &right)? {
//!     RenderOutcome::Rgba(bytes) => compositor.upload(bytes),
//!     RenderOutcome::Warmup => { /* first two frames: pipeline filling */ }
//! }
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::calibration::MatchCalibration;
use crate::detector::{
    CameraId, ChromaFormat, Detection, DetectorFrame, RawFrame, UnifiedDetector,
};
use crate::director::{MappedDetection, ViewportPosition};
use crate::gpu::GpuContext;
use crate::panner::Panner;
use crate::pipeline::{BgraPlanes, PipelineError, StitchPipeline, YuvPlanes};
use crate::projection::{self, CoverageBoundary, LShapeProjection, PanoramaExtent, Projection};
use crate::renderer::InputFormat;
use crate::rgba_readback::{RgbaReadback, RgbaReadbackError};
use crate::source::{CameraInput, StereoCameraInput};
use crate::stage::PipelineStage;
use crate::tracker::Tracker;
use crate::viewport::ViewportConfig;
use crate::yuv_stack_packer::{
    OutputTileSize, PackerError, SourceFormat, StackGridLayout, StackedAtlas, StackedPackSource,
    YuvStackPacker,
};

/// Errors from [`StitchCore`]. `Clone + Send + Sync` so consumers
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

/// Returned from every [`StitchCore::submit_frame_yuv`] /
/// [`StitchCore::submit_frame_bgra`] call.
///
/// The pipeline triple-buffers readback, so the first two calls produce
/// [`RenderOutcome::Warmup`] while the GPU fills the staging ring; from
/// the third call onward every submit produces
/// [`RenderOutcome::Rgba`] holding the tight RGBA bytes of the frame
/// submitted two frames ago.
pub enum RenderOutcome<'a> {
    /// Pipeline warm-up — submit more frames before expecting output.
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

/// Bounded-duration ring of recently-rendered panorama frames.
///
/// Solves FRICTION A16 (OBS replay). Opt-in via
/// [`StitchCore::enable_replay_buffer`]; when disabled the core
/// allocates nothing for replay and `submit_frame_*` does zero extra
/// work. Ring trimming runs per-submit and is `O(frames_evicted)`.
pub struct ReplayBuffer {
    frames: VecDeque<ReplayFrame>,
    max_duration: Duration,
}

impl ReplayBuffer {
    fn new(max_duration: Duration) -> Self {
        Self {
            frames: VecDeque::new(),
            max_duration,
        }
    }

    fn push(&mut self, frame: ReplayFrame) {
        self.frames.push_back(frame);
        // Evict from the front until the oldest kept frame is within
        // max_duration of the newest. Using wrapping subtraction on
        // Duration is not allowed, so compare directly.
        let cutoff = self
            .frames
            .back()
            .map(|f| f.captured_at)
            .unwrap_or_default()
            .saturating_sub(self.max_duration);
        while let Some(front) = self.frames.front() {
            if front.captured_at < cutoff {
                self.frames.pop_front();
            } else {
                break;
            }
        }
    }

    /// Number of frames currently buffered.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the buffer holds zero frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Maximum age of retained frames (set at enable time).
    pub fn max_duration(&self) -> Duration {
        self.max_duration
    }

    /// Iterate buffered frames oldest-to-newest.
    pub fn iter(&self) -> impl Iterator<Item = &ReplayFrame> {
        self.frames.iter()
    }

    /// Most recently buffered frame, if any.
    pub fn latest(&self) -> Option<&ReplayFrame> {
        self.frames.back()
    }

    /// Oldest buffered frame, if any. Useful for consumers that want
    /// to know the effective buffered duration
    /// (`latest.captured_at - oldest.captured_at`).
    pub fn oldest(&self) -> Option<&ReplayFrame> {
        self.frames.front()
    }

    /// The effective buffered duration: the difference between
    /// oldest and newest frame timestamps. Returns `Duration::ZERO`
    /// for empty or single-frame buffers.
    pub fn buffered_duration(&self) -> Duration {
        match (self.frames.front(), self.frames.back()) {
            (Some(first), Some(last)) => last.captured_at.saturating_sub(first.captured_at),
            _ => Duration::ZERO,
        }
    }

    /// Drop every buffered frame without changing `max_duration`.
    /// Consumers wire this to a "Clear replay" UI button so the user
    /// can start a fresh replay window after an event.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Clone every buffered frame into an owned vector.
    ///
    /// Used by consumers that want to ship the replay off the render
    /// thread (to disk, to a "Save replay" dialog, to a network
    /// stream). The buffer itself keeps the frames, so the consumer
    /// can keep recording while it exports a snapshot.
    ///
    /// Returns the vector in oldest-to-newest order, matching
    /// [`Self::iter`].
    pub fn snapshot(&self) -> Vec<ReplayFrame> {
        self.frames.iter().cloned().collect()
    }

    /// Drain every buffered frame into an owned vector, leaving the
    /// buffer empty. Same ordering contract as [`Self::snapshot`].
    /// Unlike `snapshot`, this transfers ownership — no clone cost
    /// for consumers that are about to discard the buffer anyway
    /// (e.g. a "Save + reset" UI flow).
    pub fn take(&mut self) -> Vec<ReplayFrame> {
        self.frames.drain(..).collect()
    }
}

/// Configuration for building a [`StitchCore`].
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
    /// [`LShapeProjection`] — the 2-plane L-shape that matches today's
    /// geometric model.
    pub projection: Option<Box<dyn Projection>>,
    /// Optional camera-input marker. Defaults to
    /// [`StereoCameraInput`]; future mono / N-input builds pick a
    /// different impl here.
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

/// Canonical push-first stitching core.
///
/// See the module-level docs for design rationale. `StitchCore` owns:
///
/// - A [`StitchPipeline`] for the GPU render work.
/// - An [`RgbaReadback`] triple-buffered staging ring for CPU delivery.
/// - A coverage boundary precomputed from calibration for `safe_clamp`.
/// - The active [`Projection`] and [`CameraInput`] (for future
///   N-input / alt-projection variants).
/// - Optional [`Tracker`]s and an optional [`Panner`] that together
///   drive the viewport pose, plus a pipeline-stage chain.
/// - An optional [`ReplayBuffer`].
///
/// Detection is wired through the [`UnifiedDetector`] trait: attach
/// one via [`StitchCore::set_detector`] and the core will run it on every
/// CPU-resident frame submitted (CUDA / Metal residency dispatch
/// lands in a later tranche that adds GPU-frame `submit_*` methods).
/// Raw detections are mapped to panorama coordinates and fed to the
/// attached director each submit; directors see a non-empty
/// `detections` slice on detection frames, empty otherwise.
/// Recorder hook for the push-API replay backend (FRICTION A18 /
/// plan M6.5 item 3 on the push side).
///
/// reco-core doesn't know about ffmpeg or the stacked-video file
/// format - this trait is the abstraction boundary so a concrete
/// implementation in reco-io (under the `stacked-output` feature)
/// can be plugged into [`StitchCore`] without pulling I/O types
/// into core. Consumers who only care about the pull API and go
/// through [`crate::session::StitchSession`] plus a
/// `reco_io::StitchJob::with_replay_recording(...)` builder never
/// touch this trait directly.
///
/// # Semantics
///
/// - `record_yuv` fires after every successful YUV submit via
///   [`StitchCore::submit_frame_yuv`] and
///   [`StitchCore::submit_frame_yuv_at_pose`]. It sees the tight
///   (no-stride) YUV420P planes the render consumed, so the
///   recorded replay exactly matches what the stitch pipeline saw.
/// - BGRA submit paths are not recorded today: the stacked
///   encoder is YUV-native, so recording BGRA frames would force
///   a BGRA→YUV420P conversion on the hot path. Skipped with a
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
/// [`StitchCore::enable_gpu_stacked_replay`] when the source frames
/// are already on the GPU: the pack shader reads the renderer's
/// YUV textures into a tiled atlas and reads back a single
/// YUV420P buffer via a triple-buffered staging ring. Consumers
/// receive that buffer here, two frames after the submit that
/// produced it — mirroring the RGBA readback's lag.
///
/// Unlike [`StackedReplayRecorder`], this trait does NOT fire on
/// every submit; it fires when `YuvStackPacker::poll_ready`
/// returns a complete atlas. Early submits during the warm-up
/// (first two frames) produce no `record_atlas` call at all.
/// Path-choice is decided once per session at
/// [`StitchCore::enable_gpu_stacked_replay`] and logged explicitly
/// so CPU vs GPU packing is never a silent decision.
///
/// # Thread safety
///
/// Owned by `StitchCore` on the render thread; `Send` is enough.
pub trait StackedReplayGpuRecorder: Send {
    /// Receive a packed YUV420P atlas. The bytes live in
    /// `atlas.y / u / v`; dimensions are `atlas.width × atlas.height`
    /// (Y-plane). Called at most once per `submit_frame_*` call,
    /// and only when the triple-buffer produces a ready slot.
    fn record_atlas(&mut self, atlas: &StackedAtlas);
    /// Best-effort push buffered bytes to disk.
    fn flush(&mut self) {}
    /// Finalize the recording. Called when the session ends.
    fn finish(&mut self) {}
}

pub struct StitchCore {
    pipeline: StitchPipeline,
    readback: RgbaReadback,
    output_width: u32,
    output_height: u32,

    projection: Box<dyn Projection>,
    camera_input: Box<dyn CameraInput>,
    stages: Vec<Box<dyn PipelineStage>>,

    coverage: Option<CoverageBoundary>,

    /// Per-class trackers that feed a shared [`WorldState`](crate::tracker::WorldState)
    /// consumed by [`StitchCore::panner`]. Slot-based on purpose:
    /// `ball_tracker` fills `world.ball`, `player_tracker` fills
    /// `world.players`. More slots land with future entity classes.
    ///
    /// The panner only runs when at least one tracker is registered
    /// AND a panner is set. Otherwise the pose stays at the pipeline
    /// default.
    ball_tracker: Option<Box<dyn Tracker>>,
    player_tracker: Option<Box<dyn Tracker>>,
    /// Camera-motion policy. Consumes the assembled
    /// [`WorldState`](crate::tracker::WorldState) each frame and emits
    /// a [`ViewportPosition`]. When unset, the pose stays at the
    /// pipeline default.
    panner: Option<Box<dyn Panner>>,
    /// Previous frame's resolved pose, passed to the panner in its
    /// [`PanContext`](crate::panner::PanContext) so panners can
    /// compute first-order motion deltas statelessly.
    previous_panner_pose: ViewportPosition,

    detector: Option<Box<dyn UnifiedDetector>>,
    /// How often detection runs. 1 = every frame (default), higher =
    /// skip frames. On skipped frames the director still ticks with
    /// the previously tracked detections.
    detection_interval: u64,
    /// Panorama-mapped detections from the last detection frame.
    /// Reused on skipped frames so the director retains context.
    last_detections: Vec<MappedDetection>,

    replay: Option<ReplayBuffer>,

    /// Optional stacked-video replay recorder attached via
    /// [`Self::set_stacked_recorder`]. Fires on every successful
    /// YUV submit (not BGRA — see [`StackedReplayRecorder`] docs).
    /// Decouples reco-core from the actual encoder implementation
    /// (lives in reco-io under `stacked-output`) so mobile / wasm
    /// builds that skip reco-io see no replay-recording code.
    stacked_recorder: Option<Box<dyn StackedReplayRecorder>>,

    /// Optional GPU-pack packer attached via
    /// [`Self::enable_gpu_stacked_replay`]. Holds the compute
    /// pipelines and triple-buffered staging ring. `None` when the
    /// session runs on a CPU-pack (or no replay) path.
    stacked_packer: Option<YuvStackPacker>,

    /// Optional GPU-pack atlas recorder attached via
    /// [`Self::set_stacked_gpu_recorder`]. Receives the packed atlas
    /// bytes every time [`YuvStackPacker::poll_ready`] yields a
    /// completed readback slot. `None` means the pack still runs
    /// (if enabled) but the bytes are dropped — useful when a
    /// consumer wants to attach the recorder lazily.
    stacked_gpu_recorder: Option<Box<dyn StackedReplayGpuRecorder>>,

    /// Whether `resolve_current_pose` clamps output through the
    /// coverage boundary (FRICTION A13 — "constrained look"). `true`
    /// by default so the viewport never reveals black panorama
    /// edges; toggle off when the user wants to explore the raw
    /// panorama space (e.g. to find the edge of coverage during
    /// debugging or a cinematographic effect).
    ///
    /// The public [`Self::safe_clamp`] method remains available
    /// regardless of this flag — it's the primitive consumers use
    /// for ad-hoc clamping outside the render loop.
    constrained_look: bool,

    frame_count: u64,
    session_start: Option<Instant>,
}

impl StitchCore {
    /// Build a new core. Owns the supplied [`GpuContext`].
    pub fn new(gpu: GpuContext, config: StitchCoreConfig) -> Result<Self, StitchCoreError> {
        let output_width = config.viewport.width;
        let output_height = config.viewport.height;

        let pipeline = StitchPipeline::with_gpu(
            gpu,
            config.calibration,
            config.viewport,
            config.input_width,
            config.input_height,
            config.output_format,
            config.input_format,
        )?;

        let readback = RgbaReadback::new(pipeline.gpu(), output_width, output_height)?;

        let coverage = CoverageBoundary::from_calibration(pipeline.calibration(), &pipeline.scene);

        let projection: Box<dyn Projection> = config
            .projection
            .unwrap_or_else(|| Box::new(LShapeProjection));
        let camera_input: Box<dyn CameraInput> = config
            .camera_input
            .unwrap_or_else(|| Box::new(StereoCameraInput));

        if camera_input.camera_count() != projection.camera_count() {
            return Err(StitchCoreError::Config(format!(
                "camera_input.camera_count() = {} but projection.camera_count() = {}; \
                 these must match so StitchCore receives one frame per camera plane",
                camera_input.camera_count(),
                projection.camera_count(),
            )));
        }

        let replay = config.replay_buffer_duration.map(ReplayBuffer::new);

        Ok(Self {
            pipeline,
            readback,
            output_width,
            output_height,
            projection,
            camera_input,
            stages: Vec::new(),
            coverage: Some(coverage),
            ball_tracker: None,
            player_tracker: None,
            panner: None,
            previous_panner_pose: ViewportPosition::default(),
            detector: None,
            detection_interval: 1,
            last_detections: Vec::new(),
            replay,
            stacked_recorder: None,
            stacked_packer: None,
            stacked_gpu_recorder: None,
            constrained_look: true,
            frame_count: 0,
            session_start: None,
        })
    }

    // -----------------------------------------------------------------
    // Submit / render
    // -----------------------------------------------------------------

    /// Submit a stereo YUV420P frame pair and render the current pose.
    ///
    /// Uses the director (if attached) and coverage clamping to pick
    /// the viewport, renders, and reads back RGBA. The first two calls
    /// produce [`RenderOutcome::Warmup`] while the triple-buffered
    /// staging ring fills; from the third call onward every submit
    /// yields RGBA bytes from two frames ago.
    pub fn submit_frame_yuv(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        // Feed the stacked-video replay recorder before render so
        // the recording captures the exact planes the pipeline will
        // see. Errors inside the recorder are logged by the impl;
        // never propagate them - a failing recorder must not break
        // the live stitch output.
        if let Some(ref mut recorder) = self.stacked_recorder {
            let (src_w, src_h) = self.pipeline.source_info();
            recorder.record_yuv(left, right, src_w, src_h);
        }

        // Detection first, so the director's `update` tick in
        // resolve_current_pose sees the latest tracked objects. Skipped
        // frames reuse last_detections so the director still has context.
        let ran_detection = self.detector.is_some() && self.should_run_detection();
        if ran_detection {
            let (src_w, src_h) = self.pipeline.source_info();
            let dets = self.run_yuv_detection(left, right, src_w, src_h);
            self.last_detections = self.map_detections_to_panorama(dets);
        }

        let pose = self.resolve_current_pose(ran_detection);
        let cmd = self
            .pipeline
            .render_to_target(left, right, pose.yaw, pose.pitch)?;
        // GPU stacked-replay pack runs before the readback so the
        // borrow checker sees `self.readback` free while the pack
        // runs. Queue ordering: `queue.write_texture` inside
        // `render_to_target` is already enqueued; the pack submit
        // processes the writes before its compute pass reads the
        // textures, and the subsequent stitch submit reads the
        // same textures into the render target. No-op when packer
        // is not enabled.
        self.drive_gpu_stacked_pack();
        // Split-borrow: push_replay only accesses self.replay +
        // self.session_start; self.readback keeps the rgba slice
        // alive. Inlining the replay push (instead of going through
        // `&mut self` on a helper) lets the borrow checker see the
        // fields are disjoint.
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        if let (Some(replay), Some(bytes)) = (self.replay.as_mut(), rgba) {
            replay.push(ReplayFrame {
                rgba: bytes.to_vec(),
                captured_at,
                pose,
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Submit a stereo YUV420P frame pair at an explicit pose.
    ///
    /// Same full loop as [`Self::submit_frame_yuv`] — anchors the
    /// session-start clock, runs detection when
    /// `frame_count % detection_interval == 0`, renders, reads back
    /// RGBA, pushes into the replay buffer, increments frame_count —
    /// but bypasses the director and uses the caller-supplied
    /// `(yaw, pitch)` directly. The FOV stays at whatever the
    /// pipeline currently has (set via [`Self::pipeline_mut`] or
    /// `update_calibration`).
    ///
    /// This is the canonical submit path for interactive UIs (OBS
    /// pan/zoom sliders, mouse-drag preview) where pose comes from
    /// user input rather than a director.
    pub fn submit_frame_yuv_at_pose(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        // Replay recording tap — see `submit_frame_yuv` for the
        // rationale (record-before-render so the file exactly
        // matches what the pipeline consumed).
        if let Some(ref mut recorder) = self.stacked_recorder {
            let (src_w, src_h) = self.pipeline.source_info();
            recorder.record_yuv(left, right, src_w, src_h);
        }

        // `submit_frame_yuv_at_pose` bypasses resolve_current_pose (caller
        // provides the pose directly), but detection still runs on the
        // schedule so directors stay populated for a later `current_pose()`
        // peek or a regular `submit_frame_yuv` submit.
        if self.detector.is_some() && self.should_run_detection() {
            let (src_w, src_h) = self.pipeline.source_info();
            let dets = self.run_yuv_detection(left, right, src_w, src_h);
            self.last_detections = self.map_detections_to_panorama(dets);
        }

        let cmd = self.pipeline.render_to_target(left, right, yaw, pitch)?;
        // GPU stacked-replay pack — see `submit_frame_yuv` for
        // ordering rationale. No-op when not enabled.
        self.drive_gpu_stacked_pack();
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        if let (Some(replay), Some(bytes)) = (self.replay.as_mut(), rgba) {
            replay.push(ReplayFrame {
                rgba: bytes.to_vec(),
                captured_at,
                pose: ViewportPosition {
                    yaw,
                    pitch,
                    fov_degrees: None,
                },
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Submit a stereo BGRA frame pair at an explicit pose. See
    /// [`Self::submit_frame_yuv_at_pose`] for semantics.
    ///
    /// Does not run detection (BGRA backends are not yet supported;
    /// see [`Self::submit_frame_bgra`] for the rationale).
    pub fn submit_frame_bgra_at_pose(
        &mut self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();
        let cmd = self
            .pipeline
            .render_to_target_bgra(left, right, yaw, pitch)?;
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        if let (Some(replay), Some(bytes)) = (self.replay.as_mut(), rgba) {
            replay.push(ReplayFrame {
                rgba: bytes.to_vec(),
                captured_at,
                pose: ViewportPosition {
                    yaw,
                    pitch,
                    fov_degrees: None,
                },
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Submit a stereo packed-RGBA/BGRA frame pair and render the
    /// current pose.
    ///
    /// Requires the core to have been built with [`InputFormat::Bgra`].
    /// See [`Self::submit_frame_yuv`] for return semantics.
    pub fn submit_frame_bgra(
        &mut self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
    ) -> Result<RenderOutcome<'_>, StitchCoreError> {
        self.anchor_session_start();

        // BGRA detection path: YOLO backends today consume YUV or
        // NV12 `RawFrame` variants. Wrapping BGRA bytes as a YUV
        // frame would require a color-space conversion we're not
        // paying for yet - consumers that want detection on BGRA
        // sources (OBS Browser Source, screen capture) attach a
        // detector that understands BGRA once such a backend exists.
        // For now, BGRA submits tick the director with the last
        // detections (potentially from earlier YUV submits) but do
        // not run detection themselves.

        // `fresh_detection = false`: BGRA submits never run detection by
        // design (see comment above). Directors must see this frame as
        // "reusing cached detections" even on interval ticks, otherwise
        // hysteresis counters over-fire.
        let pose = self.resolve_current_pose(false);
        let cmd = self
            .pipeline
            .render_to_target_bgra(left, right, pose.yaw, pose.pitch)?;
        let captured_at = self.session_start.map(|s| s.elapsed()).unwrap_or_default();
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        self.frame_count += 1;
        if let (Some(replay), Some(bytes)) = (self.replay.as_mut(), rgba) {
            replay.push(ReplayFrame {
                rgba: bytes.to_vec(),
                captured_at,
                pose,
            });
        }
        Ok(match rgba {
            Some(bytes) => RenderOutcome::Rgba(bytes),
            None => RenderOutcome::Warmup,
        })
    }

    /// Drain one pending readback slot without submitting a new frame.
    ///
    /// Useful at shutdown to collect the 1-2 frames still in-flight in
    /// the triple-buffered staging pipeline.
    pub fn flush(&mut self) -> Result<Option<&[u8]>, StitchCoreError> {
        Ok(self.readback.flush_pending(self.pipeline.gpu())?)
    }

    // -----------------------------------------------------------------
    // Low-level render-at-pose methods
    //
    // These produce a `wgpu::CommandBuffer` at a **caller-supplied pose**
    // without running detection, without ticking the director, and
    // without performing RGBA readback. They exist so consumers that
    // need the rendered GPU texture as input to further GPU work
    // (NV12 conversion for encoding, compositor texture import) can
    // drive the core without paying for readback.
    //
    // The M3 `StitchSession::run` pull-adapter (plan step 2) uses these
    // to route its encode loop through `StitchCore`: session owns its
    // own director + detection pipeline during the transition and
    // passes the resolved pose explicitly here. Once the session
    // migration completes, these remain as the "render primitives" for
    // multi-output consumers (record + stream, zero-copy compositor).
    // -----------------------------------------------------------------

    /// Render a stereo YUV420P frame at an explicit pose.
    ///
    /// Does not run detection, does not tick the director, does not
    /// read back RGBA. Consumers that want the full `submit_*` loop
    /// (detection + director + readback) should call
    /// [`Self::submit_frame_yuv`] instead.
    ///
    /// The caller is responsible for subsequently consuming the
    /// rendered texture (via [`Self::pipeline`] + `render_target()`)
    /// or submitting the returned command buffer to chain further
    /// GPU work.
    pub fn render_yuv_at_pose(
        &self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self.pipeline.render_to_target(left, right, yaw, pitch)?)
    }

    /// Render a stereo packed-RGBA/BGRA frame at an explicit pose.
    /// See [`Self::render_yuv_at_pose`] for semantics.
    pub fn render_bgra_at_pose(
        &self,
        left: &BgraPlanes<'_>,
        right: &BgraPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self
            .pipeline
            .render_to_target_bgra(left, right, yaw, pitch)?)
    }

    /// Render from GPU-resident RGBA textures (e.g. Bayer demosaic output).
    ///
    /// Copies the demosaiced textures into the stitch pipeline's input
    /// planes (GPU-to-GPU blit), then renders the stitch. Returns the
    /// render command buffer for `submit_render_output`.
    pub fn render_gpu_rgba_at_pose(
        &self,
        left_rgba: &wgpu::Texture,
        right_rgba: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_from_gpu_rgba(left_rgba, right_rgba, yaw, pitch)
    }

    /// Render any [`StereoFrame`](crate::source::StereoFrame) variant
    /// (YUV / NV12 / GpuResident) at an explicit pose.
    ///
    /// Thin wrapper over
    /// [`StitchPipeline::render_stereo_frame`](crate::pipeline::StitchPipeline::render_stereo_frame)
    /// that converts the pipeline error into a `StitchCoreError`. The
    /// `MetalResident` variant is NOT handled here; use
    /// [`Self::render_imported_textures_at_pose`] after importing the
    /// `CVPixelBuffer` via `MetalTextureCache`.
    pub fn render_stereo_frame_at_pose(
        &self,
        frame: &crate::source::StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<wgpu::CommandBuffer, StitchCoreError> {
        Ok(self.pipeline.render_stereo_frame(frame, yaw, pitch)?)
    }

    /// Render from four pre-imported textures at an explicit pose.
    ///
    /// Used by the macOS zero-copy path where `CVPixelBuffer` Y/UV
    /// planes are imported as wgpu textures via `MetalTextureCache`
    /// (in `metal_interop`), and the Linux zero-copy path that shares
    /// textures through the bind-group variant below.
    pub fn render_imported_textures_at_pose(
        &mut self,
        left_y: &wgpu::Texture,
        left_uv: &wgpu::Texture,
        right_y: &wgpu::Texture,
        right_uv: &wgpu::Texture,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_imported_textures(left_y, left_uv, right_y, right_uv, yaw, pitch)
    }

    /// Render from pre-built GPU texture views at an explicit pose.
    ///
    /// Used by the D3D11VA zero-copy path where NV12 plane views are
    /// created with `TextureAspect::Plane0` / `Plane1`.
    pub fn render_imported_views_at_pose(
        &mut self,
        left_y: &wgpu::TextureView,
        left_uv: &wgpu::TextureView,
        right_y: &wgpu::TextureView,
        right_uv: &wgpu::TextureView,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_imported_views(left_y, left_uv, right_y, right_uv, yaw, pitch)
    }

    /// Render from pre-configured GPU bind groups and decode slots at
    /// an explicit pose (Linux zero-copy path).
    ///
    /// Thin wrapper over
    /// [`StitchPipeline::render_gpu_frame`](crate::pipeline::StitchPipeline::render_gpu_frame).
    /// Consumers must have already called
    /// [`StitchPipeline::configure_gpu_source`] via [`Self::pipeline_mut`].
    #[cfg(target_os = "linux")]
    pub fn render_gpu_frame_at_pose(
        &mut self,
        bind_groups: &crate::pipeline::GpuSourceBindGroups,
        left_slot: u8,
        right_slot: u8,
        yaw: f32,
        pitch: f32,
    ) -> wgpu::CommandBuffer {
        self.pipeline
            .render_gpu_frame(bind_groups, left_slot, right_slot, yaw, pitch)
    }

    // -----------------------------------------------------------------
    // Tracker / panner wiring
    // -----------------------------------------------------------------

    /// Attach a singleton ball tracker. Replaces any existing one.
    ///
    /// The tracker only drives the pose when a [`Panner`] is also
    /// attached via [`set_panner`](Self::set_panner); attached without
    /// a panner it still runs so detection sinks see consistent output
    /// but the pose stays at the pipeline default.
    pub fn set_ball_tracker(&mut self, tracker: Box<dyn Tracker>) {
        log::info!(
            "StitchCore: ball tracker attached (class_id={})",
            tracker.class_id()
        );
        self.ball_tracker = Some(tracker);
    }

    /// Remove the currently attached ball tracker.
    pub fn clear_ball_tracker(&mut self) {
        self.ball_tracker = None;
    }

    /// Attach a multi-entity player tracker. Replaces any existing one.
    ///
    /// The tracker's output populates
    /// [`WorldState::players`](crate::tracker::WorldState::players)
    /// each frame. Phase-5 implementation — until that phase lands,
    /// this setter is usable from consumers but typically left unset.
    pub fn set_player_tracker(&mut self, tracker: Box<dyn Tracker>) {
        log::info!(
            "StitchCore: player tracker attached (class_id={})",
            tracker.class_id()
        );
        self.player_tracker = Some(tracker);
    }

    /// Remove the currently attached player tracker.
    pub fn clear_player_tracker(&mut self) {
        self.player_tracker = None;
    }

    /// Attach a panner. Replaces any existing one.
    ///
    /// Each frame, `resolve_current_pose` runs the registered trackers,
    /// builds a [`WorldState`], and delegates to [`Panner::decide`].
    /// Without a panner the pose stays at the pipeline default.
    ///
    /// [`WorldState`]: crate::tracker::WorldState
    pub fn set_panner(&mut self, panner: Box<dyn Panner>) {
        log::info!("StitchCore: panner attached");
        self.panner = Some(panner);
    }

    /// Remove the currently attached panner. Pose reverts to the
    /// pipeline default until a new panner is set.
    pub fn clear_panner(&mut self) {
        log::info!("StitchCore: panner detached");
        self.panner = None;
    }

    /// Attach a unified-trait detector. Replaces any existing one.
    ///
    /// The detector runs on every `submit_frame_*` call whose frame
    /// count matches [`Self::detection_interval`]. Raw detections are
    /// mapped to panorama coordinates, then handed to each registered
    /// [`Tracker`] and to the detection sink (if any). Detection errors
    /// are logged (at `warn!` level) and swallowed so a transient
    /// inference failure does not abort the render loop.
    pub fn set_detector(&mut self, detector: Box<dyn UnifiedDetector>) {
        self.detector = Some(detector);
    }

    /// Remove the currently attached detector. Cached last detections
    /// are cleared so the director does not keep seeing stale data.
    pub fn clear_detector(&mut self) {
        self.detector = None;
        self.last_detections.clear();
    }

    /// Attach a stacked-video replay recorder (M6.5 item 3, push
    /// side).
    ///
    /// Every subsequent YUV submit feeds the recorder before
    /// rendering. Errors inside the recorder are swallowed so a
    /// failing recorder cannot break the live stitch output; the
    /// recorder's own implementation is expected to log any
    /// failure. See [`StackedReplayRecorder`] for the full
    /// contract.
    ///
    /// Dropping an existing recorder via [`Self::clear_stacked_recorder`]
    /// is required before attaching a new one; otherwise the old
    /// recording is quietly abandoned.
    pub fn set_stacked_recorder(&mut self, recorder: Box<dyn StackedReplayRecorder>) {
        if self.stacked_recorder.is_some() {
            log::warn!(
                "StitchCore::set_stacked_recorder replacing an existing recorder; \
                 call clear_stacked_recorder first to finalize the previous recording"
            );
        }
        log::info!("StitchCore: stacked-video replay recorder attached");
        self.stacked_recorder = Some(recorder);
    }

    /// Drop the current replay recorder, calling `finish` first so
    /// the recording file is finalized. No-op if no recorder is
    /// attached.
    pub fn clear_stacked_recorder(&mut self) {
        if let Some(mut recorder) = self.stacked_recorder.take() {
            recorder.finish();
            log::info!("StitchCore: stacked-video replay recorder detached");
        }
    }

    /// Flush the replay recorder's buffered bytes to disk. Call
    /// periodically (e.g. once per second from a timer) so a
    /// concurrent reader sees recent frames. No-op if no recorder
    /// is attached.
    pub fn flush_stacked_recorder(&mut self) {
        if let Some(ref mut recorder) = self.stacked_recorder {
            recorder.flush();
        }
    }

    /// Enable the GPU-pack replay path (M7 pivot item 1).
    ///
    /// Builds a [`YuvStackPacker`] sized for `layout` × `output_size`
    /// and wires it into subsequent YUV submit calls. The packer's
    /// source-format variant is derived from the pipeline's input
    /// format so consumers don't risk a YUV/NV12 mismatch.
    ///
    /// Call [`Self::set_stacked_gpu_recorder`] to attach an
    /// atlas-consuming sink (typically a
    /// [`reco_io`](../../../reco_io/index.html) encoder) before the
    /// first submit, or later — the pack still runs either way and
    /// the first two submits are warmup.
    ///
    /// Emits one `log::info!` line stating the pack path has been
    /// chosen (GPU), the tile dims, `N`, and the source format — so
    /// the CPU vs GPU decision is never silent.
    ///
    /// Returns `StitchCoreError::Config` when the pipeline's input
    /// format is BGRA (the pack shader only handles YUV420P / NV12)
    /// or when the layout / output dims violate YUV420P alignment.
    pub fn enable_gpu_stacked_replay(
        &mut self,
        layout: StackGridLayout,
        output_size: OutputTileSize,
    ) -> Result<(), StitchCoreError> {
        let source_format = match self.pipeline.input_format() {
            InputFormat::Yuv420p => SourceFormat::Yuv420p,
            InputFormat::Nv12 => SourceFormat::Nv12,
            InputFormat::Bgra => {
                return Err(StitchCoreError::Config(
                    "GPU stacked replay requires YUV420P or NV12 input; BGRA pipelines must use \
                     the CPU replay-recording path"
                        .into(),
                ));
            }
        };
        let packer = YuvStackPacker::new(self.pipeline.gpu(), layout, output_size, source_format)?;
        let (atlas_w, atlas_h) = packer.atlas_dims();
        log::info!(
            "reco-core: replay pack path = GPU shader (tiles {}x{} out, N={}, atlas {}x{}, source_format={:?})",
            output_size.width,
            output_size.height,
            layout.capacity(),
            atlas_w,
            atlas_h,
            source_format,
        );
        if self.stacked_recorder.is_some() {
            log::warn!(
                "StitchCore::enable_gpu_stacked_replay: a CPU StackedReplayRecorder is also \
                 attached; both paths will run and duplicate work. Clear one to avoid \
                 redundant recording."
            );
        }
        self.stacked_packer = Some(packer);
        Ok(())
    }

    /// Disable the GPU-pack replay path and drop the packer.
    /// Also calls `finish` on any attached GPU recorder so its file
    /// is finalized. No-op when the path was not enabled.
    pub fn disable_gpu_stacked_replay(&mut self) {
        if self.stacked_packer.take().is_some() {
            log::info!("StitchCore: GPU stacked replay disabled");
        }
        self.clear_stacked_gpu_recorder();
    }

    /// Attach a GPU-pack atlas recorder. Must be called after
    /// [`Self::enable_gpu_stacked_replay`] for the pack output to
    /// reach disk — without a recorder the packer still runs but
    /// the readback bytes are dropped.
    pub fn set_stacked_gpu_recorder(&mut self, recorder: Box<dyn StackedReplayGpuRecorder>) {
        if self.stacked_gpu_recorder.is_some() {
            log::warn!(
                "StitchCore::set_stacked_gpu_recorder replacing an existing GPU recorder; \
                 call clear_stacked_gpu_recorder first to finalize the previous recording"
            );
        }
        if self.stacked_packer.is_none() {
            log::warn!(
                "StitchCore::set_stacked_gpu_recorder called before \
                 enable_gpu_stacked_replay: recorder will receive no atlases until the \
                 packer is enabled"
            );
        }
        log::info!("StitchCore: GPU stacked-replay recorder attached");
        self.stacked_gpu_recorder = Some(recorder);
    }

    /// Drop the GPU-pack atlas recorder, calling `finish` so the
    /// output file is finalized. No-op if no recorder is attached.
    pub fn clear_stacked_gpu_recorder(&mut self) {
        if let Some(mut recorder) = self.stacked_gpu_recorder.take() {
            recorder.finish();
            log::info!("StitchCore: GPU stacked-replay recorder detached");
        }
    }

    /// Flush the GPU recorder's buffered bytes to disk. No-op if no
    /// recorder is attached.
    pub fn flush_stacked_gpu_recorder(&mut self) {
        if let Some(ref mut recorder) = self.stacked_gpu_recorder {
            recorder.flush();
        }
    }

    /// Atlas dimensions `(width, height)` the current packer produces,
    /// or `None` when GPU stacked replay is not enabled. Consumers
    /// use this to open an encoder sized for the atlas.
    pub fn stacked_atlas_dims(&self) -> Option<(u32, u32)> {
        self.stacked_packer.as_ref().map(|p| p.atlas_dims())
    }

    /// Pack the GPU stacked-replay atlas from external texture
    /// views (the zero-copy entry point).
    ///
    /// Used by session-layer zero-copy submit paths where source
    /// frames live in shared / imported textures rather than the
    /// renderer's internal plane textures. Call after the stitch
    /// submit has landed; this method encodes a separate command
    /// buffer for the pack + staging copy, submits it, and polls
    /// the triple-buffer ring for a ready atlas to feed to the
    /// attached recorder.
    ///
    /// The storytelling flow (per the project principle — no silent
    /// decisions): the caller chose this path because the source is
    /// GPU-resident. The packer's configured `SourceFormat` was
    /// logged at `enable_gpu_stacked_replay` time. From here on,
    /// every call is just bytes moving through the pipeline, so no
    /// per-frame logging.
    ///
    /// No-op when the packer isn't enabled.
    ///
    /// Hard-coded to the two-camera stereo layout today; extend
    /// when `CameraInput::camera_count() > 2` lands.
    pub fn pack_gpu_stacked_replay_from_views(
        &mut self,
        left: StackedPackSource<'_>,
        right: StackedPackSource<'_>,
    ) {
        crate::profile_scope!("replay_pack_from_views");
        let Some(ref mut packer) = self.stacked_packer else {
            return;
        };
        let gpu = self.pipeline.gpu();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stitch_core_gpu_stacked_pack_ext"),
            });
        let capacity = packer.layout().capacity();
        if capacity >= 1 {
            packer.pack_tile_from_views(gpu, &mut encoder, 0, left);
        }
        if capacity >= 2 {
            packer.pack_tile_from_views(gpu, &mut encoder, 1, right);
        }
        {
            crate::profile_scope!("replay_copy_to_staging");
            packer.copy_to_staging(&mut encoder);
            gpu.queue.submit(Some(encoder.finish()));
        }

        {
            crate::profile_scope!("replay_poll_and_record");
            if let Some(atlas) = packer.poll_ready(gpu)
                && let Some(ref mut recorder) = self.stacked_gpu_recorder
            {
                recorder.record_atlas(&atlas);
            }
        }
    }

    /// Runs the GPU pack from the pipeline's internal plane
    /// textures. Used by CPU-upload submit paths where
    /// `queue.write_texture` has just populated the renderer's
    /// own textures — fires from `submit_frame_yuv*` and from the
    /// session's `process_frame` non-zero-copy branch. Zero-copy
    /// paths take [`Self::pack_gpu_stacked_replay_from_views`]
    /// instead because their source data lives in shared
    /// textures that bypass the renderer's internal planes.
    ///
    /// Delegates through the same pack + poll + record path so
    /// every entry point shares behavior.
    ///
    /// No-op when the packer isn't enabled.
    pub(crate) fn drive_gpu_stacked_pack(&mut self) {
        crate::profile_scope!("replay_drive_pack");
        if self.stacked_packer.is_none() {
            return;
        }
        // Pipeline's plane-view accessors return
        // (y_view, u_or_uv_view, v_or_dummy_view). Build the
        // StackedPackSource variant matching the packer's
        // configured source format — the packer will route to the
        // right shader kernel internally.
        let (ly, lu, lv) = self.pipeline.left_plane_views();
        let (ry, ru, rv) = self.pipeline.right_plane_views();
        // Keep bindings alive across the pack call via locals.
        let (left, right) = match self.pipeline.input_format() {
            InputFormat::Yuv420p => (
                StackedPackSource::Yuv420p {
                    y: &ly,
                    u: &lu,
                    v: &lv,
                },
                StackedPackSource::Yuv420p {
                    y: &ry,
                    u: &ru,
                    v: &rv,
                },
            ),
            InputFormat::Nv12 => (
                StackedPackSource::Nv12 { y: &ly, uv: &lu },
                StackedPackSource::Nv12 { y: &ry, uv: &ru },
            ),
            InputFormat::Bgra => {
                // Shouldn't happen: enable_gpu_stacked_replay
                // rejects BGRA up front. Defensive no-op so the
                // live render loop can't panic on an invariant
                // violation.
                log::error!(
                    "drive_gpu_stacked_pack: packer enabled but pipeline input_format is \
                     BGRA; skipping pack (this is a logic bug in StitchCore)"
                );
                return;
            }
        };
        self.pack_gpu_stacked_replay_from_views(left, right);
    }

    /// Set how often detection runs.
    ///
    /// `1` (default) = every frame, `3` = every third frame, etc.
    /// Values `< 1` are clamped to `1`. Detection is expensive
    /// (2-20 ms depending on the model and backend); skipping frames
    /// lets the render loop run faster while the director interpolates
    /// using the latest detection output.
    pub fn set_detection_interval(&mut self, interval: u64) {
        self.detection_interval = interval.max(1);
    }

    /// Current detection interval.
    pub fn detection_interval(&self) -> u64 {
        self.detection_interval
    }

    /// The resolved viewport pose for the next render, already clamped
    /// through coverage + FOV limits. Exposed so interactive consumers
    /// (OBS pan/zoom, GUI drag) can preview where the core *would*
    /// render if they submit right now.
    pub fn current_pose(&mut self) -> ViewportPosition {
        // A peek does no detection work of its own, so the director
        // sees `fresh_detection = false`. The next real submit will
        // fire the schedule-driven detection path and pass the actual
        // run result.
        self.resolve_current_pose(false)
    }

    /// Clamp a prospective `(yaw, pitch, fov)` triple through the
    /// coverage boundary. No-op if no coverage is available (e.g. the
    /// calibration produced a degenerate boundary). `fov_degrees: None`
    /// uses the pipeline's current FOV.
    ///
    /// Input is treated as world-space (matches the director-output
    /// contract). Output is user-space via a simple
    /// `user_pitch = world_pitch - rig_tilt` transform. The yaw-weighted
    /// horizon correction is a render-site concern (see Model 3 in
    /// `PoseControl::render_pose`).
    pub fn safe_clamp(&self, pose: ViewportPosition) -> ViewportPosition {
        let Some(coverage) = &self.coverage else {
            return pose;
        };
        let fov = pose
            .fov_degrees
            .unwrap_or_else(|| self.pipeline.fov())
            .min(coverage.max_fov_degrees());
        let aspect = self.pipeline.viewport().aspect_ratio();
        let clamped = coverage.safe_clamp(pose.yaw, pose.pitch, fov, aspect, 0.0);
        let rig_tilt = self.pipeline.viewport().rig_tilt;
        ViewportPosition {
            yaw: clamped.yaw,
            pitch: clamped.pitch - rig_tilt,
            fov_degrees: Some(fov),
        }
    }

    // -----------------------------------------------------------------
    // Coverage / calibration / projection introspection
    // -----------------------------------------------------------------

    /// The precomputed coverage boundary for "no-black" viewport
    /// constraining. Some calibrations produce a degenerate boundary
    /// in which case this is `None` (very rare in practice).
    pub fn coverage(&self) -> Option<&CoverageBoundary> {
        self.coverage.as_ref()
    }

    /// Whether the render loop's pose resolution clamps through the
    /// coverage boundary (FRICTION A13). `true` by default. Consumers
    /// expose this as a UI toggle ("Constrained look") so users can
    /// choose between "never show black edges" (on) and "unrestricted
    /// panning" (off).
    pub fn constrained_look(&self) -> bool {
        self.constrained_look
    }

    /// Enable or disable constrained-look clamping.
    ///
    /// When `true`, [`Self::submit_frame_yuv`] / `..._bgra` /
    /// `submit_frame_*_at_pose` pass the director's (or caller's)
    /// pose through [`Self::safe_clamp`] before rendering.
    /// When `false`, the raw pose is used verbatim; the FOV max is
    /// still respected (pipeline-set) but coverage-based yaw/pitch
    /// clamping is skipped.
    ///
    /// The public [`Self::safe_clamp`] method is unaffected — it
    /// always clamps, regardless of this flag.
    pub fn set_constrained_look(&mut self, enabled: bool) {
        self.constrained_look = enabled;
    }

    /// Toggle the constrained-look flag. Returns the new value.
    /// Consumers handling `HotkeyIntent::ToggleConstrained`
    /// wire it to this method.
    pub fn toggle_constrained_look(&mut self) -> bool {
        self.constrained_look = !self.constrained_look;
        self.constrained_look
    }

    /// Full angular extent of the stitched panorama, derived from the
    /// coverage boundary. Higher-level shortcut for analytics consumers.
    pub fn panorama_extent(&self) -> Option<PanoramaExtent> {
        self.coverage.as_ref().map(|c| {
            let (yaw_min, yaw_max) = c.yaw_range();
            let (pitch_min, pitch_max) = c.pitch_range();
            PanoramaExtent {
                yaw_min,
                yaw_max,
                pitch_min,
                pitch_max,
            }
        })
    }

    /// Short name of the active projection (for logs + UI labels).
    pub fn projection_name(&self) -> &'static str {
        self.projection.name()
    }

    /// Camera count of the active input configuration. For today's
    /// stereo L-shape this is `2`; future mono / N-input builds expose
    /// different values.
    pub fn camera_count(&self) -> u8 {
        self.camera_input.camera_count()
    }

    /// Hot-swap the calibration. Takes effect on the next submit.
    ///
    /// Re-derives the coverage boundary from the new calibration so
    /// subsequent `safe_clamp` calls respect the new no-black region.
    pub fn update_calibration(&mut self, calibration: MatchCalibration) {
        self.pipeline.update_calibration(calibration);
        self.coverage = Some(CoverageBoundary::from_calibration(
            self.pipeline.calibration(),
            &self.pipeline.scene,
        ));
    }

    // -----------------------------------------------------------------
    // Pipeline stages (slot reserved; no registered stages yet)
    // -----------------------------------------------------------------

    /// Append a pipeline stage to the mid-pipeline chain.
    ///
    /// The chain has no registered stages today; this method is the
    /// registration point for future stages (color correction, exposure
    /// normalization, remote-compute shims) without another breaking
    /// change. Stages are executed in push order.
    pub fn push_pipeline_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
    }

    /// Number of registered pipeline stages.
    pub fn pipeline_stage_count(&self) -> usize {
        self.stages.len()
    }

    // -----------------------------------------------------------------
    // Replay buffer
    // -----------------------------------------------------------------

    /// Enable (or reconfigure) the replay ring buffer.
    ///
    /// Passing `None` disables replay and drops any buffered frames,
    /// freeing the allocation. Passing `Some(duration)` creates (or
    /// resizes) the ring to retain at most `duration` of the most
    /// recent rendered frames. The ring only grows as frames arrive;
    /// no pre-allocation.
    pub fn enable_replay_buffer(&mut self, duration: Option<Duration>) {
        self.replay = duration.map(ReplayBuffer::new);
    }

    /// Borrow the replay buffer, if enabled.
    pub fn replay_buffer(&self) -> Option<&ReplayBuffer> {
        self.replay.as_ref()
    }

    /// Mutable borrow of the replay buffer. Consumers wire this to
    /// "Clear replay" / "Save replay + reset" UI buttons which call
    /// [`ReplayBuffer::clear`] or [`ReplayBuffer::take`] respectively.
    pub fn replay_buffer_mut(&mut self) -> Option<&mut ReplayBuffer> {
        self.replay.as_mut()
    }

    // -----------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------

    /// Output dimensions in pixels. Identical to
    /// `config.viewport.{width,height}` at construction.
    pub fn output_dims(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Number of frames submitted so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Shared access to the underlying pipeline.
    pub fn pipeline(&self) -> &StitchPipeline {
        &self.pipeline
    }

    /// Mutable access to the pipeline for advanced callers that need
    /// to tweak viewport / FOV / zero-copy bind groups directly.
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self.pipeline
    }

    /// The GPU context owning every resource.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    // -----------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------

    fn anchor_session_start(&mut self) {
        if self.session_start.is_none() {
            self.session_start = Some(Instant::now());
        }
    }

    fn resolve_current_pose(&mut self, fresh_detection: bool) -> ViewportPosition {
        // Pull raw director output (or default) and clamp through
        // coverage. Then write the resolved FOV back onto the pipeline
        // so the upcoming render uses it.
        //
        // `fresh_detection` is the ACTUAL run decision for this frame,
        // not the schedule-would-fire predicate. The BGRA submit path
        // deliberately skips detection (no BGRA-aware backend exists
        // today) so it must pass `false` even when the interval would
        // have fired — otherwise directors over-count hysteresis on
        // stale cached detections.
        let timestamp_ms = self
            .session_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        // Pose resolution: run all registered trackers in order, assemble
        // a [`WorldState`], and let the panner decide via the shared
        // `panner::dispatch` helper. When no panner is attached the pose
        // stays at the pipeline default.
        let _ = fresh_detection; // reserved for future freshness-aware panners
        let raw = crate::panner::dispatch(
            self.panner.as_mut(),
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            &mut self.previous_panner_pose,
            // StitchCore does not own an event sink. StitchSession
            // does the tracing when it is the active entry point.
            None,
            crate::panner::DispatchContext {
                detections: &self.last_detections,
                calibration: &self.pipeline.calibration,
                frame_index: self.frame_count,
                timestamp_ms,
                caller: "StitchCore",
            },
        )
        .map(|r| r.pose)
        .unwrap_or_default();
        let clamped = if self.constrained_look {
            self.safe_clamp(raw)
        } else {
            raw
        };
        if let Some(fov) = clamped.fov_degrees {
            self.pipeline.set_fov(fov);
        }
        clamped
    }

    fn should_run_detection(&self) -> bool {
        // Before any submit, frame_count == 0 and interval defaults to
        // 1, which covers the "run on the very first frame" case.
        self.detection_interval > 0 && self.frame_count.is_multiple_of(self.detection_interval)
    }

    /// Run the attached detector against a stereo YUV420P frame pair.
    ///
    /// Wraps each plane as [`RawFrame`] + [`DetectorFrame::Cpu`] and
    /// dispatches through the unified trait, once per camera. Errors
    /// are warned-and-dropped: a flaky inference call must not crash
    /// the render loop.
    fn run_yuv_detection(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        source_width: u32,
        source_height: u32,
    ) -> Vec<Detection> {
        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (camera, planes) in [(CameraId::Left, left), (CameraId::Right, right)] {
            let raw = RawFrame {
                y: planes.y,
                chroma: ChromaFormat::Yuv420p {
                    u: planes.u,
                    v: planes.v,
                },
                width: source_width,
                height: source_height,
            };
            match detector.detect(camera, &DetectorFrame::Cpu(raw)) {
                Ok(v) => out.extend(v),
                Err(e) => log::warn!("StitchCore detector '{}' {camera:?}: {e}", detector.name()),
            }
        }
        out
    }

    /// Map raw camera-space detections to panorama-space
    /// [`MappedDetection`]s the director can consume.
    fn map_detections_to_panorama(&self, detections: Vec<Detection>) -> Vec<MappedDetection> {
        let calibration = self.pipeline.calibration();
        let scene = &self.pipeline.scene;
        detections
            .into_iter()
            .map(|d| {
                let position = projection::camera_to_panorama(
                    d.camera,
                    d.center_x,
                    d.center_y,
                    calibration,
                    scene,
                );
                MappedDetection {
                    camera: d.camera,
                    class_id: d.class_id,
                    confidence: d.confidence,
                    camera_center: (d.center_x, d.center_y),
                    camera_size: (d.width, d.height),
                    position,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert `ReplayBuffer` trims old frames as the newest ages past
    /// `max_duration`. This is the core guarantee OBS A16 relies on:
    /// the buffer never grows unboundedly during a long session.
    #[test]
    fn replay_buffer_trims_old_frames() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(2));
        for i in 0..5 {
            buf.push(ReplayFrame {
                rgba: vec![i as u8; 4],
                captured_at: Duration::from_millis(i as u64 * 1000),
                pose: ViewportPosition::default(),
            });
        }
        // Newest is at 4s; anything older than 2s should be evicted.
        // Frames at 0s and 1s are older than (4s - 2s = 2s), so evicted.
        // Frames at 2s, 3s, 4s remain.
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.iter().next().unwrap().rgba[0], 2);
        assert_eq!(buf.latest().unwrap().rgba[0], 4);
    }

    /// Replay trimming respects `max_duration` exactly: the boundary
    /// is inclusive on the retain side.
    #[test]
    fn replay_buffer_boundary_inclusive() {
        let mut buf = ReplayBuffer::new(Duration::from_millis(100));
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(0),
            pose: ViewportPosition::default(),
        });
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(100),
            pose: ViewportPosition::default(),
        });
        // Newest - max_duration = 0, so frame at 0ms is exactly on the
        // boundary and retained.
        assert_eq!(buf.len(), 2);
    }

    /// An empty replay buffer answers `latest()` = None and is_empty.
    #[test]
    fn replay_buffer_empty_semantics() {
        let buf = ReplayBuffer::new(Duration::from_secs(1));
        assert!(buf.is_empty());
        assert!(buf.latest().is_none());
        assert!(buf.oldest().is_none());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.buffered_duration(), Duration::ZERO);
        assert_eq!(buf.max_duration(), Duration::from_secs(1));
    }

    /// A16 "Clear replay" / "Save replay" UI wiring: `clear`,
    /// `snapshot`, `take`, `buffered_duration`, `oldest`.
    #[test]
    fn replay_buffer_snapshot_and_take_preserve_ordering() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(10));
        for i in 0..3u8 {
            buf.push(ReplayFrame {
                rgba: vec![i; 4],
                captured_at: Duration::from_millis(i as u64 * 100),
                pose: ViewportPosition::default(),
            });
        }
        // snapshot returns oldest-to-newest, no consumption.
        let snap = buf.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].rgba[0], 0);
        assert_eq!(snap[2].rgba[0], 2);
        assert_eq!(buf.len(), 3, "snapshot does not drain");

        // take drains and returns owned vec in same order.
        let drained = buf.take();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].rgba[0], 0);
        assert!(buf.is_empty(), "take empties the buffer");
    }

    #[test]
    fn replay_buffer_clear_drops_frames_keeps_max_duration() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(5));
        buf.push(ReplayFrame {
            rgba: vec![0u8; 4],
            captured_at: Duration::ZERO,
            pose: ViewportPosition::default(),
        });
        assert!(!buf.is_empty());
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(
            buf.max_duration(),
            Duration::from_secs(5),
            "clear preserves the configured window"
        );
    }

    #[test]
    fn replay_buffer_duration_tracks_oldest_newest_spread() {
        let mut buf = ReplayBuffer::new(Duration::from_secs(10));
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(100),
            pose: ViewportPosition::default(),
        });
        buf.push(ReplayFrame {
            rgba: vec![],
            captured_at: Duration::from_millis(850),
            pose: ViewportPosition::default(),
        });
        assert_eq!(buf.buffered_duration(), Duration::from_millis(750));
        assert_eq!(
            buf.oldest().unwrap().captured_at,
            Duration::from_millis(100)
        );
    }

    /// `StitchCoreError` is `std::error::Error` (so downstream
    /// consumers can `?` it through `Box<dyn Error>` channels) and
    /// `Send + Sync` (so it can cross thread boundaries in a future
    /// worker-thread detection pipeline).
    #[test]
    fn stitch_core_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StitchCoreError>();
        fn assert_error<T: std::error::Error + 'static>() {}
        assert_error::<StitchCoreError>();
    }

    /// `RenderOutcome` is `Send` — needed so consumers that post
    /// rendered frames onto worker channels (or mpsc splits) can
    /// forward the enum without boxing it.
    #[test]
    fn render_outcome_warmup_constructs() {
        let outcome: RenderOutcome<'_> = RenderOutcome::Warmup;
        match outcome {
            RenderOutcome::Warmup => {}
            RenderOutcome::Rgba(_) => unreachable!("built a Warmup"),
        }
    }
}
