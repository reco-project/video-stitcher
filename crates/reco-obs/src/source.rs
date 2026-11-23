//! OBS source implementation for the Reco stitching pipeline.
//!
//! This module implements the `obs_source_info` callbacks that OBS invokes
//! to create, configure, tick, and render the stitcher source.
//!
//! ## Rendering architecture
//!
//! OBS's graphics subsystem (OpenGL/D3D11) and reco-core's wgpu pipeline
//! live in separate GPU contexts. We cannot share textures directly.
//! Instead:
//!
//! 1. `video_tick`: reco-core renders the stitched panorama to its wgpu
//!    render target, then copies it to a CPU staging buffer (RGBA readback).
//! 2. `video_render`: uploads the CPU RGBA buffer to an OBS `gs_texture_t`
//!    and draws it with the default effect.
//!
//! This incurs one GPU-to-CPU-to-GPU copy per frame. A future optimization
//! could use platform-specific interop (DMA-BUF on Linux, shared handles on
//! Windows) to avoid the CPU roundtrip.

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_void};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use reco_core::calibration::Calibration;
use reco_core::core::StitchCore;
use reco_core::core::types::RenderOutcome;
use reco_core::gpu::GpuContext;
use reco_core::render::pipeline::{FramePlaneView, RgbaPlanes, StridedYuv420pPlanes};
use reco_core::render::renderer::InputFormat;
use reco_core::render::viewport::ViewportSize;
use reco_core::stitch::{Executor, GpuExecutor, GpuExecutorConfig};

use crate::ffi;
use crate::interaction::Interaction;
use crate::resources::{SourceInput, Texture, accepts_source};

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// C string for the source ID (must live for the entire plugin lifetime).
const SOURCE_ID: &CStr = c"reco_stitcher";

/// C string for the human-readable source name.
const SOURCE_NAME: &CStr = c"Reco Panorama Stitcher";

/// Property key names (C strings).
const PROP_CONFIG_PATH: &CStr = c"config_path";
const PROP_OUTPUT_WIDTH: &CStr = c"output_width";
const PROP_OUTPUT_HEIGHT: &CStr = c"output_height";
const PROP_YAW: &CStr = c"yaw";
const PROP_PITCH: &CStr = c"pitch";
const PROP_CONSTRAINED: &CStr = c"constrained_look";
/// Name of the OBS source that feeds the left camera.
const PROP_LEFT_SOURCE: &CStr = c"left_source";
/// Name of the OBS source that feeds the right camera.
const PROP_RIGHT_SOURCE: &CStr = c"right_source";
/// Upstream pixel format (string: "i420" | "bgra"). Must match what the
/// picked OBS sources actually deliver - OBS can't negotiate, so this is
/// a manual hint.
const PROP_INPUT_FORMAT: &CStr = c"input_format";
const INPUT_FORMAT_I420: &CStr = c"i420";
const INPUT_FORMAT_BGRA: &CStr = c"bgra";
/// Opt-in replay recording. When enabled alongside a non-empty
/// [`PROP_REPLAY_PATH`], the plugin attaches a stacked-video
/// recorder to the StitchCore so every submitted YUV frame pair is
/// also written to disk as a vertically stacked Matroska file.
/// Matroska clusters flush as they are written, so a reader can
/// open the file while the recorder is still running.
#[cfg(feature = "replay")]
const PROP_REPLAY_ENABLED: &CStr = c"replay_enabled";
#[cfg(feature = "replay")]
const PROP_REPLAY_PATH: &CStr = c"replay_path";
/// Replay trigger mode. String-valued because OBS combos pass
/// strings; numeric would be less readable in the settings JSON.
/// - `"follow_obs"` (default): record only while OBS itself is
///   recording or streaming. Closest to the user's mental model
///   ("the OBS Record button controls everything the plugin
///   writes").
/// - `"always"`: record whenever the checkbox is ticked,
///   independently of OBS state, for users who genuinely want
///   replay without running an OBS recording.
#[cfg(feature = "replay")]
const PROP_REPLAY_MODE: &CStr = c"replay_mode";
#[cfg(feature = "replay")]
const REPLAY_MODE_FOLLOW_OBS: &CStr = c"follow_obs";
#[cfg(feature = "replay")]
const REPLAY_MODE_ALWAYS: &CStr = c"always";

// ---------------------------------------------------------------------------
// Source state
// ---------------------------------------------------------------------------

/// Callbacks only borrow this shared handle. The UI never borrows the engine or presentation state.
struct SourceHandle {
    engine: Mutex<RecoSource>,
    interaction: Mutex<Interaction>,
    presentation: Mutex<Presentation>,
    width: AtomicU32,
    height: AtomicU32,
}

impl Default for SourceHandle {
    fn default() -> Self {
        Self {
            engine: Mutex::new(RecoSource::new()),
            interaction: Mutex::new(Interaction::default()),
            presentation: Mutex::new(Presentation::default()),
            width: AtomicU32::new(1920),
            height: AtomicU32::new(1080),
        }
    }
}

impl SourceHandle {
    fn reset_presentation(&self, engine: &RecoSource) {
        self.width.store(engine.output_width, Ordering::Relaxed);
        self.height.store(engine.output_height, Ordering::Relaxed);
        if let Ok(mut display) = self.presentation.lock() {
            display.width = engine.output_width;
            display.height = engine.output_height;
            display.rgba.clear();
            display.dirty = false;
        }
    }
}

/// CPU output and its OBS texture. Only rendering/destruction accesses the texture, under OBS graphics ownership.
#[derive(Default)]
struct Presentation {
    rgba: Vec<u8>,
    dirty: bool,
    width: u32,
    height: u32,
    texture_width: u32,
    texture_height: u32,
    texture: Texture,
}

/// Stitching state, accessed only through the shared handle's engine mutex.
struct RecoSource {
    /// Push-first stitch core (owns pipeline + RGBA readback).
    /// None until calibration is loaded and GPU is initialized. The
    /// source drives OBS video_tick callbacks into
    /// `StitchCore::submit_frame_*_at_pose` and reads the resulting
    /// RGBA out of the triple-buffered readback ring.
    core: Option<StitchCore>,

    /// Upstream OBS source that feeds the left camera. Owned reference
    /// (obs_get_source_by_name increments refcount, we release in destroy
    /// or when the user picks a different source).
    left_source: SourceInput,
    /// Upstream OBS source that feeds the right camera. See [`Self::left_source`].
    right_source: SourceInput,
    /// Reusable tight-pack buffers for `StridedYuv420pPlanes::copy_into`. One
    /// per side so the render loop never reallocates per frame.
    left_repack: Vec<u8>,
    right_repack: Vec<u8>,
    /// Which reco-core input format the session was built with. Determines
    /// which submit path (YUV vs BGRA) we dispatch to and which OBS video
    /// formats we accept as valid input.
    input_format: InputFormat,
    /// One-shot warning flag: emitted once per unsupported video format
    /// so we don't flood the log on every frame.
    warned_unsupported_format: bool,
    /// Rolling tick counter + per-category "missed frame" counters for
    /// diagnostic heartbeat logging. Flushed every ~60 ticks so consumers
    /// can tell "plugin running but never rendered" from silent hang.
    diag_tick: u64,
    diag_missed_left: u64,
    diag_missed_right: u64,
    diag_submitted: u64,

    /// Current calibration loaded from the config file.
    calibration: Option<Calibration>,

    /// Path to the calibration JSON file.
    config_path: String,

    /// Output viewport dimensions.
    output_width: u32,
    output_height: u32,

    /// Last observed decoded dimensions, in left/right camera order.
    input_sizes: Option<[(u32, u32); 2]>,

    constrained_look: bool,

    /// User intent, independent of OBS's record/stream trigger.
    #[cfg(feature = "replay")]
    replay_enabled: bool,
    /// Output path for the stacked-video replay file. Empty string
    /// means "no output path set"; the recorder only attaches when
    /// both `replay_enabled` is true and this is non-empty.
    #[cfg(feature = "replay")]
    replay_path: String,
    /// Follow OBS's record/stream state instead of recording independently.
    #[cfg(feature = "replay")]
    replay_follow_obs: bool,
    /// The active file, so a path edit finalizes the previous recorder.
    #[cfg(feature = "replay")]
    replay_recorder_path: Option<String>,
    /// Latch each start attempt to avoid repeated file/encoder opens on the video thread.
    #[cfg(feature = "replay")]
    replay_attempted_path: Option<String>,
}

impl RecoSource {
    fn new() -> Self {
        Self {
            core: None,
            left_source: SourceInput::default(),
            right_source: SourceInput::default(),
            left_repack: Vec::new(),
            right_repack: Vec::new(),
            input_format: InputFormat::Yuv420p,
            warned_unsupported_format: false,
            diag_tick: 0,
            diag_missed_left: 0,
            diag_missed_right: 0,
            diag_submitted: 0,
            calibration: None,
            config_path: String::new(),
            output_width: 1920,
            output_height: 1080,
            input_sizes: None,
            constrained_look: true,
            #[cfg(feature = "replay")]
            replay_enabled: false,
            #[cfg(feature = "replay")]
            replay_path: String::new(),
            #[cfg(feature = "replay")]
            replay_follow_obs: true,
            #[cfg(feature = "replay")]
            replay_recorder_path: None,
            #[cfg(feature = "replay")]
            replay_attempted_path: None,
        }
    }

    /// Try to initialize (or reinitialize) the pipeline from current settings.
    ///
    /// Called after calibration is loaded or dimensions change.
    fn try_init_pipeline(&mut self) {
        // A failed rebuild must not leave a renderer using the old dimensions or calibration.
        if let Some(core) = &mut self.core {
            core.clear_stacked_recorder();
        }
        self.core = None;
        #[cfg(feature = "replay")]
        {
            if let Some(path) = self.replay_recorder_path.take() {
                log::warn!(
                    "reco-obs: replay stopped for input rebuild; preserved '{}'. Select a new output path to record again.",
                    path
                );
            }
        }
        let Some(input_sizes) = self.input_sizes else {
            return;
        };
        let calibration = match &self.calibration {
            Some(cal) => cal.clone(),
            None => {
                log::debug!("Skipping pipeline init: no calibration loaded");
                return;
            }
        };

        let gpu = match GpuContext::new_blocking() {
            Ok(g) => g,
            Err(e) => {
                log::error!("reco-obs: failed to initialize GPU: {e}");
                return;
            }
        };
        log::info!("reco-obs: GPU initialized: {}", gpu.gpu_name());

        // Forward calibration's rig tilt + roll into the viewport.
        let viewport_size = ViewportSize {
            width: self.output_width,
            height: self.output_height,
        };

        let executor = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                calibration,
                viewport_size,
                source_sizes: input_sizes.to_vec(),
                input_format: self.input_format,
                output_format: reco_core::wgpu::TextureFormat::Rgba8Unorm,
                full_range: false,
            },
        );
        let executor = match executor {
            Ok(executor) => executor,
            Err(e) => {
                log::error!("reco-obs: failed to create executor: {e}");
                return;
            }
        };
        match StitchCore::new(Executor::Gpu(Box::new(executor))) {
            Ok(session) => {
                log::info!(
                    "reco-obs: session initialized ({}x{} output, inputs={input_sizes:?}, format={:?})",
                    self.output_width,
                    self.output_height,
                    self.input_format,
                );
                self.core = Some(session);
                self.update_replay_recorder(false);
            }
            Err(e) => {
                log::error!("reco-obs: failed to create session: {e}");
            }
        }
    }

    /// Follow recording intent, but start only after a compatible input pair arrives.
    #[cfg(feature = "replay")]
    fn update_replay_recorder(&mut self, can_start: bool) {
        let trigger_on = !self.replay_follow_obs
            || crate::OBS_RECORDING_OR_STREAMING.load(std::sync::atomic::Ordering::Relaxed);
        let want_recording = self.replay_enabled
            && !self.replay_path.is_empty()
            && trigger_on
            && self.core.is_some();
        if self
            .replay_recorder_path
            .as_ref()
            .is_some_and(|path| !want_recording || path != &self.replay_path)
        {
            if let Some(core) = &mut self.core {
                core.clear_stacked_recorder();
            }
            self.replay_recorder_path = None;
        }
        if !want_recording {
            self.replay_attempted_path = None;
            return;
        }
        if !can_start
            || self.replay_recorder_path.is_some()
            || self.replay_attempted_path.as_ref() == Some(&self.replay_path)
        {
            return;
        }
        let mut video = unsafe { std::mem::zeroed::<ffi::obs_video_info>() };
        if !unsafe { ffi::obs_get_video_info(&mut video) } {
            return;
        }
        self.replay_attempted_path = Some(self.replay_path.clone());
        if self.input_format != InputFormat::Yuv420p {
            log::warn!("reco-obs: stacked replay requires I420 camera inputs");
            return;
        }
        let Some([left, right]) = self.input_sizes else {
            return;
        };
        if left != right {
            log::warn!(
                "reco-obs: stacked replay requires equal camera dimensions; stitching continues"
            );
            return;
        }
        let (input_width, input_height) = left;
        let (Ok(fps_num), Ok(fps_den)) =
            (i32::try_from(video.fps_num), i32::try_from(video.fps_den))
        else {
            log::warn!("reco-obs: unsupported OBS frame rate for replay recording");
            return;
        };
        let path = std::path::Path::new(&self.replay_path);
        // Reserve a fresh path before FFmpeg opens it. Automatic restarts must not truncate an earlier recording.
        let _reservation = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(file) => file,
            Err(error) => {
                log::warn!(
                    "reco-obs: cannot start replay at '{}' ({error}); select a new writable file path",
                    path.display()
                );
                return;
            }
        };
        let config = reco_io::stacked_video::encoder::StackedEncoderConfig {
            fps: Some((fps_num, fps_den)),
            ..Default::default()
        };
        let result = reco_io::stacked_video::replay::session_recorder(
            path,
            config,
            input_width,
            input_height,
        );
        match result {
            Ok(recorder) => {
                let core = self.core.as_mut().expect("recording requires an engine");
                if let Err(error) = core.set_stacked_recorder(recorder) {
                    log::warn!("reco-obs: replay recorder rejected: {error}");
                    return;
                }
                self.replay_recorder_path = Some(self.replay_path.clone());
                log::info!(
                    "reco-obs: replay recording started at '{}' ({}x{} per tile, {fps_num}/{fps_den} fps)",
                    path.display(),
                    input_width,
                    input_height
                );
            }
            Err(error) => log::warn!(
                "reco-obs: replay failed to open '{}' ({error}); stitching continues",
                path.display()
            ),
        }
    }

    /// No-op stub when the `replay` feature is disabled. Keeps the
    /// call sites unconditional without scattering `#[cfg]` around.
    #[cfg(not(feature = "replay"))]
    fn update_replay_recorder(&mut self, _can_start: bool) {}

    /// Resize the existing output without replacing input resources or the replay recorder.
    fn resize_output(&mut self) {
        let Some(core) = self.core.as_mut() else {
            return;
        };
        if core.resize(self.output_width, self.output_height).is_none() {
            log::warn!("reco-obs: output resize rejected; retaining the accepted dimensions");
            (self.output_width, self.output_height) = core.output_dims();
        }
    }

    /// Load calibration from the config file path.
    fn load_calibration(&mut self) {
        if self.config_path.is_empty() {
            self.calibration = None;
            self.core = None;
            return;
        }

        let path = Path::new(&self.config_path);
        match Calibration::from_file(path) {
            Ok(cal) => {
                log::info!("reco-obs: loaded calibration from {}", self.config_path);
                self.calibration = Some(cal);
            }
            Err(e) => {
                log::error!(
                    "reco-obs: failed to load calibration from {}: {e}",
                    self.config_path
                );
                self.calibration = None;
                self.core = None;
            }
        }
    }

    /// Pull the latest async frame pair from the upstream OBS sources,
    /// stitch, and stash the RGBA result for `video_render`.
    ///
    /// Returns silently (no render) when either upstream source is unset,
    /// has no current frame, or delivers a format other than the selected input format. The
    /// `warned_unsupported_format` flag ensures unsupported-format warnings
    /// fire at most once per source instance.
    fn render_and_readback(&mut self, shared: &SourceHandle) {
        self.diag_tick = self.diag_tick.wrapping_add(1);

        // Lazy re-resolve: on OBS startup, source_create for the Reco
        // source often fires before scene sources have finished loading,
        // so our initial apply_settings resolves Media / Media 2 to null.
        // Retry every 30 ticks (~0.5s at 60fps) until we get non-null
        // refs. Cheap - obs_get_source_by_name is a hashmap lookup.
        if self.diag_tick.is_multiple_of(30) {
            self.left_source.retry();
            self.right_source.retry();
            // Also re-evaluate replay attachment state here (every
            // 0.5s). The global `OBS_RECORDING_OR_STREAMING` atomic
            // is flipped from the frontend event callback, which
            // runs on a different thread; per-source state lives
            // here, so a periodic poll is how we bridge the two.
            // Cheap: one atomic load + one bool compare in the
            // common case where nothing changed.
            #[cfg(feature = "replay")]
            self.update_replay_recorder(false);
        }
        // Every ~60 ticks (~1-2 seconds) flush a diagnostic heartbeat so
        // users can tell "plugin running but no render" from "plugin hung".
        if self.diag_tick.is_multiple_of(60) {
            log::info!(
                "reco-obs: diag tick={} submitted={} \
                 missed_left={} missed_right={} session={} left_src={} right_src={}",
                self.diag_tick,
                self.diag_submitted,
                self.diag_missed_left,
                self.diag_missed_right,
                if self.core.is_some() { "ok" } else { "none" },
                if self.left_source.is_null() {
                    "null"
                } else {
                    "ok"
                },
                if self.right_source.is_null() {
                    "null"
                } else {
                    "ok"
                },
            );
        }

        if self.calibration.is_none() {
            return;
        }
        if self.left_source.is_null() || self.right_source.is_null() {
            return;
        }

        // Acquire frames (obs_source_get_frame increments the internal
        // ref; must pair with release_frame).
        let left_frame = unsafe { AsyncFrame::acquire(self.left_source.pointer) };
        let right_frame = unsafe { AsyncFrame::acquire(self.right_source.pointer) };
        if left_frame.is_none() {
            self.diag_missed_left += 1;
        }
        if right_frame.is_none() {
            self.diag_missed_right += 1;
        }

        if let (Some(left_frame), Some(right_frame)) = (&left_frame, &right_frame) {
            let l = left_frame.frame();
            let r = right_frame.frame();

            // A format mismatch cannot be submitted to the existing input textures.
            let format_ok = match self.input_format {
                InputFormat::Yuv420p => {
                    l.format == ffi::video_format::VIDEO_FORMAT_I420
                        && r.format == ffi::video_format::VIDEO_FORMAT_I420
                }
                InputFormat::Rgba => {
                    matches!(
                        l.format,
                        ffi::video_format::VIDEO_FORMAT_BGRA
                            | ffi::video_format::VIDEO_FORMAT_BGRX
                            | ffi::video_format::VIDEO_FORMAT_RGBA
                    ) && matches!(
                        r.format,
                        ffi::video_format::VIDEO_FORMAT_BGRA
                            | ffi::video_format::VIDEO_FORMAT_BGRX
                            | ffi::video_format::VIDEO_FORMAT_RGBA
                    )
                }
                InputFormat::Nv12 => false,
            };
            if !format_ok {
                if !self.warned_unsupported_format {
                    log::warn!(
                        "reco-obs: upstream format doesn't match session ({:?}, configured \
                         for {:?}; got left={:?}, right={:?}). Change the plugin's 'Input \
                         format' property to match the camera frames.",
                        self.input_format,
                        self.input_format,
                        l.format,
                        r.format,
                    );
                    self.warned_unsupported_format = true;
                }
            } else if self.input_sizes != Some([(l.width, l.height), (r.width, r.height)]) {
                let sizes = [(l.width, l.height), (r.width, r.height)];
                log::info!(
                    "reco-obs: camera dimensions changed from {:?} to {sizes:?}; rebuilding inputs",
                    self.input_sizes,
                );
                self.input_sizes = Some(sizes);
                self.try_init_pipeline();
                self.warned_unsupported_format = false;
            } else {
                if self.core.is_none() {
                    return;
                }
                self.update_replay_recorder(true);
                // Direct interaction is fitted once more before submission, including after geometry changes.
                let Ok(mut interaction) = shared.interaction.lock() else {
                    return;
                };
                let mut pose = interaction.take_request();
                drop(interaction);
                let engine = self.core.as_mut().expect("checked above");
                engine.set_constrained_look(self.constrained_look);
                let fit = pose.constrain(engine);
                if let Ok(mut interaction) = shared.interaction.lock() {
                    interaction.finish_request(&pose, fit.is_ok());
                }
                let result = fit.and_then(|()| {
                    let current = pose.current_pose();
                    match self.input_format {
                        InputFormat::Yuv420p => {
                            // Wrap OBS planes as StridedYuv420pPlanes and repack
                            // into the cached tight buffers. `copy_into` takes
                            // a single memcpy fast path when stride == width.
                            let left_strided = strided_from_obs(l);
                            let right_strided = strided_from_obs(r);
                            let left_tight = left_strided.copy_into(&mut self.left_repack);
                            let right_tight = right_strided.copy_into(&mut self.right_repack);
                            let session = self.core.as_mut().expect("checked above");
                            session.submit_frame_yuv420p_at_pose(&left_tight, &right_tight, current)
                        }
                        InputFormat::Rgba => {
                            // BGRA sources: swizzle bytes into cached RGBA
                            // buffers once per frame. Shader expects RGB in
                            // .rgb of a single sample. RGBA sources pass
                            // through without swizzle (we go through the same
                            // path because the caller would typically hand us
                            // the right order already).
                            let left_rgba = build_rgba_planes(l, &mut self.left_repack);
                            let right_rgba = build_rgba_planes(r, &mut self.right_repack);
                            let session = self.core.as_mut().expect("checked above");
                            session.submit_frame_rgba_at_pose(&left_rgba, &right_rgba, current)
                        }
                        InputFormat::Nv12 => {
                            // Not yet supported in reco-obs; guarded above.
                            // Return a Warmup so the match arms below align.
                            Ok(RenderOutcome::Warmup)
                        }
                    }
                });
                match result {
                    Ok(RenderOutcome::Rgba(rgba)) => {
                        if let Ok(mut display) = shared.presentation.lock() {
                            display.rgba.resize(rgba.len(), 0);
                            display.rgba.copy_from_slice(rgba);
                            display.dirty = true;
                        }
                        if self.diag_submitted == 0 {
                            log::info!(
                                "reco-obs: first stitched frame ready ({} bytes)",
                                rgba.len()
                            );
                        }
                        self.diag_submitted += 1;
                    }
                    Ok(RenderOutcome::Warmup) => { /* pipeline warmup */ }
                    Err(e) => {
                        log::error!("reco-obs: render/readback failed: {e}");
                    }
                }
            }
        }
    }
}

/// Acquired OBS frame, released on every return path, including a caught panic.
struct AsyncFrame {
    source: *mut ffi::obs_source_t,
    frame: *mut ffi::obs_source_frame,
}

impl AsyncFrame {
    unsafe fn acquire(source: *mut ffi::obs_source_t) -> Option<Self> {
        let frame = unsafe { ffi::obs_source_get_frame(source) };
        (!frame.is_null()).then_some(Self { source, frame })
    }

    fn frame(&self) -> &ffi::obs_source_frame {
        // The owned source reference outlives this guard, and OBS retains the frame until release.
        unsafe { &*self.frame }
    }
}

impl Drop for AsyncFrame {
    fn drop(&mut self) {
        unsafe { ffi::obs_source_release_frame(self.source, self.frame) };
    }
}

impl Presentation {
    /// Destroy the OBS texture (must be called on the OBS graphics thread).
    unsafe fn destroy_obs_texture(&mut self) {
        if !self.texture.is_null() {
            unsafe {
                ffi::gs_texture_destroy(self.texture.pointer);
            }
            self.texture.pointer = ptr::null_mut();
        }
    }

    /// Ensure the OBS texture exists with the correct dimensions.
    /// Must be called on the OBS graphics thread.
    unsafe fn ensure_obs_texture(&mut self) {
        if (self.texture_width, self.texture_height) != (self.width, self.height) {
            unsafe {
                self.destroy_obs_texture();
            }
        }
        if self.texture.is_null() && self.width > 0 && self.height > 0 {
            // GS_DYNAMIC is required for gs_texture_set_image to work;
            // without it the texture is static and all updates silently
            // fail with "Texture is not dynamic" spam in the OBS log.
            self.texture.pointer = unsafe {
                ffi::gs_texture_create(
                    self.width,
                    self.height,
                    ffi::gs_color_format::GS_RGBA,
                    1,
                    ptr::null_mut(),
                    ffi::GS_DYNAMIC,
                )
            };
            if self.texture.is_null() {
                log::error!("reco-obs: failed to create OBS texture");
            } else {
                self.texture_width = self.width;
                self.texture_height = self.height;
                log::info!(
                    "reco-obs: OBS texture created ({}x{}, GS_RGBA, dynamic)",
                    self.width,
                    self.height
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C callback implementations
// ---------------------------------------------------------------------------

/// `get_name`: return the human-readable source name.
unsafe extern "C" fn source_get_name(_type_data: *mut c_void) -> *const c_char {
    crate::ffi_catch!(ptr::null(), { SOURCE_NAME.as_ptr() })
}

/// `create`: allocate source state from settings.
unsafe extern "C" fn source_create(
    settings: *mut ffi::obs_data_t,
    _source: *mut ffi::obs_source_t,
) -> *mut c_void {
    crate::ffi_catch!(ptr::null_mut(), {
        log::info!("reco-obs: creating source");

        let src = Box::new(SourceHandle::default());
        let mut engine = src.engine.lock().expect("new source mutex");

        // Read initial settings.
        unsafe {
            apply_settings(&mut engine, settings, &src.interaction);
        }

        // Try to initialize the pipeline if calibration is available.
        engine.load_calibration();
        engine.try_init_pipeline();
        src.reset_presentation(&engine);
        drop(engine);

        Box::into_raw(src) as *mut c_void
    })
}

/// `destroy`: free source state.
unsafe extern "C" fn source_destroy(data: *mut c_void) {
    crate::ffi_catch!((), {
        if data.is_null() {
            return;
        }
        log::info!("reco-obs: destroying source");

        // OBS removes the instance from callbacks before destruction. Resource owners also clean up poisoned mutex contents.
        drop(unsafe { Box::from_raw(data as *mut SourceHandle) });
    })
}

/// `get_width`: return output width.
unsafe extern "C" fn source_get_width(data: *mut c_void) -> u32 {
    crate::ffi_catch!(0u32, {
        if data.is_null() {
            return 0;
        }
        let src = unsafe { &*(data as *const SourceHandle) };
        src.width.load(Ordering::Relaxed)
    })
}

/// `get_height`: return output height.
unsafe extern "C" fn source_get_height(data: *mut c_void) -> u32 {
    crate::ffi_catch!(0u32, {
        if data.is_null() {
            return 0;
        }
        let src = unsafe { &*(data as *const SourceHandle) };
        src.height.load(Ordering::Relaxed)
    })
}

/// `get_defaults`: set default property values.
unsafe extern "C" fn source_get_defaults(settings: *mut ffi::obs_data_t) {
    crate::ffi_catch!((), {
        unsafe {
            ffi::obs_data_set_default_int(settings, PROP_OUTPUT_WIDTH.as_ptr(), 1920);
            ffi::obs_data_set_default_bool(settings, PROP_CONSTRAINED.as_ptr(), true);
            ffi::obs_data_set_default_int(settings, PROP_OUTPUT_HEIGHT.as_ptr(), 1080);
            ffi::obs_data_set_default_string(settings, PROP_CONFIG_PATH.as_ptr(), c"".as_ptr());
            #[cfg(feature = "replay")]
            {
                ffi::obs_data_set_default_bool(settings, PROP_REPLAY_ENABLED.as_ptr(), false);
                ffi::obs_data_set_default_string(settings, PROP_REPLAY_PATH.as_ptr(), c"".as_ptr());
                ffi::obs_data_set_default_string(
                    settings,
                    PROP_REPLAY_MODE.as_ptr(),
                    REPLAY_MODE_FOLLOW_OBS.as_ptr(),
                );
            }
            ffi::obs_data_set_default_string(
                settings,
                PROP_INPUT_FORMAT.as_ptr(),
                INPUT_FORMAT_I420.as_ptr(),
            );
        }
    })
}

/// `get_properties`: define the settings UI.
unsafe extern "C" fn source_get_properties(_data: *mut c_void) -> *mut ffi::obs_properties_t {
    crate::ffi_catch!(ptr::null_mut(), {
        unsafe {
            let props = ffi::obs_properties_create();

            // Left / right upstream source pickers. Populated via
            // obs_enum_sources at property-open time.
            let left_list = ffi::obs_properties_add_list(
                props,
                PROP_LEFT_SOURCE.as_ptr(),
                c"Left camera source".as_ptr(),
                ffi::obs_combo_type::OBS_COMBO_TYPE_LIST,
                ffi::obs_combo_format::OBS_COMBO_FORMAT_STRING,
            );
            ffi::obs_enum_sources(Some(source_enum_proc_list), left_list as *mut c_void);

            let right_list = ffi::obs_properties_add_list(
                props,
                PROP_RIGHT_SOURCE.as_ptr(),
                c"Right camera source".as_ptr(),
                ffi::obs_combo_type::OBS_COMBO_TYPE_LIST,
                ffi::obs_combo_format::OBS_COMBO_FORMAT_STRING,
            );
            ffi::obs_enum_sources(Some(source_enum_proc_list), right_list as *mut c_void);

            // Input format picker. Must match the native format OBS's picked
            // sources deliver - OBS can't negotiate, and guessing wrong
            // means no output.
            let format_list = ffi::obs_properties_add_list(
                props,
                PROP_INPUT_FORMAT.as_ptr(),
                c"Input format".as_ptr(),
                ffi::obs_combo_type::OBS_COMBO_TYPE_LIST,
                ffi::obs_combo_format::OBS_COMBO_FORMAT_STRING,
            );
            ffi::obs_property_list_add_string(
                format_list,
                c"I420 (Media Source, V4L2)".as_ptr(),
                INPUT_FORMAT_I420.as_ptr(),
            );
            ffi::obs_property_list_add_string(
                format_list,
                c"Packed RGB camera (BGRA / RGBA)".as_ptr(),
                INPUT_FORMAT_BGRA.as_ptr(),
            );

            ffi::obs_properties_add_path(
                props,
                PROP_CONFIG_PATH.as_ptr(),
                c"Calibration file".as_ptr(),
                ffi::obs_path_type::OBS_PATH_FILE,
                c"JSON files (*.json)".as_ptr(),
                ptr::null(),
            );
            ffi::obs_properties_add_bool(
                props,
                PROP_CONSTRAINED.as_ptr(),
                c"Constrain view to coverage".as_ptr(),
            );

            ffi::obs_properties_add_int(
                props,
                PROP_OUTPUT_WIDTH.as_ptr(),
                c"Output width".as_ptr(),
                320,
                7680,
                1,
            );
            ffi::obs_properties_add_int(
                props,
                PROP_OUTPUT_HEIGHT.as_ptr(),
                c"Output height".as_ptr(),
                240,
                4320,
                1,
            );
            ffi::obs_properties_add_float(
                props,
                PROP_YAW.as_ptr(),
                c"Camera yaw (degrees)".as_ptr(),
                -180.0,
                180.0,
                0.1,
            );
            ffi::obs_properties_add_float(
                props,
                PROP_PITCH.as_ptr(),
                c"Camera pitch (degrees)".as_ptr(),
                -90.0,
                90.0,
                0.1,
            );

            #[cfg(feature = "replay")]
            {
                ffi::obs_properties_add_bool(
                    props,
                    PROP_REPLAY_ENABLED.as_ptr(),
                    c"Record replay (stacked video)".as_ptr(),
                );
                ffi::obs_properties_add_path(
                    props,
                    PROP_REPLAY_PATH.as_ptr(),
                    c"New replay file (.mkv, no overwrite)".as_ptr(),
                    ffi::obs_path_type::OBS_PATH_FILE_SAVE,
                    c"Matroska video (*.mkv)".as_ptr(),
                    ptr::null(),
                );
                let mode_list = ffi::obs_properties_add_list(
                    props,
                    PROP_REPLAY_MODE.as_ptr(),
                    c"Replay trigger".as_ptr(),
                    ffi::obs_combo_type::OBS_COMBO_TYPE_LIST,
                    ffi::obs_combo_format::OBS_COMBO_FORMAT_STRING,
                );
                let follow_index = ffi::obs_property_list_add_string(
                    mode_list,
                    if cfg!(have_frontend_api) {
                        c"Follow OBS Record / Stream"
                    } else {
                        c"Follow unavailable (no OBS frontend)"
                    }
                    .as_ptr(),
                    REPLAY_MODE_FOLLOW_OBS.as_ptr(),
                );
                if !cfg!(have_frontend_api) {
                    ffi::obs_property_list_item_disable(mode_list, follow_index, true);
                }
                ffi::obs_property_list_add_string(
                    mode_list,
                    c"Record independently (advanced)".as_ptr(),
                    REPLAY_MODE_ALWAYS.as_ptr(),
                );
            }

            props
        }
    })
}

/// `update`: apply changed settings.
unsafe extern "C" fn source_update(data: *mut c_void, settings: *mut ffi::obs_data_t) {
    crate::ffi_catch!((), {
        if data.is_null() {
            return;
        }
        let shared = unsafe { &*(data as *const SourceHandle) };
        let Ok(mut locked) = shared.engine.lock() else {
            return;
        };
        let src = &mut *locked;

        let old_config_path = src.config_path.clone();
        let old_output = (src.output_width, src.output_height);

        unsafe {
            apply_settings(src, settings, &shared.interaction);
        }

        // Reload calibration if the config path changed.
        if src.config_path != old_config_path {
            src.load_calibration();
        }

        let config_changed = src.config_path != old_config_path;
        // apply_settings sets src.core to None when the input format
        // changes, so we can use that as our "needs rebuild" signal.
        let format_changed = src.core.is_none() && src.calibration.is_some();

        if config_changed || format_changed {
            shared.reset_presentation(src);
            src.try_init_pipeline();
        } else {
            if (src.output_width, src.output_height) != old_output {
                src.resize_output();
                shared.reset_presentation(src);
            }
            // No pipeline rebuild needed, but replay settings may
            // have flipped. Update the recorder attachment state
            // so toggling the "Record replay" checkbox takes
            // effect immediately.
            src.update_replay_recorder(false);
        }
    })
}

/// `video_tick`: called each frame on the OBS video thread.
///
/// Stitching and input resolution happen here; interaction callbacks only accumulate input.
/// Hot path (30-60 Hz). `catch_unwind` here is the primary guard against
/// a panic in the stitch pipeline crashing the OBS host.
unsafe extern "C" fn source_video_tick(data: *mut c_void, _seconds: c_float) {
    crate::ffi_catch!((), {
        if data.is_null() {
            return;
        }
        let shared = unsafe { &*(data as *const SourceHandle) };
        let Ok(mut src) = shared.engine.lock() else {
            return;
        };
        src.render_and_readback(shared);
    })
}

/// `video_render`: called on the OBS graphics thread to draw the source.
///
/// Uploads the CPU-side RGBA buffer to an OBS texture and draws it.
/// Hot path. `catch_unwind` guards the graphics-thread upload path.
unsafe extern "C" fn source_video_render(data: *mut c_void, _effect: *mut ffi::gs_effect_t) {
    crate::ffi_catch!((), {
        if data.is_null() {
            return;
        }
        let shared = unsafe { &*(data as *const SourceHandle) };
        let Ok(mut src) = shared.presentation.lock() else {
            return;
        };
        if src.rgba.is_empty() {
            return;
        }

        unsafe {
            src.ensure_obs_texture();
        }

        if src.texture.is_null() {
            return;
        }

        // Upload new frame data if available.
        if src.dirty {
            unsafe {
                ffi::gs_texture_set_image(
                    src.texture.pointer,
                    src.rgba.as_ptr(),
                    src.width * 4,
                    false,
                );
            }
            src.dirty = false;
        }

        // Draw via OBS's helper, which respects the outer effect already
        // active when video_render is called. Manually running
        // `gs_effect_loop` here triggers "effect is already active" warnings
        // and no draw lands on screen.
        unsafe {
            ffi::obs_source_draw(src.texture.pointer, 0, 0, src.width, src.height, false);
        }
    })
}

// ---------------------------------------------------------------------------
// Interactive mouse callbacks (click-drag to pan, wheel to zoom)
// ---------------------------------------------------------------------------

// Controls derive drag sensitivity from the viewport and FOV; wheel sensitivity is configured per source.

/// `y_delta` units per standard mouse-wheel notch under the Qt
/// convention OBS inherits (see
/// https://doc.qt.io/qt-6/qwheelevent.html#angleDelta). A standard
/// notch is 15° of wheel rotation = 120 `y_delta` units. Dividing
/// `y_delta` by this gives reco-core's expected tick-count unit;
/// smooth-scroll devices send smaller values and naturally produce
/// fractional notches. This constant stays local to reco-obs
/// because it encodes an OBS/Qt convention, not a reco-core one.
const WHEEL_NOTCHES_PER_DELTA: f32 = 120.0;

/// `mouse_click`: start/end a drag for pan.
///
/// # Safety
/// Invoked by OBS on the main UI thread. `data` and `event` are valid
/// pointers for the lifetime of the call.
unsafe extern "C" fn source_mouse_click(
    data: *mut c_void,
    event: *const ffi::obs_mouse_event,
    r#type: i32,
    mouse_up: bool,
    _click_count: u32,
) {
    crate::ffi_catch!((), {
        if data.is_null() || event.is_null() {
            return;
        }
        let shared = unsafe { &*(data as *const SourceHandle) };
        let ev = unsafe { &*event };
        // Only handle left-button for drag-to-pan. Middle/right reserved for
        // future features (e.g. double-click to reset pose).
        if r#type != ffi::obs_mouse_button_type::MOUSE_LEFT as i32 {
            return;
        }
        if let Ok(mut input) = shared.interaction.lock() {
            input.click(ev.x, ev.y, mouse_up);
        }
    })
}

/// `mouse_move`: while dragging, translate pixel delta to yaw/pitch.
unsafe extern "C" fn source_mouse_move(
    data: *mut c_void,
    event: *const ffi::obs_mouse_event,
    mouse_leave: bool,
) {
    crate::ffi_catch!((), {
        if data.is_null() || event.is_null() {
            return;
        }
        let shared = unsafe { &*(data as *const SourceHandle) };
        let ev = unsafe { &*event };
        if let Ok(mut input) = shared.interaction.lock() {
            input.move_pointer(
                ev.x,
                ev.y,
                mouse_leave,
                shared.height.load(Ordering::Relaxed),
            );
        }
    })
}

/// `mouse_wheel`: scroll to zoom (adjust vertical FOV).
unsafe extern "C" fn source_mouse_wheel(
    data: *mut c_void,
    _event: *const ffi::obs_mouse_event,
    _x_delta: i32,
    y_delta: i32,
) {
    crate::ffi_catch!((), {
        if data.is_null() {
            return;
        }
        let shared = unsafe { &*(data as *const SourceHandle) };
        // y_delta is in eighths of a degree of wheel rotation
        // (Qt's QWheelEvent::angleDelta convention, inherited
        // through OBS). Divide by 120 to normalize to "notches"
        // (one standard mouse-wheel click); smooth-scroll devices
        // produce fractional notches. PoseControl handles sign
        // conventions, FOV sensitivity, and clamping internally.
        let notches = (y_delta as f32) / WHEEL_NOTCHES_PER_DELTA;
        if let Ok(mut input) = shared.interaction.lock() {
            input.pose.apply_wheel(notches);
        }
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a [`RgbaPlanes`] view over the source frame's RGBA data,
/// swizzling from BGRA into the cached buffer when needed.
///
/// OBS packed RGB formats (BGRA, BGRX, RGBA) all store 4 bytes per
/// pixel with `linesize[0]` bytes per row (may include padding). We
/// strip the stride padding into the cached buffer and reorder to
/// RGBA so the shader can sample `.rgb` directly.
fn build_rgba_planes<'a>(frame: &ffi::obs_source_frame, buffer: &'a mut Vec<u8>) -> RgbaPlanes<'a> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let stride = frame.linesize[0] as usize;
    let tight_size = w * h * 4;
    buffer.resize(tight_size, 0);
    // SAFETY: frame.data[0] is valid for linesize[0] * height bytes
    // between get_frame/release_frame, guaranteed by OBS.
    let src = unsafe { std::slice::from_raw_parts(frame.data[0] as *const u8, stride * h) };
    let bgra_mode = matches!(
        frame.format,
        ffi::video_format::VIDEO_FORMAT_BGRA | ffi::video_format::VIDEO_FORMAT_BGRX
    );
    for row in 0..h {
        let src_row = &src[row * stride..row * stride + w * 4];
        let dst_row = &mut buffer[row * w * 4..(row + 1) * w * 4];
        if bgra_mode {
            for (s, d) in src_row.chunks_exact(4).zip(dst_row.chunks_exact_mut(4)) {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        } else {
            // VIDEO_FORMAT_RGBA: already in the order the shader wants.
            dst_row.copy_from_slice(src_row);
        }
    }
    RgbaPlanes::from_rgba(buffer)
}

/// Wrap an `obs_source_frame` (I420 only) as a [`StridedYuv420pPlanes`] view.
///
/// Assumes the caller has already verified `format == VIDEO_FORMAT_I420`.
/// OBS delivers I420 with plane[0]=Y (full-res), plane[1]=U, plane[2]=V
/// (each half-res per dimension).
fn strided_from_obs<'a>(frame: &'a ffi::obs_source_frame) -> StridedYuv420pPlanes<'a> {
    let half_w = frame.width / 2;
    let half_h = frame.height / 2;
    // SAFETY: OBS guarantees data[i] points to linesize[i] * plane_height
    // bytes while the frame is held. We build slices of exactly that length
    // so FramePlaneView can traverse safely via stride indexing.
    let y_slice = unsafe {
        std::slice::from_raw_parts(
            frame.data[0] as *const u8,
            (frame.linesize[0] as usize) * (frame.height as usize),
        )
    };
    let u_slice = unsafe {
        std::slice::from_raw_parts(
            frame.data[1] as *const u8,
            (frame.linesize[1] as usize) * (half_h as usize),
        )
    };
    let v_slice = unsafe {
        std::slice::from_raw_parts(
            frame.data[2] as *const u8,
            (frame.linesize[2] as usize) * (half_h as usize),
        )
    };
    StridedYuv420pPlanes {
        y: FramePlaneView {
            data: y_slice,
            stride: frame.linesize[0],
            width: frame.width,
            height: frame.height,
        },
        u: FramePlaneView {
            data: u_slice,
            stride: frame.linesize[1],
            width: half_w,
            height: half_h,
        },
        v: FramePlaneView {
            data: v_slice,
            stride: frame.linesize[2],
            width: half_w,
            height: half_h,
        },
    }
}

/// List only source types that the acquisition boundary can accept.
unsafe extern "C" fn source_enum_proc_list(
    param: *mut c_void,
    source: *mut ffi::obs_source_t,
) -> bool {
    crate::ffi_catch!(true, {
        if param.is_null() || source.is_null() {
            return true;
        }
        if !accepts_source(unsafe { ffi::obs_source_get_output_flags(source) }) {
            return true;
        }
        let prop = param as *mut ffi::obs_property_t;
        let name_ptr = unsafe { ffi::obs_source_get_name(source) };
        if !name_ptr.is_null() {
            unsafe {
                // Use the same cstring as both label and value.
                ffi::obs_property_list_add_string(prop, name_ptr, name_ptr);
            }
        }
        true
    })
}

/// Read settings from an `obs_data_t` into the source struct.
///
/// # Safety
///
/// `settings` must be a valid pointer from OBS.
unsafe fn apply_settings(
    src: &mut RecoSource,
    settings: *mut ffi::obs_data_t,
    interaction: &Mutex<Interaction>,
) {
    unsafe {
        // Config path.
        let c_str = ffi::obs_data_get_string(settings, PROP_CONFIG_PATH.as_ptr());
        if !c_str.is_null() {
            let s = CStr::from_ptr(c_str);
            src.config_path = s.to_string_lossy().into_owned();
        }

        // Dimensions.
        let w = ffi::obs_data_get_int(settings, PROP_OUTPUT_WIDTH.as_ptr());
        if w > 0 {
            src.output_width = w as u32;
        }
        let h = ffi::obs_data_get_int(settings, PROP_OUTPUT_HEIGHT.as_ptr());
        if h > 0 {
            src.output_height = h as u32;
        }

        // Input format choice. Parse string -> InputFormat. Unknown
        // strings fall back to I420 with a one-shot warning.
        let format_ptr = ffi::obs_data_get_string(settings, PROP_INPUT_FORMAT.as_ptr());
        let new_format = if format_ptr.is_null() {
            InputFormat::Yuv420p
        } else {
            match CStr::from_ptr(format_ptr).to_str() {
                Ok("bgra") => InputFormat::Rgba,
                Ok("i420") | Ok("") => InputFormat::Yuv420p,
                Ok(other) => {
                    log::warn!("reco-obs: unknown input_format '{other}', defaulting to I420");
                    InputFormat::Yuv420p
                }
                Err(_) => InputFormat::Yuv420p,
            }
        };
        if new_format != src.input_format {
            src.input_format = new_format;
            // Existing session was built with the old format; drop it so
            // source_update rebuilds. This happens after apply_settings
            // returns (the dim / config / format-change gate reruns
            // try_init_pipeline).
            src.core = None;
            src.warned_unsupported_format = false;
        }

        // Upstream source picks. Empty string clears the slot.
        let left_ptr = ffi::obs_data_get_string(settings, PROP_LEFT_SOURCE.as_ptr());
        let left_name = if left_ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(left_ptr).to_string_lossy().into_owned()
        };
        if src.left_source.select(&left_name) {
            src.warned_unsupported_format = false;
        }

        let right_ptr = ffi::obs_data_get_string(settings, PROP_RIGHT_SOURCE.as_ptr());
        let right_name = if right_ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(right_ptr).to_string_lossy().into_owned()
        };
        if src.right_source.select(&right_name) {
            src.warned_unsupported_format = false;
        }

        // Property sliders and mouse input share the same render-space controls.
        let target_yaw_deg = ffi::obs_data_get_double(settings, PROP_YAW.as_ptr()) as f32;
        let target_pitch_deg = ffi::obs_data_get_double(settings, PROP_PITCH.as_ptr()) as f32;
        src.constrained_look = ffi::obs_data_get_bool(settings, PROP_CONSTRAINED.as_ptr());
        if let Ok(mut input) = interaction.lock() {
            input.apply_property_angles(target_yaw_deg, target_pitch_deg);
        }

        // Replay recording toggle + path. Recorder attachment itself
        // happens in `try_init_pipeline` / `update_replay_recorder` —
        // this block only captures the property state.
        #[cfg(feature = "replay")]
        {
            src.replay_enabled = ffi::obs_data_get_bool(settings, PROP_REPLAY_ENABLED.as_ptr());
            let replay_path_ptr = ffi::obs_data_get_string(settings, PROP_REPLAY_PATH.as_ptr());
            src.replay_path = if replay_path_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(replay_path_ptr)
                    .to_string_lossy()
                    .into_owned()
            };
            let mode_ptr = ffi::obs_data_get_string(settings, PROP_REPLAY_MODE.as_ptr());
            src.replay_follow_obs = if mode_ptr.is_null() {
                true
            } else {
                // Anything other than the explicit "always" string
                // falls back to follow-OBS. Keeps future mode
                // additions safe by default.
                !matches!(CStr::from_ptr(mode_ptr).to_str(), Ok("always"))
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Source info definition
// ---------------------------------------------------------------------------

/// Build the `obs_source_info` struct with all our callbacks.
///
/// This is called once from `obs_module_load`.
pub(crate) fn source_info() -> ffi::obs_source_info {
    ffi::obs_source_info {
        id: SOURCE_ID.as_ptr(),
        type_: ffi::obs_source_type::OBS_SOURCE_TYPE_INPUT,
        output_flags: ffi::OBS_SOURCE_VIDEO | ffi::OBS_SOURCE_INTERACTION,
        get_name: Some(source_get_name),
        create: Some(source_create),
        destroy: Some(source_destroy),
        get_width: Some(source_get_width),
        get_height: Some(source_get_height),
        get_defaults: Some(source_get_defaults),
        get_properties: Some(source_get_properties),
        update: Some(source_update),
        video_tick: Some(source_video_tick),
        video_render: Some(source_video_render),
        mouse_click: Some(source_mouse_click),
        mouse_move: Some(source_mouse_move),
        mouse_wheel: Some(source_mouse_wheel),
        icon_type: ffi::obs_icon_type::OBS_ICON_TYPE_CAMERA,
        // SAFETY: obs_source_info contains scalars (both enums have a zero variant),
        // pointers and optional callbacks. New optional callbacks default to null.
        ..unsafe { std::mem::zeroed() }
    }
}
