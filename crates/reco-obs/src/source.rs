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
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::calibration::MatchCalibration;
use reco_core::gpu::GpuContext;
use reco_core::pipeline::{BgraPlanes, FramePlaneView, StridedYuvPlanes};
use reco_core::renderer::InputFormat;
use reco_core::session::{LiveSessionConfig, LiveStitchSession};
use reco_core::viewport::ViewportConfig;

use crate::ffi;

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
const PROP_INPUT_WIDTH: &CStr = c"input_width";
const PROP_INPUT_HEIGHT: &CStr = c"input_height";
const PROP_YAW: &CStr = c"yaw";
const PROP_PITCH: &CStr = c"pitch";
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

// ---------------------------------------------------------------------------
// Source state
// ---------------------------------------------------------------------------

/// Per-instance state for a Reco stitcher source.
///
/// Allocated in `create`, freed in `destroy`. OBS passes `*mut c_void`
/// pointing to this through all callbacks.
struct RecoSource {
    /// High-level push session (bundles pipeline + RGBA readback).
    /// None until calibration is loaded and GPU is initialized.
    session: Option<LiveStitchSession>,

    /// Upstream OBS source that feeds the left camera. Owned reference
    /// (obs_get_source_by_name increments refcount, we release in destroy
    /// or when the user picks a different source).
    left_source: *mut ffi::obs_source_t,
    /// Upstream OBS source that feeds the right camera. See [`Self::left_source`].
    right_source: *mut ffi::obs_source_t,
    /// Source name the user picked for the left camera (used to detect changes).
    left_source_name: String,
    /// Source name the user picked for the right camera.
    right_source_name: String,
    /// Reusable tight-pack buffers for `StridedYuvPlanes::copy_into`. One
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
    /// Count of video_render invocations since source creation. Logged
    /// alongside the regular diag heartbeat so we can distinguish
    /// "stitching runs but OBS never asks us to draw" from
    /// "render runs but the texture isn't repainting".
    diag_render_calls: u64,
    /// Count of texture uploads (gs_texture_set_image calls) - should
    /// track diag_submitted when the display path is healthy.
    diag_uploads: u64,

    /// Current calibration loaded from the config file.
    calibration: Option<MatchCalibration>,

    /// Path to the calibration JSON file.
    config_path: String,

    /// Output viewport dimensions.
    output_width: u32,
    output_height: u32,

    /// Expected input frame dimensions (from the camera sources).
    input_width: u32,
    input_height: u32,

    /// Camera yaw in degrees (converted to radians for the pipeline).
    yaw_degrees: f64,

    /// Camera pitch in degrees (converted to radians for the pipeline).
    pitch_degrees: f64,

    /// CPU-side RGBA buffer from the last render (owned copy of the most
    /// recent frame from `RgbaReadback`, so `video_render` can read it on
    /// the OBS graphics thread without holding a borrow).
    rgba_buffer: Vec<u8>,

    /// Whether `rgba_buffer` has new data since the last `video_render`.
    frame_ready: AtomicBool,

    /// OBS graphics texture for uploading RGBA data.
    /// Created/destroyed on the OBS graphics thread.
    obs_texture: *mut ffi::gs_texture_t,
}

impl RecoSource {
    fn new() -> Self {
        Self {
            session: None,
            left_source: ptr::null_mut(),
            right_source: ptr::null_mut(),
            left_source_name: String::new(),
            right_source_name: String::new(),
            left_repack: Vec::new(),
            right_repack: Vec::new(),
            input_format: InputFormat::Yuv420p,
            warned_unsupported_format: false,
            diag_tick: 0,
            diag_missed_left: 0,
            diag_missed_right: 0,
            diag_submitted: 0,
            diag_render_calls: 0,
            diag_uploads: 0,
            calibration: None,
            config_path: String::new(),
            output_width: 1920,
            output_height: 1080,
            input_width: 1920,
            input_height: 1080,
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            rgba_buffer: Vec::new(),
            frame_ready: AtomicBool::new(false),
            obs_texture: ptr::null_mut(),
        }
    }

    /// Replace `slot` with a new OBS source reference (resolved from
    /// `new_name` via `obs_get_source_by_name`). Releases the old ref if
    /// one was held, and pairs `inc_showing` with `dec_showing` so the
    /// upstream source (e.g. a Media Source) keeps its decoder running
    /// while we're pulling frames from it - without that, OBS can
    /// deactivate the source when it isn't visibly rendered anywhere,
    /// which manifests as intermittent freezes in our stitched output.
    ///
    /// Safe to call with an empty name - that just releases the old ref.
    unsafe fn set_source_slot(slot: &mut *mut ffi::obs_source_t, new_name: &str) {
        unsafe {
            if !slot.is_null() {
                ffi::obs_source_dec_active(*slot);
                ffi::obs_source_dec_showing(*slot);
                ffi::obs_source_release(*slot);
                *slot = ptr::null_mut();
            }
            if new_name.is_empty() {
                return;
            }
            let cstr = match std::ffi::CString::new(new_name) {
                Ok(s) => s,
                Err(_) => {
                    log::warn!("reco-obs: source name contains NUL, ignoring");
                    return;
                }
            };
            let ptr = ffi::obs_get_source_by_name(cstr.as_ptr());
            if ptr.is_null() {
                log::warn!("reco-obs: upstream source '{new_name}' not found (not yet created?)");
            } else {
                // Hold both showing + active refs. Media Source's
                // ffmpeg decode thread pauses on active==0 even when
                // showing>0, so we need both to keep playback stable
                // while the upstream isn't visibly rendered in the scene.
                ffi::obs_source_inc_showing(ptr);
                ffi::obs_source_inc_active(ptr);
                log::info!("reco-obs: holding '{new_name}' via inc_showing + inc_active");
            }
            *slot = ptr;
        }
    }

    /// Try to initialize (or reinitialize) the pipeline from current settings.
    ///
    /// Called after calibration is loaded or dimensions change.
    fn try_init_pipeline(&mut self) {
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

        let viewport = ViewportConfig {
            width: self.output_width,
            height: self.output_height,
            ..ViewportConfig::default()
        };

        match LiveStitchSession::new(
            gpu,
            LiveSessionConfig {
                calibration,
                viewport,
                input_width: self.input_width,
                input_height: self.input_height,
                output_format: reco_core::wgpu::TextureFormat::Rgba8Unorm,
                input_format: self.input_format,
            },
        ) {
            Ok(session) => {
                log::info!(
                    "reco-obs: session initialized ({}x{} output, {}x{} input, format={:?})",
                    self.output_width,
                    self.output_height,
                    self.input_width,
                    self.input_height,
                    self.input_format,
                );
                self.session = Some(session);
                let buf_size = (self.output_width * self.output_height * 4) as usize;
                self.rgba_buffer.resize(buf_size, 0);
            }
            Err(e) => {
                log::error!("reco-obs: failed to create session: {e}");
                self.session = None;
            }
        }
    }

    /// Load calibration from the config file path.
    fn load_calibration(&mut self) {
        if self.config_path.is_empty() {
            self.calibration = None;
            self.session = None;
            return;
        }

        let path = Path::new(&self.config_path);
        match MatchCalibration::from_file(path) {
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
                self.session = None;
            }
        }
    }

    /// Pull the latest async frame pair from the upstream OBS sources,
    /// stitch, and stash the RGBA result for `video_render`.
    ///
    /// Returns silently (no render) when either upstream source is unset,
    /// has no current frame, or delivers a format other than I420. The
    /// `warned_unsupported_format` flag ensures unsupported-format warnings
    /// fire at most once per source instance.
    fn render_and_readback(&mut self) {
        self.diag_tick = self.diag_tick.wrapping_add(1);
        // Every ~60 ticks (~1-2 seconds) flush a diagnostic heartbeat so
        // users can tell "plugin running but no render" from "plugin hung".
        if self.diag_tick.is_multiple_of(60) {
            log::info!(
                "reco-obs: diag tick={} submitted={} uploads={} renders={} \
                 missed_left={} missed_right={} session={} left_src={} right_src={}",
                self.diag_tick,
                self.diag_submitted,
                self.diag_uploads,
                self.diag_render_calls,
                self.diag_missed_left,
                self.diag_missed_right,
                if self.session.is_some() { "ok" } else { "none" },
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

        if self.session.is_none() {
            return;
        }
        if self.left_source.is_null() || self.right_source.is_null() {
            return;
        }

        // Acquire frames (obs_source_get_frame increments the internal
        // ref; must pair with release_frame).
        let left_frame = unsafe { ffi::obs_source_get_frame(self.left_source) };
        let right_frame = unsafe { ffi::obs_source_get_frame(self.right_source) };
        if left_frame.is_null() {
            self.diag_missed_left += 1;
        }
        if right_frame.is_null() {
            self.diag_missed_right += 1;
        }

        if !left_frame.is_null() && !right_frame.is_null() {
            // SAFETY: both pointers non-null; OBS guarantees the frames
            // are valid between get_frame and release_frame.
            let l = unsafe { &*left_frame };
            let r = unsafe { &*right_frame };

            // Must match what the session was built with. Mismatch =>
            // warn once and skip. Camera inputs (V4L2, Media Source) are
            // I420; browser / screen capture / WebRTC are BGRA.
            let format_ok = match self.input_format {
                InputFormat::Yuv420p => {
                    l.format == ffi::video_format::VIDEO_FORMAT_I420
                        && r.format == ffi::video_format::VIDEO_FORMAT_I420
                }
                InputFormat::Bgra => {
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
                         format' property to match the source type: I420 for V4L2 / Media \
                         Source, BGRA for Browser Source / screen capture / WebRTC.",
                        self.input_format,
                        self.input_format,
                        l.format,
                        r.format,
                    );
                    self.warned_unsupported_format = true;
                }
            } else if l.width != self.input_width
                || l.height != self.input_height
                || r.width != self.input_width
                || r.height != self.input_height
            {
                if !self.warned_unsupported_format {
                    log::warn!(
                        "reco-obs: input-dim mismatch (configured {}x{}, left={}x{}, \
                         right={}x{}). Update the 'Input width/height' properties to match \
                         your camera.",
                        self.input_width,
                        self.input_height,
                        l.width,
                        l.height,
                        r.width,
                        r.height,
                    );
                    self.warned_unsupported_format = true;
                }
            } else {
                let yaw = (self.yaw_degrees as f32).to_radians();
                let pitch = (self.pitch_degrees as f32).to_radians();
                let result = match self.input_format {
                    InputFormat::Yuv420p => {
                        // Wrap OBS planes as StridedYuvPlanes and repack
                        // into the cached tight buffers. `copy_into` takes
                        // a single memcpy fast path when stride == width.
                        let left_strided = strided_from_obs(l);
                        let right_strided = strided_from_obs(r);
                        let left_tight = left_strided.copy_into(&mut self.left_repack);
                        let right_tight = right_strided.copy_into(&mut self.right_repack);
                        let session = self.session.as_mut().expect("checked above");
                        session.submit_frame(&left_tight, &right_tight, yaw, pitch)
                    }
                    InputFormat::Bgra => {
                        // BGRA sources: swizzle bytes into cached RGBA
                        // buffers once per frame. Shader expects RGB in
                        // .rgb of a single sample. RGBA sources pass
                        // through without swizzle (we go through the same
                        // path because the caller would typically hand us
                        // the right order already).
                        let left_bgra = build_bgra_planes(l, &mut self.left_repack);
                        let right_bgra = build_bgra_planes(r, &mut self.right_repack);
                        let session = self.session.as_mut().expect("checked above");
                        session.submit_frame_bgra(&left_bgra, &right_bgra, yaw, pitch)
                    }
                    InputFormat::Nv12 => {
                        // Not yet supported in reco-obs; guarded above.
                        Ok(None)
                    }
                };
                match result {
                    Ok(Some(rgba)) => {
                        self.rgba_buffer.copy_from_slice(rgba);
                        self.frame_ready.store(true, Ordering::Release);
                        if self.diag_submitted == 0 {
                            log::info!(
                                "reco-obs: first stitched frame ready ({} bytes)",
                                rgba.len()
                            );
                        }
                        self.diag_submitted += 1;
                    }
                    Ok(None) => { /* pipeline warmup */ }
                    Err(e) => {
                        log::error!("reco-obs: render/readback failed: {e}");
                    }
                }
            }
        }

        // Always release frames we acquired.
        if !left_frame.is_null() {
            unsafe { ffi::obs_source_release_frame(self.left_source, left_frame) };
        }
        if !right_frame.is_null() {
            unsafe { ffi::obs_source_release_frame(self.right_source, right_frame) };
        }
    }

    /// Destroy the OBS texture (must be called on the OBS graphics thread).
    unsafe fn destroy_obs_texture(&mut self) {
        if !self.obs_texture.is_null() {
            unsafe {
                ffi::gs_texture_destroy(self.obs_texture);
            }
            self.obs_texture = ptr::null_mut();
        }
    }

    /// Ensure the OBS texture exists with the correct dimensions.
    /// Must be called on the OBS graphics thread.
    unsafe fn ensure_obs_texture(&mut self) {
        if self.obs_texture.is_null() && self.output_width > 0 && self.output_height > 0 {
            // GS_DYNAMIC is required for gs_texture_set_image to work;
            // without it the texture is static and all updates silently
            // fail with "Texture is not dynamic" spam in the OBS log.
            self.obs_texture = unsafe {
                ffi::gs_texture_create(
                    self.output_width,
                    self.output_height,
                    ffi::gs_color_format::GS_RGBA,
                    1,
                    ptr::null(),
                    ffi::GS_DYNAMIC,
                )
            };
            if self.obs_texture.is_null() {
                log::error!("reco-obs: failed to create OBS texture");
            } else {
                log::info!(
                    "reco-obs: OBS texture created ({}x{}, GS_RGBA, dynamic)",
                    self.output_width,
                    self.output_height
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
    SOURCE_NAME.as_ptr()
}

/// `create`: allocate source state from settings.
unsafe extern "C" fn source_create(
    settings: *mut ffi::obs_data_t,
    _source: *mut ffi::obs_source_t,
) -> *mut c_void {
    log::info!("reco-obs: creating source");

    let mut src = Box::new(RecoSource::new());

    // Read initial settings.
    unsafe {
        apply_settings(&mut src, settings);
    }

    // Try to initialize the pipeline if calibration is available.
    src.load_calibration();
    src.try_init_pipeline();

    Box::into_raw(src) as *mut c_void
}

/// `destroy`: free source state.
unsafe extern "C" fn source_destroy(data: *mut c_void) {
    if data.is_null() {
        return;
    }
    log::info!("reco-obs: destroying source");

    let mut src = unsafe { Box::from_raw(data as *mut RecoSource) };

    // Release the upstream source refs we held. Must run before the Box
    // is dropped - OBS tracks ref lifetime through these calls. Pair
    // dec_showing with the inc_showing set_source_slot issued earlier.
    unsafe {
        if !src.left_source.is_null() {
            ffi::obs_source_dec_active(src.left_source);
            ffi::obs_source_dec_showing(src.left_source);
            ffi::obs_source_release(src.left_source);
            src.left_source = ptr::null_mut();
        }
        if !src.right_source.is_null() {
            ffi::obs_source_dec_active(src.right_source);
            ffi::obs_source_dec_showing(src.right_source);
            ffi::obs_source_release(src.right_source);
            src.right_source = ptr::null_mut();
        }
    }

    // Destroy OBS texture on the graphics thread.
    unsafe {
        ffi::obs_enter_graphics();
        src.destroy_obs_texture();
        ffi::obs_leave_graphics();
    }

    // src is dropped here, which drops the pipeline and GPU context.
}

/// `get_width`: return output width.
unsafe extern "C" fn source_get_width(data: *mut c_void) -> u32 {
    if data.is_null() {
        return 0;
    }
    let src = unsafe { &*(data as *const RecoSource) };
    src.output_width
}

/// `get_height`: return output height.
unsafe extern "C" fn source_get_height(data: *mut c_void) -> u32 {
    if data.is_null() {
        return 0;
    }
    let src = unsafe { &*(data as *const RecoSource) };
    src.output_height
}

/// `get_defaults`: set default property values.
unsafe extern "C" fn source_get_defaults(settings: *mut ffi::obs_data_t) {
    unsafe {
        ffi::obs_data_set_default_int(settings, PROP_OUTPUT_WIDTH.as_ptr(), 1920);
        ffi::obs_data_set_default_int(settings, PROP_OUTPUT_HEIGHT.as_ptr(), 1080);
        ffi::obs_data_set_default_int(settings, PROP_INPUT_WIDTH.as_ptr(), 1920);
        ffi::obs_data_set_default_int(settings, PROP_INPUT_HEIGHT.as_ptr(), 1080);
        ffi::obs_data_set_default_string(settings, PROP_CONFIG_PATH.as_ptr(), c"".as_ptr());
        ffi::obs_data_set_default_string(
            settings,
            PROP_INPUT_FORMAT.as_ptr(),
            INPUT_FORMAT_I420.as_ptr(),
        );
    }
}

/// `get_properties`: define the settings UI.
unsafe extern "C" fn source_get_properties(_data: *mut c_void) -> *mut ffi::obs_properties_t {
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
            c"BGRA (Browser Source, Screen Capture)".as_ptr(),
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
        ffi::obs_properties_add_int(
            props,
            PROP_INPUT_WIDTH.as_ptr(),
            c"Input width (per camera)".as_ptr(),
            320,
            7680,
            1,
        );
        ffi::obs_properties_add_int(
            props,
            PROP_INPUT_HEIGHT.as_ptr(),
            c"Input height (per camera)".as_ptr(),
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

        props
    }
}

/// `update`: apply changed settings.
unsafe extern "C" fn source_update(data: *mut c_void, settings: *mut ffi::obs_data_t) {
    if data.is_null() {
        return;
    }
    let src = unsafe { &mut *(data as *mut RecoSource) };

    let old_config_path = src.config_path.clone();
    let old_output = (src.output_width, src.output_height);
    let old_input = (src.input_width, src.input_height);

    unsafe {
        apply_settings(src, settings);
    }

    // Reload calibration if the config path changed.
    if src.config_path != old_config_path {
        src.load_calibration();
    }

    // Rebuild pipeline if dimensions or calibration changed.
    let dims_changed = (src.output_width, src.output_height) != old_output
        || (src.input_width, src.input_height) != old_input;
    let config_changed = src.config_path != old_config_path;
    // apply_settings sets src.session to None when the input format
    // changes, so we can use that as our "needs rebuild" signal.
    let format_changed = src.session.is_none() && src.calibration.is_some();

    if dims_changed || config_changed || format_changed {
        // Destroy old OBS texture since dimensions may have changed.
        unsafe {
            ffi::obs_enter_graphics();
            src.destroy_obs_texture();
            ffi::obs_leave_graphics();
        }
        src.try_init_pipeline();
    }
}

/// `video_tick`: called each frame on the OBS video thread.
///
/// We perform the wgpu render + CPU readback here, off the graphics thread.
unsafe extern "C" fn source_video_tick(data: *mut c_void, _seconds: c_float) {
    if data.is_null() {
        return;
    }
    let src = unsafe { &mut *(data as *mut RecoSource) };
    src.render_and_readback();
}

/// `video_render`: called on the OBS graphics thread to draw the source.
///
/// Uploads the CPU-side RGBA buffer to an OBS texture and draws it.
unsafe extern "C" fn source_video_render(data: *mut c_void, _effect: *mut ffi::gs_effect_t) {
    if data.is_null() {
        return;
    }
    let src = unsafe { &mut *(data as *mut RecoSource) };
    src.diag_render_calls = src.diag_render_calls.wrapping_add(1);

    if src.session.is_none() {
        return;
    }

    unsafe {
        src.ensure_obs_texture();
    }

    if src.obs_texture.is_null() {
        return;
    }

    // Upload new frame data if available.
    if src.frame_ready.load(Ordering::Acquire) {
        unsafe {
            ffi::gs_texture_set_image(
                src.obs_texture,
                src.rgba_buffer.as_ptr(),
                src.output_width * 4,
                false,
            );
        }
        src.frame_ready.store(false, Ordering::Release);
        src.diag_uploads = src.diag_uploads.wrapping_add(1);
    }

    // Draw via OBS's helper, which respects the outer effect already
    // active when video_render is called. Manually running
    // `gs_effect_loop` here triggers "effect is already active" warnings
    // and no draw lands on screen.
    unsafe {
        ffi::obs_source_draw(
            src.obs_texture,
            0,
            0,
            src.output_width,
            src.output_height,
            false,
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a [`BgraPlanes`] view over the source frame's RGBA data,
/// swizzling from BGRA into the cached buffer when needed.
///
/// OBS packed RGB formats (BGRA, BGRX, RGBA) all store 4 bytes per
/// pixel with `linesize[0]` bytes per row (may include padding). We
/// strip the stride padding into the cached buffer and reorder to
/// RGBA so the shader can sample `.rgb` directly.
fn build_bgra_planes<'a>(frame: &ffi::obs_source_frame, buffer: &'a mut Vec<u8>) -> BgraPlanes<'a> {
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
    BgraPlanes::from_rgba(buffer)
}

/// Wrap an `obs_source_frame` (I420 only) as a [`StridedYuvPlanes`] view.
///
/// Assumes the caller has already verified `format == VIDEO_FORMAT_I420`.
/// OBS delivers I420 with plane[0]=Y (full-res), plane[1]=U, plane[2]=V
/// (each half-res per dimension).
fn strided_from_obs<'a>(frame: &'a ffi::obs_source_frame) -> StridedYuvPlanes<'a> {
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
    StridedYuvPlanes {
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

/// OBS enumeration callback: appends every source name to the dropdown
/// list passed as `param`. Always returns `true` to continue iterating.
///
/// Filtering out sources that can't deliver async video (scenes,
/// transitions, audio-only inputs) requires additional bindings
/// (`obs_source_get_output_flags`). For Tier 1 we accept everything;
/// picking a bad source just means `obs_source_get_frame` returns null
/// and we skip.
unsafe extern "C" fn source_enum_proc_list(
    param: *mut c_void,
    source: *mut ffi::obs_source_t,
) -> bool {
    if param.is_null() || source.is_null() {
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
}

/// Read settings from an `obs_data_t` into the source struct.
///
/// # Safety
///
/// `settings` must be a valid pointer from OBS.
unsafe fn apply_settings(src: &mut RecoSource, settings: *mut ffi::obs_data_t) {
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
        let iw = ffi::obs_data_get_int(settings, PROP_INPUT_WIDTH.as_ptr());
        if iw > 0 {
            src.input_width = iw as u32;
        }
        let ih = ffi::obs_data_get_int(settings, PROP_INPUT_HEIGHT.as_ptr());
        if ih > 0 {
            src.input_height = ih as u32;
        }

        // Input format choice. Parse string -> InputFormat. Unknown
        // strings fall back to I420 with a one-shot warning.
        let format_ptr = ffi::obs_data_get_string(settings, PROP_INPUT_FORMAT.as_ptr());
        let new_format = if format_ptr.is_null() {
            InputFormat::Yuv420p
        } else {
            match CStr::from_ptr(format_ptr).to_str() {
                Ok("bgra") => InputFormat::Bgra,
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
            src.session = None;
            src.warned_unsupported_format = false;
        }

        // Upstream source picks. Empty string clears the slot.
        let left_ptr = ffi::obs_data_get_string(settings, PROP_LEFT_SOURCE.as_ptr());
        let left_name = if left_ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(left_ptr).to_string_lossy().into_owned()
        };
        if left_name != src.left_source_name {
            RecoSource::set_source_slot(&mut src.left_source, &left_name);
            src.left_source_name = left_name;
            // New source may deliver a different format; re-enable warning.
            src.warned_unsupported_format = false;
        }

        let right_ptr = ffi::obs_data_get_string(settings, PROP_RIGHT_SOURCE.as_ptr());
        let right_name = if right_ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(right_ptr).to_string_lossy().into_owned()
        };
        if right_name != src.right_source_name {
            RecoSource::set_source_slot(&mut src.right_source, &right_name);
            src.right_source_name = right_name;
            src.warned_unsupported_format = false;
        }

        // Viewport position.
        src.yaw_degrees = 0.0; // obs_data_get_double is not bound yet
        src.pitch_degrees = 0.0;
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
        r#type: ffi::obs_source_type::OBS_SOURCE_TYPE_INPUT,
        output_flags: ffi::OBS_SOURCE_VIDEO,
        get_name: Some(source_get_name),
        create: Some(source_create),
        destroy: Some(source_destroy),
        get_width: Some(source_get_width),
        get_height: Some(source_get_height),
        get_defaults: Some(source_get_defaults),
        get_properties: Some(source_get_properties),
        update: Some(source_update),
        activate: None,
        deactivate: None,
        show: None,
        hide: None,
        video_tick: Some(source_video_tick),
        video_render: Some(source_video_render),
        filter_video: None,
        filter_audio: None,
        enum_active_sources: None,
        save: None,
        load: None,
        mouse_click: None,
        mouse_move: None,
        mouse_wheel: None,
        focus: None,
        key_click: None,
        filter_remove: None,
        type_data: ptr::null_mut(),
        free_type_data: None,
        audio_render: None,
        enum_all_sources: None,
        transition_start: None,
        transition_stop: None,
        get_defaults2: None,
        get_properties2: None,
        audio_mix: None,
        icon_type: ffi::obs_icon_type::OBS_ICON_TYPE_CAMERA,
        media_play_pause: None,
        media_restart: None,
        media_stop: None,
        media_next: None,
        media_previous: None,
        media_get_duration: None,
        media_get_time: None,
        media_set_time: None,
        media_get_state: None,
        version: 0,
        unversioned_id: ptr::null(),
        missing_files: None,
        video_get_color_space: None,
        filter_add: None,
    }
}
