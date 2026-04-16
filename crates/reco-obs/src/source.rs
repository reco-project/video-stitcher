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
use reco_core::pipeline::{StitchPipeline, YuvPlanes};
use reco_core::renderer::InputFormat;
use reco_core::rgba_readback::RgbaReadback;
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

// ---------------------------------------------------------------------------
// Source state
// ---------------------------------------------------------------------------

/// Per-instance state for a Reco stitcher source.
///
/// Allocated in `create`, freed in `destroy`. OBS passes `*mut c_void`
/// pointing to this through all callbacks.
struct RecoSource {
    /// The wgpu stitching pipeline (None until calibration is loaded).
    pipeline: Option<StitchPipeline>,

    /// RGBA readback helper (triple-buffered staging + row-padding strip).
    /// None until the pipeline is built.
    readback: Option<RgbaReadback>,

    /// The shared GPU context.
    gpu: Option<GpuContext>,

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
            pipeline: None,
            readback: None,
            gpu: None,
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

        // Initialize GPU context if we don't have one yet.
        if self.gpu.is_none() {
            match pollster::block_on(GpuContext::new()) as Result<GpuContext, _> {
                Ok(gpu) => {
                    log::info!("reco-obs: GPU initialized: {}", gpu.gpu_name());
                    self.gpu = Some(gpu);
                }
                Err(e) => {
                    log::error!("reco-obs: failed to initialize GPU: {e}");
                    return;
                }
            }
        }

        let gpu = self.gpu.as_ref().unwrap().clone();
        let viewport = ViewportConfig {
            width: self.output_width,
            height: self.output_height,
            ..ViewportConfig::default()
        };

        match StitchPipeline::with_gpu(
            gpu,
            calibration,
            viewport,
            self.input_width,
            self.input_height,
            reco_core::wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Yuv420p,
        ) {
            Ok(pipeline) => {
                log::info!(
                    "reco-obs: pipeline initialized ({}x{} output, {}x{} input, GPU: {})",
                    self.output_width,
                    self.output_height,
                    self.input_width,
                    self.input_height,
                    pipeline.gpu_name(),
                );
                // Build the RGBA readback helper alongside the pipeline.
                // `RgbaReadback` owns the triple-buffered staging + row
                // padding strip that used to live inline in this file.
                match RgbaReadback::new(pipeline.gpu(), self.output_width, self.output_height) {
                    Ok(readback) => {
                        self.readback = Some(readback);
                    }
                    Err(e) => {
                        log::error!("reco-obs: failed to create RGBA readback: {e}");
                        self.pipeline = None;
                        self.readback = None;
                        return;
                    }
                }
                self.pipeline = Some(pipeline);
                // Pre-allocate the owned buffer that `video_render` reads.
                let buf_size = (self.output_width * self.output_height * 4) as usize;
                self.rgba_buffer.resize(buf_size, 0);
            }
            Err(e) => {
                log::error!("reco-obs: failed to create pipeline: {e}");
                self.pipeline = None;
                self.readback = None;
            }
        }
    }

    /// Load calibration from the config file path.
    fn load_calibration(&mut self) {
        if self.config_path.is_empty() {
            self.calibration = None;
            self.pipeline = None;
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
                self.pipeline = None;
            }
        }
    }

    /// Perform a render and readback cycle.
    ///
    /// Currently renders a blank frame (all-zero YUV) since we don't have
    /// live camera input wired up yet. The real integration will need an
    /// `obs_source_t` reference to pull video frames from upstream OBS
    /// sources (e.g., two V4L2 camera inputs).
    fn render_and_readback(&mut self) {
        let pipeline = match &self.pipeline {
            Some(p) => p,
            None => return,
        };
        let readback = match &mut self.readback {
            Some(r) => r,
            None => return,
        };
        let gpu = match &self.gpu {
            Some(g) => g,
            None => return,
        };

        // FRICTION: StitchPipeline expects raw YUV plane data (&[u8]) but
        // an OBS plugin consumer would naturally have obs_source_frame
        // pointers or OBS texture handles. There's no way to feed OBS's
        // frame data into the pipeline without first extracting the raw
        // plane bytes and sizes - which requires knowing the OBS video
        // format and stride layout. A higher-level API that accepts
        // width/height/stride/pointers (like a "RawFrameView") would
        // reduce this impedance mismatch.

        // TODO: Wire up actual camera frame data from OBS source inputs.
        // For now, render a test pattern (zero YUV = green in BT.601).
        let y_size = (self.input_width * self.input_height) as usize;
        let uv_size = y_size / 4;
        let y_data = vec![0u8; y_size];
        let u_data = vec![128u8; uv_size];
        let v_data = vec![128u8; uv_size];

        let left = YuvPlanes {
            y: &y_data,
            u: &u_data,
            v: &v_data,
        };
        let right = YuvPlanes {
            y: &y_data,
            u: &u_data,
            v: &v_data,
        };

        let yaw = (self.yaw_degrees as f32).to_radians();
        let pitch = (self.pitch_degrees as f32).to_radians();

        // Render + triple-buffered RGBA readback via the shared
        // `RgbaReadback` helper in reco-core. Returns `None` on the first
        // two calls (pipeline warmup) and tightly-packed RGBA thereafter,
        // so the wgpu 256-byte row padding is stripped for us.
        let cmd_buf = match pipeline.render_to_target(&left, &right, yaw, pitch) {
            Ok(cmd) => cmd,
            Err(e) => {
                log::error!("reco-obs: render failed: {e}");
                return;
            }
        };

        match readback.readback(gpu, pipeline.render_target(), cmd_buf) {
            Ok(Some(rgba)) => {
                self.rgba_buffer.copy_from_slice(rgba);
                self.frame_ready.store(true, Ordering::Release);
            }
            Ok(None) => {
                // Pipeline warmup; next tick will have data.
            }
            Err(e) => {
                log::error!("reco-obs: RGBA readback failed: {e}");
            }
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
            self.obs_texture = unsafe {
                ffi::gs_texture_create(
                    self.output_width,
                    self.output_height,
                    ffi::gs_color_format::GS_RGBA,
                    1,
                    ptr::null(),
                    0,
                )
            };
            if self.obs_texture.is_null() {
                log::error!("reco-obs: failed to create OBS texture");
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
    }
}

/// `get_properties`: define the settings UI.
unsafe extern "C" fn source_get_properties(_data: *mut c_void) -> *mut ffi::obs_properties_t {
    unsafe {
        let props = ffi::obs_properties_create();

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

    if dims_changed || config_changed {
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

    if src.pipeline.is_none() {
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
    }

    // Draw the texture using OBS's default effect.
    unsafe {
        let effect = ffi::obs_get_base_effect(ffi::obs_base_effect::OBS_EFFECT_DEFAULT);
        let param = ffi::gs_effect_get_param_by_name(effect, c"image".as_ptr());
        ffi::gs_effect_set_texture(param, src.obs_texture);

        while ffi::gs_effect_loop(effect, c"Draw".as_ptr()) {
            ffi::gs_draw_sprite(src.obs_texture, 0, src.output_width, src.output_height);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
