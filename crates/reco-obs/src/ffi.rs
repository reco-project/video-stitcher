//! Raw C FFI bindings for the OBS Studio plugin API.
//!
//! These are hand-written bindings covering only the subset of libobs we need.
//! They target OBS Studio 30+ (the `obs_source_info` struct layout must match
//! the version of OBS this plugin is loaded into).
//!
//! # Safety
//!
//! All types here are `repr(C)` to match OBS's ABI. Function pointers use
//! `Option<unsafe extern "C" fn(...)>` so unset callbacks are null pointers.

use std::os::raw::{c_char, c_float, c_int, c_void};

// ---------------------------------------------------------------------------
// Opaque OBS types (we only ever hold pointers to these)
// ---------------------------------------------------------------------------

/// Opaque OBS module handle.
#[repr(C)]
pub struct obs_module_t {
    _opaque: [u8; 0],
}

/// Opaque OBS source instance.
#[repr(C)]
pub struct obs_source_t {
    _opaque: [u8; 0],
}

/// Opaque JSON-like settings object.
#[repr(C)]
pub struct obs_data_t {
    _opaque: [u8; 0],
}

/// Opaque properties list (UI definition).
#[repr(C)]
pub struct obs_properties_t {
    _opaque: [u8; 0],
}

/// Opaque single property.
#[repr(C)]
pub struct obs_property_t {
    _opaque: [u8; 0],
}

/// Opaque GPU texture (OBS graphics subsystem).
#[repr(C)]
pub struct gs_texture_t {
    _opaque: [u8; 0],
}

/// Opaque GPU effect (shader program).
#[repr(C)]
pub struct gs_effect_t {
    _opaque: [u8; 0],
}

/// Opaque effect parameter handle.
#[repr(C)]
pub struct gs_eparam_t {
    _opaque: [u8; 0],
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// OBS source type.
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum obs_source_type {
    OBS_SOURCE_TYPE_INPUT = 0,
    OBS_SOURCE_TYPE_FILTER = 1,
    OBS_SOURCE_TYPE_TRANSITION = 2,
    OBS_SOURCE_TYPE_SCENE = 3,
}

/// OBS icon type (displayed in the source list).
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum obs_icon_type {
    OBS_ICON_TYPE_UNKNOWN = 0,
    OBS_ICON_TYPE_IMAGE = 1,
    OBS_ICON_TYPE_COLOR = 2,
    OBS_ICON_TYPE_SLIDESHOW = 3,
    OBS_ICON_TYPE_AUDIO_INPUT = 4,
    OBS_ICON_TYPE_AUDIO_OUTPUT = 5,
    OBS_ICON_TYPE_DESKTOP_CAPTURE = 6,
    OBS_ICON_TYPE_WINDOW_CAPTURE = 7,
    OBS_ICON_TYPE_GAME_CAPTURE = 8,
    OBS_ICON_TYPE_CAMERA = 9,
    OBS_ICON_TYPE_TEXT = 10,
    OBS_ICON_TYPE_MEDIA = 11,
    OBS_ICON_TYPE_BROWSER = 12,
    OBS_ICON_TYPE_CUSTOM = 13,
    OBS_ICON_TYPE_PROCESS_AUDIO_OUTPUT = 14,
    OBS_ICON_TYPE_INPUT = 15,
}

/// OBS media state (for sources that support media controls).
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum obs_media_state {
    OBS_MEDIA_STATE_NONE = 0,
    OBS_MEDIA_STATE_PLAYING = 1,
    OBS_MEDIA_STATE_OPENING = 2,
    OBS_MEDIA_STATE_BUFFERING = 3,
    OBS_MEDIA_STATE_PAUSED = 4,
    OBS_MEDIA_STATE_STOPPED = 5,
    OBS_MEDIA_STATE_ENDED = 6,
    OBS_MEDIA_STATE_ERROR = 7,
}

/// OBS color space.
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum gs_color_space {
    GS_CS_SRGB = 0,
    GS_CS_SRGB_16F = 1,
    GS_CS_709_EXTENDED = 2,
    GS_CS_709_SCRGB = 3,
}

/// OBS graphics color format.
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum gs_color_format {
    GS_UNKNOWN = 0,
    GS_A8 = 1,
    GS_R8 = 2,
    GS_RGBA = 3,
    GS_BGRX = 4,
    GS_BGRA = 5,
    GS_R10G10B10A2 = 6,
    GS_RGBA16 = 7,
    GS_R16 = 8,
    GS_RGBA16F = 9,
    GS_RGBA32F = 10,
    GS_RG16F = 11,
    GS_RG32F = 12,
    GS_R16F = 13,
    GS_R32F = 14,
    GS_DXT1 = 15,
    GS_DXT3 = 16,
    GS_DXT5 = 17,
    GS_R8G8 = 18,
    GS_RGBA_UNORM = 19,
    GS_BGRX_UNORM = 20,
    GS_BGRA_UNORM = 21,
    GS_RG16 = 22,
}

/// Path type for `obs_properties_add_path`.
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum obs_path_type {
    OBS_PATH_FILE = 0,
    OBS_PATH_FILE_SAVE = 1,
    OBS_PATH_DIRECTORY = 2,
}

// ---------------------------------------------------------------------------
// Output flag constants
// ---------------------------------------------------------------------------

/// Source provides video output (synchronous rendering via `video_render`).
pub const OBS_SOURCE_VIDEO: u32 = 1;
/// Source uses custom draw (no default effect passed to `video_render`).
#[allow(dead_code)]
pub const OBS_SOURCE_CUSTOM_DRAW: u32 = 1 << 3;

// ---------------------------------------------------------------------------
// OBS base effect enum
// ---------------------------------------------------------------------------

/// Index for `obs_get_base_effect`.
#[repr(C)]
#[allow(non_camel_case_types, dead_code)]
pub enum obs_base_effect {
    OBS_EFFECT_DEFAULT = 0,
    OBS_EFFECT_DEFAULT_RECT = 1,
    OBS_EFFECT_OPAQUE = 2,
    OBS_EFFECT_SOLID = 3,
    OBS_EFFECT_BICUBIC = 4,
    OBS_EFFECT_LANCZOS = 5,
    OBS_EFFECT_BILINEAR_LOWRES = 6,
    OBS_EFFECT_PREMULTIPLIED_ALPHA = 7,
    OBS_EFFECT_REPEAT = 8,
    OBS_EFFECT_AREA = 9,
}

// ---------------------------------------------------------------------------
// obs_source_info - the central plugin registration struct
// ---------------------------------------------------------------------------

/// Mouse event data (unused by us but needed for struct layout).
#[repr(C)]
pub struct obs_mouse_event {
    pub modifiers: u32,
    pub x: i32,
    pub y: i32,
}

/// Key event data (unused by us but needed for struct layout).
#[repr(C)]
pub struct obs_key_event {
    pub modifiers: u32,
    pub text: *const c_char,
    pub native_modifiers: u32,
    pub native_scancode: u32,
    pub native_vkey: u32,
}

/// Audio mix data (opaque placeholder).
#[repr(C)]
pub struct obs_source_audio_mix {
    _opaque: [u8; 0],
}

/// Audio output data (opaque placeholder).
#[repr(C)]
pub struct audio_output_data {
    _opaque: [u8; 0],
}

/// Async video frame (opaque placeholder).
#[repr(C)]
pub struct obs_source_frame {
    _opaque: [u8; 0],
}

/// Async audio data (opaque placeholder).
#[repr(C)]
pub struct obs_audio_data {
    _opaque: [u8; 0],
}

/// Missing files result (opaque placeholder).
#[repr(C)]
pub struct obs_missing_files_t {
    _opaque: [u8; 0],
}

/// Source enumeration callback type.
#[allow(non_camel_case_types)]
pub type obs_source_enum_proc_t = Option<
    unsafe extern "C" fn(parent: *mut obs_source_t, child: *mut obs_source_t, param: *mut c_void),
>;

/// The source info registration struct.
///
/// This struct layout must match the OBS version this plugin targets.
/// Fields are in declaration order from `obs-source.h` (OBS 30.x).
///
/// We use `Option<unsafe extern "C" fn(...)>` for all function pointers
/// so unimplemented callbacks are represented as null pointers.
#[repr(C)]
pub struct obs_source_info {
    pub id: *const c_char,
    pub r#type: obs_source_type,
    pub output_flags: u32,
    pub get_name: Option<unsafe extern "C" fn(type_data: *mut c_void) -> *const c_char>,
    pub create: Option<
        unsafe extern "C" fn(settings: *mut obs_data_t, source: *mut obs_source_t) -> *mut c_void,
    >,
    pub destroy: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub get_width: Option<unsafe extern "C" fn(data: *mut c_void) -> u32>,
    pub get_height: Option<unsafe extern "C" fn(data: *mut c_void) -> u32>,
    pub get_defaults: Option<unsafe extern "C" fn(settings: *mut obs_data_t)>,
    pub get_properties: Option<unsafe extern "C" fn(data: *mut c_void) -> *mut obs_properties_t>,
    pub update: Option<unsafe extern "C" fn(data: *mut c_void, settings: *mut obs_data_t)>,
    pub activate: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub deactivate: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub show: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub hide: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub video_tick: Option<unsafe extern "C" fn(data: *mut c_void, seconds: c_float)>,
    pub video_render: Option<unsafe extern "C" fn(data: *mut c_void, effect: *mut gs_effect_t)>,
    pub filter_video: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            frame: *mut obs_source_frame,
        ) -> *mut obs_source_frame,
    >,
    pub filter_audio: Option<
        unsafe extern "C" fn(data: *mut c_void, audio: *mut obs_audio_data) -> *mut obs_audio_data,
    >,
    pub enum_active_sources: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            enum_callback: obs_source_enum_proc_t,
            param: *mut c_void,
        ),
    >,
    pub save: Option<unsafe extern "C" fn(data: *mut c_void, settings: *mut obs_data_t)>,
    pub load: Option<unsafe extern "C" fn(data: *mut c_void, settings: *mut obs_data_t)>,
    pub mouse_click: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            event: *const obs_mouse_event,
            r#type: i32,
            mouse_up: bool,
            click_count: u32,
        ),
    >,
    pub mouse_move: Option<
        unsafe extern "C" fn(data: *mut c_void, event: *const obs_mouse_event, mouse_leave: bool),
    >,
    pub mouse_wheel: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            event: *const obs_mouse_event,
            x_delta: c_int,
            y_delta: c_int,
        ),
    >,
    pub focus: Option<unsafe extern "C" fn(data: *mut c_void, focus: bool)>,
    pub key_click:
        Option<unsafe extern "C" fn(data: *mut c_void, event: *const obs_key_event, key_up: bool)>,
    pub filter_remove: Option<unsafe extern "C" fn(data: *mut c_void, source: *mut obs_source_t)>,
    pub type_data: *mut c_void,
    pub free_type_data: Option<unsafe extern "C" fn(type_data: *mut c_void)>,
    pub audio_render: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            ts_out: *mut u64,
            audio_output: *mut obs_source_audio_mix,
            mixers: u32,
            channels: usize,
            sample_rate: usize,
        ) -> bool,
    >,
    pub enum_all_sources: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            enum_callback: obs_source_enum_proc_t,
            param: *mut c_void,
        ),
    >,
    pub transition_start: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub transition_stop: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub get_defaults2:
        Option<unsafe extern "C" fn(type_data: *mut c_void, settings: *mut obs_data_t)>,
    pub get_properties2: Option<
        unsafe extern "C" fn(data: *mut c_void, type_data: *mut c_void) -> *mut obs_properties_t,
    >,
    pub audio_mix: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            ts_out: *mut u64,
            audio_output: *mut audio_output_data,
            channels: usize,
            sample_rate: usize,
        ) -> bool,
    >,
    pub icon_type: obs_icon_type,
    pub media_play_pause: Option<unsafe extern "C" fn(data: *mut c_void, pause: bool)>,
    pub media_restart: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub media_stop: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub media_next: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub media_previous: Option<unsafe extern "C" fn(data: *mut c_void)>,
    pub media_get_duration: Option<unsafe extern "C" fn(data: *mut c_void) -> i64>,
    pub media_get_time: Option<unsafe extern "C" fn(data: *mut c_void) -> i64>,
    pub media_set_time: Option<unsafe extern "C" fn(data: *mut c_void, milliseconds: i64)>,
    pub media_get_state: Option<unsafe extern "C" fn(data: *mut c_void) -> obs_media_state>,
    pub version: u32,
    pub unversioned_id: *const c_char,
    pub missing_files: Option<unsafe extern "C" fn(data: *mut c_void) -> *mut obs_missing_files_t>,
    pub video_get_color_space: Option<
        unsafe extern "C" fn(
            data: *mut c_void,
            count: usize,
            preferred_spaces: *const gs_color_space,
        ) -> gs_color_space,
    >,
    pub filter_add: Option<unsafe extern "C" fn(data: *mut c_void, source: *mut obs_source_t)>,
}

// Safety: obs_source_info is only ever used from the OBS main/render threads
// (never sent across threads by us), but Rust requires Send + Sync for statics.
// The struct contains only raw pointers and function pointers - all thread-safe
// as long as OBS calls us on the correct threads (which it guarantees).
unsafe impl Send for obs_source_info {}
unsafe impl Sync for obs_source_info {}

// ---------------------------------------------------------------------------
// Extern functions we call into OBS
// ---------------------------------------------------------------------------

#[allow(dead_code)]
unsafe extern "C" {
    // Module registration
    pub fn obs_register_source_s(info: *const obs_source_info, size: usize);

    // Settings (obs_data_t)
    pub fn obs_data_get_string(data: *mut obs_data_t, name: *const c_char) -> *const c_char;
    pub fn obs_data_get_int(data: *mut obs_data_t, name: *const c_char) -> i64;
    pub fn obs_data_set_default_string(
        data: *mut obs_data_t,
        name: *const c_char,
        val: *const c_char,
    );
    pub fn obs_data_set_default_int(data: *mut obs_data_t, name: *const c_char, val: i64);

    // Properties (UI)
    pub fn obs_properties_create() -> *mut obs_properties_t;
    pub fn obs_properties_add_path(
        props: *mut obs_properties_t,
        name: *const c_char,
        description: *const c_char,
        path_type: obs_path_type,
        filter: *const c_char,
        default_path: *const c_char,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_int(
        props: *mut obs_properties_t,
        name: *const c_char,
        description: *const c_char,
        min: c_int,
        max: c_int,
        step: c_int,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_float(
        props: *mut obs_properties_t,
        name: *const c_char,
        description: *const c_char,
        min: f64,
        max: f64,
        step: f64,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_bool(
        props: *mut obs_properties_t,
        name: *const c_char,
        description: *const c_char,
    ) -> *mut obs_property_t;

    // Graphics (gs_*) - called only from the OBS render thread
    pub fn obs_enter_graphics();
    pub fn obs_leave_graphics();
    pub fn gs_texture_create(
        width: u32,
        height: u32,
        color_format: gs_color_format,
        levels: u32,
        data: *const *const u8,
        flags: u32,
    ) -> *mut gs_texture_t;
    pub fn gs_texture_destroy(tex: *mut gs_texture_t);
    pub fn gs_texture_set_image(
        tex: *mut gs_texture_t,
        data: *const u8,
        linesize: u32,
        invert: bool,
    );
    pub fn gs_draw_sprite(tex: *mut gs_texture_t, flip: u32, width: u32, height: u32);

    // Effects
    pub fn obs_get_base_effect(effect: obs_base_effect) -> *mut gs_effect_t;
    pub fn gs_effect_get_param_by_name(
        effect: *const gs_effect_t,
        name: *const c_char,
    ) -> *mut gs_eparam_t;
    pub fn gs_effect_set_texture(param: *mut gs_eparam_t, val: *mut gs_texture_t);
    pub fn gs_effect_loop(effect: *mut gs_effect_t, name: *const c_char) -> bool;
}
