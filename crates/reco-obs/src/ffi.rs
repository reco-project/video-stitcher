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

/// Texture creation flag: auto-generate mipmaps (from `libobs/graphics/graphics.h`).
#[allow(dead_code)]
pub const GS_BUILD_MIPMAPS: u32 = 1 << 0;
/// Texture creation flag: dynamic (CPU-writable via `gs_texture_map` /
/// `gs_texture_set_image`). Required when uploading frames from the plugin.
pub const GS_DYNAMIC: u32 = 1 << 1;
/// Texture creation flag: render target (GPU-writable).
#[allow(dead_code)]
pub const GS_RENDER_TARGET: u32 = 1 << 2;

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
/// Source delivers frames asynchronously via `obs_source_output_video`.
/// When OR'd with `OBS_SOURCE_VIDEO`, forms `OBS_SOURCE_ASYNC_VIDEO` -
/// the only mode reco-obs can currently consume via `obs_source_get_frame`.
pub const OBS_SOURCE_ASYNC: u32 = 1 << 2;
/// Source uses custom draw (no default effect passed to `video_render`).
#[allow(dead_code)]
pub const OBS_SOURCE_CUSTOM_DRAW: u32 = 1 << 3;
/// Source handles interaction callbacks (mouse / key / focus). OBS will
/// only deliver mouse_click / mouse_move / mouse_wheel events to a source
/// that declares this flag.
pub const OBS_SOURCE_INTERACTION: u32 = 1 << 5;

/// OBS mouse button enum (from `libobs/obs-interaction.h`).
#[repr(i32)]
#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum obs_mouse_button_type {
    MOUSE_LEFT = 0,
    MOUSE_MIDDLE = 1,
    MOUSE_RIGHT = 2,
}

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

/// Maximum number of planes in an async video frame (from libobs `media-io-defs.h`).
pub const MAX_AV_PLANES: usize = 8;

/// OBS video pixel format enum (subset - see `libobs/media-io/video-io.h`).
///
/// Only the variants reco-obs inspects are represented. Other values are
/// delivered by OBS but Tier 1 ingestion rejects them with a warning.
#[repr(i32)]
#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum video_format {
    /// Unset / unknown format.
    VIDEO_FORMAT_NONE = 0,
    /// Planar 4:2:0, three planes (Y, U, V). Matches our tight `YuvPlanes`.
    VIDEO_FORMAT_I420 = 1,
    /// Planar 4:2:0, two planes (Y, interleaved UV). Tier 2 target.
    VIDEO_FORMAT_NV12 = 2,
    /// Packed 4:2:2 - YVYU.
    VIDEO_FORMAT_YVYU = 3,
    /// Packed 4:2:2 - YUY2.
    VIDEO_FORMAT_YUY2 = 4,
    /// Packed 4:2:2 - UYVY.
    VIDEO_FORMAT_UYVY = 5,
    /// Packed RGBA.
    VIDEO_FORMAT_RGBA = 6,
    /// Packed BGRA.
    VIDEO_FORMAT_BGRA = 7,
    /// Packed BGRX.
    VIDEO_FORMAT_BGRX = 8,
    /// 8-bit grayscale.
    VIDEO_FORMAT_Y800 = 9,
    /// Planar 4:4:4.
    VIDEO_FORMAT_I444 = 10,
}

/// Async video frame delivered to async video sources / consumed via
/// [`obs_source_get_frame`]. Layout mirrors `struct obs_source_frame`
/// in `libobs/obs.h` for OBS 30.x.
///
/// Fields after `trc` are used internally by libobs; treat them as
/// opaque and don't rely on their offsets.
#[repr(C)]
pub struct obs_source_frame {
    /// Plane pointers. Only the first few are populated depending on `format`.
    pub data: [*mut u8; MAX_AV_PLANES],
    /// Bytes per row per plane (may include padding).
    pub linesize: [u32; MAX_AV_PLANES],
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in nanoseconds.
    pub timestamp: u64,
    /// Pixel format.
    pub format: video_format,
    /// 4x4 color matrix (column-major) for YUV->RGB conversion.
    pub color_matrix: [f32; 16],
    /// Whether the source delivers full-range YUV (vs. limited 16-235).
    pub full_range: bool,
    /// HDR max luminance hint.
    pub max_luminance: u16,
    /// Min of each color channel (for range shaping).
    pub color_range_min: [f32; 3],
    /// Max of each color channel.
    pub color_range_max: [f32; 3],
    /// If true, the frame is Y-flipped.
    pub flip: bool,
    /// Render flags (bit 0 = is linear alpha).
    pub flags: u8,
    /// Transfer characteristic (enum video_trc in libobs).
    pub trc: u8,
    /// Internal: refcount (don't touch).
    pub refs: std::sync::atomic::AtomicI32,
    /// Internal: "already rendered" flag.
    pub prev_frame: bool,
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

/// Source enumeration callback type (for `enum_active_sources` on a source).
#[allow(non_camel_case_types)]
pub type obs_source_enum_proc_t = Option<
    unsafe extern "C" fn(parent: *mut obs_source_t, child: *mut obs_source_t, param: *mut c_void),
>;

/// Global source enumeration callback (used by `obs_enum_sources`).
///
/// Return `true` to continue enumeration, `false` to stop.
#[allow(non_camel_case_types)]
pub type obs_enum_sources_cb =
    Option<unsafe extern "C" fn(param: *mut c_void, source: *mut obs_source_t) -> bool>;

/// UI dropdown kind (from `libobs/obs-properties.h`). We use
/// `OBS_COMBO_TYPE_LIST` for non-editable source pickers.
#[repr(i32)]
#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum obs_combo_type {
    /// Invalid / unset.
    OBS_COMBO_TYPE_INVALID = 0,
    /// User-editable free-form entry.
    OBS_COMBO_TYPE_EDITABLE = 1,
    /// Fixed list (what we want for source pickers).
    OBS_COMBO_TYPE_LIST = 2,
    /// Radio-button group.
    OBS_COMBO_TYPE_RADIO = 3,
}

/// Dropdown value type (we only use `STRING` for source names).
#[repr(i32)]
#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum obs_combo_format {
    /// Invalid / unset.
    OBS_COMBO_FORMAT_INVALID = 0,
    /// Integer values.
    OBS_COMBO_FORMAT_INT = 1,
    /// Float values.
    OBS_COMBO_FORMAT_FLOAT = 2,
    /// String values (what we use for source names).
    OBS_COMBO_FORMAT_STRING = 3,
    /// Bool values.
    OBS_COMBO_FORMAT_BOOL = 4,
}

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
    pub fn obs_data_get_double(data: *mut obs_data_t, name: *const c_char) -> f64;
    pub fn obs_data_set_default_double(data: *mut obs_data_t, name: *const c_char, val: f64);
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

    /// High-level draw helper that handles the effect loop internally.
    ///
    /// Use this instead of starting a new `gs_effect_loop` inside a source's
    /// `video_render` callback - OBS already has its outer effect active
    /// when it calls us, so nesting with `gs_effect_loop` triggers
    /// "effect is already active" and silently drops the draw.
    pub fn obs_source_draw(
        image: *mut gs_texture_t,
        x: c_int,
        y: c_int,
        cx: u32,
        cy: u32,
        flip: bool,
    );

    // Effects
    pub fn obs_get_base_effect(effect: obs_base_effect) -> *mut gs_effect_t;
    pub fn gs_effect_get_param_by_name(
        effect: *const gs_effect_t,
        name: *const c_char,
    ) -> *mut gs_eparam_t;
    pub fn gs_effect_set_texture(param: *mut gs_eparam_t, val: *mut gs_texture_t);
    pub fn gs_effect_loop(effect: *mut gs_effect_t, name: *const c_char) -> bool;

    // Source enumeration + lookup (Tier 1 frame ingestion)
    pub fn obs_enum_sources(enum_proc: obs_enum_sources_cb, param: *mut c_void);
    pub fn obs_get_source_by_name(name: *const c_char) -> *mut obs_source_t;
    pub fn obs_source_release(source: *mut obs_source_t);
    pub fn obs_source_get_name(source: *const obs_source_t) -> *const c_char;
    /// Returns the bitmask of OBS_SOURCE_* flags declared by the source.
    /// Bit 0 = video, bit 1 = audio, bit 2 = async, bit 3 = custom_draw, etc.
    pub fn obs_source_get_output_flags(source: *const obs_source_t) -> u32;

    // Source activation (keeps Media Source decoders running even when the
    // upstream source isn't rendered in the scene directly). We're using
    // the source's async frames via obs_source_get_frame, but OBS doesn't
    // track that as "showing", so without inc_showing it may deactivate
    // the decoder and stall playback.
    //
    // `inc_showing` alone is insufficient for Media Source (ffmpeg_source):
    // its decode thread also checks active_state, so we pair with
    // `inc_active` to hold the upstream firmly.
    pub fn obs_source_inc_showing(source: *mut obs_source_t);
    pub fn obs_source_dec_showing(source: *mut obs_source_t);
    pub fn obs_source_inc_active(source: *mut obs_source_t);
    pub fn obs_source_dec_active(source: *mut obs_source_t);

    // Dropdown property (for source pickers)
    pub fn obs_properties_add_list(
        props: *mut obs_properties_t,
        name: *const c_char,
        description: *const c_char,
        list_type: obs_combo_type,
        format: obs_combo_format,
    ) -> *mut obs_property_t;
    pub fn obs_property_list_add_string(
        p: *mut obs_property_t,
        name: *const c_char,
        val: *const c_char,
    ) -> usize;

    // Pull-based async video frame access.
    //
    // Returns the source's current async frame (ref-counted). Must be
    // paired with obs_source_release_frame to avoid leaking. The frame
    // contents are valid only between these two calls.
    pub fn obs_source_get_frame(source: *mut obs_source_t) -> *mut obs_source_frame;
    pub fn obs_source_release_frame(source: *mut obs_source_t, frame: *mut obs_source_frame);
}
