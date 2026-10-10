//! Owned OBS references and graphics resources at the plugin boundary.

use crate::ffi;
use std::ffi::CString;
use std::ptr;

/// One selected async source, with periodic retry when it has not been created yet.
#[derive(Default)]
pub(super) struct SourceInput {
    pub pointer: *mut ffi::obs_source_t,
    name: String,
    missing: bool,
}

// OBS reference counting and async-frame acquisition support callers on different threads. The engine mutex serializes use of this owned reference.
unsafe impl Send for SourceInput {}

impl SourceInput {
    pub fn is_null(&self) -> bool {
        self.pointer.is_null()
    }

    /// Apply a source selection, returning whether the selection or resolved input changed.
    pub fn select(&mut self, name: &str) -> bool {
        if self.name == name && !self.is_null() {
            return false;
        }
        let changed = self.name != name;
        self.clear();
        self.name = name.to_owned();
        self.missing = !name.is_empty();
        self.retry();
        if self.missing {
            log::warn!("reco-obs: source '{name}' is not created yet; waiting");
        }
        changed || !self.is_null()
    }

    /// Retry absent sources. Unsupported source types are not retried every tick.
    pub fn retry(&mut self) {
        if !self.missing {
            return;
        }
        self.missing = false;
        let Ok(name) = CString::new(self.name.as_str()) else {
            log::warn!("reco-obs: source name contains NUL");
            return;
        };
        let source = unsafe { ffi::obs_get_source_by_name(name.as_ptr()) };
        if source.is_null() {
            self.missing = true;
            return;
        }
        let flags = unsafe { ffi::obs_source_get_output_flags(source) };
        if !accepts_source(flags) {
            unsafe {
                ffi::obs_source_release(source);
            }
            log::warn!(
                "reco-obs: source '{}' is not async video and cannot be used as camera input",
                self.name
            );
            return;
        }
        // Keep the upstream decoder running even when its own scene item is hidden.
        unsafe {
            ffi::obs_source_inc_showing(source);
            ffi::obs_source_inc_active(source);
        }
        self.pointer = source;
        log::info!("reco-obs: selected async source '{}'", self.name);
    }

    fn clear(&mut self) {
        let pointer = std::mem::replace(&mut self.pointer, ptr::null_mut());
        if !pointer.is_null() {
            unsafe {
                ffi::obs_source_dec_active(pointer);
                ffi::obs_source_dec_showing(pointer);
                ffi::obs_source_release(pointer);
            }
        }
    }
}

impl Drop for SourceInput {
    fn drop(&mut self) {
        self.clear();
    }
}

/// The picker and the acquisition boundary accept the same source capabilities.
pub(super) fn accepts_source(flags: u32) -> bool {
    flags & ffi::OBS_SOURCE_ASYNC_VIDEO == ffi::OBS_SOURCE_ASYNC_VIDEO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_async_video_sources_can_be_owned_as_inputs() {
        for flags in [
            0,
            ffi::OBS_SOURCE_VIDEO,
            ffi::OBS_SOURCE_AUDIO,
            ffi::OBS_SOURCE_ASYNC,
            ffi::OBS_SOURCE_VIDEO | ffi::OBS_SOURCE_INTERACTION,
        ] {
            assert!(!accepts_source(flags));
        }
        assert!(accepts_source(ffi::OBS_SOURCE_ASYNC_VIDEO));
        assert!(accepts_source(
            ffi::OBS_SOURCE_ASYNC_VIDEO | ffi::OBS_SOURCE_AUDIO
        ));
    }

    #[test]
    #[ignore = "initializes a standalone libobs host; run in isolation with a desktop session"]
    fn native_selection_rejects_self_and_releases_late_sources() {
        struct Host;
        unsafe extern "C" fn barrier(_: *mut std::ffi::c_void) {}
        impl Drop for Host {
            fn drop(&mut self) {
                unsafe {
                    ffi::obs_queue_task(
                        ffi::obs_task_type_OBS_TASK_DESTROY,
                        Some(barrier),
                        ptr::null_mut(),
                        true,
                    );
                    ffi::obs_shutdown();
                }
            }
        }
        struct Source(*mut ffi::obs_source_t);
        impl Drop for Source {
            fn drop(&mut self) {
                unsafe {
                    ffi::obs_source_release(self.0);
                }
            }
        }
        unsafe {
            assert!(ffi::obs_startup(
                c"en-US".as_ptr(),
                ptr::null(),
                ptr::null_mut()
            ));
            let _host = Host;
            let info = crate::source::source_info();
            ffi::obs_register_source_s(&info, std::mem::size_of_val(&info));
            let mut camera_info = crate::source::source_info();
            camera_info.id = c"reco_test_async".as_ptr();
            camera_info.output_flags = ffi::OBS_SOURCE_ASYNC_VIDEO;
            camera_info.video_tick = None;
            camera_info.video_render = None;
            ffi::obs_register_source_s(&camera_info, std::mem::size_of_val(&camera_info));
            let parent = Source(ffi::obs_source_create(
                c"reco_stitcher".as_ptr(),
                c"native-parent".as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            ));
            assert!(!parent.0.is_null());
            let settings = ffi::obs_source_get_settings(parent.0);
            ffi::obs_data_set_string(
                settings,
                c"right_source".as_ptr(),
                c"native-parent".as_ptr(),
            );
            info.update.unwrap()(ffi::obs_obj_get_data(parent.0.cast()), settings);
            ffi::obs_data_release(settings);

            let mut input = SourceInput::default();
            input.select("native-late");
            assert!(input.is_null());
            assert!(input.missing);
            let camera = Source(ffi::obs_source_create(
                c"reco_test_async".as_ptr(),
                c"native-late".as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            ));
            assert!(!camera.0.is_null());
            input.retry();
            assert_eq!(input.pointer, camera.0);
            assert!(!input.missing);

            let properties = ffi::obs_source_properties(parent.0);
            let list = ffi::obs_properties_get(properties, c"left_source".as_ptr());
            let names: Vec<_> = (0..ffi::obs_property_list_item_count(list))
                .map(|index| {
                    std::ffi::CStr::from_ptr(ffi::obs_property_list_item_string(list, index))
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            ffi::obs_properties_destroy(properties);
            assert!(names.iter().any(|name| name == "native-late"));
            assert!(!names.iter().any(|name| name == "native-parent"));
            drop(camera);
            let retained = Source(ffi::obs_get_source_by_name(c"native-late".as_ptr()));
            assert!(!retained.0.is_null());
            drop(retained);
            input.select("");
            let released = Source(ffi::obs_get_source_by_name(c"native-late".as_ptr()));
            assert!(released.0.is_null());
            drop(parent);
            let released = Source(ffi::obs_get_source_by_name(c"native-parent".as_ptr()));
            assert!(released.0.is_null());
        }
    }
}

/// Texture access requires OBS graphics ownership; destruction obtains it automatically.
#[derive(Default)]
pub(super) struct Texture {
    pub pointer: *mut ffi::gs_texture_t,
}

// The resource stays in OBS's graphics context. The presentation mutex serializes its handle; texture operations require that graphics context.
unsafe impl Send for Texture {}

impl Texture {
    pub fn is_null(&self) -> bool {
        self.pointer.is_null()
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe {
                ffi::obs_enter_graphics();
                ffi::gs_texture_destroy(self.pointer);
                ffi::obs_leave_graphics();
            }
        }
    }
}
