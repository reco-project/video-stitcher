//! NvBufSurfTransform-based NVMM detection preprocessing for Jetson.
//!
//! Replaces the broken EGL/CUDA interop path with NVIDIA's hardware-accelerated
//! transform API. Converts NVMM NV12 camera frames to CUDA RGBA u8 at model
//! resolution with letterboxing in a single GPU operation.
//!
//! Flow: NVMM NV12 → NvBufSurfTransform(GPU, NV12→RGBA, resize, letterbox)
//!       → CUDA RGBA u8 1280×1280 → normalize kernel → TRT f32 CHW

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::interop::cuda::CUdeviceptr;

// Values verified via offsetof()/printf on Jetson Orin Nano, JetPack 6.

/// NvBufSurfTransformCompute_GPU = 1 (uses CUDA kernels internally).
const NVBUF_TRANSFORM_COMPUTE_GPU: i32 = 1;

/// NVBUF_MEM_CUDA_DEVICE = 2.
const NVBUF_MEM_CUDA_DEVICE: i32 = 2;

/// NVBUF_COLOR_FORMAT_RGBA = 19 (NOT 4 - that's BGRA).
const NVBUF_COLOR_FORMAT_RGBA: i32 = 19;

/// NVBUF_LAYOUT_PITCH = 0.
const NVBUF_LAYOUT_PITCH: i32 = 0;

/// NvBufSurfTransformInter_Bilinear = 1.
const NVBUF_TRANSFORM_FILTER_BILINEAR: i32 = 1;

/// NVBUFSURF_TRANSFORM_CROP_DST = 1 << 1 = 2.
const NVBUF_TRANSFORM_CROP_DST: u32 = 2;
/// NVBUFSURF_TRANSFORM_FILTER = 1 << 2 = 4.
const NVBUF_TRANSFORM_FILTER: u32 = 4;

/// NvBufSurface creation parameters (sizeof=32, verified on aarch64).
#[repr(C)]
#[derive(Clone)]
struct NvBufSurfaceCreateParams {
    gpu_id: u32,
    width: u32,
    height: u32,
    size: u32,
    is_contiguous: i32,
    color_format: i32,
    layout: i32,
    mem_type: i32,
}

/// Minimal NvBufSurfaceParams for reading pitch and data_ptr.
/// Matches the first 48 bytes of the full 384-byte struct defined
/// in nvbufsurface.h. Offsets verified via offsetof() on aarch64.
#[repr(C)]
struct NvBufSurfaceParamsHeader {
    _width: u32,
    _height: u32,
    pitch: u32,
    _color_format: u32,
    _layout: u32,
    _pad0: u32,
    _buffer_desc: i64,
    _data_size: u32,
    _pad1: u32,
    data_ptr: *mut c_void,
}

/// Minimal NvBufSurface for accessing num_filled and surface_list.
/// Matches the first 32 bytes of the full 64-byte struct.
#[repr(C)]
struct NvBufSurfaceHeader {
    _gpu_id: u32,
    _batch_size: u32,
    num_filled: u32,
    _is_contiguous: u8,
    _pad0: [u8; 3],
    _mem_type: u32,
    _pad1: u32,
    surface_list: *mut NvBufSurfaceParamsHeader,
}

/// Rectangle for crop/placement operations (sizeof=16).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NvBufSurfTransformRect {
    top: u32,    // offset 0
    left: u32,   // offset 4
    width: u32,  // offset 8
    height: u32, // offset 12
}

/// Transform session parameters (sizeof=16, must be set per-thread).
#[repr(C)]
struct NvBufSurfTransformSessionParams {
    compute_mode: i32,        // offset 0
    gpu_id: i32,              // offset 4
    cuda_stream: *mut c_void, // offset 8
}

/// Transform parameters for a single operation (sizeof=32 on aarch64).
#[repr(C)]
struct NvBufSurfTransformParams {
    transform_flag: u32,                     // offset 0
    transform_flip: i32,                     // offset 4
    transform_filter: i32,                   // offset 8
    _pad: i32,                               // offset 12 (padding to align pointers at 16)
    src_rect: *const NvBufSurfTransformRect, // offset 16
    dst_rect: *const NvBufSurfTransformRect, // offset 24
}

/// Dynamically loaded NvBufSurface + NvBufSurfTransform functions.
struct NvBufFunctions {
    _lib_surface: libloading::Library,
    _lib_transform: libloading::Library,

    // NvBufSurface management
    create: unsafe extern "C" fn(*mut *mut c_void, u32, *const NvBufSurfaceCreateParams) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,

    // NvBufSurfTransform
    set_session_params: unsafe extern "C" fn(*const NvBufSurfTransformSessionParams) -> i32,
    transform:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const NvBufSurfTransformParams) -> i32,
}

unsafe impl Send for NvBufFunctions {}
unsafe impl Sync for NvBufFunctions {}

static NVBUF: OnceLock<Option<NvBufFunctions>> = OnceLock::new();

fn load_nvbuf() -> Option<&'static NvBufFunctions> {
    NVBUF
        .get_or_init(|| unsafe {
            let lib_surface = [
                "libnvbufsurface.so",
                "/usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so",
                "/usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface.so",
            ]
            .iter()
            .find_map(|p| libloading::Library::new(*p).ok())?;

            let lib_transform = [
                "libnvbufsurftransform.so",
                "/usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurftransform.so",
                "/usr/lib/aarch64-linux-gnu/tegra/libnvbufsurftransform.so",
            ]
            .iter()
            .find_map(|p| libloading::Library::new(*p).ok())?;

            let fns = NvBufFunctions {
                create: *lib_surface.get(b"NvBufSurfaceCreate").ok()?,
                destroy: *lib_surface.get(b"NvBufSurfaceDestroy").ok()?,
                set_session_params: *lib_transform
                    .get(b"NvBufSurfTransformSetSessionParams")
                    .ok()?,
                transform: *lib_transform.get(b"NvBufSurfTransform").ok()?,
                _lib_surface: lib_surface,
                _lib_transform: lib_transform,
            };
            Some(fns)
        })
        .as_ref()
}

/// Returns true if NvBufSurfTransform libraries are available (Jetson only).
pub fn is_available() -> bool {
    load_nvbuf().is_some()
}

/// Letterboxed RGBA detection buffer backed by CUDA device memory.
///
/// Pre-filled with grey (pixel value 114) for letterbox padding.
/// NvBufSurfTransform writes the resized content into the `dst_rect`
/// region, leaving the grey border intact.
pub struct NvBufDetectionSurface {
    surface: *mut c_void,
    /// CUDA device pointer to the RGBA pixel data.
    pub data_ptr: CUdeviceptr,
    /// Pitch in bytes (width * 4 for RGBA, possibly aligned).
    pub pitch: u32,
    /// Model input size (square).
    pub size: u32,
    /// Destination rect where content is placed (letterbox geometry).
    dst_rect: NvBufSurfTransformRect,
    /// Scale factor applied during resize.
    pub scale: f32,
    /// Offset from top of the letterbox padding (for coordinate mapping).
    pub pad_top: u32,
    /// Offset from left of the letterbox padding.
    pub pad_left: u32,
    session_initialized: bool,
}

// SAFETY: The NvBufSurface and CUDA pointers are not thread-bound.
// Access is serialized by the caller (one detection per camera at a time).
unsafe impl Send for NvBufDetectionSurface {}

impl NvBufDetectionSurface {
    /// Allocate a CUDA-device-backed RGBA surface for detection preprocessing.
    ///
    /// `model_size` is the square input dimension (e.g. 1280 for YOLOv8).
    /// `src_width`/`src_height` define the source frame dimensions for
    /// computing the letterbox geometry.
    pub fn new(model_size: u32, src_width: u32, src_height: u32) -> Result<Self, String> {
        let fns = load_nvbuf().ok_or("NvBufSurfTransform libraries not available")?;
        crate::interop::cuda::cuda_ensure_context().map_err(|e| format!("CUDA context: {e}"))?;

        // Compute letterbox geometry: scale to fit, center
        let scale_w = model_size as f32 / src_width as f32;
        let scale_h = model_size as f32 / src_height as f32;
        let scale = scale_w.min(scale_h);
        let content_w = (src_width as f32 * scale).round() as u32;
        let content_h = (src_height as f32 * scale).round() as u32;
        let pad_left = (model_size - content_w) / 2;
        let pad_top = (model_size - content_h) / 2;

        let dst_rect = NvBufSurfTransformRect {
            top: pad_top,
            left: pad_left,
            width: content_w,
            height: content_h,
        };

        let params = NvBufSurfaceCreateParams {
            gpu_id: 0,
            width: model_size,
            height: model_size,
            size: 0,
            is_contiguous: 1,
            color_format: NVBUF_COLOR_FORMAT_RGBA,
            layout: NVBUF_LAYOUT_PITCH,
            mem_type: NVBUF_MEM_CUDA_DEVICE,
        };

        let mut surface: *mut c_void = std::ptr::null_mut();
        let ret = unsafe { (fns.create)(&mut surface, 1, &params) };
        if ret != 0 {
            return Err(format!("NvBufSurfaceCreate failed: {ret}"));
        }

        let (data_ptr, pitch) = unsafe {
            let hdr = surface as *mut NvBufSurfaceHeader;
            (*hdr).num_filled = 1;
            let params = &*(*hdr).surface_list;
            (params.data_ptr as u64, params.pitch)
        };

        if data_ptr == 0 {
            unsafe { (fns.destroy)(surface) };
            return Err("NvBufSurfaceCreate returned null data pointer".into());
        }

        // Fill entire surface with grey (114) for letterbox padding.
        // RGBA grey = [114, 114, 114, 255] but memset fills byte-by-byte,
        // so we fill with 114 (close enough for detection, alpha is ignored).
        let total_bytes = pitch as usize * model_size as usize;
        crate::interop::cuda::cuda_memset_d8(data_ptr, 114, total_bytes)
            .map_err(|e| format!("grey fill: {e}"))?;

        log::info!(
            "NvBufDetectionSurface: {}x{} RGBA, pitch={}, content={}x{} at ({},{}), scale={:.4}",
            model_size,
            model_size,
            pitch,
            content_w,
            content_h,
            pad_left,
            pad_top,
            scale,
        );

        Ok(Self {
            surface,
            data_ptr,
            pitch,
            size: model_size,
            dst_rect,
            scale,
            pad_top,
            pad_left,
            session_initialized: false,
        })
    }

    /// Transform an NVMM NV12 source surface into this RGBA detection buffer.
    ///
    /// `src_surface` is the raw `NvBufSurface*` pointer from a GStreamer NVMM
    /// buffer. The transform resizes from source resolution to the letterbox
    /// content area, converting NV12 → RGBA.
    ///
    /// After this call, `self.data_ptr` contains the letterboxed RGBA u8 frame
    /// ready for the normalize kernel.
    ///
    /// # Safety
    ///
    /// `src_surface` must point to a valid, mapped NvBufSurface with at least
    /// one filled NV12 buffer.
    pub unsafe fn transform_from_nvmm(&mut self, src_surface: *mut c_void) -> Result<(), String> {
        crate::profile_scope!("nvbuf_surf_transform");
        let fns = load_nvbuf().ok_or("NvBufSurfTransform libraries not available")?;

        // Initialize per-thread session on first call
        if !self.session_initialized {
            let session = NvBufSurfTransformSessionParams {
                compute_mode: NVBUF_TRANSFORM_COMPUTE_GPU,
                gpu_id: 0,
                cuda_stream: std::ptr::null_mut(),
            };
            let ret = unsafe { (fns.set_session_params)(&session) };
            if ret != 0 {
                return Err(format!("NvBufSurfTransformSetSessionParams failed: {ret}"));
            }
            self.session_initialized = true;
        }

        let params = NvBufSurfTransformParams {
            transform_flag: NVBUF_TRANSFORM_CROP_DST | NVBUF_TRANSFORM_FILTER,
            transform_flip: 0,
            transform_filter: NVBUF_TRANSFORM_FILTER_BILINEAR,
            _pad: 0,
            src_rect: std::ptr::null(),
            dst_rect: &self.dst_rect,
        };

        let ret = unsafe { (fns.transform)(src_surface, self.surface, &params) };
        if ret != 0 {
            return Err(format!("NvBufSurfTransform failed: {ret}"));
        }

        Ok(())
    }

    /// Re-fill the surface with grey (114). Call this if the source resolution
    /// changes and the letterbox geometry needs to be recalculated.
    pub fn refill_grey(&self) -> Result<(), String> {
        let total_bytes = self.pitch as usize * self.size as usize;
        crate::interop::cuda::cuda_memset_d8(self.data_ptr, 114, total_bytes)
            .map_err(|e| format!("grey fill: {e}"))?;
        Ok(())
    }
}

impl Drop for NvBufDetectionSurface {
    fn drop(&mut self) {
        if let Some(fns) = load_nvbuf() {
            unsafe {
                (fns.destroy)(self.surface);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_geometry_16_9() {
        // 1920x1080 (16:9) into 1280x1280 square
        let scale_w: f32 = 1280.0 / 1920.0;
        let scale_h: f32 = 1280.0 / 1080.0;
        let scale = scale_w.min(scale_h);
        let content_w = (1920.0_f32 * scale) as u32;
        let content_h = (1080.0_f32 * scale) as u32;
        let pad_left = (1280 - content_w) / 2;
        let pad_top = (1280 - content_h) / 2;

        assert_eq!(content_w, 1280);
        assert_eq!(content_h, 720);
        assert_eq!(pad_left, 0);
        assert_eq!(pad_top, 280);
        assert!((scale - 0.6667).abs() < 0.001);
    }

    #[test]
    fn letterbox_geometry_4_3() {
        // 4032x3040 (4:3) into 1280x1280 square
        let scale_w: f32 = 1280.0 / 4032.0;
        let scale_h: f32 = 1280.0 / 3040.0;
        let scale = scale_w.min(scale_h);
        let content_w = (4032.0_f32 * scale) as u32;
        let content_h = (3040.0_f32 * scale) as u32;
        let pad_left = (1280 - content_w) / 2;
        let pad_top = (1280 - content_h) / 2;

        assert_eq!(content_w, 1280);
        assert_eq!(content_h, 965);
        assert_eq!(pad_left, 0);
        assert_eq!(pad_top, 157);
    }

    #[test]
    fn nvbuf_not_available_on_desktop() {
        // On non-Jetson, libraries won't be found
        if !is_available() {
            assert!(NvBufDetectionSurface::new(1280, 1920, 1080).is_err());
        }
    }
}
