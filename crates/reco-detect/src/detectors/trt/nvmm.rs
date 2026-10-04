//! Model-sized NVMM letterboxing owned by the native TensorRT detector.

use std::ffi::c_void;
use std::sync::OnceLock;

use super::{InputLayout, Result};
use reco_core::interop::cuda::{CUdeviceptr, cuda_ensure_context, cuda_memset_d8};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    _libraries: Option<[libloading::Library; 2]>,

    // NvBufSurface management
    create: unsafe extern "C" fn(*mut *mut c_void, u32, *const NvBufSurfaceCreateParams) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,

    // NvBufSurfTransform
    set_session_params: unsafe extern "C" fn(*const NvBufSurfTransformSessionParams) -> i32,
    transform:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const NvBufSurfTransformParams) -> i32,
}

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
                _libraries: Some([lib_surface, lib_transform]),
            };
            Some(fns)
        })
        .as_ref()
}

pub(super) fn is_available() -> bool {
    load_nvbuf().is_some()
}

/// One reusable model-sized allocation. Inference finishes before the next camera reuses it.
pub(super) struct NvmmPreprocessor {
    surface: *mut c_void,
    functions: &'static NvBufFunctions,
    pub(super) data_ptr: CUdeviceptr,
    pub(super) pitch: usize,
    size: u32,
    content: Option<NvBufSurfTransformRect>,
}

// SAFETY: native allocation access is serialized by Detector::detect; transforms are synchronous.
unsafe impl Send for NvmmPreprocessor {}

impl NvmmPreprocessor {
    pub(super) fn new(size: u32) -> Result<Self> {
        let functions = load_nvbuf().ok_or("NvBufSurfTransform libraries not available")?;
        Self::allocate(size, functions, Self::fill_grey)
    }

    fn allocate(
        size: u32,
        functions: &'static NvBufFunctions,
        fill: impl FnOnce(&Self) -> Result<()>,
    ) -> Result<Self> {
        if size == 0 || size > reco_core::calibration::MAX_DIM {
            return Err("invalid NVMM model dimensions".into());
        }
        cuda_ensure_context()?;
        let params = NvBufSurfaceCreateParams {
            gpu_id: 0,
            width: size,
            height: size,
            size: 0,
            is_contiguous: 1,
            color_format: NVBUF_COLOR_FORMAT_RGBA,
            layout: NVBUF_LAYOUT_PITCH,
            mem_type: NVBUF_MEM_CUDA_DEVICE,
        };
        let mut surface = std::ptr::null_mut();
        let status = unsafe { (functions.create)(&mut surface, 1, &params) };
        if status != 0 {
            return Err(format!("NvBufSurfaceCreate failed: {status}").into());
        }
        if surface.is_null() {
            return Err("NvBufSurfaceCreate returned null surface".into());
        }
        // Establish ownership before any fallible metadata validation or device fill.
        let mut prepared = Self {
            surface,
            functions,
            data_ptr: 0,
            pitch: 0,
            size,
            content: None,
        };
        let header = unsafe { &mut *surface.cast::<NvBufSurfaceHeader>() };
        if header.surface_list.is_null() {
            return Err("NvBufSurfaceCreate returned null surface list".into());
        }
        header.num_filled = 1;
        let plane = unsafe { &*header.surface_list };
        prepared.data_ptr = plane.data_ptr as u64;
        prepared.pitch = plane.pitch as usize;
        super::validate_plane(prepared.data_ptr, prepared.pitch, size as usize * 4, size)?;
        fill(&prepared)?;
        Ok(prepared)
    }

    fn fill_grey(&self) -> Result<()> {
        // Alpha is ignored by normalization.
        cuda_memset_d8(self.data_ptr, 114, self.pitch * self.size as usize)?;
        Ok(())
    }

    /// Source must remain valid until the synchronous transform returns.
    ///
    /// # Safety
    /// The pointer must reference a filled NV12 NvBufSurface matching the layout.
    pub(super) unsafe fn transform(
        &mut self,
        source: *mut c_void,
        layout: InputLayout,
    ) -> Result<()> {
        reco_core::profile_scope!("nvbuf_surf_transform");
        if source.is_null() {
            return Err("NVMM source surface is null".into());
        }
        cuda_ensure_context()?;
        let rect = NvBufSurfTransformRect {
            top: layout.padding.1,
            left: layout.padding.0,
            width: layout.resized.0,
            height: layout.resized.1,
        };
        if self.content.is_some_and(|previous| previous != rect) {
            self.fill_grey()?;
        }
        self.content = Some(rect);
        // Session parameters belong to the calling thread, not this movable buffer.
        // Set before every transform so another library's settings cannot leak in.
        let session = NvBufSurfTransformSessionParams {
            compute_mode: NVBUF_TRANSFORM_COMPUTE_GPU,
            gpu_id: 0,
            cuda_stream: std::ptr::null_mut(),
        };
        let status = unsafe { (self.functions.set_session_params)(&session) };
        if status != 0 {
            return Err(format!("NvBufSurfTransformSetSessionParams failed: {status}").into());
        }
        let params = NvBufSurfTransformParams {
            transform_flag: NVBUF_TRANSFORM_CROP_DST | NVBUF_TRANSFORM_FILTER,
            transform_flip: 0,
            transform_filter: NVBUF_TRANSFORM_FILTER_BILINEAR,
            _pad: 0,
            src_rect: std::ptr::null(),
            dst_rect: &rect,
        };
        let status = unsafe { (self.functions.transform)(source, self.surface, &params) };
        if status != 0 {
            return Err(format!("NvBufSurfTransform failed: {status}").into());
        }
        Ok(())
    }
}

impl Drop for NvmmPreprocessor {
    fn drop(&mut self) {
        // Match CUDA allocation context even if the detector moved between threads.
        if let Err(error) = cuda_ensure_context() {
            log::error!("NVMM surface cleanup context: {error}");
            return;
        }
        let status = unsafe { (self.functions.destroy)(self.surface) };
        if status != 0 {
            log::error!("NvBufSurfaceDestroy failed: {status}");
        }
    }
}

#[cfg(test)]
mod tests;
