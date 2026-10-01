//! CUDA memory, copies and external-resource interop.
//!
//! Enables NVDEC-decoded frames to reach wgpu textures without
//! leaving the GPU. Imports Vulkan-owned image memory on Linux into
//! persistent CUDA mappings.
//!
//! ## Architecture (inspired by [Gyroflow](https://github.com/gyroflow/gyroflow)'s approach)
//!
//! ```text
//! NVDEC decode → CUDA device memory (FFmpeg)
//!        ↓  cuMemcpy2D (GPU-to-GPU, no CPU)
//! Vulkan-owned linear image (exported fd, imported by CUDA)
//!        ↓  shared allocation
//! wgpu texture (via HAL) → render pipeline
//! ```
//!
//! ## Platform Support
//!
//! - **Linux**: POSIX file descriptor sharing (`VK_KHR_external_memory_fd`)
//! - **Windows**: CUDA memory and kernel operations, without graphics imports.
//! - CUDA is loaded dynamically — no compile-time CUDA SDK dependency.

use std::ffi::c_void;
use std::sync::OnceLock;
use thiserror::Error;

/// Errors from CUDA interop.
#[derive(Debug, Error)]
pub enum CudaInteropError {
    /// CUDA runtime not available (driver not installed).
    #[error("CUDA not available: {0}")]
    NotAvailable(String),

    /// CUDA API call returned an error.
    #[error("CUDA error {code} in {function}")]
    CudaError { function: &'static str, code: i32 },

    /// Vulkan interop failed.
    #[error("Vulkan interop: {0}")]
    VulkanError(String),

    /// The wgpu backend is not Vulkan (interop requires Vulkan).
    #[error("CUDA interop requires the Vulkan backend")]
    NotVulkan,

    /// CUDA allocation, decode and Vulkan rendering selected different devices.
    #[error("CUDA interop requires the renderer and current CUDA device to match CUDA device 0")]
    DeviceMismatch,
}

// ── CUDA types ──────────────────────────────────────────────────────

/// CUDA device pointer (64-bit).
pub type CUdeviceptr = u64;

type CUdevice = i32;
type CUresult = i32;

const CUDA_SUCCESS: CUresult = 0;

/// Memory types for cuMemcpy2D.
const CU_MEMORYTYPE_HOST: u32 = 1;
const CU_MEMORYTYPE_DEVICE: u32 = 2;

/// CUDA UUID (16 bytes, matching Vulkan's device UUID).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

/// 2D memory copy descriptor.
#[repr(C)]
#[derive(Clone)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: u32,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *const c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: u32,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *const c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

/// Opaque external memory handle for cross-API sharing.
#[cfg(target_os = "linux")]
type CUexternalMemory = *mut c_void;

/// Handle descriptor for importing external memory into CUDA.
#[cfg(target_os = "linux")]
#[repr(C)]
struct CudaExternalMemoryHandleDesc {
    handle_type: u32,
    handle: CudaExternalMemoryHandle,
    size: u64,
    flags: u32,
    _reserved: [u32; 16],
}

/// Retains the CUDA handle union's C layout, including on Linux.
#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaExternalMemoryWin32Handle {
    handle: *mut c_void,
    name: *const c_void,
}

#[cfg(target_os = "linux")]
#[repr(C)]
union CudaExternalMemoryHandle {
    fd: i32,
    win32: CudaExternalMemoryWin32Handle,
}

/// Buffer descriptor for mapping a region of external memory.
#[cfg(target_os = "linux")]
#[repr(C)]
struct CudaExternalMemoryBufferDesc {
    offset: u64,
    size: u64,
    flags: u32,
    _reserved: [u32; 16],
}

// ── Dynamic loader ──────────────────────────────────────────────────

/// Opaque CUDA context handle.
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;

/// Dynamically loaded CUDA functions.
///
/// Loaded once from `libcuda.so.1` (Linux) or `nvcuda.dll` (Windows).
/// Only the functions needed for memory, kernels and interop are loaded.
struct CudaFunctions {
    _lib_cuda: libloading::Library,
    _lib_cudart: Option<libloading::Library>,

    // Device management
    cuda_get_device: unsafe extern "C" fn(*mut CUdevice) -> CUresult,
    cu_device_get_uuid: unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,

    // Context management
    cu_init: unsafe extern "C" fn(u32) -> CUresult,
    cu_device_get: unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult,
    cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    cu_ctx_get_current: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    cu_ctx_set_current: unsafe extern "C" fn(CUcontext) -> CUresult,
    #[cfg(target_os = "linux")]
    cu_ctx_push_current: unsafe extern "C" fn(CUcontext) -> CUresult,
    #[cfg(target_os = "linux")]
    cu_ctx_pop_current: unsafe extern "C" fn(*mut CUcontext) -> CUresult,

    // 2D copy
    cu_memcpy_2d_v2: unsafe extern "C" fn(*const CudaMemcpy2D) -> CUresult,

    // Device memory management
    cu_mem_alloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_dtoh_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cu_memset_d8_v2: unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult,
    cu_mem_get_info_v2: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,

    // External memory imported from the rendering API.
    #[cfg(target_os = "linux")]
    cu_import_external_memory: unsafe extern "C" fn(
        *mut CUexternalMemory,
        *const CudaExternalMemoryHandleDesc,
    ) -> CUresult,
    #[cfg(target_os = "linux")]
    cu_external_memory_get_mapped_buffer: unsafe extern "C" fn(
        *mut CUdeviceptr,
        CUexternalMemory,
        *const CudaExternalMemoryBufferDesc,
    ) -> CUresult,
    #[cfg(target_os = "linux")]
    cu_destroy_external_memory: unsafe extern "C" fn(CUexternalMemory) -> CUresult,

    // Module / kernel launch
    cu_module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cu_module_unload: unsafe extern "C" fn(CUmodule) -> CUresult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const std::ffi::c_char) -> CUresult,
    cu_launch_kernel: unsafe extern "C" fn(
        CUfunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
}

// SAFETY: CudaFunctions contains function pointers and a library handle.
// The library remains loaded for the process lifetime (via OnceLock),
// so function pointers remain valid. CUDA calls are synchronized via
// cuCtxSynchronize.
unsafe impl Send for CudaFunctions {}
unsafe impl Sync for CudaFunctions {}

/// Global CUDA function table, loaded once.
static CUDA: OnceLock<Result<CudaFunctions, String>> = OnceLock::new();

impl CudaFunctions {
    /// Try to load CUDA libraries and resolve symbols.
    fn load() -> Result<Self, String> {
        unsafe {
            // Load the CUDA driver API library
            #[cfg(target_os = "linux")]
            let lib_cuda = libloading::Library::new("libcuda.so.1")
                .map_err(|e| format!("Failed to load libcuda.so.1: {e}"))?;
            #[cfg(target_os = "windows")]
            let lib_cuda = libloading::Library::new("nvcuda.dll")
                .map_err(|e| format!("Failed to load nvcuda.dll: {e}"))?;

            // Optionally load cudart for cudaGetDevice
            #[cfg(target_os = "linux")]
            let lib_cudart = libloading::Library::new("libcudart.so")
                .ok()
                .or_else(|| libloading::Library::new("libcudart.so.12").ok())
                .or_else(|| libloading::Library::new("libcudart.so.11").ok());
            #[cfg(target_os = "windows")]
            let lib_cudart = {
                // Try various cudart versions
                (0..=20)
                    .rev()
                    .find_map(|v| libloading::Library::new(format!("cudart64_{v}0.dll")).ok())
            };

            // Resolve cudaGetDevice from cudart (or fall back to cuCtxGetDevice)
            let cuda_get_device = if let Some(ref cudart) = lib_cudart {
                *cudart
                    .get::<unsafe extern "C" fn(*mut CUdevice) -> CUresult>(b"cudaGetDevice\0")
                    .map_err(|e| format!("cudaGetDevice: {e}"))?
            } else {
                *lib_cuda
                    .get::<unsafe extern "C" fn(*mut CUdevice) -> CUresult>(b"cuCtxGetDevice\0")
                    .map_err(|e| format!("cuCtxGetDevice: {e}"))?
            };

            macro_rules! load_sym {
                ($lib:expr, $name:literal) => {
                    *$lib
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| format!(concat!($name, ": {}"), e))?
                };
            }

            Ok(CudaFunctions {
                cuda_get_device,
                cu_device_get_uuid: load_sym!(lib_cuda, "cuDeviceGetUuid"),
                cu_ctx_synchronize: load_sym!(lib_cuda, "cuCtxSynchronize"),
                cu_init: load_sym!(lib_cuda, "cuInit"),
                cu_device_get: load_sym!(lib_cuda, "cuDeviceGet"),
                cu_device_primary_ctx_retain: load_sym!(lib_cuda, "cuDevicePrimaryCtxRetain"),
                cu_ctx_get_current: load_sym!(lib_cuda, "cuCtxGetCurrent"),
                cu_ctx_set_current: load_sym!(lib_cuda, "cuCtxSetCurrent"),
                #[cfg(target_os = "linux")]
                cu_ctx_push_current: load_sym!(lib_cuda, "cuCtxPushCurrent_v2"),
                #[cfg(target_os = "linux")]
                cu_ctx_pop_current: load_sym!(lib_cuda, "cuCtxPopCurrent_v2"),
                cu_memcpy_2d_v2: load_sym!(lib_cuda, "cuMemcpy2D_v2"),
                cu_mem_alloc_v2: load_sym!(lib_cuda, "cuMemAlloc_v2"),
                cu_mem_free_v2: load_sym!(lib_cuda, "cuMemFree_v2"),
                cu_mem_get_info_v2: load_sym!(lib_cuda, "cuMemGetInfo_v2"),
                cu_memcpy_dtoh_v2: load_sym!(lib_cuda, "cuMemcpyDtoH_v2"),
                cu_memset_d8_v2: load_sym!(lib_cuda, "cuMemsetD8_v2"),
                #[cfg(target_os = "linux")]
                cu_import_external_memory: load_sym!(lib_cuda, "cuImportExternalMemory"),
                #[cfg(target_os = "linux")]
                cu_external_memory_get_mapped_buffer: load_sym!(
                    lib_cuda,
                    "cuExternalMemoryGetMappedBuffer"
                ),
                #[cfg(target_os = "linux")]
                cu_destroy_external_memory: load_sym!(lib_cuda, "cuDestroyExternalMemory"),
                cu_module_load_data: load_sym!(lib_cuda, "cuModuleLoadData"),
                cu_module_unload: load_sym!(lib_cuda, "cuModuleUnload"),
                cu_module_get_function: load_sym!(lib_cuda, "cuModuleGetFunction"),
                cu_launch_kernel: load_sym!(lib_cuda, "cuLaunchKernel"),
                _lib_cuda: lib_cuda,
                _lib_cudart: lib_cudart,
            })
        }
    }
}

/// Get the global CUDA function table, loading it on first call.
fn cuda() -> Result<&'static CudaFunctions, CudaInteropError> {
    CUDA.get_or_init(CudaFunctions::load)
        .as_ref()
        .map_err(|e| CudaInteropError::NotAvailable(e.clone()))
}

/// Check a CUDA return code and convert to our error type.
fn check_cuda(function: &'static str, result: CUresult) -> Result<(), CudaInteropError> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaInteropError::CudaError {
            function,
            code: result,
        })
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Check if CUDA is available on this system.
///
/// Returns `true` if `libcuda.so.1` (or `nvcuda.dll`) can be loaded
/// and the required driver symbols are present.
pub fn is_cuda_available() -> bool {
    cuda().is_ok()
}

/// Persistent CUDA mapping of Vulkan-owned, dedicated linear image memory.
#[cfg(target_os = "linux")]
pub(crate) struct CudaImageMemory {
    ext_mem: CUexternalMemory,
    context: CUcontext,
    pub device_ptr: CUdeviceptr,
}

#[cfg(target_os = "linux")]
impl CudaImageMemory {
    pub(crate) fn import(fd: std::os::fd::OwnedFd, size: u64) -> Result<Self, CudaInteropError> {
        use std::os::fd::{AsRawFd, IntoRawFd};
        let cuda = cuda()?;
        cuda_ensure_context()?;
        let mut context = std::ptr::null_mut();
        unsafe {
            check_cuda("cuCtxGetCurrent", (cuda.cu_ctx_get_current)(&mut context))?;
        }
        let desc = CudaExternalMemoryHandleDesc {
            handle_type: 1, // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
            handle: CudaExternalMemoryHandle { fd: fd.as_raw_fd() },
            size,
            flags: 1, // CUDA_EXTERNAL_MEMORY_DEDICATED
            _reserved: [0; 16],
        };
        let mut ext_mem = std::ptr::null_mut();
        unsafe {
            check_cuda(
                "cuImportExternalMemory",
                (cuda.cu_import_external_memory)(&mut ext_mem, &desc),
            )?;
        }
        // CUDA consumes the descriptor only on successful import.
        let _ = fd.into_raw_fd();
        let mut memory = Self {
            ext_mem,
            context,
            device_ptr: 0,
        };
        let buffer = CudaExternalMemoryBufferDesc {
            offset: 0,
            size,
            flags: 0,
            _reserved: [0; 16],
        };
        unsafe {
            check_cuda(
                "cuExternalMemoryGetMappedBuffer",
                (cuda.cu_external_memory_get_mapped_buffer)(
                    &mut memory.device_ptr,
                    memory.ext_mem,
                    &buffer,
                ),
            )?;
        }
        Ok(memory)
    }
}

// The mapping is persistent and handles are only mutated during exclusive drop.
// Users synchronize CUDA and Vulkan before reuse; drop restores its own context.
#[cfg(target_os = "linux")]
unsafe impl Send for CudaImageMemory {}
#[cfg(target_os = "linux")]
unsafe impl Sync for CudaImageMemory {}

#[cfg(target_os = "linux")]
impl Drop for CudaImageMemory {
    fn drop(&mut self) {
        let Ok(cuda) = cuda() else {
            return;
        };
        unsafe {
            // wgpu can retire a texture on a different thread. Use the mapping's
            // context without replacing that thread's current CUDA context.
            let result = (cuda.cu_ctx_push_current)(self.context);
            if result != CUDA_SUCCESS {
                log::warn!("CUDA image mapping cleanup context failed: {result}");
                return;
            }
            if self.device_ptr != 0 {
                let result = (cuda.cu_mem_free_v2)(self.device_ptr);
                if result != CUDA_SUCCESS {
                    log::warn!("cuMemFree(image) failed: {result}");
                }
            }
            let result = (cuda.cu_destroy_external_memory)(self.ext_mem);
            if result != CUDA_SUCCESS {
                log::warn!("cuDestroyExternalMemory(image) failed: {result}");
            }
            let mut context = std::ptr::null_mut();
            let result = (cuda.cu_ctx_pop_current)(&mut context);
            if result != CUDA_SUCCESS {
                log::warn!("CUDA image mapping cleanup context restore failed: {result}");
            }
        }
    }
}

/// Get the UUID of the current CUDA device.
///
/// Used to match the CUDA device with the Vulkan physical device,
/// ensuring the shared memory is on the correct GPU.
pub fn get_cuda_device_uuid() -> Result<[u8; 16], CudaInteropError> {
    cuda_ensure_context()?;
    let cuda = cuda()?;
    unsafe {
        let mut device: CUdevice = 0;
        check_cuda("cudaGetDevice", (cuda.cuda_get_device)(&mut device))?;

        let mut uuid = CUuuid { bytes: [0; 16] };
        check_cuda(
            "cuDeviceGetUuid",
            (cuda.cu_device_get_uuid)(&mut uuid, device),
        )?;

        Ok(uuid.bytes)
    }
}

/// Device selected by decoders that request the default CUDA device.
#[cfg(target_os = "linux")]
pub(crate) fn default_cuda_device_uuid() -> Result<[u8; 16], CudaInteropError> {
    let cuda = cuda()?;
    unsafe {
        check_cuda("cuInit", (cuda.cu_init)(0))?;
        let mut device = 0;
        check_cuda("cuDeviceGet", (cuda.cu_device_get)(&mut device, 0))?;
        let mut uuid = CUuuid { bytes: [0; 16] };
        check_cuda(
            "cuDeviceGetUuid",
            (cuda.cu_device_get_uuid)(&mut uuid, device),
        )?;
        Ok(uuid.bytes)
    }
}

/// Synchronize the CUDA context (wait for all GPU work to complete).
pub fn cuda_synchronize() -> Result<(), CudaInteropError> {
    let cuda = cuda()?;
    unsafe {
        check_cuda("cuCtxSynchronize", (cuda.cu_ctx_synchronize)())?;
    }
    Ok(())
}

/// Query (free, total) device memory in bytes via the CUDA driver.
///
/// `free` is the OS-wide free device memory (matches `nvidia-smi`), so
/// on NVIDIA it reflects the COMBINED footprint of the wgpu/Vulkan
/// allocations, the NVDEC decode pool, and the TensorRT engine - the
/// honest budget for sizing the lookahead VRAM pool (a Vulkan-only
/// `VK_EXT_memory_budget` query would miss the CUDA-side allocations).
/// Requires a current CUDA context, which this ensures internally.
pub fn cuda_mem_get_info() -> Result<(usize, usize), CudaInteropError> {
    let cuda = cuda()?;
    cuda_ensure_context()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        check_cuda(
            "cuMemGetInfo_v2",
            (cuda.cu_mem_get_info_v2)(&mut free, &mut total),
        )?;
    }
    Ok((free, total))
}

/// Ensure a CUDA context is current on this thread.
///
/// FFmpeg's NVDEC backend pushes/pops its CUDA context around decode calls,
/// which may leave no context current after `avcodec_receive_frame`. This
/// function retains the primary context for device 0 and sets it as current
/// if no context is active. Safe to call multiple times (idempotent).
pub fn cuda_ensure_context() -> Result<(), CudaInteropError> {
    let cuda = cuda()?;
    unsafe {
        // cuInit must be called before any other driver API function.
        // It is safe to call multiple times (idempotent).
        check_cuda("cuInit", (cuda.cu_init)(0))?;

        let mut ctx: CUcontext = std::ptr::null_mut();
        check_cuda("cuCtxGetCurrent", (cuda.cu_ctx_get_current)(&mut ctx))?;

        if ctx.is_null() {
            // No context current — retain and set the primary context
            let mut device: CUdevice = 0;
            check_cuda("cuDeviceGet", (cuda.cu_device_get)(&mut device, 0))?;
            let mut primary_ctx: CUcontext = std::ptr::null_mut();
            check_cuda(
                "cuDevicePrimaryCtxRetain",
                (cuda.cu_device_primary_ctx_retain)(&mut primary_ctx, device),
            )?;
            check_cuda("cuCtxSetCurrent", (cuda.cu_ctx_set_current)(primary_ctx))?;
            log::debug!(
                "CUDA primary context set on thread {:?}",
                std::thread::current().name()
            );
        }
    }
    Ok(())
}

/// Copy a 2D region from host memory (CPU) to a CUDA device pointer.
///
/// Used by detector CPU-upload fallbacks (live camera → TrtGpuDetector)
/// where the source buffer lives in system memory but the detector
/// needs a CUDA device pointer. Cost is the PCIe/bus transfer time —
/// about 1-2 ms for a 1080p NV12 Y+UV plane on Jetson.
pub fn cuda_memcpy_htod_2d(
    dst: CUdeviceptr,
    dst_pitch: usize,
    src: *const u8,
    src_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<(), CudaInteropError> {
    let cuda = cuda()?;

    let desc = CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: CU_MEMORYTYPE_HOST,
        src_host: src as *const c_void,
        src_device: 0,
        src_array: std::ptr::null(),
        src_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: CU_MEMORYTYPE_DEVICE,
        dst_host: std::ptr::null_mut(),
        dst_device: dst,
        dst_array: std::ptr::null(),
        dst_pitch,
        width_in_bytes: width_bytes,
        height,
    };

    unsafe {
        check_cuda("cuMemcpy2D_v2 (H2D)", (cuda.cu_memcpy_2d_v2)(&desc))?;
    }
    Ok(())
}

/// Copy a 2D region between CUDA device pointers (GPU-to-GPU, no CPU involved).
///
/// This replaces the CPU round-trip (`av_hwframe_transfer_data` → swscale → upload).
/// For NV12→NV12 copy, this is a simple device memcpy on the GPU.
pub fn cuda_2d_copy(
    dst: CUdeviceptr,
    dst_pitch: usize,
    src: CUdeviceptr,
    src_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<(), CudaInteropError> {
    let cuda = cuda()?;

    let desc = CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: CU_MEMORYTYPE_DEVICE,
        src_host: std::ptr::null(),
        src_device: src,
        src_array: std::ptr::null(),
        src_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: CU_MEMORYTYPE_DEVICE,
        dst_host: std::ptr::null_mut(),
        dst_device: dst,
        dst_array: std::ptr::null(),
        dst_pitch,
        width_in_bytes: width_bytes,
        height,
    };

    unsafe {
        check_cuda("cuMemcpy2D_v2", (cuda.cu_memcpy_2d_v2)(&desc))?;
    }

    Ok(())
}

// ── Device memory management ────────────────────────────────────────

/// Allocate `size` bytes of device memory.
///
/// Returns a CUDA device pointer. The caller must free it with
/// [`cuda_mem_free`] when done.
pub fn cuda_mem_alloc(size: usize) -> Result<CUdeviceptr, CudaInteropError> {
    let cuda = cuda()?;
    let mut ptr: CUdeviceptr = 0;
    unsafe {
        check_cuda("cuMemAlloc_v2", (cuda.cu_mem_alloc_v2)(&mut ptr, size))?;
    }
    Ok(ptr)
}

/// Free device memory previously allocated with [`cuda_mem_alloc`].
pub fn cuda_mem_free(ptr: CUdeviceptr) -> Result<(), CudaInteropError> {
    let cuda = cuda()?;
    unsafe {
        check_cuda("cuMemFree_v2", (cuda.cu_mem_free_v2)(ptr))?;
    }
    Ok(())
}

/// Copy `size` bytes from device memory to host memory.
///
/// # Safety
/// `dst` must point to at least `size` bytes of valid host memory.
pub unsafe fn cuda_memcpy_dtoh(
    dst: *mut c_void,
    src: CUdeviceptr,
    size: usize,
) -> Result<(), CudaInteropError> {
    let cuda = cuda()?;
    unsafe {
        check_cuda("cuMemcpyDtoH_v2", (cuda.cu_memcpy_dtoh_v2)(dst, src, size))?;
    }
    Ok(())
}

/// Copy a 2D region from device memory to host memory.
///
/// Like [`cuda_2d_copy`] but the destination is host memory.
pub fn cuda_2d_copy_dtoh(
    dst: *mut c_void,
    dst_pitch: usize,
    src: CUdeviceptr,
    src_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<(), CudaInteropError> {
    let cuda = cuda()?;

    let desc = CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: CU_MEMORYTYPE_DEVICE,
        src_host: std::ptr::null(),
        src_device: src,
        src_array: std::ptr::null(),
        src_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: CU_MEMORYTYPE_HOST,
        dst_host: dst,
        dst_device: 0,
        dst_array: std::ptr::null(),
        dst_pitch,
        width_in_bytes: width_bytes,
        height,
    };

    unsafe {
        check_cuda("cuMemcpy2D_v2", (cuda.cu_memcpy_2d_v2)(&desc))?;
    }

    Ok(())
}

/// Fill `count` bytes of device memory with `value`.
pub fn cuda_memset_d8(ptr: CUdeviceptr, value: u8, count: usize) -> Result<(), CudaInteropError> {
    let cuda = cuda()?;
    unsafe {
        check_cuda("cuMemsetD8_v2", (cuda.cu_memset_d8_v2)(ptr, value, count))?;
    }
    Ok(())
}

// ── CUDA kernel management ─────────────────────────────────────────

/// A loaded CUDA kernel (module + function), ready for launch.
///
/// Created from PTX source via [`CudaKernel::from_ptx`]. The module
/// is unloaded when the kernel is dropped.
pub struct CudaKernel {
    module: CUmodule,
    function: CUfunction,
}

// SAFETY: CudaKernel holds opaque CUDA handles that are thread-safe
// when used with proper context management (cuda_ensure_context).
unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

impl CudaKernel {
    /// Load a kernel from PTX source.
    ///
    /// `ptx` must be a null-terminated PTX string. `func_name` is the
    /// `extern "C" __global__` entry point name.
    pub fn from_ptx(ptx: &[u8], func_name: &str) -> Result<Self, CudaInteropError> {
        let cuda = cuda()?;
        let mut module: CUmodule = std::ptr::null_mut();
        let mut function: CUfunction = std::ptr::null_mut();
        let c_name = std::ffi::CString::new(func_name)
            .map_err(|_| CudaInteropError::NotAvailable("invalid kernel name".into()))?;

        unsafe {
            check_cuda(
                "cuModuleLoadData",
                (cuda.cu_module_load_data)(&mut module, ptx.as_ptr().cast()),
            )?;
            check_cuda(
                "cuModuleGetFunction",
                (cuda.cu_module_get_function)(&mut function, module, c_name.as_ptr()),
            )?;
        }

        log::debug!("CUDA kernel '{func_name}' loaded from PTX");
        Ok(Self { module, function })
    }

    /// Launch the kernel with the given grid/block dimensions and arguments.
    ///
    /// # Safety
    /// `args` must be an array of pointers to the kernel arguments,
    /// matching the kernel's parameter list in order and type.
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), CudaInteropError> {
        let cuda = cuda()?;
        unsafe {
            check_cuda(
                "cuLaunchKernel",
                (cuda.cu_launch_kernel)(
                    self.function,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_mem,
                    std::ptr::null_mut(), // default stream
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
            )?;
        }
        Ok(())
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        if let Ok(cuda) = cuda() {
            // Ensure a CUDA context is current on this thread before
            // calling cuModuleUnload. Drop may run on a different thread
            // than the one that created the module (e.g., after the decode
            // thread's context was popped).
            if let Err(e) = cuda_ensure_context() {
                log::warn!("CudaKernel drop: failed to set CUDA context: {e}");
                return;
            }
            unsafe {
                let _ = (cuda.cu_module_unload)(self.module);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_available() {
        // This test is informational — it passes on both CUDA and non-CUDA systems.
        if is_cuda_available() {
            println!("CUDA is available");
            let uuid = get_cuda_device_uuid().expect("should get UUID");
            println!(
                "CUDA device UUID: {}",
                uuid.iter().map(|b| format!("{b:02x}")).collect::<String>()
            );
        } else {
            println!("CUDA not available (expected on non-NVIDIA systems)");
        }
    }
}
