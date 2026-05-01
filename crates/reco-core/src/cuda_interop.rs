//! CUDA/Vulkan interop for zero-copy GPU frame sharing.
//!
//! Enables NVDEC-decoded frames to reach wgpu textures without
//! leaving the GPU. Uses CUDA's Virtual Memory Management (VMM) API
//! to allocate shareable device memory, exports it as a POSIX file
//! descriptor (Linux) or Win32 handle (Windows), and imports it into
//! Vulkan via external memory extensions.
//!
//! ## Architecture (inspired by [Gyroflow](https://github.com/gyroflow/gyroflow)'s approach)
//!
//! ```text
//! NVDEC decode → CUDA device memory (FFmpeg)
//!        ↓  cuMemcpy2D (GPU-to-GPU, no CPU)
//! Shared CUDA/Vulkan buffer (allocated via VMM, exported as fd)
//!        ↓  Vulkan external memory import
//! wgpu texture (via HAL) → render pipeline
//! ```
//!
//! ## Platform Support
//!
//! - **Linux**: POSIX file descriptor sharing (`VK_KHR_external_memory_fd`)
//! - **Windows**: Win32 handle sharing (`VK_KHR_external_memory_win32`)
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
}

// ── CUDA types ──────────────────────────────────────────────────────

/// CUDA device pointer (64-bit).
pub type CUdeviceptr = u64;

type CUdevice = i32;
type CUresult = i32;

const CUDA_SUCCESS: CUresult = 0;

/// Memory allocation type: pinned (required for shareable allocations).
const CU_MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
/// Memory location type: device.
const CU_MEM_LOCATION_TYPE_DEVICE: u32 = 1;

/// Shareable handle types.
#[cfg(target_os = "linux")]
const CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: u32 = 1;
#[cfg(target_os = "windows")]
const CU_MEM_HANDLE_TYPE_WIN32: u32 = 2;

/// Memory access flags.
const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: u32 = 3;

/// Granularity query: minimum allocation size.
const CU_MEM_ALLOC_GRANULARITY_MINIMUM: u32 = 0;

/// Memory types for cuMemcpy2D.
const CU_MEMORYTYPE_HOST: u32 = 1;
const CU_MEMORYTYPE_DEVICE: u32 = 2;

/// CUDA UUID (16 bytes, matching Vulkan's device UUID).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

/// Allocation properties for CUDA VMM.
#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemAllocationProp {
    alloc_type: u32,
    handle_type: u32,
    location: CUmemLocation,
    win32_handle_value: *mut c_void,
    _reserved: [u64; 8],
}

/// Memory location descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemLocation {
    location_type: u32,
    id: i32,
}

/// Memory access descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
struct CUmemAccessDesc {
    location: CUmemLocation,
    flags: u32,
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

type CUmemGenericAllocationHandle = u64;

// ── Dynamic loader ──────────────────────────────────────────────────

/// Opaque CUDA context handle.
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;

/// Dynamically loaded CUDA functions.
///
/// Loaded once from `libcuda.so.1` (Linux) or `nvcuda.dll` (Windows).
/// Only the functions needed for VMM + interop are loaded.
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

    // VMM allocation
    cu_mem_get_allocation_granularity:
        unsafe extern "C" fn(*mut usize, *const CUmemAllocationProp, u32) -> CUresult,
    cu_mem_address_reserve:
        unsafe extern "C" fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64) -> CUresult,
    cu_mem_create: unsafe extern "C" fn(
        *mut CUmemGenericAllocationHandle,
        usize,
        *const CUmemAllocationProp,
        u64,
    ) -> CUresult,
    cu_mem_export_to_shareable_handle:
        unsafe extern "C" fn(*mut c_void, CUmemGenericAllocationHandle, u32, u64) -> CUresult,
    cu_mem_map: unsafe extern "C" fn(
        CUdeviceptr,
        usize,
        usize,
        CUmemGenericAllocationHandle,
        u64,
    ) -> CUresult,
    cu_mem_set_access:
        unsafe extern "C" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult,
    cu_mem_release: unsafe extern "C" fn(CUmemGenericAllocationHandle) -> CUresult,
    cu_mem_unmap: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,
    cu_mem_address_free: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,

    // 2D copy
    cu_memcpy_2d_v2: unsafe extern "C" fn(*const CudaMemcpy2D) -> CUresult,

    // Device memory management
    cu_mem_alloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_dtoh_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cu_memset_d8_v2: unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult,

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
                cu_mem_get_allocation_granularity: load_sym!(
                    lib_cuda,
                    "cuMemGetAllocationGranularity"
                ),
                cu_mem_address_reserve: load_sym!(lib_cuda, "cuMemAddressReserve"),
                cu_mem_create: load_sym!(lib_cuda, "cuMemCreate"),
                cu_mem_export_to_shareable_handle: load_sym!(
                    lib_cuda,
                    "cuMemExportToShareableHandle"
                ),
                cu_mem_map: load_sym!(lib_cuda, "cuMemMap"),
                cu_mem_set_access: load_sym!(lib_cuda, "cuMemSetAccess"),
                cu_mem_release: load_sym!(lib_cuda, "cuMemRelease"),
                cu_mem_unmap: load_sym!(lib_cuda, "cuMemUnmap"),
                cu_mem_address_free: load_sym!(lib_cuda, "cuMemAddressFree"),
                cu_memcpy_2d_v2: load_sym!(lib_cuda, "cuMemcpy2D_v2"),
                cu_mem_alloc_v2: load_sym!(lib_cuda, "cuMemAlloc_v2"),
                cu_mem_free_v2: load_sym!(lib_cuda, "cuMemFree_v2"),
                cu_memcpy_dtoh_v2: load_sym!(lib_cuda, "cuMemcpyDtoH_v2"),
                cu_memset_d8_v2: load_sym!(lib_cuda, "cuMemsetD8_v2"),
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

/// Shared CUDA/Vulkan memory allocation.
///
/// Holds both the CUDA device pointer and the metadata needed for cleanup.
/// When dropped, unmaps and frees the CUDA VMM allocation.
pub struct CudaSharedMemory {
    /// CUDA device pointer to the shared allocation.
    pub device_ptr: CUdeviceptr,
    /// Size of the allocation in bytes (rounded up to granularity).
    pub alloc_size: usize,
    /// The shareable handle (POSIX fd on Linux, Win32 handle on Windows).
    /// Only valid until imported by Vulkan — after that it is closed.
    #[cfg(target_os = "linux")]
    pub shared_handle: i32,
    #[cfg(target_os = "windows")]
    pub shared_handle: *mut c_void,
}

impl Drop for CudaSharedMemory {
    fn drop(&mut self) {
        if let Ok(cuda) = cuda() {
            unsafe {
                let unmap_rc = (cuda.cu_mem_unmap)(self.device_ptr, self.alloc_size);
                if unmap_rc != CUDA_SUCCESS {
                    log::warn!(
                        "cuMemUnmap(0x{:x}, {}) failed with error {}",
                        self.device_ptr,
                        self.alloc_size,
                        unmap_rc,
                    );
                }
                let free_rc = (cuda.cu_mem_address_free)(self.device_ptr, self.alloc_size);
                if free_rc != CUDA_SUCCESS {
                    log::warn!(
                        "cuMemAddressFree(0x{:x}, {}) failed with error {}",
                        self.device_ptr,
                        self.alloc_size,
                        free_rc,
                    );
                }
            }
        }
        // Note: shared_handle (fd) is NOT closed here. Vulkan's vkAllocateMemory
        // with VkImportMemoryFdInfoKHR takes ownership of the fd per the spec.
        // It is closed when vkFreeMemory runs (via the SharedTexture drop_callback).
    }
}

/// Check if CUDA is available on this system.
///
/// Returns `true` if `libcuda.so.1` (or `nvcuda.dll`) can be loaded
/// and all required VMM symbols are present.
pub fn is_cuda_available() -> bool {
    cuda().is_ok()
}

/// Get the UUID of the current CUDA device.
///
/// Used to match the CUDA device with the Vulkan physical device,
/// ensuring the shared memory is on the correct GPU.
pub fn get_cuda_device_uuid() -> Result<[u8; 16], CudaInteropError> {
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

/// Allocate CUDA device memory that can be shared with Vulkan.
///
/// Uses the CUDA Virtual Memory Management (VMM) API:
/// 1. `cuMemCreate` with shareable handle type
/// 2. `cuMemExportToShareableHandle` → POSIX fd (Linux) or Win32 handle (Windows)
/// 3. `cuMemMap` + `cuMemSetAccess` to make it usable from CUDA
///
/// The returned `CudaSharedMemory` owns the allocation and will clean up on drop.
pub fn allocate_shared_memory(size: usize) -> Result<CudaSharedMemory, CudaInteropError> {
    let cuda = cuda()?;

    unsafe {
        // Get current device
        let mut device: CUdevice = 0;
        check_cuda("cudaGetDevice", (cuda.cuda_get_device)(&mut device))?;

        #[cfg(target_os = "linux")]
        let handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        #[cfg(target_os = "windows")]
        let handle_type = CU_MEM_HANDLE_TYPE_WIN32;

        // On Windows, CUDA requires a valid OBJECT_ATTRIBUTES pointer
        // in win32HandleMetaData for CU_MEM_HANDLE_TYPE_WIN32 allocations.
        // Passing null causes CUDA_ERROR_INVALID_VALUE. The security
        // descriptor grants access to all users (Gyroflow pattern).
        #[cfg(target_os = "windows")]
        let win32_handle_value: *mut c_void = {
            use windows::Wdk::Foundation::OBJECT_ATTRIBUTES;
            use windows::Win32::Foundation::{HANDLE, OBJECT_ATTRIBUTE_FLAGS};
            use windows::Win32::Security::Authorization::{
                ConvertStringSecurityDescriptorToSecurityDescriptorA, SDDL_REVISION_1,
            };
            use windows::Win32::Security::PSECURITY_DESCRIPTOR;
            use windows::core::PCSTR;

            static OBJ_ATTRS_PTR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let ptr = *OBJ_ATTRS_PTR.get_or_init(|| {
                let sddl = c"D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
                let mut sec_desc = PSECURITY_DESCRIPTOR::default();
                let result = ConvertStringSecurityDescriptorToSecurityDescriptorA(
                    PCSTR::from_raw(sddl.as_ptr() as *const u8),
                    SDDL_REVISION_1,
                    &mut sec_desc,
                    None,
                );
                if result.is_err() {
                    log::warn!("Failed to create security descriptor for CUDA Win32 handle");
                    return 0;
                }
                let attrs = Box::new(OBJECT_ATTRIBUTES {
                    Length: std::mem::size_of::<OBJECT_ATTRIBUTES>() as u32,
                    RootDirectory: HANDLE::default(),
                    ObjectName: std::ptr::null_mut(),
                    Attributes: OBJECT_ATTRIBUTE_FLAGS::default(),
                    SecurityDescriptor: sec_desc.0 as *const _,
                    SecurityQualityOfService: std::ptr::null_mut(),
                });
                let ptr = Box::into_raw(attrs) as usize;
                // Intentionally leak: CUDA needs this for the process lifetime.
                // The security descriptor is also leaked (LocalFree not called).
                log::debug!("CUDA Win32 OBJECT_ATTRIBUTES at 0x{ptr:x}");
                ptr
            });
            ptr as *mut c_void
        };
        #[cfg(not(target_os = "windows"))]
        let win32_handle_value: *mut c_void = std::ptr::null_mut();

        let prop = CUmemAllocationProp {
            alloc_type: CU_MEM_ALLOCATION_TYPE_PINNED,
            handle_type,
            location: CUmemLocation {
                location_type: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            win32_handle_value,
            _reserved: [0; 8],
        };

        // Get minimum allocation granularity
        let mut granularity: usize = 0;
        check_cuda(
            "cuMemGetAllocationGranularity",
            (cuda.cu_mem_get_allocation_granularity)(
                &mut granularity,
                &prop,
                CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
        )?;

        // Round up to granularity
        let alloc_size = size.div_ceil(granularity) * granularity;

        // Reserve virtual address space
        let mut device_ptr: CUdeviceptr = 0;
        check_cuda(
            "cuMemAddressReserve",
            (cuda.cu_mem_address_reserve)(&mut device_ptr, alloc_size, granularity, 0, 0),
        )?;

        // Create the physical allocation
        let mut alloc_handle: CUmemGenericAllocationHandle = 0;
        check_cuda(
            "cuMemCreate",
            (cuda.cu_mem_create)(&mut alloc_handle, alloc_size, &prop, 0),
        )?;

        // Export shareable handle
        #[cfg(target_os = "linux")]
        let mut shared_handle: i32 = -1;
        #[cfg(target_os = "windows")]
        let mut shared_handle: *mut c_void = std::ptr::null_mut();

        check_cuda(
            "cuMemExportToShareableHandle",
            (cuda.cu_mem_export_to_shareable_handle)(
                &mut shared_handle as *mut _ as *mut c_void,
                alloc_handle,
                handle_type,
                0,
            ),
        )?;

        // Map the allocation into the virtual address space
        check_cuda(
            "cuMemMap",
            (cuda.cu_mem_map)(device_ptr, alloc_size, 0, alloc_handle, 0),
        )?;

        // Release the handle (the mapping holds a reference)
        check_cuda("cuMemRelease", (cuda.cu_mem_release)(alloc_handle))?;

        // Set read/write access
        let access_desc = CUmemAccessDesc {
            location: CUmemLocation {
                location_type: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device,
            },
            flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        check_cuda(
            "cuMemSetAccess",
            (cuda.cu_mem_set_access)(device_ptr, alloc_size, &access_desc, 1),
        )?;

        log::debug!(
            "CUDA shared memory allocated: {} bytes at 0x{:x} (handle={:?})",
            alloc_size,
            device_ptr,
            shared_handle,
        );

        Ok(CudaSharedMemory {
            device_ptr,
            alloc_size,
            shared_handle,
        })
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

    #[test]
    fn test_shared_memory_allocation() {
        if !is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let size = 1920 * 1080; // ~2MB
        let mem = allocate_shared_memory(size).expect("should allocate shared memory");
        assert!(mem.device_ptr != 0, "device_ptr should be non-null");
        assert!(mem.alloc_size >= size, "alloc_size should be >= requested");
        #[cfg(target_os = "linux")]
        assert!(mem.shared_handle >= 0, "shared fd should be valid");

        println!(
            "Shared memory: {} bytes at 0x{:x}, handle={:?}",
            mem.alloc_size, mem.device_ptr, mem.shared_handle
        );
        // Drop will clean up
    }
}
