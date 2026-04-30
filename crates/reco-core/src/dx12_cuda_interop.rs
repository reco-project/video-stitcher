//! DX12 side of CUDA/DX12 interop for NVIDIA Windows zero-copy.
//!
//! Mirrors [`vulkan_interop`](crate::vulkan_interop) for Windows:
//! imports CUDA-exported shared memory into DX12, then wraps the
//! resulting resource into a [`wgpu::Texture`] via the HAL.
//!
//! ## Flow
//! 1. CUDA allocates shareable memory and exports a Win32 NT handle
//! 2. DX12 opens the handle as a committed resource
//! 3. Wraps into `wgpu::Texture` via `create_texture_from_hal`
//! 4. Decode threads write via `cuMemcpy2D` + `cuCtxSynchronize`
//! 5. wgpu renders from the DX12 texture (same physical memory)

use crate::cuda_interop::{CudaInteropError, CudaSharedMemory};
use crate::gpu::GpuContext;

/// A wgpu texture backed by CUDA shared memory (DX12 import).
pub struct SharedTexture {
    /// The wgpu texture, usable in bind groups and render passes.
    pub texture: wgpu::Texture,
    /// The CUDA device pointer to the shared memory.
    pub cuda_ptr: crate::cuda_interop::CUdeviceptr,
    /// Pitch (row stride in bytes) of the DX12 resource.
    pub pitch: usize,
    /// Keeps the CUDA allocation alive.
    _shared_mem: CudaSharedMemory,
}

// SAFETY: SharedTexture's only non-Send field is the *mut c_void handle
// in CudaSharedMemory, which is a Win32 NT handle that can be used from
// any thread. The wgpu::Texture and CUDA device pointer are also thread-safe.
unsafe impl Send for SharedTexture {}
unsafe impl Sync for SharedTexture {}

/// Create a wgpu texture backed by CUDA shared memory via DX12 import.
pub fn create_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<SharedTexture, CudaInteropError> {
    use wgpu::hal::api::Dx12;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

    let bpp = format_bytes_per_pixel(format);
    let row_bytes = width as usize * bpp;
    let pitch = row_bytes.div_ceil(256) * 256;
    let alloc_size = pitch * height as usize;

    let shared_mem = crate::cuda_interop::allocate_shared_memory(alloc_size)?;
    let cuda_ptr = shared_mem.device_ptr;
    let win32_handle = shared_mem.shared_handle;

    let d3d12_resource: ID3D12Resource = unsafe {
        let hal_device_guard = gpu
            .device()
            .as_hal::<Dx12>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let raw_device = hal_device_guard.raw_device();
        let handle = HANDLE(win32_handle);
        let mut resource: Option<ID3D12Resource> = None;
        raw_device
            .OpenSharedHandle(handle, &mut resource)
            .map_err(|e| CudaInteropError::VulkanError(format!("DX12 OpenSharedHandle: {e}")))?;
        resource
            .ok_or_else(|| CudaInteropError::VulkanError("OpenSharedHandle returned None".into()))?
    };

    let wgpu_texture = unsafe {
        let hal_texture = wgpu::hal::dx12::Device::texture_from_raw(
            d3d12_resource,
            format,
            wgpu::TextureDimension::D2,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            1,
            1,
        );

        gpu.device().create_texture_from_hal::<Dx12>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("cuda_dx12_shared"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )
    };

    log::info!(
        "DX12/CUDA shared texture: {}x{} {:?}, pitch={}, cuda_ptr=0x{:x}",
        width,
        height,
        format,
        pitch,
        cuda_ptr,
    );

    Ok(SharedTexture {
        texture: wgpu_texture,
        cuda_ptr,
        pitch,
        _shared_mem: shared_mem,
    })
}

fn format_bytes_per_pixel(format: wgpu::TextureFormat) -> usize {
    match format {
        wgpu::TextureFormat::R8Unorm => 1,
        wgpu::TextureFormat::Rg8Unorm => 2,
        wgpu::TextureFormat::R16Unorm => 2,
        wgpu::TextureFormat::Rg16Unorm => 4,
        wgpu::TextureFormat::Rgba8Unorm => 4,
        _ => 4,
    }
}
