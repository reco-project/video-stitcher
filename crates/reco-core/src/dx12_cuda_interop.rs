//! DX12/CUDA interop for NVIDIA Windows zero-copy.
//!
//! DX12 allocates shared TEXTURE2D resources, CUDA imports them via
//! `cuImportExternalMemory` + `cuExternalMemoryGetMappedMipmappedArray`.
//! Decode threads write via `cuMemcpy2D` (device→array), wgpu renders
//! from the same DX12 textures.
//!
//! ## Flow
//! 1. DX12 `CreateCommittedResource(TEXTURE2D, D3D12_HEAP_FLAG_SHARED)`
//! 2. `GetResourceAllocationInfo` for actual allocation size
//! 3. `CreateSharedHandle` → NT handle
//! 4. CUDA `cuImportExternalMemory(D3D12_RESOURCE)`
//! 5. `cuExternalMemoryGetMappedMipmappedArray` → CUmipmappedArray
//! 6. `cuMipmappedArrayGetLevel(0)` → CUarray
//! 7. Decode threads: `cuMemcpy2D(CU_MEMORYTYPE_ARRAY)` + `cuCtxSynchronize`
//! 8. wgpu renders from the DX12 texture (same physical memory)

use crate::cuda_interop::CudaInteropError;
use crate::gpu::GpuContext;
use std::ffi::c_void;

/// A wgpu texture backed by a DX12 TEXTURE2D shared with CUDA.
///
/// CUDA accesses the texture via a `CUarray` (not a linear device
/// pointer) because DX12 TEXTURE2D resources use tiled memory layouts.
pub struct SharedTexture {
    /// The wgpu texture for rendering.
    pub texture: wgpu::Texture,
    /// CUDA array handle mapped to the same physical memory.
    pub cuda_array: *mut c_void,
    /// CUDA external memory handle (freed on drop).
    _ext_mem: *mut c_void,
    /// CUDA mipmapped array handle (freed on drop before ext_mem).
    _mipmapped_array: *mut c_void,
}

unsafe impl Send for SharedTexture {}
unsafe impl Sync for SharedTexture {}

impl Drop for SharedTexture {
    fn drop(&mut self) {
        // Order matters: destroy mipmapped array before external memory.
        if !self._mipmapped_array.is_null() {
            let _ = crate::cuda_interop::destroy_mipmapped_array(self._mipmapped_array);
        }
        if !self._ext_mem.is_null() {
            let _ = crate::cuda_interop::destroy_external_memory(self._ext_mem);
        }
    }
}

/// Create a wgpu texture backed by DX12/CUDA shared memory.
///
/// DX12 allocates a TEXTURE2D with `D3D12_HEAP_FLAG_SHARED`, exports an
/// NT handle, and CUDA imports it as a mipmapped array. The returned
/// texture can be used in wgpu bind groups, and `cuda_array` can be
/// written to via `cuMemcpy2D` with `CU_MEMORYTYPE_ARRAY` destination.
pub fn create_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<SharedTexture, CudaInteropError> {
    use wgpu::hal::api::Dx12;
    use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::Interface;

    let (dxgi_format, num_channels) = match format {
        wgpu::TextureFormat::R8Unorm => (DXGI_FORMAT_R8_UNORM, 1u32),
        wgpu::TextureFormat::Rg8Unorm => (DXGI_FORMAT_R8G8_UNORM, 2),
        wgpu::TextureFormat::R16Unorm => (DXGI_FORMAT_R16_UNORM, 1),
        wgpu::TextureFormat::Rg16Unorm => (DXGI_FORMAT_R16G16_UNORM, 2),
        _ => (DXGI_FORMAT_R8_UNORM, 1),
    };

    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: dxgi_format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
    };

    // Create a DX12 committed resource with SHARED heap flag.
    let (d3d12_resource, alloc_size) = unsafe {
        let hal_device_guard = gpu
            .device()
            .as_hal::<Dx12>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let raw_device = hal_device_guard.raw_device();

        // DX12 textures use tiled memory; the actual allocation may
        // be larger than width*height*bpp. CUDA needs the real size.
        let alloc_info = raw_device.GetResourceAllocationInfo(0, &[desc]);
        let alloc_size = alloc_info.SizeInBytes as usize;

        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };

        let mut resource: Option<ID3D12Resource> = None;
        raw_device
            .CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
            .map_err(|e| {
                CudaInteropError::VulkanError(format!("DX12 CreateCommittedResource: {e}"))
            })?;
        let resource = resource.ok_or_else(|| {
            CudaInteropError::VulkanError("CreateCommittedResource returned None".into())
        })?;
        (resource, alloc_size)
    };

    // Get shared NT handle for CUDA import.
    let shared_handle: HANDLE = unsafe {
        let hal_device_guard = gpu
            .device()
            .as_hal::<Dx12>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let raw_device = hal_device_guard.raw_device();
        raw_device
            .CreateSharedHandle(&d3d12_resource, None, GENERIC_ALL.0, None)
            .map_err(|e| CudaInteropError::VulkanError(format!("DX12 CreateSharedHandle: {e}")))?
    };

    // CUDA imports the D3D12 texture as a mipmapped array.
    let (cuda_array, ext_mem, mipmapped_array) = crate::cuda_interop::import_d3d12_shared_texture(
        shared_handle.0 as *mut c_void,
        alloc_size,
        width,
        height,
        num_channels,
    )?;

    unsafe {
        let _ = CloseHandle(shared_handle);
    }

    // Wrap the DX12 texture into wgpu.
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
        "DX12/CUDA shared texture (array): {}x{} {:?}, alloc={}KB, array={:?}",
        width,
        height,
        format,
        alloc_size / 1024,
        cuda_array,
    );

    Ok(SharedTexture {
        texture: wgpu_texture,
        cuda_array,
        _ext_mem: ext_mem,
        _mipmapped_array: mipmapped_array,
    })
}
