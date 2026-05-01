//! DX12/CUDA interop for NVIDIA Windows zero-copy.
//!
//! DX12 allocates shared buffers, CUDA imports them via
//! `cuImportExternalMemory`. Decode threads write via `cuMemcpy2D`,
//! wgpu renders from the same DX12 resources.
//!
//! ## Flow
//! 1. DX12 creates a committed buffer with `D3D12_HEAP_FLAG_SHARED`
//! 2. Gets NT shared handle via `CreateSharedResourceHandle`
//! 3. CUDA imports via `cuImportExternalMemory(D3D12_RESOURCE)`
//! 4. Maps to CUDA device pointer via `cuExternalMemoryGetMappedBuffer`
//! 5. Decode threads: `cuMemcpy2D` + `cuCtxSynchronize`
//! 6. wgpu renders from the DX12 buffer (same physical memory)

use crate::cuda_interop::CudaInteropError;
use crate::gpu::GpuContext;
use std::ffi::c_void;

/// A wgpu texture backed by a DX12 resource shared with CUDA.
pub struct SharedTexture {
    /// The wgpu texture for rendering.
    pub texture: wgpu::Texture,
    /// CUDA device pointer mapped to the same physical memory.
    pub cuda_ptr: crate::cuda_interop::CUdeviceptr,
    /// Row pitch in bytes.
    pub pitch: usize,
    /// CUDA external memory handle (freed on drop).
    _ext_mem: *mut c_void,
}

unsafe impl Send for SharedTexture {}
unsafe impl Sync for SharedTexture {}

impl Drop for SharedTexture {
    fn drop(&mut self) {
        if !self._ext_mem.is_null() {
            let _ = crate::cuda_interop::destroy_external_memory(self._ext_mem);
        }
    }
}

/// Create a wgpu texture backed by DX12/CUDA shared memory.
///
/// DX12 allocates a buffer with `D3D12_HEAP_FLAG_SHARED`, exports an
/// NT handle, and CUDA imports it. The returned texture can be used
/// in wgpu bind groups, and `cuda_ptr` can be written to via
/// `cuMemcpy2D`.
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
    use windows::Win32::Graphics::Dxgi::IDXGIResource1;
    use windows::core::Interface;

    let bpp = format_bytes_per_pixel(format);
    let row_bytes = width as usize * bpp;
    let pitch = row_bytes.div_ceil(256) * 256;
    let buffer_size = pitch * height as usize;

    let dxgi_format = match format {
        wgpu::TextureFormat::R8Unorm => DXGI_FORMAT_R8_UNORM,
        wgpu::TextureFormat::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        wgpu::TextureFormat::R16Unorm => DXGI_FORMAT_R16_UNORM,
        wgpu::TextureFormat::Rg16Unorm => DXGI_FORMAT_R16G16_UNORM,
        _ => DXGI_FORMAT_R8_UNORM,
    };

    // Create a DX12 committed resource with SHARED heap flag.
    let d3d12_resource: ID3D12Resource = unsafe {
        let hal_device_guard = gpu
            .device()
            .as_hal::<Dx12>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let raw_device = hal_device_guard.raw_device();

        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
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
        resource.ok_or_else(|| {
            CudaInteropError::VulkanError("CreateCommittedResource returned None".into())
        })?
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

    // CUDA imports the D3D12 resource via the shared handle.
    let (cuda_ptr, ext_mem) = crate::cuda_interop::import_d3d12_shared_handle(
        shared_handle.0 as *mut c_void,
        buffer_size,
    )?;

    unsafe {
        let _ = CloseHandle(shared_handle);
    }

    // Wrap the DX12 texture into wgpu directly - same resource,
    // CUDA writes via cuExternalMemoryGetMappedBuffer, wgpu reads.
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
        _ext_mem: ext_mem,
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
