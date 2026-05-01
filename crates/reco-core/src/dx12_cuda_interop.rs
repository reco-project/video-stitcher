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

/// A wgpu texture backed by a DX12 buffer shared with CUDA.
pub struct SharedTexture {
    /// The wgpu texture for rendering.
    pub texture: wgpu::Texture,
    /// CUDA device pointer mapped to the same memory as `shared_buffer`.
    pub cuda_ptr: crate::cuda_interop::CUdeviceptr,
    /// Row pitch in bytes (aligned to 256 for DX12).
    pub pitch: usize,
    /// DX12 shared buffer that CUDA writes to.
    /// Per-frame: copy this buffer → texture before rendering.
    pub shared_buffer: wgpu::Buffer,
    /// Buffer size in bytes.
    pub buffer_size: usize,
    /// Texture width (for buffer→texture copy).
    pub width: u32,
    /// Texture height (for buffer→texture copy).
    pub height: u32,
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

        // Use a BUFFER resource so CUDA can map it as a linear device
        // pointer via cuExternalMemoryGetMappedBuffer. Textures would
        // need cuExternalMemoryGetMappedMipmappedArray which doesn't
        // give us a device pointer for cuMemcpy2D.
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: buffer_size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
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

    // Create a regular wgpu texture for rendering. CUDA writes to
    // the shared buffer, then we copy buffer→texture per frame.
    // This adds ~0.1ms but avoids the CUDA array vs device pointer
    // mismatch with DX12 texture resources.
    let wgpu_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("cuda_dx12_render"),
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
    });

    // Wrap the DX12 buffer as a wgpu buffer for the copy source.
    let shared_buffer = unsafe {
        let hal_buffer =
            wgpu::hal::dx12::Device::buffer_from_raw(d3d12_resource, buffer_size as u64);
        gpu.device().create_buffer_from_hal::<Dx12>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some("cuda_dx12_shared_buf"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
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
        shared_buffer,
        buffer_size,
        width,
        height,
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
