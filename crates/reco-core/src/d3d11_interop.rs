//! D3D11VA to wgpu zero-copy interop via staging textures.
//!
//! Copies FFmpeg's D3D11VA decoded NV12 frames to shared staging textures,
//! then imports those into wgpu via DX12 shared handles. Eliminates the
//! GPU -> CPU -> GPU roundtrip (~50ms/frame on Surface-class hardware).
//!
//! ## Architecture
//!
//! ```text
//! D3D11VA decode pool (FFmpeg-owned)
//!        |  CopySubresourceRegion (~0.2ms GPU-GPU)
//! D3D11 staging texture (SHARED_NTHANDLE)
//!        |  NT handle -> DX12 OpenSharedHandle
//! wgpu NV12 texture -> Plane0 (R8Unorm Y) + Plane1 (Rg8Unorm UV)
//!        |
//! stitch render pass
//! ```

use crate::gpu::GpuContext;
use std::ffi::c_void;
use thiserror::Error;

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, HMODULE, LUID};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_QUERY, D3D11_QUERY_DESC, D3D11_RESOURCE_MISC_SHARED,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Query,
    ID3D11Texture2D,
};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, IDXGIResource1,
};
use windows::core::Interface;

/// Errors from D3D11VA interop.
#[derive(Debug, Error)]
pub enum D3d11InteropError {
    #[error("D3D11: {0}")]
    D3d11(String),

    #[error("DXGI: {0}")]
    Dxgi(String),

    #[error("wgpu NV12 texture format not available")]
    Nv12NotSupported,

    #[error("wgpu backend is not DX12")]
    NotDx12,

    #[error("staging copy failed: {0}")]
    StagingCopy(String),
}

impl From<windows::core::Error> for D3d11InteropError {
    fn from(e: windows::core::Error) -> Self {
        Self::D3d11(e.to_string())
    }
}

impl From<D3d11InteropError> for crate::session::SessionError {
    fn from(e: D3d11InteropError) -> Self {
        crate::session::SessionError::ZeroCopy(e.to_string())
    }
}

/// Double-buffered NV12 staging pool for D3D11VA -> wgpu zero-copy.
///
/// Pre-allocates 4 staging textures (2 per camera, ping-pong) with shared
/// handles. Each staging texture is imported into wgpu as an NV12 texture
/// with pre-built Y and UV plane views.
pub struct D3d11StagingPool {
    _device: ID3D11Device,
    context: ID3D11DeviceContext,
    staging: [ID3D11Texture2D; 4],
    _wgpu_textures: [wgpu::Texture; 4],
    y_views: [wgpu::TextureView; 4],
    uv_views: [wgpu::TextureView; 4],
    event_query: ID3D11Query,
    width: u32,
    height: u32,
}

impl D3d11StagingPool {
    /// Create a staging pool on the same physical adapter as wgpu's DX12 device.
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Result<Self, D3d11InteropError> {
        use wgpu::hal::api::Dx12;

        if !gpu.is_dx12() {
            return Err(D3d11InteropError::NotDx12);
        }

        // Get the adapter LUID from wgpu so we create D3D11 on the same GPU.
        let adapter_luid = gpu.adapter_info.device as u64;

        // Find the matching DXGI adapter by LUID.
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        let adapter = find_adapter_by_luid(&factory, adapter_luid)?;
        let adapter_desc = unsafe { adapter.GetDesc1()? };
        log::info!(
            "D3D11 interop: matched adapter '{}' (LUID {}:{})",
            String::from_utf16_lossy(
                &adapter_desc
                    .Description
                    .iter()
                    .take_while(|c| **c != 0)
                    .copied()
                    .collect::<Vec<_>>()
            ),
            adapter_desc.AdapterLuid.HighPart,
            adapter_desc.AdapterLuid.LowPart,
        );

        // Create D3D11 device on that adapter.
        let mut d3d11_device: Option<ID3D11Device> = None;
        let mut d3d11_context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut d3d11_device),
                None,
                Some(&mut d3d11_context),
            )?;
        }
        let device = d3d11_device
            .ok_or_else(|| D3d11InteropError::D3d11("D3D11CreateDevice returned None".into()))?;
        let context = d3d11_context.ok_or_else(|| {
            D3d11InteropError::D3d11("D3D11CreateDevice returned no context".into())
        })?;

        // Create event query for staging copy synchronization.
        let query_desc = D3D11_QUERY_DESC {
            Query: D3D11_QUERY(0), // D3D11_QUERY_EVENT = 0
            MiscFlags: 0,
        };
        let event_query: ID3D11Query = unsafe {
            let mut query = None;
            device.CreateQuery(&query_desc, Some(&mut query))?;
            query.ok_or_else(|| D3d11InteropError::D3d11("CreateQuery returned None".into()))?
        };

        // Create 4 NV12 staging textures with shared handles.
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_NV12,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: 0,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0)
                as u32,
        };

        let mut staging_textures: Vec<ID3D11Texture2D> = Vec::with_capacity(4);
        let mut wgpu_textures: Vec<wgpu::Texture> = Vec::with_capacity(4);
        let mut y_views: Vec<wgpu::TextureView> = Vec::with_capacity(4);
        let mut uv_views: Vec<wgpu::TextureView> = Vec::with_capacity(4);

        for i in 0..4 {
            let staging: ID3D11Texture2D = unsafe {
                let mut tex = None;
                device.CreateTexture2D(&desc, None, Some(&mut tex))?;
                tex.ok_or_else(|| D3d11InteropError::D3d11("CreateTexture2D returned None".into()))?
            };

            // Get shared NT handle.
            let dxgi_resource: IDXGIResource1 = staging.cast()?;
            let handle: HANDLE =
                unsafe { dxgi_resource.CreateSharedHandle(None, GENERIC_ALL.0, None)? };

            // Import into wgpu via DX12 HAL.
            let wgpu_texture = unsafe { import_d3d11_shared_handle(gpu, handle, width, height)? };

            // Close the NT handle (DX12 has its own reference now).
            unsafe {
                let _ = CloseHandle(handle);
            }

            // Create NV12 plane views.
            let y_view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("d3d11_y_{i}")),
                format: Some(wgpu::TextureFormat::R8Unorm),
                dimension: None,
                aspect: wgpu::TextureAspect::Plane0,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
                usage: None,
            });
            let uv_view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("d3d11_uv_{i}")),
                format: Some(wgpu::TextureFormat::Rg8Unorm),
                dimension: None,
                aspect: wgpu::TextureAspect::Plane1,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
                usage: None,
            });

            staging_textures.push(staging);
            wgpu_textures.push(wgpu_texture);
            y_views.push(y_view);
            uv_views.push(uv_view);
        }

        log::info!(
            "D3D11VA staging pool ready: {}x{} NV12, 4 slots",
            width,
            height
        );

        Ok(Self {
            _device: device,
            context,
            staging: staging_textures.try_into().unwrap(),
            _wgpu_textures: wgpu_textures.try_into().unwrap(),
            y_views: y_views.try_into().unwrap(),
            uv_views: uv_views.try_into().unwrap(),
            event_query,
            width,
            height,
        })
    }

    /// Copy a D3D11VA decoded frame from the decode pool to a staging slot.
    ///
    /// Performs `CopySubresourceRegion` (GPU-to-GPU, ~0.2ms) then waits
    /// for completion via an event query before returning.
    pub fn stage_frame(
        &self,
        src_texture: *mut c_void,
        array_slice: usize,
        slot: usize,
    ) -> Result<(), D3d11InteropError> {
        if src_texture.is_null() {
            return Err(D3d11InteropError::StagingCopy("null source texture".into()));
        }
        if slot >= 4 {
            return Err(D3d11InteropError::StagingCopy(format!(
                "slot {slot} out of range (0..4)"
            )));
        }

        unsafe {
            // Reconstruct the ID3D11Texture2D from the raw pointer.
            // SAFETY: FFmpeg guarantees data[0] is a valid ID3D11Texture2D*
            // for the lifetime of the decoded frame. We AddRef via clone
            // to get our own reference, then release at scope end.
            let unknown: windows::core::IUnknown = windows::core::IUnknown::from_raw(src_texture);
            let src: ID3D11Texture2D = unknown.cast()?;
            // Re-leak the original pointer so FFmpeg keeps its reference.
            std::mem::forget(unknown);

            // D3D11CalcSubresource(MipSlice=0, ArraySlice, MipLevels=1) = ArraySlice
            let src_subresource = array_slice as u32;

            self.context.CopySubresourceRegion(
                &self.staging[slot],
                0, // dst subresource
                0,
                0,
                0, // dst x, y, z
                &src,
                src_subresource,
                None, // full region
            );

            // Flush and wait for the copy to complete.
            self.context.Flush();
            self.context.End(&self.event_query);
            loop {
                let mut done: u32 = 0;
                let hr = self.context.GetData(
                    &self.event_query,
                    Some(&mut done as *mut u32 as *mut c_void),
                    std::mem::size_of::<u32>() as u32,
                    0,
                );
                if hr.is_ok() && done != 0 {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        Ok(())
    }

    /// Y plane view (R8Unorm, full resolution) for the given slot.
    pub fn y_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.y_views[slot]
    }

    /// UV plane view (Rg8Unorm, half resolution) for the given slot.
    pub fn uv_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.uv_views[slot]
    }

    /// Staging texture width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Staging texture height.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Find a DXGI adapter matching the given adapter device ID (LUID encoding).
fn find_adapter_by_luid(
    factory: &IDXGIFactory1,
    device_id: u64,
) -> Result<IDXGIAdapter1, D3d11InteropError> {
    // wgpu stores the LUID as a u64 in AdapterInfo.device.
    // Reconstruct the LUID: low 32 bits = LowPart, high 32 bits = HighPart.
    let target_luid = LUID {
        LowPart: device_id as u32,
        HighPart: (device_id >> 32) as i32,
    };

    let mut i = 0u32;
    loop {
        let adapter: IDXGIAdapter1 = match unsafe { factory.EnumAdapters1(i) } {
            Ok(a) => a,
            Err(_) => break,
        };
        let desc = unsafe { adapter.GetDesc1()? };
        if desc.AdapterLuid.LowPart == target_luid.LowPart
            && desc.AdapterLuid.HighPart == target_luid.HighPart
        {
            return Ok(adapter);
        }
        i += 1;
    }

    Err(D3d11InteropError::Dxgi(format!(
        "no DXGI adapter with LUID {}:{} (device_id={})",
        target_luid.HighPart, target_luid.LowPart, device_id
    )))
}

/// Import a D3D11 shared handle into wgpu as an NV12 texture.
///
/// # Safety
/// `handle` must be a valid NT shared handle from `CreateSharedHandle`.
unsafe fn import_d3d11_shared_handle(
    gpu: &GpuContext,
    handle: HANDLE,
    width: u32,
    height: u32,
) -> Result<wgpu::Texture, D3d11InteropError> {
    use wgpu::hal::api::Dx12;

    // Open the shared handle as a D3D12 resource.
    let d3d12_resource: ID3D12Resource = {
        let hal_device_guard = gpu
            .device()
            .as_hal::<Dx12>()
            .ok_or(D3d11InteropError::NotDx12)?;
        let raw_device = hal_device_guard.raw_device();
        unsafe { raw_device.OpenSharedHandle(handle)? }
    };

    // Wrap the D3D12 resource in a wgpu HAL texture.
    let hal_texture = unsafe {
        wgpu::hal::dx12::Device::texture_from_raw(
            d3d12_resource,
            wgpu::TextureFormat::NV12,
            wgpu::TextureDimension::D2,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            1, // mip levels
            1, // sample count
        )
    };

    let desc = wgpu::TextureDescriptor {
        label: Some("d3d11_staging_nv12"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::NV12,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    // Wrap the HAL texture into a wgpu::Texture.
    let texture = unsafe {
        gpu.device()
            .create_texture_from_hal::<Dx12>(hal_texture, &desc)
    };

    Ok(texture)
}
