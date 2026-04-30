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

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_QUERY, D3D11_QUERY_DESC,
    D3D11_RESOURCE_MISC_SHARED, D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING, ID3D11Device, ID3D11DeviceContext, ID3D11Query,
    ID3D11Texture2D,
};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::IDXGIResource1;
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
    /// CPU-readable staging texture for detection readback.
    /// Created lazily on first detection frame.
    readback_staging: Option<ID3D11Texture2D>,
    width: u32,
    height: u32,
}

impl D3d11StagingPool {
    /// Create a staging pool using FFmpeg's D3D11 device.
    ///
    /// `d3d11_device_ptr` and `d3d11_context_ptr` must be the raw
    /// `ID3D11Device*` and `ID3D11DeviceContext*` from FFmpeg's
    /// `AVD3D11VADeviceContext`. Using FFmpeg's device ensures
    /// `CopySubresourceRegion` operates on same-device resources.
    ///
    /// # Safety
    /// The raw pointers must be valid COM interfaces from FFmpeg's
    /// hw_device_ctx. They are AddRef'd internally and remain valid
    /// for the lifetime of the pool.
    pub unsafe fn new(
        gpu: &GpuContext,
        d3d11_device_ptr: *mut c_void,
        d3d11_context_ptr: *mut c_void,
        width: u32,
        height: u32,
    ) -> Result<Self, D3d11InteropError> {
        if !gpu.is_dx12() {
            return Err(D3d11InteropError::NotDx12);
        }
        if d3d11_device_ptr.is_null() || d3d11_context_ptr.is_null() {
            return Err(D3d11InteropError::D3d11(
                "null D3D11 device/context from FFmpeg".into(),
            ));
        }

        // Wrap FFmpeg's raw COM pointers. from_raw takes ownership
        // (no AddRef), so we clone to AddRef for our own reference,
        // then forget the original to avoid double-Release.
        let device: ID3D11Device = {
            let raw = ID3D11Device::from_raw(d3d11_device_ptr);
            let cloned = raw.clone(); // AddRef
            std::mem::forget(raw); // don't Release FFmpeg's ref
            cloned
        };
        let context: ID3D11DeviceContext = {
            let raw = ID3D11DeviceContext::from_raw(d3d11_context_ptr);
            let cloned = raw.clone();
            std::mem::forget(raw);
            cloned
        };

        log::info!("D3D11 interop: using FFmpeg's D3D11 device ({d3d11_device_ptr:?})");

        // Create event query for staging copy synchronization.
        let query_desc = D3D11_QUERY_DESC {
            Query: D3D11_QUERY(0), // D3D11_QUERY_EVENT = 0
            MiscFlags: 0,
        };
        let event_query: ID3D11Query = {
            let mut query: Option<ID3D11Query> = None;
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
                let mut tex: Option<ID3D11Texture2D> = None;
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
            readback_staging: None,
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

    /// Read back NV12 data from a staging slot to CPU memory.
    ///
    /// Returns `(y_data, uv_data)` where `y_data` is the full-resolution
    /// luma plane and `uv_data` is the half-resolution interleaved chroma.
    /// Only called on detection frames (every N frames).
    pub fn readback_nv12(&mut self, slot: usize) -> Result<(Vec<u8>, Vec<u8>), D3d11InteropError> {
        if slot >= 4 {
            return Err(D3d11InteropError::StagingCopy(format!(
                "readback slot {slot} out of range"
            )));
        }

        unsafe {
            // Lazily create the CPU-readable staging texture.
            if self.readback_staging.is_none() {
                let desc = D3D11_TEXTURE2D_DESC {
                    Width: self.width,
                    Height: self.height,
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: DXGI_FORMAT_NV12,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Usage: D3D11_USAGE_STAGING,
                    BindFlags: 0,
                    CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                    MiscFlags: 0,
                };
                let mut tex: Option<ID3D11Texture2D> = None;
                self._device.CreateTexture2D(&desc, None, Some(&mut tex))?;
                self.readback_staging = Some(tex.ok_or_else(|| {
                    D3d11InteropError::D3d11("readback staging texture creation failed".into())
                })?);
                log::info!(
                    "D3D11VA readback staging texture created: {}x{} NV12",
                    self.width,
                    self.height
                );
            }

            let readback = self.readback_staging.as_ref().unwrap();

            // Copy from the shared staging slot to the CPU-readable staging texture.
            self.context.CopyResource(readback, &self.staging[slot]);
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

            // Map the readback texture for CPU read.
            let mut mapped = std::mem::zeroed();
            self.context
                .Map(readback, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;

            let row_pitch = mapped.RowPitch as usize;

            // NV12 layout: Y plane is height rows, UV plane is height/2 rows.
            // Row pitch may be larger than width due to alignment.
            let w = self.width as usize;
            let h = self.height as usize;

            // Copy Y plane (tightly packed, removing row padding).
            let mut y_data = vec![0u8; w * h];
            let src = mapped.pData as *const u8;
            for row in 0..h {
                std::ptr::copy_nonoverlapping(
                    src.add(row * row_pitch),
                    y_data.as_mut_ptr().add(row * w),
                    w,
                );
            }

            // UV plane starts at offset row_pitch * height in NV12 layout.
            let uv_h = h / 2;
            let mut uv_data = vec![0u8; w * uv_h];
            let uv_src = src.add(row_pitch * h);
            for row in 0..uv_h {
                std::ptr::copy_nonoverlapping(
                    uv_src.add(row * row_pitch),
                    uv_data.as_mut_ptr().add(row * w),
                    w,
                );
            }

            self.context.Unmap(readback, 0);

            Ok((y_data, uv_data))
        }
    }
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
    let d3d12_resource: ID3D12Resource = unsafe {
        let hal_device_guard = gpu
            .device()
            .as_hal::<Dx12>()
            .ok_or(D3d11InteropError::NotDx12)?;
        let raw_device = hal_device_guard.raw_device();
        let mut resource: Option<ID3D12Resource> = None;
        raw_device.OpenSharedHandle(handle, &mut resource)?;
        resource.ok_or_else(|| D3d11InteropError::D3d11("OpenSharedHandle returned None".into()))?
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

    // Wrap the HAL texture into a wgpu::Texture.
    let texture = unsafe {
        gpu.device().create_texture_from_hal::<Dx12>(
            hal_texture,
            &wgpu::TextureDescriptor {
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
            },
        )
    };

    Ok(texture)
}
