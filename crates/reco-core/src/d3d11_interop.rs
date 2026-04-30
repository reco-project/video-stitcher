//! D3D11VA to wgpu zero-copy interop via staging textures.
//!
//! Copies FFmpeg's D3D11VA decoded NV12 frames to shared staging textures,
//! then imports those into wgpu via DX12 shared handles. Eliminates the
//! GPU -> CPU -> GPU roundtrip (~50ms/frame on Surface-class hardware).
//!
//! ## Architecture (decode-thread staging)
//!
//! ```text
//! Decode thread (left/right):
//!   D3D11VA decode → CopySubresourceRegion → Flush + event query
//!   (staging copy is on the same thread as decode, zero context contention)
//!
//! Main thread:
//!   recv(slot_index) → render from pre-built wgpu NV12 plane views
//!   (no D3D11 work at all)
//!
//! Shared staging textures (created once at init):
//!   D3D11 NV12 (SHARED_NTHANDLE) → NT handle → DX12 OpenSharedHandle
//!   → wgpu Texture → Plane0 (R8Unorm Y) + Plane1 (Rg8Unorm UV)
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

/// Slots per camera for staging. Triple buffering ensures frame N never
/// overwrites a slot still being rendered by frame N-1.
pub const SLOTS_PER_CAMERA: usize = 3;
const TOTAL_SLOTS: usize = SLOTS_PER_CAMERA * 2;

/// D3D11-side staging copier for one camera. Lives in the decode thread.
///
/// Holds the D3D11 staging textures and event query. The decode thread
/// calls [`stage_frame`](Self::stage_frame) right after decoding, so the
/// event query only waits for this one decode+copy operation (no
/// cross-thread context contention).
pub struct D3d11StagingCopier {
    device: ID3D11Device,
    /// Shared D3D11 immediate context behind a mutex.
    /// Both decode threads share the same context (from FFmpeg's
    /// hw_device_ctx). D3D11 immediate contexts are NOT thread-safe -
    /// concurrent calls cause frame ordering corruption.
    context: std::sync::Arc<std::sync::Mutex<ID3D11DeviceContext>>,
    staging: Vec<ID3D11Texture2D>,
    event_query: ID3D11Query,
    /// CPU-readable staging texture for detection readback.
    readback_staging: Option<ID3D11Texture2D>,
    width: u32,
    height: u32,
    slot_count: usize,
}

unsafe impl Send for D3d11StagingCopier {}

/// wgpu-side views for all staging slots. Lives in the session.
///
/// Created once at init by importing the D3D11 staging textures via
/// shared NT handles. The session renders directly from these views
/// without any D3D11 work on the main thread.
pub struct D3d11WgpuViews {
    _wgpu_textures: Vec<wgpu::Texture>,
    y_views: Vec<wgpu::TextureView>,
    uv_views: Vec<wgpu::TextureView>,
}

/// Create the full staging infrastructure: copiers for decode threads
/// and wgpu views for the session.
///
/// Creates `SLOTS_PER_CAMERA` staging textures per camera on the given
/// D3D11 device, imports them into wgpu via DX12 shared handles, and
/// returns paired copiers + views.
///
/// # Safety
/// `d3d11_device_ptr` and `d3d11_context_ptr` must be valid COM
/// interfaces from FFmpeg's `AVD3D11VADeviceContext`.
pub unsafe fn create_staging_pair(
    gpu: &GpuContext,
    d3d11_device_ptr: *mut c_void,
    d3d11_context_ptr: *mut c_void,
    width: u32,
    height: u32,
) -> Result<(D3d11StagingCopier, D3d11StagingCopier, D3d11WgpuViews), D3d11InteropError> {
    if !gpu.is_dx12() {
        return Err(D3d11InteropError::NotDx12);
    }
    if d3d11_device_ptr.is_null() || d3d11_context_ptr.is_null() {
        return Err(D3d11InteropError::D3d11(
            "null D3D11 device/context from FFmpeg".into(),
        ));
    }

    let device: ID3D11Device = unsafe {
        let raw = ID3D11Device::from_raw(d3d11_device_ptr);
        let cloned = raw.clone();
        std::mem::forget(raw);
        cloned
    };
    let context: ID3D11DeviceContext = unsafe {
        let raw = ID3D11DeviceContext::from_raw(d3d11_context_ptr);
        let cloned = raw.clone();
        std::mem::forget(raw);
        cloned
    };

    log::info!("D3D11 interop: using FFmpeg's D3D11 device ({d3d11_device_ptr:?})");

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
        MiscFlags: (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0) as u32,
    };

    let mut left_staging: Vec<ID3D11Texture2D> = Vec::with_capacity(SLOTS_PER_CAMERA);
    let mut right_staging: Vec<ID3D11Texture2D> = Vec::with_capacity(SLOTS_PER_CAMERA);
    let mut wgpu_textures: Vec<wgpu::Texture> = Vec::with_capacity(TOTAL_SLOTS);
    let mut y_views: Vec<wgpu::TextureView> = Vec::with_capacity(TOTAL_SLOTS);
    let mut uv_views: Vec<wgpu::TextureView> = Vec::with_capacity(TOTAL_SLOTS);

    for i in 0..TOTAL_SLOTS {
        let staging: ID3D11Texture2D = unsafe {
            let mut tex: Option<ID3D11Texture2D> = None;
            device.CreateTexture2D(&desc, None, Some(&mut tex))?;
            tex.ok_or_else(|| D3d11InteropError::D3d11("CreateTexture2D returned None".into()))?
        };

        let dxgi_resource: IDXGIResource1 = staging.cast()?;
        let handle: HANDLE =
            unsafe { dxgi_resource.CreateSharedHandle(None, GENERIC_ALL.0, None)? };
        let wgpu_texture = unsafe { import_d3d11_shared_handle(gpu, handle, width, height)? };
        unsafe {
            let _ = CloseHandle(handle);
        }

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

        if i < SLOTS_PER_CAMERA {
            left_staging.push(staging);
        } else {
            right_staging.push(staging);
        }
        wgpu_textures.push(wgpu_texture);
        y_views.push(y_view);
        uv_views.push(uv_view);
    }

    let query_desc = D3D11_QUERY_DESC {
        Query: D3D11_QUERY(0),
        MiscFlags: 0,
    };
    let left_query: ID3D11Query = unsafe {
        let mut q: Option<ID3D11Query> = None;
        device.CreateQuery(&query_desc, Some(&mut q))?;
        q.ok_or_else(|| D3d11InteropError::D3d11("CreateQuery returned None".into()))?
    };
    let right_query: ID3D11Query = unsafe {
        let mut q: Option<ID3D11Query> = None;
        device.CreateQuery(&query_desc, Some(&mut q))?;
        q.ok_or_else(|| D3d11InteropError::D3d11("CreateQuery returned None".into()))?
    };

    log::info!(
        "D3D11VA staging pool ready: {}x{} NV12, {} slots ({} per camera), decode-thread staging",
        width,
        height,
        TOTAL_SLOTS,
        SLOTS_PER_CAMERA
    );

    let shared_context = std::sync::Arc::new(std::sync::Mutex::new(context));

    let left_copier = D3d11StagingCopier {
        device: device.clone(),
        context: std::sync::Arc::clone(&shared_context),
        staging: left_staging,
        event_query: left_query,
        readback_staging: None,
        width,
        height,
        slot_count: SLOTS_PER_CAMERA,
    };

    let right_copier = D3d11StagingCopier {
        device,
        context: shared_context,
        staging: right_staging,
        event_query: right_query,
        readback_staging: None,
        width,
        height,
        slot_count: SLOTS_PER_CAMERA,
    };

    let views = D3d11WgpuViews {
        _wgpu_textures: wgpu_textures,
        y_views,
        uv_views,
    };

    Ok((left_copier, right_copier, views))
}

impl D3d11StagingCopier {
    /// Copy a D3D11VA decoded frame to a staging slot.
    ///
    /// Called by the decode thread right after decoding. The event query
    /// only waits for this one copy (no cross-thread contention).
    pub fn stage_frame(
        &self,
        src_texture: *mut c_void,
        array_slice: usize,
        slot: usize,
    ) -> Result<(), D3d11InteropError> {
        if src_texture.is_null() {
            return Err(D3d11InteropError::StagingCopy("null source texture".into()));
        }
        if slot >= self.slot_count {
            return Err(D3d11InteropError::StagingCopy(format!(
                "slot {slot} out of range (0..{})",
                self.slot_count
            )));
        }

        unsafe {
            let unknown: windows::core::IUnknown = windows::core::IUnknown::from_raw(src_texture);
            let src: ID3D11Texture2D = unknown.cast()?;
            std::mem::forget(unknown);

            let ctx = self.context.lock().unwrap();
            ctx.CopySubresourceRegion(
                &self.staging[slot],
                0,
                0,
                0,
                0,
                &src,
                array_slice as u32,
                None,
            );
            ctx.Flush();

            ctx.End(&self.event_query);
            loop {
                let mut done: u32 = 0;
                let hr = ctx.GetData(
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

    /// Read back NV12 data from a staging slot to CPU memory.
    ///
    /// Returns `(y_data, uv_data)`. Only called on detection frames.
    pub fn readback_nv12(&mut self, slot: usize) -> Result<(Vec<u8>, Vec<u8>), D3d11InteropError> {
        if slot >= self.slot_count {
            return Err(D3d11InteropError::StagingCopy(format!(
                "readback slot {slot} out of range (0..{})",
                self.slot_count
            )));
        }

        unsafe {
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
                self.device.CreateTexture2D(&desc, None, Some(&mut tex))?;
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

            let ctx = self.context.lock().unwrap();
            ctx.CopyResource(readback, &self.staging[slot]);
            ctx.Flush();
            ctx.End(&self.event_query);
            loop {
                let mut done: u32 = 0;
                let hr = ctx.GetData(
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

            let mut mapped = std::mem::zeroed();
            ctx.Map(readback, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;

            let row_pitch = mapped.RowPitch as usize;
            let w = self.width as usize;
            let h = self.height as usize;

            let mut y_data = vec![0u8; w * h];
            let src = mapped.pData as *const u8;
            for row in 0..h {
                std::ptr::copy_nonoverlapping(
                    src.add(row * row_pitch),
                    y_data.as_mut_ptr().add(row * w),
                    w,
                );
            }

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

            ctx.Unmap(readback, 0);

            Ok((y_data, uv_data))
        }
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

impl D3d11WgpuViews {
    /// Y plane view (R8Unorm, full resolution) for the given slot.
    ///
    /// Slot layout: [left_0, left_1, left_2, right_0, right_1, right_2]
    pub fn y_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.y_views[slot]
    }

    /// UV plane view (Rg8Unorm, half resolution) for the given slot.
    pub fn uv_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.uv_views[slot]
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
            1,
            1,
        )
    };

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
