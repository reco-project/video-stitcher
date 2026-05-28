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
//! D3D11 staging texture (SHARED_NTHANDLE, on FFmpeg's D3D11 device)
//!        |  NT handle -> DX12 OpenSharedHandle
//! wgpu NV12 texture -> Plane0 (R8Unorm Y) + Plane1 (Rg8Unorm UV)
//!        |
//! stitch render pass
//! ```
//!
//! The staging pool is initialized lazily on the first `stage_frame` call.
//! It extracts the D3D11 device from the source texture via `GetDevice`,
//! ensuring staging textures live on the same device as FFmpeg's decode
//! pool. This avoids cross-device copies which hang on NVIDIA GPUs.

use crate::gpu::GpuContext;
use std::ffi::c_void;
use thiserror::Error;

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, S_OK};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_QUERY, D3D11_QUERY_DESC, D3D11_RESOURCE_MISC_SHARED, D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11DeviceContext, ID3D11Multithread, ID3D11Query,
    ID3D11Texture2D,
};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_SAMPLE_DESC,
};
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

impl From<D3d11InteropError> for crate::session::types::SessionError {
    fn from(e: D3d11InteropError) -> Self {
        crate::session::types::SessionError::ZeroCopy(e.to_string())
    }
}

/// Lazily-initialized D3D11 staging resources.
///
/// Created on the first `stage_frame` call using the device obtained from
/// the source texture, so all D3D11 operations stay on a single device.
struct StagingState {
    context: ID3D11DeviceContext,
    staging: Vec<ID3D11Texture2D>,
    _wgpu_textures: Vec<wgpu::Texture>,
    y_views: Vec<wgpu::TextureView>,
    uv_views: Vec<wgpu::TextureView>,
    event_query: ID3D11Query,
    cuda_nv12: Option<Vec<crate::interop::cuda::CudaImportedNv12>>,
}

/// Double-buffered NV12 staging pool for D3D11VA -> wgpu zero-copy.
///
/// Allocates 4 staging textures (2 per camera, ping-pong) with shared
/// handles. Each staging texture is imported into wgpu as an NV12 texture
/// with pre-built Y and UV plane views.
///
/// The pool is constructed lightweight; the D3D11 device and staging
/// textures are created lazily on the first `stage_frame` call by
/// extracting FFmpeg's own device from the source texture. This ensures
/// `CopySubresourceRegion` never crosses device boundaries, which would
/// hang on NVIDIA hardware.
pub struct D3d11StagingPool {
    /// wgpu device, needed for DX12 shared handle import during lazy init.
    wgpu_device: wgpu::Device,
    width: u32,
    height: u32,
    /// Number of staging slots.
    n_slots: usize,
    /// Pixel format (NV12 8-bit or P010 10-bit).
    pixel_format: crate::render::renderer::GpuPixelFormat,
    /// Import D3D11 textures into CUDA for GPU-resident detection.
    enable_cuda: bool,
    /// Lazily initialized on first `stage_frame` call.
    state: Option<StagingState>,
}

impl D3d11StagingPool {
    /// Create a staging pool that will bind to FFmpeg's D3D11 device lazily.
    ///
    /// This only validates the wgpu backend and stores the dimensions.
    /// The actual D3D11 device, staging textures, and wgpu imports are
    /// created on the first `stage_frame` call.
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        n_slots: usize,
        enable_cuda: bool,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<Self, D3d11InteropError> {
        if !gpu.is_dx12() {
            return Err(D3d11InteropError::NotDx12);
        }

        Ok(Self {
            wgpu_device: gpu.device().clone(),
            width,
            height,
            n_slots,
            pixel_format,
            enable_cuda,
            state: None,
        })
    }

    /// Initialize staging resources using FFmpeg's D3D11 device.
    ///
    /// Extracts the device from `src_texture` via `ID3D11DeviceChild::GetDevice`,
    /// then creates staging textures and wgpu imports on that device.
    fn ensure_initialized(
        &mut self,
        src_texture: &ID3D11Texture2D,
    ) -> Result<(), D3d11InteropError> {
        if self.state.is_some() {
            return Ok(());
        }

        // Get FFmpeg's D3D11 device from the source texture.
        //
        // ID3D11Texture2D derefs to ID3D11Resource derefs to ID3D11DeviceChild,
        // so GetDevice is callable directly. It returns the device that created
        // this texture, which is FFmpeg's D3D11VA device. Creating staging
        // textures on this same device ensures CopySubresourceRegion stays
        // within one device.
        let device = unsafe { src_texture.GetDevice()? };

        // Enable multithread protection on FFmpeg's device. FFmpeg's decode
        // threads and our staging thread share the immediate context. D3D11
        // immediate contexts are single-threaded by default; this adds an
        // internal critical section that serializes access.
        let multithread: ID3D11Multithread = device.cast()?;
        unsafe { multithread.SetMultithreadProtected(true) };

        let context = unsafe { device.GetImmediateContext()? };

        log::info!("D3D11VA staging pool: using FFmpeg's device with multithread protection");

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

        // Create 4 staging textures with shared handles on FFmpeg's device.
        let dxgi_format = match self.pixel_format {
            crate::render::renderer::GpuPixelFormat::P010 => DXGI_FORMAT_P010,
            _ => DXGI_FORMAT_NV12,
        };
        let desc = D3D11_TEXTURE2D_DESC {
            Width: self.width,
            Height: self.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: dxgi_format,
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

        let n = self.n_slots;
        let mut staging_textures: Vec<ID3D11Texture2D> = Vec::with_capacity(n);
        let mut wgpu_textures: Vec<wgpu::Texture> = Vec::with_capacity(n);
        let mut y_views: Vec<wgpu::TextureView> = Vec::with_capacity(n);
        let mut uv_views: Vec<wgpu::TextureView> = Vec::with_capacity(n);
        let mut cuda_imports: Vec<Option<crate::interop::cuda::CudaImportedNv12>> =
            Vec::with_capacity(n);

        for i in 0..n {
            let staging: ID3D11Texture2D = unsafe {
                let mut tex = None;
                device.CreateTexture2D(&desc, None, Some(&mut tex))?;
                tex.ok_or_else(|| D3d11InteropError::D3d11("CreateTexture2D returned None".into()))?
            };

            // Get shared NT handle (kept open for both DX12 and CUDA import).
            let dxgi_resource: IDXGIResource1 = staging.cast()?;
            let handle: HANDLE =
                unsafe { dxgi_resource.CreateSharedHandle(None, GENERIC_ALL.0, None)? };

            // Import into wgpu via DX12 HAL.
            let wgpu_texture = unsafe {
                import_d3d11_shared_handle(
                    &self.wgpu_device,
                    handle,
                    self.width,
                    self.height,
                    self.pixel_format.wgpu_format(),
                )?
            };

            // Import into CUDA for AI detection (NVIDIA only, non-fatal).
            let nv12_pitch = self.width as usize;
            let cuda_import = crate::interop::cuda::cuda_import_d3d11_nv12(
                handle.0,
                self.width,
                self.height,
                nv12_pitch,
            );
            match &cuda_import {
                Ok(m) => log::debug!(
                    "CUDA imported staging slot {i}: Y=0x{:x} UV=0x{:x}",
                    m.y_ptr,
                    m.uv_ptr
                ),
                Err(e) => log::debug!("CUDA import for slot {i} skipped: {e}"),
            }
            cuda_imports.push(cuda_import.ok());

            unsafe {
                let _ = CloseHandle(handle);
            }

            // Create Y/UV plane views (R8/Rg8 for NV12, R16/Rg16 for P010).
            let y_view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("d3d11_y_{i}")),
                format: Some(self.pixel_format.y_format()),
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
                format: Some(self.pixel_format.uv_format()),
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
            "D3D11VA staging pool ready: {}x{} {:?}, {n} slots (same-device)",
            self.width,
            self.height,
            self.pixel_format
        );

        let cuda_nv12 = if self.enable_cuda && cuda_imports.iter().all(|c| c.is_some()) {
            let v: Vec<_> = cuda_imports.into_iter().map(|c| c.unwrap()).collect();
            log::info!("CUDA detection enabled on D3D11VA staging textures");
            Some(v)
        } else {
            log::info!(
                "CUDA detection not available on D3D11VA staging (non-NVIDIA or CUDA not found)"
            );
            None
        };

        self.state = Some(StagingState {
            context,
            staging: staging_textures,
            _wgpu_textures: wgpu_textures,
            y_views,
            uv_views,
            cuda_nv12,
            event_query,
        });

        Ok(())
    }

    /// Copy a D3D11VA decoded frame from the decode pool to a staging slot.
    ///
    /// On the first call, extracts the D3D11 device from the source texture
    /// and creates all staging resources on that device. Subsequent calls
    /// reuse the initialized state.
    ///
    /// Performs `CopySubresourceRegion` (GPU-to-GPU, ~0.2ms) then waits
    /// for completion via an event query before returning.
    pub fn stage_frame(
        &mut self,
        src_texture: *mut c_void,
        array_slice: usize,
        slot: usize,
    ) -> Result<(), D3d11InteropError> {
        if src_texture.is_null() {
            return Err(D3d11InteropError::StagingCopy("null source texture".into()));
        }
        if slot >= self.n_slots {
            return Err(D3d11InteropError::StagingCopy(format!(
                "slot {slot} out of range (0..{})",
                self.n_slots
            )));
        }

        unsafe {
            // Reconstruct the ID3D11Texture2D from the raw pointer.
            // SAFETY: FFmpeg guarantees data[0] is a valid ID3D11Texture2D*
            // for the lifetime of the decoded frame. We AddRef via clone
            // to get our own reference, then release at scope end.
            let unknown: windows::core::IUnknown = windows::core::IUnknown::from_raw(src_texture);
            let src: ID3D11Texture2D = unknown.cast()?;
            std::mem::forget(unknown);

            self.ensure_initialized(&src)?;
            let state = self.state.as_ref().unwrap();

            let src_subresource = array_slice as u32;

            state.context.CopySubresourceRegion(
                &state.staging[slot],
                0,
                0,
                0,
                0,
                &src,
                src_subresource,
                None,
            );

            state.context.Flush();
            state.context.End(&state.event_query);
            let mut spins: u32 = 0;
            loop {
                let mut done: i32 = 0;
                let hr = (Interface::vtable(&state.context).GetData)(
                    Interface::as_raw(&state.context),
                    Interface::as_raw(&state.event_query),
                    &mut done as *mut i32 as *mut c_void,
                    std::mem::size_of::<i32>() as u32,
                    0,
                );
                if hr == S_OK {
                    break;
                }
                if hr.is_err() {
                    return Err(D3d11InteropError::StagingCopy(format!(
                        "GetData failed after {spins} polls: HRESULT {:#x}",
                        hr.0
                    )));
                }
                // S_FALSE: not ready yet
                spins += 1;
                if spins > 1_000_000 {
                    return Err(D3d11InteropError::StagingCopy(
                        "GetData timed out (>1M polls)".into(),
                    ));
                }
                std::hint::spin_loop();
            }
        }

        Ok(())
    }

    /// Y plane view (R8Unorm, full resolution) for the given slot.
    ///
    /// # Panics
    /// Panics if called before the first `stage_frame` (pool not initialized).
    pub fn y_view(&self, slot: usize) -> &wgpu::TextureView {
        &self
            .state
            .as_ref()
            .expect("staging pool not initialized")
            .y_views[slot]
    }

    /// UV plane view (Rg8Unorm, half resolution) for the given slot.
    ///
    /// # Panics
    /// Panics if called before the first `stage_frame` (pool not initialized).
    pub fn uv_view(&self, slot: usize) -> &wgpu::TextureView {
        &self
            .state
            .as_ref()
            .expect("staging pool not initialized")
            .uv_views[slot]
    }

    /// Number of staging slots.
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    /// CUDA NV12 pointers for the given slot, if CUDA import succeeded.
    ///
    /// Returns `(y_ptr, uv_ptr, pitch)` for feeding to the GPU detection
    /// pipeline. None on non-NVIDIA hardware or if CUDA is unavailable.
    pub fn cuda_nv12_ptrs(
        &self,
        slot: usize,
    ) -> Option<(
        crate::interop::cuda::CUdeviceptr,
        crate::interop::cuda::CUdeviceptr,
        usize,
    )> {
        let state = self.state.as_ref()?;
        let imports = state.cuda_nv12.as_ref()?;
        let m = &imports[slot];
        Some((m.y_ptr, m.uv_ptr, m.pitch))
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

/// Import a D3D11 shared handle into wgpu as an NV12 texture.
///
/// # Safety
/// `handle` must be a valid NT shared handle from `CreateSharedHandle`.
unsafe fn import_d3d11_shared_handle(
    device: &wgpu::Device,
    handle: HANDLE,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture, D3d11InteropError> {
    use wgpu::hal::api::Dx12;

    // Open the shared handle as a D3D12 resource.
    let d3d12_resource: ID3D12Resource = {
        let hal_device_guard =
            unsafe { device.as_hal::<Dx12>().ok_or(D3d11InteropError::NotDx12)? };
        let raw_device = hal_device_guard.raw_device();
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            raw_device.OpenSharedHandle(handle, &mut resource)?;
        }
        resource.ok_or_else(|| D3d11InteropError::D3d11("OpenSharedHandle returned None".into()))?
    };

    // Wrap the D3D12 resource in a wgpu HAL texture.
    let hal_texture = unsafe {
        wgpu::hal::dx12::Device::texture_from_raw(
            d3d12_resource,
            format,
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
        label: Some("d3d11_staging"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    // Wrap the HAL texture into a wgpu::Texture.
    let texture = unsafe { device.create_texture_from_hal::<Dx12>(hal_texture, &desc) };

    Ok(texture)
}
