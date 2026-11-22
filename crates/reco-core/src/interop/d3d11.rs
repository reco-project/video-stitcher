//! D3D11VA to wgpu zero-copy interop via staging textures.
//!
//! Copies FFmpeg's D3D11VA decoded NV12 frames to shared staging textures,
//! then imports those into wgpu via DX12 shared handles. Eliminates the
//! GPU -> CPU -> GPU roundtrip.
//!
//! ## Architecture
//!
//! ```text
//! D3D11VA decode pool (FFmpeg-owned)
//!        |  CopySubresourceRegion
//! D3D11 staging texture (SHARED_NTHANDLE, on FFmpeg's D3D11 device)
//!        |  NT handle -> DX12 OpenSharedHandle
//! wgpu NV12 texture -> Plane0 (R8Unorm Y) + Plane1 (Rg8Unorm UV)
//!        |
//! stitch render pass
//! ```
//!
//! The staging pool is initialized lazily on the first `stage_frames` call.
//! It extracts the D3D11 device from the source texture via `GetDevice`,
//! ensuring staging textures live on the same device as FFmpeg's decode
//! pool. This avoids cross-device copies which hang on NVIDIA GPUs.

use crate::gpu::GpuContext;
use std::ffi::c_void;
use thiserror::Error;

use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, S_OK};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_QUERY, D3D11_QUERY_DESC, D3D11_RESOURCE_MISC_SHARED,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11Device,
    ID3D11DeviceContext, ID3D11Multithread, ID3D11Query, ID3D11Texture2D,
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
/// Created on the first `stage_frames` call using the device obtained from
/// the source texture, so all D3D11 operations stay on a single device.
struct StagingState {
    /// FFmpeg's D3D11 device that owns the decode pool and the staging
    /// textures. `stage_frames` rejects source frames from any other device:
    /// a cross-device `CopySubresourceRegion` faults the GPU.
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    staging: Vec<ID3D11Texture2D>,
    _wgpu_textures: Vec<wgpu::Texture>,
    y_views: Vec<wgpu::TextureView>,
    uv_views: Vec<wgpu::TextureView>,
    event_query: ID3D11Query,
}

/// NV12/P010 staging pool for D3D11VA -> wgpu zero-copy.
///
/// Each frame slot holds one texture per camera, at that camera's extent.
/// Textures are imported into wgpu with pre-built Y and UV plane views.
///
/// The pool is constructed lightweight; the D3D11 device and staging
/// textures are created lazily on the first `stage_frames` call by
/// extracting FFmpeg's own device from the source texture. This ensures
/// `CopySubresourceRegion` never crosses device boundaries, which would
/// hang on NVIDIA hardware.
pub struct D3d11StagingPool {
    /// wgpu device, needed for DX12 shared handle import during lazy init.
    wgpu_device: wgpu::Device,
    source_sizes: Vec<(u32, u32)>,
    /// Number of complete frame-set slots.
    n_slots: usize,
    /// Pixel format (NV12 8-bit or P010 10-bit).
    pixel_format: crate::render::renderer::GpuPixelFormat,
    /// Lazily initialized on first `stage_frames` call.
    state: Option<StagingState>,
}

impl D3d11StagingPool {
    /// Create a staging pool that will bind to FFmpeg's D3D11 device lazily.
    ///
    /// Validates the backend, dimensions and capacity without allocating textures.
    /// The actual D3D11 device, staging textures, and wgpu imports are
    /// created on the first `stage_frames` call.
    pub fn new(
        gpu: &GpuContext,
        source_sizes: &[(u32, u32)],
        n_slots: usize,
        pixel_format: crate::render::renderer::GpuPixelFormat,
    ) -> Result<Self, D3d11InteropError> {
        if !gpu.is_dx12() {
            return Err(D3d11InteropError::NotDx12);
        }

        if source_sizes.is_empty()
            || source_sizes
                .iter()
                .any(|&(w, h)| w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0)
            || n_slots == 0
            || n_slots.checked_mul(source_sizes.len()).is_none()
        {
            return Err(D3d11InteropError::StagingCopy(
                "invalid staging dimensions or capacity".into(),
            ));
        }
        crate::lookahead::image_bytes(source_sizes, pixel_format.bytes_per_sample())
            .and_then(|bytes| {
                bytes
                    .checked_mul(n_slots)
                    .ok_or_else(|| "D3D11 staging capacity overflow".into())
            })
            .map_err(D3d11InteropError::StagingCopy)?;

        Ok(Self {
            wgpu_device: gpu.device().clone(),
            source_sizes: source_sizes.to_vec(),
            n_slots,
            pixel_format,
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
        // Returns the previous state; we don't need it.
        let _ = unsafe { multithread.SetMultithreadProtected(true) };

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

        // Create staging textures with shared handles on FFmpeg's device.
        let dxgi_format = match self.pixel_format {
            crate::render::renderer::GpuPixelFormat::P010 => DXGI_FORMAT_P010,
            _ => DXGI_FORMAT_NV12,
        };
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 0,
            Height: 0,
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

        let n = self.n_slots * self.source_sizes.len();
        let mut staging_textures: Vec<ID3D11Texture2D> = Vec::with_capacity(n);
        let mut wgpu_textures: Vec<wgpu::Texture> = Vec::with_capacity(n);
        let mut y_views: Vec<wgpu::TextureView> = Vec::with_capacity(n);
        let mut uv_views: Vec<wgpu::TextureView> = Vec::with_capacity(n);

        let bytes_per_frame =
            crate::lookahead::image_bytes(&self.source_sizes, self.pixel_format.bytes_per_sample())
                .map_err(D3d11InteropError::StagingCopy)?;
        let total_mb = (bytes_per_frame * self.n_slots) as f64 / (1024.0 * 1024.0);

        for i in 0..n {
            let (width, height) = self.source_sizes[i % self.source_sizes.len()];
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                ..desc
            };
            let per_slot_mb =
                width as f64 * height as f64 * 1.5 * self.pixel_format.bytes_per_sample() as f64
                    / (1024.0 * 1024.0);
            let staging: ID3D11Texture2D = unsafe {
                let mut tex = None;
                device
                    .CreateTexture2D(&desc, None, Some(&mut tex))
                    .map_err(|e| {
                        D3d11InteropError::D3d11(format!(
                            "staging VRAM allocation failed at slot {i}/{n} \
                             (~{per_slot_mb:.0} MB/slot, ~{total_mb:.0} MB total). \
                             Reduce --lookahead or use --no-zero-copy. ({e})"
                        ))
                    })?;
                tex.ok_or_else(|| {
                    D3d11InteropError::D3d11(format!(
                        "CreateTexture2D returned None at slot {i}/{n}"
                    ))
                })?
            };

            // The NT handle is needed only during DX12 import.
            let dxgi_resource: IDXGIResource1 = staging.cast()?;
            let handle: HANDLE =
                unsafe { dxgi_resource.CreateSharedHandle(None, GENERIC_ALL.0, None)? };

            // Import into wgpu via DX12 HAL. create_texture_from_hal can panic
            // through wgpu's uncaptured-error handler on a bad/post-OOM device
            // state, so guard it with catch_unwind (mirrors VramPool::new).
            let import_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                import_d3d11_shared_handle(
                    &self.wgpu_device,
                    handle,
                    width,
                    height,
                    self.pixel_format.wgpu_format(),
                )
            }));
            unsafe {
                let _ = CloseHandle(handle);
            }
            let wgpu_texture = match import_result {
                Ok(Ok(tex)) => tex,
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(D3d11InteropError::D3d11(format!(
                        "staging texture import (wgpu) failed at slot {i}/{n} \
                         (~{per_slot_mb:.0} MB/slot, ~{total_mb:.0} MB total). \
                         Reduce --lookahead or use --no-zero-copy."
                    )));
                }
            };

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
            "D3D11VA staging pool ready: {:?} {:?}, {} frame slots (same-device)",
            self.source_sizes,
            self.pixel_format,
            self.n_slots
        );

        self.state = Some(StagingState {
            device,
            context,
            staging: staging_textures,
            _wgpu_textures: wgpu_textures,
            y_views,
            uv_views,
            event_query,
        });

        Ok(())
    }

    /// Copy one complete frame set and await GPU completion before releasing its sources.
    ///
    /// All sources must belong to the same decode device. Source textures may be
    /// decoder-aligned, but only each camera's configured image extent is copied.
    ///
    /// # Safety
    /// Each non-null texture must be a live ID3D11Texture2D COM pointer retained
    /// by the caller until this call returns.
    pub unsafe fn stage_frames(
        &mut self,
        frames: &[crate::source::D3d11CameraFrame],
        slot: usize,
    ) -> Result<(), D3d11InteropError> {
        if slot >= self.n_slots || frames.len() != self.source_sizes.len() {
            return Err(D3d11InteropError::StagingCopy(
                "invalid frame slot or camera count".into(),
            ));
        }
        if frames.iter().any(|frame| frame.texture.is_null()) {
            return Err(D3d11InteropError::StagingCopy("null source texture".into()));
        }
        unsafe {
            let sources: Vec<ID3D11Texture2D> = frames
                .iter()
                .map(|frame| {
                    // Borrow FFmpeg's reference; cast obtains our own AddRef.
                    let unknown = std::mem::ManuallyDrop::new(windows::core::IUnknown::from_raw(
                        frame.texture,
                    ));
                    unknown.cast()
                })
                .collect::<windows::core::Result<_>>()?;
            let device = if let Some(state) = &self.state {
                state.device.clone()
            } else {
                sources[0].GetDevice()?
            };
            // Validate the entire batch before allocating or issuing any GPU copies.
            for (camera, (src, frame)) in sources.iter().zip(frames).enumerate() {
                if Interface::as_raw(&src.GetDevice()?) != Interface::as_raw(&device) {
                    return Err(D3d11InteropError::StagingCopy(
                        "source frame is on a different D3D11 device than the staging pool; the decoders must share one hw device".into()
                    ));
                }
                let mut desc = D3D11_TEXTURE2D_DESC::default();
                src.GetDesc(&mut desc);
                validate_source(
                    &desc,
                    frame.slice,
                    self.source_sizes[camera],
                    self.pixel_format,
                )?;
            }
            self.ensure_initialized(&sources[0])?;
            let state = self.state.as_ref().unwrap();
            for (camera, (src, frame)) in sources.iter().zip(frames).enumerate() {
                let (width, height) = self.source_sizes[camera];
                let region = D3D11_BOX {
                    left: 0,
                    top: 0,
                    front: 0,
                    right: width,
                    bottom: height,
                    back: 1,
                };
                state.context.CopySubresourceRegion(
                    &state.staging[slot * frames.len() + camera],
                    0,
                    0,
                    0,
                    0,
                    src,
                    frame.slice as u32,
                    Some(&region),
                );
            }
            state.context.End(&state.event_query);
            state.context.Flush();
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

    /// Plane views for one staged frame set, in camera order.
    pub(crate) fn views(&self, slot: usize) -> Option<Vec<(wgpu::TextureView, wgpu::TextureView)>> {
        let state = self.state.as_ref()?;
        if slot >= self.n_slots {
            return None;
        }
        let first = slot * self.source_sizes.len();
        Some(
            (first..first + self.source_sizes.len())
                .map(|i| (state.y_views[i].clone(), state.uv_views[i].clone()))
                .collect(),
        )
    }

    /// Frame slot assigned round-robin after reducing the full-width sequence number.
    pub(crate) fn slot(&self, sequence: u64) -> usize {
        (sequence % self.n_slots as u64) as usize
    }

    pub(crate) fn matches(
        &self,
        sizes: &[(u32, u32)],
        slots: usize,
        format: crate::render::renderer::GpuPixelFormat,
    ) -> bool {
        self.source_sizes == sizes && self.n_slots == slots && self.pixel_format == format
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

fn validate_source(
    desc: &D3D11_TEXTURE2D_DESC,
    slice: usize,
    (width, height): (u32, u32),
    format: crate::render::renderer::GpuPixelFormat,
) -> Result<(), D3d11InteropError> {
    let expected_format = match format {
        crate::render::renderer::GpuPixelFormat::P010 => DXGI_FORMAT_P010,
        _ => DXGI_FORMAT_NV12,
    };
    if desc.Width < width
        || desc.Height < height
        || desc.Format != expected_format
        || desc.MipLevels != 1
        || slice >= desc.ArraySize as usize
        || desc.SampleDesc.Count != 1
        || desc.SampleDesc.Quality != 0
    {
        return Err(D3d11InteropError::StagingCopy(
            "source texture extent, format, array slice or sample layout does not match its camera"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::renderer::GpuPixelFormat;

    #[test]
    #[ignore = "requires a DX12 adapter"]
    fn frame_slot_layout_and_invalid_batches_do_not_allocate_native_resources() {
        let gpu = GpuContext::new_blocking().unwrap();
        for sizes in [
            vec![(1920, 1080)],
            vec![(1920, 1080), (1280, 720)],
            vec![(1280, 720), (1920, 1080)],
        ] {
            for format in [GpuPixelFormat::Nv12, GpuPixelFormat::P010] {
                for capacity in [2, 7, 16] {
                    let mut pool = D3d11StagingPool::new(&gpu, &sizes, capacity, format).unwrap();
                    for sequence in [0, 1, 2, 17, u64::MAX] {
                        assert_eq!(pool.slot(sequence), (sequence % capacity as u64) as usize);
                    }
                    assert!(pool.matches(&sizes, capacity, format));
                    assert!(!pool.matches(&sizes, capacity + 1, format));
                    assert!(!pool.matches(&[(640, 480)], capacity, format));
                    assert!(!pool.matches(
                        &sizes,
                        capacity,
                        if format == GpuPixelFormat::Nv12 {
                            GpuPixelFormat::P010
                        } else {
                            GpuPixelFormat::Nv12
                        }
                    ));
                    let null_frames: Vec<_> = sizes
                        .iter()
                        .map(|_| unsafe {
                            crate::source::D3d11CameraFrame::from_raw(std::ptr::null_mut(), 0, ())
                        })
                        .collect();
                    // Null pointers are rejected before attempting to borrow a COM reference.
                    unsafe {
                        assert!(pool.stage_frames(&[], 0).is_err());
                        assert!(pool.stage_frames(&null_frames, capacity).is_err());
                        assert!(pool.stage_frames(&null_frames, 0).is_err());
                    }
                    assert!(pool.state.is_none());
                    assert!(pool.views(0).is_none());
                }
            }
        }
        for sizes in [vec![], vec![(0, 1080)], vec![(1920, 1079)]] {
            assert!(D3d11StagingPool::new(&gpu, &sizes, 2, GpuPixelFormat::Nv12).is_err());
        }
        for capacity in [0, usize::MAX] {
            assert!(
                D3d11StagingPool::new(&gpu, &[(1920, 1080); 2], capacity, GpuPixelFormat::P010)
                    .is_err()
            );
        }
    }

    #[test]
    fn copy_region_accepts_decoder_padding_but_rejects_invalid_subresources() {
        for (format, dxgi) in [
            (GpuPixelFormat::Nv12, DXGI_FORMAT_NV12),
            (GpuPixelFormat::P010, DXGI_FORMAT_P010),
        ] {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: 1920,
                Height: 1088,
                MipLevels: 1,
                ArraySize: 8,
                Format: dxgi,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            assert!(validate_source(&desc, 7, (1920, 1080), format).is_ok());
            assert!(validate_source(&desc, 0, (1280, 720), format).is_ok());
            for slice in [8, usize::MAX] {
                assert!(validate_source(&desc, slice, (1920, 1080), format).is_err());
            }
            for bad in [
                D3D11_TEXTURE2D_DESC {
                    Width: 1918,
                    ..desc
                },
                D3D11_TEXTURE2D_DESC {
                    Height: 1078,
                    ..desc
                },
                D3D11_TEXTURE2D_DESC {
                    MipLevels: 2,
                    ..desc
                },
                D3D11_TEXTURE2D_DESC {
                    ArraySize: 0,
                    ..desc
                },
                D3D11_TEXTURE2D_DESC {
                    Format: if dxgi == DXGI_FORMAT_NV12 {
                        DXGI_FORMAT_P010
                    } else {
                        DXGI_FORMAT_NV12
                    },
                    ..desc
                },
                D3D11_TEXTURE2D_DESC {
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 2,
                        Quality: 0,
                    },
                    ..desc
                },
            ] {
                assert!(validate_source(&bad, 0, (1920, 1080), format).is_err());
            }
        }
    }
}
