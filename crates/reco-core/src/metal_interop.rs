//! Metal/VideoToolbox interop for macOS zero-copy decode.
//!
//! Imports VideoToolbox-decoded `CVPixelBuffer`s into wgpu textures via
//! `CVMetalTextureCache`, avoiding CPU copies. The flow:
//!
//! 1. VideoToolbox decodes H.264 -> `CVPixelBuffer` (IOSurface-backed)
//! 2. `CVMetalTextureCacheCreateTextureFromImage` maps each NV12 plane to an `MTLTexture`
//! 3. `wgpu::Device::create_texture_from_hal::<Metal>()` wraps it as a `wgpu::Texture`
//!
//! ## References
//! - Gyroflow `zero_copy.rs` and `wgpu_interop_metal.rs`
//! - Apple CoreVideo `CVMetalTextureCache` documentation
//! - wgpu HAL interop API (`texture_from_raw`, `create_texture_from_hal`)

use std::ffi::c_void;

use crate::gpu::GpuContext;

// ---------------------------------------------------------------------------
// CoreVideo / CoreFoundation FFI
// ---------------------------------------------------------------------------

type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CVReturn = i32;

/// Opaque CoreVideo pixel buffer reference.
///
/// On VideoToolbox decode, FFmpeg stores this at `frame->data[3]`.
/// Backed by an IOSurface, it can be mapped to Metal textures.
pub type CVPixelBufferRef = *mut c_void;

type CVMetalTextureCacheRef = *const c_void;
type CVMetalTextureRef = *mut c_void;

const K_CV_RETURN_SUCCESS: CVReturn = 0;

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    fn CVMetalTextureCacheCreate(
        allocator: CFAllocatorRef,
        cache_attributes: CFDictionaryRef,
        metal_device: *const c_void,
        texture_attributes: CFDictionaryRef,
        cache_out: *mut CVMetalTextureCacheRef,
    ) -> CVReturn;

    fn CVMetalTextureCacheCreateTextureFromImage(
        allocator: CFAllocatorRef,
        texture_cache: CVMetalTextureCacheRef,
        source_image: CVPixelBufferRef,
        texture_attributes: CFDictionaryRef,
        pixel_format: u64, // MTLPixelFormat (NSUInteger = u64 on 64-bit)
        width: u64,
        height: u64,
        plane_index: u64,
        texture_out: *mut CVMetalTextureRef,
    ) -> CVReturn;

    fn CVMetalTextureGetTexture(image: CVMetalTextureRef) -> *mut c_void;

    fn CVPixelBufferGetPixelFormatType(pixel_buffer: CVPixelBufferRef) -> u32;
    fn CVPixelBufferGetWidthOfPlane(pixel_buffer: CVPixelBufferRef, plane_index: u64) -> u64;
    fn CVPixelBufferGetHeightOfPlane(pixel_buffer: CVPixelBufferRef, plane_index: u64) -> u64;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
}

// ---------------------------------------------------------------------------
// MTLPixelFormat constants (matching objc2-metal values)
// ---------------------------------------------------------------------------

/// `MTLPixelFormat::R8Unorm` - single-channel 8-bit, used for Y plane.
const MTL_PIXEL_FORMAT_R8_UNORM: u64 = 10;
/// `MTLPixelFormat::RG8Unorm` - two-channel 8-bit, used for interleaved UV plane.
const MTL_PIXEL_FORMAT_RG8_UNORM: u64 = 30;

// NV12 FourCC values from VideoToolbox
const K_CV_PIXEL_FORMAT_420_YP_CB_CR_8_BI_PLANAR_VIDEO_RANGE: u32 = 0x34323076; // '420v'
const K_CV_PIXEL_FORMAT_420_YP_CB_CR_8_BI_PLANAR_FULL_RANGE: u32 = 0x34323066; // '420f'

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from Metal/VideoToolbox interop operations.
#[derive(Debug, thiserror::Error)]
pub enum MetalInteropError {
    /// The wgpu backend is not Metal.
    #[error("wgpu backend is not Metal")]
    NotMetal,

    /// Failed to create the CVMetalTextureCache.
    #[error("CVMetalTextureCacheCreate failed (CVReturn {0})")]
    CacheCreationFailed(i32),

    /// Failed to create a Metal texture from a CVPixelBuffer plane.
    #[error("CVMetalTextureCacheCreateTextureFromImage failed (CVReturn {0})")]
    TextureImportFailed(i32),

    /// The CVMetalTextureGetTexture call returned a null pointer.
    #[error("CVMetalTextureGetTexture returned null")]
    NullTexture,

    /// The CVPixelBuffer has an unsupported pixel format.
    #[error("unsupported CVPixelBuffer format: 0x{0:08x}")]
    UnsupportedFormat(u32),
}

// ---------------------------------------------------------------------------
// MetalTextureCache
// ---------------------------------------------------------------------------

/// A cache for importing VideoToolbox `CVPixelBuffer`s as wgpu textures.
///
/// Wraps Apple's `CVMetalTextureCache`, which bridges CoreVideo pixel buffers
/// to Metal textures via IOSurface without CPU copies.
///
/// Create one per `GpuContext` and reuse across frames.
pub struct MetalTextureCache {
    cv_cache: CVMetalTextureCacheRef,
}

// CVMetalTextureCacheRef is thread-safe per Apple docs.
unsafe impl Send for MetalTextureCache {}
unsafe impl Sync for MetalTextureCache {}

impl MetalTextureCache {
    /// Create a new texture cache backed by the Metal device from `gpu`.
    ///
    /// Returns `Err(NotMetal)` if the wgpu backend is not Metal.
    pub fn new(gpu: &GpuContext) -> Result<Self, MetalInteropError> {
        use wgpu::hal::api::Metal;

        let device_ptr = unsafe {
            let hal_device = gpu
                .device
                .as_hal::<Metal>()
                .ok_or(MetalInteropError::NotMetal)?;
            let raw_device = hal_device.raw_device();
            objc2::rc::Retained::as_ptr(raw_device) as *const c_void
        };

        let mut cache: CVMetalTextureCacheRef = std::ptr::null();
        let ret = unsafe {
            CVMetalTextureCacheCreate(
                std::ptr::null(), // default allocator
                std::ptr::null(), // no cache attributes
                device_ptr,
                std::ptr::null(), // no texture attributes
                &mut cache,
            )
        };

        if ret != K_CV_RETURN_SUCCESS {
            return Err(MetalInteropError::CacheCreationFailed(ret));
        }

        Ok(Self { cv_cache: cache })
    }

    /// Import a single NV12 plane from a `CVPixelBuffer` as a wgpu texture.
    ///
    /// - `plane_index` 0 = Y plane (`R8Unorm`), 1 = UV plane (`Rg8Unorm`)
    ///
    /// The returned `ImportedPlaneTexture` keeps the underlying
    /// `CVMetalTextureRef` alive. Drop it when the GPU is done reading.
    pub fn import_plane(
        &self,
        cv_pixel_buffer: CVPixelBufferRef,
        plane_index: u64,
        gpu: &GpuContext,
    ) -> Result<ImportedPlaneTexture, MetalInteropError> {
        let width = unsafe { CVPixelBufferGetWidthOfPlane(cv_pixel_buffer, plane_index) } as u32;
        let height = unsafe { CVPixelBufferGetHeightOfPlane(cv_pixel_buffer, plane_index) } as u32;

        let (mtl_format, wgpu_format) = match plane_index {
            0 => (MTL_PIXEL_FORMAT_R8_UNORM, wgpu::TextureFormat::R8Unorm),
            1 => (MTL_PIXEL_FORMAT_RG8_UNORM, wgpu::TextureFormat::Rg8Unorm),
            _ => unreachable!("NV12 only has planes 0 and 1"),
        };

        // Create a Metal texture view of this CVPixelBuffer plane
        let mut cv_texture: CVMetalTextureRef = std::ptr::null_mut();
        let ret = unsafe {
            CVMetalTextureCacheCreateTextureFromImage(
                std::ptr::null(), // default allocator
                self.cv_cache,
                cv_pixel_buffer,
                std::ptr::null(), // no texture attributes
                mtl_format,
                width as u64,
                height as u64,
                plane_index,
                &mut cv_texture,
            )
        };

        if ret != K_CV_RETURN_SUCCESS {
            return Err(MetalInteropError::TextureImportFailed(ret));
        }

        // Extract the raw MTLTexture pointer
        let mtl_texture_ptr = unsafe { CVMetalTextureGetTexture(cv_texture) };
        if mtl_texture_ptr.is_null() {
            unsafe { CFRelease(cv_texture as *const c_void) };
            return Err(MetalInteropError::NullTexture);
        }

        // Wrap as a wgpu texture via the Metal HAL
        let wgpu_texture =
            Self::wrap_mtl_texture(mtl_texture_ptr, width, height, wgpu_format, gpu)?;

        Ok(ImportedPlaneTexture {
            texture: wgpu_texture,
            _cv_texture_ref: cv_texture,
        })
    }

    /// Import both NV12 planes (Y + UV) from a `CVPixelBuffer`.
    ///
    /// Returns `(y_texture, uv_texture)` ready for use in shader bind groups.
    pub fn import_nv12(
        &self,
        cv_pixel_buffer: CVPixelBufferRef,
        gpu: &GpuContext,
    ) -> Result<(ImportedPlaneTexture, ImportedPlaneTexture), MetalInteropError> {
        // Validate pixel format
        let format = unsafe { CVPixelBufferGetPixelFormatType(cv_pixel_buffer) };
        if format != K_CV_PIXEL_FORMAT_420_YP_CB_CR_8_BI_PLANAR_VIDEO_RANGE
            && format != K_CV_PIXEL_FORMAT_420_YP_CB_CR_8_BI_PLANAR_FULL_RANGE
        {
            return Err(MetalInteropError::UnsupportedFormat(format));
        }

        let y = self.import_plane(cv_pixel_buffer, 0, gpu)?;
        let uv = self.import_plane(cv_pixel_buffer, 1, gpu)?;
        Ok((y, uv))
    }

    /// Wrap a raw `MTLTexture` pointer as a `wgpu::Texture` via the HAL.
    fn wrap_mtl_texture(
        mtl_texture_ptr: *mut c_void,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        gpu: &GpuContext,
    ) -> Result<wgpu::Texture, MetalInteropError> {
        use objc2::rc::Retained;
        use objc2_metal::MTLTexture;
        use objc2_metal::MTLTextureType;
        use wgpu::hal::api::Metal;

        // Retain the MTLTexture (bump reference count so it outlives the CVMetalTextureRef)
        let retained = unsafe {
            Retained::retain(mtl_texture_ptr as *mut objc2::runtime::ProtocolObject<dyn MTLTexture>)
        }
        .ok_or(MetalInteropError::NullTexture)?;

        // Create HAL-level texture
        let hal_texture = unsafe {
            <Metal as wgpu::hal::Api>::Device::texture_from_raw(
                retained,
                format,
                MTLTextureType::Type2D,
                1, // array layers
                1, // mip levels
                wgpu::hal::CopyExtent {
                    width,
                    height,
                    depth: 1,
                },
            )
        };

        // Wrap into wgpu texture
        let texture = unsafe {
            gpu.device.create_texture_from_hal::<Metal>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some("vt_imported"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )
        };

        Ok(texture)
    }
}

impl Drop for MetalTextureCache {
    fn drop(&mut self) {
        if !self.cv_cache.is_null() {
            unsafe { CFRelease(self.cv_cache) };
        }
    }
}

// ---------------------------------------------------------------------------
// ImportedPlaneTexture
// ---------------------------------------------------------------------------

/// A wgpu texture imported from a single NV12 plane of a VideoToolbox frame.
///
/// Holds the `CVMetalTextureRef` alive so the underlying `MTLTexture` remains
/// valid. Drop this after the GPU render pass that reads it has completed.
pub struct ImportedPlaneTexture {
    /// The wgpu texture, usable in bind groups.
    pub texture: wgpu::Texture,
    /// The CoreVideo texture reference (released on drop via `CFRelease`).
    _cv_texture_ref: CVMetalTextureRef,
}

impl Drop for ImportedPlaneTexture {
    fn drop(&mut self) {
        if !self._cv_texture_ref.is_null() {
            unsafe { CFRelease(self._cv_texture_ref as *const c_void) };
        }
    }
}

/// Validate that a `CVPixelBuffer` has a supported NV12 pixel format.
///
/// Returns `true` for `420v` (video range) and `420f` (full range) NV12 formats.
pub fn is_supported_format(cv_pixel_buffer: CVPixelBufferRef) -> bool {
    let format = unsafe { CVPixelBufferGetPixelFormatType(cv_pixel_buffer) };
    format == K_CV_PIXEL_FORMAT_420_YP_CB_CR_8_BI_PLANAR_VIDEO_RANGE
        || format == K_CV_PIXEL_FORMAT_420_YP_CB_CR_8_BI_PLANAR_FULL_RANGE
}
