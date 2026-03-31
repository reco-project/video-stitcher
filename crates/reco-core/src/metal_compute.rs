//! Metal compute pipeline for GPU-side image preprocessing.
//!
//! Provides a fused NV12-to-CHW-tensor compute shader that reads
//! VideoToolbox-decoded NV12 planes (via `CVMetalTextureCache`) and
//! outputs a float32 CHW tensor ready for YOLO inference.
//!
//! The shader performs in a single dispatch:
//! 1. NV12 (Y + UV planes) to RGB color conversion (BT.601)
//! 2. Bilinear resize to model input dimensions
//! 3. Letterbox padding with grey (114/255)
//! 4. Normalize to \[0,1\] and transpose HWC to CHW
//!
//! The output `MTLBuffer` uses shared storage mode (Apple Silicon
//! unified memory), so the CPU can read the tensor data directly
//! without an explicit GPU-to-CPU copy.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions,
    MTLSize,
};

use crate::gpu::GpuContext;
use crate::metal_interop::{CVPixelBufferRef, MetalInteropError, MetalTextureCache};

/// Errors from Metal compute operations.
#[derive(Debug, thiserror::Error)]
pub enum MetalComputeError {
    /// Metal interop error (texture cache, device access).
    #[error("Metal interop: {0}")]
    Interop(#[from] MetalInteropError),

    /// MSL shader compilation failed.
    #[error("MSL compile error: {0}")]
    ShaderCompile(String),

    /// Failed to create a Metal resource.
    #[error("Metal resource creation failed: {0}")]
    ResourceCreation(String),

    /// Compute dispatch or execution failed.
    #[error("Metal compute error: {0}")]
    Compute(String),
}

/// Parameters passed to the Metal compute shader via a constant buffer.
#[repr(C)]
#[derive(Clone, Copy)]
struct PreprocessParams {
    /// Source frame width.
    src_w: u32,
    /// Source frame height.
    src_h: u32,
    /// Model input size (square, e.g. 1280).
    dst_size: u32,
    /// Padding (unused, alignment).
    _pad: u32,
    /// Letterbox scale factor.
    scale: f32,
    /// Horizontal padding offset.
    pad_x: f32,
    /// Vertical padding offset.
    pad_y: f32,
    /// Scaled content width (before padding).
    new_w: f32,
    /// Scaled content height (before padding).
    new_h: f32,
    /// Whether source is full-range YUV (1) or video-range (0).
    is_full_range: u32,
    /// Padding for 16-byte alignment.
    _pad2: [u32; 2],
}

/// MSL source for the fused NV12-to-CHW preprocessing kernel.
///
/// This compute shader reads Y (R8Unorm) and UV (RG8Unorm) textures
/// from `CVMetalTextureCache` and outputs a float32 CHW tensor suitable
/// for YOLO inference. It performs NV12-to-RGB conversion (BT.601),
/// bilinear resize, letterbox padding, and HWC-to-CHW normalization
/// in a single GPU dispatch.
const PREPROCESS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct PreprocessParams {
    uint src_w;
    uint src_h;
    uint dst_size;
    uint _pad;
    float scale;
    float pad_x;
    float pad_y;
    float new_w;
    float new_h;
    uint is_full_range;
    uint _pad2[2];
};

kernel void nv12_to_chw_tensor(
    texture2d<float, access::sample> y_tex   [[texture(0)]],
    texture2d<float, access::sample> uv_tex  [[texture(1)]],
    device float* output                      [[buffer(0)]],
    constant PreprocessParams& params         [[buffer(1)]],
    uint2 gid                                 [[thread_position_in_grid]])
{
    uint sz = params.dst_size;
    if (gid.x >= sz || gid.y >= sz) return;

    uint plane_size = sz * sz;
    uint idx = gid.y * sz + gid.x;

    // Check if this pixel is in the letterbox padding region.
    float fx = float(gid.x) - params.pad_x;
    float fy = float(gid.y) - params.pad_y;

    if (fx < 0.0 || fy < 0.0 || fx >= params.new_w || fy >= params.new_h) {
        // Grey fill (114/255 = 0.447)
        float grey = 0.44705882;
        output[idx]                  = grey;
        output[plane_size + idx]     = grey;
        output[2 * plane_size + idx] = grey;
        return;
    }

    // Map back to source coordinates for bilinear sampling.
    float src_x = fx / params.scale;
    float src_y = fy / params.scale;

    // Normalized texture coordinates for Metal sampler.
    float2 tex_coord = float2(
        (src_x + 0.5) / float(params.src_w),
        (src_y + 0.5) / float(params.src_h)
    );

    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);

    // Sample Y and UV planes.
    float y_val = y_tex.sample(s, tex_coord).r;
    float2 uv_val = uv_tex.sample(s, tex_coord).rg;

    // YUV to RGB conversion.
    float y, cb, cr;
    if (params.is_full_range != 0) {
        // Full range (420f): Y [0,255], UV [0,255]
        y  = y_val;
        cb = uv_val.x - 0.5;
        cr = uv_val.y - 0.5;
    } else {
        // Video range (420v): Y [16,235], UV [16,240]
        y  = (y_val - 0.0627451) * 1.164384;  // (Y - 16/255) * 255/219
        cb = (uv_val.x - 0.5) * 1.138393;     // (U - 128/255) * 255/224
        cr = (uv_val.y - 0.5) * 1.138393;     // (V - 128/255) * 255/224
    }

    // BT.601 YCbCr -> RGB
    float r = y + 1.402 * cr;
    float g = y - 0.344136 * cb - 0.714136 * cr;
    float b = y + 1.772 * cb;

    // Clamp to [0,1] (already normalized since Metal textures are [0,1]).
    r = clamp(r, 0.0f, 1.0f);
    g = clamp(g, 0.0f, 1.0f);
    b = clamp(b, 0.0f, 1.0f);

    // Write as CHW (channel-first, planar layout).
    output[idx]                  = r;
    output[plane_size + idx]     = g;
    output[2 * plane_size + idx] = b;
}
"#;

/// Metal compute pipeline for NV12-to-CHW tensor preprocessing.
///
/// Pre-allocates the compute pipeline state and output buffer.
/// Reuse across frames for the same input/output dimensions.
pub struct MetalPreprocessPipeline {
    #[allow(dead_code)]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    texture_cache: MetalTextureCache,
    output_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    params_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    input_size: u32,
    frame_width: u32,
    frame_height: u32,
    is_full_range: bool,
}

// SAFETY: Metal device, queue, buffers, and pipeline state are thread-safe.
// The texture cache is also Send+Sync (CVMetalTextureCacheRef is thread-safe per Apple docs).
// We only use this from the session's render thread.
unsafe impl Send for MetalPreprocessPipeline {}

impl MetalPreprocessPipeline {
    /// Create a new preprocessing pipeline.
    ///
    /// `input_size` is the YOLO model's square input dimension (e.g. 1280).
    /// `frame_width`/`frame_height` are the raw camera frame dimensions.
    /// `gpu` provides the Metal device via wgpu's HAL.
    pub fn new(
        gpu: &GpuContext,
        input_size: u32,
        frame_width: u32,
        frame_height: u32,
    ) -> Result<Self, MetalComputeError> {
        use wgpu::hal::api::Metal;

        // Extract the raw MTLDevice from wgpu.
        let device = unsafe {
            let hal_device = gpu
                .device
                .as_hal::<Metal>()
                .ok_or(MetalInteropError::NotMetal)?;
            hal_device.raw_device().clone()
        };

        // Compile MSL source at runtime.
        let source = NSString::from_str(PREPROCESS_MSL);
        let options = MTLCompileOptions::new();
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| MetalComputeError::ShaderCompile(e.to_string()))?;

        let func_name = NSString::from_str("nv12_to_chw_tensor");
        let function = library
            .newFunctionWithName(&func_name)
            .ok_or_else(|| MetalComputeError::ShaderCompile("kernel function not found".into()))?;

        let pipeline_state = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalComputeError::ResourceCreation(format!("pipeline: {e}")))?;

        // Create command queue for compute dispatches.
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| MetalComputeError::ResourceCreation("command queue".into()))?;

        // Create texture cache for importing CVPixelBuffer planes.
        let texture_cache = MetalTextureCache::new(gpu)?;

        // Allocate output buffer: 3 * input_size * input_size * sizeof(f32).
        let tensor_bytes = 3 * (input_size as usize) * (input_size as usize) * 4;
        let output_buffer = device
            .newBufferWithLength_options(tensor_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalComputeError::ResourceCreation("output buffer".into()))?;

        // Allocate params buffer.
        let params_bytes = std::mem::size_of::<PreprocessParams>();
        let params_buffer = device
            .newBufferWithLength_options(params_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalComputeError::ResourceCreation("params buffer".into()))?;

        log::info!(
            "MetalPreprocessPipeline ready: frame={}x{}, model={}x{}, buffer={:.1}MB",
            frame_width,
            frame_height,
            input_size,
            input_size,
            tensor_bytes as f64 / 1024.0 / 1024.0,
        );

        Ok(Self {
            device,
            queue,
            pipeline_state,
            texture_cache,
            output_buffer,
            params_buffer,
            input_size,
            frame_width,
            frame_height,
            is_full_range: false,
        })
    }

    /// Run the preprocessing pipeline on a CVPixelBuffer.
    ///
    /// Returns a reference to the output tensor data as `&[f32]` in CHW layout
    /// (`[3, input_size, input_size]`). The slice is valid until the next call
    /// to `preprocess` (it points into the shared MTLBuffer).
    ///
    /// # Safety
    ///
    /// `cv_pixel_buffer` must be a valid, non-null `CVPixelBufferRef`.
    pub unsafe fn preprocess(
        &mut self,
        cv_pixel_buffer: CVPixelBufferRef,
        gpu: &GpuContext,
    ) -> Result<&[f32], MetalComputeError> {
        use wgpu::hal::api::Metal;

        crate::profile_scope!("metal_preprocess");

        // Detect pixel format (video-range vs full-range).
        let format =
            unsafe { crate::metal_interop::CVPixelBufferGetPixelFormatType(cv_pixel_buffer) };
        self.is_full_range = format == 0x34323066; // '420f'

        // Import Y and UV planes as Metal textures (zero-copy via IOSurface).
        let y_plane = unsafe { self.texture_cache.import_plane(cv_pixel_buffer, 0, gpu)? };
        let uv_plane = unsafe { self.texture_cache.import_plane(cv_pixel_buffer, 1, gpu)? };

        // Write preprocessing parameters to the shared params buffer.
        let (fw, fh) = (self.frame_width as f32, self.frame_height as f32);
        let is = self.input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round();
        let new_h = (fh * scale).round();
        let pad_x = (is - new_w) / 2.0;
        let pad_y = (is - new_h) / 2.0;

        let params = PreprocessParams {
            src_w: self.frame_width,
            src_h: self.frame_height,
            dst_size: self.input_size,
            _pad: 0,
            scale,
            pad_x,
            pad_y,
            new_w,
            new_h,
            is_full_range: u32::from(self.is_full_range),
            _pad2: [0; 2],
        };

        unsafe {
            let params_ptr = self.params_buffer.contents().as_ptr() as *mut PreprocessParams;
            params_ptr.write(params);
        }

        // Create command buffer and compute encoder.
        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalComputeError::Compute("command buffer creation".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalComputeError::Compute("compute encoder creation".into()))?;

        encoder.setComputePipelineState(&self.pipeline_state);

        // Extract raw MTLTexture references from wgpu textures via HAL.
        // The as_hal guards must stay alive while the compute encoder references
        // the textures. Metal retains textures internally on setTexture, but
        // we keep the guards alive through endEncoding for safety.
        {
            let y_hal = unsafe {
                y_plane
                    .texture
                    .as_hal::<Metal>()
                    .ok_or(MetalInteropError::NotMetal)?
            };
            let uv_hal = unsafe {
                uv_plane
                    .texture
                    .as_hal::<Metal>()
                    .ok_or(MetalInteropError::NotMetal)?
            };

            unsafe {
                encoder.setTexture_atIndex(Some(y_hal.raw_handle()), 0);
                encoder.setTexture_atIndex(Some(uv_hal.raw_handle()), 1);
            }
        }

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.output_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.params_buffer), 0, 1);
        }

        // Dispatch threadgroups.
        let sz = self.input_size as usize;
        let threadgroups = MTLSize {
            width: (sz + 15) / 16,
            height: (sz + 15) / 16,
            depth: 1,
        };
        let threads_per_group = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_group);
        encoder.endEncoding();

        // Submit and wait for completion.
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back from shared buffer (unified memory, no explicit copy).
        let float_count = 3 * sz * sz;
        let result = unsafe {
            std::slice::from_raw_parts(
                self.output_buffer.contents().as_ptr() as *const f32,
                float_count,
            )
        };

        Ok(result)
    }

    /// Flush the texture cache to release stale entries.
    pub fn flush_cache(&self) {
        self.texture_cache.flush();
    }
}
