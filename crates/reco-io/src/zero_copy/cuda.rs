//! Shared CUDA/Vulkan staging for decoded file and live images.

use crate::ffmpeg::decoder::{DecodeError, GpuFrame};
use reco_core::gpu::GpuContext;
use reco_core::interop::{
    cuda::{CudaInteropError, cuda_2d_copy, cuda_ensure_context, cuda_synchronize},
    cuda_lease::{CudaLease, CudaSlotPool},
    vulkan::{Nv12Plane, create_nv12_shared_texture},
};
use reco_core::render::{color::YuvColor, renderer::GpuPixelFormat};

pub(crate) fn allocate(
    gpu: &GpuContext,
    sizes: &[(u32, u32)],
    format: GpuPixelFormat,
    color: YuvColor,
    rotations: [i32; 2],
) -> Result<Vec<CudaSlotPool>, CudaInteropError> {
    // NVIDIA driver workaround for R16Unorm: see issue #134.
    let _warmup = if format == GpuPixelFormat::P010 {
        Some((
            create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Y, format)?,
            create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Uv, format)?,
        ))
    } else {
        None
    };
    sizes
        .iter()
        .zip(rotations)
        .map(|(&size, rotation)| CudaSlotPool::new(gpu, size, format, color, rotation))
        .collect()
}

pub(crate) fn stage(frame: &GpuFrame, lease: &mut CudaLease) -> Result<(), DecodeError> {
    let dst = lease.cuda_frame();
    if frame.width != dst.width
        || frame.height != dst.height
        || frame.pixel_format != lease.pixel_format()
    {
        return Err(DecodeError::ConversionError(
            "decoded frame layout does not match shared CUDA textures".into(),
        ));
    }
    let stage = || {
        cuda_ensure_context()?;
        let width = dst.width as usize * frame.pixel_format.bytes_per_sample();
        let copies = cuda_2d_copy(
            dst.y_ptr,
            dst.y_pitch,
            frame.y_ptr,
            frame.y_pitch,
            width,
            dst.height as usize,
        )
        .and_then(|()| {
            cuda_2d_copy(
                dst.uv_ptr,
                dst.uv_pitch,
                frame.uv_ptr,
                frame.uv_pitch,
                width,
                dst.height as usize / 2,
            )
        });
        // Even if the second copy fails, an earlier write can still be in flight.
        let completed = cuda_synchronize();
        copies.and(completed)
    };
    stage().map_err(|e| DecodeError::ConversionError(format!("CUDA staging: {e}")))
}
