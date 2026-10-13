//! Shared CUDA/Vulkan staging for decoded file and live images.

use crate::ffmpeg::decoder::{DecodeError, GpuFrame};
use reco_core::interop::cuda::{cuda_2d_copy, cuda_ensure_context, cuda_synchronize};
use reco_core::interop::zero_copy::GpuBufInfo;
#[cfg(target_os = "linux")]
use reco_core::interop::{
    shared_textures::SharedCameraTextures,
    vulkan::{Nv12Plane, create_nv12_shared_texture},
};
#[cfg(target_os = "linux")]
use reco_core::render::renderer::GpuPixelFormat;
#[cfg(target_os = "linux")]
use reco_core::{gpu::GpuContext, interop::SharedTextureSet};
#[cfg(target_os = "linux")]
use std::sync::mpsc::{Receiver, sync_channel};

#[cfg(target_os = "linux")]
pub(crate) fn allocate(
    gpu: &GpuContext,
    sizes: &[(u32, u32)],
    format: GpuPixelFormat,
) -> Result<(SharedTextureSet, Vec<Receiver<u8>>), reco_core::interop::cuda::CudaInteropError> {
    // NVIDIA driver workaround for R16Unorm: see issue #134.
    let _warmup = if format == GpuPixelFormat::P010 {
        Some((
            create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Y, format)?,
            create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Uv, format)?,
        ))
    } else {
        None
    };
    let mut cameras = Vec::with_capacity(sizes.len());
    let mut receivers = Vec::with_capacity(sizes.len());
    for &(width, height) in sizes {
        let y0 = create_nv12_shared_texture(gpu, width, height, Nv12Plane::Y, format)?;
        let uv0 = create_nv12_shared_texture(gpu, width, height, Nv12Plane::Uv, format)?;
        let y1 = create_nv12_shared_texture(gpu, width, height, Nv12Plane::Y, format)?;
        let uv1 = create_nv12_shared_texture(gpu, width, height, Nv12Plane::Uv, format)?;
        let buf = GpuBufInfo {
            y_ptr: [y0.cuda_ptr, y1.cuda_ptr],
            uv_ptr: [uv0.cuda_ptr, uv1.cuda_ptr],
            y_pitch: [y0.pitch, y1.pitch],
            uv_pitch: [uv0.pitch, uv1.pitch],
            width,
            height,
            pixel_format: format,
        };
        let (tx, rx) = sync_channel(2);
        tx.send(0).expect("seed slot");
        tx.send(1).expect("seed slot");
        receivers.push(rx);
        cameras.push(SharedCameraTextures {
            textures: [y0, uv0, y1, uv1],
            buf,
            slot_free_tx: tx,
        });
    }
    Ok((SharedTextureSet { cameras }, receivers))
}

pub(crate) fn stage(frame: &GpuFrame, buf: &GpuBufInfo, slot: u8) -> Result<(), DecodeError> {
    if slot > 1
        || frame.width != buf.width
        || frame.height != buf.height
        || frame.pixel_format != buf.pixel_format
    {
        return Err(DecodeError::ConversionError(
            "decoded frame layout does not match shared CUDA textures".into(),
        ));
    }
    let copy = || {
        cuda_ensure_context()?;
        let s = usize::from(slot);
        let width = buf.width as usize * buf.pixel_format.bytes_per_sample();
        cuda_2d_copy(
            buf.y_ptr[s],
            buf.y_pitch[s],
            frame.y_ptr,
            frame.y_pitch,
            width,
            buf.height as usize,
        )?;
        cuda_2d_copy(
            buf.uv_ptr[s],
            buf.uv_pitch[s],
            frame.uv_ptr,
            frame.uv_pitch,
            width,
            buf.height as usize / 2,
        )?;
        // Publish only after both planes are complete, and before releasing the decoded owner.
        cuda_synchronize()
    };
    copy().map_err(|e| DecodeError::ConversionError(format!("CUDA staging: {e}")))
}
