//! Retained decoded images are paired before entering the shared CUDA/Vulkan slots.

use super::*;
use crate::ffmpeg::decoder::{DecodeBackend, VideoDecoder};
use reco_core::interop::cuda_lease::CudaSlotPool;
use reco_core::render::color::YuvColor;
use reco_core::render::renderer::GpuPixelFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    Host,
    Cuda(GpuPixelFormat),
}

/// Choose one residency for the entire source before any images are published.
pub(super) fn negotiate(
    shared: &Shared,
    camera: usize,
    decoder: &VideoDecoder,
    cancelled: impl Fn() -> bool,
) -> Result<Layout, SourceError> {
    if !shared.native_requested {
        return Ok(Layout::Host);
    }
    let layout = if decoder.backend() == DecodeBackend::Cuda {
        Layout::Cuda(decoder.pixel_format())
    } else {
        Layout::Host
    };
    let mut state = shared.lock();
    state.inputs[camera].layout = Some(layout);
    if state.inputs.iter().all(|input| input.layout.is_some()) {
        state.delivery = Some(
            if state
                .inputs
                .iter()
                .all(|input| input.layout == Some(layout))
            {
                layout
            } else {
                Layout::Host
            },
        );
        shared.wake.notify_all();
    }
    loop {
        if cancelled() {
            return Err(read_error("RTSP native opening interrupted or timed out"));
        }
        for input in &state.inputs {
            if let Some(Err(error)) = &input.terminal {
                return Err(error.clone());
            }
        }
        if let Some(delivery) = state.delivery {
            return Ok(delivery);
        }
        state = shared
            .wake
            .wait_timeout(state, Duration::from_millis(10))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}

pub(super) struct Delivery {
    pools: Vec<CudaSlotPool>,
    pub format: GpuPixelFormat,
}

impl Delivery {
    pub fn new(
        gpu: &reco_core::gpu::GpuContext,
        sizes: &[(u32, u32)],
        format: GpuPixelFormat,
        color: YuvColor,
        rotations: [i32; 2],
    ) -> Result<Self, SourceError> {
        let pools = crate::zero_copy::cuda::allocate(gpu, sizes, format, color, rotations)
            .map_err(read_error)?;
        Ok(Self { pools, format })
    }

    pub fn stage(&self, captures: &[Capture<Image>]) -> Result<FrameSet, SourceError> {
        let cameras = self
            .pools
            .iter()
            .zip(captures)
            .map(|(pool, capture)| {
                let Image::Cuda(frame) = &capture.image else {
                    return Err(read_error("RTSP native image residency changed"));
                };
                let mut lease = pool.try_acquire().ok_or_else(|| {
                    read_error("RTSP native delivery still in use by the consumer")
                })?;
                crate::zero_copy::cuda::stage(frame, &mut lease).map_err(read_error)?;
                Ok(lease)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FrameSet::GpuResident(cameras))
    }
}
