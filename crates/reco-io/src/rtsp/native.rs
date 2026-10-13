//! Retained decoded images are paired before entering the shared CUDA/Vulkan slots.

use super::*;
use crate::ffmpeg::decoder::{DecodeBackend, VideoDecoder};
use reco_core::interop::SharedTextureSet;
use reco_core::render::renderer::GpuPixelFormat;
use std::sync::mpsc::Receiver;

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
    pub shared: SharedTextureSet,
    free: Vec<Receiver<u8>>,
    pub format: GpuPixelFormat,
}

impl Delivery {
    pub fn new(
        gpu: &reco_core::gpu::GpuContext,
        sizes: &[(u32, u32)],
        format: GpuPixelFormat,
    ) -> Result<Self, SourceError> {
        let (shared, free) =
            crate::zero_copy::cuda::allocate(gpu, sizes, format).map_err(read_error)?;
        Ok(Self {
            shared,
            free,
            format,
        })
    }

    pub fn stage(&self, captures: &[Capture<Image>]) -> Result<FrameSet, SourceError> {
        let mut slots = Vec::with_capacity(captures.len());
        let result = (|| {
            for (camera, capture) in captures.iter().enumerate() {
                let Image::Cuda(frame) = &capture.image else {
                    return Err(read_error("RTSP native image residency changed"));
                };
                // The session returns slots only after staging and detection have finished.
                // A consumer that retains both deliveries must release one before reading again.
                let slot = self.free[camera]
                    .try_recv()
                    .map_err(|_| read_error("RTSP native delivery still in use by the consumer"))?;
                slots.push(slot);
                crate::zero_copy::cuda::stage(frame, &self.shared.cameras[camera].buf, slot)
                    .map_err(read_error)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            for (camera, slot) in slots.into_iter().enumerate() {
                let _ = self.shared.cameras[camera].slot_free_tx.try_send(slot);
            }
            return Err(error);
        }
        Ok(FrameSet::GpuResident { slots })
    }
}
