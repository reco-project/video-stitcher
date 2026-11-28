//! Linux CUDA/Vulkan zero-copy native file source.

use std::sync::{Arc, atomic::AtomicBool};

use reco_core::gpu::GpuContext;
use reco_core::interop::cuda_lease::CudaLease;
use reco_core::render::{color::YuvColor, renderer::GpuPixelFormat};
use reco_core::source::{FrameSet, SourceError, SourceFrame, SourceInfo};

use super::SkipOutcome;
use crate::decode_stream::OwnedFrames;
use crate::ffmpeg::decoder::DecodeBackend;
use crate::stitch_job::InputPath;

pub(super) struct NativeSource {
    frames: OwnedFrames<CudaLease>,
}

impl NativeSource {
    pub(super) fn supports_backend(backend: DecodeBackend) -> bool {
        matches!(backend, DecodeBackend::Cuda)
    }

    pub(super) fn allows_mixed_camera_sizes() -> bool {
        true
    }

    pub(super) fn supports_input_count(count: usize) -> bool {
        matches!(count, 1 | 2)
    }

    pub(super) fn try_open(
        inputs: &[InputPath],
        gpu: &GpuContext,
        sync_offset: i64,
        info: &SourceInfo,
        pixel_format: GpuPixelFormat,
        color: YuvColor,
        rotations: [i32; 2],
    ) -> Result<Option<Self>, SourceError> {
        let pools = crate::zero_copy::cuda::allocate(
            gpu,
            &info.camera_sizes,
            pixel_format,
            color,
            rotations,
        )
        .map_err(|error| SourceError::Init {
            path: inputs[0].first_path().display().to_string(),
            reason: error.to_string(),
        })?;
        let decoders = inputs.iter().cloned().zip(pools).collect();
        let frames = crate::zero_copy::spawn_decode_threads_gpu(
            decoders,
            sync_offset,
            Arc::new(AtomicBool::new(false)),
        );
        log::info!(
            "SmartFileSource: GPU zero-copy ({:?}, {pixel_format:?})",
            info.camera_sizes
        );

        Ok(Some(Self { frames }))
    }

    pub(super) fn decode_mode(&self) -> &'static str {
        "GPU zero-copy (CUDA/Vulkan)"
    }

    pub(super) fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        Ok(self
            .frames
            .receive()?
            .map(|cameras| SourceFrame::untimed(FrameSet::GpuResident(cameras))))
    }

    pub(super) fn skip_frames(
        &mut self,
        count: u64,
        _info: &SourceInfo,
    ) -> Result<SkipOutcome, SourceError> {
        for skipped in 0..count {
            if self.frames.receive()?.is_none() {
                return Ok(SkipOutcome {
                    skipped,
                    exhausted: true,
                });
            }
        }
        log::debug!("GPU zero-copy: skipped {count} frames for start_time");
        Ok(SkipOutcome {
            skipped: count,
            exhausted: false,
        })
    }

    pub(super) fn start_decoding(&mut self) {}
}
