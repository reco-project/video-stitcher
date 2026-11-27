//! macOS VideoToolbox/Metal zero-copy native file source.

use reco_core::gpu::GpuContext;
use reco_core::interop::metal::RetainedCVPixelBuffer;
use reco_core::render::renderer::GpuPixelFormat;
use reco_core::source::{FrameSet, SourceError, SourceFrame, SourceInfo};

use super::SkipOutcome;
use crate::decode_stream::OwnedFrames;
use crate::ffmpeg::decoder::DecodeBackend;
use crate::stitch_job::InputPath;

pub(super) struct NativeSource {
    frames: OwnedFrames<RetainedCVPixelBuffer>,
}

impl NativeSource {
    pub(super) fn supports_backend(backend: DecodeBackend) -> bool {
        matches!(backend, DecodeBackend::VideoToolbox)
    }

    pub(super) fn allows_mixed_camera_sizes() -> bool {
        false
    }

    pub(super) fn supports_input_count(count: usize) -> bool {
        count == 2
    }

    pub(super) fn try_open(
        inputs: &[InputPath],
        _gpu: &GpuContext,
        sync_offset: i64,
        info: &SourceInfo,
        _pixel_format: GpuPixelFormat,
        full_range: bool,
    ) -> Result<Option<Self>, SourceError> {
        let frames = crate::zero_copy::spawn_vt_decode_frames(inputs, sync_offset);

        log::info!(
            "SmartFileSource: Metal zero-copy ({}x{}, VideoToolbox decode{})",
            info.camera_sizes[0].0,
            info.camera_sizes[0].1,
            if full_range { ", full-range" } else { "" }
        );

        Ok(Some(Self { frames }))
    }

    pub(super) fn decode_mode(&self) -> &'static str {
        "Metal zero-copy (VideoToolbox)"
    }

    pub(super) fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        match self.frames.receive()? {
            Some(frames) => Ok(Some(SourceFrame::untimed(FrameSet::MetalResident(frames)))),
            None => Ok(None),
        }
    }

    pub(super) fn skip_frames(
        &mut self,
        count: u64,
        _info: &SourceInfo,
    ) -> Result<SkipOutcome, SourceError> {
        for i in 0..count {
            if self.next_frame()?.is_none() {
                return Ok(SkipOutcome {
                    skipped: i,
                    exhausted: true,
                });
            }
        }
        Ok(SkipOutcome {
            skipped: count,
            exhausted: false,
        })
    }

    pub(super) fn start_decoding(&mut self) {}
}
