//! Fallback native source stub for platforms without GPU zero-copy decode.

use std::convert::Infallible;

use reco_core::gpu::GpuContext;
use reco_core::render::renderer::GpuPixelFormat;
use reco_core::source::{SourceError, SourceFrame, SourceInfo};

use super::SkipOutcome;
use crate::ffmpeg::decoder::DecodeBackend;
use crate::stitch_job::InputPath;

pub(super) struct NativeSource(Infallible);

impl NativeSource {
    pub(super) fn supports_backend(_backend: DecodeBackend) -> bool {
        false
    }

    pub(super) fn allows_mixed_camera_sizes() -> bool {
        false
    }

    pub(super) fn supports_input_count(_count: usize) -> bool {
        false
    }

    pub(super) fn try_open(
        _inputs: &[InputPath],
        _gpu: &GpuContext,
        _sync_offset: i64,
        _info: &SourceInfo,
        _pixel_format: GpuPixelFormat,
        _color: reco_core::render::color::YuvColor,
        _rotations: [i32; 2],
    ) -> Result<Option<Self>, SourceError> {
        log::info!("SmartFileSource: zero-copy not yet implemented for this platform, using CPU");
        Ok(None)
    }

    pub(super) fn decode_mode(&self) -> &'static str {
        match self.0 {}
    }

    pub(super) fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        match self.0 {}
    }

    pub(super) fn skip_frames(
        &mut self,
        _count: u64,
        _info: &SourceInfo,
    ) -> Result<SkipOutcome, SourceError> {
        match self.0 {}
    }

    pub(super) fn start_decoding(&mut self) {
        match self.0 {}
    }
}
