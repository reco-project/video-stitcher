//! Live push-based stitching for compositor/callback consumers.
//!
//! [`LiveStitchSession`] bundles a [`StitchPipeline`] with an
//! `RgbaReadback` helper so consumers that receive frames one at a
//! time on a callback thread (OBS `video_tick`, V4L2 capture, WebRTC
//! ingest) can call a single `submit_frame` and get back a tightly
//! packed RGBA buffer ready for compositor upload.
//!
//! Contrast with `StitchSession`, which drives a pull-based
//! [`FrameSource`](crate::source::FrameSource) loop - the wrong shape
//! when frames arrive asynchronously from a C callback.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use reco_core::session::{LiveStitchSession, LiveSessionConfig};
//!
//! let mut session = LiveStitchSession::new(gpu, LiveSessionConfig {
//!     calibration,
//!     viewport,
//!     input_width, input_height,
//!     output_format: wgpu::TextureFormat::Rgba8Unorm,
//!     input_format: InputFormat::Yuv420p,
//! })?;
//!
//! // Inside the OBS video_tick callback:
//! if let Some(rgba) = session.submit_frame(&left, &right, yaw, pitch)? {
//!     // rgba is &[u8] of length output_width * output_height * 4
//!     compositor.upload(rgba);
//! }
//! ```

use thiserror::Error;

use crate::calibration::MatchCalibration;
use crate::gpu::GpuContext;
use crate::pipeline::{PipelineError, StitchPipeline, YuvPlanes};
use crate::renderer::InputFormat;
use crate::rgba_readback::{RgbaReadback, RgbaReadbackError};
use crate::viewport::ViewportConfig;

/// Configuration for creating a [`LiveStitchSession`].
pub struct LiveSessionConfig {
    /// Camera calibration data.
    pub calibration: MatchCalibration,
    /// Output viewport (dimensions, blend width, FOV).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels (per camera).
    pub input_width: u32,
    /// Input frame height in pixels (per camera).
    pub input_height: u32,
    /// GPU render target format. Use `Rgba8Unorm` for most compositor
    /// consumers; `Bgra8Unorm` matches native Windows/DirectX formats.
    pub output_format: wgpu::TextureFormat,
    /// Input pixel format.
    pub input_format: InputFormat,
}

/// Errors from [`LiveStitchSession`].
#[derive(Debug, Error)]
pub enum LiveSessionError {
    /// Pipeline initialization or render failed.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    /// Readback staging/mapping failed.
    #[error("readback: {0}")]
    Readback(#[from] RgbaReadbackError),
}

/// High-level wrapper for push-based stitching with RGBA readback.
///
/// See the module-level docs for usage.
pub struct LiveStitchSession {
    pipeline: StitchPipeline,
    readback: RgbaReadback,
    output_width: u32,
    output_height: u32,
}

impl LiveStitchSession {
    /// Build a new session. Owns the provided [`GpuContext`].
    pub fn new(gpu: GpuContext, config: LiveSessionConfig) -> Result<Self, LiveSessionError> {
        let output_width = config.viewport.width;
        let output_height = config.viewport.height;

        let pipeline = StitchPipeline::with_gpu(
            gpu,
            config.calibration,
            config.viewport,
            config.input_width,
            config.input_height,
            config.output_format,
            config.input_format,
        )?;

        let readback = RgbaReadback::new(pipeline.gpu(), output_width, output_height)?;

        Ok(Self {
            pipeline,
            readback,
            output_width,
            output_height,
        })
    }

    /// Submit a stereo YUV420P frame and request an RGBA render.
    ///
    /// Returns the tightly-packed RGBA bytes from two frames ago (triple-
    /// buffered staging), or `None` during pipeline warmup (first two
    /// calls). The returned slice is `output_width * output_height * 4`
    /// bytes; it borrows from the session's internal buffers and is
    /// valid until the next `submit_frame` call.
    ///
    /// `yaw` and `pitch` are in radians.
    pub fn submit_frame(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Option<&[u8]>, LiveSessionError> {
        let cmd = self.pipeline.render_to_target(left, right, yaw, pitch)?;
        let rgba =
            self.readback
                .readback(self.pipeline.gpu(), self.pipeline.render_target(), cmd)?;
        Ok(rgba)
    }

    /// Output dimensions in pixels.
    pub fn output_dims(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }

    /// Shared access to the wrapped pipeline (for `update_camera_params`,
    /// pose introspection, etc.).
    pub fn pipeline(&self) -> &StitchPipeline {
        &self.pipeline
    }

    /// Mutable access to the wrapped pipeline.
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self.pipeline
    }

    /// Access the GPU context that owns the render resources.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    /// Drain one pending readback slot without submitting a new frame.
    ///
    /// Useful at shutdown to collect the 1-2 frames still in-flight in
    /// the triple-buffered staging pipeline. Returns `None` when no
    /// frames remain.
    pub fn flush(&mut self) -> Result<Option<&[u8]>, LiveSessionError> {
        let gpu = self.pipeline.gpu();
        let rgba = self.readback.flush_pending(gpu)?;
        Ok(rgba)
    }
}
