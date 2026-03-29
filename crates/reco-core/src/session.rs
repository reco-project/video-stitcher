//! High-level stitching session.
//!
//! [`StitchSession`] bundles the GPU pipeline with the NV12 converter,
//! providing a single entry point for rendering and encoding stitched
//! panoramic frames. This keeps encode orchestration inside `reco-core`
//! so that every consumer (CLI, GUI, OBS plugin, cloud worker) gets the
//! same optimized frame loop without duplicating pipeline plumbing.
//!
//! ## Two-level API
//!
//! - [`StitchSession::process_frame`] - render one frame and submit it
//!   to an encoder. Use this for interactive/GUI applications or when
//!   the caller controls the frame loop (e.g. zero-copy GPU decode).
//!
//! - [`StitchSession::run`] - batch-process an entire [`FrameSource`]
//!   into an encoder, with optional progress reporting and interrupt
//!   support. Use this for CLI batch encoding.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::calibration::MatchCalibration;
use crate::encoder::{EncodeError, Encoder, OutputFrame, PixelFormat};
use crate::gpu::{GpuContext, GpuError};
use crate::nv12_converter::{Nv12Converter, Nv12Error};
use crate::pipeline::{PipelineError, StitchPipeline};
use crate::renderer::InputFormat;
use crate::source::{FrameSource, SourceError, StereoFrame};
use crate::viewport::ViewportConfig;

use thiserror::Error;

/// Configuration for creating a [`StitchSession`].
pub struct SessionConfig {
    /// Camera calibration data.
    pub calibration: MatchCalibration,
    /// Output viewport (dimensions, blend width, FOV).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels.
    pub input_width: u32,
    /// Input frame height in pixels.
    pub input_height: u32,
    /// GPU render target format (typically `Rgba8Unorm` for encoding).
    pub output_format: wgpu::TextureFormat,
    /// Input pixel format (YUV420P or NV12).
    pub input_format: InputFormat,
}

/// Progress information passed to the progress callback.
#[derive(Debug, Clone)]
pub struct FrameProgress {
    /// Number of frames processed so far.
    pub frames_completed: u64,
    /// Elapsed wall-clock time since the run started.
    pub elapsed: std::time::Duration,
}

/// Callback for progress reporting during [`StitchSession::run`].
pub type ProgressCallback = Box<dyn FnMut(&FrameProgress) + Send>;

/// Errors from [`StitchSession`].
#[derive(Debug, Error)]
pub enum SessionError {
    /// GPU initialization error.
    #[error("GPU: {0}")]
    Gpu(#[from] GpuError),

    /// GPU pipeline error.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),

    /// NV12 conversion error.
    #[error("NV12 converter: {0}")]
    Nv12(#[from] Nv12Error),

    /// Encoder error.
    #[error("encoder: {0}")]
    Encode(#[from] EncodeError),

    /// Source error.
    #[error("source: {0}")]
    Source(#[from] SourceError),
}

/// A high-level stitching session that owns the GPU pipeline and NV12 converter.
///
/// Created once per encoding job or application lifetime. Provides
/// [`process_frame`](Self::process_frame) for per-frame control and
/// [`run`](Self::run) for batch processing.
pub struct StitchSession {
    pipeline: StitchPipeline,
    nv12_converter: Nv12Converter,
    frame_count: u64,
}

impl StitchSession {
    /// Create a new session, initializing the GPU automatically.
    pub async fn new(config: SessionConfig) -> Result<Self, SessionError> {
        let gpu = GpuContext::new().await?;
        Self::with_gpu(gpu, config)
    }

    /// Create a session with an existing GPU context.
    ///
    /// Use this when the caller needs to control GPU selection (e.g.
    /// for zero-copy decode where the GPU must match the CUDA device).
    pub fn with_gpu(gpu: GpuContext, config: SessionConfig) -> Result<Self, SessionError> {
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

        let nv12_converter = Nv12Converter::new(pipeline.gpu(), output_width, output_height)?;

        Ok(Self {
            pipeline,
            nv12_converter,
            frame_count: 0,
        })
    }

    /// Render a single CPU-resident stereo frame and submit it to the encoder.
    ///
    /// Handles YUV420P and NV12 input formats. For GPU-resident frames
    /// (zero-copy path), use [`submit_render_output`](Self::submit_render_output)
    /// instead.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_process_frame")
    )]
    pub fn process_frame(
        &mut self,
        frame: &StereoFrame,
        yaw: f32,
        pitch: f32,
        encoder: &mut dyn Encoder,
    ) -> Result<(), SessionError> {
        let render_buf = self.pipeline.render_stereo_frame(frame, yaw, pitch);
        self.submit_render_output(render_buf, encoder)
    }

    /// Render from GPU-resident textures and submit to the encoder.
    ///
    /// Used with the zero-copy path where decode threads write directly
    /// to shared GPU textures. The caller must configure bind groups via
    /// [`pipeline_mut()`](Self::pipeline_mut) and call
    /// [`StitchPipeline::render_gpu_frame`] to get the command buffer,
    /// then pass it here.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_submit_render")
    )]
    pub fn submit_render_output(
        &mut self,
        render_commands: wgpu::CommandBuffer,
        encoder: &mut dyn Encoder,
    ) -> Result<(), SessionError> {
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_commands,
        )?;

        encoder.submit(OutputFrame {
            data: nv12_data.to_vec(),
            width: self.nv12_converter.width(),
            height: self.nv12_converter.height(),
            format: PixelFormat::Nv12,
            pts_us: 0, // PTS handled by encoder from frame sequence
        })?;

        self.frame_count += 1;
        Ok(())
    }

    /// Batch-process frames from a source into an encoder.
    ///
    /// Runs the full decode-render-encode loop until the source is
    /// exhausted, the frame limit is reached, or the interrupt flag
    /// is set. Returns the number of frames processed.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_run")
    )]
    pub fn run(
        &mut self,
        source: &mut dyn FrameSource,
        encoder: &mut dyn Encoder,
        frame_limit: u64,
        interrupted: &AtomicBool,
        mut on_progress: Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        let start = std::time::Instant::now();
        let yaw = 0.0_f32;
        let pitch = 0.0_f32;

        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let frame = {
                crate::profile_scope!("wait_decode");
                match source.next_frame()? {
                    Some(f) => f,
                    None => break,
                }
            };

            self.process_frame(&frame, yaw, pitch, encoder)?;

            if let Some(ref mut cb) = on_progress {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        Ok(self.frame_count)
    }

    /// Render a CPU-resident frame and return NV12 data without encoding.
    ///
    /// Useful when the caller manages encoding separately (e.g. async
    /// encode thread for live camera capture).
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_render_nv12")
    )]
    pub fn render_to_nv12(
        &mut self,
        frame: &StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<&[u8], SessionError> {
        let render_buf = self.pipeline.render_stereo_frame(frame, yaw, pitch);
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_buf,
        )?;
        self.frame_count += 1;
        Ok(nv12_data)
    }

    /// Convert a pre-rendered frame to NV12 without encoding.
    ///
    /// Like [`render_to_nv12`](Self::render_to_nv12) but accepts an
    /// already-submitted render command buffer. Used with the NV12
    /// camera path and zero-copy GPU decode.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_convert_nv12")
    )]
    pub fn convert_to_nv12(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<&[u8], SessionError> {
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_commands,
        )?;
        self.frame_count += 1;
        Ok(nv12_data)
    }

    /// Number of frames processed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Shared reference to the underlying pipeline.
    pub fn pipeline(&self) -> &StitchPipeline {
        &self.pipeline
    }

    /// Mutable reference to the underlying pipeline.
    ///
    /// Needed for zero-copy setup (configure_gpu_source) and viewport
    /// changes (resize, set_fov).
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self.pipeline
    }

    /// Shared reference to the GPU context.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    /// The name of the GPU this session is running on.
    pub fn gpu_name(&self) -> &str {
        self.pipeline.gpu_name()
    }
}
