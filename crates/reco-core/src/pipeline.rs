//! Stitch pipeline orchestration.
//!
//! The [`StitchPipeline`] coordinates all stages: GPU setup, frame ingestion,
//! rendering, viewport cropping, and output encoding. It is the primary
//! entry point for consumers of `reco-core`.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use reco_core::pipeline::StitchPipeline;
//! use reco_core::calibration::MatchCalibration;
//! use reco_core::viewport::ViewportConfig;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let calibration: MatchCalibration = todo!("load from JSON");
//! let viewport = ViewportConfig::default();
//!
//! let pipeline = StitchPipeline::new(calibration, viewport, 1920, 1080).await?;
//! # Ok(())
//! # }
//! ```

use crate::calibration::MatchCalibration;
use crate::director::ViewportPosition;
use crate::gpu::{GpuContext, GpuError};
use crate::renderer::{RenderError, Renderer};
use crate::scene::SceneGeometry;
use crate::viewport::{ResolvedViewport, ViewportConfig};

use thiserror::Error;

/// Errors from the stitch pipeline.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// GPU initialization failed.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// Render error.
    #[error("render error: {0}")]
    Render(#[from] RenderError),
}

/// The main stitching pipeline.
///
/// Owns the GPU context, scene geometry, and renderer. Consumers provide
/// RGBA frames and receive stitched output via [`Self::process_frame`].
pub struct StitchPipeline {
    /// GPU device and queue.
    pub gpu: GpuContext,
    /// 3D scene layout computed from calibration.
    pub scene: SceneGeometry,
    /// Calibration data (camera intrinsics + layout).
    pub calibration: MatchCalibration,
    /// Output viewport configuration.
    pub viewport: ViewportConfig,
    /// GPU renderer (textures, pipelines, bind groups).
    renderer: Renderer,
}

impl StitchPipeline {
    /// Create a new stitch pipeline.
    ///
    /// Initializes the GPU, computes the scene geometry from the
    /// calibration data, and creates the render pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Gpu`] if no compatible GPU is found.
    pub async fn new(
        calibration: MatchCalibration,
        viewport: ViewportConfig,
        input_width: u32,
        input_height: u32,
    ) -> Result<Self, PipelineError> {
        let gpu = GpuContext::new().await?;
        let scene = SceneGeometry::from_layout(&calibration.layout);
        let renderer = Renderer::new(
            &gpu,
            viewport.width,
            viewport.height,
            input_width,
            input_height,
        );

        log::info!(
            "Pipeline initialized: {}x{} output, GPU: {}",
            viewport.width,
            viewport.height,
            gpu.adapter_info.name
        );

        Ok(Self {
            gpu,
            scene,
            calibration,
            viewport,
            renderer,
        })
    }

    /// Process a single frame through the GPU pipeline.
    ///
    /// Uploads left and right RGBA frames to the GPU, renders the stitched
    /// panorama at the given viewport position, and reads back the result.
    ///
    /// # Arguments
    ///
    /// - `left_rgba`: Raw RGBA pixel data for the left camera frame
    /// - `right_rgba`: Raw RGBA pixel data for the right camera frame
    /// - `yaw`: Horizontal pan angle in radians (0 = center/seam)
    /// - `pitch`: Vertical tilt angle in radians (0 = level)
    ///
    /// # Returns
    ///
    /// RGBA pixel data for the stitched output frame
    /// (`viewport.width * viewport.height * 4` bytes).
    pub fn process_frame(
        &self,
        left_rgba: &[u8],
        right_rgba: &[u8],
        yaw: f32,
        pitch: f32,
    ) -> Result<Vec<u8>, PipelineError> {
        self.renderer.upload_left_frame(&self.gpu, left_rgba);
        self.renderer.upload_right_frame(&self.gpu, right_rgba);

        let viewport = ResolvedViewport {
            config: self.viewport.clone(),
            position: ViewportPosition { yaw, pitch },
        };

        Ok(self
            .renderer
            .render_frame(&self.gpu, &self.scene, &self.calibration, &viewport, 0.0)?)
    }
}
