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
//! let pipeline = StitchPipeline::new(calibration, viewport).await?;
//! # Ok(())
//! # }
//! ```

use crate::calibration::MatchCalibration;
use crate::gpu::{GpuContext, GpuError};
use crate::scene::SceneGeometry;
use crate::viewport::ViewportConfig;

use thiserror::Error;

/// Errors from the stitch pipeline.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// GPU initialization failed.
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),
}

/// The main stitching pipeline.
///
/// Owns the GPU context and scene geometry. Consumers provide frames
/// via the encoder trait and receive output frames for encoding.
pub struct StitchPipeline {
    /// GPU device and queue.
    pub gpu: GpuContext,
    /// 3D scene layout computed from calibration.
    pub scene: SceneGeometry,
    /// Calibration data (camera intrinsics + layout).
    pub calibration: MatchCalibration,
    /// Output viewport configuration.
    pub viewport: ViewportConfig,
}

impl StitchPipeline {
    /// Create a new stitch pipeline.
    ///
    /// Initializes the GPU, computes the scene geometry from the
    /// calibration data, and prepares the rendering pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Gpu`] if no compatible GPU is found.
    pub async fn new(
        calibration: MatchCalibration,
        viewport: ViewportConfig,
    ) -> Result<Self, PipelineError> {
        let gpu = GpuContext::new().await?;
        let scene = SceneGeometry::from_layout(&calibration.layout);

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
        })
    }
}
