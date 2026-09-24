//! CPU executor: validated calibration and synchronous software stitching.

use super::{StitchError, StitchExecutor};
use crate::calibration::Calibration;
use crate::geometry::RenderPose;
use crate::projection::ProjectionGeometry;
use crate::render::planes::{Nv12Planes, Yuv420pPlanes};
use crate::render::viewport::ViewportSize;
use crate::stitch::cpu::stitch_rgba;

/// CPU software backend - pure Rust, no GPU. The portable / GPU-less path.
pub struct CpuExecutor {
    pub(crate) calib: Calibration,
    pub(crate) viewport_size: ViewportSize,
    pub(crate) cam: (u32, u32),
    pub(crate) full_range: bool,
}

impl CpuExecutor {
    /// Configure a CPU executor for a fixed source size and output
    /// viewport. The projection is the calibration document's projection
    /// (zero-copy: the document is the engine), so a document edit
    /// switches the projection automatically.
    pub fn new(
        calib: Calibration,
        viewport_size: ViewportSize,
        cam_w: u32,
        cam_h: u32,
        full_range: bool,
    ) -> Result<Self, StitchError> {
        calib
            .validate()
            .map_err(|e| StitchError::InvalidConfig(e.to_string()))?;
        viewport_size
            .validate()
            .map_err(StitchError::InvalidConfig)?;
        if cam_w < 2 || cam_h < 2 {
            return Err(StitchError::InvalidConfig(format!(
                "source dimensions must be >= 2, got {cam_w}x{cam_h}"
            )));
        }
        log::debug!(
            "CpuExecutor: projection '{}' from the calibration projection",
            calib.projection.geometry().name()
        );

        Ok(Self {
            calib,
            viewport_size,
            cam: (cam_w, cam_h),
            full_range,
        })
    }

    /// The projection in effect: a borrow of the document's projection.
    pub(crate) fn geometry(&self) -> &dyn ProjectionGeometry {
        self.calib.projection.geometry()
    }

    /// Stitch one NV12 frame set to RGBA at the configured output
    /// size. `&self` on purpose: the CPU stitch is stateless per call
    /// (the [`StitchExecutor`] trait's `&mut self` accommodates the
    /// GPU arm's readback ring).
    pub fn stitch_nv12(
        &self,
        planes: &[Nv12Planes<'_>],
        pose: RenderPose,
    ) -> Result<Vec<u8>, StitchError> {
        stitch_rgba(
            planes,
            self.cam,
            &self.calib,
            &self.viewport_size,
            pose,
            self.full_range,
        )
    }

    /// Stitch one YUV420P frame set to RGBA at the configured output
    /// size. The CPU kernel is format-flexible per call (unlike the
    /// GPU pipeline, which fixes its input format at construction);
    /// the [`StitchExecutor`] trait covers the NV12 contract, this
    /// inherent entry covers planar YUV sources (file decode).
    pub fn stitch_yuv(
        &self,
        planes: &[Yuv420pPlanes<'_>],
        pose: RenderPose,
    ) -> Result<Vec<u8>, StitchError> {
        crate::stitch::cpu::stitch_rgba_yuv420p(
            planes,
            self.cam,
            &self.calib,
            &self.viewport_size,
            pose,
            self.full_range,
        )
    }
}

impl StitchExecutor for CpuExecutor {
    fn stitch(
        &mut self,
        planes: &[Nv12Planes<'_>],
        pose: RenderPose,
    ) -> Result<Vec<u8>, StitchError> {
        // Plane-size + camera-count validation lives in stitch_rgba, which
        // returns a typed error instead of panicking on a short/truncated frame.
        self.stitch_nv12(planes, pose)
    }

    fn output_dims(&self) -> (u32, u32) {
        (self.viewport_size.width, self.viewport_size.height)
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}
