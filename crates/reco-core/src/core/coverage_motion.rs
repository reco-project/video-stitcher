//! Shared coverage for world-space tracking and direct render-space interaction.

use super::{StitchCore, StitchCoreError};
use crate::geometry::{RenderPose, WorldPose};
use crate::projection::coverage::controller::{Controller, MapConfig};

impl StitchCore {
    /// Resolve direct preview input without waiting for coverage construction.
    ///
    /// While coverage is being prepared, returns validated input unchanged, permitting black edges. Once ready, fits the current pose through coverage. Does not change the user's constrained-look setting or advance the tracking clock. Interactive recording follows this behavior; offline export and automatic tracking use strict pose resolution instead.
    pub fn constrain_preview_pose(
        &mut self,
        previous: RenderPose,
        requested: RenderPose,
    ) -> Result<RenderPose, StitchCoreError> {
        validate_interactive_poses(previous, requested)?;
        if !self.constrained_look || self.prepare_coverage_preview()?.is_none() {
            return Ok(requested);
        }
        self.constrain_render_pose(previous, requested)
    }

    /// Constrain direct manipulation without advancing the tracking clock or adding inertia.
    ///
    /// Both arguments and the result are render-space poses, directly usable by render-at-pose methods. Feed the accepted result back into the input control so blocked movement does not accumulate. Yaw requests use the nearest periodic equivalent to the previous pose. Movement follows a supported segment and slides at contact; zoom may move the view inward to fit. A previous pose invalidated by geometry changes or re-enabling constraints is fitted anew. Disabled constraints return the request unchanged. The derived coverage map is built on first use.
    pub fn constrain_render_pose(
        &mut self,
        previous: RenderPose,
        requested: RenderPose,
    ) -> Result<RenderPose, StitchCoreError> {
        validate_interactive_poses(previous, requested)?;
        if !self.constrained_look {
            return Ok(requested);
        }
        self.prepare_coverage()?;
        self.coverage_motion
            .as_mut()
            .expect("prepared coverage")
            .constrain(previous, requested)
            .map_err(StitchCoreError::Config)
    }

    /// Prepare coverage for the accepted calibration and viewport, returning its vertical FOV cap.
    ///
    /// Optional warm-up: strict pose constraints call this lazily. Repeated calls preserve the map and motion state. A failed construction leaves the accepted geometry unchanged. Strict geometry edits and viewport resizes rebuild an existing map; preview edits invalidate it. No-op and blend-only edits preserve it.
    pub fn prepare_coverage(&mut self) -> Result<f32, StitchCoreError> {
        // Strict consumers own synchronous readiness, not a pending preview job.
        self.pending_coverage = None;
        if let Some(controller) = &self.coverage_motion {
            if controller.matches_viewport(self.executor.viewport_size()) {
                return Ok(controller.cap());
            }
            let rebuilt = controller
                .rebuild(self.calibration(), self.executor.viewport_size())
                .map_err(StitchCoreError::Config)?;
            let cap = rebuilt.cap();
            self.coverage_motion = Some(rebuilt);
            return Ok(cap);
        }
        let controller = Controller::new(
            self.calibration(),
            self.executor.viewport_size(),
            MapConfig::default(),
        )
        .map_err(StitchCoreError::Config)?;
        let cap = controller.cap();
        self.coverage_motion = Some(controller);
        Ok(cap)
    }

    /// Resolve one world-space request into a render-space pose over one output frame.
    ///
    /// `frame_seconds` is the finite positive output frame duration, not processing time. Indices must be consecutive; repeating the current index returns the same pose without advancing motion. The first index after geometry changes or a constrained-look toggle establishes a new stream. Do not feed the returned render-space pose back as a world-space request.
    ///
    /// Disabling constrained look bypasses coverage but still applies the world-to-render orientation. Explicit render-at-pose methods accept the result directly and do not advance this controller a second time.
    pub fn advance_pose(
        &mut self,
        requested: WorldPose,
        index: u64,
        frame_seconds: f32,
    ) -> Result<RenderPose, StitchCoreError> {
        if !frame_seconds.is_finite()
            || frame_seconds <= 0.0
            || !requested.yaw.is_finite()
            || !requested.pitch.is_finite()
            || !requested.fov_degrees.is_finite()
            || !(1.0..=179.0).contains(&requested.fov_degrees)
        {
            return Err(StitchCoreError::Config(
                "invalid presentation pose or frame duration".into(),
            ));
        }
        let pose = if !self.constrained_look {
            self.calibration().world_to_render_pose(requested)
        } else {
            self.prepare_coverage()?;
            self.coverage_motion
                .as_mut()
                .expect("prepared coverage")
                .advance(
                    self.executor.calibration(),
                    self.executor.viewport_size(),
                    requested,
                    index,
                    frame_seconds,
                )
                .map_err(StitchCoreError::Config)?
        };
        if let Some(sink) = self.event_sink.as_deref_mut() {
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::RenderPoseResolved {
                    frame_index: index,
                    pose,
                },
            );
        }
        Ok(pose)
    }
}

fn validate_interactive_poses(
    previous: RenderPose,
    requested: RenderPose,
) -> Result<(), StitchCoreError> {
    if [previous, requested].iter().any(|p| {
        !p.yaw.is_finite()
            || !p.pitch.is_finite()
            || !p.fov_degrees.is_finite()
            || !(1.0..=179.0).contains(&p.fov_degrees)
    }) {
        return Err(StitchCoreError::Config("invalid interactive pose".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
