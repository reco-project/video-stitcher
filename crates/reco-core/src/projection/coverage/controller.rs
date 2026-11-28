//! Fixed-timestep presentation control over a calibration-derived coverage map.
use super::*;
use crate::geometry::WorldPose;

/// Internal construction precision and angular chart. Coverage certifies the viewport exterior, not enclosed source holes or blending.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MapConfig {
    /// Render-space yaw chart, in radians.
    pub yaw: [f32; 2],
    /// Render-space pitch chart, in radians.
    pub pitch: [f32; 2],
    /// Number of chart cells along yaw and pitch.
    pub cells: [usize; 2],
    /// Minimum and maximum vertical FOV represented by the map, in degrees.
    pub fov: [f32; 2],
    /// Required connected viewport-centre yaw travel, in render-space radians. By default, use the span of mapped input-camera centres when available.
    pub required_yaw: Option<[f32; 2]>,
    /// Map construction precision in vertical FOV degrees.
    pub tolerance: f32,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            yaw: [-std::f32::consts::PI, std::f32::consts::PI],
            pitch: [-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2],
            cells: [512, 256],
            fov: [1.0, 178.0],
            required_yaw: None,
            tolerance: 0.05,
        }
    }
}

#[cfg(test)]
mod tests;

pub(crate) struct Controller {
    config: MapConfig,
    size: ViewportSize,
    roof: roof::Roof,
    cap: f32,
    state: Option<(RenderPose, [f32; 2], u64)>,
    route: Option<motion::Route>,
}

impl Controller {
    pub(crate) fn new(
        calibration: &Calibration,
        size: &ViewportSize,
        config: MapConfig,
    ) -> Result<Self, String> {
        Self::new_cancellable(calibration, size, config, None)
    }

    pub(crate) fn new_cancellable(
        calibration: &Calibration,
        size: &ViewportSize,
        config: MapConfig,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Self, String> {
        if matches!(
            calibration.projection,
            crate::calibration::Projection::Panorama(_)
        ) {
            return Err(
                "fixed panorama output does not support perspective coverage constraints".into(),
            );
        }
        let started = std::time::Instant::now();
        let roof = roof::Roof::build_cancellable(
            calibration,
            size,
            roof::Grid {
                yaw: config.yaw,
                pitch: config.pitch,
                columns: config.cells[0],
                rows: config.cells[1],
            },
            config.fov,
            config.tolerance,
            cancelled,
        )
        .ok_or("invalid coverage map configuration")?;
        let span = config
            .required_yaw
            .or_else(|| mapped_camera_span(calibration));
        let cap = roof
            .connected_cap(span, 1e-6)
            .ok_or("no connected coverage at the supported angular precision")?;
        log::info!(
            "Coverage motion: connected FOV cap {cap:.3} degrees, initialized in {:?}",
            started.elapsed()
        );
        Ok(Self {
            config,
            size: size.clone(),
            roof,
            cap,
            state: None,
            route: None,
        })
    }

    pub(crate) fn rebuild(
        &self,
        calibration: &Calibration,
        size: &ViewportSize,
    ) -> Result<Self, String> {
        Self::new(calibration, size, self.config.clone())
    }

    pub(crate) fn reset(&mut self) {
        self.state = None;
        if let Some(route) = &mut self.route {
            route.reset();
        }
    }

    pub(crate) fn matches_viewport(&self, size: &ViewportSize) -> bool {
        self.size.aspect_ratio() == size.aspect_ratio()
    }

    #[cfg(test)]
    pub(crate) fn current(&self) -> Option<RenderPose> {
        self.state.map(|s| s.0)
    }

    #[cfg(test)]
    pub(crate) fn clone_reset(&self) -> Self {
        Self {
            config: self.config.clone(),
            size: self.size.clone(),
            roof: self.roof.clone(),
            cap: self.cap,
            state: None,
            route: None,
        }
    }

    pub(crate) fn cap(&self) -> f32 {
        self.cap
    }

    pub(crate) fn constrain(
        &mut self,
        previous: RenderPose,
        mut requested: RenderPose,
    ) -> Result<RenderPose, String> {
        requested.fov_degrees = requested.fov_degrees.clamp(self.config.fov[0], self.cap);
        let previous = if self.roof.contains(previous) {
            previous
        } else {
            self.roof
                .supported_vertices(requested.fov_degrees)
                .map(|(_, p)| self.roof.near_yaw(p, requested.yaw))
                .min_by(|a, b| {
                    let cost = |p: &RenderPose| {
                        (f64::from(p.yaw) - f64::from(requested.yaw))
                            .hypot(f64::from(p.pitch) - f64::from(requested.pitch))
                    };
                    cost(a).total_cmp(&cost(b))
                })
                .ok_or("no supported interactive coverage pose")?
        };
        Ok(self.roof.constrain(previous, requested))
    }

    pub(crate) fn advance(
        &mut self,
        calibration: &Calibration,
        size: &ViewportSize,
        requested: WorldPose,
        index: u64,
        frame_seconds: f32,
    ) -> Result<RenderPose, String> {
        if !frame_seconds.is_finite()
            || frame_seconds <= 0.0
            || !requested.yaw.is_finite()
            || !requested.pitch.is_finite()
            || !requested.fov_degrees.is_finite()
            || requested.fov_degrees < self.config.fov[0]
        {
            return Err("invalid coverage request or FOV below map minimum".into());
        }
        if let Some((pose, _, previous_index)) = self.state {
            if index == previous_index {
                return Ok(pose);
            }
            if Some(index) != previous_index.checked_add(1) {
                return Err("coverage presentation clock is discontinuous".into());
            }
        }
        let target = calibration
            .virtual_camera()
            .yaw_pitch_to_direction(requested.yaw, requested.pitch);
        let fov = requested.fov_degrees.min(self.cap);
        let route = self
            .route
            .get_or_insert_with(|| motion::Route::new(calibration, &self.roof));
        let (pose, velocity) = match self.state {
            None => (
                motion::initial(calibration, size, &target, fov, &self.roof, route)
                    .ok_or("no supported initial coverage pose")?,
                [0.0; 2],
            ),
            Some((previous, velocity, _)) => {
                let next = motion::solve_motion(
                    calibration,
                    size,
                    motion::MotionWindow {
                        previous,
                        velocity,
                        dt: frame_seconds,
                    },
                    &target,
                    fov,
                    &self.roof,
                    route,
                )
                .ok_or("no admissible coverage motion step")?;
                log::trace!(
                    "coverage motion queries={} objective={:?}",
                    next.support_queries,
                    next.objective
                );
                (next.pose, next.velocity)
            }
        };
        self.state = Some((pose, velocity, index));
        Ok(pose)
    }
}

fn mapped_camera_span(calibration: &Calibration) -> Option<[f32; 2]> {
    let projection = calibration.projection.geometry();
    if projection.camera_count() < 2 {
        return None;
    }
    let camera = calibration.virtual_camera();
    let mut span = [f32::INFINITY, f32::NEG_INFINITY];
    for index in 0..projection.camera_count() {
        let mapped = calibration.map_source_point(index, [0.5, 0.5]).ok()?;
        let (yaw, _) = crate::geometry::world_to_render_angles(
            &camera,
            mapped.yaw,
            mapped.pitch,
            calibration.framing.tilt as f32,
            calibration.framing.roll as f32,
        );
        if !yaw.is_finite() {
            return None;
        }
        span[0] = span[0].min(yaw);
        span[1] = span[1].max(yaw);
    }
    (span[0] < span[1]).then_some(span)
}
