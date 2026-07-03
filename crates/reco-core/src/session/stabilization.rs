//! Viewport stabilization corrections.
//!
//! The session applies stabilization to the virtual-camera pose before
//! coverage clamping. Two sources can contribute:
//!
//! - [`StabilizationTrack`] - frame-indexed yaw/pitch offsets, e.g.
//!   from pre-extracted gyro data.
//! - [`RoiStabilizer`] - a pose-level compensator that uses selected
//!   field ROI points as visual anchors. It keeps the ROI projection
//!   close to a low-pass reference so short-frame panner jitter moves
//!   the virtual camera instead of shaking the presented output.

use crate::calibration::FieldRoi;
use crate::geometry::ViewportPosition;
use crate::render::scene::SceneGeometry;

const DEFAULT_ROI_FOLLOW_ALPHA: f32 = 0.08;
const DEFAULT_ROI_STRENGTH: f32 = 1.0;
const DEFAULT_ROI_MAX_CORRECTION_RAD: f32 = 0.08;
const DEFAULT_ROI_MIN_VISIBLE_POINTS: usize = 2;
const DEFAULT_PROJECTION_EPSILON_RAD: f32 = 1.0e-3;
const MAX_ANCHOR_SCREEN_MARGIN: f32 = 1.0;

/// One normalized gyroscope sample.
///
/// Axis convention follows `reco-calibrate`'s telemetry normalization:
/// `x` is camera-right angular velocity, `y` is camera-down/vertical
/// angular velocity, and `z` is optical-axis roll. This MVP consumes
/// only `x` (pitch) and `y` (yaw); roll needs a dynamic render-roll path.
#[derive(Debug, Clone, Copy)]
pub struct GyroSample {
    /// Timestamp in seconds from the start of the source video.
    pub t: f64,
    /// Angular velocity around the camera-right axis, radians/second.
    pub x: f64,
    /// Angular velocity around the camera-down/vertical axis, radians/second.
    pub y: f64,
    /// Angular velocity around the optical axis, radians/second.
    pub z: f64,
}

/// Additive correction for one output frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StabilizationCorrection {
    /// Yaw offset to add to the viewport pose, in radians.
    pub yaw: f32,
    /// Pitch offset to add to the viewport pose, in radians.
    pub pitch: f32,
}

/// Precomputed correction track, indexed by rendered output frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StabilizationTrack {
    corrections: Vec<StabilizationCorrection>,
}

/// Tuning for ROI-anchor stabilization.
#[derive(Debug, Clone)]
pub struct RoiStabilizationConfig {
    /// Field ROI points in panorama `[yaw, pitch]` radians.
    pub roi: FieldRoi,
    /// How much of the solved correction to apply, usually `0.0..=1.0`.
    pub strength: f32,
    /// Low-pass follow factor for anchor reference positions. Lower
    /// values reject more jitter but lag intentional pans longer.
    pub follow_alpha: f32,
    /// Per-axis correction cap in radians.
    pub max_correction_rad: f32,
    /// Minimum projected anchors needed before applying correction.
    pub min_visible_points: usize,
    /// Finite-difference step used to estimate projection sensitivity.
    pub projection_epsilon_rad: f32,
}

impl RoiStabilizationConfig {
    /// Create ROI stabilization with conservative defaults.
    pub fn new(roi: FieldRoi) -> Self {
        Self {
            roi,
            strength: DEFAULT_ROI_STRENGTH,
            follow_alpha: DEFAULT_ROI_FOLLOW_ALPHA,
            max_correction_rad: DEFAULT_ROI_MAX_CORRECTION_RAD,
            min_visible_points: DEFAULT_ROI_MIN_VISIBLE_POINTS,
            projection_epsilon_rad: DEFAULT_PROJECTION_EPSILON_RAD,
        }
    }

    /// Validate and clamp values that are safe to clamp.
    pub fn sanitized(mut self) -> Result<Self, String> {
        if self.roi.points.len() < 2 {
            return Err(format!(
                "ROI stabilization needs at least 2 ROI points, got {}",
                self.roi.points.len()
            ));
        }
        if self
            .roi
            .points
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err("ROI stabilization points must be finite".into());
        }
        if !self.strength.is_finite() {
            return Err(format!(
                "ROI stabilization strength must be finite, got {}",
                self.strength
            ));
        }
        if !self.follow_alpha.is_finite() {
            return Err(format!(
                "ROI stabilization follow_alpha must be finite, got {}",
                self.follow_alpha
            ));
        }
        if !self.max_correction_rad.is_finite() || self.max_correction_rad <= 0.0 {
            return Err(format!(
                "ROI stabilization max_correction_rad must be positive and finite, got {}",
                self.max_correction_rad
            ));
        }
        if !self.projection_epsilon_rad.is_finite() || self.projection_epsilon_rad <= 0.0 {
            return Err(format!(
                "ROI stabilization projection_epsilon_rad must be positive and finite, got {}",
                self.projection_epsilon_rad
            ));
        }

        self.follow_alpha = self.follow_alpha.clamp(0.0, 1.0);
        self.min_visible_points = self.min_visible_points.clamp(1, self.roi.points.len());
        self.projection_epsilon_rad = self.projection_epsilon_rad.clamp(1.0e-5, 0.05);
        Ok(self)
    }

    /// Set correction strength.
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    /// Set the reference follow alpha.
    pub fn with_follow_alpha(mut self, follow_alpha: f32) -> Self {
        self.follow_alpha = follow_alpha;
        self
    }

    /// Set the per-axis correction cap in radians.
    pub fn with_max_correction_rad(mut self, max_correction_rad: f32) -> Self {
        self.max_correction_rad = max_correction_rad;
        self
    }
}

/// Runtime ROI-anchor stabilizer.
#[derive(Debug, Clone)]
pub struct RoiStabilizer {
    config: RoiStabilizationConfig,
    reference_points: Vec<Option<ScreenPoint>>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ViewportStabilizer {
    track: Option<StabilizationTrack>,
    roi: Option<RoiStabilizer>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StabilizationContext<'a> {
    pub(crate) frame_index: u64,
    pub(crate) fov_degrees: f32,
    pub(crate) aspect: f32,
    pub(crate) rig_tilt: f32,
    pub(crate) rig_roll: f32,
    pub(crate) scene: &'a SceneGeometry,
}

#[derive(Debug, Clone, Copy)]
struct ScreenPoint {
    x: f32,
    y: f32,
}

impl RoiStabilizer {
    /// Build an ROI stabilizer from validated configuration.
    pub fn new(config: RoiStabilizationConfig) -> Result<Self, String> {
        let config = config.sanitized()?;
        Ok(Self {
            reference_points: vec![None; config.roi.points.len()],
            config,
        })
    }

    /// Reset the low-pass anchor references.
    pub fn reset(&mut self) {
        self.reference_points.fill(None);
    }

    /// Compute and apply an ROI-anchor correction to a pose.
    pub fn stabilize_pose(
        &mut self,
        pose: ViewportPosition,
        fov_degrees: f32,
        aspect: f32,
        rig_tilt: f32,
        rig_roll: f32,
        scene: &SceneGeometry,
    ) -> (ViewportPosition, StabilizationCorrection) {
        let current = self.project_points(pose, fov_degrees, aspect, rig_tilt, rig_roll, scene);

        let mut initialized = 0usize;
        for (reference, projected) in self.reference_points.iter_mut().zip(current.iter()) {
            if reference.is_none()
                && let Some(point) = *projected
            {
                *reference = Some(point);
                initialized += 1;
            }
        }

        let correction = if initialized == current.iter().filter(|p| p.is_some()).count() {
            StabilizationCorrection::default()
        } else {
            self.solve_correction(pose, fov_degrees, aspect, rig_tilt, rig_roll, scene)
                .unwrap_or_default()
        };

        self.update_references(&current);

        let stabilized = ViewportPosition {
            yaw: pose.yaw + correction.yaw,
            pitch: pose.pitch + correction.pitch,
            fov_degrees: pose.fov_degrees,
        };
        (stabilized, correction)
    }

    fn project_points(
        &self,
        pose: ViewportPosition,
        fov_degrees: f32,
        aspect: f32,
        rig_tilt: f32,
        rig_roll: f32,
        scene: &SceneGeometry,
    ) -> Vec<Option<ScreenPoint>> {
        self.config
            .roi
            .points
            .iter()
            .map(|point| {
                project_roi_point(point, pose, fov_degrees, aspect, rig_tilt, rig_roll, scene)
            })
            .collect()
    }

    fn solve_correction(
        &self,
        pose: ViewportPosition,
        fov_degrees: f32,
        aspect: f32,
        rig_tilt: f32,
        rig_roll: f32,
        scene: &SceneGeometry,
    ) -> Option<StabilizationCorrection> {
        let eps = self.config.projection_epsilon_rad;
        let yaw_pose = ViewportPosition {
            yaw: pose.yaw + eps,
            ..pose
        };
        let pitch_pose = ViewportPosition {
            pitch: pose.pitch + eps,
            ..pose
        };

        let mut a00 = 0.0_f32;
        let mut a01 = 0.0_f32;
        let mut a11 = 0.0_f32;
        let mut b0 = 0.0_f32;
        let mut b1 = 0.0_f32;
        let mut used = 0usize;

        for (idx, anchor) in self.config.roi.points.iter().enumerate() {
            let Some(reference) = self.reference_points[idx] else {
                continue;
            };
            let Some(current) =
                project_roi_point(anchor, pose, fov_degrees, aspect, rig_tilt, rig_roll, scene)
            else {
                continue;
            };
            if !screen_point_usable(current) || !screen_point_usable(reference) {
                continue;
            }
            let Some(yaw_projected) = project_roi_point(
                anchor,
                yaw_pose,
                fov_degrees,
                aspect,
                rig_tilt,
                rig_roll,
                scene,
            ) else {
                continue;
            };
            let Some(pitch_projected) = project_roi_point(
                anchor,
                pitch_pose,
                fov_degrees,
                aspect,
                rig_tilt,
                rig_roll,
                scene,
            ) else {
                continue;
            };

            let j_yaw_x = (yaw_projected.x - current.x) / eps;
            let j_yaw_y = (yaw_projected.y - current.y) / eps;
            let j_pitch_x = (pitch_projected.x - current.x) / eps;
            let j_pitch_y = (pitch_projected.y - current.y) / eps;
            if !j_yaw_x.is_finite()
                || !j_yaw_y.is_finite()
                || !j_pitch_x.is_finite()
                || !j_pitch_y.is_finite()
            {
                continue;
            }

            let err_x = reference.x - current.x;
            let err_y = reference.y - current.y;
            a00 += j_yaw_x * j_yaw_x + j_yaw_y * j_yaw_y;
            a01 += j_yaw_x * j_pitch_x + j_yaw_y * j_pitch_y;
            a11 += j_pitch_x * j_pitch_x + j_pitch_y * j_pitch_y;
            b0 += j_yaw_x * err_x + j_yaw_y * err_y;
            b1 += j_pitch_x * err_x + j_pitch_y * err_y;
            used += 1;
        }

        if used < self.config.min_visible_points {
            return None;
        }

        let det = a00 * a11 - a01 * a01;
        if det.abs() < 1.0e-9 || !det.is_finite() {
            return None;
        }

        let max_correction = self.config.max_correction_rad;
        let strength = self.config.strength;
        Some(StabilizationCorrection {
            yaw: (((b0 * a11) - (b1 * a01)) / det * strength)
                .clamp(-max_correction, max_correction),
            pitch: (((a00 * b1) - (a01 * b0)) / det * strength)
                .clamp(-max_correction, max_correction),
        })
    }

    fn update_references(&mut self, current: &[Option<ScreenPoint>]) {
        let alpha = self.config.follow_alpha;
        for (reference, projected) in self.reference_points.iter_mut().zip(current.iter()) {
            let Some(projected) = *projected else {
                *reference = None;
                continue;
            };
            match reference {
                Some(reference) => {
                    reference.x += (projected.x - reference.x) * alpha;
                    reference.y += (projected.y - reference.y) * alpha;
                }
                None => *reference = Some(projected),
            }
        }
    }
}

impl ViewportStabilizer {
    pub(crate) fn set_track(&mut self, track: StabilizationTrack) {
        self.track = Some(track);
    }

    pub(crate) fn set_roi(&mut self, roi: RoiStabilizer) {
        self.roi = Some(roi);
    }

    pub(crate) fn stabilize_pose(
        &mut self,
        pose: ViewportPosition,
        ctx: StabilizationContext<'_>,
    ) -> (ViewportPosition, StabilizationCorrection) {
        let mut stabilized = pose;
        let mut total = StabilizationCorrection::default();

        if let Some(track) = self.track.as_ref() {
            let correction = track.correction_for_frame(ctx.frame_index);
            stabilized.yaw += correction.yaw;
            stabilized.pitch += correction.pitch;
            total.yaw += correction.yaw;
            total.pitch += correction.pitch;
        }

        if let Some(roi) = self.roi.as_mut() {
            let (pose, correction) = roi.stabilize_pose(
                stabilized,
                ctx.fov_degrees,
                ctx.aspect,
                ctx.rig_tilt,
                ctx.rig_roll,
                ctx.scene,
            );
            stabilized = pose;
            total.yaw += correction.yaw;
            total.pitch += correction.pitch;
        }

        (stabilized, total)
    }
}

fn project_roi_point(
    point: &[f64; 2],
    pose: ViewportPosition,
    fov_degrees: f32,
    aspect: f32,
    rig_tilt: f32,
    rig_roll: f32,
    scene: &SceneGeometry,
) -> Option<ScreenPoint> {
    crate::projection::panorama_to_viewport(
        point[0] as f32,
        point[1] as f32,
        pose.yaw,
        pose.pitch,
        fov_degrees,
        aspect,
        rig_tilt,
        rig_roll,
        scene,
    )
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .map(|(x, y)| ScreenPoint { x, y })
}

fn screen_point_usable(point: ScreenPoint) -> bool {
    (-MAX_ANCHOR_SCREEN_MARGIN..=1.0 + MAX_ANCHOR_SCREEN_MARGIN).contains(&point.x)
        && (-MAX_ANCHOR_SCREEN_MARGIN..=1.0 + MAX_ANCHOR_SCREEN_MARGIN).contains(&point.y)
}

#[derive(Clone, Copy)]
struct IntegratedPose {
    t: f64,
    yaw: f32,
    pitch: f32,
}

impl StabilizationTrack {
    /// Build a correction track from normalized gyro samples.
    ///
    /// `strength` is allowed to be negative so callers can flip the
    /// sign while testing camera-axis conventions without recompiling.
    pub fn from_gyro_samples(
        samples: &[GyroSample],
        fps: f64,
        output_start_secs: f64,
        output_duration_secs: Option<f64>,
        smoothing_window_secs: f64,
        max_correction_rad: f32,
        strength: f32,
    ) -> Result<Self, String> {
        if samples.len() < 2 {
            return Err("gyro stabilization needs at least two samples".into());
        }
        if fps <= 0.0 || !fps.is_finite() {
            return Err(format!(
                "gyro stabilization needs a positive fps, got {fps}"
            ));
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        let integrated = integrate_gyro(&sorted);
        if integrated.len() < 2 {
            return Err("gyro stabilization could not integrate a usable orientation path".into());
        }

        let start = output_start_secs.max(0.0);
        let available = (integrated.last().unwrap().t - start).max(0.0);
        let duration = output_duration_secs
            .filter(|d| d.is_finite() && *d > 0.0)
            .map_or(available, |d| d.min(available));
        if duration <= 0.0 {
            return Err("gyro stabilization has no samples in the requested output range".into());
        }

        let frame_count = (duration * fps).ceil().max(1.0) as usize;
        let mut raw = Vec::with_capacity(frame_count);
        for frame in 0..frame_count {
            let t = start + frame as f64 / fps;
            raw.push(sample_integrated(&integrated, t));
        }

        let half_window = ((smoothing_window_secs.max(0.0) * fps).round() as usize / 2).max(1);
        let mut corrections = Vec::with_capacity(raw.len());
        let max_correction = max_correction_rad.abs();
        for i in 0..raw.len() {
            let lo = i.saturating_sub(half_window);
            let hi = (i + half_window + 1).min(raw.len());
            let n = (hi - lo) as f32;
            let mut smooth_yaw = 0.0_f32;
            let mut smooth_pitch = 0.0_f32;
            for pose in &raw[lo..hi] {
                smooth_yaw += pose.yaw;
                smooth_pitch += pose.pitch;
            }
            smooth_yaw /= n;
            smooth_pitch /= n;

            corrections.push(StabilizationCorrection {
                yaw: ((smooth_yaw - raw[i].yaw) * strength).clamp(-max_correction, max_correction),
                pitch: ((smooth_pitch - raw[i].pitch) * strength)
                    .clamp(-max_correction, max_correction),
            });
        }

        Ok(Self { corrections })
    }

    /// Number of frame corrections in the track.
    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    /// Whether the track contains no corrections.
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    /// Correction for a rendered output frame. Missing frames receive no correction.
    pub fn correction_for_frame(&self, frame_index: u64) -> StabilizationCorrection {
        self.corrections
            .get(frame_index as usize)
            .copied()
            .unwrap_or_default()
    }
}

fn integrate_gyro(samples: &[GyroSample]) -> Vec<IntegratedPose> {
    let mut out = Vec::with_capacity(samples.len());
    let mut yaw = 0.0_f32;
    let mut pitch = 0.0_f32;
    out.push(IntegratedPose {
        t: samples[0].t,
        yaw,
        pitch,
    });

    for pair in samples.windows(2) {
        let prev = pair[0];
        let cur = pair[1];
        let dt = cur.t - prev.t;
        if dt <= 0.0 || dt > 0.5 || !dt.is_finite() {
            continue;
        }

        // Trapezoidal integration damps sample-to-sample noise a touch
        // without changing the shape of the high-frequency shake.
        let avg_yaw_rate = 0.5 * (prev.y + cur.y);
        let avg_pitch_rate = 0.5 * (prev.x + cur.x);
        yaw += (avg_yaw_rate * dt) as f32;
        pitch += (avg_pitch_rate * dt) as f32;

        out.push(IntegratedPose {
            t: cur.t,
            yaw,
            pitch,
        });
    }

    out
}

fn sample_integrated(samples: &[IntegratedPose], t: f64) -> IntegratedPose {
    let idx = samples.partition_point(|p| p.t <= t);
    if idx == 0 {
        return IntegratedPose {
            t,
            yaw: samples[0].yaw,
            pitch: samples[0].pitch,
        };
    }
    if idx >= samples.len() {
        let last = samples[samples.len() - 1];
        return IntegratedPose {
            t,
            yaw: last.yaw,
            pitch: last.pitch,
        };
    }

    let a = samples[idx - 1];
    let b = samples[idx];
    let span = b.t - a.t;
    if span <= 0.0 {
        return IntegratedPose {
            t,
            yaw: a.yaw,
            pitch: a.pitch,
        };
    }
    let f = ((t - a.t) / span).clamp(0.0, 1.0) as f32;
    IntegratedPose {
        t,
        yaw: a.yaw + (b.yaw - a.yaw) * f,
        pitch: a.pitch + (b.pitch - a.pitch) * f,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Calibration, FieldRoi, Framing, LShapeTopology, Lens};
    use crate::render::scene::SceneGeometry;

    fn test_scene() -> SceneGeometry {
        let lens = Lens {
            width: 3840,
            height: 2160,
            fx: 1796.32,
            fy: 1797.22,
            cx: 1919.37,
            cy: 1063.17,
            distortion: [0.0342, 0.0677, -0.0741, 0.0299],
            correction: 1.0,
        };
        let cal = Calibration {
            schema_version: 1,
            lenses: vec![lens.clone(), lens],
            topology: LShapeTopology {
                intersect: 0.5446,
                x_ty: 0.00476,
                x_rz: 0.00753,
                z_rx: -0.00431,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            }
            .into(),
            framing: Framing {
                axis_offset: 0.2398,
                tilt: 0.0,
                roll: 0.0,
            },
            sync_offset: 0,
            field_roi: None,
        };
        SceneGeometry::for_calibration(&cal, 16.0 / 9.0)
    }

    fn test_roi() -> FieldRoi {
        FieldRoi {
            points: vec![[-0.12, -0.05], [0.12, -0.05], [0.12, 0.07], [-0.12, 0.07]],
        }
    }

    #[test]
    fn zero_gyro_produces_zero_correction() {
        let samples: Vec<GyroSample> = (0..200)
            .map(|i| GyroSample {
                t: i as f64 / 100.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            })
            .collect();
        let track =
            StabilizationTrack::from_gyro_samples(&samples, 30.0, 0.0, Some(1.0), 0.5, 0.1, 1.0)
                .unwrap();

        assert!(!track.is_empty());
        for frame in 0..track.len() as u64 {
            assert_eq!(
                track.correction_for_frame(frame),
                StabilizationCorrection::default()
            );
        }
    }

    #[test]
    fn impulse_is_capped_and_can_flip_sign() {
        let samples: Vec<GyroSample> = (0..200)
            .map(|i| {
                let t = i as f64 / 100.0;
                GyroSample {
                    t,
                    x: 0.0,
                    y: if (0.9..=1.0).contains(&t) { 2.0 } else { 0.0 },
                    z: 0.0,
                }
            })
            .collect();
        let track =
            StabilizationTrack::from_gyro_samples(&samples, 30.0, 0.0, Some(2.0), 0.7, 0.01, 1.0)
                .unwrap();
        let flipped =
            StabilizationTrack::from_gyro_samples(&samples, 30.0, 0.0, Some(2.0), 0.7, 0.01, -1.0)
                .unwrap();

        let strongest = (0..track.len() as u64)
            .map(|frame| track.correction_for_frame(frame).yaw)
            .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
            .unwrap();
        assert!(strongest.abs() > 0.0);
        assert!(strongest.abs() <= 0.01);

        let same_frame = (0..track.len() as u64)
            .max_by(|a, b| {
                track
                    .correction_for_frame(*a)
                    .yaw
                    .abs()
                    .partial_cmp(&track.correction_for_frame(*b).yaw.abs())
                    .unwrap()
            })
            .unwrap();
        assert!(
            (track.correction_for_frame(same_frame).yaw
                + flipped.correction_for_frame(same_frame).yaw)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn roi_stabilization_requires_two_points() {
        let cfg = RoiStabilizationConfig::new(FieldRoi {
            points: vec![[0.0, 0.0]],
        });

        assert!(RoiStabilizer::new(cfg).is_err());
    }

    #[test]
    fn roi_stabilizer_cancels_short_pose_jitter() {
        let scene = test_scene();
        let cfg = RoiStabilizationConfig::new(test_roi())
            .with_follow_alpha(0.0)
            .with_max_correction_rad(0.1);
        let mut stabilizer = RoiStabilizer::new(cfg).unwrap();
        let base = ViewportPosition {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: Some(75.0),
        };

        let (_, first_correction) =
            stabilizer.stabilize_pose(base, 75.0, 16.0 / 9.0, 0.0, 0.0, &scene);
        assert_eq!(first_correction, StabilizationCorrection::default());

        let jittered = ViewportPosition {
            yaw: 0.02,
            pitch: -0.015,
            fov_degrees: Some(75.0),
        };
        let (stabilized, correction) =
            stabilizer.stabilize_pose(jittered, 75.0, 16.0 / 9.0, 0.0, 0.0, &scene);

        assert!(correction.yaw.abs() > 0.0);
        assert!(correction.pitch.abs() > 0.0);
        assert!(
            stabilized.yaw.abs() < jittered.yaw.abs() * 0.5,
            "expected yaw jitter to shrink: raw={} stabilized={}",
            jittered.yaw,
            stabilized.yaw
        );
        assert!(
            stabilized.pitch.abs() < jittered.pitch.abs() * 0.5,
            "expected pitch jitter to shrink: raw={} stabilized={}",
            jittered.pitch,
            stabilized.pitch
        );
    }
}
