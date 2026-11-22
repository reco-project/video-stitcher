//! Calibrated four-corner image support, shared by editors and detection consumers.
//!
//! Edges are the minor great-circle arcs between observed virtual-camera rays. They appear straight in a perspective view and generally curved in raw lens images. This defines image support, not a physical field pose or scale.

use nalgebra::{Matrix4, Perspective3, Vector3, Vector4};

use super::{SourceMapError, SourcePoint, SourceRays};
use crate::calibration::{Calibration, CalibrationError};
use crate::geometry::{FAR_PLANE, NEAR_PLANE, RenderPose, WorldPose, view_matrix};
use crate::render::viewport::ViewportSize;
use crate::stitch::{BlendRule, SurfaceMap};

fn dominant_source(
    maps: &[(Box<dyn SurfaceMap>, BlendRule)],
    ndc: [f64; 2],
) -> Option<SourcePoint> {
    let mut best = None;
    let mut best_weight = 0.0;
    for (camera, (map, blend)) in maps.iter().enumerate() {
        if let Some(uv) = map.sample_uv(ndc) {
            let alpha = blend.alpha(uv.edge);
            best_weight *= 1.0 - alpha;
            if alpha > best_weight {
                best_weight = alpha;
                best = Some(SourcePoint {
                    camera,
                    position: [uv.u, uv.v],
                });
            }
        }
    }
    best
}

fn valid_point(point: [f64; 2]) -> bool {
    point
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
}

fn projection(cal: &Calibration, size: &ViewportSize, pose: RenderPose) -> Option<Matrix4<f64>> {
    if size.width == 0
        || size.height == 0
        || ![pose.yaw, pose.pitch, pose.fov_degrees]
            .iter()
            .all(|v| v.is_finite())
        || !(1.0..=179.0).contains(&pose.fov_degrees)
    {
        return None;
    }
    let view = view_matrix(
        &cal.virtual_camera(),
        pose.yaw,
        pose.pitch,
        cal.framing.tilt as f32,
        cal.framing.roll as f32,
    );
    let perspective = Perspective3::new(
        size.width as f32 / size.height as f32,
        pose.render_fov().to_radians(),
        NEAR_PLANE,
        FAR_PLANE,
    );
    Some((perspective.to_homogeneous() * view).cast())
}

fn project_ray(matrix: &Matrix4<f64>, ray: &Vector3<f64>) -> Option<[f64; 2]> {
    let clip = matrix * ray.push(0.0);
    (clip.w > 1e-9).then(|| [0.5 * (clip.x / clip.w + 1.0), 0.5 * (1.0 - clip.y / clip.w)])
}

impl Calibration {
    /// Pick the strongest visible source contributor at a normalized top-left-origin viewport point.
    ///
    /// Uses the renderer's surface maps and compositing weights. Unsupported/black points return `None`, never a clamped source coordinate. The calibration must be validated.
    pub fn pick_source_point(
        &self,
        size: &ViewportSize,
        pose: RenderPose,
        point: [f64; 2],
    ) -> Option<SourcePoint> {
        if !valid_point(point) || projection(self, size, pose).is_none() {
            return None;
        }
        dominant_source(
            &self.surface_maps(size, pose),
            [2.0 * point[0] - 1.0, 1.0 - 2.0 * point[1]],
        )
    }

    /// Project a raw-source observation into a perspective viewport, independently of the field editor.
    ///
    /// Returns `None` behind the viewer; finite coordinates outside `[0,1]` remain outside rather than snapping to an edge. This locates an observation, not a guarantee it wins the seam blend. The calibration must be validated.
    pub fn project_source_point(
        &self,
        point: SourcePoint,
        size: &ViewportSize,
        pose: RenderPose,
    ) -> Result<Option<[f64; 2]>, SourceMapError> {
        if !valid_point(point.position) {
            return Err(SourceMapError::InvalidPoint);
        }
        if point.camera >= self.projection.camera_count() || point.camera >= self.lenses.len() {
            return Err(SourceMapError::InvalidCamera);
        }
        let map =
            self.projection
                .geometry()
                .source_rays(point.camera, &self.lenses, &self.framing)?;
        let ray = map(point.position).ok_or(SourceMapError::NoDirection)?;
        Ok(projection(self, size, pose).and_then(|matrix| project_ray(&matrix, &ray)))
    }

    /// Prepare the document's tracking boundary for allocation-free source-point tests.
    ///
    /// Rebuild after changing corners, optics, projection, framing or blending. `None` means no region. Four supported, distinct observations may be supplied in any order; all four must be vertices of a nondegenerate convex region. Corners need not share the hemisphere centered on their mean ray. The document's observation order is preserved.
    pub fn field_boundary(&self) -> Result<Option<FieldBoundary>, CalibrationError> {
        let Some(roi) = &self.field_roi else {
            return Ok(None);
        };
        let invalid = |reason: &str| CalibrationError::InvalidFieldRegion(reason.into());
        if self.lenses.len() != self.projection.camera_count() {
            return Err(invalid("lens count does not match the projection"));
        }
        let maps = (0..self.projection.camera_count())
            .map(|camera| {
                self.projection
                    .geometry()
                    .source_rays(camera, &self.lenses, &self.framing)
                    .map_err(|e| invalid(&e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut rays = [Vector3::zeros(); 4];
        for (ray, point) in rays.iter_mut().zip(roi.corners) {
            if !valid_point(point.position) {
                return Err(invalid("corners must be within their source images"));
            }
            *ray = maps
                .get(point.camera)
                .and_then(|map| map(point.position))
                .ok_or_else(|| invalid("corner has no supported source ray"))?;
        }
        let center = rays
            .iter()
            .copied()
            .sum::<Vector3<f64>>()
            .try_normalize(1e-9)
            .ok_or_else(|| invalid("corners do not enclose a finite region"))?;
        // Four observations have only three distinct cycles up to rotation and
        // reversal. Test supporting great-circle edges directly, without a
        // perspective chart that would exclude wide fields. Every observation
        // must remain a vertex; never turn an interior point into a triangle.
        let (rays, normals) = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3]]
            .into_iter()
            .find_map(|order| {
                let ordered = order.map(|i| rays[i]);
                let mut normals = [Vector3::zeros(); 4];
                for i in 0..4 {
                    let mut normal = ordered[i]
                        .cross(&ordered[(i + 1) % 4])
                        .try_normalize(1e-9)?;
                    if normal.dot(&center) < 0.0 {
                        normal = -normal;
                    }
                    if (2..4).any(|offset| normal.dot(&ordered[(i + offset) % 4]) <= 1e-8) {
                        return None;
                    }
                    normals[i] = normal;
                }
                Some((ordered, normals))
            })
            .ok_or_else(|| invalid("select four distinct corners enclosing a convex region"))?;
        let angles = self.virtual_camera().direction_to_yaw_pitch(&center.cast());
        let pose = self.world_to_render_pose(WorldPose {
            yaw: angles.yaw,
            pitch: angles.pitch,
            fov_degrees: 90.0,
        });
        let size = ViewportSize {
            width: 1,
            height: 1,
        };
        let reference = projection(self, &size, pose).ok_or_else(|| invalid("invalid framing"))?;
        Ok(Some(FieldBoundary {
            maps,
            rays,
            normals,
            reference,
            panorama: if let crate::calibration::Projection::Panorama(pan) = &self.projection {
                Some((
                    view_matrix(
                        &self.virtual_camera(),
                        0.0,
                        0.0,
                        self.framing.tilt as f32,
                        self.framing.roll as f32,
                    )
                    .cast(),
                    pan.arc_deg.to_radians() * 0.5,
                    (pan.vertical_fov_deg.to_radians() * 0.5).tan(),
                ))
            } else {
                None
            },
            surfaces: self
                .projection
                .geometry()
                .surface_maps(&super::ProjectionContext {
                    lenses: &self.lenses,
                    framing: &self.framing,
                    viewport_size: &size,
                    pose,
                    clip_depth: false,
                }),
        }))
    }
}

/// Derived common field boundary; no per-camera polygon approximations or image masks.
pub struct FieldBoundary {
    maps: Vec<SourceRays>,
    rays: [Vector3<f64>; 4],
    normals: [Vector3<f64>; 4],
    reference: Matrix4<f64>,
    panorama: Option<(Matrix4<f64>, f64, f64)>,
    surfaces: Vec<(Box<dyn SurfaceMap>, BlendRule)>,
}

impl FieldBoundary {
    /// Whether this camera is the strongest contributor at the observation's calibrated ray.
    ///
    /// Uses ordered surface blending, not the field's halfway line or view-dependent depth clipping. Test a detection's primary anchor once, independently of its other field-containment anchors, so boxes crossing the seam are not dropped by both cameras.
    pub fn owns(&self, point: SourcePoint) -> bool {
        if !valid_point(point.position) {
            return false;
        }
        let Some(ray) = self
            .maps
            .get(point.camera)
            .and_then(|map| map(point.position))
        else {
            return false;
        };
        if self.maps.len() == 1 {
            return true;
        }
        let ndc = if let Some((view, half_arc, tan_half_v)) = &self.panorama {
            let local = view * ray.push(0.0);
            [
                local.x.atan2(-local.z) / half_arc,
                local.y / (local.x.hypot(local.z) * tan_half_v),
            ]
        } else {
            let Some(p) = project_ray(&self.reference, &ray) else {
                return false;
            };
            [2.0 * p[0] - 1.0, 1.0 - 2.0 * p[1]]
        };
        dominant_source(&self.surfaces, ndc).is_some_and(|owner| owner.camera == point.camera)
    }

    /// Test a source observation against all four calibrated edges, including the boundary itself.
    ///
    /// Invalid, unsupported and unknown-camera points are outside. This operates on coordinates only, identically for CPU and GPU inference results, without allocations per query.
    pub fn contains(&self, point: SourcePoint) -> bool {
        valid_point(point.position)
            && self
                .maps
                .get(point.camera)
                .and_then(|map| map(point.position))
                .is_some_and(|ray| self.normals.iter().all(|normal| normal.dot(&ray) >= -1e-9))
    }

    /// Clip the outline to the current perspective view using the calibration that prepared this boundary.
    ///
    /// Coordinates are normalized with a top-left origin. This display-only result must not be saved as a new ROI. Empty means the region is outside the view. Pan, zoom, aspect and framing affect only this projection.
    pub fn outline(
        &self,
        calibration: &Calibration,
        size: &ViewportSize,
        pose: RenderPose,
    ) -> Vec<[f64; 2]> {
        let Some(matrix) = projection(calibration, size, pose) else {
            return vec![];
        };
        let mut vertices = self.rays.map(|ray| matrix * ray.push(0.0)).to_vec();
        // Homogeneous side-plane clipping also rejects rays behind the view without projecting through infinity.
        for plane in [
            Vector4::new(1.0, 0.0, 0.0, 1.0),
            Vector4::new(-1.0, 0.0, 0.0, 1.0),
            Vector4::new(0.0, 1.0, 0.0, 1.0),
            Vector4::new(0.0, -1.0, 0.0, 1.0),
        ] {
            let mut clipped = Vec::with_capacity(vertices.len() + 1);
            for i in 0..vertices.len() {
                let a = vertices[i];
                let b = vertices[(i + 1) % vertices.len()];
                let da = plane.dot(&a);
                let db = plane.dot(&b);
                if da >= 0.0 {
                    clipped.push(a);
                }
                if (da >= 0.0) != (db >= 0.0) {
                    clipped.push(a + (b - a) * (da / (da - db)));
                }
            }
            vertices = clipped;
        }
        vertices
            .into_iter()
            .filter(|p| p.w > 1e-9)
            .map(|p| [0.5 * (p.x / p.w + 1.0), 0.5 * (1.0 - p.y / p.w)])
            .collect()
    }
}

#[cfg(test)]
mod tests;
