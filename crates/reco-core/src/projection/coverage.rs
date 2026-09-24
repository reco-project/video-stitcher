//! Continuous source support and constrained presentation motion.

mod boundary;
pub(crate) mod controller;
mod motion;
#[cfg(test)]
mod rendering;
mod roof;

#[cfg(test)]
use super::{Cylinder, LShape};
#[cfg(test)]
use crate::calibration::{Framing, Lens};
use crate::{
    calibration::Calibration,
    geometry::RenderPose,
    render::viewport::ViewportSize,
    stitch::{BlendRule, SurfaceMap},
};

#[cfg(test)]
fn covered(surfaces: &[(Box<dyn SurfaceMap>, BlendRule)], size: &ViewportSize) -> (bool, usize) {
    let mut stack = vec![([0, 0], [size.width - 1, size.height - 1])];
    let mut work = 0;
    while let Some((lo, hi)) = stack.pop() {
        work += 1;
        let (x, y) = super::interval::ndc_rect(lo, hi, size.width as f64, size.height as f64);
        if surfaces
            .iter()
            .any(|(s, _)| s.covers_ndc([x.lo, y.lo], [x.hi, y.hi], 0.0))
        {
            continue;
        }
        for point in [
            lo,
            hi,
            [lo[0], hi[1]],
            [hi[0], lo[1]],
            [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2],
        ] {
            if !surfaces
                .iter()
                .any(|(s, _)| s.sample_uv(size.pixel_ndc(point[0], point[1])).is_some())
            {
                return (false, work);
            }
        }
        if lo == hi {
            continue;
        }
        let axis = usize::from(hi[1] - lo[1] > hi[0] - lo[0]);
        let middle = lo[axis] + (hi[axis] - lo[axis]) / 2;
        let mut end = hi;
        end[axis] = middle;
        let mut start = lo;
        start[axis] = middle + 1;
        stack.push((start, hi));
        stack.push((lo, end));
    }
    (true, work)
}

#[cfg(test)]
fn exhaustive(surfaces: &[(Box<dyn SurfaceMap>, BlendRule)], size: &ViewportSize) -> usize {
    (0..size.height)
        .flat_map(|y| (0..size.width).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            surfaces
                .iter()
                .all(|(s, _)| s.sample_uv(size.pixel_ndc(x, y)).is_none())
        })
        .count()
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Certificate {
    /// Bounds establish support at every point of the queried rectangle.
    Covered,
    /// At least one point has no supporting source.
    Uncovered,
    /// The numerical domain or work limit prevented a conclusion.
    Unknown,
}

#[cfg(test)]
fn certify(surfaces: &[(Box<dyn SurfaceMap>, BlendRule)], rotation: f64) -> Certificate {
    certify_rect(surfaces, [-1.0; 2], [1.0; 2], rotation, 2048)
}

#[cfg(test)]
fn certify_rect(
    surfaces: &[(Box<dyn SurfaceMap>, BlendRule)],
    min: [f64; 2],
    max: [f64; 2],
    rotation: f64,
    budget: usize,
) -> Certificate {
    certify_regions(surfaces, &[(min, max)], rotation, budget)
}

fn certify_regions(
    surfaces: &[(Box<dyn SurfaceMap>, BlendRule)],
    regions: &[([f64; 2], [f64; 2])],
    rotation: f64,
    budget: usize,
) -> Certificate {
    if budget == 0
        || regions.is_empty()
        || !rotation.is_finite()
        || !(0.0..=std::f64::consts::PI).contains(&rotation)
        || regions.iter().any(|(min, max)| {
            (0..2).any(|i| !min[i].is_finite() || !max[i].is_finite() || min[i] > max[i])
        })
    {
        return Certificate::Unknown;
    }
    let mut stack: Vec<_> = regions
        .iter()
        .map(|&(min, max)| (min, max, 0_u32))
        .collect();
    let mut work = 0;
    while let Some((lo, hi, depth)) = stack.pop() {
        if work == budget {
            return Certificate::Unknown;
        }
        work += 1;
        if surfaces.iter().any(|(s, _)| s.covers_ndc(lo, hi, rotation)) {
            continue;
        }
        for p in [
            lo,
            hi,
            [lo[0], hi[1]],
            [hi[0], lo[1]],
            [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5],
        ] {
            if surfaces.iter().all(|(s, _)| s.sample_uv(p).is_none()) {
                return Certificate::Uncovered;
            }
        }
        if depth >= 40 {
            return Certificate::Unknown;
        }
        let axis = usize::from(hi[1] - lo[1] > hi[0] - lo[0]);
        let middle = (lo[axis] + hi[axis]) * 0.5;
        let mut end = hi;
        end[axis] = middle;
        let mut start = lo;
        start[axis] = middle;
        stack.push((start, hi, depth + 1));
        stack.push((lo, end, depth + 1));
    }
    Certificate::Covered
}

#[cfg(test)]
fn certify_pose(cal: &Calibration, size: &ViewportSize, pose: RenderPose) -> Certificate {
    if !valid_query(cal, size, pose) {
        return Certificate::Unknown;
    }
    let maps = cal.surface_maps(size, pose);
    certify(&maps, 0.0)
}

fn valid_query(cal: &Calibration, size: &ViewportSize, pose: RenderPose) -> bool {
    cal.validate().is_ok()
        && size.validate().is_ok()
        && pose.yaw.is_finite()
        && pose.pitch.is_finite()
        && (1.0..=179.0).contains(&pose.fov_degrees)
}

fn interpolate(a: RenderPose, b: RenderPose, t: f32) -> RenderPose {
    RenderPose {
        yaw: a.yaw + (b.yaw - a.yaw) * t,
        pitch: a.pitch + (b.pitch - a.pitch) * t,
        fov_degrees: a.fov_degrees + (b.fov_degrees - a.fov_degrees) * t,
    }
}

#[cfg(test)]
fn certify_sweep_enclosure(
    cal: &Calibration,
    size: &ViewportSize,
    a: RenderPose,
    b: RenderPose,
    rectangles: usize,
) -> Certificate {
    let rotation = f64::from((b.yaw - a.yaw).abs() + (b.pitch - a.pitch).abs()) * 0.5;
    let mid = RenderPose {
        fov_degrees: a.fov_degrees.max(b.fov_degrees),
        ..interpolate(a, b, 0.5)
    };
    certify_orientation_enclosure(cal, size, mid, rotation, rectangles)
}

#[cfg(test)]
fn certify_orientation_enclosure(
    cal: &Calibration,
    size: &ViewportSize,
    mid: RenderPose,
    rotation: f64,
    rectangles: usize,
) -> Certificate {
    let rotation = rotation + 2e-6;
    let aspect = f64::from(size.aspect_ratio());
    let v = (f64::from(mid.fov_degrees).to_radians() * 0.5).tan();
    let h = v * aspect;
    let e = 2.0 * (rotation * 0.5).sin() * (1.0 + h * h + v * v).sqrt();
    if e >= 1.0 || rotation > 0.5 {
        return Certificate::Unknown;
    }
    let x = (h + e) / (h * (1.0 - e)) * (1.0 + 1e-6);
    let y = (v + e) / (v * (1.0 - e)) * (1.0 + 1e-6);
    let maps = cal.surface_maps(size, mid);
    // Temporal subdivision cannot compensate for an insufficient spatial
    // budget, even when both endpoints are the same covered pose.
    certify_rect(&maps, [-x, -y], [x, y], rotation, rectangles)
}

#[cfg(test)]
fn fit_fov(cal: &Calibration, size: &ViewportSize, requested: RenderPose) -> Option<RenderPose> {
    if !valid_query(cal, size, requested) {
        return None;
    }
    if certify_pose(cal, size, requested) == Certificate::Covered {
        return Some(requested);
    }
    let mut fitted = RenderPose {
        fov_degrees: 1.0,
        ..requested
    };
    if certify_pose(cal, size, fitted) != Certificate::Covered {
        return None;
    }
    let mut high = requested.fov_degrees;
    for _ in 0..16 {
        let candidate = RenderPose {
            fov_degrees: (fitted.fov_degrees + high) * 0.5,
            ..requested
        };
        if certify_pose(cal, size, candidate) == Certificate::Covered {
            fitted = candidate;
        } else {
            high = candidate.fov_degrees;
        }
    }
    Some(fitted)
}

#[cfg(test)]
fn stereo() -> Calibration {
    let l = Lens::fisheye(
        3840,
        2160,
        1796.32,
        1797.22,
        1919.37,
        1063.17,
        [0.0342, 0.0677, -0.0741, 0.0299],
    );
    Calibration::new(
        vec![l.clone(), l],
        LShape {
            intersect: 0.5446,
            x_ty: 0.00476,
            x_rz: 0.00753,
            z_rx: -0.00431,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.0,
        },
        Framing {
            axis_offset: 0.2398,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

#[cfg(test)]
mod tests;
