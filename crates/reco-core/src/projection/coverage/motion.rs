//! Bounded motion and contact-limited zoom against a required coverage map.

use super::*;
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

mod braking;
mod routing;
pub(super) use routing::Route;
#[cfg(test)]
mod diagnostics;
#[cfg(test)]
mod motion_tests;
#[cfg(test)]
mod scoring_tests;
#[cfg(test)]
mod sweep;

#[cfg(all(test, feature = "gpu"))]
mod rendering;

const RATE: f32 = 0.9;
// Limits apply to yaw/pitch coordinates at a fixed controller timestep.
const ACCELERATION: f32 = 0.6;
// Vertical FOV degrees per second. A discrete request must not teleport zoom.
const ZOOM_RATE: f32 = 30.0;
const ZOOM_TOLERANCE: f64 = 0.002;
const CONTACT_TOLERANCE: f32 = 1e-6;

fn view(cal: &Calibration, p: RenderPose) -> Matrix4<f32> {
    let camera = cal.virtual_camera();
    crate::geometry::view_matrix(
        &camera,
        p.yaw,
        p.pitch,
        cal.framing.tilt as f32,
        cal.framing.roll as f32,
    )
}

#[cfg(test)]
fn direction(cal: &Calibration, p: RenderPose) -> Vector3<f32> {
    view_direction(&view(cal, p))
}

fn view_direction(view: &Matrix4<f32>) -> Vector3<f32> {
    let inverse = view.try_inverse().unwrap();
    (inverse * Vector4::new(0.0, 0.0, -1.0, 0.0))
        .xyz()
        .normalize()
}

fn aim(cal: &Calibration, direction: &Vector3<f32>, fov_degrees: f32) -> RenderPose {
    let camera = cal.virtual_camera();
    let world = camera.direction_to_yaw_pitch(direction);
    let (yaw, pitch) = crate::geometry::world_to_render_angles(
        &camera,
        world.yaw,
        world.pitch,
        cal.framing.tilt as f32,
        cal.framing.roll as f32,
    );
    RenderPose {
        yaw,
        pitch,
        fov_degrees,
    }
}

#[cfg(test)]
fn screen(
    cal: &Calibration,
    size: &ViewportSize,
    p: RenderPose,
    target: &Vector3<f32>,
) -> Option<[f64; 2]> {
    let ray = view(cal, p) * Vector4::new(target.x, target.y, target.z, 0.0);
    project_ray(ray, screen_scale(size, p.fov_degrees))
}

fn screen_scale(size: &ViewportSize, fov: f32) -> [f64; 2] {
    let vertical = f64::from(fov).to_radians().mul_add(0.5, 0.0).tan();
    [vertical * f64::from(size.aspect_ratio()), vertical]
}

fn project_ray(ray: Vector4<f32>, scale: [f64; 2]) -> Option<[f64; 2]> {
    if ray.z >= -1e-6 {
        return None;
    }
    Some([
        f64::from(ray.x / -ray.z) / scale[0],
        f64::from(ray.y / -ray.z) / scale[1],
    ])
}

// Lexicographic: behind-camera class, visibility error, centering error.
// Keep the class separate so it cannot round away small angular improvements.
fn score(cal: &Calibration, size: &ViewportSize, p: RenderPose, target: &Vector3<f32>) -> [f64; 3] {
    let view = view(cal, p);
    score_view(&view, screen_scale(size, p.fov_degrees), target, || {
        view_direction(&view).cast()
    })
}

fn score_view(
    view: &Matrix4<f32>,
    scale: [f64; 2],
    target: &Vector3<f32>,
    forward: impl FnOnce() -> Vector3<f64>,
) -> [f64; 3] {
    let ray = view * Vector4::new(target.x, target.y, target.z, 0.0);
    let Some([x, y]) = project_ray(ray, scale) else {
        let forward = forward();
        let target = target.cast::<f64>();
        // atan2 retains angular progress even near the antipodal direction,
        // where a cosine difference can round to zero.
        let error = forward.cross(&target).norm().atan2(forward.dot(&target));
        return [1.0, error, error];
    };
    let outside = (x.abs() - 0.96).max(0.0).powi(2) + (y.abs() - 0.96).max(0.0).powi(2);
    [
        0.0,
        if outside > 1e-10 { outside } else { 0.0 },
        x * x + y * y,
    ]
}

fn better<const N: usize>(a: [f64; N], b: [f64; N]) -> bool {
    a.iter()
        .zip(b)
        .map(|(a, b)| a.total_cmp(&b))
        .find(|order| !order.is_eq())
        .is_some_and(|order| order.is_lt())
}

pub(super) fn distance(a: RenderPose, b: RenderPose) -> f32 {
    (a.yaw - b.yaw).hypot(a.pitch - b.pitch)
}

pub(super) fn initial(
    cal: &Calibration,
    size: &ViewportSize,
    target: &Vector3<f32>,
    fov: f32,
    roof: &roof::Roof,
    route: &Route,
) -> Option<RenderPose> {
    if !valid_query(
        cal,
        size,
        RenderPose {
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: fov,
        },
    ) || !target.iter().all(|v| v.is_finite())
        || (target.norm_squared() - 1.0).abs() > 1e-4
    {
        return None;
    }
    let requested = aim(cal, target, fov);
    if !requested.yaw.is_finite() || !requested.pitch.is_finite() {
        return None;
    }
    if roof.contains(requested) {
        return Some(requested);
    }
    let scale = screen_scale(size, fov);
    roof.supported_vertices(fov)
        .map(|(i, p)| (p, route.score(i, scale, target)))
        .min_by(|(_, a), (_, b)| {
            a[0].total_cmp(&b[0])
                .then_with(|| a[1].total_cmp(&b[1]))
                .then_with(|| a[2].total_cmp(&b[2]))
        })
        .map(|(p, _)| p)
}

pub(super) struct Advance {
    pub(super) pose: RenderPose,
    pub(super) velocity: [f32; 2],
    pub(super) support_queries: usize,
    pub(super) objective: [f64; 4],
}

#[derive(Clone, Copy)]
pub(super) struct MotionWindow {
    pub(super) previous: RenderPose,
    pub(super) velocity: [f32; 2],
    pub(super) dt: f32,
}

fn valid_motion(
    cal: &Calibration,
    size: &ViewportSize,
    motion: &MotionWindow,
    target: &Vector3<f32>,
    fov: f32,
) -> bool {
    motion.dt.is_finite()
        && motion.dt > 0.0
        && valid_query(cal, size, motion.previous)
        && (1.0..=179.0).contains(&fov)
        && motion.velocity.iter().all(|v| v.is_finite())
        && motion.velocity[0].hypot(motion.velocity[1]) <= RATE + 1e-5
        && target.iter().all(|v| v.is_finite())
        && (target.norm_squared() - 1.0).abs() <= 1e-4
        && (ACCELERATION * motion.dt * motion.dt).is_finite()
}

impl MotionWindow {
    fn admits(&self, p: RenderPose) -> bool {
        let v = self.velocity_at(p);
        v[0].hypot(v[1]) <= RATE + 1e-5
            && (v[0] - self.velocity[0]).hypot(v[1] - self.velocity[1]) / self.dt
                <= ACCELERATION + 0.001
    }
    fn stopping_distance(&self, speed: f32) -> f32 {
        let decrement = ACCELERATION * self.dt;
        let steps = (speed / decrement).floor();
        self.dt * (steps * speed - decrement * steps * (steps + 1.0) * 0.5)
    }

    fn aim_step(&self, target: RenderPose) -> RenderPose {
        let distance = distance(self.previous, target);
        if distance < 1e-8 {
            return self.project(target);
        }
        let (mut lo, mut hi) = (0.0, RATE);
        for _ in 0..24 {
            let speed = (lo + hi) * 0.5;
            if self.dt * speed + self.stopping_distance(speed) < distance {
                lo = speed;
            } else {
                hi = speed;
            }
        }
        self.project(interpolate(self.previous, target, self.dt * lo / distance))
    }

    fn velocity_at(&self, p: RenderPose) -> [f32; 2] {
        [
            (p.yaw - self.previous.yaw) / self.dt,
            (p.pitch - self.previous.pitch) / self.dt,
        ]
    }

    fn stopping_pose(&self, p: RenderPose) -> RenderPose {
        let v = self.velocity_at(p);
        let speed = v[0].hypot(v[1]);
        if speed < 1e-8 {
            return p;
        }
        let distance = self.stopping_distance(speed);
        RenderPose {
            yaw: p.yaw + v[0] * distance / speed,
            pitch: p.pitch + v[1] * distance / speed,
            ..p
        }
    }

    fn braking_step(&self) -> RenderPose {
        let speed = self.velocity[0].hypot(self.velocity[1]);
        let scale = if speed > 0.0 {
            (1.0 - ACCELERATION * self.dt / speed).max(0.0)
        } else {
            0.0
        };
        let pose = RenderPose {
            yaw: self.previous.yaw + self.velocity[0] * scale * self.dt,
            pitch: self.previous.pitch + self.velocity[1] * scale * self.dt,
            ..self.previous
        };
        if self.admits(pose) {
            pose
        } else {
            self.project(pose)
        }
    }

    // Euclidean projection onto the intersection of speed and acceleration disks.
    fn project(&self, p: RenderPose) -> RenderPose {
        let wanted = Vector2::new(
            f64::from(p.yaw - self.previous.yaw),
            f64::from(p.pitch - self.previous.pitch),
        );
        let centre = Vector2::new(
            f64::from(self.velocity[0] * self.dt),
            f64::from(self.velocity[1] * self.dt),
        );
        let r = f64::from(RATE * self.dt);
        let s = f64::from(ACCELERATION * self.dt * self.dt);
        let within = |v: Vector2<f64>| v.norm() <= r + 1e-12 && (v - centre).norm() <= s + 1e-12;
        let on_disk = |v: Vector2<f64>, c: Vector2<f64>, radius: f64| {
            let delta = v - c;
            c + delta * (radius / delta.norm().max(radius)).min(1.0)
        };
        let mut candidates = vec![
            wanted,
            on_disk(wanted, Vector2::zeros(), r),
            on_disk(wanted, centre, s),
        ];
        let d = centre.norm();
        if d > 1e-12 && d <= r + s && d >= (r - s).abs() {
            let along = (r * r - s * s + d * d) / (2.0 * d);
            let cross = (r * r - along * along).max(0.0).sqrt();
            let axis = centre / d;
            let normal = Vector2::new(-axis.y, axis.x);
            candidates.extend([axis * along + normal * cross, axis * along - normal * cross]);
        }
        let q = candidates
            .into_iter()
            .filter(|v| within(*v))
            .min_by(|a, b| {
                (a - wanted)
                    .norm_squared()
                    .total_cmp(&(b - wanted).norm_squared())
            })
            .unwrap();
        let rounded = RenderPose {
            yaw: self.previous.yaw + q.x as f32,
            pitch: self.previous.pitch + q.y as f32,
            ..self.previous
        };
        if self.admits(rounded) {
            return rounded;
        }
        // A continuous disk contact may round outside the acceleration bound.
        // Select a feasible adjacent f32 pose instead of relaxing that bound.
        [rounded.yaw.next_down(), rounded.yaw, rounded.yaw.next_up()]
            .into_iter()
            .flat_map(|yaw| {
                [
                    rounded.pitch.next_down(),
                    rounded.pitch,
                    rounded.pitch.next_up(),
                ]
                .into_iter()
                .map(move |pitch| RenderPose {
                    yaw,
                    pitch,
                    ..rounded
                })
            })
            .filter(|p| self.admits(*p))
            .min_by(|a, b| {
                let cost = |p: &RenderPose| {
                    (f64::from(p.yaw) - f64::from(self.previous.yaw) - q.x).powi(2)
                        + (f64::from(p.pitch) - f64::from(self.previous.pitch) - q.y).powi(2)
                };
                cost(a).total_cmp(&cost(b))
            })
            .unwrap_or(rounded)
    }
}

pub(super) fn solve_motion(
    cal: &Calibration,
    size: &ViewportSize,
    motion: MotionWindow,
    target: &Vector3<f32>,
    requested_fov: f32,
    roof: &roof::Roof,
    route: &mut Route,
) -> Option<Advance> {
    let previous = motion.previous;
    let dt = motion.dt;
    if !valid_motion(cal, size, &motion, target, requested_fov) {
        return None;
    }
    let zoom_step = ZOOM_RATE * dt;
    let fov = requested_fov.clamp(
        previous.fov_degrees - zoom_step,
        previous.fov_degrees + zoom_step,
    );
    let floor = previous.fov_degrees;
    let window = MotionWindow {
        previous: RenderPose {
            fov_degrees: fov,
            ..previous
        },
        ..motion
    };
    let mut guide = None;
    let mut wanted = aim(cal, target, fov);
    wanted.yaw = previous.yaw
        + (wanted.yaw - previous.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    if route.active(target, floor) || !roof.segment(previous, wanted) {
        guide = route.guide(cal, size, previous, target, roof);
        if let Some(p) = guide {
            wanted = p;
        }
    }
    let objective = |p: RenderPose| {
        if let Some(guide) = guide {
            let rest = window.stopping_pose(p);
            return [
                (f64::from(fov) - f64::from(p.fov_degrees)).abs(),
                0.0,
                0.0,
                f64::from(rest.yaw - guide.yaw).powi(2)
                    + f64::from(rest.pitch - guide.pitch).powi(2),
            ];
        }
        let now = score(cal, size, p, target);
        let rest = score(cal, size, window.stopping_pose(p), target);
        // Take the worse visibility pair, not independent maxima from
        // incompatible front/behind error scales.
        let visibility = if better([now[0], now[1]], [rest[0], rest[1]]) {
            rest
        } else {
            now
        };
        [
            (f64::from(fov) - f64::from(p.fov_degrees)).abs(),
            visibility[0],
            visibility[1],
            rest[2],
        ]
    };
    // Sub-millidegree proof/search noise must not dominate target placement.
    let improves = |a: [f64; 4], b: [f64; 4]| {
        if (a[0] - b[0]).abs() > ZOOM_TOLERANCE {
            a[0] < b[0]
        } else {
            better([a[1], a[2], a[3]], [b[1], b[2], b[3]])
        }
    };
    let requested = window.aim_step(wanted);
    let brake = window.braking_step();
    let radius = RATE * dt;
    let mut queries = 0;
    let permitted = |p: RenderPose| roof.contains(p);
    let certified = |p: RenderPose| {
        if !window.admits(p) {
            return false;
        }
        let Some(path) = braking::path(&window, p) else {
            return false;
        };
        if !permitted(p) || !permitted(*path.last().unwrap()) {
            return false;
        }
        roof.segment(previous, p) && path.windows(2).all(|pair| roof.segment(pair[0], pair[1]))
    };
    // The unconstrained controller already supplies a bounded aiming step.
    // Only search contacts when coverage actually prevents that move.
    queries += 1;
    if certified(requested) {
        return Some(Advance {
            pose: requested,
            velocity: window.velocity_at(requested),
            support_queries: queries,
            objective: objective(requested),
        });
    }
    let preliminary = |p: RenderPose| {
        let p = RenderPose {
            fov_degrees: floor.min(fov),
            ..p
        };
        permitted(p) && permitted(window.stopping_pose(p))
    };
    // Fit zoom at each candidate position, including the real movement and
    // a stopping continuation. Holding FOV is allowed; teleporting is not.
    let fit = |mut p: RenderPose| {
        if !window.admits(p) {
            return None;
        }
        let path = braking::path(&window, p)?;
        // Zoom changes neither the braking positions nor their constant-FOV
        // ceiling. Compute that ceiling once for all trials at this position.
        let ceiling = path.windows(2).try_fold(f64::INFINITY, |ceiling, pair| {
            roof.segment_ceiling(pair[0], pair[1])
                .map(|value| ceiling.min(value))
        })?;
        let supported = |p: RenderPose| {
            f64::from(p.fov_degrees) + roof::FOV_EPSILON <= ceiling && roof.segment(previous, p)
        };
        if supported(p) {
            return Some(p);
        }
        if fov == floor {
            return None;
        }
        p.fov_degrees = floor;
        if !supported(p) {
            return None;
        }
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..10 {
            if (hi - lo) * (fov - floor).abs() < 0.001 {
                break;
            }
            let mid = (lo + hi) * 0.5;
            let candidate = RenderPose {
                fov_degrees: floor + mid * (fov - floor),
                ..p
            };
            if supported(candidate) {
                lo = mid;
                p = candidate;
            } else {
                hi = mid;
            }
        }
        Some(p)
    };
    let mut best = None;
    let mut best_score = [f64::INFINITY; 4];
    for p in [brake, requested] {
        queries += 1;
        if let Some(p) = fit(p) {
            let cost = objective(p);
            if improves(cost, best_score) {
                best = Some(p);
                best_score = cost;
            }
        }
    }
    // A stationary/straight-braking continuation may be unavailable at a
    // corner; search the reachable disk before giving up.
    if best.is_none() {
        for i in 0..64 {
            let angle = i as f32 * std::f32::consts::TAU / 64.0;
            queries += 1;
            if let Some(p) = fit(window.project(RenderPose {
                yaw: previous.yaw + radius * angle.cos(),
                pitch: previous.pitch + radius * angle.sin(),
                ..window.previous
            })) {
                let cost = objective(p);
                if improves(cost, best_score) {
                    best = Some(p);
                    best_score = cost;
                }
            }
        }
    }
    let mut best = best?;
    let mut step = 2.0 * ACCELERATION * dt * dt;
    for _ in 0..12 {
        if best_score[0] == 0.0
            && best_score[1] == 0.0
            && best_score[2] == 0.0
            && best_score[3] < 1e-8
        {
            break;
        }
        let ring: Vec<_> = (0..8)
            .map(|i| {
                let angle = i as f32 * std::f32::consts::TAU / 8.0;
                let p = window.project(RenderPose {
                    yaw: best.yaw + step * angle.cos(),
                    pitch: best.pitch + step * angle.sin(),
                    ..window.previous
                });
                queries += 1;
                (p, preliminary(p))
            })
            .collect();
        let mut consider = |p: RenderPose| {
            if !improves(objective(p), best_score) {
                return;
            }
            if let Some(p) = fit(p) {
                let cost = objective(p);
                if improves(cost, best_score) {
                    best = p;
                    best_score = cost;
                }
            }
        };
        for &(p, possible) in &ring {
            if possible {
                consider(p);
            }
        }
        // Refine feasible/infeasible contacts to avoid angular quantization.
        for i in 0..ring.len() {
            let (a, a_fit) = ring[i];
            let (b, b_fit) = ring[(i + 1) % ring.len()];
            if a_fit == b_fit {
                continue;
            }
            let (mut inside, mut outside) = if a_fit { (a, b) } else { (b, a) };
            for _ in 0..20 {
                if distance(inside, outside) < CONTACT_TOLERANCE {
                    break;
                }
                let mid = interpolate(inside, outside, 0.5);
                queries += 1;
                if preliminary(mid) {
                    inside = mid;
                } else {
                    outside = mid;
                }
            }
            consider(inside);
        }
        step *= 0.5;
    }
    Some(Advance {
        pose: best,
        velocity: window.velocity_at(best),
        support_queries: queries,
        objective: best_score,
    })
}

#[cfg(test)]
fn footprint(cal: &Calibration, size: &ViewportSize, p: RenderPose) -> Vec<[f32; 2]> {
    let inverse = view(cal, p).try_inverse().unwrap();
    let vertical = (p.fov_degrees.to_radians() * 0.5).tan();
    let horizontal = vertical * size.aspect_ratio();
    (0..4)
        .flat_map(|edge| {
            (0..16).map(move |i| {
                let t = i as f32 / 16.0;
                match edge {
                    0 => [-1.0 + 2.0 * t, -1.0],
                    1 => [1.0, -1.0 + 2.0 * t],
                    2 => [1.0 - 2.0 * t, 1.0],
                    _ => [-1.0, 1.0 - 2.0 * t],
                }
            })
        })
        .map(|[x, y]| {
            let ray = (inverse * Vector4::new(x * horizontal, y * vertical, -1.0, 0.0))
                .xyz()
                .normalize();
            let q = aim(cal, &ray, p.fov_degrees);
            [q.yaw, q.pitch]
        })
        .collect()
}

#[cfg(test)]
mod unit_tests;
