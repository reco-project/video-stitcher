//! Bounded adaptive certification of a continuous viewport motion.

use super::*;

// Count geometry queries, rather than limiting every branch to the same depth.
const QUERY_BUDGET: usize = 2048;

pub(super) fn swept(
    cal: &Calibration,
    size: &ViewportSize,
    from: RenderPose,
    to: RenderPose,
) -> bool {
    certify_motion(cal, size, from, to, QUERY_BUDGET) == Certificate::Covered
}

pub(super) fn certify_motion(
    cal: &Calibration,
    size: &ViewportSize,
    from: RenderPose,
    to: RenderPose,
    queries: usize,
) -> Certificate {
    certify_path(cal, size, &[from, to], queries, 2048)
}

// One budget covers both frame intervals and refinement within an interval.
pub(super) fn certify_path(
    cal: &Calibration,
    size: &ViewportSize,
    path: &[RenderPose],
    mut queries: usize,
    rectangles: usize,
) -> Certificate {
    if path.is_empty() || path.iter().any(|p| !valid_query(cal, size, *p)) {
        return Certificate::Unknown;
    }
    enum Work<'a> {
        Frames(&'a [RenderPose]),
        Motion(RenderPose, RenderPose),
    }
    let mut pending = vec![Work::Frames(path)];
    while let Some(work) = pending.pop() {
        if queries == 0 {
            return Certificate::Unknown;
        }
        let (from, to, witness) = match &work {
            Work::Motion(a, b) => (*a, *b, interpolate(*a, *b, 0.5)),
            Work::Frames(poses) => {
                let mut lo = poses[0];
                let mut hi = lo;
                for p in *poses {
                    lo.yaw = lo.yaw.min(p.yaw);
                    lo.pitch = lo.pitch.min(p.pitch);
                    hi.yaw = hi.yaw.max(p.yaw);
                    hi.pitch = hi.pitch.max(p.pitch);
                    lo.fov_degrees = lo.fov_degrees.min(p.fov_degrees);
                    hi.fov_degrees = hi.fov_degrees.max(p.fov_degrees);
                }
                (lo, hi, poses[poses.len() / 2])
            }
        };
        queries -= 1;
        if boundary::enclosure(cal, size, from, to, rectangles) == Certificate::Covered {
            continue;
        }
        if queries == 0 {
            return Certificate::Unknown;
        }
        queries -= 1;
        // An uncovered enclosure need not mean an uncovered path. Only an
        // actual view on the path supplies a counterexample.
        if boundary::pose(cal, size, witness) == Certificate::Uncovered {
            return Certificate::Uncovered;
        }
        match work {
            Work::Frames(poses) if poses.len() > 2 => {
                let mid = poses.len() / 2;
                pending.push(Work::Frames(&poses[mid..]));
                pending.push(Work::Frames(&poses[..=mid]));
            }
            Work::Frames(poses) => {
                pending.push(Work::Motion(poses[0], poses[poses.len() - 1]));
            }
            Work::Motion(a, b) => {
                if witness == a || witness == b {
                    return Certificate::Unknown;
                }
                pending.push(Work::Motion(witness, b));
                pending.push(Work::Motion(a, witness));
            }
        }
    }
    Certificate::Covered
}

#[test]
fn budget_exhaustion_is_not_coverage() {
    let cal = stereo();
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let pose = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 35.0,
    };
    assert_eq!(
        certify_motion(&cal, &size, pose, pose, 0),
        Certificate::Unknown
    );
    assert_eq!(
        certify_motion(&cal, &size, pose, pose, 1),
        Certificate::Covered
    );
    let outside = RenderPose { yaw: 3.0, ..pose };
    assert_eq!(
        certify_motion(&cal, &size, outside, outside, QUERY_BUDGET),
        Certificate::Uncovered
    );
    assert!(!swept(&cal, &size, pose, outside));
    let invalid = RenderPose {
        fov_degrees: f32::NAN,
        ..pose
    };
    assert_eq!(
        certify_motion(&cal, &size, pose, invalid, QUERY_BUDGET),
        Certificate::Unknown
    );
}

#[test]
fn covered_endpoints_do_not_imply_covered_motion() {
    let cal = stereo();
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let from = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 35.0,
    };
    // Interpolation follows the full turn, not the shortest angular path.
    let to = RenderPose {
        yaw: std::f32::consts::TAU,
        ..from
    };
    assert_eq!(certify_pose(&cal, &size, from), Certificate::Covered);
    assert_eq!(certify_pose(&cal, &size, to), Certificate::Covered);
    assert_eq!(
        certify_motion(&cal, &size, from, to, QUERY_BUDGET),
        Certificate::Uncovered
    );
}
