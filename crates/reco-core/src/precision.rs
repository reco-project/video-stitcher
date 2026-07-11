//! Numeric precision policy: the tolerances that are *decisions*.
//!
//! One home for the epsilons whose value is a project-level choice
//! rather than a property of one algorithm. Deliberately granular: a
//! single blanket epsilon would either starve tight geometry (KB4
//! convergence wants 1e-10) or admit degenerate documents (validation
//! wants a human-scale bound), so each tolerance is named for the
//! question it answers and documented with what breaks at each extreme.
//!
//! Algorithm-internal constants (Newton-Raphson convergence, the
//! behind-camera homogeneous guard) stay next to their algorithm on
//! purpose: their values are derived from that code's numerics, not
//! from policy, and hoisting them here would invite "tuning" that
//! silently breaks the CPU/GPU agreement oracle.

/// Minimum positive value accepted by document validation for focal
/// lengths and the L-shape's axis offset.
///
/// Too loose and a near-zero focal length or camera offset reaches the
/// geometry, dividing by ~zero or normalizing a ~zero vector into NaN
/// poses. Too tight and legitimate hand-edited calibrations start
/// failing validation for values that render fine.
pub(crate) const VALIDATION_EPSILON: f64 = 1e-6;
