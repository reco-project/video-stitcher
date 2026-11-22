//! Central KB4 inversion with bounded root isolation and safeguarded Newton iteration.
#[derive(Default)]
struct Roots {
    values: [f64; 4],
    len: usize,
}

impl Roots {
    fn push(&mut self, value: f64) -> Option<()> {
        if self.len == 0 || (self.values[self.len - 1] - value).abs() > 1e-10 {
            *self.values.get_mut(self.len)? = value;
            self.len += 1;
        }
        Some(())
    }
}

fn polynomial_roots(coefficients: &[f64], lower: f64, upper: f64) -> Option<Roots> {
    let mut degree = coefficients.len() - 1;
    while degree > 0 && coefficients[degree] == 0.0 {
        degree -= 1;
    }
    let mut roots = Roots::default();
    if degree == 0 {
        return Some(roots);
    }
    if degree == 1 {
        let root = -coefficients[0] / coefficients[1];
        if root >= lower && root <= upper {
            roots.push(root)?;
        }
        return Some(roots);
    }
    let value = |x: f64| {
        coefficients[..=degree]
            .iter()
            .rev()
            .fold(0.0, |sum, c| sum * x + c)
    };
    let near_zero = |x: f64, y: f64| {
        let magnitude = coefficients[..=degree]
            .iter()
            .rev()
            .fold(0.0, |sum, c| sum * x.abs() + c.abs());
        // Cover Horner rounding for degree at most four without merging nearby distinct roots.
        y.abs() <= 16.0 * f64::EPSILON * magnitude
    };
    let mut derivative = [0.0; 4];
    for i in 1..=degree {
        derivative[i - 1] = coefficients[i] * i as f64;
    }
    let stationary = polynomial_roots(&derivative[..degree], lower, upper)?;
    let mut lo = lower;
    for hi in stationary.values[..stationary.len]
        .iter()
        .copied()
        .chain(std::iter::once(upper))
    {
        let (left, right) = (value(lo), value(hi));
        let left_zero = near_zero(lo, left);
        let right_zero = near_zero(hi, right);
        if left_zero {
            roots.push(lo)?;
        }
        if !left_zero && !right_zero && left.is_sign_positive() != right.is_sign_positive() {
            let (mut a, mut b) = (lo, hi);
            for _ in 0..48 {
                let mid = (a + b) * 0.5;
                if value(mid).is_sign_positive() == left.is_sign_positive() {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            roots.push((a + b) * 0.5)?;
        }
        lo = hi;
    }
    if near_zero(upper, value(upper)) {
        roots.push(upper)?;
    }
    Some(roots)
}

/// Solve only the increasing branch connected to the optical axis, before its first stationary point.
pub(super) fn theta(radius: f64, distortion: &[f64; 4], correction: f64) -> Option<f64> {
    if radius == 0.0 && distortion.iter().all(|k| (k * correction).is_finite()) {
        return Some(0.0);
    }
    Inverse::new(distortion, correction)?.theta(radius)
}

/// A prepared central branch, reused across source correspondence queries.
pub(crate) struct Inverse {
    coefficients: [f64; 4],
    upper: f64,
    max_radius: f64,
}

impl Inverse {
    pub(crate) fn new(distortion: &[f64; 4], correction: f64) -> Option<Self> {
        let coefficients = distortion.map(|k| k * correction);
        if coefficients.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if coefficients.iter().all(|&k| k == 0.0) {
            let upper = std::f64::consts::FRAC_PI_2 - f64::EPSILON;
            return Some(Self {
                coefficients,
                upper,
                max_radius: upper,
            });
        }
        let end = std::f64::consts::FRAC_PI_2 - f64::EPSILON;
        let scale = coefficients.iter().map(|v| v.abs()).fold(1.0, f64::max);
        // The radial derivative is a quartic in theta squared. Its first zero bounds the central branch.
        let stationary = polynomial_roots(
            &[
                1.0 / scale,
                3.0 * (coefficients[0] / scale),
                5.0 * (coefficients[1] / scale),
                7.0 * (coefficients[2] / scale),
                9.0 * (coefficients[3] / scale),
            ],
            0.0,
            end * end,
        )?;
        let upper = if stationary.len == 0 {
            end
        } else {
            stationary.values[0].sqrt().min(end)
        };
        let max_radius = super::kb4::theta_d(upper, &coefficients);
        if !max_radius.is_finite() || max_radius <= 0.0 {
            return None;
        }
        Some(Self {
            coefficients,
            upper,
            max_radius,
        })
    }

    pub(crate) fn theta(&self, radius: f64) -> Option<f64> {
        if !radius.is_finite() || radius < 0.0 || radius > self.max_radius {
            return None;
        }
        if radius == 0.0 || self.coefficients.iter().all(|&k| k == 0.0) {
            return Some(radius);
        }
        let coefficients = self.coefficients;
        let mut upper = self.upper;
        let mut lower = 0.0;
        let mut theta = radius.min(upper * 0.5);
        for _ in 0..48 {
            let residual = super::kb4::theta_d(theta, &coefficients) - radius;
            if residual.abs() <= 1e-10 {
                return Some(theta);
            }
            if residual < 0.0 {
                lower = theta;
            } else {
                upper = theta;
            }
            let derivative = super::kb4::theta_d_prime(theta, &coefficients);
            let next = theta - residual / derivative;
            theta = if next > lower && next < upper {
                next
            } else {
                (lower + upper) * 0.5
            };
        }
        None
    }
}

pub(super) fn sensor_margin(lens: &crate::calibration::Lens) -> Option<f64> {
    let d = lens.distortion.map(|k| k * f64::from(lens.correction));
    if d.iter().any(|v| !v.is_finite())
        || [lens.fx, lens.fy, lens.cx, lens.cy]
            .iter()
            .any(|v| !v.is_finite())
        || lens.fx <= 0.0
        || lens.fy <= 0.0
        || lens.width == 0
        || lens.height == 0
        || !(0.0..=1.0).contains(&lens.correction)
    {
        return None;
    }
    let end = std::f64::consts::FRAC_PI_2 - 1e-6;
    let roots = polynomial_roots(
        &[0.99, 3.0 * d[0], 5.0 * d[1], 7.0 * d[2], 9.0 * d[3]],
        0.0,
        end * end,
    )?;
    let limit = if roots.len == 0 {
        end
    } else {
        roots.values[0].sqrt()
    };
    let capacity = super::kb4::theta_d(limit, &d);
    let corner = (lens.cx.max(f64::from(lens.width) - lens.cx) / lens.fx)
        .hypot(lens.cy.max(f64::from(lens.height) - lens.cy) / lens.fy);
    Some(capacity - corner - 1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_capacity_rejects_invalid_intrinsics() {
        let valid =
            crate::calibration::Lens::fisheye(640, 480, 400.0, 400.0, 320.0, 240.0, [0.0; 4]);
        assert!(super::super::sensor_domain_margin(&valid) > 0.0);
        for field in 0..4 {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut lens = valid.clone();
                *[&mut lens.fx, &mut lens.fy, &mut lens.cx, &mut lens.cy][field] = invalid;
                assert_eq!(super::super::sensor_domain_margin(&lens), f64::NEG_INFINITY);
            }
        }
        let mut lens = valid;
        lens.width = 0;
        assert_eq!(super::super::sensor_domain_margin(&lens), f64::NEG_INFINITY);
    }

    #[test]
    fn recovers_central_root_when_newton_chooses_a_negative_or_later_root() {
        for (coefficients, radius, expected) in [
            ([0.75, 0.0, -0.125, 0.0], 1.625, 1.0),
            ([0.5, 0.0, -0.08, 0.0], 1.42, 1.0),
        ] {
            assert!((theta(radius, &coefficients, 1.0).unwrap() - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_nonfinite_inputs_and_preserves_the_optical_axis() {
        for radius in [
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            std::f64::consts::FRAC_PI_2,
        ] {
            assert!(theta(radius, &[0.0; 4], 1.0).is_none());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(theta(0.5, &[value, 0.0, 0.0, 0.0], 1.0).is_none());
            assert!(theta(0.5, &[0.1; 4], value).is_none());
        }
        for coefficient in [-1e300, -1.0, 0.0, 1.0, 1e300] {
            assert_eq!(theta(0.0, &[coefficient; 4], 1.0), Some(0.0));
        }
    }

    fn with_roots(roots: &[f64]) -> Vec<f64> {
        let mut coefficients = vec![1.0];
        for root in roots {
            let mut product = vec![0.0; coefficients.len() + 1];
            for (i, c) in coefficients.iter().enumerate() {
                product[i] += c;
                product[i + 1] -= c / root;
            }
            coefficients = product;
        }
        coefficients
    }

    #[test]
    fn finds_close_repeated_and_endpoint_stationary_points() {
        for expected in [
            vec![0.49, 0.5],
            vec![0.5, 0.5],
            vec![0.1, 0.1001, 0.9, 1.9],
            vec![0.5, 0.50001, 1.5, 1.50001],
            vec![0.01, 0.1, 1.0, 2.46],
            vec![],
        ] {
            let coefficients = with_roots(&expected);
            let actual = polynomial_roots(&coefficients, 0.0, 2.46).unwrap();
            let mut unique = expected.clone();
            unique.dedup();
            assert_eq!(
                actual.len,
                unique.len(),
                "expected={expected:?}, actual={:?}",
                actual.values
            );
            for (&a, b) in actual.values.iter().zip(unique) {
                assert!((a - b).abs() < 2e-7, "actual={a}, expected={expected:?}");
            }
        }
        let actual = polynomial_roots(&[0.0, -1.0, 1.0], 0.0, 2.46).unwrap();
        assert_eq!(actual.len, 2);
        assert_eq!(actual.values[0], 0.0);
        assert!((actual.values[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn root_capacity_failure_is_reported_instead_of_panicking() {
        let mut roots = Roots::default();
        for i in 0..4 {
            roots.push(i as f64).unwrap();
        }
        assert_eq!(roots.push(4.0), None);
        assert_eq!(roots.len, 4);
    }

    #[test]
    fn central_branch_round_trips_varied_coefficients_and_corrections() {
        let mut random = 271828_u64;
        let mut next = || {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((random >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        for _ in 0..2000 {
            let d = std::array::from_fn(|_| next() * 4.0 - 2.0);
            let correction = next();
            let coefficients = d.map(|k| k * correction);
            let stationary = polynomial_roots(
                &[
                    1.0,
                    3.0 * coefficients[0],
                    5.0 * coefficients[1],
                    7.0 * coefficients[2],
                    9.0 * coefficients[3],
                ],
                0.0,
                std::f64::consts::FRAC_PI_2.powi(2),
            )
            .unwrap();
            let upper = if stationary.len == 0 {
                std::f64::consts::FRAC_PI_2
            } else {
                stationary.values[0].sqrt()
            };
            for i in 0..100 {
                let expected = upper * (i as f64 / 101.0);
                let radius = crate::lens::kb4::theta_d(expected, &coefficients);
                let actual = theta(radius, &d, correction).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "d={d:?}, expected={expected}, actual={actual}"
                );
            }
        }
    }
}
