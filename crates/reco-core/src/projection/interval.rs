//! Conservative scalar bounds for source-map rectangle checks.

use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Self {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            };
        }
        Self {
            lo: lo.next_down(),
            hi: hi.next_up(),
        }
    }

    pub fn point(x: f64) -> Self {
        Self::new(x, x)
    }

    pub fn square(self) -> Self {
        let lo = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            self.lo.abs().min(self.hi.abs()).powi(2)
        };
        Self::new(lo, self.lo.abs().max(self.hi.abs()).powi(2))
    }

    pub fn sqrt(self) -> Self {
        Self::new(self.lo.max(0.0).sqrt(), self.hi.sqrt())
    }

    pub fn inside(self, lo: f64, hi: f64) -> bool {
        self.lo >= lo && self.hi <= hi
    }

    pub fn abs_max(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    pub fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }
}

impl Add for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.lo + rhs.lo, self.hi + rhs.hi)
    }
}
impl Sub for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.lo - rhs.hi, self.hi - rhs.lo)
    }
}
impl Mul for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        if products.iter().any(|x| !x.is_finite()) {
            return Self::new(f64::NEG_INFINITY, f64::INFINITY);
        }
        Self::new(
            products.into_iter().fold(f64::INFINITY, f64::min),
            products.into_iter().fold(f64::NEG_INFINITY, f64::max),
        )
    }
}
impl Div for Interval {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs.lo <= 0.0 && rhs.hi >= 0.0 {
            return Self::new(f64::NEG_INFINITY, f64::INFINITY);
        }
        self * Self::new(1.0 / rhs.hi, 1.0 / rhs.lo)
    }
}

#[cfg(test)]
pub(crate) fn ndc_rect(
    min: [u32; 2],
    max: [u32; 2],
    width: f64,
    height: f64,
) -> (Interval, Interval) {
    (
        Interval::new(
            (min[0] as f64 + 0.5) / width * 2.0 - 1.0,
            (max[0] as f64 + 0.5) / width * 2.0 - 1.0,
        ),
        Interval::new(
            1.0 - (max[1] as f64 + 0.5) / height * 2.0,
            1.0 - (min[1] as f64 + 0.5) / height * 2.0,
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_encloses_interior_scalar_evaluations() {
        let mut seed = 17_u64;
        let mut random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 32) as u32) as f64 / u32::MAX as f64
        };
        for _ in 0..10000 {
            let a = random() * 20.0 - 10.0;
            let b = a + random() * 10.0;
            let c = random() * 20.0 - 10.0;
            let d = c + random() * 10.0;
            let (x, y) = (Interval::new(a, b), Interval::new(c, d));
            let contains = |i: Interval, value: f64| {
                assert!(i.lo <= value && value <= i.hi, "{i:?} excludes {value}")
            };
            for t in [0.0, random(), 1.0] {
                for s in [0.0, random(), 1.0] {
                    let (p, q) = (a + (b - a) * t, c + (d - c) * s);
                    contains(x + y, p + q);
                    contains(x - y, p - q);
                    contains(x * y, p * q);
                    if q != 0.0 {
                        contains(x / y, p / q);
                    }
                    contains(x.square(), p * p);
                    contains(x.square().sqrt(), p.abs());
                }
            }
        }
    }

    #[test]
    fn invalid_overflow_and_zero_divisors_widen_instead_of_accepting() {
        for i in [
            Interval::new(2.0, 1.0),
            Interval::point(f64::NAN),
            Interval::point(f64::MAX) * Interval::point(2.0),
            Interval::point(1.0) / Interval::new(-1.0, 1.0),
            Interval::point(-1.0).sqrt(),
        ] {
            assert!(!i.finite());
            assert!(!i.inside(0.0, 1.0));
        }
    }
}
