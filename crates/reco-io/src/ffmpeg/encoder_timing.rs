//! Presentation-time validation and fixed-rate timestamp generation.

use super::encoder::EncodeError;
use ffmpeg_next::Rational;

pub(super) const TIME_BASE: Rational = Rational(1, 1_000_000);

pub(super) struct Timeline {
    frame_duration: Rational,
    pub frames: i64,
    last: Option<(i64, bool)>,
}

impl Timeline {
    pub fn new(fps: Rational) -> Self {
        Self {
            frame_duration: Rational(fps.1, fps.0),
            frames: 0,
            last: None,
        }
    }

    pub fn next(&self, supplied: Option<i64>) -> Result<i64, EncodeError> {
        let timed = supplied.is_some();
        if self.last.is_some_and(|(_, previous)| previous != timed) {
            return Err(EncodeError::InvalidConfig(
                "cannot mix explicit timestamps and fixed-rate frames in one output".into(),
            ));
        }
        let pts = supplied.unwrap_or_else(|| self.fixed_timestamp(self.frames));
        if pts < 0 || self.last.is_some_and(|(last, _)| pts <= last) {
            return Err(EncodeError::InvalidConfig(
                "presentation timestamps must be nonnegative and strictly increasing".into(),
            ));
        }
        pts.checked_add(self.duration_us())
            .ok_or_else(|| EncodeError::InvalidConfig("presentation time overflow".into()))?;
        Ok(pts)
    }

    pub fn accept(&mut self, pts: i64, timed: bool) {
        self.last = Some((pts, timed));
        self.frames += 1;
    }

    pub fn duration_us(&self) -> i64 {
        self.fixed_timestamp(1).max(1)
    }

    pub fn end_us(&self) -> i64 {
        match self.last {
            Some((last, true)) => last + self.duration_us(),
            _ => self.fixed_timestamp(self.frames),
        }
    }

    fn fixed_timestamp(&self, frame: i64) -> i64 {
        // SAFETY: pure rational rescaling; the encoder validates positive FPS.
        unsafe {
            ffmpeg_next::sys::av_rescale_q(frame, self.frame_duration.into(), TIME_BASE.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_times_preserve_gaps_and_origin() {
        let mut timeline = Timeline::new(Rational(30, 1));
        for pts in [1_000_000, 1_033_333, 1_133_333] {
            assert_eq!(timeline.next(Some(pts)).unwrap(), pts);
            timeline.accept(pts, true);
        }
        assert_eq!(timeline.end_us(), 1_166_666);
    }

    #[test]
    fn rational_cadence_does_not_accumulate_rounding() {
        let mut timeline = Timeline::new(Rational(30000, 1001));
        for frame in 0..30000 {
            let pts = timeline.next(None).unwrap();
            let expected = ((frame * 1_001_000i64) + 15) / 30;
            assert_eq!(pts, expected);
            timeline.accept(pts, false);
        }
        assert_eq!(timeline.end_us(), 1_001_000_000);
    }

    #[test]
    fn rejects_invalid_times_without_advancing() {
        let mut timeline = Timeline::new(Rational(30, 1));
        assert!(timeline.next(Some(-1)).is_err());
        let pts = timeline.next(Some(100)).unwrap();
        timeline.accept(pts, true);
        for pts in [0, 99, 100, i64::MAX] {
            assert!(timeline.next(Some(pts)).is_err());
        }
        assert!(timeline.next(None).is_err());
        assert_eq!(timeline.frames, 1);
        assert_eq!(timeline.next(Some(101)).unwrap(), 101);

        let mut fixed = Timeline::new(Rational(30, 1));
        fixed.accept(fixed.next(None).unwrap(), false);
        assert!(fixed.next(Some(33_333)).is_err());
    }
}
