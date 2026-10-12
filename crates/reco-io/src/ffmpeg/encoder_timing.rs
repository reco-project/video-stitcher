//! Presentation-time validation and fixed-rate timestamp generation.

use super::encoder::EncodeError;
use ffmpeg_next::Rational;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use super::encoder::{StreamMonitor, StreamStatus};

pub(super) const TIME_BASE: Rational = Rational(1, 1_000_000);

/// One wall clock for shared encoded packets, never for ordinary offline output.
pub(super) struct BroadcastClock {
    origin: Option<(i64, Instant)>,
    primary: bool,
    stream: Option<StreamMonitor>,
    interrupted: Option<Arc<AtomicBool>>,
}

impl BroadcastClock {
    pub(super) fn new(
        primary: bool,
        stream: Option<StreamMonitor>,
        interrupted: Option<Arc<AtomicBool>>,
    ) -> Option<Self> {
        (primary || stream.is_some()).then_some(Self {
            origin: None,
            primary,
            stream,
            interrupted,
        })
    }

    fn remaining(&mut self, time_us: i64, now: Instant) -> Duration {
        let (first_time, first_wall) = *self.origin.get_or_insert((time_us, now));
        let media = Duration::from_micros(time_us.saturating_sub(first_time).max(0) as u64);
        media.saturating_sub(now.duration_since(first_wall))
    }

    pub(super) fn wait(&mut self, time_us: i64) -> bool {
        reco_core::profile_scope!("broadcast_wait");
        loop {
            if self
                .interrupted
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return false;
            }
            if !self.primary
                && !self.stream.as_ref().is_some_and(|stream| {
                    matches!(
                        stream.status(),
                        StreamStatus::Connecting | StreamStatus::Connected
                    )
                })
            {
                return false;
            }
            let remaining = self.remaining(time_us, Instant::now());
            if remaining.is_zero() {
                return true;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(10)));
        }
    }
}

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
    fn broadcast_clock_preserves_gaps_without_restarting_after_lateness() {
        let mut clock = BroadcastClock::new(true, None, None).unwrap();
        let start = Instant::now();
        assert_eq!(clock.remaining(-66_667, start), Duration::ZERO);
        assert_eq!(
            clock.remaining(33_333, start + Duration::from_millis(20)),
            Duration::from_millis(80)
        );
        assert_eq!(
            clock.remaining(66_666, start + Duration::from_secs(1)),
            Duration::ZERO
        );
        assert_eq!(
            clock.remaining(1_433_333, start + Duration::from_secs(1)),
            Duration::from_millis(500)
        );
        assert!(BroadcastClock::new(false, None, None).is_none());
    }

    #[test]
    fn cancellation_interrupts_a_large_presentation_gap() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut clock = BroadcastClock::new(true, None, Some(flag.clone())).unwrap();
        clock.wait(0);
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            flag.store(true, Ordering::Relaxed);
        });
        let start = Instant::now();
        clock.wait(3_600_000_000);
        assert!(start.elapsed() < Duration::from_secs(1));
        worker.join().unwrap();
    }

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
