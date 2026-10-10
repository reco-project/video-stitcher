//! Live calibration from in-memory frame pair streams.
//!
//! Drives the calibration pipeline from any source of synchronized
//! YUV420P frame pairs (an OBS plugin's dual-source buffer, a Jetson
//! V4L2 capture, a WebRTC ingest, etc.). Consumers implement
//! `LiveFramePairSource` on whatever their upstream source is and
//! pass it to `calibrate_from_live`; the function pulls N pairs
//! and feeds them to the existing [`calibrate`](crate::calibrate)
//! primitive.
//!
//! # When to use this vs. `calibrate_videos`
//!
//! - `calibrate_videos` (file-based) when both cameras have been
//!   recorded to files and you want a one-shot calibration for
//!   post-production.
//! - `calibrate_from_live` (stream-based) when you want to
//!   calibrate **in the middle of a live session** — the OBS
//!   "Calibrate from current sources" workflow,
//!   Jetson on-device calibration with the cameras already
//!   streaming, mobile "re-calibrate" button.
//! - `calibrate` directly when you have your
//!   own capture policy and just want the core solver. Prefer
//!   `calibrate_from_live` if the "collect N pairs with a
//!   per-pair timeout" policy fits what you already want to
//!   implement.
//!
//! # Example adapter
//!
//! Consumers wrap their upstream source in a small adapter that
//! implements `LiveFramePairSource`:
//!
//! ```ignore
//! struct ObsCalibSource<'a> {
//!     buffer: &'a mut TimestampedIngestBuffer<ObsFrameHandle>,
//!     left_id: SourceId,
//!     right_id: SourceId,
//! }
//!
//! impl<'a> LiveFramePairSource for ObsCalibSource<'a> {
//!     fn next_pair(&mut self, timeout: Duration)
//!         -> Option<(Yuv420pFrame, Yuv420pFrame)>
//!     {
//!         let deadline = Instant::now() + timeout;
//!         loop {
//!             if let Some(tuple) = self.buffer.try_emit() {
//!                 let left  = yuv_from_handle(&tuple.frames[0].handle)?;
//!                 let right = yuv_from_handle(&tuple.frames[1].handle)?;
//!                 return Some((left, right));
//!             }
//!             if Instant::now() >= deadline { return None; }
//!             std::thread::sleep(Duration::from_millis(10));
//!         }
//!     }
//! }
//! ```
//!
//! Adapter lives in the consumer because the handle type `H` and
//! the conversion-to-Yuv420pFrame depend on the consumer's framework.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use reco_core::calibration::Lens;

use crate::error::CalibrateError;
use crate::types::{CalibrationProgress, CalibrationResult, CalibrationStep, Yuv420pFrame};

/// A source of synchronized YUV420P frame pairs for live calibration.
///
/// Consumers implement this trait on a small adapter that wraps their
/// upstream source (OBS dual-source buffer, V4L2 capture, WebRTC ingest,
/// `TimestampedIngestBuffer`). The trait is `Send` so calibration can
/// run on a worker thread.
pub trait LiveFramePairSource: Send {
    /// Retrieve the next synchronized `(left, right)` frame pair, or
    /// return `None` if no pair arrives within `timeout`.
    ///
    /// Implementations should block up to `timeout` and return as
    /// soon as a pair is available. Returning `None` terminates the
    /// capture loop with
    /// [`CalibrateFromLiveError::Timeout`].
    fn next_pair(&mut self, timeout: Duration) -> Option<(Yuv420pFrame, Yuv420pFrame)>;
}

/// Options for [`calibrate_from_live`].
#[derive(Debug, Clone)]
pub struct CalibrateFromLiveOptions {
    /// Number of frame pairs to collect before running the solver.
    /// Must be in 1..=16. The first pair fits; later pairs independently verify alignment.
    pub num_pairs: usize,
    /// Per-pair timeout. If a single pair takes longer than this to
    /// arrive, the whole call fails with
    /// [`CalibrateFromLiveError::Timeout`] (the live source is
    /// probably stalled).
    pub timeout_per_pair: Duration,
}

impl Default for CalibrateFromLiveOptions {
    fn default() -> Self {
        Self {
            num_pairs: 4,
            timeout_per_pair: Duration::from_secs(5),
        }
    }
}

/// Errors from [`calibrate_from_live`]. `Clone + Send + Sync` so a
/// calibration worker thread can ship the result through an mpsc
/// channel to the UI thread.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CalibrateFromLiveError {
    /// The requested count is outside 1..=16.
    #[error("expected 1 to 16 frame pairs, requested {requested}")]
    InvalidPairCount {
        /// Requested count.
        requested: usize,
    },
    /// A single frame pair took longer than the per-pair timeout to
    /// arrive. Usually indicates the live source stalled or one
    /// camera stopped producing frames.
    #[error(
        "live frame pair timeout ({timeout:?}) at pair {captured} of {requested}: \
         upstream source stalled?"
    )]
    Timeout {
        /// The timeout that was exceeded.
        timeout: Duration,
        /// How many pairs had been captured before the timeout.
        captured: usize,
        /// How many were requested.
        requested: usize,
    },
    /// The caller requested cancellation via the interrupted flag.
    #[error("calibration cancelled")]
    Cancelled,
    /// The underlying calibration solver returned an error.
    #[error("calibration: {0}")]
    Calibrate(#[from] CalibrateError),
}

/// Drive the calibration pipeline from a live frame-pair source.
///
/// Collects `options.num_pairs` synchronized frame pairs from
/// `source`, then runs the calibration solver via
/// [`crate::calibrate`]. Reports progress through `on_progress`
/// using the standard [`CalibrationStep`] vocabulary
/// (`ExtractingFrames` during capture, then `Optimizing`).
///
/// # Arguments
///
/// - `source`: live frame-pair source — the adapter the consumer
///   wrote around their upstream buffer.
/// - `left_params` / `right_params`: per-camera intrinsics (from a
///   lens-profile lookup or previous calibration). Missing optics are estimated.
/// - `options`: pair count and timeout.
/// - `on_progress`: per-step progress callback.
/// - `interrupted`: caller can set this to abort capture early; the
///   function then returns [`CalibrateFromLiveError::Cancelled`].
///
/// # Errors
///
/// See [`CalibrateFromLiveError`]. All variants are `Clone + Send + Sync`
/// so the result can ship across a worker-thread channel.
pub fn calibrate_from_live(
    source: &mut dyn LiveFramePairSource,
    left_params: Option<&Lens>,
    right_params: Option<&Lens>,
    options: &CalibrateFromLiveOptions,
    on_progress: &mut dyn FnMut(&CalibrationProgress),
    interrupted: &AtomicBool,
) -> Result<CalibrationResult, CalibrateFromLiveError> {
    if options.num_pairs == 0 || options.num_pairs > 16 {
        return Err(CalibrateFromLiveError::InvalidPairCount {
            requested: options.num_pairs,
        });
    }

    let start = Instant::now();
    let mut pairs: Vec<(Yuv420pFrame, Yuv420pFrame)> = Vec::with_capacity(options.num_pairs);

    emit_progress(
        on_progress,
        CalibrationStep::ExtractingFrames,
        format!(
            "capturing {} frame pairs (timeout {:?}/pair)",
            options.num_pairs, options.timeout_per_pair
        ),
    );

    while pairs.len() < options.num_pairs {
        if interrupted.load(Ordering::Relaxed) {
            return Err(CalibrateFromLiveError::Cancelled);
        }
        match source.next_pair(options.timeout_per_pair) {
            Some(pair) => {
                pairs.push(pair);
                emit_progress(
                    on_progress,
                    CalibrationStep::ExtractingFrames,
                    format!(
                        "captured {}/{} (elapsed {:.1}s)",
                        pairs.len(),
                        options.num_pairs,
                        start.elapsed().as_secs_f32()
                    ),
                );
            }
            None => {
                return Err(CalibrateFromLiveError::Timeout {
                    timeout: options.timeout_per_pair,
                    captured: pairs.len(),
                    requested: options.num_pairs,
                });
            }
        }
    }

    emit_progress(
        on_progress,
        CalibrationStep::Optimizing,
        "Fitting and verifying image alignment",
    );
    let result = super::calibrate(&pairs, [left_params, right_params], interrupted)?;
    Ok(result)
}

fn emit_progress(
    on_progress: &mut dyn FnMut(&CalibrationProgress),
    step: CalibrationStep,
    detail: impl Into<String>,
) {
    on_progress(&CalibrationProgress {
        step,
        detail: detail.into(),
    });
}

// Compile-time assertion: `CalibrateFromLiveError` is
// `Clone + Send + Sync` so a calibration worker can post its typed
// result through an mpsc channel without stringifying. Regresses if
// a future variant introduces a non-Clone `#[from]`.
const _: fn() = || {
    fn assert_clone_send_sync<T: Clone + Send + Sync + 'static>() {}
    assert_clone_send_sync::<CalibrateFromLiveError>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    /// A fake live source that hands out N pre-built YUV frame pairs,
    /// then returns `None` to simulate a stalled upstream.
    struct FakeSource {
        pairs: Vec<(Yuv420pFrame, Yuv420pFrame)>,
        next_idx: usize,
        // Count how many times `next_pair` was called (success or None)
        // so tests can verify the timeout path.
        calls: Arc<AtomicU32>,
    }

    impl FakeSource {
        fn new(count: usize, calls: Arc<AtomicU32>) -> Self {
            let pairs = (0..count).map(|_| (dummy_yuv(), dummy_yuv())).collect();
            Self {
                pairs,
                next_idx: 0,
                calls,
            }
        }
    }

    impl LiveFramePairSource for FakeSource {
        fn next_pair(&mut self, _timeout: Duration) -> Option<(Yuv420pFrame, Yuv420pFrame)> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.next_idx < self.pairs.len() {
                let p = self.pairs[self.next_idx].clone();
                self.next_idx += 1;
                Some(p)
            } else {
                None
            }
        }
    }

    /// A small valid frame for capture and rejection tests.
    fn dummy_yuv() -> Yuv420pFrame {
        let w = 64u32;
        let h = 64u32;
        Yuv420pFrame {
            y: vec![128u8; (w * h) as usize],
            u: vec![128u8; ((w * h) / 4) as usize],
            v: vec![128u8; ((w * h) / 4) as usize],
            width: w,
            height: h,
            timestamp_us: 0,
        }
    }

    #[test]
    fn rejects_invalid_pair_counts_before_capture() {
        let calls = Arc::new(AtomicU32::new(0));
        let mut src = FakeSource::new(5, calls.clone());
        for num_pairs in [0, 17, usize::MAX] {
            let opts = CalibrateFromLiveOptions {
                num_pairs,
                ..Default::default()
            };
            let err = calibrate_from_live(
                &mut src,
                None,
                None,
                &opts,
                &mut |_| {},
                &AtomicBool::new(false),
            )
            .unwrap_err();
            assert!(
                matches!(err, CalibrateFromLiveError::InvalidPairCount { requested } if requested == num_pairs)
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn fake_source_returns_none_after_exhaustion() {
        let calls = Arc::new(AtomicU32::new(0));
        let mut src = FakeSource::new(2, calls.clone());
        assert!(src.next_pair(Duration::from_millis(1)).is_some());
        assert!(src.next_pair(Duration::from_millis(1)).is_some());
        assert!(src.next_pair(Duration::from_millis(1)).is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn capture_honors_cancellation_and_reports_partial_timeout() {
        let calls = Arc::new(AtomicU32::new(0));
        let mut source = FakeSource::new(1, calls.clone());
        let options = CalibrateFromLiveOptions {
            num_pairs: 2,
            ..Default::default()
        };
        let interrupted = AtomicBool::new(true);
        assert!(matches!(
            calibrate_from_live(&mut source, None, None, &options, &mut |_| {}, &interrupted),
            Err(CalibrateFromLiveError::Cancelled)
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        interrupted.store(false, Ordering::Relaxed);
        assert!(matches!(
            calibrate_from_live(&mut source, None, None, &options, &mut |_| {}, &interrupted),
            Err(CalibrateFromLiveError::Timeout {
                captured: 1,
                requested: 2,
                ..
            })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn timeout_error_carries_captured_and_requested_counts() {
        let err = CalibrateFromLiveError::Timeout {
            timeout: Duration::from_secs(5),
            captured: 7,
            requested: 20,
        };
        let msg = err.to_string();
        assert!(msg.contains("5s"));
        assert!(msg.contains("7"));
        assert!(msg.contains("20"));
    }

    #[test]
    fn default_options_collect_fitting_and_confirmation_pairs() {
        let o = CalibrateFromLiveOptions::default();
        assert_eq!(o.num_pairs, 4);
        assert_eq!(o.timeout_per_pair, Duration::from_secs(5));
    }
}
