//! Shared CLI helpers: progress reporting, platform detection.

use std::cell::Cell;

/// Progress reporter that prints frame count and FPS every N frames.
///
/// Reports two rates when driven from the session's own elapsed
/// clock via [`Self::report_with_elapsed`]:
///
/// - **cumulative fps**: frames / elapsed (biased by startup when
///   the fallback clock is used; accurate when the session clock
///   is passed in)
/// - **recent fps**: frames / time since last checkpoint — the
///   steady-state rate, unaffected by startup costs
///
/// Useful because a benchmarking run's "final cumulative fps" still
/// carries early-frame warmup suppression: NVDEC context init, NVENC
/// encoder open, shader compile, triple-buffer warmup, ORT BFCArena
/// extensions, etc. can each cost tens to hundreds of milliseconds
/// spread across the first 10-30 frames. Cumulative averages those
/// in forever; recent tells you what the pipeline is actually doing
/// right now.
pub struct ProgressReporter {
    /// Fallback start time, used only by [`Self::report`] callers
    /// that don't supply the session's elapsed clock (camera /
    /// libcamera capture paths). Anchored at [`Self::new`] (before
    /// GPU init, so strictly pre-warmup); session-driven paths
    /// should prefer [`Self::report_with_elapsed`].
    #[allow(
        dead_code,
        reason = "only read by `report()`, which is gstreamer/libcamera-path-only under default features"
    )]
    start: std::time::Instant,
    interval: u64,
    /// Last `(frame_count, elapsed)` checkpoint — used to compute
    /// the recent-window fps as a delta against the previous
    /// report. Cell makes this interior-mutable without forcing
    /// `&mut self` on `report*` (ergonomic in Slint/session
    /// callback closures that capture by value).
    last_checkpoint: Cell<Option<(u64, std::time::Duration)>>,
}

impl ProgressReporter {
    /// Create a reporter that prints every `interval` frames.
    pub fn new(interval: u64) -> Self {
        Self {
            start: std::time::Instant::now(),
            interval,
            last_checkpoint: Cell::new(None),
        }
    }

    /// Report progress if frame_count is at the interval boundary.
    ///
    /// Uses the reporter's own wall clock (which starts at
    /// [`Self::new`], so it includes any pre-session setup the
    /// caller did). Prefer [`Self::report_with_elapsed`] when the
    /// caller has access to the session's own clock — it measures
    /// only the decode/render/encode loop and excludes GPU init.
    #[allow(
        dead_code,
        reason = "used only by gstreamer/libcamera capture paths under their features"
    )]
    pub fn report(&self, frame_count: u64) {
        self.report_with_elapsed(frame_count, self.start.elapsed());
    }

    /// Report progress using an externally-supplied elapsed time.
    ///
    /// Call this from a session progress callback that already
    /// carries the session's own clock (e.g.
    /// `FrameProgress::elapsed`). That clock starts after GPU +
    /// encoder init, so neither the cumulative nor the recent rate
    /// is diluted by startup costs.
    ///
    /// Emits a line like:
    /// ```text
    /// Processed 900 frames (555.6 / 588.2 fps)
    /// ```
    /// where the first rate is cumulative and the second is the
    /// recent-window rate since the last report.
    pub fn report_with_elapsed(&self, frame_count: u64, elapsed: std::time::Duration) {
        use std::io::Write;
        if frame_count == 0 || !frame_count.is_multiple_of(self.interval) {
            return;
        }
        let elapsed_secs = elapsed.as_secs_f64();
        let fps_cum = if elapsed_secs > 0.0 {
            frame_count as f64 / elapsed_secs
        } else {
            0.0
        };

        let fps_recent = match self.last_checkpoint.get() {
            Some((prev_frames, prev_elapsed))
                if frame_count > prev_frames && elapsed > prev_elapsed =>
            {
                let df = (frame_count - prev_frames) as f64;
                let dt = (elapsed - prev_elapsed).as_secs_f64();
                df / dt
            }
            // First checkpoint (or monotonic clock skew): recent ==
            // cumulative. Downstream viewer sees both columns agree
            // until the second report arrives.
            _ => fps_cum,
        };
        self.last_checkpoint.set(Some((frame_count, elapsed)));

        print!("\rProcessed {frame_count} frames ({fps_cum:.1} / {fps_recent:.1} fps)");
        let _ = std::io::stdout().flush();
    }

    /// Print the final summary line.
    #[cfg(any(feature = "gstreamer", feature = "libcamera"))]
    pub fn finish(&self, frame_count: u64, output_path: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let fps = frame_count as f64 / elapsed;
        println!(
            "\n\nDone: {frame_count} frames in {elapsed:.1}s ({fps:.1} fps) \u{2192} {output_path}"
        );
    }
}

/// Detect NVIDIA Jetson (Tegra) platform.
#[cfg(feature = "gstreamer")]
pub fn is_tegra() -> bool {
    std::path::Path::new("/etc/nv_tegra_release").exists()
        || std::fs::read_to_string("/proc/device-tree/compatible")
            .is_ok_and(|s| s.contains("nvidia,tegra"))
}

/// Finalize session outputs after capture, preserving the first error.
/// Keep capture sources inside the callback so they stop before finalization.
#[cfg(any(feature = "gstreamer", feature = "libcamera"))]
pub fn with_session_finalization<T>(
    session: &mut reco_core::session::StitchSession,
    capture: impl FnOnce(&mut reco_core::session::StitchSession) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let result = capture(session);
    #[cfg(feature = "replay")]
    session.clear_stacked_gpu_recorder();
    let finished = session.finish();
    if result.is_err()
        && let Err(error) = &finished
    {
        log::warn!("Output finalization also failed after capture error: {error}");
    }
    let value = result?;
    finished?;
    Ok(value)
}

#[cfg(all(test, any(feature = "gstreamer", feature = "libcamera")))]
mod tests {
    use super::*;
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::geometry::RenderPose;
    use reco_core::projection::Cylinder;
    use reco_core::render::viewport::ViewportSize;
    use reco_core::session::{SinkDelivery, SinkErrorPolicy, SinkOptions, StitchSession};
    use reco_core::sink::{OutputFrame, OutputSink, PixelFormat, SinkError, SinkInput};
    use reco_core::source::{FrameSet, Yuv420pData};
    use reco_core::stitch::{CpuExecutor, Executor};
    use std::sync::{Arc, Mutex};

    struct CaptureOwner(Arc<Mutex<Vec<&'static str>>>);
    impl Drop for CaptureOwner {
        fn drop(&mut self) {
            self.0.lock().unwrap().push("capture stopped");
        }
    }

    struct RecordingSink {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail_finish: bool,
    }
    impl OutputSink for RecordingSink {
        fn name(&self) -> &str {
            "recording"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(PixelFormat::Nv12)
        }
        fn consume(&mut self, _: OutputFrame<'_>) -> Result<(), SinkError> {
            self.events.lock().unwrap().push("frame");
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            self.events.lock().unwrap().push("finished");
            if self.fail_finish {
                Err(SinkError::Finish {
                    reason: "finalization failure".into(),
                })
            } else {
                Ok(())
            }
        }
    }

    fn capture_finalization(gpu: bool) {
        for fail_capture in [true, false] {
            for fail_finish in [false, true] {
                let calibration = Calibration::new(
                    vec![Lens::flat(64, 32)],
                    Cylinder {
                        radius: 32.0,
                        ..Default::default()
                    },
                    Framing {
                        axis_offset: 0.0,
                        tilt: 0.0,
                        roll: 0.0,
                    },
                );
                let size = ViewportSize {
                    width: 64,
                    height: 32,
                };
                let mut session = if gpu {
                    StitchSession::builder()
                        .calibration(calibration)
                        .source_sizes(vec![(64, 32)])
                        .viewport_size(size)
                        .build()
                        .unwrap()
                } else {
                    StitchSession::with_executor(Executor::Cpu(Box::new(
                        CpuExecutor::new(calibration, size, vec![(64, 32)], false).unwrap(),
                    )))
                    .unwrap()
                };
                session.core_mut().set_constrained_look(false);
                let events = Arc::new(Mutex::new(Vec::new()));
                session
                    .add_sink(
                        Box::new(RecordingSink {
                            events: events.clone(),
                            fail_finish,
                        }),
                        SinkOptions {
                            delivery: SinkDelivery::Inline,
                            on_error: SinkErrorPolicy::Abort,
                        },
                    )
                    .unwrap();
                let result = with_session_finalization(&mut session, |session| {
                    let _capture = CaptureOwner(events.clone());
                    let images = FrameSet::Yuv420p(vec![Arc::new(Yuv420pData {
                        size: (64, 32),
                        y: vec![80; 64 * 32],
                        u: vec![128; 32 * 16],
                        v: vec![128; 32 * 16],
                    })]);
                    for index in 0..2 {
                        session.render_frame(
                            &images,
                            RenderPose::default(),
                            Some(index * 33_333),
                        )?;
                    }
                    if fail_capture {
                        anyhow::bail!("capture failure");
                    }
                    Ok(7)
                });
                let events = events.lock().unwrap();
                assert_eq!(events.iter().filter(|&&event| event == "frame").count(), 2);
                assert_eq!(
                    events.iter().filter(|&&event| event == "finished").count(),
                    1
                );
                assert!(
                    events
                        .iter()
                        .position(|&event| event == "capture stopped")
                        .unwrap()
                        < events
                            .iter()
                            .position(|&event| event == "finished")
                            .unwrap()
                );
                match (fail_capture, fail_finish) {
                    (true, _) => {
                        assert!(result.unwrap_err().to_string().contains("capture failure"))
                    }
                    (false, true) => assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("finalization failure")
                    ),
                    (false, false) => assert_eq!(result.unwrap(), 7),
                }
            }
        }
    }

    #[test]
    fn cpu_capture_finalizes_after_success_or_failure() {
        capture_finalization(false);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn gpu_capture_finalizes_after_success_or_failure() {
        capture_finalization(true);
    }
}
