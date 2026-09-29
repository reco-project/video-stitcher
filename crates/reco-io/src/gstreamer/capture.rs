//! One capture clock and lifecycle, parallel extraction and cancellable frame waits.

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use reco_core::source::SourceError;
use std::sync::atomic::{AtomicBool, Ordering};

mod sync;

#[cfg(test)]
mod pool_tests;

struct Capture {
    pipeline: gst::Pipeline,
    sinks: [AppSink; 2],
    bus: gst::Bus,
}

impl Capture {
    fn start(descriptions: [&str; 2]) -> Result<Self, SourceError> {
        let init_error = |reason: String| SourceError::Init {
            path: "capture".into(),
            reason,
        };
        gst::init().map_err(|e| init_error(e.to_string()))?;
        let pipeline = gst::Pipeline::new();
        let branch = |description: &str, label: &str| -> Result<AppSink, SourceError> {
            let error = |reason: String| SourceError::Init {
                path: label.into(),
                reason,
            };
            let bin = gst::parse::bin_from_description(description, false)
                .map_err(|e| error(e.to_string()))?;
            bin.set_property("name", label);
            let sink = bin
                .by_name("sink")
                .and_then(|element| element.downcast::<AppSink>().ok())
                .ok_or_else(|| error("capture appsink not found".into()))?;
            sink.set_max_buffers(2);
            sink.set_drop(true);
            // Capture has no thumbnail consumer; do not retain an extra last-sample reference.
            sink.set_property("enable-last-sample", false);
            // A silent camera must not block the other branch's startup or EOS.
            sink.set_property("async", false);
            // EOS must not wait for a consumer that has stopped reading.
            sink.set_wait_on_eos(false);
            pipeline.add(&bin).map_err(|e| error(e.to_string()))?;
            Ok(sink)
        };
        let sinks = [
            branch(descriptions[0], "left")?,
            branch(descriptions[1], "right")?,
        ];
        let bus = pipeline
            .bus()
            .ok_or_else(|| init_error("capture bus not found".into()))?;
        let capture = Self {
            pipeline,
            sinks,
            bus,
        };
        capture
            .pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| {
                capture
                    .check_error()
                    .err()
                    .unwrap_or_else(|| init_error(format!("start capture: {e}")))
            })?;
        Ok(capture)
    }

    fn check_error(&self) -> Result<(), SourceError> {
        if let Some(message) = self
            .bus
            .pop_filtered(&[gst::MessageType::Error, gst::MessageType::ClockLost])
        {
            match message.view() {
                gst::MessageView::Error(error) => {
                    return Err(SourceError::Read {
                        reason: format!(
                            "{}: {} ({:?})",
                            error.src().map(|src| src.path_string()).unwrap_or_default(),
                            error.error(),
                            error.debug()
                        ),
                    });
                }
                gst::MessageView::ClockLost(_) => {
                    return Err(SourceError::Read {
                        reason: "capture clock lost; reopen the source".into(),
                    });
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    fn stop(&self) -> Result<(), SourceError> {
        self.pipeline
            .set_state(gst::State::Null)
            .map(|_| ())
            .map_err(|e| SourceError::Read {
                reason: format!("stop capture: {e}"),
            })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            log::error!("{error}");
        }
    }
}

#[derive(Debug)]
pub(super) struct Captured<T> {
    pub images: T,
    pub pts_us: i64,
    arrived_us: Option<i64>,
}

#[derive(Debug)]
pub(super) struct CapturedPair<T> {
    pub images: [T; 2],
    pub pts_us: i64,
    pub capture: reco_core::source::CaptureTiming,
}

fn sample_time_us(sample: &gst::Sample) -> Result<i64, SourceError> {
    let running = sample.buffer().and_then(|buffer| {
        sample
            .segment()?
            .downcast_ref::<gst::ClockTime>()?
            .to_running_time(buffer.pts())
    });
    running
        .and_then(|time| i64::try_from(time.useconds()).ok())
        .ok_or_else(|| SourceError::Read {
            reason: "capture sample has no representable running-time timestamp".into(),
        })
}

pub(super) struct CapturePair<T> {
    capture: Capture,
    receivers: Vec<std::sync::mpsc::Receiver<Option<Captured<T>>>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    errors: std::sync::mpsc::Receiver<SourceError>,
    stopped: bool,
    origin_us: Option<i64>,
    sync: Option<sync::SyncState<T>>,
}

impl<T: Clone + Send + 'static> CapturePair<T> {
    pub(super) fn start(
        descriptions: [&str; 2],
        extract: impl Fn(&gst::Sample) -> Result<T, SourceError> + Clone + Send + 'static,
        policy: Option<(crate::frame_sync::Config, u32)>,
    ) -> Result<Self, SourceError> {
        // Each worker reports at most one terminal error.
        let (error_tx, errors) = std::sync::mpsc::sync_channel(2);
        // Partial initialization owns and cleans up the pipeline and workers already started.
        let sync = policy
            .map(|(config, fps)| sync::SyncState::new(config, fps))
            .transpose()?;
        let mut pair = Self {
            capture: Capture::start(descriptions)?,
            receivers: Vec::with_capacity(2),
            workers: Vec::with_capacity(2),
            errors,
            stopped: false,
            origin_us: None,
            sync,
        };
        for (sink, label) in pair.capture.sinks.iter().zip(["left", "right"]) {
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            let errors = error_tx.clone();
            let sink = sink.clone();
            let extract = extract.clone();
            let wake = pair.sync.as_ref().map(|state| state.wake.clone());
            let pipeline = pair.capture.pipeline.clone();
            let worker = std::thread::Builder::new()
                .name(format!("capture_{label}"))
                .spawn(move || {
                    let mut previous_time = None;
                    while let Ok(sample) = sink.pull_sample() {
                        let frame = match sample_time_us(&sample).and_then(|pts_us| {
                            if previous_time.is_some_and(|previous| pts_us <= previous) {
                                return Err(SourceError::Read {
                                    reason: "capture timeline reset or did not advance; reopen the source".into(),
                                });
                            }
                            previous_time = Some(pts_us);
                            let images = extract(&sample)?;
                            Ok(Captured {
                                images,
                                pts_us,
                                arrived_us: if wake.is_some() { Some(sync::clock_us(&pipeline)?) } else { None },
                            })
                        }) {
                            Ok(frame) => frame,
                            Err(error) => {
                                // Errors must remain visible even if the peer has stalled.
                                let _ = errors.send(SourceError::Read {
                                    reason: format!("{label}: {error}"),
                                });
                                if let Some(wake) = &wake { sync::notify(wake); }
                                return;
                            }
                        };
                        drop(sample);
                        if tx.send(Some(frame)).is_err() {
                            return;
                        }
                        if let Some(wake) = &wake { sync::notify(wake); }
                    }
                    let _ = tx.send(None);
                    if let Some(wake) = &wake { sync::notify(wake); }
                })
                .map_err(|e| SourceError::Init {
                    path: label.into(),
                    reason: format!("start capture worker: {e}"),
                })?;
            pair.receivers.push(rx);
            pair.workers.push(worker);
        }
        Ok(pair)
    }

    pub(super) fn next(
        &mut self,
        interrupted: &AtomicBool,
    ) -> Result<Option<CapturedPair<T>>, SourceError> {
        if self.stopped {
            return Ok(None);
        }
        let result = if self.sync.is_some() {
            self.next_synced(interrupted)
        } else {
            self.next_frames(interrupted).map(|pair| {
                pair.map(|[left, right]| {
                    // One reference camera supplies output cadence. A common clock
                    // does not make arrival-order pairing timestamp alignment.
                    let origin = *self.origin_us.get_or_insert(left.pts_us);
                    CapturedPair {
                        images: [left.images, right.images],
                        pts_us: left.pts_us - origin,
                        capture: reco_core::source::CaptureTiming {
                            times_us: vec![left.pts_us, right.pts_us],
                            at_us: left.pts_us.max(right.pts_us),
                            max_age_us: None,
                        },
                    }
                })
            })
        };
        if !matches!(result, Ok(Some(_))) {
            let stopped = self.stop();
            if result.is_ok() {
                stopped?;
            } else if let Err(error) = stopped {
                log::error!("{error}");
            }
        }
        result
    }

    fn next_frames(
        &self,
        interrupted: &AtomicBool,
    ) -> Result<Option<[Captured<T>; 2]>, SourceError> {
        let mut pending = [None, None];
        loop {
            self.check_errors()?;
            if interrupted.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let mut ended = false;
            for (receiver, frame) in self.receivers.iter().zip(&mut pending) {
                if frame.is_none() {
                    // Bound idle waits so a stalled input cannot hide peer errors or cancellation.
                    match receiver.recv_timeout(std::time::Duration::from_millis(20)) {
                        Ok(next) => {
                            ended |= next.is_none();
                            *frame = next;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            self.check_errors()?;
                            return Err(SourceError::Read {
                                reason: "capture worker disconnected without a terminal result"
                                    .into(),
                            });
                        }
                    }
                }
            }
            self.check_errors()?;
            if ended || interrupted.load(Ordering::Relaxed) {
                return Ok(None);
            }
            if pending.iter().all(Option::is_some) {
                return Ok(Some(pending.map(Option::unwrap)));
            }
        }
    }
}

impl<T> CapturePair<T> {
    #[cfg(test)]
    pub(super) fn test_app_sources(&self) -> [gstreamer_app::AppSrc; 2] {
        ["left", "right"].map(|label| {
            self.capture
                .pipeline
                .by_name(label)
                .unwrap()
                .downcast::<gst::Bin>()
                .unwrap()
                .by_name("source")
                .unwrap()
                .downcast::<gstreamer_app::AppSrc>()
                .unwrap()
        })
    }

    fn check_errors(&self) -> Result<(), SourceError> {
        self.capture.check_error()?;
        if let Ok(error) = self.errors.try_recv() {
            return Err(error);
        }
        Ok(())
    }

    pub(super) fn is_stopped(&self) -> bool {
        self.stopped
    }

    pub(super) fn stop(&mut self) -> Result<(), SourceError> {
        self.stopped = true;
        self.sync = None;
        // Disconnect full sends, stop native pulls, then join. No worker is detached.
        self.receivers.clear();
        let mut result = self.capture.stop();
        for worker in self.workers.drain(..) {
            if worker.join().is_err() && result.is_ok() {
                result = Err(SourceError::Read {
                    reason: "capture worker panicked".into(),
                });
            }
        }
        result
    }
}

impl<T> Drop for CapturePair<T> {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            log::error!("{error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    const IDLE: &str = "appsrc is-live=true ! appsink name=sink sync=false";
    const FINITE: &str = "videotestsrc num-buffers=2 ! video/x-raw,format=I420,width=64,height=32 ! appsink name=sink sync=false";

    fn sample_pair(descriptions: [&str; 2]) -> Result<CapturePair<gst::Sample>, SourceError> {
        CapturePair::start(descriptions, |sample| Ok(sample.clone()), None)
    }

    fn observer<T>(pair: &CapturePair<T>) -> gst::Pipeline {
        pair.capture.pipeline.clone()
    }

    fn assert_stopped(pipeline: &gst::Pipeline) {
        assert_eq!(pipeline.current_state(), gst::State::Null);
        for branch in pipeline.children() {
            assert_eq!(branch.current_state(), gst::State::Null);
        }
    }

    #[test]
    fn sample_time_uses_segment_running_time_and_rejects_missing_timing() {
        gst::init().unwrap();
        let mut segment = gst::FormattedSegment::<gst::ClockTime>::new();
        segment.set_start(gst::ClockTime::from_seconds(5));
        segment.set_base(gst::ClockTime::from_seconds(2));
        let mut buffer = gst::Buffer::new();
        buffer
            .get_mut()
            .unwrap()
            .set_pts(gst::ClockTime::from_seconds(6));
        let sample = gst::Sample::builder()
            .buffer(&buffer)
            .segment(&segment)
            .build();
        assert_eq!(sample_time_us(&sample).unwrap(), 3_000_000);
        drop(sample);
        buffer.get_mut().unwrap().set_pts(None);
        assert!(
            sample_time_us(
                &gst::Sample::builder()
                    .buffer(&buffer)
                    .segment(&segment)
                    .build()
            )
            .is_err()
        );
        assert!(sample_time_us(&gst::Sample::builder().build()).is_err());
        buffer.get_mut().unwrap().set_pts(gst::ClockTime::ZERO);
        let segment = gst::FormattedSegment::<gst::ClockTime>::new();
        assert_eq!(
            sample_time_us(
                &gst::Sample::builder()
                    .buffer(&buffer)
                    .segment(&segment)
                    .build()
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn shared_clock_preserves_skew_and_a_silent_peer_does_not_block_capture() {
        for first in 0..2 {
            let description =
                "appsrc name=source is-live=true format=time ! appsink name=sink sync=false";
            let (seen_tx, seen_rx) = mpsc::channel();
            let mut pair = CapturePair::start(
                [description; 2],
                move |sample| {
                    let time = sample_time_us(sample)?;
                    seen_tx.send(time).unwrap();
                    Ok(time)
                },
                None,
            )
            .unwrap();
            let sources = pair.test_app_sources();
            let pipeline = observer(&pair);
            assert_eq!(pipeline.current_state(), gst::State::Playing);
            assert!(pipeline.clock().is_some());
            for source in &sources {
                assert_eq!(source.clock(), pipeline.clock());
                assert_eq!(source.base_time(), pipeline.base_time());
            }
            for (times, output) in [([500_000, 600_000], 0), ([700_000, 700_000], 200_000)] {
                for camera in [first, 1 - first] {
                    let time = times[camera];
                    let mut buffer = gst::Buffer::from_mut_slice(vec![camera as u8]);
                    buffer
                        .get_mut()
                        .unwrap()
                        .set_pts(gst::ClockTime::from_useconds(time));
                    sources[camera].push_buffer(buffer).unwrap();
                    // Wait for actual extraction before delivering the peer. No sleep
                    // or assumed thread timing, and acquisition timestamps stay intact.
                    assert_eq!(
                        seen_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
                        time as i64
                    );
                }
                let captured = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
                assert_eq!(captured.images, times.map(|time| time as i64));
                assert_eq!(captured.capture.times_us, times.map(|time| time as i64));
                assert_eq!(captured.pts_us, output);
            }
            pair.stop().unwrap();
            assert_stopped(&pipeline);
        }
    }

    #[test]
    fn clock_loss_stops_both_branches_instead_of_rebasing_time() {
        let mut pair = sample_pair([IDLE; 2]).unwrap();
        let pipeline = observer(&pair);
        pair.capture
            .bus
            .post(
                gst::message::ClockLost::builder(&pipeline.clock().unwrap())
                    .src(&pipeline)
                    .build(),
            )
            .unwrap();
        // A clock failure must remain visible even if cancellation is also set.
        let error = pair.next(&AtomicBool::new(true)).unwrap_err();
        assert!(error.to_string().contains("clock lost"));
        assert!(pair.is_stopped());
        assert_stopped(&pipeline);
    }

    #[test]
    fn each_camera_rejects_duplicate_or_backward_time_with_a_silent_peer() {
        let description =
            "appsrc name=source is-live=true format=time ! appsink name=sink sync=false";
        for failed in [0, 1] {
            for invalid_time in [5_000_000, 0] {
                let mut pair = sample_pair([description; 2]).unwrap();
                let pipeline = observer(&pair);
                let sources = pair.test_app_sources();
                let push = |camera: usize, time| {
                    let mut buffer = gst::Buffer::from_mut_slice(vec![camera as u8]);
                    buffer
                        .get_mut()
                        .unwrap()
                        .set_pts(gst::ClockTime::from_useconds(time));
                    sources[camera].push_buffer(buffer).unwrap();
                };
                for camera in [0, 1] {
                    push(camera, 5_000_000);
                }
                assert!(pair.next(&AtomicBool::new(false)).unwrap().is_some());
                push(failed, invalid_time);
                let cancel = Arc::new(AtomicBool::new(false));
                let flag = cancel.clone();
                let (done_tx, done_rx) = mpsc::channel();
                let worker = std::thread::spawn(move || {
                    let result = pair.next(&flag);
                    drop(pair);
                    done_tx.send(result).unwrap();
                });
                let result = match done_rx.recv_timeout(Duration::from_secs(2)) {
                    Ok(result) => result,
                    Err(_) => {
                        cancel.store(true, Ordering::Relaxed);
                        done_rx.recv_timeout(Duration::from_secs(5)).unwrap()
                    }
                };
                worker.join().unwrap();
                assert_stopped(&pipeline);
                let error =
                    result.expect_err("bad camera time must fail without waiting for its peer");
                let message = error.to_string();
                assert!(message.contains("timeline reset"), "{message}");
                assert!(
                    message.contains(if failed == 0 { "left" } else { "right" }),
                    "{message}"
                );
            }
        }
    }

    #[test]
    fn capture_reference_clock_preserves_gaps_and_rejects_resets() {
        let description =
            "appsrc name=source is-live=true format=time ! appsink name=sink sync=false";
        for invalid_time in [5_141_667, 5_100_000] {
            let mut pair = sample_pair([description, description]).unwrap();
            let pipeline = observer(&pair);
            let sources = pair.test_app_sources();
            let (done_tx, done_rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                for (index, time) in [5_000_000, 5_033_333, 5_133_333, 5_141_667, invalid_time]
                    .into_iter()
                    .enumerate()
                {
                    for (camera, source) in sources.iter().enumerate() {
                        let mut buffer = gst::Buffer::from_mut_slice(vec![index as u8]);
                        // Deliberately skewed samples must not be retimestamped as aligned.
                        buffer
                            .get_mut()
                            .unwrap()
                            .set_pts(gst::ClockTime::from_useconds(
                                time + camera as u64 * 90_000_000,
                            ));
                        source.push_buffer(buffer).unwrap();
                    }
                    let result = pair.next(&AtomicBool::new(false));
                    if index == 4 {
                        assert!(result.unwrap_err().to_string().contains("timeline reset"));
                        assert!(pair.is_stopped());
                    } else {
                        let captured = result.unwrap().unwrap();
                        assert_eq!(captured.pts_us, time as i64 - 5_000_000);
                        for image in captured.images {
                            assert_eq!(
                                image.buffer().unwrap().map_readable().unwrap().as_slice(),
                                &[index as u8]
                            );
                        }
                    }
                }
                drop(pair);
                done_tx.send(()).unwrap();
            });
            if done_rx.recv_timeout(Duration::from_secs(10)).is_err() {
                pipeline.set_state(gst::State::Null).unwrap();
                worker.join().unwrap();
                panic!("capture timestamp test stalled");
            }
            worker.join().unwrap();
            assert_stopped(&pipeline);
        }
    }

    #[test]
    fn finite_inputs_drain_then_report_eos_and_stop_both_branches() {
        let mut pair = sample_pair([FINITE, FINITE]).unwrap();
        let pipeline = observer(&pair);
        let interrupted = AtomicBool::new(false);
        for _ in 0..2 {
            let frames = pair.next(&interrupted).unwrap().unwrap();
            for frame in frames.images {
                assert_eq!(frame.buffer().unwrap().size(), 64 * 32 * 3 / 2);
            }
        }
        assert!(pair.next(&interrupted).unwrap().is_none());
        assert!(pair.is_stopped());
        assert_stopped(&pipeline);
        assert!(pair.next(&interrupted).unwrap().is_none());
    }

    #[test]
    fn cancellation_interrupts_a_starved_camera_in_either_position() {
        for idle in [0, 1] {
            let mut descriptions = [FINITE, FINITE];
            descriptions[idle] = IDLE;
            let mut pair = sample_pair(descriptions).unwrap();
            let pipeline = observer(&pair);
            let interrupted = Arc::new(AtomicBool::new(false));
            let flag = interrupted.clone();
            let (entered_tx, entered_rx) = mpsc::channel();
            let (done_tx, done_rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                entered_tx.send(()).unwrap();
                let result = pair.next(&flag);
                drop(pair);
                done_tx.send(result).unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            // With one live-but-silent appsrc this cannot complete by itself.
            assert!(done_rx.recv_timeout(Duration::from_millis(100)).is_err());
            interrupted.store(true, Ordering::Relaxed);
            assert!(
                done_rx
                    .recv_timeout(Duration::from_secs(5))
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            worker.join().unwrap();
            assert_stopped(&pipeline);
        }
    }

    #[test]
    fn runtime_error_is_not_hidden_by_a_starved_peer() {
        for failed in [0, 1] {
            let mut descriptions = [IDLE, IDLE];
            descriptions[failed] =
                "videotestsrc ! identity error-after=1 ! appsink name=sink sync=false";
            let mut pair = sample_pair(descriptions).unwrap();
            let pipeline = observer(&pair);
            let (tx, rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                let result = pair.next(&AtomicBool::new(false));
                drop(pair);
                tx.send(result).unwrap();
            });
            let error = rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(if failed == 0 { "left" } else { "right" })
            );
            worker.join().unwrap();
            assert_stopped(&pipeline);
        }
    }

    #[test]
    fn bus_error_takes_precedence_over_eos_and_cancellation() {
        for failed in [0, 1] {
            let empty = "videotestsrc num-buffers=0 ! appsink name=sink sync=false";
            let mut pair = sample_pair([empty, empty]).unwrap();
            let pipeline = observer(&pair);
            pair.capture
                .bus
                .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Eos])
                .expect("empty sources must reach EOS");
            pair.capture
                .bus
                .post(
                    gst::message::Error::builder(gst::ResourceError::Read, "capture disconnected")
                        .src(&pair.capture.sinks[failed])
                        .build(),
                )
                .unwrap();
            let error = pair.next(&AtomicBool::new(failed == 1)).unwrap_err();
            assert!(error.to_string().contains("capture disconnected"));
            assert_stopped(&pipeline);
        }
    }

    #[test]
    fn eos_does_not_wait_for_a_starved_peer() {
        for ended in [0, 1] {
            let mut descriptions = [IDLE, IDLE];
            descriptions[ended] = "videotestsrc num-buffers=0 ! appsink name=sink sync=false";
            let mut pair = sample_pair(descriptions).unwrap();
            let pipeline = observer(&pair);
            let (tx, rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                let result = pair.next(&AtomicBool::new(false));
                drop(pair);
                tx.send(result).unwrap();
            });
            assert!(
                rx.recv_timeout(Duration::from_secs(5))
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            worker.join().unwrap();
            assert_stopped(&pipeline);
        }
    }

    #[test]
    fn drop_stops_full_queues_without_a_consumer() {
        let pair = sample_pair([FINITE, FINITE]).unwrap();
        let pipeline = observer(&pair);
        pair.capture
            .bus
            .timed_pop_filtered(gst::ClockTime::from_seconds(5), &[gst::MessageType::Eos])
            .expect("finite sources must reach EOS even with unconsumed samples");
        drop(pair);
        assert_stopped(&pipeline);
    }

    #[test]
    fn invalid_second_branch_returns_an_error() {
        assert!(sample_pair([FINITE, "element_that_does_not_exist"]).is_err());
    }

    #[test]
    fn drop_joins_workers_blocked_on_full_frame_queues() {
        let running = "videotestsrc ! appsink name=sink sync=false";
        let token = Arc::new(());
        let weak = Arc::downgrade(&token);
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (full_tx, full_rx) = mpsc::channel();
        let pair = CapturePair::start(
            [running, running],
            move |_| {
                std::hint::black_box(&token);
                if count.fetch_add(1, Ordering::Relaxed) == 5 {
                    full_tx.send(()).unwrap();
                }
                Ok(())
            },
            None,
        )
        .unwrap();
        let pipeline = observer(&pair);
        // Each worker can extract only three frames before its two-slot queue blocks.
        full_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(weak.upgrade().is_some());
        let (closed_tx, closed_rx) = mpsc::channel();
        let closer = std::thread::spawn(move || {
            drop(pair);
            closed_tx.send(()).unwrap();
        });
        closed_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        closer.join().unwrap();
        assert!(
            weak.upgrade().is_none(),
            "worker closures must be released before drop returns"
        );
        assert_stopped(&pipeline);
    }

    #[test]
    fn extraction_error_is_visible_while_the_other_input_is_starved() {
        for failed in [0, 1] {
            let mut descriptions = [IDLE, IDLE];
            descriptions[failed] = FINITE;
            let mut pair = CapturePair::<()>::start(
                descriptions,
                |_| {
                    Err(SourceError::Read {
                        reason: "extraction failed".into(),
                    })
                },
                None,
            )
            .unwrap();
            let (tx, rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                let result = pair.next(&AtomicBool::new(false));
                drop(pair);
                tx.send(result).unwrap();
            });
            let error = rx
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err();
            assert!(error.to_string().contains("extraction failed"));
            assert!(
                error
                    .to_string()
                    .contains(if failed == 0 { "left" } else { "right" })
            );
            worker.join().unwrap();
        }
    }
}
