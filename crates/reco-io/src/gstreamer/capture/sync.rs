//! Clock-driven adapter for the transport-independent selector.

use super::*;
use crate::frame_sync::{Capture as Image, Config, Selector};
use reco_core::source::CaptureTiming;
use std::sync::{Arc, Condvar, Mutex, mpsc::TryRecvError};
use std::time::Duration;

type Wake = Arc<(Mutex<bool>, Condvar)>;

pub(super) struct SyncState<T> {
    pub(super) wake: Wake,
    selector: Selector<T>,
    config: Config,
    fps: u32,
    remainder: u32,
    slot: Option<i64>,
    ended: [bool; 2],
}

fn read_error(error: impl std::fmt::Display) -> SourceError {
    SourceError::Read {
        reason: error.to_string(),
    }
}

impl<T> SyncState<T> {
    pub(super) fn new(config: Config, fps: u32) -> Result<Self, SourceError> {
        let invalid = |reason: String| SourceError::Init {
            path: "capture synchronization".into(),
            reason,
        };
        if fps == 0 || fps > 1_000_000 {
            return Err(invalid(
                "capture cadence must be between 1 and 1000000 fps".into(),
            ));
        }
        Ok(Self {
            wake: Arc::new((Mutex::new(false), Condvar::new())),
            selector: Selector::new(2, config).map_err(|e| invalid(e.to_string()))?,
            config,
            fps,
            remainder: 0,
            slot: None,
            ended: [false; 2],
        })
    }

    fn advance(&mut self, at_us: i64) -> Result<(), SourceError> {
        let slot = self.slot.unwrap();
        // Preserve the cadence's phase, including fractional microseconds.
        // A future-of-slot capture may require skipping an already covered slot.
        let ticks = (((at_us - slot) as u128 + 1) * u128::from(self.fps)
            - u128::from(self.remainder))
        .div_ceil(1_000_000);
        let numerator = ticks * 1_000_000 + u128::from(self.remainder);
        let step = i64::try_from(numerator / u128::from(self.fps))
            .map_err(|_| read_error("capture cadence overflow"))?;
        self.slot = Some(
            slot.checked_add(step)
                .ok_or_else(|| read_error("capture cadence overflow"))?,
        );
        self.remainder = (numerator % u128::from(self.fps)) as u32;
        Ok(())
    }
}

pub(super) fn clock_us(pipeline: &gst::Pipeline) -> Result<i64, SourceError> {
    pipeline
        .current_running_time()
        .and_then(|time| i64::try_from(time.useconds()).ok())
        .ok_or_else(|| read_error("capture has no representable running clock"))
}

pub(super) fn notify(wake: &Wake) {
    *wake.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
    wake.1.notify_one();
}

impl<T: Clone + Send + 'static> CapturePair<T> {
    pub(super) fn next_synced(
        &mut self,
        interrupted: &AtomicBool,
    ) -> Result<Option<CapturedPair<T>>, SourceError> {
        let mut resuming = true;
        loop {
            self.check_errors()?;
            if interrupted.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let state = self.sync.as_mut().unwrap();
            // Clear before draining. A notification during the drain then prevents sleeping.
            *state.wake.0.lock().unwrap_or_else(|e| e.into_inner()) = false;
            let mut latest = None::<i64>;
            for (camera, receiver) in self.receivers.iter().enumerate() {
                if state.ended[camera] {
                    continue;
                }
                // Bound work even when producers continuously fill their queues.
                for _ in 0..2 {
                    match receiver.try_recv() {
                        Ok(Some(frame)) => {
                            latest =
                                Some(latest.map_or(frame.pts_us, |last| last.max(frame.pts_us)));
                            state
                                .selector
                                .push(
                                    camera,
                                    Image {
                                        time_us: frame.pts_us,
                                        image: frame.images,
                                    },
                                    frame.arrived_us.ok_or_else(|| {
                                        read_error("missing capture receipt time")
                                    })?,
                                )
                                .map_err(read_error)?;
                        }
                        Ok(None) => {
                            state.ended[camera] = true;
                            break;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            self.check_errors()?;
                            return Err(read_error(
                                "capture worker disconnected without a terminal result",
                            ));
                        }
                    }
                }
            }
            if state
                .ended
                .iter()
                .enumerate()
                .any(|(camera, ended)| *ended && !state.selector.has_pending(camera))
            {
                return Ok(None);
            }
            if state.slot.is_none() {
                state.slot = latest;
            }
            let now = clock_us(&self.capture.pipeline)?;
            let mut wait_us = 20_000;
            if let Some(mut slot) = state.slot {
                if resuming {
                    // A stalled consumer skips missed slots, not capture identity or age.
                    // Once this call starts waiting, its deadline must not move.
                    if now.saturating_sub(slot) > state.config.wait_us {
                        state.advance(now - 1)?;
                        slot = state.slot.unwrap();
                    }
                    resuming = false;
                }
                match state.selector.poll(slot, now) {
                    Ok(Some(selection)) => {
                        state.advance(selection.at_us)?;
                        let origin = *self.origin_us.get_or_insert(selection.at_us);
                        let mut captures = selection.captures.into_iter();
                        let left = captures.next().unwrap();
                        let right = captures.next().unwrap();
                        return Ok(Some(CapturedPair {
                            images: [left.image, right.image],
                            pts_us: selection.at_us - origin,
                            capture: CaptureTiming {
                                times_us: vec![left.time_us, right.time_us],
                                at_us: selection.at_us,
                                max_age_us: Some(state.config.max_age_us as u64),
                            },
                        }));
                    }
                    Ok(None) => {
                        let deadline = slot
                            .checked_add(state.config.wait_us)
                            .ok_or_else(|| read_error("capture deadline overflow"))?;
                        wait_us = if now < slot {
                            slot - now
                        } else {
                            deadline - now
                        };
                    }
                    Err(error) => return Err(read_error(error)),
                }
            }
            let ready = state.wake.0.lock().unwrap_or_else(|e| e.into_inner());
            let _guard = state
                .wake
                .1
                .wait_timeout_while(
                    ready,
                    Duration::from_micros(wait_us.clamp(1, 20_000) as u64),
                    |ready| !*ready,
                )
                .unwrap_or_else(|e| e.into_inner());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gstreamer_app::AppSrc;

    const INPUT: &str =
        "appsrc name=source is-live=true format=time ! appsink name=sink sync=false";

    fn pair() -> CapturePair<Arc<gst::Sample>> {
        CapturePair::start(
            [INPUT; 2],
            |sample| Ok(Arc::new(sample.clone())),
            Some((
                Config {
                    wait_us: 80_000,
                    skew_us: 20_000,
                    max_age_us: 250_000,
                    capacity: 2,
                },
                10,
            )),
        )
        .unwrap()
    }

    fn push(source: &AppSrc, time: i64) {
        let mut buffer = gst::Buffer::from_mut_slice(vec![42]);
        buffer
            .get_mut()
            .unwrap()
            .set_pts(gst::ClockTime::from_useconds(time as u64));
        source.push_buffer(buffer).unwrap();
    }

    fn wait_until(pair: &CapturePair<Arc<gst::Sample>>, time: i64) {
        loop {
            let remaining = time - clock_us(&pair.capture.pipeline).unwrap();
            if remaining <= 0 {
                return;
            }
            std::thread::sleep(Duration::from_micros(remaining as u64));
        }
    }

    #[test]
    fn queued_live_burst_keeps_both_due_frames() {
        let mut pair = CapturePair::start(
            [INPUT; 2],
            |sample| Ok(Arc::new(sample.clone())),
            Some((
                Config {
                    wait_us: 150_000,
                    skew_us: 40_000,
                    max_age_us: 500_000,
                    capacity: 4,
                },
                30,
            )),
        )
        .unwrap();
        let inputs = pair.test_app_sources();
        let start = clock_us(&pair.capture.pipeline).unwrap();
        for input in &inputs {
            push(input, start);
        }
        pair.next(&AtomicBool::new(false)).unwrap().unwrap();
        wait_until(&pair, start + 80_000);
        for input in &inputs {
            push(input, start + 33_333);
            push(input, start + 66_666);
        }
        for offset in [33_333, 66_666] {
            let frame = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            assert_eq!(frame.pts_us, offset);
            assert_eq!(frame.capture.times_us, [start + offset; 2]);
        }
        pair.stop().unwrap();
    }

    #[test]
    fn live_drop_reuses_same_owner_and_recovers_in_both_directions() {
        for missing in 0..2 {
            let mut pair = pair();
            let inputs = pair.test_app_sources();
            let start = clock_us(&pair.capture.pipeline).unwrap();
            for input in &inputs {
                push(input, start);
            }
            let first = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            let weak = Arc::downgrade(&first.images[missing]);
            wait_until(&pair, start + 100_000);
            push(&inputs[1 - missing], start + 100_000);
            let held = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            assert!(Arc::ptr_eq(&first.images[missing], &held.images[missing]));
            assert_eq!(held.capture.times_us[missing], start);
            assert_eq!(held.capture.times_us[1 - missing], start + 100_000);
            assert_eq!(held.capture.at_us, start + 100_000);
            assert_eq!(held.capture.max_age_us, Some(250_000));
            assert_eq!(held.pts_us, 100_000);
            assert!(clock_us(&pair.capture.pipeline).unwrap() >= start + 180_000);
            wait_until(&pair, start + 200_000);
            for input in &inputs {
                push(input, start + 200_000);
            }
            let recovered = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            assert_eq!(recovered.capture.times_us, [start + 200_000; 2]);
            assert!(!Arc::ptr_eq(
                &first.images[missing],
                &recovered.images[missing]
            ));
            drop(first);
            drop(held);
            assert!(weak.upgrade().is_none());
            for input in &inputs {
                input.end_of_stream().unwrap();
            }
            assert!(pair.next(&AtomicBool::new(false)).unwrap().is_none());
            assert!(pair.is_stopped());
        }
    }

    #[test]
    fn delayed_live_backlog_holds_existing_owners_until_cameras_rejoin() {
        for delayed in 0..2 {
            let mut pair = pair();
            let inputs = pair.test_app_sources();
            let start = clock_us(&pair.capture.pipeline).unwrap();
            for input in &inputs {
                push(input, start);
            }
            pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            wait_until(&pair, start + 100_000);
            push(&inputs[1 - delayed], start + 100_000);
            let held = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            assert_eq!(held.capture.times_us[delayed], start);
            assert_eq!(held.capture.times_us[1 - delayed], start + 100_000);
            wait_until(&pair, start + 200_000);
            push(&inputs[delayed], start + 33_333);
            push(&inputs[1 - delayed], start + 200_000);
            let repeated = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            assert_eq!(repeated.pts_us, 200_000);
            assert_eq!(repeated.capture.times_us, held.capture.times_us);
            for camera in 0..2 {
                assert!(Arc::ptr_eq(&held.images[camera], &repeated.images[camera]));
            }
            wait_until(&pair, start + 300_000);
            for input in &inputs {
                push(input, start + 300_000);
            }
            let recovered = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            assert_eq!(recovered.capture.times_us, [start + 300_000; 2]);
            pair.stop().unwrap();
        }
    }

    #[test]
    fn both_frozen_inputs_expire_and_release_their_owners() {
        let mut pair = pair();
        let inputs = pair.test_app_sources();
        let start = clock_us(&pair.capture.pipeline).unwrap();
        for input in &inputs {
            push(input, start);
        }
        let first = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
        let weak = Arc::downgrade(&first.images[0]);
        drop(first);
        for pts in [100_000, 200_000] {
            assert_eq!(
                pair.next(&AtomicBool::new(false)).unwrap().unwrap().pts_us,
                pts
            );
        }
        assert!(
            pair.next(&AtomicBool::new(false))
                .unwrap_err()
                .to_string()
                .contains("age limit")
        );
        assert!(pair.is_stopped());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn waiting_for_initial_pair_observes_cancellation_and_worker_failure() {
        let mut pair = pair();
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = cancelled.clone();
        let cancel = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            flag.store(true, Ordering::Relaxed);
        });
        assert!(pair.next(&cancelled).unwrap().is_none());
        cancel.join().unwrap();
        assert!(pair.is_stopped());

        let mut pair = self::pair();
        let inputs = pair.test_app_sources();
        push(&inputs[0], 0);
        push(&inputs[0], 0);
        let error = pair.next(&AtomicBool::new(false)).unwrap_err().to_string();
        assert!(error.contains("did not advance"), "{error}");
        assert!(pair.is_stopped());
    }

    #[test]
    fn live_consumer_stall_skips_backlog_without_replaying_or_stopping() {
        let input = "videotestsrc is-live=true ! video/x-raw,format=I420,width=64,height=32,framerate=30/1 ! appsink name=sink sync=false";
        let mut pair = CapturePair::start(
            [input; 2],
            |sample| Ok(Arc::new(sample.clone())),
            Some((
                Config {
                    wait_us: 80_000,
                    skew_us: 20_000,
                    max_age_us: 250_000,
                    capacity: 2,
                },
                30,
            )),
        )
        .unwrap();
        let first = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
        drop(first);
        // Long enough to fill worker and appsink queues while both cameras continue.
        std::thread::sleep(Duration::from_millis(650));
        let resumed_at = clock_us(&pair.capture.pipeline).unwrap();
        let mut previous = 0;
        for _ in 0..6 {
            let frame = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
            assert!(
                frame.capture.at_us >= resumed_at - 100_000,
                "replayed old output: assessment={}, resumed={resumed_at}",
                frame.capture.at_us
            );
            assert!(frame.pts_us > previous);
            previous = frame.pts_us;
            assert!(
                frame
                    .capture
                    .times_us
                    .iter()
                    .all(|&time| frame.capture.at_us - time <= 250_000)
            );
        }
        assert!(!pair.is_stopped());
    }

    #[test]
    fn backlogged_startup_and_repeated_stalls_release_bounded_image_owners() {
        use std::sync::atomic::AtomicUsize;
        struct OwnedSample {
            _sample: gst::Sample,
            count: Arc<AtomicUsize>,
        }
        impl Drop for OwnedSample {
            fn drop(&mut self) {
                self.count.fetch_sub(1, Ordering::Relaxed);
            }
        }
        for capacity in [1, 4] {
            let input = "videotestsrc is-live=true ! video/x-raw,format=I420,width=64,height=32,framerate=30/1 ! appsink name=sink sync=false";
            let count = Arc::new(AtomicUsize::new(0));
            let peak = Arc::new(AtomicUsize::new(0));
            let owned = count.clone();
            let high = peak.clone();
            let mut pair = CapturePair::start(
                [input; 2],
                move |sample| {
                    high.fetch_max(owned.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
                    Ok(Arc::new(OwnedSample {
                        _sample: sample.clone(),
                        count: owned.clone(),
                    }))
                },
                Some((
                    Config {
                        wait_us: 80_000,
                        skew_us: 20_000,
                        max_age_us: 250_000,
                        capacity,
                    },
                    30,
                )),
            )
            .unwrap();
            let mut previous = None;
            for delay in [650, 140, 380] {
                std::thread::sleep(Duration::from_millis(delay));
                let now = clock_us(&pair.capture.pipeline).unwrap();
                for _ in 0..3 {
                    let frame = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
                    assert!(frame.capture.at_us >= now - 100_000);
                    assert!(previous.is_none_or(|last| frame.pts_us > last));
                    previous = Some(frame.pts_us);
                }
            }
            // Per camera: selector queue, previous selection, worker queue and one blocked extraction.
            // Appsink's not-yet-extracted samples and downstream GPU reads are separate native owners.
            assert!(peak.load(Ordering::Relaxed) <= 2 * (capacity + 1 + 2 + 1));
            pair.stop().unwrap();
            assert_eq!(count.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn resuming_after_both_cameras_freeze_does_not_renew_old_images() {
        let mut pair = pair();
        let inputs = pair.test_app_sources();
        let start = clock_us(&pair.capture.pipeline).unwrap();
        for input in &inputs {
            push(input, start);
        }
        let first = pair.next(&AtomicBool::new(false)).unwrap().unwrap();
        let weak = Arc::downgrade(&first.images[0]);
        drop(first);
        std::thread::sleep(Duration::from_millis(400));
        let error = pair.next(&AtomicBool::new(false)).unwrap_err();
        assert!(error.to_string().contains("age limit"), "{error}");
        assert!(weak.upgrade().is_none());
        assert!(pair.is_stopped());
    }

    #[test]
    fn live_cadence_handles_clock_rate_drift_and_short_camera_loss() {
        for wait_us in [15_000, 80_000] {
            let config = Config {
                wait_us,
                skew_us: 20_000,
                max_age_us: 250_000,
                capacity: 4,
            };
            let mut state = SyncState::<i64>::new(config, 30).unwrap();
            state.slot = Some(0);
            let mut indices = [0i64; 2];
            let mut fresh = [0; 2];
            let mut previous = [-1; 2];
            // Ten minutes of source time, simulated without wall-clock sleeps.
            for _ in 0..18_000 {
                let slot = state.slot.unwrap();
                let now = slot + config.wait_us;
                for (camera, index) in indices.iter_mut().enumerate() {
                    loop {
                        // Cameras run at 29.97 and 30.003 Hz with different phases.
                        let time = *index * 1_000_000_000 / [29_970, 30_003][camera]
                            + camera as i64 * 2_000;
                        let arrival = time + 2_000;
                        if arrival > now {
                            break;
                        }
                        if *index % 101 != 50 {
                            state
                                .selector
                                .push(
                                    camera,
                                    Image {
                                        time_us: time,
                                        image: time,
                                    },
                                    arrival,
                                )
                                .unwrap();
                        }
                        *index += 1;
                    }
                }
                let set = state
                    .selector
                    .poll(slot, now)
                    .unwrap_or_else(|error| {
                        panic!("{error}: slot={slot}, previous={previous:?}, indices={indices:?}")
                    })
                    .unwrap();
                for (camera, capture) in set.captures.iter().enumerate() {
                    assert!(capture.time_us >= previous[camera]);
                    fresh[camera] += usize::from(capture.time_us != previous[camera]);
                    previous[camera] = capture.time_us;
                    assert!(set.at_us - capture.time_us <= config.max_age_us);
                }
                state.advance(set.at_us).unwrap();
            }
            assert!(fresh.iter().all(|&count| count > 17_000), "{fresh:?}");
        }
    }

    #[test]
    fn fractional_cadence_does_not_accumulate_rounding_drift() {
        let mut state = SyncState::<()>::new(
            Config {
                wait_us: 0,
                skew_us: 0,
                max_age_us: 0,
                capacity: 1,
            },
            30,
        )
        .unwrap();
        state.slot = Some(0);
        for _ in 0..30 {
            state.advance(state.slot.unwrap()).unwrap();
        }
        assert_eq!(state.slot, Some(1_000_000));
        state.advance(1_070_000).unwrap();
        assert_eq!(state.slot, Some(1_100_000));
        assert!(state.advance(i64::MAX).is_err());
    }
}
