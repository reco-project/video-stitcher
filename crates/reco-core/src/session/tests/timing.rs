//! Image/timestamp association across synchronous and delayed sink delivery.

use super::*;
use crate::source::SourceFrame;

type TimedImage = (Option<i64>, u8);
struct TimedImages(Arc<Mutex<Vec<TimedImage>>>);

struct TimedSource<'a> {
    times: std::iter::Copied<std::slice::Iter<'a, Option<i64>>>,
    index: u8,
}

impl<'a> TimedSource<'a> {
    fn new(times: &'a [Option<i64>]) -> Self {
        Self {
            times: times.iter().copied(),
            index: 0,
        }
    }
}

impl FrameSource for TimedSource<'_> {
    fn info(&self) -> SourceInfo {
        MockSource::new(0).info()
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        Ok(self.times.next().map(|pts_us| {
            let mut images = solid_frame();
            let FrameSet::Yuv420p(cameras) = &mut images else {
                unreachable!()
            };
            for camera in cameras {
                Arc::make_mut(camera).y.fill(48 + self.index * 40);
            }
            self.index += 1;
            SourceFrame {
                images,
                pts_us,
                capture: None,
            }
        }))
    }
}

impl OutputSink for TimedImages {
    fn name(&self) -> &str {
        "timed-images"
    }
    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        let centre = (frame.height / 2 * frame.width + frame.width / 2) as usize;
        self.0
            .lock()
            .unwrap()
            .push((frame.pts_us, frame.data[centre]));
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

fn check_delivery(mut session: StitchSession, count: usize, threaded: bool) {
    let received = Arc::new(Mutex::new(Vec::new()));
    session
        .add_sink(
            Box::new(TimedImages(received.clone())),
            if threaded {
                super::super::SinkOptions::threaded(1)
            } else {
                super::super::SinkOptions::inline_lossy()
            },
        )
        .unwrap();
    let times = [Some(0), Some(33_333), Some(133_333), Some(141_667)];
    let pose = crate::geometry::RenderPose::default();
    for (i, &time) in times.iter().take(count).enumerate() {
        let mut image = solid_frame();
        let FrameSet::Yuv420p(cameras) = &mut image else {
            unreachable!()
        };
        for camera in cameras {
            Arc::make_mut(camera).y.fill(48 + i as u8 * 40);
        }
        session.render_frame(&image, pose, time).unwrap();
    }
    assert_eq!(session.frame_count(), count as u64);
    session.finish().unwrap();
    session.finish().unwrap();
    assert_eq!(
        session.frame_count(),
        count as u64,
        "drain must not count renders twice"
    );
    let received = received.lock().unwrap();
    assert_eq!(received.len(), count);
    assert_eq!(
        received.iter().map(|f| f.0).collect::<Vec<_>>(),
        times[..count]
    );
    for pair in received.windows(2) {
        assert!(
            pair[1].1 > pair[0].1 + 20,
            "timestamp must stay with its image: {received:?}"
        );
    }
}

#[test]
fn cpu_output_times_stay_with_images() {
    for count in 0..=4 {
        for threaded in [false, true] {
            check_delivery(build_cpu_session(), count, threaded);
        }
    }
}

fn check_source_delivery(mut session: StitchSession, count: usize, depth: usize, threaded: bool) {
    let times = [Some(0), Some(33_333), Some(133_333), Some(141_667)];
    let received = Arc::new(Mutex::new(Vec::new()));
    session.core_mut().set_constrained_look(false);
    session.set_lookahead(depth, usize::MAX);
    session
        .add_sink(
            Box::new(TimedImages(received.clone())),
            if threaded {
                super::super::SinkOptions::threaded(1)
            } else {
                super::super::SinkOptions::inline_lossy()
            },
        )
        .unwrap();
    let result = session
        .run(
            &mut TimedSource::new(&times[..count]),
            u64::MAX,
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
    assert_eq!(result.frames_processed, count as u64);
    session.finish().unwrap();
    let received = received.lock().unwrap();
    assert_eq!(
        received.iter().map(|f| f.0).collect::<Vec<_>>(),
        times[..count]
    );
    for pair in received.windows(2) {
        assert!(
            pair[1].1 > pair[0].1 + 20,
            "image association lost: {received:?}"
        );
    }
}

#[test]
fn cpu_source_times_survive_immediate_lookahead_and_short_drain() {
    for (count, depth) in [(0, 8), (1, 8), (2, 8), (4, 2), (4, 0)] {
        for threaded in [false, true] {
            check_source_delivery(build_cpu_session(), count, depth, threaded);
        }
    }
}

#[test]
fn invalid_source_times_fail_even_without_an_output_sink() {
    for times in [
        vec![Some(-1)],
        vec![Some(0), Some(0)],
        vec![Some(2), Some(1)],
        vec![None, Some(1)],
        vec![Some(0), None],
    ] {
        for depth in [0, 2] {
            let mut session = build_cpu_session();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, usize::MAX);
            let error = session
                .run(
                    &mut TimedSource::new(&times),
                    u64::MAX,
                    &AtomicBool::new(false),
                    None,
                )
                .unwrap_err();
            assert!(
                error.to_string().contains("source presentation timestamps"),
                "{error}"
            );
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires GPU render and readback"]
fn gpu_output_times_stay_with_images_from_source_through_lookahead() {
    for (count, depth) in [(0, 8), (1, 8), (2, 8), (4, 2), (4, 0)] {
        for threaded in [false, true] {
            check_source_delivery(
                build_test_session(None, None, 1).unwrap(),
                count,
                depth,
                threaded,
            );
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires GPU render and readback"]
fn gpu_output_times_stay_with_images_including_short_drain() {
    for count in 0..=4 {
        for threaded in [false, true] {
            check_delivery(build_test_session(None, None, 1).unwrap(), count, threaded);
        }
    }
}
