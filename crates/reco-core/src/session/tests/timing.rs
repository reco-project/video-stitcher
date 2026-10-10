//! Image/timestamp association across synchronous and delayed sink delivery.

use super::*;
use crate::source::SourceFrame;

type TimedImage = (Option<i64>, u8, u32, u32);
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
            .push((frame.pts_us, frame.data[centre], frame.width, frame.height));
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

#[test]
fn cpu_resized_output_preserves_frame_geometry_in_both_delivery_modes() {
    for threaded in [false, true] {
        let mut session = build_cpu_session();
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
        let dimensions = [(64, 64), (80, 48), (48, 80), (64, 64)];
        for (index, &(width, height)) in dimensions.iter().enumerate() {
            session.core_mut().resize_preview(width, height).unwrap();
            session
                .render_frame(
                    &solid_frame(),
                    crate::geometry::RenderPose::default(),
                    Some(index as i64),
                )
                .unwrap();
        }
        session.finish().unwrap();
        assert_eq!(
            received
                .lock()
                .unwrap()
                .iter()
                .map(|frame| (frame.2, frame.3))
                .collect::<Vec<_>>(),
            dimensions,
            "threaded={threaded}"
        );
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
#[cfg(feature = "gpu")]
#[ignore = "requires GPU render and readback"]
fn gpu_resize_discards_timestamps_with_their_pending_images() {
    for count in [1, 3] {
        for (round_trip, threaded) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut session = build_test_session(None, None, 1).unwrap();
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
            let pose = crate::geometry::RenderPose::default();
            let image = solid_frame();
            session.render_frame(&image, pose, Some(10)).unwrap();
            session.render_frame(&image, pose, Some(20)).unwrap();
            assert!(received.lock().unwrap().is_empty());
            let size = session.core().output_dims();
            session
                .core_mut()
                .resize_preview(size.0 + 16, size.1 + 8)
                .unwrap();
            if round_trip {
                session.core_mut().resize_preview(size.0, size.1).unwrap();
            }
            for index in 0..count {
                session
                    .render_frame(&image, pose, Some(1_000 + index))
                    .unwrap();
            }
            session.finish().unwrap();
            let received = received.lock().unwrap();
            assert_eq!(
                received.iter().map(|frame| frame.0).collect::<Vec<_>>(),
                (0..count)
                    .map(|index| Some(1_000 + index))
                    .collect::<Vec<_>>()
            );
            let expected_size = if round_trip {
                size
            } else {
                (size.0 + 16, size.1 + 8)
            };
            assert!(
                received
                    .iter()
                    .all(|frame| (frame.2, frame.3) == expected_size)
            );
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires GPU render and readback"]
fn unchanged_or_rejected_resize_preserves_pending_images_and_timestamps() {
    for invalid in [false, true] {
        let mut session = build_test_session(None, None, 1).unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        session
            .add_sink(
                Box::new(TimedImages(received.clone())),
                super::super::SinkOptions::threaded(1),
            )
            .unwrap();
        let pose = crate::geometry::RenderPose::default();
        let mut image = solid_frame();
        session.render_frame(&image, pose, Some(10)).unwrap();
        let dimensions = session.core().output_dims();
        if invalid {
            assert!(session.core_mut().resize_preview(0, dimensions.1).is_err());
        } else {
            session
                .core_mut()
                .resize_preview(dimensions.0, dimensions.1)
                .unwrap();
        }
        let FrameSet::Yuv420p(cameras) = &mut image else {
            unreachable!()
        };
        for camera in cameras {
            Arc::make_mut(camera).y.fill(220);
        }
        session.render_frame(&image, pose, Some(20)).unwrap();
        session.finish().unwrap();
        let received = received.lock().unwrap();
        assert_eq!(
            received.iter().map(|frame| frame.0).collect::<Vec<_>>(),
            [Some(10), Some(20)]
        );
        assert!(received[1].1 > received[0].1 + 20, "{received:?}");
        assert!(
            received
                .iter()
                .all(|frame| (frame.2, frame.3) == dimensions)
        );
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
