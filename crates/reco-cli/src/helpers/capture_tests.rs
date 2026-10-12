//! Capture command frame limits, timing and shutdown through the shared session.

use super::tests::{session, session_with_format};
use super::{run_capture, run_capture_source};
use reco_core::session::{SinkOptions, types::RunCompletion};
use reco_core::sink::{OutputFrame, OutputSink, PixelFormat, SinkError, SinkInput};
use reco_core::source::{
    FrameSet, FrameSource, Nv12Data, SourceError, SourceFrame, SourceInfo, Yuv420pData,
};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

const TIMES: [i64; 4] = [0, 33_333, 133_333, 141_667];

struct Capture {
    nv12: bool,
    times: Vec<i64>,
    reads: usize,
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl FrameSource for Capture {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            camera_sizes: vec![(64, 32)],
            fps: 30.0,
            fps_rational: Some((30, 1)),
            total_frames: None,
        }
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        let Some(&pts) = self.times.get(self.reads) else {
            return Ok(None);
        };
        self.reads += 1;
        let y = vec![80; 64 * 32];
        let images = if self.nv12 {
            FrameSet::Nv12(vec![Arc::new(Nv12Data {
                size: (64, 32),
                y,
                uv: vec![128; 64 * 16],
            })])
        } else {
            FrameSet::Yuv420p(vec![Arc::new(Yuv420pData {
                size: (64, 32),
                y,
                u: vec![128; 32 * 16],
                v: vec![128; 32 * 16],
            })])
        };
        Ok(Some(SourceFrame {
            images,
            pts_us: Some(pts),
            capture: None,
        }))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.events.lock().unwrap().push("stopped");
    }
}

struct Recording {
    times: Arc<Mutex<Vec<Option<i64>>>>,
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl OutputSink for Recording {
    fn name(&self) -> &str {
        "capture timing"
    }
    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(PixelFormat::Nv12)
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        self.times.lock().unwrap().push(frame.pts_us);
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        self.events.lock().unwrap().push("finished");
        Ok(())
    }
}

#[test]
fn capture_counts_every_frame_preserves_timing_and_finalizes() {
    check_capture_counts(false, &[0, 1, 3, u64::MAX], &[0, 2, 8]);
}

#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_capture_uses_shared_timing_and_lookahead() {
    check_capture_counts(true, &[3], &[0, 2]);
}

fn check_capture_counts(gpu: bool, limits: &[u64], depths: &[usize]) {
    for nv12 in [false, true] {
        for &depth in depths {
            for &limit in limits {
                let mut session = session_with_format(gpu, nv12);
                session.core_mut().set_constrained_look(false);
                session.set_lookahead(depth, 1024 * 1024);
                let times = Arc::new(Mutex::new(Vec::new()));
                let events = Arc::new(Mutex::new(Vec::new()));
                session
                    .add_sink(
                        Box::new(Recording {
                            times: times.clone(),
                            events: events.clone(),
                        }),
                        SinkOptions::threaded(1),
                    )
                    .unwrap();
                let result = run_capture(&mut session, |session| {
                    let mut capture = Capture {
                        nv12,
                        times: TIMES.to_vec(),
                        reads: 0,
                        events: events.clone(),
                    };
                    let result =
                        run_capture_source(session, &mut capture, limit, &AtomicBool::new(false))?;
                    if limit == 0 {
                        assert_eq!(capture.reads, 0);
                    }
                    Ok(result)
                })
                .unwrap();
                let count = TIMES.len().min(limit as usize);
                assert_eq!(result.frames_processed, count as u64);
                assert_eq!(
                    *times.lock().unwrap(),
                    TIMES[..count]
                        .iter()
                        .map(|&time| Some(time))
                        .collect::<Vec<_>>()
                );
                assert_eq!(*events.lock().unwrap(), ["stopped", "finished"]);
            }
        }
    }
}

#[test]
fn capture_cancellation_and_clock_reset_still_finalize() {
    for depth in [0, 2] {
        for cancelled in [false, true] {
            let mut session = session(false);
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, 1024 * 1024);
            let events = Arc::new(Mutex::new(Vec::new()));
            let times = Arc::new(Mutex::new(Vec::new()));
            session
                .add_sink(
                    Box::new(Recording {
                        times: times.clone(),
                        events: events.clone(),
                    }),
                    SinkOptions::threaded(1),
                )
                .unwrap();
            let result = run_capture(&mut session, |session| {
                let mut capture = Capture {
                    nv12: true,
                    times: vec![0, 33_333, 0],
                    reads: 0,
                    events: events.clone(),
                };
                let result = run_capture_source(
                    session,
                    &mut capture,
                    u64::MAX,
                    &AtomicBool::new(cancelled),
                );
                if cancelled {
                    assert_eq!(capture.reads, 0);
                }
                Ok(result?)
            });
            if cancelled {
                assert_eq!(result.unwrap().completion, RunCompletion::Cancelled);
                assert!(times.lock().unwrap().is_empty());
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("presentation timestamps")
                );
            }
            assert_eq!(*events.lock().unwrap(), ["stopped", "finished"]);
        }
    }
}
