//! Supplied-source ownership, formats and failure finalization through the real job.

use super::*;
use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::projection::LShape;
use reco_core::source::{FrameSet, Nv12Data, SourceError, SourceFrame, SourceInfo, Yuv420pData};
use std::sync::atomic::Ordering;

const W: u32 = 64;
const H: u32 = 32;

struct Images {
    stopped: Arc<AtomicBool>,
    index: usize,
    nv12: bool,
    native: bool,
    fail: bool,
}

impl Images {
    fn new(stopped: Arc<AtomicBool>) -> Self {
        Self {
            stopped,
            index: 0,
            nv12: false,
            native: false,
            fail: false,
        }
    }
}

impl Drop for Images {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
    }
}

impl FrameSource for Images {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            camera_sizes: vec![(W, H); 2],
            fps: 30.0,
            fps_rational: Some((30, 1)),
            total_frames: None,
        }
    }

    fn is_gpu_resident(&self) -> bool {
        self.native
    }

    #[cfg(feature = "gpu")]
    fn input_format(&self) -> reco_core::render::renderer::InputFormat {
        use reco_core::render::renderer::InputFormat;
        if self.nv12 {
            InputFormat::Nv12
        } else {
            InputFormat::Yuv420p
        }
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        if self.index == 12 {
            return if self.fail {
                Err(SourceError::Read {
                    reason: "capture failed".into(),
                })
            } else {
                Ok(None)
            };
        }
        let y = vec![60 + 30 * (self.index % 4) as u8; (W * H) as usize];
        let images = if self.nv12 {
            let image = Arc::new(Nv12Data {
                size: (W, H),
                y,
                uv: vec![128; (W * H / 2) as usize],
            });
            FrameSet::Nv12(vec![image.clone(), image])
        } else {
            let image = Arc::new(Yuv420pData {
                size: (W, H),
                y,
                u: vec![128; (W * H / 4) as usize],
                v: vec![128; (W * H / 4) as usize],
            });
            FrameSet::Yuv420p(vec![image.clone(), image])
        };
        let pts_us = self.index as i64 * 1_000_000 / 30 + if self.index >= 2 { 66_667 } else { 0 };
        self.index += 1;
        Ok(Some(SourceFrame {
            images,
            pts_us: Some(pts_us),
            capture: None,
        }))
    }
}

fn calibration() -> Calibration {
    let lens = Lens::fisheye(W, H, 32.0, 32.0, 32.0, 16.0, [0.0; 4]);
    Calibration::new(
        vec![lens.clone(), lens],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.0,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

#[test]
fn supplied_source_rejects_inapplicable_options_before_creating_output() {
    for option in 0..5 {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("unexpected.mp4");
        let stopped = Arc::new(AtomicBool::new(false));
        let mut source = Images::new(stopped.clone());
        source.native = option == 3;
        let mut cal = calibration();
        if option == 4 {
            cal.projection = reco_core::projection::Cylinder::default().into();
        }
        let job = StitchJob::from_source(source, cal, &output).cpu();
        let job = match option {
            0 => job.start_time(1.0),
            1 => job.sync_offset(0),
            2 => job.audio(AudioMode::CopyFrom(0)),
            _ => job,
        };
        assert!(job.run(&Arc::new(AtomicBool::new(false))).is_err());
        assert!(stopped.load(Ordering::SeqCst));
        assert!(!output.exists());
    }
}

fn lifecycle(gpu: bool) {
    for nv12 in [false, true] {
        for lookahead in [0.0, 0.1] {
            for fail in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let output = dir.path().join("capture.mp4");
                let stopped = Arc::new(AtomicBool::new(false));
                let mut source = Images::new(stopped.clone());
                source.nv12 = nv12;
                source.fail = fail;
                let checked = Arc::new(AtomicBool::new(false));
                let finalizing = checked.clone();
                let mut job = StitchJob::from_source(source, calibration(), &output)
                    .resolution(W, H)
                    .encoder_name("libx264")
                    .quality(Quality::Value(100))
                    .lookahead(lookahead)
                    .lookahead_memory_limit(1024 * 1024)
                    .on_session(|session, _| {
                        session.core_mut().set_constrained_look(false);
                        Ok(())
                    })
                    .on_finalizing(move || {
                        assert!(
                            stopped.load(Ordering::SeqCst),
                            "capture must stop before finalizing"
                        );
                        finalizing.store(true, Ordering::SeqCst);
                    });
                #[cfg(feature = "stacked-output")]
                let replay = dir.path().join("replay.mkv");
                #[cfg(feature = "stacked-output")]
                if !nv12 && lookahead > 0.0 {
                    job = job.with_replay_recording(&replay);
                }
                if !gpu {
                    job = job.cpu();
                }
                let result = job.run(&Arc::new(AtomicBool::new(false)));
                if fail {
                    assert!(result.unwrap_err().to_string().contains("capture failed"));
                } else {
                    assert_eq!(result.unwrap().frames_processed, 12);
                }
                assert!(checked.load(Ordering::SeqCst));
                let mut decoder =
                    crate::ffmpeg::decoder::VideoDecoder::open_software(&output).unwrap();
                let mut times = Vec::new();
                while let Some(frame) = decoder.next_frame().unwrap() {
                    assert!(frame.y.iter().any(|&pixel| pixel > 32));
                    times.push(frame.timestamp_us);
                }
                if fail {
                    assert!(!times.is_empty());
                } else {
                    assert_eq!(times.len(), 12);
                    for (index, actual) in times.iter().enumerate() {
                        let expected =
                            index as i64 * 1_000_000 / 30 + if index >= 2 { 66_667 } else { 0 };
                        assert!(
                            (actual - expected).abs() <= 1,
                            "container timestamp rounding: {actual} vs {expected}"
                        );
                    }
                }
                #[cfg(feature = "stacked-output")]
                if !nv12 && lookahead > 0.0 {
                    let mut replay =
                        crate::ffmpeg::decoder::VideoDecoder::open_software(&replay).unwrap();
                    let mut frames = 0;
                    while replay.next_frame().unwrap().is_some() {
                        frames += 1;
                    }
                    assert_eq!(frames, 12);
                }
            }
        }
    }
}

#[test]
fn supplied_host_formats_preserve_timing_and_stop_before_finalization() {
    lifecycle(false);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a working Vulkan/Metal/DX12 device"]
fn supplied_host_formats_configure_the_gpu_before_reading() {
    lifecycle(true);
}
