//! Camera/image identity through capture, inference, lookahead and encoded delivery.

use super::super::*;
use crate::adapters::FfmpegFileEncoder;
use crate::ffmpeg::{decoder::VideoDecoder, encoder::EncoderConfig};
use gstreamer_app::AppSrc;
use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::geometry::CameraIndex;
use reco_core::projection::LShape;
use reco_core::render::viewport::ViewportSize;
use reco_core::session::{SinkOptions, StitchSession};
use reco_core::sink::{OutputFrame, OutputSink, SinkError, SinkInput};
use reco_core::source::FrameSource;
use reco_core::stitch::{CpuExecutor, Executor};
use std::sync::Mutex;

const TIMES: [u64; 4] = [1_000_000, 1_033_333, 1_133_333, 1_141_667];
const MARKERS: [[u8; 2]; 4] = [[48, 144], [80, 176], [112, 208], [64, 160]];
const W: u32 = 64;
const H: u32 = 32;
const PIXELS: usize = (W * H) as usize;

fn capture<T: Send + Sync + 'static>(
    format: &str,
    extract: fn(&[u8], u32, u32) -> Result<T, SourceError>,
    policy: Option<(crate::frame_sync::Config, u32)>,
) -> (HostCapture<T>, [AppSrc; 2]) {
    let description = format!(
        "appsrc name=source is-live=true format=time caps=video/x-raw,format={format},width={W},height={H},framerate=30/1 ! appsink name=sink sync=false"
    );
    let pair = super::super::super::capture::CapturePair::start(
        [&description, &description],
        move |sample| extract_sample(sample, W, H, extract).map(Arc::new),
        policy,
    )
    .unwrap();
    let inputs = pair.test_app_sources();
    (
        HostCapture {
            pair,
            info: SourceInfo {
                camera_sizes: vec![(W, H); 2],
                fps: 30.0,
                fps_rational: Some((30, 1)),
                total_frames: None,
            },
            interrupted: Arc::new(AtomicBool::new(false)),
        },
        inputs,
    )
}

struct ScriptedCapture {
    source: Box<dyn FrameSource>,
    inputs: [AppSrc; 2],
    index: usize,
}

impl ScriptedCapture {
    fn new(nv12: bool) -> Self {
        let (source, inputs): (Box<dyn FrameSource>, _) = if nv12 {
            let (capture, inputs) = capture("NV12", extract_nv12, None);
            (Box::new(GstreamerNv12CameraSource { capture }), inputs)
        } else {
            let (capture, inputs) = capture("I420", extract_i420, None);
            (Box::new(GstreamerCameraSource { capture }), inputs)
        };
        Self {
            source,
            inputs,
            index: 0,
        }
    }
}

impl FrameSource for ScriptedCapture {
    fn info(&self) -> SourceInfo {
        self.source.info()
    }
    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        if self.index == TIMES.len() {
            for input in &self.inputs {
                input.end_of_stream().unwrap();
            }
        } else if self.index < TIMES.len() {
            // Deliver one pair per read so queue overflow cannot randomly remove
            // fixture images. Alternating send order must not swap camera identity.
            for camera in [self.index % 2, 1 - self.index % 2] {
                let mut bytes = vec![128; PIXELS * 3 / 2];
                bytes[..PIXELS].fill(MARKERS[self.index][camera]);
                let mut buffer = gst::Buffer::from_mut_slice(bytes);
                buffer
                    .get_mut()
                    .unwrap()
                    .set_pts(gst::ClockTime::from_useconds(TIMES[self.index]));
                self.inputs[camera].push_buffer(buffer).unwrap();
            }
        }
        let result = self.source.next_frame()?;
        if self.index < TIMES.len() {
            let frame = result.as_ref().expect("capture ended before its input");
            assert_eq!(frame.pts_us, Some((TIMES[self.index] - TIMES[0]) as i64));
            assert_eq!(
                frame.capture.as_ref().unwrap().times_us,
                [TIMES[self.index] as i64; 2]
            );
            let planes: Vec<&[u8]> = match &frame.images {
                FrameSet::Yuv420p(cameras) => cameras.iter().map(|f| f.y.as_slice()).collect(),
                FrameSet::Nv12(cameras) => cameras.iter().map(|f| f.y.as_slice()).collect(),
                _ => panic!("expected host capture"),
            };
            assert_eq!(planes.len(), 2);
            for (camera, plane) in planes.iter().enumerate() {
                assert!(plane.iter().all(|&y| y == MARKERS[self.index][camera]));
            }
        } else {
            assert!(result.is_none());
            assert!(self.source.is_exhausted());
        }
        self.index += 1;
        Ok(result)
    }
}

type Measurements = Arc<Mutex<Vec<(CameraIndex, u8)>>>;
struct RecordingDetector(Measurements);

impl Detector for RecordingDetector {
    fn name(&self) -> &'static str {
        "image-identity"
    }
    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let DetectorFrame::Yuv(image) = frame else {
            panic!("expected host detector input");
        };
        self.0.lock().unwrap().push((camera, image.y[0]));
        Ok(Vec::new())
    }
}

fn session() -> StitchSession {
    let lens = Lens::fisheye(W, H, 32.0, 32.0, 32.0, 16.0, [0.0; 4]);
    let calibration = Calibration::new(
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
    );
    let executor = {
        let source_sizes = vec![(W, H); calibration.projection.camera_count()];
        CpuExecutor::new(
            calibration,
            ViewportSize {
                width: W,
                height: H,
            },
            source_sizes,
            false,
        )
    }
    .unwrap();
    let mut session = StitchSession::with_executor(Executor::Cpu(Box::new(executor))).unwrap();
    session.core_mut().set_constrained_look(false);
    session
}

fn verify_image(
    index: usize,
    pts: Option<i64>,
    y: &[u8],
    time_tolerance: u64,
) -> Result<(), String> {
    if index >= TIMES.len()
        || pts.is_none_or(|pts| pts.abs_diff((TIMES[index] - TIMES[0]) as i64) > time_tolerance)
    {
        return Err(format!("unexpected output {index} at {pts:?}"));
    }
    // The symmetric rig has a hard seam at centre. Sample inside each half,
    // away from the seam and image edges; tolerate only encoding/rounding.
    for (camera, x) in [W / 4, W * 3 / 4].into_iter().enumerate() {
        let actual = y[(H / 2 * W + x) as usize];
        if actual.abs_diff(MARKERS[index][camera]) > 3 {
            return Err(format!(
                "image {index}, camera {camera}: luma {actual}, expected {}",
                MARKERS[index][camera]
            ));
        }
    }
    Ok(())
}

struct CheckedEncoder {
    encoder: FfmpegFileEncoder,
    index: usize,
}

impl OutputSink for CheckedEncoder {
    fn name(&self) -> &str {
        "checked-encoder"
    }
    fn wants(&self) -> SinkInput {
        self.encoder.wants()
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        verify_image(self.index, frame.pts_us, frame.data, 0).map_err(|reason| {
            SinkError::Consume {
                frame_index: Some(self.index as u64),
                reason: format!("before encoding: {reason}"),
            }
        })?;
        self.encoder.consume(frame)?;
        self.index += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        self.encoder.finish()
    }
}

fn check_encoded(path: &Path) {
    let mut decoder = VideoDecoder::open_software(path).unwrap();
    for index in 0..TIMES.len() {
        let frame = decoder
            .next_frame()
            .unwrap()
            .expect("missing encoded image");
        // The decoder converts the container time base back to integer microseconds.
        verify_image(index, Some(frame.timestamp_us), &frame.y, 1).unwrap();
    }
    assert!(decoder.next_frame().unwrap().is_none());
}

#[test]
fn image_oracle_rejects_swapped_stale_black_and_retimed_images() {
    let image = |markers: [u8; 2]| {
        let mut y = vec![0; PIXELS];
        for row in y.chunks_exact_mut(W as usize) {
            row[..W as usize / 2].fill(markers[0]);
            row[W as usize / 2..].fill(markers[1]);
        }
        y
    };
    let pts = Some((TIMES[1] - TIMES[0]) as i64);
    assert!(verify_image(1, pts, &image(MARKERS[1]), 0).is_ok());
    for wrong in [
        [MARKERS[1][1], MARKERS[1][0]],
        [MARKERS[0][0], MARKERS[1][1]],
        [MARKERS[1][0], MARKERS[0][1]],
        [16, 16],
    ] {
        assert!(verify_image(1, pts, &image(wrong), 1).is_err());
    }
    assert!(verify_image(1, Some(0), &image(MARKERS[1]), 1).is_err());
    assert!(verify_image(1, None, &image(MARKERS[1]), 1).is_err());
    assert!(verify_image(TIMES.len(), pts, &image(MARKERS[1]), 1).is_err());
}

#[test]
fn capture_identity_survives_detection_lookahead_and_encoded_file() {
    for nv12 in [false, true] {
        for (depth, interval) in [(0, 1), (0, 2), (2, 2), (8, 1)] {
            let mut source = ScriptedCapture::new(nv12);
            let mut session = session();
            let measurements = Measurements::default();
            session.set_detector(Box::new(RecordingDetector(measurements.clone())));
            session.set_detection_interval(interval);
            session.set_lookahead(depth, usize::MAX);
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("capture.mp4");
            let encoder = FfmpegFileEncoder::new(
                &path,
                W,
                H,
                (30, 1),
                &EncoderConfig {
                    encoder_name: Some("libx264".into()),
                    // Keep compression noise smaller than the image-identity tolerance.
                    quality: crate::output::Quality::Value(100),
                    ..Default::default()
                },
            )
            .unwrap();
            session
                .add_sink(
                    Box::new(CheckedEncoder { encoder, index: 0 }),
                    SinkOptions::threaded(1),
                )
                .unwrap();
            let result = session
                .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
                .unwrap();
            assert_eq!(result.frames_processed, TIMES.len() as u64);
            session.finish().unwrap();
            session.finish().unwrap();
            drop(session);
            assert!(source.next_frame().unwrap().is_none());
            let expected: Vec<_> = MARKERS
                .iter()
                .step_by(interval as usize)
                .flat_map(|markers| markers.iter().enumerate().map(|(camera, &y)| (camera, y)))
                .collect();
            assert_eq!(
                *measurements.lock().unwrap(),
                expected,
                "NV12={nv12}, depth={depth}, interval={interval}"
            );
            check_encoded(&path);
        }
    }
}

struct DroppedCapture {
    source: Box<dyn FrameSource>,
    inputs: [AppSrc; 2],
    start: i64,
    index: usize,
    missing: usize,
}

impl FrameSource for DroppedCapture {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            fps: 10.0,
            fps_rational: Some((10, 1)),
            ..self.source.info()
        }
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        use gstreamer::prelude::*;
        if self.index == 3 {
            for input in &self.inputs {
                input.end_of_stream().unwrap();
            }
            return self.source.next_frame();
        }
        let time = self.start + self.index as i64 * 100_000;
        loop {
            let now = self.inputs[0].current_running_time().unwrap().useconds() as i64;
            if now >= time {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros((time - now) as u64));
        }
        for (camera, &marker) in MARKERS[self.index].iter().enumerate() {
            if self.index == 1 && camera == self.missing {
                continue;
            }
            let mut bytes = vec![128; PIXELS * 3 / 2];
            bytes[..PIXELS].fill(marker);
            let mut buffer = gst::Buffer::from_mut_slice(bytes);
            buffer
                .get_mut()
                .unwrap()
                .set_pts(gst::ClockTime::from_useconds(time as u64));
            self.inputs[camera].push_buffer(buffer).unwrap();
        }
        let frame = self.source.next_frame()?.unwrap();
        assert_eq!(frame.pts_us, Some(self.index as i64 * 100_000));
        let timing = frame.capture.as_ref().unwrap();
        assert_eq!(timing.at_us, time);
        assert_eq!(timing.max_age_us, Some(250_000));
        for camera in 0..2 {
            assert_eq!(
                timing.times_us[camera],
                if self.index == 1 && camera == self.missing {
                    self.start
                } else {
                    time
                }
            );
        }
        self.index += 1;
        Ok(Some(frame))
    }
}

#[test]
fn bounded_live_reuse_survives_inference_lookahead_and_encoding() {
    use gstreamer::prelude::*;
    for nv12 in [false, true] {
        for depth in [0, 4] {
            for missing in 0..2 {
                let policy = Some((
                    crate::frame_sync::Config {
                        wait_us: 80_000,
                        skew_us: 20_000,
                        max_age_us: 250_000,
                        capacity: 2,
                    },
                    10,
                ));
                let (source, inputs): (Box<dyn FrameSource>, _) = if nv12 {
                    let (capture, inputs) = capture("NV12", extract_nv12, policy);
                    (Box::new(GstreamerNv12CameraSource { capture }), inputs)
                } else {
                    let (capture, inputs) = capture("I420", extract_i420, policy);
                    (Box::new(GstreamerCameraSource { capture }), inputs)
                };
                let mut session = session();
                let measurements = Measurements::default();
                session.set_detector(Box::new(RecordingDetector(measurements.clone())));
                session.set_lookahead(depth, usize::MAX);
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("reuse.mp4");
                let encoder = FfmpegFileEncoder::new(
                    &path,
                    W,
                    H,
                    (10, 1),
                    &EncoderConfig {
                        encoder_name: Some("libx264".into()),
                        quality: crate::output::Quality::Value(100),
                        ..Default::default()
                    },
                )
                .unwrap();
                session
                    .add_sink(Box::new(encoder), SinkOptions::threaded(1))
                    .unwrap();
                let start = inputs[0].current_running_time().unwrap().useconds() as i64;
                let mut source = DroppedCapture {
                    source,
                    inputs,
                    start,
                    index: 0,
                    missing,
                };
                let result = session
                    .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
                    .unwrap();
                assert_eq!(result.frames_processed, 3);
                session.finish().unwrap();
                let expected: Vec<_> = MARKERS[..3]
                    .iter()
                    .enumerate()
                    .flat_map(|(index, markers)| {
                        markers.iter().enumerate().filter_map(move |(camera, &y)| {
                            (!(index == 1 && camera == missing)).then_some((camera, y))
                        })
                    })
                    .collect();
                assert_eq!(*measurements.lock().unwrap(), expected);
                let mut decoder = VideoDecoder::open_software(&path).unwrap();
                for index in 0..3 {
                    let frame = decoder.next_frame().unwrap().unwrap();
                    assert!(frame.timestamp_us.abs_diff(index as i64 * 100_000) <= 1);
                    for (camera, x) in [W / 4, W * 3 / 4].into_iter().enumerate() {
                        let marker = MARKERS[if index == 1 && camera == missing {
                            0
                        } else {
                            index
                        }][camera];
                        assert!(
                            frame.y[(H / 2 * W + x) as usize].abs_diff(marker) <= 3,
                            "nv12={nv12}, depth={depth}, missing={missing}, output={index}, camera={camera}"
                        );
                    }
                }
                assert!(decoder.next_frame().unwrap().is_none());
            }
        }
    }
}
