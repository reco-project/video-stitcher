//! Finite-input integrity across extraction and worker boundaries.

#![cfg(feature = "ffmpeg")]

use ffmpeg_next as ffmpeg;
use reco_core::source::{FrameSet, FrameSource};
use reco_io::InputPath;
use reco_io::adapters::FfmpegFileSource;
use reco_io::ffmpeg::decoder::{DecodeBackend, DecodeError, VideoDecoder};

const FRAMES: usize = 48;
const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;

#[test]
fn recording_rotation_changes_follow_the_camera_baseline_not_the_other_camera() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.mp4");
    fixture(&first);
    for (angle, expected) in [(90, 270), (180, 180), (270, 90), (360, 0), (-180, 180)] {
        let later = dir.path().join(format!("rotation-{angle}.mp4"));
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-nostdin",
                "-display_rotation",
                &angle.to_string(),
                "-i",
            ])
            .arg(&first)
            .args(["-map", "0:v:0", "-c", "copy"])
            .arg(&later)
            .output()
            .expect("ffmpeg fixture remuxer");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            VideoDecoder::open_software(&later).unwrap().rotation(),
            expected
        );
        let reports = reco_io::adapters::inspect_recordings(&[InputPath::Chained(vec![
            first.clone(),
            later.clone(),
        ])]);
        assert_eq!(reports[0].different, [false, expected != 0]);
        assert_eq!(reports[0].recording_warnings.len(), 2);
        assert!(reports[0].recording_warnings[0].is_empty());
        assert_eq!(reports[0].recording_warnings[1].is_empty(), expected == 0);
        if expected == 0 {
            assert!(reports[0].warnings.is_empty());
        } else {
            assert_eq!(
                reports[0].warnings,
                [format!(
                    "rotation-{angle}.mp4: rotation {expected}° (expected 0°)."
                )]
            );
            let reordered = reco_io::adapters::inspect_recordings(&[InputPath::Chained(vec![
                later.clone(),
                first.clone(),
            ])]);
            assert_eq!(reordered[0].different, [false, true]);
            assert!(reordered[0].recording_warnings[0].is_empty());
            assert_eq!(reordered[0].recording_warnings[1], reordered[0].warnings[0]);
            assert_eq!(
                reordered[0].warnings,
                [format!("first.mp4: rotation 0° (expected {expected}°).")]
            );
        }
        let cameras = reco_io::adapters::inspect_recordings(&[
            InputPath::Single(first.clone()),
            InputPath::Single(later),
        ]);
        assert!(
            cameras
                .iter()
                .all(|report| report.warnings.is_empty() && report.different == [false])
        );
    }
}

#[test]
fn chained_recordings_warn_on_later_settings_before_playback_reaches_them() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.mp4");
    let resized = dir.path().join("resized.mp4");
    let faster = dir.path().join("faster.mp4");
    fixture_size(&first, "libx264", 128, 96);
    fixture_size(&resized, "libx264", 256, 192);
    fixture_rate(&faster, "libx264", 128, 96, (60, 1));
    for later in [&resized, &faster] {
        let warnings = reco_io::adapters::inspect_recordings(&[InputPath::Chained(vec![
            first.clone(),
            later.clone(),
        ])]);
        assert_eq!(warnings[0].warnings.len(), 1);
        assert_eq!(warnings[0].different, vec![false, true]);
        assert!(warnings[0].recording_warnings[0].is_empty());
        assert_eq!(warnings[0].recording_warnings[1], warnings[0].warnings[0]);
        assert_eq!(warnings[0].first_size, Some((128, 96)));
        assert_eq!(warnings[0].first_rate, Some((30, 1)));
        assert!(warnings[0].warnings[0].contains(later.file_name().unwrap().to_str().unwrap()));
        let reordered = reco_io::adapters::inspect_recordings(&[InputPath::Chained(vec![
            later.clone(),
            first.clone(),
        ])]);
        assert_eq!(
            reordered[0].first_size,
            Some(if later == &resized {
                (256, 192)
            } else {
                (128, 96)
            })
        );
        assert_eq!(
            reordered[0].first_rate,
            Some(if later == &faster { (60, 1) } else { (30, 1) })
        );
        assert_eq!(reordered[0].different, vec![false, true]);
        assert!(reordered[0].warnings[0].contains("first.mp4"));
    }
    let matching = reco_io::adapters::inspect_recordings(&[InputPath::Chained(vec![
        first.clone(),
        first.clone(),
    ])]);
    assert!(matching[0].warnings.is_empty());
    assert_eq!(matching[0].different, vec![false, false]);
    let differences = reco_io::adapters::inspect_recordings(&[
        InputPath::Single(first.clone()),
        InputPath::Single(resized),
    ]);
    assert_eq!(differences[0].different, vec![false]);
    assert_eq!(differences[1].different, vec![true]);
    assert!(differences[0].recording_warnings[0].is_empty());
    assert!(differences[1].recording_warnings[0].contains("resolutions differ"));
    assert!(
        differences.iter().all(|report| report.warnings.len() == 1
            && report.warnings[0].contains("resolutions differ"))
    );
    let differences = reco_io::adapters::inspect_recordings(&[
        InputPath::Single(first.clone()),
        InputPath::Single(faster),
    ]);
    assert!(
        differences.iter().all(|report| report.warnings.len() == 1
            && report.warnings[0].contains("without resampling"))
    );
    let unreadable = reco_io::adapters::inspect_recordings(&[InputPath::Chained(vec![
        dir.path().join("missing.mp4"),
        first,
    ])]);
    assert!(unreadable[0].warnings[0].contains("could not be checked"));
    assert_eq!(
        unreadable[0].recording_warnings[0],
        unreadable[0].warnings[0]
    );
    assert!(unreadable[0].recording_warnings[1].is_empty());
}

#[test]
fn independent_input_dimensions_survive_decode_and_cpu_render() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::geometry::RenderPose;
    use reco_core::projection::LShape;
    use reco_core::render::viewport::ViewportSize;
    use reco_core::stitch::CpuExecutor;
    let dir = tempfile::tempdir().unwrap();
    let sizes = [(128, 96), (256, 192), (192, 64)];
    let paths = sizes.map(|(w, h)| {
        let path = dir.path().join(format!("{w}x{h}.mp4"));
        fixture_size(&path, "libx264", w, h);
        path
    });
    for order in [[0, 1], [1, 0], [0, 2], [2, 0]] {
        let inputs = order.map(|i| InputPath::Single(paths[i].clone()));
        let expected = order.map(|i| sizes[i]);
        let mut source = FfmpegFileSource::open_inputs(&inputs, 0, true).unwrap();
        assert_eq!(source.info().camera_sizes, expected);
        let cal = Calibration::new(
            vec![Lens::flat(128, 96); 2],
            LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let cpu = CpuExecutor::new(
            cal,
            ViewportSize {
                width: 64,
                height: 36,
            },
            expected.to_vec(),
            false,
        )
        .unwrap();
        for _ in 0..FRAMES {
            let FrameSet::Yuv420p(images) = source.next_frame().unwrap().unwrap().images else {
                panic!("expected host frames")
            };
            assert_eq!(images.iter().map(|i| i.size).collect::<Vec<_>>(), expected);
            let rendered = cpu
                .stitch_yuv(
                    &images.iter().map(|i| i.as_planes()).collect::<Vec<_>>(),
                    RenderPose::default(),
                )
                .unwrap();
            assert_eq!(rendered.len(), 64 * 36 * 4);
        }
        assert!(source.next_frame().unwrap().is_none());
    }
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan external-memory interop"]
fn mixed_native_inputs_preserve_pixels_detection_order_and_lookahead() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
    use reco_core::projection::LShape;
    use reco_core::render::{renderer::InputFormat, viewport::ViewportSize};
    use reco_core::session::{SinkOptions, StitchSession};
    use reco_core::sink::{OutputFrame, OutputSink, SinkError, SinkInput};
    use reco_core::stitch::{Executor, GpuExecutor, GpuExecutorConfig};
    use std::sync::{Arc, Mutex, atomic::AtomicBool};

    struct Capture(Arc<Mutex<Vec<Vec<u8>>>>);
    impl OutputSink for Capture {
        fn name(&self) -> &str {
            "capture"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(reco_core::sink::PixelFormat::Nv12)
        }
        fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
            self.0.lock().unwrap().push(frame.data.to_vec());
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            Ok(())
        }
    }
    struct CheckInputs {
        sizes: Vec<(u32, u32)>,
        counts: Arc<Mutex<[usize; 2]>>,
    }
    impl Detector for CheckInputs {
        fn name(&self) -> &'static str {
            "native-pixel-oracle"
        }
        fn prefers_cuda_nv12(&self) -> bool {
            true
        }
        fn detect(
            &mut self,
            camera: usize,
            frame: &DetectorFrame<'_>,
        ) -> Result<Vec<Detection>, DetectorError> {
            let DetectorFrame::CudaNv12(frame) = frame else {
                panic!("native delivery required")
            };
            assert_eq!((frame.width, frame.height), self.sizes[camera]);
            let mut counts = self.counts.lock().unwrap();
            let mut sample = [0u8; 1];
            let offset = frame.y_pitch * (frame.height as usize - 1) + frame.width as usize - 1;
            // SAFETY: this is one byte inside the live decoded Y allocation.
            unsafe {
                reco_core::interop::cuda::cuda_memcpy_dtoh(
                    sample.as_mut_ptr().cast(),
                    frame.y_ptr + offset as u64,
                    1,
                )
                .unwrap();
            }
            assert!(sample[0].abs_diff(32 + counts[camera] as u8 * 4) <= 2);
            counts[camera] += 1;
            Ok(Vec::new())
        }
    }
    let gpu = reco_core::gpu::GpuContext::new_blocking().unwrap();
    assert!(
        gpu.supports_zero_copy(),
        "NVIDIA renderer required: {}",
        gpu.gpu_name()
    );
    let dir = tempfile::tempdir().unwrap();
    let sizes = [(128, 96), (256, 192), (192, 64)];
    let paths = sizes.map(|(w, h)| {
        let path = dir.path().join(format!("{w}x{h}.mp4"));
        fixture_size(&path, "libx264", w, h);
        path
    });
    for order in [[0, 1], [1, 0], [0, 2], [2, 0]] {
        let inputs = order.map(|i| InputPath::Single(paths[i].clone()));
        let source_sizes: Vec<_> = order.iter().map(|&i| sizes[i]).collect();
        let run = |native: bool, depth: usize| {
            let mut source = if native {
                reco_io::SmartFileSource::open(&inputs, &gpu, 0).unwrap()
            } else {
                reco_io::SmartFileSource::open_cpu_only(&inputs, 0, true).unwrap()
            };
            assert_eq!(source.is_gpu_resident(), native);
            assert_eq!(source.info().camera_sizes, source_sizes);
            let cal = Calibration::new(
                vec![Lens::flat(128, 96); 2],
                LShape {
                    intersect: 0.5,
                    x_ty: 0.0,
                    x_rz: 0.0,
                    z_rx: 0.0,
                    x_rx: 0.0,
                    z_rz: 0.0,
                    blend_width: 0.05,
                },
                Framing {
                    axis_offset: 0.25,
                    tilt: 0.0,
                    roll: 0.0,
                },
            );
            let mut config = GpuExecutorConfig::new(
                cal,
                source_sizes.clone(),
                if native {
                    InputFormat::Nv12
                } else {
                    InputFormat::Yuv420p
                },
            );
            config.viewport_size = ViewportSize {
                width: 64,
                height: 36,
            };
            let executor = GpuExecutor::new(gpu.clone(), config).unwrap();
            let mut session =
                StitchSession::with_executor(Executor::Gpu(Box::new(executor))).unwrap();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, usize::MAX);
            let counts = Arc::new(Mutex::new([0; 2]));
            if native {
                session
                    .setup_gpu_source(source.shared_texture_set().unwrap())
                    .unwrap();
                session.set_detector(Box::new(CheckInputs {
                    sizes: source_sizes.clone(),
                    counts: counts.clone(),
                }));
                session.set_detection_interval(1);
            }
            let output = Arc::new(Mutex::new(Vec::new()));
            session
                .add_sink(
                    Box::new(Capture(output.clone())),
                    SinkOptions::inline_lossy(),
                )
                .unwrap();
            assert_eq!(
                session
                    .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
                    .unwrap()
                    .frames_processed,
                FRAMES as u64
            );
            session.finish().unwrap();
            if native {
                assert_eq!(*counts.lock().unwrap(), [FRAMES; 2]);
            }
            let frames = output.lock().unwrap().clone();
            assert_eq!(frames.len(), FRAMES);
            frames
        };
        let reference = run(false, 0);
        for depth in [0, 3] {
            let actual = run(true, depth);
            for (expected, actual) in reference.iter().zip(actual) {
                assert_eq!(expected.len(), actual.len());
                assert!(
                    expected
                        .iter()
                        .zip(actual)
                        .all(|(&a, b)| a.abs_diff(b) <= 2)
                );
            }
        }
    }
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVDEC and a non-matching Vulkan adapter"]
fn cuda_decode_uses_host_transfer_for_an_incompatible_renderer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    assert_eq!(
        VideoDecoder::open(&path).unwrap().backend(),
        DecodeBackend::Cuda
    );
    let gpu = reco_core::gpu::GpuContext::new_blocking().unwrap();
    assert!(!gpu.supports_zero_copy());
    let input = InputPath::Single(path);
    for count in [1, 2] {
        let mut source =
            reco_io::SmartFileSource::open(&vec![input.clone(); count], &gpu, 0).unwrap();
        assert!(!source.is_gpu_resident());
        for index in 0..FRAMES {
            let FrameSet::Yuv420p(cameras) = source.next_frame().unwrap().unwrap().images else {
                panic!("expected host-backed frames");
            };
            assert_eq!(cameras.len(), count);
            for frame in cameras {
                assert_pixels(&frame.y, &frame.u, &frame.v, index);
            }
        }
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.is_exhausted());
    }
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan external-memory interop"]
fn cuda_mono_session_preserves_pixels_order_and_buffered_tail() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::Cylinder;
    use reco_core::render::renderer::InputFormat;
    use reco_core::session::{SinkOptions, StitchSession};
    use reco_core::sink::{OutputFrame, OutputSink, SinkError, SinkInput};
    use reco_core::stitch::{Executor, GpuExecutor, GpuExecutorConfig};
    use std::sync::{Arc, Mutex, atomic::AtomicBool};

    struct Capture(Arc<Mutex<Vec<Vec<u8>>>>);
    impl OutputSink for Capture {
        fn name(&self) -> &str {
            "capture"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(reco_core::sink::PixelFormat::Nv12)
        }
        fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
            self.0.lock().unwrap().push(frame.data.to_vec());
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            Ok(())
        }
    }
    for (width, height) in [(WIDTH, HEIGHT), (7500, 2500)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reordered.mp4");
        fixture_size(&path, "libx264", width, height);
        let input = InputPath::Single(path);
        let run = |native: bool, depth: usize, limit: u64, budget: usize| {
            let gpu = reco_core::gpu::GpuContext::new_blocking().unwrap();
            let mut source = if native {
                reco_io::SmartFileSource::open(std::slice::from_ref(&input), &gpu, 0).unwrap()
            } else {
                reco_io::SmartFileSource::open_cpu_only(std::slice::from_ref(&input), 0, true)
                    .unwrap()
            };
            assert_eq!(
                source.is_gpu_resident(),
                native,
                "adapter={}, zero_copy={}, mode={}, size={width}x{height}",
                gpu.gpu_name(),
                gpu.supports_zero_copy(),
                source.decode_mode()
            );
            let cal = Calibration::new(
                vec![Lens::flat(width, height)],
                Cylinder {
                    radius: f64::from(width),
                    ..Default::default()
                },
                Framing {
                    axis_offset: 0.0,
                    tilt: 0.0,
                    roll: 0.0,
                },
            );
            let mut config = {
                let source_sizes = vec![(width, height); cal.projection.camera_count()];
                GpuExecutorConfig::new(
                    cal,
                    source_sizes,
                    if native {
                        InputFormat::Nv12
                    } else {
                        InputFormat::Yuv420p
                    },
                )
            };
            config.viewport_size = reco_core::render::viewport::ViewportSize {
                width: 64,
                height: 48,
            };
            let executor = GpuExecutor::new(gpu, config).unwrap();
            let mut session =
                StitchSession::with_executor(Executor::Gpu(Box::new(executor))).unwrap();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, budget);
            if let Some(shared) = source.shared_texture_set() {
                session.setup_gpu_source(shared).unwrap();
            }
            let frames = Arc::new(Mutex::new(Vec::new()));
            session
                .add_sink(
                    Box::new(Capture(frames.clone())),
                    SinkOptions::inline_lossy(),
                )
                .unwrap();
            let result = session
                .run(&mut source, limit, &AtomicBool::new(false), None)
                .map_err(|error| error.to_string())?;
            let reuse = native && depth <= 5 && limit == u64::MAX && budget == usize::MAX;
            if reuse {
                struct HostFrame {
                    info: reco_core::source::SourceInfo,
                    frame: Option<FrameSet>,
                }
                impl FrameSource for HostFrame {
                    fn info(&self) -> reco_core::source::SourceInfo {
                        self.info.clone()
                    }
                    fn next_frame(
                        &mut self,
                    ) -> Result<
                        Option<reco_core::source::SourceFrame>,
                        reco_core::source::SourceError,
                    > {
                        Ok(self
                            .frame
                            .take()
                            .map(reco_core::source::SourceFrame::untimed))
                    }
                }
                // Switch ownership after real decoded frames have been staged
                // and rendered. The one-frame host allowance is smaller than
                // the native pool; rendering must sample the new upload.
                let image = reco_core::source::Nv12Data {
                    size: (width, height),
                    y: vec![128; width as usize * height as usize],
                    uv: vec![128; width as usize * height as usize / 2],
                };
                session.set_lookahead(5, image.y.capacity() + image.uv.capacity());
                let mut host = HostFrame {
                    info: source.info(),
                    frame: Some(FrameSet::Nv12(vec![image.into()])),
                };
                assert_eq!(
                    session
                        .run(&mut host, u64::MAX, &AtomicBool::new(false), None)
                        .unwrap()
                        .frames_processed,
                    result.frames_processed + 1
                );
            }
            session.finish().unwrap();
            let mut frames = frames.lock().unwrap().clone();
            assert_eq!(
                result.frames_processed as usize + usize::from(reuse),
                frames.len()
            );
            if reuse {
                let center = frames.last().unwrap()[24 * 64 + 32];
                assert!(
                    center.abs_diff(128) <= 2,
                    "host frame center={center}, input={width}x{height}"
                );
                frames.pop();
            }
            Ok::<_, String>(frames)
        };
        let reference = run(false, 0, u64::MAX, usize::MAX).unwrap();
        assert_eq!(reference.len(), FRAMES);
        assert!(reference.windows(2).all(|pair| pair[0] != pair[1]));
        for depth in [0, 5, FRAMES + 3] {
            for limit in [1, 2, 7, u64::MAX] {
                let actual = run(true, depth, limit, usize::MAX).unwrap();
                assert_eq!(actual.len(), FRAMES.min(limit as usize));
                for (index, (expected, actual)) in reference.iter().zip(&actual).enumerate() {
                    assert_eq!(expected.len(), actual.len());
                    let max_error = expected
                        .iter()
                        .zip(actual)
                        .map(|(&a, &b)| a.abs_diff(b))
                        .max()
                        .unwrap();
                    assert!(
                        max_error <= 2,
                        "{width}x{height} frame {index}, depth {depth}, limit {limit}: error {max_error}"
                    );
                }
            }
        }
        // Depth five retains eleven native staging slots (5 + 2 + 4).
        let exact_budget = width as usize * height as usize * 3 / 2 * 11;
        assert_eq!(run(true, 5, u64::MAX, exact_budget).unwrap().len(), FRAMES);
        let error = run(true, 5, u64::MAX, exact_budget - 1).unwrap_err();
        assert!(error.contains("image storage"), "{error}");
    }
}

fn fixture(path: &std::path::Path) {
    fixture_encoding(path, "libx264");
}

#[test]
fn file_counts_follow_chains_and_alignment_without_limiting_decode() {
    assert_file_counts(false);
}

#[test]
fn unusable_rate_is_unknown_not_thirty_fps() {
    assert_source_rates(false);
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan external-memory interop"]
fn native_sources_preserve_unknown_rates_and_reject_mismatches() {
    assert_source_rates(true);
}

fn assert_source_rates(native: bool) {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::{Cylinder, LShape};
    use std::sync::{Arc, atomic::AtomicBool};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unusable-rate.mp4");
    fixture_rate(&path, "libx264", WIDTH, HEIGHT, (2000, 1));
    let normal = dir.path().join("normal.mp4");
    fixture(&normal);
    #[cfg(feature = "gpu")]
    let gpu = native.then(|| reco_core::gpu::GpuContext::new_blocking().unwrap());
    let open = |inputs: &[InputPath]| {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &gpu {
            return reco_io::SmartFileSource::open(inputs, gpu, 0);
        }
        reco_io::SmartFileSource::open_cpu_only(inputs, 0, true)
    };
    for paths in [
        vec![path.clone()],
        vec![path.clone(), normal.clone()],
        vec![normal.clone(), path.clone()],
    ] {
        let inputs: Vec<_> = paths.into_iter().map(InputPath::Single).collect();
        let mut source = open(&inputs).unwrap();
        assert_eq!(source.is_gpu_resident(), native);
        assert_eq!(source.info().fps_rational, None);
        assert_eq!(source.info().fps, 0.0);
        if !native {
            let error = source.skip_frames(1).unwrap_err();
            assert!(
                error.to_string().contains("frame rate is unknown"),
                "{error}"
            );
        }
        source.start_decoding();
        assert!(
            source.next_frame().unwrap().is_some(),
            "missing timing must not prevent raw decoding"
        );
        drop(source);

        let lenses = vec![Lens::flat(WIDTH, HEIGHT); inputs.len()];
        let framing = Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        };
        let cal = if inputs.len() == 1 {
            Calibration::new(lenses, Cylinder::default(), framing)
        } else {
            Calibration::new(
                lenses,
                LShape {
                    intersect: 0.5,
                    x_ty: 0.0,
                    x_rz: 0.0,
                    z_rx: 0.0,
                    x_rx: 0.0,
                    z_rz: 0.0,
                    blend_width: 0.05,
                },
                framing,
            )
        };
        let output = dir.path().join("output.mp4");
        let job = || {
            let job = reco_io::StitchJob::with_calibration(inputs.clone(), cal.clone(), &output)
                .on_session(|_, _| panic!("invalid timing must fail before session setup"));
            if native { job } else { job.cpu() }
        };
        for existing in [false, true] {
            if existing {
                std::fs::write(&output, b"existing output").unwrap();
            }
            let error = job().run(&Arc::new(AtomicBool::new(false))).unwrap_err();
            assert!(
                error.to_string().contains("frame rate is unknown"),
                "{error}"
            );
            if existing {
                assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
                std::fs::remove_file(&output).unwrap();
            } else {
                assert!(!output.exists());
            }
        }
    }
    assert!(matches!(
        reco_io::ffmpeg::calibration_io::probe_video(&path),
        Err(reco_io::ffmpeg::calibration_io::CalibrationIoError::UnknownFrameRate)
    ));
    assert!(matches!(
        reco_io::ffmpeg::calibration_io::extract_frames(&path, &[0]),
        Err(reco_io::ffmpeg::calibration_io::CalibrationIoError::UnknownFrameRate)
    ));
    {
        let layout = reco_io::stacked_video::GridLayout::vstack(WIDTH, HEIGHT / 2, 2).unwrap();
        let mut source =
            reco_io::stacked_video::source::StackedSource::open(layout, &path).unwrap();
        assert_eq!(source.info().fps_rational, None);
        assert_eq!(source.info().fps, 0.0);
        assert!(source.next_tuple().unwrap().is_some());
    }

    let different = dir.path().join("different.mp4");
    fixture_rate(&different, "libx264", WIDTH, HEIGHT, (60, 1));
    for paths in [
        [normal.clone(), different.clone()],
        [different, normal.clone()],
    ] {
        let error = open(&paths.map(InputPath::Single))
            .err()
            .expect("mismatched input rates must fail at source setup");
        assert!(error.to_string().contains("frame rate mismatch"), "{error}");
    }
    let fractional = dir.path().join("fractional.mp4");
    fixture_rate(&fractional, "libx264", WIDTH, HEIGHT, (30000, 1001));
    let inputs = [InputPath::Single(fractional), InputPath::Single(normal)];
    let source = open(&inputs).unwrap();
    assert_eq!(source.info().fps_rational, Some((30000, 1001)));
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan external-memory interop"]
fn native_file_counts_match_host_sources() {
    assert_file_counts(true);
}

fn assert_file_counts(native: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("input.mp4");
    fixture(&path);
    let single = InputPath::Single(path.clone());
    let chain = InputPath::Chained(vec![path.clone(), path]);
    #[cfg(feature = "gpu")]
    let gpu = native.then(|| reco_core::gpu::GpuContext::new_blocking().unwrap());
    let open = |inputs: &[InputPath], offset| {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &gpu {
            return reco_io::SmartFileSource::open(inputs, gpu, offset);
        }
        reco_io::SmartFileSource::open_cpu_only(inputs, offset, true)
    };
    for (inputs, offset, expected) in [
        (vec![single.clone()], 7, FRAMES),
        (vec![chain.clone()], 0, 2 * FRAMES),
        (vec![single.clone(), chain.clone()], 7, FRAMES),
        (vec![single.clone(), chain.clone()], -7, FRAMES - 7),
        (vec![chain.clone(), single.clone()], 7, FRAMES - 7),
        (vec![chain, single.clone()], -7, FRAMES),
        (vec![single.clone(), single.clone()], i64::MIN, 0),
    ] {
        let mut source = open(&inputs, offset).unwrap();
        assert_eq!(source.is_gpu_resident(), native);
        assert_eq!(source.info().total_frames, Some(expected as u64));
        assert_eq!(source.total_frames(), Some(expected as u64));
        source.start_decoding();
        let mut decoded = 0;
        while let Some(frame) = source.next_frame().unwrap() {
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            if let FrameSet::GpuResident { slots } = frame.images {
                for (camera, slot) in source
                    .shared_texture_set()
                    .unwrap()
                    .cameras
                    .iter()
                    .zip(slots)
                {
                    let _ = camera.slot_free_tx.send(slot);
                }
            }
            #[cfg(not(all(feature = "gpu", target_os = "linux")))]
            drop(frame);
            decoded += 1;
        }
        assert_eq!(decoded, expected);
    }

    let incomplete = InputPath::Chained(vec![
        single.first_path().to_path_buf(),
        dir.path().join("missing.mp4"),
    ]);
    for inputs in [vec![incomplete.clone()], vec![single.clone(), incomplete]] {
        let source = reco_io::SmartFileSource::open_cpu_only(&inputs, 0, true).unwrap();
        assert_eq!(source.info().total_frames, None);
        assert_eq!(source.total_frames(), None);
    }
    let source = reco_io::SmartFileSource::open_cpu_only(&vec![single; 3], 7, true).unwrap();
    assert_eq!(source.total_frames(), Some(FRAMES as u64));
}

#[test]
fn job_setup_and_completion_are_explicit() {
    use reco_core::session::types::RunCompletion;
    use reco_io::stitch_job::StitchError;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    fixture(&input);
    let output = dir.path().join("output.mp4");
    let job = || {
        let lens =
            reco_core::calibration::Lens::fisheye(WIDTH, HEIGHT, 64.0, 64.0, 64.0, 48.0, [0.0; 4]);
        let cal = reco_core::calibration::Calibration::new(
            vec![lens.clone(), lens],
            reco_core::projection::LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.05,
            },
            reco_core::calibration::Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        reco_io::StitchJob::with_calibration([input.clone(), input.clone()], cal, &output)
            .cpu()
            .resolution(64, 64)
            .audio(reco_io::output::AudioMode::Disabled)
            .encoder_name("libx264")
            .on_session(|session, _| {
                session.core_mut().set_constrained_look(false);
                Ok(())
            })
    };
    let error = job()
        .on_session(|_, _| Err(StitchError::Other("setup failure".into())))
        .on_session(|_, _| panic!("must not run after failure"))
        .run(&std::sync::Arc::new(AtomicBool::new(false)))
        .unwrap_err();
    assert_eq!(error.to_string(), "setup failure");
    assert!(!output.exists(), "failed setup must not create an output");

    let flag = Arc::new(AtomicBool::new(false));
    let result = job()
        .start_time(0.051)
        .end_time(0.149)
        .on_session(|_, source| {
            assert_eq!(source.info().total_frames, Some(FRAMES as u64));
            Ok(())
        })
        .run(&flag)
        .unwrap();
    assert_eq!(result.frames_processed, 3);
    for start in [2.0, 20.0] {
        let error = job().start_time(start).run(&flag).unwrap_err();
        assert!(
            error.to_string().contains("no video frames remain"),
            "{error}"
        );
    }

    for cancel_during_frames in [true, false] {
        let flag = Arc::new(AtomicBool::new(false));
        let progress_flag = flag.clone();
        let finish_flag = flag.clone();
        let result = job()
            .max_frames(6)
            .on_progress(move |p| {
                if cancel_during_frames && p.frames_completed == 3 {
                    progress_flag.store(true, Ordering::Relaxed);
                }
            })
            .on_finalizing(move || finish_flag.store(true, Ordering::Relaxed))
            .run(&flag)
            .unwrap();
        let expected = if cancel_during_frames { 3 } else { 6 };
        assert_eq!(result.frames_processed, expected);
        assert_eq!(
            result.completion,
            if cancel_during_frames {
                RunCompletion::Cancelled
            } else {
                RunCompletion::Completed
            }
        );
        let mut decoded = VideoDecoder::open_software(&output).unwrap();
        let mut count = 0;
        while decoded.next_frame().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(
            count, expected,
            "partial output must be finalized and decodable"
        );
    }

    struct FailingFinish;
    impl reco_core::sink::OutputSink for FailingFinish {
        fn name(&self) -> &str {
            "failing-finish"
        }
        fn wants(&self) -> reco_core::sink::SinkInput {
            reco_core::sink::SinkInput::CpuBytes(reco_core::sink::PixelFormat::Nv12)
        }
        fn consume(
            &mut self,
            _: reco_core::sink::OutputFrame<'_>,
        ) -> Result<(), reco_core::sink::SinkError> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), reco_core::sink::SinkError> {
            Err(reco_core::sink::SinkError::Finish {
                reason: "finalization failure".into(),
            })
        }
    }
    let flag = Arc::new(AtomicBool::new(false));
    let signal = flag.clone();
    let result = job()
        .on_session(|session, _| {
            session.add_sink(
                Box::new(FailingFinish),
                reco_core::session::SinkOptions::threaded(1),
            )?;
            Ok(())
        })
        .on_progress(move |_| signal.store(true, Ordering::Relaxed))
        .run(&flag);
    assert!(flag.load(Ordering::Relaxed));
    assert!(matches!(result, Err(StitchError::Session(_))), "{result:?}");
}

#[test]
fn host_file_job_preserves_pixels_timing_and_lookahead_tail() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::Cylinder;
    use std::sync::{Arc, atomic::AtomicBool};

    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    fixture(&input);
    let modes: &[bool] = if cfg!(feature = "gpu") {
        &[true]
    } else {
        &[true, false]
    };
    for &software in modes {
        let output = dir.path().join(format!("output-{software}.mp4"));
        let calibration = Calibration::new(
            vec![Lens::flat(WIDTH, HEIGHT)],
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
        let mut job = reco_io::StitchJob::with_calibration([input.clone()], calibration, &output)
            .resolution(64, 32)
            .audio(reco_io::output::AudioMode::Disabled)
            .encoder_name("libx264")
            .lookahead(0.1)
            .lookahead_memory_limit(1024 * 1024)
            .on_session(|session, source| {
                assert!(!source.is_gpu_resident());
                session.core_mut().set_constrained_look(false);
                Ok(())
            });
        if software {
            job = job.cpu();
        }
        let result = job.run(&Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!(result.frames_processed, FRAMES as u64);
        assert_eq!(result.gpu_name, "software (CPU)");
        let mut decoder = VideoDecoder::open_software(&output).unwrap();
        assert_eq!(decoder.frame_rate(), Some((30, 1).into()));
        for index in 0..FRAMES {
            let frame = decoder
                .next_frame()
                .unwrap()
                .expect("missing rendered frame");
            assert_eq!((frame.width, frame.height), (64, 32));
            assert!((frame.timestamp_us - index as i64 * 1_000_000 / 30).abs() <= 1);
            // These source colors remain inside RGB gamut during stitching.
            if (16..32).contains(&index) {
                for (plane, expected) in [
                    (&frame.y, 32 + index as u8 * 4),
                    (&frame.u, 96),
                    (&frame.v, 160),
                ] {
                    assert!(
                        plane.iter().all(|value| value.abs_diff(expected) <= 4),
                        "frame {index}: expected {expected}"
                    );
                }
            }
        }
        assert!(decoder.next_frame().unwrap().is_none());
    }
}

fn failed_job_finalizes_its_rendered_prefix(software: bool) {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
    use reco_core::projection::Cylinder;
    use reco_core::session::{SinkDelivery, SinkErrorPolicy, SinkOptions};
    use reco_core::sink::{OutputFrame, OutputSink, PixelFormat, SinkError, SinkInput};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    };

    struct FailAfter(usize);
    impl Detector for FailAfter {
        fn name(&self) -> &'static str {
            "injected-midstream-failure"
        }
        fn detect(
            &mut self,
            _: usize,
            _: &DetectorFrame<'_>,
        ) -> Result<Vec<Detection>, DetectorError> {
            if self.0 == 0 {
                return Err(DetectorError::UnsupportedFrameKind);
            }
            self.0 -= 1;
            Ok(vec![])
        }
    }
    struct Prefix {
        pts: Arc<Mutex<Vec<Option<i64>>>>,
        finishes: Arc<AtomicU64>,
        fail_finish: bool,
    }
    impl OutputSink for Prefix {
        fn name(&self) -> &'static str {
            "prefix"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(PixelFormat::Nv12)
        }
        fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
            self.pts.lock().unwrap().push(frame.pts_us);
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            self.finishes.fetch_add(1, Ordering::Relaxed);
            if self.fail_finish {
                return Err(SinkError::Finish {
                    reason: "secondary finish failure".into(),
                });
            }
            Ok(())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    fixture(&input);
    for lookahead in [0.0, 0.1] {
        for fail_finish in [false, true] {
            let output = dir
                .path()
                .join(format!("prefix-{lookahead}-{fail_finish}.mp4"));
            let calibration = Calibration::new(
                vec![Lens::flat(WIDTH, HEIGHT)],
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
            let processed = Arc::new(AtomicU64::new(0));
            let progress = processed.clone();
            let finalized = Arc::new(AtomicBool::new(false));
            let notify = finalized.clone();
            let pts = Arc::new(Mutex::new(vec![]));
            let finishes = Arc::new(AtomicU64::new(0));
            let observer = Prefix {
                pts: pts.clone(),
                finishes: finishes.clone(),
                fail_finish,
            };
            let mut job =
                reco_io::StitchJob::with_calibration([input.clone()], calibration, &output)
                    .resolution(64, 32)
                    .audio(reco_io::output::AudioMode::Disabled)
                    .encoder_name("libx264")
                    .lookahead(lookahead)
                    .on_session(move |session, _| {
                        session.core_mut().set_constrained_look(false);
                        session.set_detector(Box::new(FailAfter(12)));
                        session.add_sink(
                            Box::new(observer),
                            SinkOptions {
                                delivery: SinkDelivery::Inline,
                                on_error: SinkErrorPolicy::Abort,
                            },
                        )?;
                        Ok(())
                    })
                    .on_progress(move |frame| {
                        progress.store(frame.frames_completed, Ordering::Relaxed);
                    })
                    .on_finalizing(move || {
                        notify.store(true, Ordering::Relaxed);
                    });
            if software {
                job = job.cpu();
            }
            let error = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
            assert!(
                error.to_string().contains("injected-midstream-failure"),
                "{error}"
            );
            let expected = processed.load(Ordering::Relaxed) as usize;
            assert!(expected > 0);
            assert_eq!(
                pts.lock().unwrap().len(),
                expected,
                "submitted images were lost on failure"
            );
            assert_eq!(finishes.load(Ordering::Relaxed), 1);
            assert!(finalized.load(Ordering::Relaxed));
            let mut decoded = VideoDecoder::open_software(&output).unwrap();
            let mut count = 0;
            while let Some(frame) = decoded.next_frame().unwrap() {
                let expected_pts =
                    pts.lock().unwrap()[count].unwrap_or(count as i64 * 1_000_000 / 30);
                assert!(frame.timestamp_us.abs_diff(expected_pts) <= 1);
                count += 1;
            }
            assert_eq!(count, expected, "recording must retain the rendered prefix");
        }
    }
}

#[test]
fn failed_cpu_job_finalizes_its_rendered_prefix() {
    failed_job_finalizes_its_rendered_prefix(true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a Vulkan adapter and runtime encoding support"]
fn failed_gpu_job_finalizes_its_rendered_prefix() {
    failed_job_finalizes_its_rendered_prefix(false);
}

fn fixture_encoding(path: &std::path::Path, codec_name: &str) {
    fixture_size(path, codec_name, WIDTH, HEIGHT);
}

fn fixture_size(path: &std::path::Path, codec_name: &str, width: u32, height: u32) {
    fixture_rate(path, codec_name, width, height, (30, 1));
}

fn fixture_rate(
    path: &std::path::Path,
    codec_name: &str,
    width: u32,
    height: u32,
    rate: (i32, i32),
) {
    fixture_color(
        path,
        codec_name,
        width,
        height,
        rate,
        ffmpeg::color::Range::Unspecified,
    );
}

fn fixture_color(
    path: &std::path::Path,
    codec_name: &str,
    width: u32,
    height: u32,
    rate: (i32, i32),
    range: ffmpeg::color::Range,
) {
    ffmpeg::init().unwrap();
    let codec = ffmpeg::encoder::find_by_name(codec_name).expect("test encoder");
    let mut output = ffmpeg::format::output(path).unwrap();
    let mut stream = output.add_stream(codec).unwrap();
    let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .unwrap();
    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_format(ffmpeg::format::Pixel::YUV420P);
    encoder.set_color_range(range);
    if range != ffmpeg::color::Range::Unspecified {
        // Require an explicit VUI range flag even for the default limited range.
        encoder.set_colorspace(ffmpeg::color::Space::BT709);
    }
    encoder.set_time_base((rate.1, rate.0));
    encoder.set_frame_rate(Some(rate));
    encoder.set_gop(48);
    encoder.set_max_b_frames(if codec_name == "libx264" { 3 } else { 0 });
    encoder.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
    stream.set_parameters(&encoder);
    let mut options = ffmpeg::Dictionary::new();
    if codec_name == "libx264" {
        options.set("crf", "10");
        options.set("x264-params", "b-adapt=0:scenecut=0");
    }
    let mut encoder = encoder.open_with(options).unwrap();
    stream.set_parameters(&encoder);
    output.write_header().unwrap();
    let time_base = output.stream(0).unwrap().time_base();

    let drain = |encoder: &mut ffmpeg::encoder::video::Encoder,
                 output: &mut ffmpeg::format::context::Output| {
        loop {
            let mut packet = ffmpeg::Packet::empty();
            match encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(0);
                    packet.set_duration(1);
                    packet.rescale_ts((rate.1, rate.0), time_base);
                    packet.write_interleaved(output).unwrap();
                }
                Err(ffmpeg::Error::Eof) => break,
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => break,
                Err(error) => panic!("encode: {error}"),
            }
        }
    };
    for i in 0..FRAMES {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width, height);
        frame.data_mut(0).fill(32 + i as u8 * 4);
        frame.data_mut(1).fill(96);
        frame.data_mut(2).fill(160);
        frame.set_pts(Some(i as i64));
        encoder.send_frame(&frame).unwrap();
        drain(&mut encoder, &mut output);
    }
    encoder.send_eof().unwrap();
    drain(&mut encoder, &mut output);
    output.write_trailer().unwrap();
}

fn assert_pixels(y: &[u8], u: &[u8], v: &[u8], index: usize) {
    for (plane, expected) in [(y, 32 + (index % FRAMES) as u8 * 4), (u, 96), (v, 160)] {
        assert!(
            plane.iter().all(|value| value.abs_diff(expected) <= 2),
            "frame {index}: expected plane value {expected}"
        );
    }
}

fn drain_cpu(decoder: &mut VideoDecoder, count: usize) {
    for index in 0..count {
        let frame = decoder.next_frame().unwrap().expect("missing frame");
        assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT));
        assert_eq!(frame.timestamp_us, index as i64 * 1_000_000 / 30);
        assert_pixels(&frame.y, &frame.u, &frame.v, index);
    }
    for _ in 0..3 {
        assert!(decoder.next_frame().unwrap().is_none());
    }
}

#[test]
fn reordered_frames_drain_and_seek_without_loss() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    for input in [
        InputPath::Single(path.clone()),
        InputPath::Chained(vec![path.clone(), path.clone()]),
    ] {
        let count = if matches!(input, InputPath::Chained(_)) {
            FRAMES * 2
        } else {
            FRAMES
        };
        let mut decoder = VideoDecoder::open_input_software(&input).unwrap();
        drain_cpu(&mut decoder, count);
        decoder.seek_to_secs(0.0).unwrap();
        drain_cpu(&mut decoder, count);
    }
}

#[test]
#[cfg(feature = "gpu")]
fn wrong_backend_is_error_without_consuming_a_frame() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let mut decoder = VideoDecoder::open_software(&path).unwrap();
    assert!(matches!(
        decoder.next_frame_gpu(),
        Err(DecodeError::BackendMismatch { .. })
    ));
    drain_cpu(&mut decoder, FRAMES);
}

#[test]
fn missing_later_segment_is_a_read_error_not_successful_eof() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let input = InputPath::Chained(vec![path, dir.path().join("missing.mp4")]);
    let mut decoder = VideoDecoder::open_input_software(&input).unwrap();
    let mut received = 0;
    loop {
        match decoder.next_frame() {
            Ok(Some(_)) => received += 1,
            Ok(None) => panic!("missing segment reported as EOF"),
            Err(DecodeError::Ffmpeg(_)) => break,
            Err(error) => panic!("unexpected error: {error}"),
        }
    }
    assert!(received > 0, "must exercise a mid-stream failure");
    let mut source = FfmpegFileSource::open_inputs(std::slice::from_ref(&input), 0, true).unwrap();
    let mut delivered = 0;
    loop {
        match source.next_frame() {
            Ok(Some(_)) => delivered += 1,
            Ok(None) => panic!("worker failure reported as EOF"),
            Err(_) => break,
        }
    }
    assert_eq!(delivered, received);
    assert!(!source.is_exhausted());

    // The absent segment prevents seeking into it. The respawned worker
    // must expose that seek failure, not disconnect as successful EOF.
    assert!(decoder.seek_to_secs(1000.0 / 30.0).is_err());
    source.seek(1000).unwrap();
    assert!(source.next_frame().is_err());
    assert!(!source.is_exhausted());
    source.seek(0).unwrap();
    assert!(source.next_frame().unwrap().is_some());

    let lens =
        reco_core::calibration::Lens::fisheye(WIDTH, HEIGHT, 64.0, 64.0, 64.0, 48.0, [0.0; 4]);
    let calibration = reco_core::calibration::Calibration::new(
        vec![lens.clone(), lens],
        reco_core::projection::LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        reco_core::calibration::Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let result = reco_io::StitchJob::with_calibration(
        [input.clone(), input],
        calibration,
        dir.path().join("out.mp4"),
    )
    .cpu()
    .resolution(64, 64)
    .audio(reco_io::output::AudioMode::Disabled)
    .encoder_name("libx264")
    .run(&std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
        false,
    )));
    assert!(
        matches!(
            result,
            Err(reco_io::stitch_job::StitchError::Session(
                reco_core::session::types::SessionError::Source(_)
            ))
        ),
        "{result:?}"
    );
}

#[test]
fn file_source_delivers_every_frame_and_persists_eof() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let input = InputPath::Single(path);
    for inputs in [vec![input.clone()], vec![input.clone(); 2], vec![input; 3]] {
        let mut source = FfmpegFileSource::open_inputs(&inputs, 0, true).unwrap();
        assert_eq!(source.decode_backend(), DecodeBackend::Software);
        for index in 0..FRAMES {
            let FrameSet::Yuv420p(cameras) = source.next_frame().unwrap().unwrap().images else {
                panic!("expected CPU frames");
            };
            assert_eq!(cameras.len(), inputs.len());
            for frame in cameras {
                assert_pixels(&frame.y, &frame.u, &frame.v, index);
            }
        }
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.is_exhausted());
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.try_next_frame().unwrap().is_none());
        source.seek(0).unwrap();
        assert!(source.next_frame().unwrap().is_some());
        source.seek(FRAMES as u64 + 1).unwrap();
        assert!(source.next_frame().unwrap().is_none());
        assert!(source.is_exhausted());
    }
}

#[test]
fn repeated_seek_replacements_deliver_only_the_requested_generation() {
    assert_seek_replacements(true);
}

#[test]
fn seek_frame_respects_container_timestamp_precision() {
    let dir = tempfile::tempdir().unwrap();
    for (extension, codec) in [("mkv", "ffv1"), ("mp4", "libx264")] {
        for rate in [(30, 1), (30000, 1001)] {
            let path = dir.path().join(format!("seek-{}.{}", rate.1, extension));
            fixture_rate(&path, codec, WIDTH, HEIGHT, rate);
            let input = InputPath::Single(path);
            for offset in [-3_i64, 0, 3] {
                let mut source =
                    FfmpegFileSource::open_inputs(&[input.clone(), input.clone()], offset, true)
                        .unwrap();
                // Cross the backward-seek worker replacement and forward-discard paths.
                for target in [24, 1, 2, 4, 31, 7, 0, 13, 11] {
                    source.seek(target).unwrap();
                    let FrameSet::Yuv420p(images) = source.next_frame().unwrap().unwrap().images
                    else {
                        panic!("expected host frames");
                    };
                    for (camera, image) in images.iter().enumerate() {
                        let skipped = if (camera == 0 && offset < 0) || (camera == 1 && offset > 0)
                        {
                            offset.unsigned_abs()
                        } else {
                            0
                        };
                        assert_pixels(&image.y, &image.u, &image.v, (target + skipped) as usize);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC"]
fn cuda_file_source_seek_replacements_deliver_requested_pixels() {
    assert_seek_replacements(false);
}

fn assert_seek_replacements(software_decode: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seek.mp4");
    fixture(&path);
    let input = InputPath::Single(path);
    for (count, offset) in [(1, 0_i64), (2, 0), (2, -3), (2, 3), (3, 0)] {
        let mut source =
            FfmpegFileSource::open_inputs(&vec![input.clone(); count], offset, software_decode)
                .unwrap();
        assert_eq!(
            source.decode_backend(),
            if software_decode {
                DecodeBackend::Software
            } else {
                DecodeBackend::Cuda
            }
        );
        for cycle in 0..6 {
            source.seek(43).unwrap();
            assert!(source.next_frame().unwrap().is_some());
            // All targets lie inside the input and its GOP. Queue backward
            // replacements while workers may be decoding/discarding frames.
            for frame in [42, 36, 30, 24, 18] {
                source.seek(frame).unwrap();
            }
            let target = cycle % 3 * 6;
            source.seek(target as u64).unwrap();
            let FrameSet::Yuv420p(cameras) = source.next_frame().unwrap().unwrap().images else {
                panic!("expected CPU frames");
            };
            assert_eq!(cameras.len(), count);
            for (camera, frame) in cameras.into_iter().enumerate() {
                let skipped =
                    if count == 2 && ((offset > 0 && camera == 1) || (offset < 0 && camera == 0)) {
                        offset.unsigned_abs() as usize
                    } else {
                        0
                    };
                assert_pixels(&frame.y, &frame.u, &frame.v, target + skipped);
            }
        }
    }
}

/// Opt-in hardware gate: fails rather than silently skipping when CUDA is absent.
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC"]
#[cfg(feature = "gpu")]
fn cuda_drains_reordered_frames_and_seeks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    for input in [
        InputPath::Single(path.clone()),
        InputPath::Chained(vec![path.clone(), path.clone()]),
    ] {
        let count = if matches!(input, InputPath::Chained(_)) {
            FRAMES * 2
        } else {
            FRAMES
        };
        let mut decoder = VideoDecoder::open_input(&input).unwrap();
        assert_eq!(decoder.backend(), DecodeBackend::Cuda);
        for _ in 0..2 {
            for index in 0..count {
                let frame = decoder
                    .next_frame_gpu()
                    .unwrap()
                    .expect("missing GPU frame");
                assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT));
                assert_eq!(frame.timestamp_us, index as i64 * 1_000_000 / 30);
            }
            for _ in 0..3 {
                assert!(decoder.next_frame_gpu().unwrap().is_none());
            }
            assert!(
                decoder.next_frame().unwrap().is_none(),
                "stranded frames after GPU EOF"
            );
            decoder.seek_to_secs(0.0).unwrap();
        }
        // Validate this hardware decoder's decoded pixel identities as well.
        drain_cpu(&mut decoder, count);
    }
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan external-memory interop"]
fn incompatible_detector_releases_native_decode_workers() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
    use reco_core::projection::Cylinder;
    use reco_core::render::{renderer::InputFormat, viewport::ViewportSize};
    use reco_core::session::StitchSession;
    use std::sync::atomic::AtomicBool;

    struct RejectInput(bool);
    impl Detector for RejectInput {
        fn name(&self) -> &'static str {
            "incompatible"
        }
        fn prefers_cuda_nv12(&self) -> bool {
            self.0
        }
        fn detect(
            &mut self,
            _: usize,
            frame: &DetectorFrame<'_>,
        ) -> Result<Vec<Detection>, DetectorError> {
            assert_eq!(
                frame.variant_name(),
                if self.0 { "CudaNv12" } else { "WgpuNv12" }
            );
            Err(DetectorError::UnsupportedFrameKind)
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let gpu = reco_core::gpu::GpuContext::new_blocking().unwrap();
    let input = InputPath::Single(path);
    for depth in [0, 3] {
        for cuda in [false, true] {
            let mut source =
                reco_io::SmartFileSource::open(std::slice::from_ref(&input), &gpu, 0).unwrap();
            assert!(source.is_gpu_resident());
            let calibration = Calibration::new(
                vec![Lens::flat(WIDTH, HEIGHT)],
                Cylinder {
                    radius: 1.0,
                    arc_deg: 180.0,
                    surface_height: Some(1.5),
                },
                Framing {
                    axis_offset: 0.0,
                    tilt: 0.0,
                    roll: 0.0,
                },
            );
            let mut session = StitchSession::builder()
                .gpu(gpu.clone())
                .calibration(calibration)
                .source_sizes(vec![(WIDTH, HEIGHT)])
                .input_format(InputFormat::Nv12)
                .viewport_size(ViewportSize {
                    width: 64,
                    height: 36,
                })
                .detector(Box::new(RejectInput(cuda)))
                .build()
                .unwrap();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(depth, usize::MAX);
            session
                .setup_gpu_source(source.shared_texture_set().unwrap())
                .unwrap();
            let error = session
                .run(&mut source, u64::MAX, &AtomicBool::new(false), None)
                .unwrap_err();
            assert!(
                error.to_string().contains("detector 'incompatible'"),
                "{error}"
            );
            assert_eq!(session.telemetry_snapshot().frames_processed, 0);
            // The session remains alive while the producer joins its decode worker.
            drop(source);
            session.finish().unwrap();
        }
    }
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC and Vulkan external-memory interop"]
fn cuda_workers_preserve_completion_failure_and_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reordered.mp4");
    fixture(&path);
    let software_path = dir.path().join("software.mkv");
    fixture_encoding(&software_path, "ffv1");
    let gpu = reco_core::gpu::GpuContext::new_blocking().unwrap();
    let input = InputPath::Single(path.clone());

    for offset in [-3_i64, 0, 3] {
        let mut source =
            reco_io::SmartFileSource::open(&[input.clone(), input.clone()], &gpu, offset).unwrap();
        assert!(source.is_gpu_resident(), "requires the CUDA zero-copy path");
        assert_eq!(
            source.skip_frames((FRAMES + 1) as u64).unwrap(),
            FRAMES as u64 - offset.unsigned_abs()
        );
        assert!(source.is_exhausted());
        assert!(source.next_frame().unwrap().is_none());
        assert_eq!(source.skip_frames(1).unwrap(), 0);
    }

    // A shorter input terminates the pair, and dropping the source releases
    // the longer producer even when it is waiting for a reusable slot.
    let chain = InputPath::Chained(vec![path.clone(), path]);
    for (left, right) in [(&chain, &input), (&input, &chain)] {
        let mut source =
            reco_io::SmartFileSource::open(&[left.clone(), right.clone()], &gpu, 0).unwrap();
        assert!(source.is_gpu_resident());
        assert_eq!(
            source.skip_frames((FRAMES * 2) as u64).unwrap(),
            FRAMES as u64
        );
    }

    let software_input = InputPath::Single(software_path);
    for (left, right) in [(&input, &software_input), (&software_input, &input)] {
        let mut source =
            reco_io::SmartFileSource::open(&[left.clone(), right.clone()], &gpu, 0).unwrap();
        assert!(source.is_gpu_resident());
        let error = source
            .next_frame()
            .err()
            .expect("software frame must not become EOF");
        assert!(
            error
                .to_string()
                .contains("unexpected decoded frame format"),
            "{error}"
        );
        assert!(!source.is_exhausted());
        assert_eq!(
            source.next_frame().err().unwrap().to_string(),
            error.to_string()
        );
        assert_eq!(
            source.skip_frames(1).unwrap_err().to_string(),
            error.to_string()
        );
    }

    for offset in [0, 3] {
        let mut source =
            reco_io::SmartFileSource::open(std::slice::from_ref(&input), &gpu, offset).unwrap();
        assert!(source.is_gpu_resident());
        assert_eq!(
            source.skip_frames((FRAMES + 1) as u64).unwrap(),
            FRAMES as u64
        );
        assert!(source.is_exhausted());
    }

    let smaller = dir.path().join("smaller.mp4");
    fixture_size(&smaller, "libx264", 64, 64);
    let resized = InputPath::Chained(vec![input.first_path().to_path_buf(), smaller]);
    let mut source =
        reco_io::SmartFileSource::open(&[resized.clone(), resized.clone()], &gpu, 0).unwrap();
    assert!(source.is_gpu_resident());
    assert!(source.skip_frames((FRAMES * 2) as u64).is_err());
    assert!(!source.is_exhausted());
    drop(source);

    let mut source =
        reco_io::SmartFileSource::open(&[input.clone(), input.clone()], &gpu, 0).unwrap();
    assert!(source.is_gpu_resident());
    assert_eq!(source.skip_frames(1).unwrap(), 1);
    drop(source);
}

#[test]
fn dimensions_cannot_change_under_a_fixed_frame_layout() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.mp4");
    let second = dir.path().join("smaller.mp4");
    fixture(&first);
    fixture_size(&second, "libx264", 64, 64);
    let input = InputPath::Chained(vec![first, second]);
    let mut decoder = VideoDecoder::open_input_software(&input).unwrap();
    let mut count = 0;
    loop {
        match decoder.next_frame() {
            Ok(Some(_)) => count += 1,
            Ok(None) => panic!("changed dimensions must be rejected"),
            Err(error) => {
                assert!(
                    error.to_string().contains("decoded dimensions changed"),
                    "{error}"
                );
                break;
            }
        }
    }
    assert_eq!(count, FRAMES);
}

#[test]
fn explicit_color_range_cannot_change_under_a_fixed_source_description() {
    check_segment_color_ranges(false);
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore = "requires NVIDIA CUDA/NVDEC"]
fn cuda_explicit_color_range_cannot_change_under_a_fixed_source_description() {
    check_segment_color_ranges(true);
}

fn check_segment_color_ranges(hardware: bool) {
    let dir = tempfile::tempdir().unwrap();
    let paths = [dir.path().join("limited.mp4"), dir.path().join("full.mp4")];
    for (path, range) in paths
        .iter()
        .zip([ffmpeg::color::Range::MPEG, ffmpeg::color::Range::JPEG])
    {
        fixture_color(path, "libx264", WIDTH, HEIGHT, (30, 1), range);
    }
    for order in [[0, 1], [1, 0], [0, 0], [1, 1]] {
        let input = InputPath::Chained(order.map(|index| paths[index].clone()).to_vec());
        let mut decoder = if hardware {
            VideoDecoder::open_input(&input).unwrap()
        } else {
            VideoDecoder::open_input_software(&input).unwrap()
        };
        if hardware {
            assert_eq!(decoder.backend(), DecodeBackend::Cuda);
        }
        assert_eq!(decoder.is_full_range(), order[0] == 1);
        let mut count = 0;
        loop {
            #[cfg(feature = "gpu")]
            let received = if hardware {
                decoder.next_frame_gpu().map(|frame| frame.is_some())
            } else {
                decoder.next_frame().map(|frame| frame.is_some())
            };
            #[cfg(not(feature = "gpu"))]
            let received = decoder.next_frame().map(|frame| frame.is_some());
            match received {
                Ok(true) => count += 1,
                Ok(false) => {
                    assert_eq!(order[0], order[1], "changed color range must be rejected");
                    assert_eq!(count, FRAMES * 2);
                    break;
                }
                Err(error) => {
                    assert_ne!(order[0], order[1], "{error}");
                    assert!(
                        error.to_string().contains("decoded color range changed"),
                        "{error}"
                    );
                    // FFmpeg can report new metadata on still-buffered B pictures.
                    assert!((FRAMES - 3..=FRAMES).contains(&count), "count={count}");
                    break;
                }
            }
        }
    }
}
