//! Native input staging tested with ordinary GPU textures, without Jetson APIs.

use super::*;

#[test]
fn native_detection_dispatch_preserves_ownership_camera_order_and_capture_retries() {
    struct Probe {
        calls: Arc<Mutex<Vec<usize>>>,
        owners: [std::sync::Weak<()>; 2],
    }
    impl Detector for Probe {
        fn name(&self) -> &'static str {
            "nvmm-lifetime-probe"
        }
        fn detect(
            &mut self,
            camera: usize,
            frame: &DetectorFrame<'_>,
        ) -> Result<Vec<Detection>, DetectorError> {
            let DetectorFrame::Nvmm(image) = frame else {
                panic!("expected native image");
            };
            assert_eq!((image.info().width, image.info().height), (4, 2));
            assert!(self.owners[camera].upgrade().is_some());
            let mut calls = self.calls.lock().unwrap();
            calls.push(camera);
            if calls.len() == 1 {
                return Err(DetectorError::InferenceFailed(
                    "temporary transform failure".into(),
                ));
            }
            Ok(Vec::new())
        }
    }
    for buffered in [false, true] {
        let owners = [Arc::new(()), Arc::new(())];
        let weak = owners.each_ref().map(Arc::downgrade);
        let frame =
            FrameSet::NvmmResident(owners.map(crate::source::NvmmFrame::lifetime_test_frame));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut session = build_cpu_session();
        session.set_detector(Box::new(Probe {
            calls: calls.clone(),
            owners: weak.clone(),
        }));
        for (index, times_us) in [[10, 10], [10, 10], [10, 10], [20, 20]]
            .into_iter()
            .enumerate()
        {
            let capture = crate::source::CaptureTiming {
                at_us: times_us[0],
                times_us: times_us.to_vec(),
                max_age_us: None,
            };
            if buffered {
                session
                    .detect_and_track_only(
                        &frame,
                        std::time::Duration::ZERO,
                        index as u64,
                        Some(&capture),
                        1.0 / 30.0,
                    )
                    .unwrap();
            } else {
                // Stop at pose validation after real immediate detection, before native rendering.
                let error = session
                    .process_frame_any(
                        Some(&frame),
                        std::time::Duration::ZERO,
                        None,
                        crate::telemetry::FrameTiming::default(),
                        std::time::Instant::now(),
                        &FrameLoopContext {
                            capture: Some(capture),
                            frame_seconds: 0.0,
                            ..Default::default()
                        },
                    )
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("invalid presentation pose or frame duration"),
                    "{error}"
                );
            }
        }
        assert_eq!(*calls.lock().unwrap(), [0, 1, 0, 0, 1]);
        assert!(weak.iter().all(|owner| owner.upgrade().is_some()));
        drop(frame);
        assert!(weak.iter().all(|owner| owner.upgrade().is_none()));
    }
}

#[test]
#[ignore = "requires GPU staging and readback, no native NVMM import"]
fn nvmm_staged_pixels_survive_capture_reuse() {
    type ImageWithTime = (Vec<u8>, Option<i64>);
    struct Pixels(Arc<Mutex<Vec<ImageWithTime>>>);
    impl OutputSink for Pixels {
        fn name(&self) -> &str {
            "pixels"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(crate::sink::PixelFormat::Nv12)
        }
        fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
            self.0
                .lock()
                .unwrap()
                .push((frame.data.to_vec(), frame.pts_us));
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            Ok(())
        }
    }
    let pixels = Arc::new(Mutex::new(Vec::new()));
    let mut session = StitchSession::builder()
        .calibration(test_calibration())
        .source_sizes(vec![(W, H); 2])
        .input_format(crate::render::renderer::InputFormat::Nv12)
        .viewport_size(ViewportSize {
            width: W,
            height: H,
        })
        .build()
        .unwrap();
    session.core_mut().set_constrained_look(false);
    let queue = session.core.gpu().queue.clone();
    let textures: [wgpu::Texture; 4] = std::array::from_fn(|i| {
        let uv = i % 2 == 1;
        session
            .core
            .gpu()
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("simulated capture"),
                size: wgpu::Extent3d {
                    width: if uv { W / 2 } else { W },
                    height: if uv { H / 2 } else { H },
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if uv {
                    wgpu::TextureFormat::Rg8Unorm
                } else {
                    wgpu::TextureFormat::R8Unorm
                },
                usage: wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
    });
    let upload = |values: [u8; 4]| {
        for (texture, value) in textures.iter().zip(values) {
            queue.write_texture(
                texture.as_image_copy(),
                &vec![value; (W * texture.height()) as usize],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(W),
                    rows_per_image: None,
                },
                texture.size(),
            );
        }
    };
    session
        .gpu_exec()
        .create_lookahead_pool(1, crate::render::renderer::GpuPixelFormat::Nv12)
        .unwrap();
    use crate::gpu::yuv_stack_packer::{OutputTileSize, StackGridLayout, StackedAtlas};
    struct Recorder(Arc<Mutex<Vec<(u8, u8)>>>);
    impl crate::core::types::StackedReplayGpuRecorder for Recorder {
        fn record_atlas(&mut self, atlas: &StackedAtlas) {
            self.0
                .lock()
                .unwrap()
                .push((atlas.y[0], atlas.y[(W * H) as usize]));
        }
    }
    for staged in [false, true] {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        session
            .enable_gpu_stacked_replay(
                StackGridLayout::vstack(W, H, 2).unwrap(),
                OutputTileSize::unscaled(W, H),
            )
            .unwrap();
        session.set_stacked_gpu_recorder(Box::new(Recorder(recorded.clone())));
        session
            .add_sink(
                Box::new(Pixels(pixels.clone())),
                crate::session::sinks::SinkOptions::inline_lossy(),
            )
            .unwrap();
        for i in 0..4 {
            let pts_us = Some([0, 33_333, 133_333, 141_667][i as usize]);
            upload([64 + i * 20, 100, 180 - i * 10, 150]);
            if staged {
                let slot = session
                    .gpu_exec()
                    .stage_textures_to_pool(&[
                        (&textures[0], &textures[1]),
                        (&textures[2], &textures[3]),
                    ])
                    .unwrap()
                    .unwrap();
                // The producer overwrites its original allocation before render.
                upload([0, 0, 0, 0]);
                session.current_vram_slot = Some(slot);
                let ctx = FrameLoopContext {
                    tracking_complete: true,
                    capture: None,
                    frame_seconds: 1.0 / 30.0,
                };
                session
                    .process_frame_any(
                        None,
                        std::time::Duration::ZERO,
                        pts_us,
                        crate::telemetry::FrameTiming::default(),
                        std::time::Instant::now(),
                        &ctx,
                    )
                    .unwrap();
                session.gpu_exec().release_pool_slot(slot);
                session.current_vram_slot = None;
            } else {
                let pose = session.advance_render_pose(1.0 / 30.0).unwrap();
                let [ly, lu, ry, ru] = textures
                    .each_ref()
                    .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
                session
                    .render_frame_imported_nv12(&[(&ly, &lu), (&ry, &ru)], pose, pts_us)
                    .unwrap();
            }
        }
        session.clear_stacked_gpu_recorder();
        session.finish().unwrap();
        assert_eq!(
            *recorded.lock().unwrap(),
            [(64, 180), (84, 170), (104, 160), (124, 150)],
            "staged={staged}"
        );
    }
    let pixels = pixels.lock().unwrap();
    assert_eq!(pixels.len(), 8);
    assert_ne!(
        pixels[0].0, pixels[1].0,
        "fixture must distinguish successive images"
    );
    assert_eq!(
        pixels[..4],
        pixels[4..],
        "staged rendering must preserve pixels and ordering"
    );
    assert_eq!(
        pixels[..4].iter().map(|image| image.1).collect::<Vec<_>>(),
        [Some(0), Some(33_333), Some(133_333), Some(141_667)]
    );
}
