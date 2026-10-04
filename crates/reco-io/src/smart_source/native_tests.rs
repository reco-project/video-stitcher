//! Native source-frame identity checks at the detector and output boundaries.

use super::*;
use crate::InputPath;
use crate::ffmpeg::decoder::{DecodeBackend, VideoDecoder};
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::gpu::GpuContext;
use reco_core::wgpu;
use std::sync::{Arc, Mutex};

#[derive(Default, Clone)]
struct CapturedFrames(Arc<Mutex<Vec<Vec<u8>>>>);

impl reco_core::sink::OutputSink for CapturedFrames {
    fn name(&self) -> &str {
        "capture"
    }
    fn wants(&self) -> reco_core::sink::SinkInput {
        reco_core::sink::SinkInput::CpuBytes(reco_core::sink::PixelFormat::Nv12)
    }
    fn consume(
        &mut self,
        frame: reco_core::sink::OutputFrame<'_>,
    ) -> Result<(), reco_core::sink::SinkError> {
        self.0.lock().unwrap().push(frame.data.to_vec());
        Ok(())
    }
    fn finish(&mut self) -> Result<(), reco_core::sink::SinkError> {
        Ok(())
    }
}

struct LumaProbe {
    gpu: GpuContext,
    samples: Arc<Mutex<Vec<(usize, u8)>>>,
    sizes: Vec<(u32, u32)>,
    #[cfg(target_os = "macos")]
    cache: reco_core::interop::metal::MetalTextureCache,
}

impl Detector for LumaProbe {
    fn name(&self) -> &'static str {
        "luma-probe"
    }
    fn detect(
        &mut self,
        camera: usize,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        #[cfg(target_os = "macos")]
        let imported = match frame {
            DetectorFrame::Metal {
                cv_pixel_buffer, ..
            } => Some(unsafe { self.cache.import_nv12(*cv_pixel_buffer, &self.gpu) }.unwrap()),
            _ => None,
        };
        let (y_view, size) = match frame {
            DetectorFrame::WgpuNv12 {
                y_view,
                width,
                height,
                ..
            } => ((*y_view).clone(), (*width, *height)),
            #[cfg(target_os = "macos")]
            DetectorFrame::Metal { width, height, .. } => (
                imported
                    .as_ref()
                    .unwrap()
                    .0
                    .texture
                    .create_view(&Default::default()),
                (*width, *height),
            ),
            _ => panic!("expected native detector input"),
        };
        assert_eq!(size, self.sizes[camera]);
        assert_eq!(
            (y_view.texture().width(), y_view.texture().height()),
            self.sizes[camera]
        );
        let device = self.gpu.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var y: texture_2d<f32>; @group(0) @binding(1) var<storage, read_write> value: u32; @compute @workgroup_size(1) fn main() { value = u32(round(textureLoad(y, vec2i(0), 0).r * 255.)); }".into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, 4);
        self.gpu.queue().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = readback.slice(..).get_mapped_range();
        let value = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u8;
        self.samples.lock().unwrap().push((camera, value));
        Ok(Vec::new())
    }
}

#[test]
#[ignore = "requires native Windows/Metal decode and RECO_DECODE_TEST_VIDEO/RECO_DECODE_TEST_VIDEO_OTHER, different-sized same-format fixtures with distinct luma per frame"]
fn native_camera_frames_keep_dimensions_and_identity_with_and_without_lookahead() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::{Cylinder, LShape};
    use reco_core::render::{renderer::InputFormat, viewport::ViewportSize};
    use reco_core::stitch::{Executor, GpuExecutor, GpuExecutorConfig};

    let inputs: Vec<_> = ["RECO_DECODE_TEST_VIDEO", "RECO_DECODE_TEST_VIDEO_OTHER"]
        .map(|key| InputPath::Single(std::env::var(key).unwrap().into()))
        .into();
    let gpu = GpuContext::new_blocking().unwrap();
    let mut sizes = Vec::new();
    let mut reference = Vec::new();
    for input in &inputs {
        let probe = VideoDecoder::open_input(input).unwrap();
        #[cfg(target_os = "windows")]
        assert_eq!(probe.backend(), DecodeBackend::D3d11va);
        #[cfg(target_os = "macos")]
        assert_eq!(probe.backend(), DecodeBackend::VideoToolbox);
        sizes.push((probe.width(), probe.height()));
        drop(probe);
        let mut decoder = VideoDecoder::open_input_software(input).unwrap();
        let mut luma = Vec::new();
        while let Some(frame) = decoder.next_frame().unwrap() {
            luma.push(frame.y[0]);
        }
        assert!(luma.len() >= 24);
        assert!(
            luma.windows(2).all(|p| p[0].abs_diff(p[1]) > 2),
            "fixture must distinguish consecutive frames"
        );
        reference.push(luma);
    }
    assert_ne!(sizes[0], sizes[1]);

    for order in [vec![0], vec![1], vec![0, 0], vec![0, 1], vec![1, 0]] {
        let selected: Vec<_> = order.iter().map(|&i| inputs[i].clone()).collect();
        let source_sizes: Vec<_> = order.iter().map(|&i| sizes[i]).collect();
        for offset in [-3_i64, 0, 3] {
            let mut baseline = None;
            for lookahead in [0, 2, 8] {
                for interval in [1, 3] {
                    let lenses: Vec<_> = source_sizes
                        .iter()
                        .map(|&(w, h)| {
                            Lens::fisheye(
                                w,
                                h,
                                w as f64 / 2.0,
                                w as f64 / 2.0,
                                w as f64 / 2.0,
                                h as f64 / 2.0,
                                [0.0; 4],
                            )
                        })
                        .collect();
                    let framing = Framing {
                        axis_offset: 0.25,
                        tilt: 0.0,
                        roll: 0.0,
                    };
                    let calibration = if order.len() == 1 {
                        Calibration::new(
                            lenses,
                            Cylinder {
                                radius: 20.0,
                                arc_deg: 180.0,
                                surface_height: None,
                            },
                            framing,
                        )
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
                    let mut config = GpuExecutorConfig::new(
                        calibration,
                        source_sizes.clone(),
                        InputFormat::Nv12,
                    );
                    config.viewport_size = ViewportSize {
                        width: 64,
                        height: 64,
                    };
                    let mut session = reco_core::session::StitchSession::with_executor(
                        Executor::Gpu(Box::new(GpuExecutor::new(gpu.clone(), config).unwrap())),
                    )
                    .unwrap();
                    session.core_mut().set_constrained_look(false);
                    session.set_lookahead(lookahead, usize::MAX);
                    session.set_detection_interval(interval);
                    let rendered = CapturedFrames::default();
                    session
                        .add_sink(
                            Box::new(rendered.clone()),
                            reco_core::session::SinkOptions::inline_lossy(),
                        )
                        .unwrap();
                    let samples = Arc::new(Mutex::new(Vec::new()));
                    session.set_detector(Box::new(LumaProbe {
                        gpu: gpu.clone(),
                        samples: samples.clone(),
                        sizes: source_sizes.clone(),
                        #[cfg(target_os = "macos")]
                        cache: reco_core::interop::metal::MetalTextureCache::new(&gpu).unwrap(),
                    }));

                    let automatic = SmartFileSource::open(&selected, &gpu, offset).unwrap();
                    assert_eq!(automatic.is_gpu_resident(), order == [0, 0]);
                    let info = automatic.info();
                    let format = automatic.pixel_format;
                    let full_range = automatic.full_range;
                    let rotations = [automatic.left_rotation, automatic.right_rotation];
                    drop(automatic);
                    // Exercise the native adapter without weakening automatic source admission.
                    let mut source = SmartFileSource::open_zero_copy(
                        &selected,
                        &gpu,
                        offset,
                        info,
                        format,
                        full_range,
                        rotations[0],
                        rotations[1],
                    )
                    .unwrap();
                    let result = session
                        .run(
                            &mut source,
                            u64::MAX,
                            &std::sync::atomic::AtomicBool::new(false),
                            None,
                        )
                        .unwrap();
                    session.finish().unwrap();
                    let skips: Vec<_> = (0..order.len())
                        .map(|camera| {
                            if order.len() == 2 && camera == usize::from(offset > 0) {
                                offset.unsigned_abs() as usize
                            } else {
                                0
                            }
                        })
                        .collect();
                    let count = order
                        .iter()
                        .enumerate()
                        .map(|(camera, &i)| reference[i].len() - skips[camera])
                        .min()
                        .unwrap();
                    let mut wanted = Vec::new();
                    for frame in (0..count).step_by(interval as usize) {
                        for (camera, &i) in order.iter().enumerate() {
                            wanted.push((camera, reference[i][frame + skips[camera]]));
                        }
                    }
                    assert_eq!(
                        *samples.lock().unwrap(),
                        wanted,
                        "order={order:?}, offset={offset}, lookahead={lookahead}, interval={interval}"
                    );
                    assert_eq!(result.frames_processed as usize, count);
                    let rendered = rendered.0.lock().unwrap();
                    assert_eq!(rendered.len(), count);
                    if let Some(expected) = &baseline {
                        assert_eq!(
                            &*rendered, expected,
                            "rendered order={order:?}, offset={offset}, lookahead={lookahead}, interval={interval}"
                        );
                    } else {
                        baseline = Some(rendered.clone());
                    }
                }
            }
        }
    }
}
