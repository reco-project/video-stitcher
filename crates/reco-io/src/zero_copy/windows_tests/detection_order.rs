//! Native source-frame identity checks at the detector and output boundaries.

use super::*;
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::gpu::GpuContext;
use reco_core::wgpu;
use std::sync::{Arc, Mutex};

struct LumaProbe {
    gpu: GpuContext,
    samples: Arc<Mutex<Vec<(usize, u8)>>>,
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
        let DetectorFrame::WgpuNv12 { y_view, .. } = frame else {
            panic!("D3D11 must use native staging views")
        };
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
                    resource: wgpu::BindingResource::TextureView(y_view),
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
#[ignore = "requires D3D11VA and RECO_DECODE_TEST_VIDEO with distinct luma at (0,0) per frame"]
fn d3d11_detection_reads_current_frame_with_and_without_lookahead() {
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::render::{renderer::InputFormat, viewport::ViewportSize};
    use reco_core::stitch::{Executor, GpuExecutor, GpuExecutorConfig};
    let input = input();
    let mut reference = VideoDecoder::open_input_software(&input).unwrap();
    let (w, h) = (reference.width(), reference.height());
    let mut expected = Vec::new();
    while let Some(frame) = reference.next_frame().unwrap() {
        expected.push(frame.y[0]);
    }
    assert!(expected.len() >= 8);
    assert!(
        expected.windows(2).all(|p| p[0].abs_diff(p[1]) > 2),
        "fixture must distinguish consecutive source frames"
    );
    let gpu = GpuContext::new_blocking().unwrap();
    for lookahead in [0, 2] {
        for interval in [1, 3] {
            let lens = || {
                Lens::fisheye(
                    w,
                    h,
                    w as f64 / 2.0,
                    w as f64 / 2.0,
                    w as f64 / 2.0,
                    h as f64 / 2.0,
                    [0.0; 4],
                )
            };
            let calibration = Calibration::new(
                vec![lens(), lens()],
                reco_core::projection::LShape {
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
            let config = GpuExecutorConfig {
                viewport_size: ViewportSize {
                    width: 64,
                    height: 64,
                },
                ..GpuExecutorConfig::new(calibration, w, h, InputFormat::Nv12)
            };
            let mut session = reco_core::session::StitchSession::with_executor(Executor::Gpu(
                Box::new(GpuExecutor::new(gpu.clone(), config).unwrap()),
            ))
            .unwrap();
            session.core_mut().set_constrained_look(false);
            session.set_lookahead(lookahead);
            session.set_detection_interval(interval);
            let samples = Arc::new(Mutex::new(Vec::new()));
            session.set_detector(Box::new(LumaProbe {
                gpu: gpu.clone(),
                samples: samples.clone(),
            }));
            let mut source = crate::SmartFileSource::open(&input, &input, &gpu, 3).unwrap();
            assert!(source.is_gpu_resident());
            let result = session
                .run(
                    &mut source,
                    u64::MAX,
                    &std::sync::atomic::AtomicBool::new(false),
                    None,
                )
                .unwrap();
            session.finish().unwrap();
            let wanted: Vec<_> = (0..expected.len() - 3)
                .step_by(interval as usize)
                .flat_map(|i| [(0, expected[i]), (1, expected[i + 3])])
                .collect();
            assert_eq!(
                *samples.lock().unwrap(),
                wanted,
                "lookahead={lookahead}, interval={interval}"
            );
            assert_eq!(result.frames_processed as usize, expected.len() - 3);
        }
    }
}
