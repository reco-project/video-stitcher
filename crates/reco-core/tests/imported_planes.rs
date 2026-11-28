//! Imported camera views preserve pixels, ownership and output timing across residency changes.

#![cfg(feature = "gpu")]

use std::sync::{Arc, Mutex};

use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::geometry::RenderPose;
use reco_core::gpu::GpuContext;
use reco_core::projection::{Cylinder, LShape};
use reco_core::render::renderer::{GpuPixelFormat, InputFormat};
use reco_core::render::viewport::ViewportSize;
use reco_core::session::{SinkOptions, StitchSession};
use reco_core::sink::{OutputFrame, OutputSink, PixelFormat, SinkError, SinkInput};
use reco_core::source::{FrameSet, Nv12Data};
use reco_core::stitch::{CpuExecutor, Executor, GpuExecutor, GpuExecutorConfig};

type Captured = Arc<Mutex<Vec<(Vec<u8>, Option<i64>)>>>;

fn output_texture(gpu: &GpuContext, width: u32, height: u32) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("caller-owned output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn read_pixels(gpu: &GpuContext, texture: &wgpu::Texture) -> Vec<u8> {
    let mut reader =
        reco_core::gpu::rgba_readback::RgbaReadback::new(gpu, texture.width(), texture.height())
            .unwrap();
    let empty = gpu
        .device()
        .create_command_encoder(&Default::default())
        .finish();
    assert!(reader.readback(gpu, texture, empty).unwrap().is_none());
    reader.flush_pending(gpu).unwrap().unwrap().to_vec()
}

#[test]
fn caller_owned_outputs_preserve_each_image_without_intermediate_waits() {
    let Some(gpu) = gpu() else { return };
    for sizes in [vec![(64, 32)], vec![(64, 32); 2], vec![(64, 32), (128, 16)]] {
        let (mut direct, _) = session(&gpu, &sizes, InputFormat::Nv12);
        let mut outputs = Vec::new();
        let mut last_submission = None;
        for index in 0..6 {
            let images: Vec<_> = sizes
                .iter()
                .enumerate()
                .map(|(camera, size)| image(*size, camera, index))
                .collect();
            let pose = RenderPose {
                yaw: -0.25 + f32::from(index) * 0.1,
                pitch: 0.05,
                fov_degrees: 45.0 + f32::from(index) * 3.0,
            };
            let target = output_texture(&gpu, 128, 72);
            last_submission = Some(
                direct
                    .core_mut()
                    .render_to_view(
                        &FrameSet::Nv12(images),
                        pose,
                        &target.create_view(&Default::default()),
                    )
                    .unwrap(),
            );
            outputs.push(target);
        }
        gpu.device()
            .poll(wgpu::PollType::Wait {
                submission_index: last_submission,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        let mut prior = None;
        for target in outputs {
            let actual = read_pixels(&gpu, &target);
            assert!(actual.chunks_exact(4).any(|pixel| pixel[0] > 16));
            if let Some(previous) = prior {
                assert_ne!(actual, previous);
            }
            prior = Some(actual);
        }
        assert_eq!(direct.core_mut().frame_count(), 0);
    }
}

#[test]
fn caller_owned_output_composes_with_conversion_before_target_reuse() {
    use reco_core::gpu::yuv_converter::YuvConverter;

    let Some(gpu) = gpu() else { return };
    for sizes in [vec![(64, 32)], vec![(64, 32), (128, 16)]] {
        let (mut reference, expected) = session(&gpu, &sizes, InputFormat::Nv12);
        let (mut direct, _) = session(&gpu, &sizes, InputFormat::Nv12);
        let target = output_texture(&gpu, 128, 72);
        let view = target.create_view(&Default::default());
        let mut converter = YuvConverter::new(&gpu, 128, 72).unwrap();
        let mut actual = Vec::new();
        for index in 0..7 {
            let data: Vec<_> = sizes
                .iter()
                .enumerate()
                .map(|(camera, &size)| image(size, camera, index))
                .collect();
            let pose = RenderPose {
                yaw: f32::from(index) * 0.08 - 0.25,
                pitch: 0.03,
                fov_degrees: 60.0,
            };
            reference
                .render_frame(&FrameSet::Nv12(data.clone()), pose, None)
                .unwrap();
            direct
                .core_mut()
                .render_to_view(&FrameSet::Nv12(data), pose, &view)
                .unwrap();
            // Rendering is already submitted. Conversion on the same queue precedes the next render into this target, without a CPU wait between them.
            let empty = gpu
                .device()
                .create_command_encoder(&Default::default())
                .finish();
            if let Some(bytes) = converter
                .convert_and_readback(&gpu, &target, empty)
                .unwrap()
            {
                actual.push(bytes.to_vec());
            }
        }
        while let Some(bytes) = converter.flush_pending(&gpu).unwrap() {
            actual.push(bytes.to_vec());
        }
        reference.finish().unwrap();
        let expected = expected.lock().unwrap();
        assert_eq!(actual.len(), 7);
        assert_eq!(expected.len(), actual.len());
        for (actual, (expected, _)) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
        assert!(actual.windows(2).all(|pair| pair[0] != pair[1]));
    }
}

#[test]
fn caller_target_rejection_preserves_subsequent_host_rendering() {
    let Some(gpu) = gpu() else { return };
    let (mut session, _) = session(&gpu, &[(64, 32); 2], InputFormat::Nv12);
    let data = image((64, 32), 0, 2);
    let (y, _) = views(&gpu, &data, GpuPixelFormat::Nv12);
    let host = FrameSet::Nv12(vec![data; 2]);
    for invalid in [output_texture(&gpu, 64, 72), output_texture(&gpu, 128, 64)] {
        let view = invalid.create_view(&Default::default());
        assert!(
            session
                .core_mut()
                .render_to_view(&host, RenderPose::default(), &view)
                .is_err()
        );
    }
    // A sampled input plane cannot serve as an RGBA output attachment.
    assert!(
        session
            .core_mut()
            .render_to_view(&host, RenderPose::default(), &y)
            .is_err()
    );
    for (mips, layers, samples, format, usage) in [
        (
            2,
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        (
            1,
            2,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        (
            1,
            1,
            4,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        (
            1,
            1,
            1,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        (
            1,
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
    ] {
        let target = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("incompatible caller output"),
            size: wgpu::Extent3d {
                width: 128,
                height: 72,
                depth_or_array_layers: layers,
            },
            mip_level_count: mips,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        assert!(
            session
                .core_mut()
                .render_to_view(&host, RenderPose::default(), &view)
                .is_err()
        );
    }
    let target = output_texture(&gpu, 128, 72);
    let view = target.create_view(&Default::default());
    session
        .core_mut()
        .render_to_view(&host, RenderPose::default(), &view)
        .unwrap();
    let expected = read_pixels(&gpu, &target);
    // Surface-style sRGB allocations may expose an unorm view matching the renderer.
    let surface = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("sRGB allocation with unorm render view"),
        size: target.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    });
    let view = surface.create_view(&wgpu::TextureViewDescriptor {
        format: Some(wgpu::TextureFormat::Rgba8Unorm),
        ..Default::default()
    });
    session
        .core_mut()
        .render_to_view(&host, RenderPose::default(), &view)
        .unwrap();
    assert_eq!(read_pixels(&gpu, &surface), expected);
}
struct Capture(Captured, PixelFormat);
impl OutputSink for Capture {
    fn name(&self) -> &str {
        "capture"
    }
    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(self.1)
    }
    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        assert_eq!(frame.format, self.1);
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

fn calibration(count: usize) -> Calibration {
    let lenses = vec![Lens::flat(64, 32); count];
    let framing = Framing {
        axis_offset: 0.25,
        tilt: 0.0,
        roll: 0.0,
    };
    if count == 1 {
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
                blend_width: 0.08,
            },
            framing,
        )
    }
}

fn session(
    gpu: &GpuContext,
    sizes: &[(u32, u32)],
    format: InputFormat,
) -> (StitchSession, Captured) {
    let mut config = GpuExecutorConfig::new(calibration(sizes.len()), sizes.to_vec(), format);
    config.viewport_size = ViewportSize {
        width: 128,
        height: 72,
    };
    let executor = GpuExecutor::new(gpu.clone(), config).unwrap();
    let mut session = StitchSession::with_executor(Executor::Gpu(Box::new(executor))).unwrap();
    session.core_mut().set_constrained_look(false);
    let output = Captured::default();
    session
        .add_sink(
            Box::new(Capture(output.clone(), PixelFormat::Nv12)),
            SinkOptions::inline_lossy(),
        )
        .unwrap();
    (session, output)
}

fn image((width, height): (u32, u32), camera: usize, frame: u8) -> Arc<Nv12Data> {
    Arc::new(Nv12Data {
        size: (width, height),
        y: (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    32 + ((x * 70 / width
                        + y * 40 / height
                        + camera as u32 * 40
                        + u32::from(frame) * 21)
                        % 180) as u8
                })
            })
            .collect(),
        uv: [100 + camera as u8 * 40, 155 - frame * 5].repeat((width * height / 4) as usize),
    })
}

fn texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    data: &[u8],
    row: u32,
) -> wgpu::Texture {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("imported plane fixture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        texture.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: None,
        },
        texture.size(),
    );
    texture
}

fn views(
    gpu: &GpuContext,
    data: &Nv12Data,
    format: GpuPixelFormat,
) -> (wgpu::TextureView, wgpu::TextureView) {
    let convert = |bytes: &[u8]| -> Vec<u8> {
        match format {
            GpuPixelFormat::Nv12 => bytes.to_vec(),
            // Valid P010 samples, quantized to the nearest lower 10-bit level.
            GpuPixelFormat::P010 => bytes
                .iter()
                .flat_map(|&v| ((u16::from(v) * 257) & 0xffc0).to_le_bytes())
                .collect(),
            _ => unreachable!(),
        }
    };
    let (w, h) = data.size;
    let y = texture(
        gpu,
        w,
        h,
        format.y_format(),
        &convert(&data.y),
        w * format.bytes_per_sample() as u32,
    );
    let uv = texture(
        gpu,
        w / 2,
        h / 2,
        format.uv_format(),
        &convert(&data.uv),
        w * format.bytes_per_sample() as u32,
    );
    (
        y.create_view(&Default::default()),
        uv.create_view(&Default::default()),
    )
}

fn gpu() -> Option<GpuContext> {
    match GpuContext::new_blocking() {
        Ok(gpu) => {
            eprintln!("imported planes adapter: {}", gpu.gpu_name());
            Some(gpu)
        }
        Err(error) if std::env::var_os("RECO_REQUIRE_GPU").is_none() => {
            eprintln!("GPU unavailable: {error}");
            None
        }
        Err(error) => panic!("GPU required: {error}"),
    }
}

#[test]
fn imported_planes_match_host_pixels_and_timestamps_across_shape_and_residency_changes() {
    let Some(gpu) = gpu() else { return };
    for sizes in [
        vec![(64, 32)],
        vec![(64, 32); 2],
        vec![(64, 32), (128, 64)],
        vec![(128, 64), (64, 32)],
        vec![(64, 32), (128, 16)],
    ] {
        for format in [GpuPixelFormat::Nv12, GpuPixelFormat::P010] {
            if format == GpuPixelFormat::P010
                && !gpu
                    .device()
                    .features()
                    .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
            {
                continue;
            }
            let (mut reference, expected) = session(&gpu, &sizes, InputFormat::Nv12);
            let (mut imported, actual) = session(&gpu, &sizes, InputFormat::Nv12);
            for frame in 0..6 {
                let pose = RenderPose {
                    yaw: [-0.4, 0.0, 0.4][frame as usize % 3],
                    pitch: 0.0,
                    fov_degrees: 70.0,
                };
                let time = Some([0, 33_333, 140_000, 190_000, 210_000, 330_000][frame as usize]);
                let data: Vec<_> = sizes
                    .iter()
                    .enumerate()
                    .map(|(camera, &size)| image(size, camera, frame))
                    .collect();
                let host = FrameSet::Nv12(data.clone());
                reference.render_frame(&host, pose, time).unwrap();
                if frame % 3 == 2 {
                    imported.render_frame(&host, pose, time).unwrap();
                } else {
                    // Drop texture handles before rendering, then views before completion.
                    // wgpu must retain the submitted allocations; no explicit destroy/reuse.
                    let planes: Vec<_> =
                        data.iter().map(|data| views(&gpu, data, format)).collect();
                    let refs: Vec<_> = planes.iter().map(|(y, uv)| (y, uv)).collect();
                    imported
                        .render_frame_imported_nv12(&refs, pose, time)
                        .unwrap();
                }
            }
            reference.finish().unwrap();
            imported.finish().unwrap();
            assert_eq!(imported.frame_count(), 6);
            let (expected, actual) = (expected.lock().unwrap(), actual.lock().unwrap());
            assert_eq!(actual.len(), 6);
            assert_eq!(expected.len(), 6);
            assert_ne!(expected[0].0, expected[1].0);
            for ((a, at), (b, bt)) in actual.iter().zip(expected.iter()) {
                assert_eq!(at, bt);
                assert_eq!(a.len(), b.len());
                let max_error = a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                assert!(
                    max_error <= if format == GpuPixelFormat::P010 { 2 } else { 0 },
                    "{sizes:?} {format:?}: error {max_error}"
                );
            }
        }
    }
}

#[test]
fn gpu_p010_import_preserves_adjacent_ten_bit_luma_codes() {
    let Some(gpu) = gpu() else { return };
    if !gpu
        .device()
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    {
        return;
    }
    let (width, height) = (64, 32);
    let mut config =
        GpuExecutorConfig::new(calibration(1), vec![(width, height)], InputFormat::Nv12);
    config.viewport_size = ViewportSize { width, height };
    config.output_format = wgpu::TextureFormat::Rgb10a2Unorm;
    let executor = GpuExecutor::new(gpu.clone(), config).unwrap();
    let mut session = StitchSession::with_executor(Executor::Gpu(Box::new(executor))).unwrap();
    let captured = Captured::default();
    session
        .add_sink(
            Box::new(Capture(captured.clone(), PixelFormat::P010)),
            SinkOptions::inline_lossy(),
        )
        .unwrap();

    let gpu_format = GpuPixelFormat::P010;
    let neutral = (512u16 << 6)
        .to_le_bytes()
        .repeat((width * height / 2) as usize);
    let uv = texture(
        &gpu,
        width / 2,
        height / 2,
        gpu_format.uv_format(),
        &neutral,
        width * 2,
    )
    .create_view(&Default::default());
    for code in [512u16, 513] {
        let luma = (code << 6).to_le_bytes().repeat((width * height) as usize);
        let y = texture(&gpu, width, height, gpu_format.y_format(), &luma, width * 2)
            .create_view(&Default::default());
        session
            .render_frame_imported_nv12(&[(&y, &uv)], RenderPose::default(), Some(i64::from(code)))
            .unwrap();
    }
    session.finish().unwrap();
    let result = captured.lock().unwrap();
    assert_eq!(result.len(), 2);
    let center = ((height / 2 * width + width / 2) * 2) as usize;
    for ((bytes, time), code) in result.iter().zip([512u16, 513]) {
        assert_eq!(*time, Some(i64::from(code)));
        assert_eq!(
            u16::from_le_bytes(bytes[center..center + 2].try_into().unwrap()),
            code << 6
        );
    }
}

#[test]
fn returning_to_the_same_host_image_after_import_does_not_use_imported_pixels() {
    let Some(gpu) = gpu() else { return };
    for sizes in [vec![(64, 32)], vec![(64, 32); 2], vec![(64, 32), (128, 64)]] {
        let (mut session, captured) = session(&gpu, &sizes, InputFormat::Nv12);
        let host = FrameSet::Nv12(
            sizes
                .iter()
                .enumerate()
                .map(|(camera, &size)| image(size, camera, 0))
                .collect(),
        );
        let other: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(camera, &size)| image(size, camera, 3))
            .collect();
        let planes: Vec<_> = other
            .iter()
            .map(|data| views(&gpu, data, GpuPixelFormat::Nv12))
            .collect();
        let refs: Vec<_> = planes.iter().map(|(y, uv)| (y, uv)).collect();
        let pose = RenderPose::default();
        session.render_frame(&host, pose, Some(0)).unwrap();
        session
            .render_frame_imported_nv12(&refs, pose, Some(33_333))
            .unwrap();
        session.render_frame(&host, pose, Some(66_667)).unwrap();
        session.finish().unwrap();
        let output = captured.lock().unwrap();
        assert_eq!(output.len(), 3);
        assert_ne!(output[0].0, output[1].0);
        assert_eq!(output[0].0, output[2].0);
    }
}

#[test]
fn imported_planes_reject_wrong_count_shape_and_format_without_submission() {
    let Some(gpu) = gpu() else { return };
    let (mut session, output) = session(&gpu, &[(64, 32); 2], InputFormat::Nv12);
    let (y, uv) = views(&gpu, &image((64, 32), 0, 0), GpuPixelFormat::Nv12);
    let (wrong_y, wrong_uv) = views(&gpu, &image((128, 16), 0, 0), GpuPixelFormat::Nv12);
    for cameras in [
        vec![],
        vec![(&y, &uv)],
        vec![(&y, &uv); 3],
        vec![(&y, &uv), (&wrong_y, &wrong_uv)],
        vec![(&y, &uv), (&y, &y)],
        vec![(&y, &uv), (&y, &wrong_uv)],
    ] {
        assert!(
            session
                .render_frame_imported_nv12(&cameras, RenderPose::default(), Some(0))
                .is_err()
        );
        assert_eq!(session.frame_count(), 0);
    }
    if gpu
        .device()
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    {
        let (y16, uv16) = views(&gpu, &image((64, 32), 1, 0), GpuPixelFormat::P010);
        for cameras in [[(&y, &uv), (&y16, &uv16)], [(&y16, &uv16), (&y, &uv)]] {
            assert!(
                session
                    .render_frame_imported_nv12(&cameras, RenderPose::default(), None)
                    .is_err()
            );
        }
        assert_eq!(session.frame_count(), 0);
    }
    assert!(output.lock().unwrap().is_empty());
    session
        .render_frame_imported_nv12(&[(&y, &uv); 2], RenderPose::default(), Some(0))
        .unwrap();
    session.finish().unwrap();
    assert_eq!(output.lock().unwrap().len(), 1);
    for format in [InputFormat::Rgba, InputFormat::Yuv420p] {
        let (mut session, _) = self::session(&gpu, &[(64, 32); 2], format);
        assert!(
            session
                .render_frame_imported_nv12(&[(&y, &uv); 2], RenderPose::default(), None)
                .is_err()
        );
    }
    let executor = CpuExecutor::new(
        calibration(2),
        ViewportSize {
            width: 128,
            height: 72,
        },
        vec![(64, 32); 2],
        false,
    )
    .unwrap();
    let mut cpu = StitchSession::with_executor(Executor::Cpu(Box::new(executor))).unwrap();
    assert!(matches!(
        cpu.render_frame_imported_nv12(&[(&y, &uv); 2], RenderPose::default(), None),
        Err(reco_core::session::types::SessionError::Core(
            reco_core::core::types::StitchCoreError::RequiresGpu
        ))
    ));
    assert!(matches!(
        cpu.render_frame_gpu_rgba(y.texture(), y.texture(), RenderPose::default(), None),
        Err(reco_core::session::types::SessionError::Core(
            reco_core::core::types::StitchCoreError::RequiresGpu
        ))
    ));
}

#[test]
fn imported_planes_preserve_stereo_replay_restrictions() {
    use reco_core::gpu::yuv_stack_packer::{OutputTileSize, StackGridLayout};

    let Some(gpu) = gpu() else { return };
    for (sizes, accepted) in [
        (vec![(64, 32)], false),
        (vec![(64, 32); 2], true),
        (vec![(64, 32), (128, 16)], false),
    ] {
        let (mut session, _) = session(&gpu, &sizes, InputFormat::Nv12);
        assert_eq!(
            session
                .enable_gpu_stacked_replay(
                    StackGridLayout::vstack(64, 32, 2).unwrap(),
                    OutputTileSize::unscaled(64, 32),
                )
                .is_ok(),
            accepted
        );
        let planes: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(camera, &size)| views(&gpu, &image(size, camera, 0), GpuPixelFormat::Nv12))
            .collect();
        let refs: Vec<_> = planes.iter().map(|(y, uv)| (y, uv)).collect();
        session
            .render_frame_imported_nv12(&refs, RenderPose::default(), None)
            .unwrap();
        session.finish().unwrap();
    }
}
