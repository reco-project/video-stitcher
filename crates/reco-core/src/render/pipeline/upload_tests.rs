//! Host-upload identity, borrowed-plane invalidation and pixel ownership.

use super::*;
use crate::{
    calibration::{Framing, Lens},
    gpu::rgba_readback::RgbaReadback,
    projection::LShape,
    source::FrameSet,
};

fn pipeline(gpu: &GpuContext, format: InputFormat) -> StitchPipeline {
    let calibration = Calibration::new(
        vec![Lens::flat(64, 32); 2],
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
    StitchPipeline::with_gpu(
        gpu.clone(),
        &calibration.projection.geometry().gpu_program().unwrap(),
        calibration,
        ViewportSize {
            width: 64,
            height: 32,
        },
        vec![(64, 32); 2],
        wgpu::TextureFormat::Rgba8Unorm,
        format,
    )
    .unwrap()
}

fn frames(format: InputFormat, value: u8) -> FrameSet {
    if format == InputFormat::Yuv420p {
        FrameSet::Yuv420p(
            (0..2)
                .map(|_| {
                    Arc::new(Yuv420pData {
                        size: (64, 32),
                        y: vec![value; 64 * 32],
                        u: vec![128; 32 * 16],
                        v: vec![128; 32 * 16],
                    })
                })
                .collect(),
        )
    } else {
        FrameSet::Nv12(
            (0..2)
                .map(|_| {
                    Arc::new(Nv12Data {
                        size: (64, 32),
                        y: vec![value; 64 * 32],
                        uv: vec![128; 64 * 16],
                    })
                })
                .collect(),
        )
    }
}

fn pixels(pipeline: &StitchPipeline, commands: wgpu::CommandBuffer) -> Vec<u8> {
    let gpu = pipeline.gpu();
    let mut readback = RgbaReadback::new(gpu, 64, 32).unwrap();
    readback
        .readback(gpu, pipeline.render_target(), commands)
        .unwrap();
    readback.flush_pending(gpu).unwrap().unwrap().to_vec()
}

fn render(pipeline: &mut StitchPipeline, frames: &FrameSet) -> Vec<u8> {
    let commands = pipeline
        .render_frame_set(frames, RenderPose::default())
        .unwrap();
    pixels(pipeline, commands)
}

#[test]
fn repeated_host_frames_remain_correct_after_mutation_and_borrowed_uploads() {
    let gpu = match GpuContext::new_blocking() {
        Ok(gpu) => gpu,
        Err(error) if std::env::var_os("RECO_REQUIRE_GPU").is_none() => {
            eprintln!("GPU unavailable: {error}");
            return;
        }
        Err(error) => panic!("GPU required: {error}"),
    };
    eprintln!("host-upload test: {}", gpu.gpu_name());
    for format in [InputFormat::Yuv420p, InputFormat::Nv12] {
        let mut pipeline = pipeline(&gpu, format);
        let mut input = frames(format, 90);
        assert_eq!(pipeline.upload_frame_set(&input).unwrap(), 2);
        assert_eq!(pipeline.upload_frame_set(&input).unwrap(), 0);
        let original = render(&mut pipeline, &input);
        assert!(original.chunks_exact(4).any(|rgba| rgba[0] > 16));
        assert_eq!(pipeline.upload_frame_set(&input).unwrap(), 0);
        assert_eq!(render(&mut pipeline, &input), original);
        if let FrameSet::Nv12(cameras) = frames(format, 180) {
            let pose = RenderPose::default();
            let left = cameras[0].as_planes();
            let right = cameras[1].as_planes();
            let commands = pipeline.render_nv12(&[left, right], pose).unwrap();
            assert!(pipeline.uploaded.iter().all(Option::is_none));
            assert_ne!(pixels(&pipeline, commands), original);
            assert_eq!(pipeline.upload_frame_set(&input).unwrap(), 2);
            assert_eq!(render(&mut pipeline, &input), original);

            let invalid = Nv12Planes { uv: &[], ..right };
            assert!(pipeline.render_nv12(&[left, invalid], pose).is_err());
            assert_eq!(
                render(&mut pipeline, &input),
                original,
                "restore even a partially overwritten upload"
            );
        }

        let mut invalid = frames(format, 180);
        match &mut invalid {
            FrameSet::Yuv420p(cameras) => {
                Arc::make_mut(&mut cameras[1]).y.clear();
            }
            FrameSet::Nv12(cameras) => {
                Arc::make_mut(&mut cameras[1]).y.clear();
            }
            _ => unreachable!(),
        }
        assert!(
            pipeline
                .render_frame_set(&invalid, RenderPose::default())
                .is_err()
        );
        assert_eq!(pipeline.upload_frame_set(&input).unwrap(), 0);
        assert_eq!(render(&mut pipeline, &input), original);

        match &mut input {
            FrameSet::Yuv420p(cameras) => {
                assert!(Arc::get_mut(&mut cameras[1]).is_none());
                Arc::make_mut(&mut cameras[1]).y.fill(160);
            }
            FrameSet::Nv12(cameras) => {
                assert!(Arc::get_mut(&mut cameras[1]).is_none());
                Arc::make_mut(&mut cameras[1]).y.fill(160);
            }
            _ => unreachable!(),
        }
        assert_eq!(
            pipeline.upload_frame_set(&input).unwrap(),
            1,
            "only the copied-on-write camera should re-upload"
        );
        let changed = render(&mut pipeline, &input);
        assert_ne!(
            changed, original,
            "changed pixels must upload after copy-on-write"
        );
        assert_eq!(pipeline.upload_frame_set(&input).unwrap(), 0);
        assert_eq!(render(&mut pipeline, &input), changed);

        // Identity must not prolong the lifetime of the decoded pixel vectors.
        match &input {
            FrameSet::Yuv420p(cameras) => {
                assert!(cameras.iter().all(|c| Arc::strong_count(c) == 1))
            }
            FrameSet::Nv12(cameras) => assert!(cameras.iter().all(|c| Arc::strong_count(c) == 1)),
            _ => unreachable!(),
        }
        drop(input);
        assert!(pipeline.uploaded.iter().all(|cached| match cached {
            Some(UploadedImage::Yuv420p(weak)) => weak.upgrade().is_none(),
            Some(UploadedImage::Nv12(weak)) => weak.upgrade().is_none(),
            None => false,
        }));
        let fresh = frames(format, 90);
        assert_eq!(pipeline.upload_frame_set(&fresh).unwrap(), 2);
        assert_eq!(render(&mut pipeline, &fresh), original);
    }
}

#[test]
fn sampled_nv12_and_p010_match_independent_color_vectors() {
    use crate::render::color::tests::{P010_VECTORS, RGB_PATCHES, vectors};
    use crate::render::color::{YuvColor, YuvMatrix};
    use crate::stitch::test_support::gpu_or_skip;

    let Some(gpu) = gpu_or_skip() else { return };
    let mut pipeline = pipeline(&gpu, InputFormat::Nv12);
    let assert_center = |rgba: &[u8], expected: [u8; 3]| {
        let center = (16 * 64 + 32) * 4;
        let rgb = &rgba[center..center + 3];
        assert!(
            rgb.iter().zip(expected).all(|(&a, b)| a.abs_diff(b) <= 2),
            "{rgb:?} vs {expected:?}"
        );
    };
    for (color, codes) in vectors() {
        pipeline.set_source_color(color);
        for ([y, cb, cr], rgb) in codes.into_iter().zip(RGB_PATCHES) {
            let y = [y; 64 * 32];
            let uv = [cb, cr].repeat(32 * 16);
            let input = Nv12Planes { y: &y, uv: &uv };
            let commands = pipeline
                .render_nv12(&[input; 2], RenderPose::default())
                .unwrap();
            assert_center(&pixels(&pipeline, commands), rgb);
        }
    }

    if !gpu
        .device()
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    {
        return;
    }
    let make_plane = |format, width, height| {
        gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: None,
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
        })
    };
    let y_plane = make_plane(wgpu::TextureFormat::R16Unorm, 64, 32);
    let uv_plane = make_plane(wgpu::TextureFormat::Rg16Unorm, 32, 16);
    let y_view = y_plane.create_view(&Default::default());
    let uv_view = uv_plane.create_view(&Default::default());
    let upload = |texture: &wgpu::Texture, data: &[u16]| {
        gpu.queue().write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: None,
            },
            texture.size(),
        );
    };
    for (full_range, [y, cb, cr], rgb) in P010_VECTORS {
        pipeline.set_source_color(YuvColor {
            matrix: YuvMatrix::Bt2020Ncl,
            full_range,
            ..Default::default()
        });
        upload(&y_plane, &[y << 6; 64 * 32]);
        upload(&uv_plane, &[cb << 6, cr << 6].repeat(32 * 16));
        let commands = pipeline
            .render_imported_views(&[(&y_view, &uv_view); 2], RenderPose::default())
            .unwrap();
        assert_center(&pixels(&pipeline, commands), rgb);
    }
}
