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
        let alternate = frames(format, 180);
        let commands = match &alternate {
            FrameSet::Yuv420p(cameras) => pipeline
                .render_to_target(
                    &cameras[0].as_planes(),
                    &cameras[1].as_planes(),
                    RenderPose::default(),
                )
                .unwrap(),
            FrameSet::Nv12(cameras) => pipeline
                .render_nv12(
                    &cameras.iter().map(|c| c.as_planes()).collect::<Vec<_>>(),
                    RenderPose::default(),
                )
                .unwrap(),
            _ => unreachable!(),
        };
        assert!(pipeline.uploaded.iter().all(Option::is_none));
        assert_ne!(pixels(&pipeline, commands), original);
        assert_eq!(pipeline.upload_frame_set(&input).unwrap(), 2);
        assert_eq!(render(&mut pipeline, &input), original);

        let failed = match &alternate {
            FrameSet::Yuv420p(cameras) => {
                let mut invalid = cameras[1].as_planes();
                invalid.u = &[];
                pipeline.render_to_target(&cameras[0].as_planes(), &invalid, RenderPose::default())
            }
            FrameSet::Nv12(cameras) => {
                let mut invalid = cameras[1].as_planes();
                invalid.uv = &[];
                pipeline.render_nv12(&[cameras[0].as_planes(), invalid], RenderPose::default())
            }
            _ => unreachable!(),
        };
        assert!(failed.is_err());
        assert_eq!(
            render(&mut pipeline, &input),
            original,
            "restore even a partially overwritten upload"
        );

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
