//! Camera admission and model-input handoff, without loading an inference runtime.

use super::*;
use reco_core::gpu::GpuContext;
use std::sync::{Arc, Mutex};

struct InputProbe {
    calls: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl Detector for InputProbe {
    fn name(&self) -> &'static str {
        "input-probe"
    }
    fn detect(
        &mut self,
        camera: usize,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let DetectorFrame::PreprocessedChw {
            data,
            input_size,
            src_width,
            src_height,
        } = frame
        else {
            return Err(DetectorError::UnsupportedFrameKind);
        };
        assert_eq!(*input_size, 8);
        assert_eq!(data.len(), 3 * 8 * 8);
        assert_eq!((*src_width, *src_height), [(6, 4), (4, 6)][camera]);
        let (padding, content) = if camera == 0 {
            (6 * 8 + 3, 3 * 8 + 3)
        } else {
            (3 * 8 + 6, 3 * 8 + 3)
        };
        assert!((data[padding] - 114.0 / 255.0).abs() < 1e-6);
        assert!((data[content] - (200 - camera * 100) as f32 / 255.0).abs() < 0.004);
        self.calls
            .lock()
            .unwrap()
            .push((camera, data.as_ptr() as usize));
        Ok(Vec::new())
    }
}

#[test]
fn declared_cameras_share_one_model_tensor_and_invalid_frames_do_not_reach_inference() {
    let gpu = match GpuContext::new_blocking() {
        Ok(gpu) => gpu,
        Err(error) if std::env::var_os("RECO_REQUIRE_GPU").is_none() => {
            eprintln!("GPU unavailable: {error}");
            return;
        }
        Err(error) => panic!("GPU required: {error}"),
    };
    eprintln!("wrapper adapter: {}", gpu.gpu_name());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let sizes = vec![(6, 4), (4, 6)];
    let mut detector = WgpuPreprocessingDetector::new(
        Box::new(InputProbe {
            calls: calls.clone(),
        }),
        gpu.device().clone(),
        gpu.queue().clone(),
        8,
        sizes.clone(),
    )
    .unwrap();
    let planes: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(camera, &(w, h))| {
            [false, true].map(|uv| {
                let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("wrapper fixture"),
                    size: wgpu::Extent3d {
                        width: if uv { w / 2 } else { w },
                        height: if uv { h / 2 } else { h },
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
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                gpu.queue().write_texture(
                    texture.as_image_copy(),
                    &vec![
                        if uv { 128 } else { (200 - camera * 100) as u8 };
                        (w * texture.height()) as usize
                    ],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(w),
                        rows_per_image: None,
                    },
                    texture.size(),
                );
                texture.create_view(&Default::default())
            })
        })
        .collect();
    let frame = |camera: usize, (width, height)| DetectorFrame::WgpuNv12 {
        y_view: &planes[camera][0],
        uv_view: &planes[camera][1],
        width,
        height,
        rotation: 0,
    };
    assert!(detector.detect(2, &frame(0, sizes[0])).is_err());
    assert!(detector.detect(0, &frame(0, sizes[1])).is_err());
    assert!(detector.detect(0, &frame(1, sizes[0])).is_err());
    assert!(calls.lock().unwrap().is_empty());
    for camera in [0, 1, 0] {
        detector
            .detect(camera, &frame(camera, sizes[camera]))
            .unwrap();
    }
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.iter().map(|call| call.0).collect::<Vec<_>>(),
        [0, 1, 0]
    );
    assert!(calls.iter().all(|call| call.1 == calls[0].1));
}
