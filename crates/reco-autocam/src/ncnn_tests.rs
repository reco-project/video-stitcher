//! NCNN selection at the shared consumer boundary.

use reco_core::detect::{
    DetectionTarget,
    detector::{ChromaPlanes, Detector, DetectorFrame, YuvView},
    panner::Panner,
    tracker::Tracker,
};

use super::*;

struct Target {
    sizes: Vec<(u32, u32)>,
    detector: Option<Box<dyn Detector>>,
}

impl DetectionTarget for Target {
    fn set_detection_filter(&mut self, _: Option<Box<dyn reco_core::detect::DetectionFilter>>) {}
    fn set_detector(&mut self, detector: Box<dyn Detector>) {
        self.detector = Some(detector);
    }
    fn set_detection_interval(&mut self, _: u64) {}
    fn set_ball_tracker(&mut self, _: Box<dyn Tracker>) {}
    fn set_player_tracker(&mut self, _: Box<dyn Tracker>) {}
    fn set_panner(&mut self, _: Box<dyn Panner>) {}
    fn source_sizes(&self) -> &[(u32, u32)] {
        &self.sizes
    }
    fn gpu(&self) -> Option<&reco_core::gpu::GpuContext> {
        None
    }
}

#[test]
fn host_ncnn_supports_mixed_sizes_in_both_camera_orders() {
    let model = tempfile::tempdir().unwrap();
    std::fs::write(
        model.path().join("model.ncnn.param"),
        "7767517\n2 2\nInput input 0 1 in0\nMemoryData output 0 1 out0 0=1 1=5\n",
    )
    .unwrap();
    let weights: Vec<u8> = [320.0_f32, 320.0, 40.0, 40.0, 0.9]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    std::fs::write(model.path().join("model.ncnn.bin"), weights).unwrap();
    for sizes in [vec![(6, 4), (4, 6)], vec![(4, 6), (6, 4)]] {
        let mut target = Target {
            sizes,
            detector: None,
        };
        let config = AutocamConfig::new(model.path());
        assert!(setup_autocam(&mut target, &config, 30.0, false).unwrap());
        let detector = target.detector.as_mut().unwrap();
        assert_eq!(detector.name(), "ncnn");
        for (camera, &(width, height)) in target.sizes.iter().enumerate() {
            let detections = detector
                .detect(
                    camera,
                    &DetectorFrame::Yuv(YuvView {
                        y: &[100; 24],
                        chroma: ChromaPlanes::Nv12 { uv: &[128; 12] },
                        width,
                        height,
                    }),
                )
                .unwrap();
            assert_eq!(detections.len(), 1);
            assert_eq!(detections[0].camera, camera);
            assert!((detections[0].center_x - 0.5).abs() < 0.01);
            assert!((detections[0].center_y - 0.5).abs() < 0.01);
        }
    }
}

#[test]
fn gpu_resident_ncnn_fails_setup_instead_of_silently_skipping_detection() {
    let model = tempfile::tempdir().unwrap();
    let mut target = Target {
        sizes: vec![(6, 4); 2],
        detector: None,
    };
    let error =
        setup_autocam(&mut target, &AutocamConfig::new(model.path()), 30.0, true).unwrap_err();
    assert!(error.to_string().contains("NCNN requires host YUV"));
    assert!(target.detector.is_none());
}
