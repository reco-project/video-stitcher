//! Detection backend implementations.
//!
//! Each submodule implements a specific backend (CPU ORT, GPU ORT,
//! Metal, native TensorRT). Shared postprocessing logic lives here.

#[cfg(feature = "ort")]
pub mod cpu;
#[cfg(all(feature = "ort", target_os = "macos"))]
pub mod metal;
#[cfg(all(feature = "ort", any(target_os = "linux", target_os = "windows")))]
pub mod ort_gpu;
#[cfg(feature = "tensorrt-native")]
pub mod trt;

use std::path::Path;

use reco_core::detector::{CameraId, Detection};

/// Parse YOLO end-to-end NMS output `[1, N, 6]` into detections.
///
/// Each row is `[x1, y1, x2, y2, confidence, class_id]` in letterboxed
/// pixel coordinates. Coordinates are un-letterboxed and normalized to `[0, 1]`.
///
/// This is the single shared implementation used by all detector backends.
#[allow(clippy::too_many_arguments)]
pub fn postprocess(
    data: &[f32],
    n: usize,
    camera: CameraId,
    confidence_threshold: f32,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    frame_width: u32,
    frame_height: u32,
) -> Vec<Detection> {
    let expected_len = n * 6;
    if data.len() < expected_len {
        log::error!(
            "YOLO output buffer too small: got {} floats, expected {} ({} detections x 6)",
            data.len(),
            expected_len,
            n,
        );
        return Vec::new();
    }

    let mut detections = Vec::new();
    let fw = frame_width as f32;
    let fh = frame_height as f32;

    for i in 0..n {
        let base = i * 6;
        let conf = data[base + 4];
        if conf < confidence_threshold {
            continue;
        }

        let x1 = data[base];
        let y1 = data[base + 1];
        let x2 = data[base + 2];
        let y2 = data[base + 3];
        let class_id = data[base + 5] as u16;

        // Un-letterbox: map from padded input coords to original frame coords.
        let orig_x1 = (x1 - pad_x) / scale;
        let orig_y1 = (y1 - pad_y) / scale;
        let orig_x2 = (x2 - pad_x) / scale;
        let orig_y2 = (y2 - pad_y) / scale;

        // Normalize to [0, 1] center + size.
        let cx = ((orig_x1 + orig_x2) / 2.0) / fw;
        let cy = ((orig_y1 + orig_y2) / 2.0) / fh;
        // Use .abs() for defensive width/height (handles inverted coords from TRT).
        let w = (orig_x2 - orig_x1).abs() / fw;
        let h = (orig_y2 - orig_y1).abs() / fh;

        // Skip if center is outside the frame.
        if !(0.0..=1.0).contains(&cx) || !(0.0..=1.0).contains(&cy) {
            continue;
        }

        detections.push(Detection {
            camera,
            class_id,
            confidence: conf,
            center_x: cx.clamp(0.0, 1.0),
            center_y: cy.clamp(0.0, 1.0),
            width: w.clamp(0.0, 1.0),
            height: h.clamp(0.0, 1.0),
        });
    }

    detections
}

/// Read class labels from a sidecar `.labels` file (one name per line).
///
/// If the file doesn't exist, returns an empty vec. Labels are used
/// for log messages and class ID resolution in directors.
pub fn read_labels_file(path: impl AsRef<Path>) -> Vec<String> {
    match std::fs::read_to_string(path.as_ref()) {
        Ok(contents) => contents
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detector::CameraId;

    /// Build a synthetic [1, N, 6] flat tensor with the given detections.
    /// Each detection is [x1, y1, x2, y2, confidence, class_id].
    fn make_tensor(rows: &[[f32; 6]]) -> Vec<f32> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    // Common test parameters: 640x640 model input, 1920x1080 source frame.
    const FRAME_W: u32 = 1920;
    const FRAME_H: u32 = 1080;
    const MODEL_SIZE: f32 = 640.0;

    fn scale() -> f32 {
        (MODEL_SIZE / FRAME_W as f32).min(MODEL_SIZE / FRAME_H as f32)
    }
    fn pad_x() -> f32 {
        (MODEL_SIZE - (FRAME_W as f32 * scale()).round()) / 2.0
    }
    fn pad_y() -> f32 {
        (MODEL_SIZE - (FRAME_H as f32 * scale()).round()) / 2.0
    }

    #[test]
    fn postprocess_valid_detection() {
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, 0.85, 0.0]]);

        let dets = postprocess(
            &data,
            1,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert_eq!(d.class_id, 0);
        assert!((d.confidence - 0.85).abs() < 1e-6);
        assert!(d.center_x >= 0.0 && d.center_x <= 1.0);
        assert!(d.center_y >= 0.0 && d.center_y <= 1.0);
        assert!(d.width > 0.0);
        assert!(d.height > 0.0);
    }

    #[test]
    fn postprocess_multiple_detections() {
        let data = make_tensor(&[
            [100.0, 200.0, 120.0, 220.0, 0.80, 0.0],
            [400.0, 300.0, 500.0, 400.0, 0.70, 1.0],
        ]);

        let dets = postprocess(
            &data,
            2,
            CameraId::Right,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert_eq!(dets.len(), 2);
        assert_eq!(dets[0].class_id, 0);
        assert_eq!(dets[1].class_id, 1);
    }

    #[test]
    fn postprocess_zero_detections() {
        let data: Vec<f32> = vec![];

        let dets = postprocess(
            &data,
            0,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert!(dets.is_empty());
    }

    #[test]
    fn postprocess_filters_low_confidence() {
        let data = make_tensor(&[
            [300.0, 300.0, 340.0, 340.0, 0.05, 0.0], // below threshold
            [300.0, 300.0, 340.0, 340.0, 0.50, 0.0], // above threshold
        ]);

        let dets = postprocess(
            &data,
            2,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert_eq!(dets.len(), 1);
        assert!((dets[0].confidence - 0.50).abs() < 1e-6);
    }

    #[test]
    fn postprocess_at_exact_threshold_passes() {
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, 0.10, 0.0]]);

        let dets = postprocess(
            &data,
            1,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert_eq!(dets.len(), 1);
    }

    #[test]
    fn postprocess_filters_out_of_bounds_detection() {
        let data = make_tensor(&[[-500.0, -500.0, -400.0, -400.0, 0.90, 0.0]]);

        let dets = postprocess(
            &data,
            1,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert!(
            dets.is_empty(),
            "out-of-bounds detection should be filtered"
        );
    }

    #[test]
    fn postprocess_empty_n_with_short_buffer() {
        let data = vec![1.0, 2.0, 3.0];

        let dets = postprocess(
            &data,
            0,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );
        assert!(dets.is_empty());
    }

    #[test]
    fn postprocess_preserves_class_id() {
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, 0.90, 5.0]]);

        let dets = postprocess(
            &data,
            1,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 5);
    }

    #[test]
    fn postprocess_center_maps_to_half() {
        let cx = pad_x() + (FRAME_W as f32 * scale()) / 2.0;
        let cy = pad_y() + (FRAME_H as f32 * scale()) / 2.0;
        let half = 10.0;
        let data = make_tensor(&[[cx - half, cy - half, cx + half, cy + half, 0.90, 0.0]]);

        let dets = postprocess(
            &data,
            1,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert_eq!(dets.len(), 1);
        assert!(
            (dets[0].center_x - 0.5).abs() < 0.02,
            "center_x should be ~0.5, got {}",
            dets[0].center_x
        );
        assert!(
            (dets[0].center_y - 0.5).abs() < 0.02,
            "center_y should be ~0.5, got {}",
            dets[0].center_y
        );
    }

    #[test]
    fn postprocess_preserves_camera_id() {
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, 0.90, 0.0]]);

        let dets_left = postprocess(
            &data,
            1,
            CameraId::Left,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );
        let dets_right = postprocess(
            &data,
            1,
            CameraId::Right,
            0.10,
            scale(),
            pad_x(),
            pad_y(),
            FRAME_W,
            FRAME_H,
        );

        assert_eq!(dets_left[0].camera, CameraId::Left);
        assert_eq!(dets_right[0].camera, CameraId::Right);
    }
}
