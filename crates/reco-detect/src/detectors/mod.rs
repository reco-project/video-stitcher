//! Detection backend implementations.
//!
//! Each submodule implements a specific backend (CPU ORT, GPU ORT,
//! Metal, native TensorRT). Shared postprocessing logic lives here.

#[cfg(feature = "ort")]
pub mod cpu;
#[cfg(all(feature = "ort", target_os = "macos"))]
pub mod metal;
#[cfg(feature = "ncnn")]
pub mod ncnn;
#[cfg(all(feature = "ort", any(target_os = "linux", target_os = "windows")))]
pub mod ort_gpu;
#[cfg(feature = "tensorrt-native")]
pub mod trt;

use std::path::Path;

use reco_core::detect::detector::{CameraId, Detection};

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
        // NaN-safe filter: `NaN < x` is always false, so a naive comparison
        // would let NaN confidences through and poison the director's EMA
        // (`f32::clamp` preserves NaN). Reject non-finite confidence first.
        if !conf.is_finite() || conf < confidence_threshold {
            continue;
        }

        let x1 = data[base];
        let y1 = data[base + 1];
        let x2 = data[base + 2];
        let y2 = data[base + 3];
        if !x1.is_finite() || !y1.is_finite() || !x2.is_finite() || !y2.is_finite() {
            continue;
        }
        // `f32 as u16` is saturating in release but UB-adjacent for NaN in
        // some toolchains; reject explicitly.
        let class_id_f = data[base + 5];
        if !class_id_f.is_finite() || !(0.0..=u16::MAX as f32).contains(&class_id_f) {
            continue;
        }
        let class_id = class_id_f as u16;

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

/// Parse a raw seg-YOLO ball-detector output `[1, N, 6]` into detections.
///
/// TEMPORARY adapter for an external single-class ball detector whose
/// export differs from the stock end-to-end-NMS YOLO this codebase
/// assumes:
/// - rows are `[cx, cy, w, h, obj, conf]` - center+size (not corners),
///   confidence in col 5 (not col 4), col 4 is a constant objectness;
/// - the output is PRE-NMS (all N anchors), so we conf-filter then run a
///   greedy IoU NMS;
/// - single class ("ball"), emitted as class id 0 per the model metadata.
///
/// Selected when the ONNX session reports more than one output (this
/// model also emits segmentation heads we ignore). Coordinates are
/// un-letterboxed and normalized to `[0, 1]` exactly as [`postprocess`].
#[allow(clippy::too_many_arguments)]
pub fn postprocess_balldet(
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
    /// Single-class ball model: metadata declares class 0 = "ball".
    const BALL_CLASS_ID: u16 = 0;
    let expected_len = n * 6;
    if data.len() < expected_len {
        log::error!(
            "ball-detector output buffer too small: got {} floats, expected {} ({n} x 6)",
            data.len(),
            expected_len,
        );
        return Vec::new();
    }

    let fw = frame_width as f32;
    let fh = frame_height as f32;
    let mut cands: Vec<Detection> = Vec::new();
    for i in 0..n {
        let base = i * 6;
        let conf = data[base + 5];
        if !conf.is_finite() || conf < confidence_threshold {
            continue;
        }
        let (cx_p, cy_p, w_p, h_p) = (data[base], data[base + 1], data[base + 2], data[base + 3]);
        if !cx_p.is_finite() || !cy_p.is_finite() || !w_p.is_finite() || !h_p.is_finite() {
            continue;
        }
        // Un-letterbox center+size from padded input px to original-frame
        // normalized coords.
        let cx = ((cx_p - pad_x) / scale) / fw;
        let cy = ((cy_p - pad_y) / scale) / fh;
        let w = (w_p / scale) / fw;
        let h = (h_p / scale) / fh;
        if !(0.0..=1.0).contains(&cx) || !(0.0..=1.0).contains(&cy) {
            continue;
        }
        cands.push(Detection {
            camera,
            class_id: BALL_CLASS_ID,
            confidence: conf,
            center_x: cx.clamp(0.0, 1.0),
            center_y: cy.clamp(0.0, 1.0),
            width: w.clamp(0.0, 1.0),
            height: h.clamp(0.0, 1.0),
        });
    }
    greedy_nms(cands, 0.45)
}

/// IoU of two normalized center+size detection boxes.
fn box_iou(a: &Detection, b: &Detection) -> f32 {
    let (ax1, ay1) = (a.center_x - a.width / 2.0, a.center_y - a.height / 2.0);
    let (ax2, ay2) = (a.center_x + a.width / 2.0, a.center_y + a.height / 2.0);
    let (bx1, by1) = (b.center_x - b.width / 2.0, b.center_y - b.height / 2.0);
    let (bx2, by2) = (b.center_x + b.width / 2.0, b.center_y + b.height / 2.0);
    let iw = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let ih = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = iw * ih;
    let union = a.width * a.height + b.width * b.height - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Greedy IoU non-maximum suppression: keep highest-confidence boxes,
/// drop any later box overlapping a kept one beyond `iou_thresh`.
fn greedy_nms(mut dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        if keep.iter().all(|k| box_iou(k, &d) < iou_thresh) {
            keep.push(d);
        }
    }
    keep
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
        Err(_) => {
            log::debug!(
                "No labels file at {}, using defaults",
                path.as_ref().display()
            );
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detect::detector::CameraId;

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
    fn postprocess_rejects_nan_confidence() {
        // B-28 regression: NaN confidence must not pass the threshold check.
        // `NaN < threshold` is false, so without an explicit is_finite guard
        // the row would be kept and NaN would propagate into the director EMA.
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, f32::NAN, 0.0]]);
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
            "NaN confidence must be filtered, got {dets:?}"
        );
    }

    #[test]
    fn postprocess_rejects_infinite_confidence() {
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, f32::INFINITY, 0.0]]);
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
        assert!(dets.is_empty());
    }

    #[test]
    fn postprocess_rejects_nan_bbox_coords() {
        for bad_idx in 0..4 {
            let mut row = [300.0, 300.0, 340.0, 340.0, 0.80, 0.0];
            row[bad_idx] = f32::NAN;
            let data = make_tensor(&[row]);
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
                "NaN at bbox index {bad_idx} must be filtered"
            );
        }
    }

    #[test]
    fn postprocess_rejects_nan_class_id() {
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, 0.80, f32::NAN]]);
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
        assert!(dets.is_empty());
    }

    #[test]
    fn postprocess_rejects_out_of_range_class_id() {
        // Class id that would saturate u16 on cast; reject rather than pretend.
        let data = make_tensor(&[[300.0, 300.0, 340.0, 340.0, 0.80, 1e9]]);
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
        assert!(dets.is_empty());
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

    #[test]
    fn postprocess_balldet_decodes_cxcywh_with_conf_in_col5() {
        // [cx, cy, w, h, obj=1, conf]; scale=1, no pad, 1000x1000 frame.
        let data = vec![500.0, 250.0, 40.0, 40.0, 1.0, 0.90];
        let dets = postprocess_balldet(&data, 1, CameraId::Left, 0.25, 1.0, 0.0, 0.0, 1000, 1000);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 0, "single ball class");
        assert!((dets[0].confidence - 0.90).abs() < 1e-6, "conf from col 5");
        assert!((dets[0].center_x - 0.5).abs() < 1e-3, "cx 500/1000");
        assert!((dets[0].center_y - 0.25).abs() < 1e-3, "cy 250/1000");
        assert!((dets[0].width - 0.04).abs() < 1e-3, "w 40/1000");
    }

    #[test]
    fn postprocess_balldet_conf_filters_and_nms_dedups() {
        let data = vec![
            500.0, 250.0, 40.0, 40.0, 1.0, 0.10, // below 0.25 -> dropped
            500.0, 250.0, 40.0, 40.0, 1.0, 0.90, // kept (highest)
            505.0, 252.0, 40.0, 40.0, 1.0, 0.80, // overlaps -> NMS-dropped
            100.0, 100.0, 30.0, 30.0, 1.0, 0.70, // far -> kept
        ];
        let dets = postprocess_balldet(&data, 4, CameraId::Left, 0.25, 1.0, 0.0, 0.0, 1000, 1000);
        assert_eq!(dets.len(), 2, "low-conf filtered + overlap NMS-deduped");
        assert!(
            (dets[0].confidence - 0.90).abs() < 1e-6,
            "highest-conf first"
        );
    }
}
