//! YOLO ONNX detector for ball detection on raw camera frames.
//!
//! Runs a YOLOv26n model exported with end-to-end NMS (output shape `[1, 300, 6]`).
//! Pre-processing converts YUV camera frames to RGB, letterboxes to the model's
//! input size, and normalizes to `[0, 1]`. Post-processing maps detections back
//! to normalized camera coordinates.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use reco_core::detector::{CameraId, ChromaFormat, Detection, Detector, RawFrame};

/// YOLO-based object detector using ONNX Runtime.
///
/// Loads an exported YOLO model (`.onnx`) and runs inference on raw camera
/// frames. The model must have end-to-end NMS built in (output shape
/// `[1, N, 6]` where each detection is `[x1, y1, x2, y2, confidence, class_id]`
/// in pixel coordinates).
pub struct YoloDetector {
    session: Session,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
}

impl YoloDetector {
    /// Load a YOLO ONNX model from a file path.
    ///
    /// The model is expected to take `[1, 3, H, W]` float32 input and produce
    /// `[1, N, 6]` float32 output with built-in NMS.
    ///
    /// Class labels are auto-detected from the ONNX model's `names` metadata
    /// (standard for Ultralytics exports). Falls back to `["ball"]` if not found.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ort::Error> {
        Self::with_config(path, 0.10, Vec::new())
    }

    /// Load a YOLO ONNX model with custom confidence threshold and class labels.
    ///
    /// If `labels` is empty, labels are auto-detected from the model's `names`
    /// metadata field (Ultralytics convention).
    pub fn with_config(
        path: impl AsRef<Path>,
        confidence_threshold: f32,
        labels: Vec<String>,
    ) -> Result<Self, ort::Error> {
        let (session, input_size, labels) = crate::create_ort_session(path.as_ref(), labels)?;

        log::info!(
            "YoloDetector loaded: input={}x{}, {} classes, conf_thresh={}",
            input_size,
            input_size,
            labels.len(),
            confidence_threshold,
        );

        Ok(Self {
            session,
            input_size,
            confidence_threshold,
            labels,
        })
    }

    /// Class names from the model (index = class_id, value = label string).
    ///
    /// Consumers (directors, setup code) use this to resolve a
    /// [`Detection::class_id`] to a human-readable label, or to find the
    /// class ID for a given label name.
    pub fn class_names(&self) -> &[String] {
        &self.labels
    }

    /// Convert a raw YUV frame to a flat RGB float32 buffer in CHW layout,
    /// letterboxed to `input_size x input_size`, normalized to `[0, 1]`.
    ///
    /// Returns `(rgb_chw, scale, pad_x, pad_y)` where scale and padding are
    /// needed to map detection coordinates back to the original frame.
    fn preprocess(&self, frame: &RawFrame<'_>) -> (Vec<f32>, f32, f32, f32) {
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        let is = self.input_size as f32;

        // Letterbox: scale to fit, then pad.
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (self.input_size - new_w) as f32 / 2.0;
        let pad_y = (self.input_size - new_h) as f32 / 2.0;

        let sz = self.input_size as usize;
        let mut rgb_chw = vec![114.0 / 255.0_f32; 3 * sz * sz]; // grey fill

        // For each pixel in the letterboxed region, sample from the source frame
        // with nearest-neighbor, convert YUV->RGB inline.
        let pad_x_i = pad_x as u32;
        let pad_y_i = pad_y as u32;

        for dy in 0..new_h {
            for dx in 0..new_w {
                // Map back to source coordinates.
                let sx = ((dx as f32) / scale) as u32;
                let sy = ((dy as f32) / scale) as u32;
                let sx = sx.min(frame.width - 1);
                let sy = sy.min(frame.height - 1);

                let y_val = frame.y[(sy * frame.width + sx) as usize] as f32;
                let (u_val, v_val) = chroma_sample(frame, sx, sy);

                // BT.709 full-range YUV -> RGB (matches fisheye.wgsl)
                let r = (y_val + 1.5748 * (v_val - 128.0)).clamp(0.0, 255.0);
                let g =
                    (y_val - 0.1873 * (u_val - 128.0) - 0.4681 * (v_val - 128.0)).clamp(0.0, 255.0);
                let b = (y_val + 1.8556 * (u_val - 128.0)).clamp(0.0, 255.0);

                let ox = (pad_x_i + dx) as usize;
                let oy = (pad_y_i + dy) as usize;

                let plane = sz * sz;
                rgb_chw[oy * sz + ox] = r / 255.0;
                rgb_chw[plane + oy * sz + ox] = g / 255.0;
                rgb_chw[2 * plane + oy * sz + ox] = b / 255.0;
            }
        }

        (rgb_chw, scale, pad_x, pad_y)
    }

    /// Parse model output into detections. Delegates to [`postprocess`].
    #[allow(clippy::too_many_arguments)]
    fn postprocess(
        &self,
        data: &[f32],
        n: usize,
        camera: CameraId,
        scale: f32,
        pad_x: f32,
        pad_y: f32,
        frame_width: u32,
        frame_height: u32,
    ) -> Vec<Detection> {
        postprocess(
            data,
            n,
            camera,
            self.confidence_threshold,
            scale,
            pad_x,
            pad_y,
            frame_width,
            frame_height,
        )
    }
}

/// Sample chroma (U, V) values at a given pixel position.
fn chroma_sample(frame: &RawFrame<'_>, x: u32, y: u32) -> (f32, f32) {
    let cx = (x / 2) as usize;
    let cy = (y / 2) as usize;
    let cw = (frame.width / 2) as usize;

    match &frame.chroma {
        ChromaFormat::Yuv420p { u, v } => {
            let idx = cy * cw + cx;
            (u[idx] as f32, v[idx] as f32)
        }
        ChromaFormat::Nv12 { uv } => {
            // Interleaved: U at even indices, V at odd indices.
            let idx = cy * (frame.width as usize) + cx * 2;
            (uv[idx] as f32, uv[idx + 1] as f32)
        }
    }
}

/// Parse YOLO end-to-end NMS output into detections.
///
/// Shared between CPU ([`YoloDetector`]) and GPU
/// ([`GpuYoloDetector`](super::gpu_detector::GpuYoloDetector)) detectors.
///
/// `data` is a flat `[1, N, 6]` tensor where each row is
/// `[x1, y1, x2, y2, confidence, class_id]` in letterboxed pixel coordinates.
/// Returns detections in normalized `[0, 1]` frame coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn postprocess(
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
        let class_id = data[base + 5] as usize;

        // Map from letterboxed pixel coords to original frame coords.
        let orig_x1 = (x1 - pad_x) / scale;
        let orig_y1 = (y1 - pad_y) / scale;
        let orig_x2 = (x2 - pad_x) / scale;
        let orig_y2 = (y2 - pad_y) / scale;

        // Convert to normalized center + size.
        let cx = ((orig_x1 + orig_x2) / 2.0) / frame_width as f32;
        let cy = ((orig_y1 + orig_y2) / 2.0) / frame_height as f32;
        let w = (orig_x2 - orig_x1) / frame_width as f32;
        let h = (orig_y2 - orig_y1) / frame_height as f32;

        // Skip out-of-bounds detections.
        if !(0.0..=1.0).contains(&cx) || !(0.0..=1.0).contains(&cy) {
            continue;
        }

        detections.push(Detection {
            camera,
            class_id: class_id as u16,
            confidence: conf,
            center_x: cx.clamp(0.0, 1.0),
            center_y: cy.clamp(0.0, 1.0),
            width: w.clamp(0.0, 1.0),
            height: h.clamp(0.0, 1.0),
        });
    }

    detections
}

impl Detector for YoloDetector {
    fn detect(&mut self, camera: CameraId, frame: &RawFrame<'_>) -> Vec<Detection> {
        reco_core::profile_scope!("yolo_detect");

        let (rgb_chw, scale, pad_x, pad_y) = self.preprocess(frame);

        let sz = self.input_size as usize;
        let input_tensor = match Tensor::from_array(([1, 3, sz, sz], rgb_chw)) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Failed to create input tensor: {e}");
                return Vec::new();
            }
        };

        let outputs = match self.session.run(ort::inputs![input_tensor]) {
            Ok(o) => o,
            Err(e) => {
                log::error!("YOLO inference failed: {e}");
                return Vec::new();
            }
        };

        let (n, data) = match outputs[0].try_extract_tensor::<f32>() {
            Ok((shape, slice)) => {
                // Output shape: [1, N, 6]. Copy to owned vec to release borrow.
                (shape[1] as usize, slice.to_vec())
            }
            Err(e) => {
                log::error!("Failed to extract YOLO output tensor: {e}");
                return Vec::new();
            }
        };
        drop(outputs);

        let detections = self.postprocess(
            &data,
            n,
            camera,
            scale,
            pad_x,
            pad_y,
            frame.width,
            frame.height,
        );

        if !detections.is_empty() {
            log::debug!(
                "camera {:?}: {} detection(s) — {}",
                camera,
                detections.len(),
                detections
                    .iter()
                    .map(|d| {
                        let name = self
                            .labels
                            .get(d.class_id as usize)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!(
                            "{}({:.0}%@{:.2},{:.2})",
                            name,
                            d.confidence * 100.0,
                            d.center_x,
                            d.center_y
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        detections
    }
}

/// Parse class labels from ONNX model's `names` metadata field.
///
/// Ultralytics exports include a `names` field like:
/// `{0: 'person', 1: 'bicycle', 2: 'car', ...}`
///
/// Returns `None` if the metadata is missing or unparseable.
pub(crate) fn parse_onnx_names(session: &Session) -> Option<Vec<String>> {
    let metadata = session.metadata().ok()?;
    let names_str = metadata.custom("names")?;

    // Parse Python-dict-style string: {0: 'person', 1: 'bicycle', ...}
    let inner = names_str.trim().strip_prefix('{')?.strip_suffix('}')?;
    if inner.is_empty() {
        return None;
    }

    let mut labels: Vec<(usize, String)> = Vec::new();
    for entry in inner.split(',') {
        let entry = entry.trim();
        let (idx_str, name) = entry.split_once(':')?;
        let idx: usize = idx_str.trim().parse().ok()?;
        let name = name.trim().trim_matches('\'').trim_matches('"').to_string();
        labels.push((idx, name));
    }

    labels.sort_by_key(|(idx, _)| *idx);

    // Build a dense label vector (fill gaps with "class_N").
    let max_idx = labels.last()?.0;
    let mut result = Vec::with_capacity(max_idx + 1);
    let mut label_iter = labels.into_iter().peekable();
    for i in 0..=max_idx {
        if label_iter.peek().is_some_and(|(idx, _)| *idx == i) {
            result.push(label_iter.next().unwrap().1);
        } else {
            result.push(format!("class_{i}"));
        }
    }

    log::info!(
        "Auto-detected {} class labels from model metadata",
        result.len()
    );
    Some(result)
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
    // With model input 640: scale = min(640/1920, 640/1080) = 0.333..
    // new_w = 640, new_h = 360. pad_x = 0, pad_y = 140.
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

    // ---- Valid detection output ----

    #[test]
    fn postprocess_valid_detection() {
        // Place a detection roughly in the center of the letterboxed image.
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
        // center_x and center_y should be in [0, 1].
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
        assert_eq!(dets[0].class_id, 0); // "ball"
        assert_eq!(dets[1].class_id, 1); // "player"
    }

    // ---- Empty output ----

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

    // ---- Confidence threshold filtering ----

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
        // Confidence exactly at the threshold: conf (0.10) is NOT < 0.10,
        // so the detection passes the filter.
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

    // ---- Out-of-bounds detection filtering ----

    #[test]
    fn postprocess_filters_out_of_bounds_detection() {
        // Place detection far outside the frame (negative after un-letterbox).
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

    // ---- Undersized buffer bounds check ----

    #[test]
    fn postprocess_empty_n_with_short_buffer() {
        // Buffer is too short for any full row, but n=0 so no iteration.
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

    // ---- Class ID propagation ----

    #[test]
    fn postprocess_preserves_class_id() {
        // class_id = 5 should be stored as-is in the Detection.
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

    // ---- Coordinate mapping accuracy ----

    #[test]
    fn postprocess_center_maps_to_half() {
        // Place detection at the exact center of the content area.
        // Content center in letterboxed coords:
        //   x = pad_x + new_w/2 = 0 + 320 = 320
        //   y = pad_y + new_h/2 = 140 + 180 = 320
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

    // ---- Camera ID propagation ----

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
