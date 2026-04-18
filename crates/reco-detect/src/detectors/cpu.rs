//! YOLO ONNX detector for ball detection on raw camera frames.
//!
//! Runs a YOLOv26n model exported with end-to-end NMS (output shape `[1, 300, 6]`).
//! Pre-processing converts YUV camera frames to RGB, letterboxes to the model's
//! input size, and normalizes to `[0, 1]`. Post-processing maps detections back
//! to normalized camera coordinates.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use reco_core::detector::{
    CameraId, ChromaFormat, Detection, DetectorError, DetectorFrame, RawFrame, UnifiedDetector,
};

use super::postprocess;

/// YOLO-based object detector using ONNX Runtime on CPU.
///
/// Loads an exported YOLO model (`.onnx`) and runs inference on raw camera
/// frames. The model must have end-to-end NMS built in (output shape
/// `[1, N, 6]` where each detection is `[x1, y1, x2, y2, confidence, class_id]`
/// in pixel coordinates).
pub struct CpuYoloDetector {
    session: Session,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
}

impl CpuYoloDetector {
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
        let (session, input_size, labels) =
            crate::ort_session::create_ort_session(path.as_ref(), labels)?;

        log::info!(
            "CpuYoloDetector loaded: input={}x{}, {} classes, conf_thresh={}",
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

impl CpuYoloDetector {
    /// Core inference path shared by the legacy [`Detector`] impl and
    /// the [`UnifiedDetector`] impl. Returns a typed
    /// [`DetectorError`] so the unified-trait consumer can react to
    /// failure; the legacy impl collapses the error to a log + empty
    /// vec for backwards compatibility.
    fn detect_raw(
        &mut self,
        camera: CameraId,
        frame: &RawFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        reco_core::profile_scope!("yolo_detect");

        let (rgb_chw, scale, pad_x, pad_y) = self.preprocess(frame);

        let sz = self.input_size as usize;
        let input_tensor = Tensor::from_array(([1, 3, sz, sz], rgb_chw))
            .map_err(|e| DetectorError::InferenceFailed(format!("tensor build: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .map_err(|e| DetectorError::InferenceFailed(format!("ort run: {e}")))?;

        let (n, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map(|(shape, slice)| (shape[1] as usize, slice.to_vec()))
            .map_err(|e| DetectorError::InferenceFailed(format!("output extract: {e}")))?;
        drop(outputs);

        let detections = postprocess(
            &data,
            n,
            camera,
            self.confidence_threshold,
            scale,
            pad_x,
            pad_y,
            frame.width,
            frame.height,
        );

        if !detections.is_empty() {
            log::debug!(
                "camera {:?}: {} detection(s) - {}",
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

        Ok(detections)
    }
}

impl UnifiedDetector for CpuYoloDetector {
    fn name(&self) -> &'static str {
        "ort-cpu"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        // `DetectorFrame` is `#[non_exhaustive]`, so we pattern-match
        // the variants we can handle and catch everything else (today:
        // CUDA on Linux/Windows, Metal on macOS/iOS; tomorrow: any
        // new residency variant) with a single wildcard arm that
        // returns a typed "not my frame" error. A future `StitchCore`
        // dispatch layer routes the frame to a GPU backend instead of
        // silently producing zero detections.
        match frame {
            DetectorFrame::Cpu(raw) => self.detect_raw(camera, raw),
            _ => Err(DetectorError::UnsupportedFrameKind),
        }
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
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
    parse_names_dict_string(&names_str)
}

/// Maximum class count accepted when building a dense label vector.
///
/// N-C1 (deep-review-2026-04-18): the ONNX `names` metadata is
/// attacker-controlled (user-supplied model file). A crafted entry
/// like `{9999999999: 'ball'}` would drive a multi-GB
/// `Vec::with_capacity` and a matching loop. Cap the dense index at
/// a generous ceiling - realistic models top out at ~1200 classes
/// (LVIS); 10_000 leaves comfortable headroom while rejecting the
/// OOM primitive.
const MAX_CLASS_COUNT: usize = 10_000;

/// Fuzz entry point: drives [`parse_names_dict_string`] without
/// requiring a real ONNX session. See `fuzz/fuzz_targets/onnx_names.rs`
/// and the N-C1 OOM cap fix. `__` prefix + `doc(hidden)` keeps this
/// out of the normal public API.
#[doc(hidden)]
pub fn __fuzz_parse_names_dict_string(names_str: &str) -> Option<Vec<String>> {
    parse_names_dict_string(names_str)
}

/// Pure string parser for Ultralytics-style `names` metadata.
///
/// Extracted from [`parse_onnx_names`] so the OOM guard can be
/// exercised without spinning up an ONNX session.
fn parse_names_dict_string(names_str: &str) -> Option<Vec<String>> {
    // Parse Python-dict-style string: {0: 'person', 1: 'bicycle', ...}
    let inner = names_str.trim().strip_prefix('{')?.strip_suffix('}')?;
    if inner.is_empty() {
        return None;
    }

    let mut labels: Vec<(usize, String)> = inner
        .split(',')
        .filter_map(|entry| {
            let entry = entry.trim();
            let (idx_str, name) = entry.split_once(':')?;
            let idx: usize = idx_str.trim().parse().ok()?;
            let name = name.trim().trim_matches('\'').trim_matches('"').to_string();
            Some((idx, name))
        })
        .collect();

    labels.sort_by_key(|(idx, _)| *idx);

    let max_idx = labels.last()?.0;
    if max_idx >= MAX_CLASS_COUNT {
        log::warn!(
            "parse_onnx_names: max class index {max_idx} exceeds cap {MAX_CLASS_COUNT}; \
             refusing to build dense label vector (possible malicious model metadata)"
        );
        return None;
    }
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

    #[test]
    fn parse_names_dict_string_happy_path() {
        let input = "{0: 'person', 1: 'bicycle', 2: 'car'}";
        let labels = parse_names_dict_string(input).unwrap();
        assert_eq!(labels, vec!["person", "bicycle", "car"]);
    }

    #[test]
    fn parse_names_dict_string_fills_gaps() {
        let input = "{0: 'ball', 3: 'goal'}";
        let labels = parse_names_dict_string(input).unwrap();
        assert_eq!(labels, vec!["ball", "class_1", "class_2", "goal"]);
    }

    #[test]
    fn parse_names_dict_string_rejects_oom_index() {
        // N-C1 regression: attacker-crafted ONNX metadata with an
        // enormous class index would drive Vec::with_capacity(idx+1)
        // into a multi-GB allocation and a billion-iteration loop.
        // Guard rejects the dense build instead of allocating.
        let input = "{9999999999: 'ball'}";
        assert!(
            parse_names_dict_string(input).is_none(),
            "must refuse huge class index"
        );
    }

    #[test]
    fn parse_names_dict_string_rejects_index_at_cap() {
        // Exact boundary: MAX_CLASS_COUNT itself is the rejection point
        // (cap is exclusive). Anything >= 10_000 refused.
        let input = format!("{{{MAX_CLASS_COUNT}: 'class'}}");
        assert!(parse_names_dict_string(&input).is_none());
    }

    #[test]
    fn parse_names_dict_string_accepts_just_below_cap() {
        let just_below = MAX_CLASS_COUNT - 1;
        let input = format!("{{0: 'a', {just_below}: 'b'}}");
        let labels = parse_names_dict_string(&input).unwrap();
        assert_eq!(labels.len(), MAX_CLASS_COUNT);
        assert_eq!(labels[0], "a");
        assert_eq!(labels[just_below], "b");
    }

    #[test]
    fn parse_names_dict_string_rejects_empty() {
        assert!(parse_names_dict_string("{}").is_none());
    }

    #[test]
    fn parse_names_dict_string_rejects_non_dict() {
        assert!(parse_names_dict_string("[0: 'x']").is_none());
        assert!(parse_names_dict_string("random garbage").is_none());
    }

    /// Compile-time: `CpuYoloDetector` must satisfy the `UnifiedDetector`
    /// bounds so a future `StitchCore::set_detector` signature taking
    /// `Box<dyn UnifiedDetector>` accepts it. This catches a regression
    /// where a field accidentally becomes non-`Send` (shared mutable
    /// state, raw pointer, etc.).
    #[test]
    fn cpu_yolo_detector_is_unified_detector_send() {
        fn assert_unified_send<T: UnifiedDetector + Send + 'static>() {}
        assert_unified_send::<CpuYoloDetector>();
    }
}
