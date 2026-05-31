//! YOLO ONNX detector for ball detection on raw camera frames.
//!
//! Runs a YOLOv26n model exported with end-to-end NMS (output shape `[1, 300, 6]`).
//! Pre-processing converts YUV camera frames to RGB, letterboxes to the model's
//! input size, and normalizes to `[0, 1]`. Post-processing maps detections back
//! to normalized camera coordinates.
//!
//! ## Canonical preprocessing spec (all backends must match)
//!
//! - **Color**: BT.601 full-range YUV -> RGB (matches JPEG/OpenCV
//!   training pipeline and NPP's `nppiNV12ToRGB`)
//! - **Resize**: bilinear interpolation
//! - **Letterbox**: grey fill at `rgb(114, 114, 114)`, centered
//! - **Normalize**: divide by 255.0 to `[0, 1]`
//! - **Layout**: CHW float32, `[1, 3, H, W]`

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;
use reco_core::detect::detector::{
    CameraId, ChromaFormat, Detection, DetectorError, DetectorFrame, RawFrame, UnifiedDetector,
};

use super::{postprocess, postprocess_balldet};

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
    /// Pre-allocated preprocess scratch for `3 * input_size * input_size`
    /// f32 (4.9 MB at sz=640). Fresh allocation per frame was one of the
    /// M7 plan §M7.5 hotspots; this buffer is filled in place and handed
    /// to `TensorRef::from_array_view` without moving ownership.
    rgb_chw_buf: Vec<f32>,
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

        let sz = input_size as usize;
        let rgb_chw_buf = vec![114.0 / 255.0_f32; 3 * sz * sz];

        Ok(Self {
            session,
            input_size,
            confidence_threshold,
            labels,
            rgb_chw_buf,
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

    /// Model input size (square dimension, e.g. 640 or 1280).
    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    /// Fill `self.rgb_chw_buf` from a raw YUV frame: flat RGB float32 in
    /// CHW layout, letterboxed to `input_size x input_size`, normalized
    /// to `[0, 1]`. The buffer is reused across frames so no allocation
    /// happens per call.
    ///
    /// Returns `(scale, pad_x, pad_y)` for mapping detection coordinates
    /// back to the original frame.
    fn preprocess(&mut self, frame: &RawFrame<'_>) -> (f32, f32, f32) {
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        let is = self.input_size as f32;

        // Letterbox: scale to fit, then pad.
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (self.input_size - new_w) as f32 / 2.0;
        let pad_y = (self.input_size - new_h) as f32 / 2.0;

        let sz = self.input_size as usize;

        // Refill grey pad. Resetting the whole buffer is cheaper than
        // tracking which pixels were last written — at 640×640×3 f32
        // this is a single ~5 MB contiguous write ~2× / sec at 30 fps
        // on the detection interval, not per-frame.
        let grey = 114.0 / 255.0_f32;
        self.rgb_chw_buf.fill(grey);

        let pad_x_i = pad_x as u32;
        let pad_y_i = pad_y as u32;
        let plane = sz * sz;

        let w_max = frame.width - 1;
        let h_max = frame.height - 1;

        for dy in 0..new_h {
            for dx in 0..new_w {
                // Bilinear interpolation: map destination pixel to
                // fractional source coordinates, sample 4 neighbors.
                let src_x = (dx as f32) / scale;
                let src_y = (dy as f32) / scale;
                let x0 = (src_x.floor() as u32).min(w_max);
                let y0 = (src_y.floor() as u32).min(h_max);
                let x1 = (x0 + 1).min(w_max);
                let y1 = (y0 + 1).min(h_max);
                let fx = src_x - src_x.floor();
                let fy = src_y - src_y.floor();

                let sample_rgb = |sx: u32, sy: u32| -> (f32, f32, f32) {
                    let y_val = frame.y[(sy * frame.width + sx) as usize] as f32;
                    let (u_val, v_val) = chroma_sample(frame, sx, sy);
                    // BT.601 full-range YUV -> RGB (matches JPEG/OpenCV
                    // training pipeline and NPP GPU path)
                    let r = (y_val + 1.402 * (v_val - 128.0)).clamp(0.0, 255.0);
                    let g = (y_val - 0.344136 * (u_val - 128.0) - 0.714136 * (v_val - 128.0))
                        .clamp(0.0, 255.0);
                    let b = (y_val + 1.772 * (u_val - 128.0)).clamp(0.0, 255.0);
                    (r, g, b)
                };

                let (r00, g00, b00) = sample_rgb(x0, y0);
                let (r10, g10, b10) = sample_rgb(x1, y0);
                let (r01, g01, b01) = sample_rgb(x0, y1);
                let (r11, g11, b11) = sample_rgb(x1, y1);

                let lerp = |a: f32, b: f32, c: f32, d: f32| -> f32 {
                    a * (1.0 - fx) * (1.0 - fy)
                        + b * fx * (1.0 - fy)
                        + c * (1.0 - fx) * fy
                        + d * fx * fy
                };
                let r = lerp(r00, r10, r01, r11);
                let g = lerp(g00, g10, g01, g11);
                let b = lerp(b00, b10, b01, b11);

                let ox = (pad_x_i + dx) as usize;
                let oy = (pad_y_i + dy) as usize;

                self.rgb_chw_buf[oy * sz + ox] = r / 255.0;
                self.rgb_chw_buf[plane + oy * sz + ox] = g / 255.0;
                self.rgb_chw_buf[2 * plane + oy * sz + ox] = b / 255.0;
            }
        }

        (scale, pad_x, pad_y)
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

        let (scale, pad_x, pad_y) = {
            reco_core::profile_scope!("yolo_preprocess");
            self.preprocess(frame)
        };

        let sz = self.input_size as usize;
        let input_tensor =
            TensorRef::from_array_view(([1, 3, sz, sz], self.rgb_chw_buf.as_slice()))
                .map_err(|e| DetectorError::InferenceFailed(format!("tensor build: {e}")))?;

        let outputs = {
            reco_core::profile_scope!("yolo_inference");
            self.session
                .run(ort::inputs![input_tensor])
                .map_err(|e| DetectorError::InferenceFailed(format!("ort run: {e}")))?
        };

        // Borrow the output tensor's backing buffer instead of cloning
        // it into a Vec. `outputs` owns it; postprocess finishes before
        // the drop below. (plan M7 item 5)
        // Stock YOLO exports one [1,N,6] end-to-end-NMS output; the
        // external ball detector emits multiple (boxes + seg heads) with a
        // raw pre-NMS, cxcywh, conf-in-col-5 layout. Pick the decoder by
        // output count - both share a signature.
        let postproc = if outputs.len() > 1 {
            postprocess_balldet
        } else {
            postprocess
        };
        let (shape, slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("output extract: {e}")))?;
        let n = shape[1] as usize;

        let detections = postproc(
            slice,
            n,
            camera,
            self.confidence_threshold,
            scale,
            pad_x,
            pad_y,
            frame.width,
            frame.height,
        );
        drop(outputs);

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

    fn detect_preprocessed(
        &mut self,
        camera: CameraId,
        data: &[f32],
        input_size: u32,
        src_width: u32,
        src_height: u32,
    ) -> Result<Vec<Detection>, DetectorError> {
        reco_core::profile_scope!("yolo_detect_preprocessed");

        let sz = input_size as usize;
        let expected = 3 * sz * sz;
        if data.len() != expected {
            return Err(DetectorError::InferenceFailed(format!(
                "PreprocessedChw: expected {expected} floats, got {}",
                data.len()
            )));
        }

        let input_tensor = TensorRef::from_array_view(([1, 3, sz, sz], data))
            .map_err(|e| DetectorError::InferenceFailed(format!("tensor build: {e}")))?;

        let outputs = {
            reco_core::profile_scope!("yolo_inference");
            self.session
                .run(ort::inputs![input_tensor])
                .map_err(|e| DetectorError::InferenceFailed(format!("ort run: {e}")))?
        };

        let postproc = if outputs.len() > 1 {
            postprocess_balldet
        } else {
            postprocess
        };
        let (shape, slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("output extract: {e}")))?;
        let n = shape[1] as usize;

        // Recompute letterbox params to match the preprocessor's layout
        let fw = src_width as f32;
        let fh = src_height as f32;
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let pad_x = (is - (fw * scale).round()) / 2.0;
        let pad_y = (is - (fh * scale).round()) / 2.0;

        let detections = postproc(
            slice,
            n,
            camera,
            self.confidence_threshold,
            scale,
            pad_x,
            pad_y,
            src_width,
            src_height,
        );
        drop(outputs);

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
            DetectorFrame::PreprocessedChw {
                data,
                input_size,
                src_width,
                src_height,
            } => self.detect_preprocessed(camera, data, *input_size, *src_width, *src_height),
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
