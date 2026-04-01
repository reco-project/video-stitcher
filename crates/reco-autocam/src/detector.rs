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
        Self::with_config(path, 0.05, Vec::new())
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
        #[allow(unused_mut)]
        let mut builder = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        // Try TensorRT EP first (JIT-compiles for any GPU arch including Blackwell),
        // then CUDA EP, then fall back to CPU.
        #[cfg(feature = "tensorrt")]
        let mut builder = {
            match builder.with_execution_providers([ort::ep::TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path("/tmp/reco-trt-cache")
                .with_timing_cache(true)
                .with_timing_cache_path("/tmp/reco-trt-cache")
                .with_builder_optimization_level(3)
                .build()])
            {
                Ok(b) => {
                    log::info!("YoloDetector: TensorRT execution provider enabled");
                    b
                }
                Err(e) => {
                    log::warn!("YoloDetector: TensorRT EP failed ({e}), falling back to CPU");
                    e.recover()
                }
            }
        };

        // Try CUDA execution provider, fall back to CPU.
        #[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
        let mut builder = {
            match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
                Ok(b) => {
                    log::info!("YoloDetector: CUDA execution provider enabled");
                    b
                }
                Err(e) => {
                    log::warn!("YoloDetector: CUDA EP failed ({e}), falling back to CPU");
                    e.recover()
                }
            }
        };

        let session = builder.commit_from_file(path.as_ref())?;

        // Extract input size from model metadata.
        let input_size = match session.inputs()[0].dtype() {
            ort::value::ValueType::Tensor { shape, .. } => {
                // shape[2] is height, shape[3] is width (BCHW).
                let h = shape[2];
                if h > 0 { h as u32 } else { 1280 }
            }
            _ => 1280,
        };

        // Auto-detect labels from model metadata if not provided.
        let labels = if labels.is_empty() {
            parse_onnx_names(&session).unwrap_or_else(|| vec!["ball".into()])
        } else {
            labels
        };

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

                // BT.601 YUV -> RGB
                let r = (y_val + 1.402 * (v_val - 128.0)).clamp(0.0, 255.0);
                let g = (y_val - 0.344136 * (u_val - 128.0) - 0.714136 * (v_val - 128.0))
                    .clamp(0.0, 255.0);
                let b = (y_val + 1.772 * (u_val - 128.0)).clamp(0.0, 255.0);

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
            &self.labels,
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
    labels: &[String],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    frame_width: u32,
    frame_height: u32,
) -> Vec<Detection> {
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

        let label = labels
            .get(class_id)
            .cloned()
            .unwrap_or_else(|| format!("class_{class_id}"));

        detections.push(Detection {
            camera,
            label,
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
                    .map(|d| format!(
                        "{}({:.0}%@{:.2},{:.2})",
                        d.label,
                        d.confidence * 100.0,
                        d.center_x,
                        d.center_y
                    ))
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
