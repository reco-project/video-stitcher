//! NCNN-based YOLO detector for ARM/embedded platforms.
//!
//! Uses the [NCNN](https://github.com/Tencent/ncnn) inference framework
//! which is optimized for ARM NEON. Achieves ~67ms per frame on RPi5
//! at 640px input (vs 130ms for ONNX Runtime).
//!
//! The model must be exported to NCNN format (`.param` + `.bin` files)
//! via `yolo export format=ncnn`.
//!
//! Unlike the end-to-end NMS models used by ORT/TRT, NCNN models
//! output raw predictions that need NMS postprocessing.

use std::path::Path;

use ncnn_rs::{Mat, Net, Option as NcnnOption};
use reco_core::detector::{CameraId, ChromaFormat, Detection, Detector, RawFrame};

/// YOLO detector using NCNN inference (optimized for ARM).
///
/// Loads an NCNN model (`.param` + `.bin`) and runs inference on CPU.
/// Best suited for Raspberry Pi 5 and other ARM SBCs where ONNX Runtime
/// is too slow.
pub struct NcnnYoloDetector {
    net: Net,
    input_size: u32,
    confidence_threshold: f32,
    nms_threshold: f32,
    labels: Vec<String>,
    // Pre-computed letterbox parameters.
    scale: f32,
    new_w: u32,
    new_h: u32,
    pad_x: f32,
    pad_y: f32,
    frame_width: u32,
    frame_height: u32,
}

impl NcnnYoloDetector {
    /// Create a detector from an NCNN model directory.
    ///
    /// The `model_dir` should contain `model.ncnn.param` and `model.ncnn.bin`
    /// (as produced by `yolo export format=ncnn`).
    ///
    /// `input_size` is the model's input resolution (typically 640).
    /// `labels` are class names; if empty, detections have numeric IDs only.
    pub fn new(
        model_dir: impl AsRef<Path>,
        input_size: u32,
        frame_width: u32,
        frame_height: u32,
        confidence_threshold: f32,
        labels: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model_dir = model_dir.as_ref();
        let param_path = model_dir.join("model.ncnn.param");
        let bin_path = model_dir.join("model.ncnn.bin");

        if !param_path.exists() {
            return Err(format!("NCNN param file not found: {}", param_path.display()).into());
        }
        if !bin_path.exists() {
            return Err(format!("NCNN bin file not found: {}", bin_path.display()).into());
        }

        let mut opt = NcnnOption::new();
        opt.set_num_threads(4); // RPi5 has 4 cores
        opt.set_use_vulkan_compute(false); // CPU only for now

        let mut net = Net::new();
        net.set_option(&opt);
        net.load_param(param_path.to_str().ok_or("invalid path")?)?;
        net.load_model(bin_path.to_str().ok_or("invalid path")?)?;

        // Pre-compute letterbox parameters.
        let (fw, fh) = (frame_width as f32, frame_height as f32);
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (input_size - new_w) as f32 / 2.0;
        let pad_y = (input_size - new_h) as f32 / 2.0;

        log::info!(
            "NcnnYoloDetector ready: input={}x{}, frame={}x{}, scale={:.3}, \
             labels={}",
            input_size,
            input_size,
            frame_width,
            frame_height,
            scale,
            labels.len(),
        );

        Ok(Self {
            net,
            input_size,
            confidence_threshold,
            nms_threshold: 0.45,
            labels,
            scale,
            new_w,
            new_h,
            pad_x,
            pad_y,
            frame_width,
            frame_height,
        })
    }

    /// Access class labels.
    pub fn class_names(&self) -> &[String] {
        &self.labels
    }

    /// Preprocess a raw frame into an NCNN Mat (letterboxed, normalized RGB).
    fn preprocess(&self, frame: &RawFrame<'_>) -> Mat {
        let sz = self.input_size as usize;

        // Create a grey-filled letterbox buffer (HWC, RGB, float32).
        let mut rgb = vec![114.0f32 / 255.0; sz * sz * 3];

        let fw = frame.width as usize;
        let fh = frame.height as usize;
        let new_w = self.new_w as usize;
        let new_h = self.new_h as usize;
        let pad_x = self.pad_x as usize;
        let pad_y = self.pad_y as usize;

        // Simple nearest-neighbor resize + YUV->RGB + normalize.
        for dy in 0..new_h {
            for dx in 0..new_w {
                let sx = ((dx as f32) / self.scale) as u32;
                let sy = ((dy as f32) / self.scale) as u32;
                let sx = sx.min(frame.width - 1);
                let sy = sy.min(frame.height - 1);

                let y_val = frame.y[(sy * frame.width + sx) as usize] as f32;
                let (u_val, v_val) = match &frame.chroma {
                    ChromaFormat::Yuv420p { u, v } => {
                        let cx = (sx / 2) as usize;
                        let cy = (sy / 2) as usize;
                        let cw = fw / 2;
                        (u[cy * cw + cx] as f32, v[cy * cw + cx] as f32)
                    }
                    ChromaFormat::Nv12 { uv } => {
                        let cx = (sx / 2) as usize;
                        let cy = (sy / 2) as usize;
                        (uv[cy * fw + cx * 2] as f32, uv[cy * fw + cx * 2 + 1] as f32)
                    }
                };

                // BT.709 full-range YUV -> RGB, normalized to [0,1].
                let r = (y_val + 1.5748 * (v_val - 128.0)).clamp(0.0, 255.0) / 255.0;
                let g = (y_val - 0.1873 * (u_val - 128.0) - 0.4681 * (v_val - 128.0))
                    .clamp(0.0, 255.0)
                    / 255.0;
                let b = (y_val + 1.8556 * (u_val - 128.0)).clamp(0.0, 255.0) / 255.0;

                let ox = pad_x + dx;
                let oy = pad_y + dy;
                let idx = (oy * sz + ox) * 3;
                rgb[idx] = r;
                rgb[idx + 1] = g;
                rgb[idx + 2] = b;
            }
        }

        // Convert HWC float to NCNN Mat (CHW format).
        // NCNN's Mat::from_pixels_resize expects u8, so we'll use Mat::new_3d
        // and fill it manually in CHW order.
        let mut mat = Mat::new_3d(self.input_size as i32, self.input_size as i32, 3);
        let data = mat.data_mut::<f32>();
        let plane = sz * sz;
        for y in 0..sz {
            for x in 0..sz {
                let src = (y * sz + x) * 3;
                data[y * sz + x] = rgb[src]; // R
                data[plane + y * sz + x] = rgb[src + 1]; // G
                data[2 * plane + y * sz + x] = rgb[src + 2]; // B
            }
        }

        mat
    }

    /// Parse YOLO output (non-NMS) and apply NMS.
    fn postprocess(&self, output: &Mat, camera: CameraId) -> Vec<Detection> {
        // YOLO26 NCNN output shape: [num_classes+4, num_proposals]
        // Transposed from the ONNX [1, num_proposals, num_classes+4] format.
        let num_proposals = output.w() as usize;
        let num_features = output.h() as usize;
        let num_classes = num_features.saturating_sub(4);

        if num_classes == 0 || num_proposals == 0 {
            return Vec::new();
        }

        let data = output.data::<f32>();

        // Collect candidates above confidence threshold.
        let mut candidates: Vec<(Detection, f32)> = Vec::new();

        for i in 0..num_proposals {
            // Find best class.
            let mut best_class = 0u16;
            let mut best_score = 0.0f32;
            for c in 0..num_classes {
                let score = data[(4 + c) * num_proposals + i];
                if score > best_score {
                    best_score = score;
                    best_class = c as u16;
                }
            }

            if best_score < self.confidence_threshold {
                continue;
            }

            // Extract bbox (cx, cy, w, h in letterboxed pixel coords).
            let cx = data[0 * num_proposals + i];
            let cy = data[1 * num_proposals + i];
            let w = data[2 * num_proposals + i];
            let h = data[3 * num_proposals + i];

            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            // Un-letterbox.
            let orig_x1 = (x1 - self.pad_x) / self.scale;
            let orig_y1 = (y1 - self.pad_y) / self.scale;
            let orig_x2 = (x2 - self.pad_x) / self.scale;
            let orig_y2 = (y2 - self.pad_y) / self.scale;

            // Normalize to [0,1].
            let fw = self.frame_width as f32;
            let fh = self.frame_height as f32;
            let ncx = ((orig_x1 + orig_x2) / 2.0) / fw;
            let ncy = ((orig_y1 + orig_y2) / 2.0) / fh;
            let nw = (orig_x2 - orig_x1).abs() / fw;
            let nh = (orig_y2 - orig_y1).abs() / fh;

            if ncx < 0.0 || ncx > 1.0 || ncy < 0.0 || ncy > 1.0 {
                continue;
            }

            candidates.push((
                Detection {
                    camera,
                    class_id: best_class,
                    confidence: best_score,
                    center_x: ncx.clamp(0.0, 1.0),
                    center_y: ncy.clamp(0.0, 1.0),
                    width: nw.clamp(0.0, 1.0),
                    height: nh.clamp(0.0, 1.0),
                },
                best_score,
            ));
        }

        // Sort by confidence descending.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Simple NMS.
        let mut keep = Vec::new();
        let mut suppressed = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            keep.push(candidates[i].0);
            for j in (i + 1)..candidates.len() {
                if suppressed[j] || candidates[i].0.class_id != candidates[j].0.class_id {
                    continue;
                }
                if iou(&candidates[i].0, &candidates[j].0) > self.nms_threshold {
                    suppressed[j] = true;
                }
            }
        }

        keep
    }
}

impl Detector for NcnnYoloDetector {
    fn detect(&mut self, camera: CameraId, frame: &RawFrame<'_>) -> Vec<Detection> {
        reco_core::profile_scope!("ncnn_yolo_detect");

        let input = self.preprocess(frame);

        let mut extractor = self.net.create_extractor();
        if let Err(e) = extractor.input("in0", &input) {
            log::error!("NCNN input failed: {e}");
            return Vec::new();
        }

        let output = match extractor.extract("out0") {
            Ok(mat) => mat,
            Err(e) => {
                log::error!("NCNN inference failed: {e}");
                return Vec::new();
            }
        };

        let detections = self.postprocess(&output, camera);

        if !detections.is_empty() {
            log::debug!(
                "NCNN camera {:?}: {} detection(s)",
                camera,
                detections.len(),
            );
        }

        detections
    }
}

/// Intersection over Union for NMS.
fn iou(a: &Detection, b: &Detection) -> f32 {
    let a_x1 = a.center_x - a.width / 2.0;
    let a_y1 = a.center_y - a.height / 2.0;
    let a_x2 = a.center_x + a.width / 2.0;
    let a_y2 = a.center_y + a.height / 2.0;

    let b_x1 = b.center_x - b.width / 2.0;
    let b_y1 = b.center_y - b.height / 2.0;
    let b_x2 = b.center_x + b.width / 2.0;
    let b_y2 = b.center_y + b.height / 2.0;

    let inter_x1 = a_x1.max(b_x1);
    let inter_y1 = a_y1.max(b_y1);
    let inter_x2 = a_x2.min(b_x2);
    let inter_y2 = a_y2.min(b_y2);

    let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
    let a_area = a.width * a.height;
    let b_area = b.width * b.height;
    let union_area = a_area + b_area - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}
