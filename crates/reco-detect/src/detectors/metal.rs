//! Metal-resident YOLO detector for the macOS zero-copy pipeline.
//!
//! Runs the entire detection pipeline on GPU via Metal compute shaders:
//! NV12 color conversion, resize with letterbox, normalize + CHW transpose,
//! then inference via either:
//! - **CoreML native** (`.mlmodelc`) - dispatches to Neural Engine for best perf
//! - **ORT with CoreML EP** (`.onnx`) - runtime ONNX-to-CoreML conversion
//!
//! Only the small detection output (~7KB for `[1, 300, 6]`) is processed on CPU.
//!
//! The Metal output buffer uses shared storage mode (Apple Silicon unified
//! memory), so the preprocessed tensor is CPU-accessible without an
//! explicit GPU-to-CPU copy.

use std::path::Path;

use reco_core::coreml_inference::CoreMlModel;
use reco_core::detector::{CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector};
use reco_core::gpu::GpuContext;
use reco_core::metal_compute::MetalPreprocessPipeline;
use reco_core::metal_interop::CVPixelBufferRef;

use super::postprocess;

/// Inference backend for the Metal YOLO detector.
enum InferenceBackend {
    /// Native CoreML inference (`.mlmodelc` bundle).
    /// Dispatches to Neural Engine / GPU / CPU as CoreML sees fit.
    CoreMlNative(CoreMlModel),
    /// ORT inference with CoreML EP (`.onnx` model).
    /// Runtime ONNX-to-CoreML conversion, may fall back to CPU.
    OrtSession {
        session: ort::session::Session,
        input_size: u32,
    },
}

/// YOLO detector that operates on Metal-resident NV12 frames.
///
/// Uses a Metal compute shader for preprocessing. For inference, supports
/// both native CoreML (`.mlmodelc`) for Neural Engine acceleration and
/// ORT with CoreML EP (`.onnx`) as a fallback.
///
/// Created via [`MetalYoloDetector::try_new`].
pub struct MetalYoloDetector {
    /// Owned clone of the session's `GpuContext`. Needed by the Metal
    /// preprocess pipeline on every `detect()` call; before the M3
    /// trait collapse the session passed a `&GpuContext` parameter
    /// but `UnifiedDetector::detect` has no such parameter, so the
    /// detector now owns its own handle. Cloning is cheap (wgpu
    /// types are reference-counted).
    gpu: GpuContext,
    backend: InferenceBackend,
    preprocess: MetalPreprocessPipeline,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
    // Pre-computed letterbox parameters (constant for fixed frame dimensions).
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    /// Frame counter for periodic texture cache flush.
    frame_counter: u64,
}

impl MetalYoloDetector {
    /// Create a Metal YOLO detector.
    ///
    /// The model path can be either:
    /// - A `.mlmodelc` directory: uses native CoreML inference (Neural Engine)
    /// - A `.onnx` file: uses ORT with CoreML EP (runtime conversion)
    ///
    /// `frame_width`/`frame_height` are the raw camera frame dimensions.
    /// Letterbox parameters are pre-computed from these dimensions.
    pub fn try_new(
        model_path: impl AsRef<Path>,
        gpu: &GpuContext,
        frame_width: u32,
        frame_height: u32,
        confidence_threshold: f32,
        labels: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = model_path.as_ref();

        let (backend, input_size, labels) = if path.extension().is_some_and(|ext| ext == "mlmodelc")
            || path.to_str().is_some_and(|s| s.ends_with(".mlmodelc"))
        {
            // Native CoreML: extract input size from directory name convention
            // (e.g. "ball_v0_960.mlmodelc") or default to checking the path.
            let input_size = extract_input_size_from_path(path).unwrap_or(1280);

            let coreml = CoreMlModel::load(
                path,
                "images",  // YOLO input tensor name
                "output0", // YOLO output tensor name
                [1, 3, input_size as i64, input_size as i64],
            )?;

            log::info!(
                "MetalYoloDetector: using native CoreML (ANE/GPU), input={input_size}x{input_size}"
            );
            let labels = if labels.is_empty() {
                vec!["ball".into()]
            } else {
                labels
            };
            (InferenceBackend::CoreMlNative(coreml), input_size, labels)
        } else {
            // ORT with CoreML EP fallback (uses shared session builder).
            let (session, input_size, labels) =
                crate::ort_session::create_ort_session(path, labels)?;
            (
                InferenceBackend::OrtSession {
                    session,
                    input_size,
                },
                input_size,
                labels,
            )
        };

        // Pre-compute letterbox parameters.
        let (fw, fh) = (frame_width as f32, frame_height as f32);
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (input_size - new_w) as f32 / 2.0;
        let pad_y = (input_size - new_h) as f32 / 2.0;

        // Create Metal preprocessing pipeline.
        let preprocess = MetalPreprocessPipeline::new(gpu, input_size, frame_width, frame_height)
            .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;

        let tensor_bytes = 3 * (input_size as usize) * (input_size as usize) * 4;
        log::info!(
            "MetalYoloDetector ready: input={}x{}, frame={}x{}, scale={:.3}, pad=({:.1},{:.1}), \
             buffer={:.1}MB",
            input_size,
            input_size,
            frame_width,
            frame_height,
            scale,
            pad_x,
            pad_y,
            tensor_bytes as f64 / 1024.0 / 1024.0,
        );

        Ok(Self {
            gpu: gpu.clone(),
            backend,
            preprocess,
            input_size,
            confidence_threshold,
            labels,
            scale,
            pad_x,
            pad_y,
            frame_counter: 0,
        })
    }

    /// Run inference and return (n_detections, output_data).
    fn run_inference(&mut self, tensor_data: &mut [f32]) -> Option<(usize, Vec<f32>)> {
        match &mut self.backend {
            InferenceBackend::CoreMlNative(coreml) => {
                reco_core::profile_scope!("coreml_native_inference");
                match coreml.predict(tensor_data.as_mut_ptr(), tensor_data.len()) {
                    Ok(result) => Some(result),
                    Err(e) => {
                        log::error!("CoreML native inference failed: {e}");
                        None
                    }
                }
            }
            InferenceBackend::OrtSession {
                session,
                input_size,
            } => {
                reco_core::profile_scope!("metal_ort_inference");
                let sz = *input_size as i64;
                let tensor = match ort::value::TensorRef::from_array_view((
                    [1i64, 3, sz, sz],
                    &*tensor_data,
                )) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!("Failed to create ORT tensor: {e}");
                        return None;
                    }
                };

                let outputs = match session.run(ort::inputs![tensor]) {
                    Ok(o) => o,
                    Err(e) => {
                        log::error!("ORT inference failed: {e}");
                        return None;
                    }
                };

                match outputs[0].try_extract_tensor::<f32>() {
                    Ok((shape, slice)) => Some((shape[1] as usize, slice.to_vec())),
                    Err(e) => {
                        log::error!("Failed to extract YOLO output: {e}");
                        None
                    }
                }
            }
        }
    }
}

impl UnifiedDetector for MetalYoloDetector {
    fn name(&self) -> &'static str {
        "metal-coreml"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        // Metal-residency backend: matches `Metal { cv_pixel_buffer,
        // width, height }`; all other residencies (Cpu, Cuda, future
        // `#[non_exhaustive]` additions) are routed to
        // `UnsupportedFrameKind` via a single wildcard arm so the
        // dispatcher picks a different backend.
        //
        // The `GpuContext` needed by `MetalPreprocessPipeline` is
        // owned by this struct (cloned at construction time); wgpu
        // types are reference-counted so the clone is cheap and shares
        // the session's device/queue.
        let (cv_pixel_buffer, width, height) = match frame {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            DetectorFrame::Metal {
                cv_pixel_buffer,
                width,
                height,
            } => (*cv_pixel_buffer, *width, *height),
            _ => return Err(DetectorError::UnsupportedFrameKind),
        };

        reco_core::profile_scope!("metal_yolo_detect");

        // Step 1: Metal compute preprocess (NV12 -> CHW f32 tensor).
        let tensor_data = {
            reco_core::profile_scope!("metal_preprocess");
            // SAFETY: caller guarantees cv_pixel_buffer is valid
            // (session wraps it as a RetainedCVPixelBuffer).
            unsafe { self.preprocess.preprocess(cv_pixel_buffer, &self.gpu) }
                .map_err(|e| DetectorError::InferenceFailed(format!("metal preprocess: {e}")))?
        };

        // Step 2: Run inference (CoreML native or ORT).
        let mut tensor_owned = tensor_data.to_vec();
        let (n, data) = self.run_inference(&mut tensor_owned).ok_or_else(|| {
            DetectorError::InferenceFailed("metal inference returned None".into())
        })?;

        // Step 3: Postprocess on CPU (shared between all backends).
        let detections = postprocess(
            &data,
            n,
            camera,
            self.confidence_threshold,
            self.scale,
            self.pad_x,
            self.pad_y,
            width,
            height,
        );

        if !detections.is_empty() {
            log::debug!(
                "Metal camera {:?}: {} detection(s) - {}",
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

        // Periodic texture cache flush.
        self.frame_counter += 1;
        if self.frame_counter.is_multiple_of(60) {
            self.preprocess.flush_cache();
        }

        Ok(detections)
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

/// Try to extract input size from a model path like "ball_v0_960.mlmodelc".
fn extract_input_size_from_path(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    // Look for a trailing number after the last underscore.
    let last_part = stem.rsplit('_').next()?;
    last_part
        .parse::<u32>()
        .ok()
        .filter(|&s| s >= 320 && s <= 2048)
}
