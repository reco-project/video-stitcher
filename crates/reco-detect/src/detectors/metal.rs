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

use crate::coreml_inference::CoreMlModel;
use crate::metal_compute::MetalPreprocessPipeline;
use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::geometry::CameraIndex;
use reco_core::gpu::GpuContext;

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
    /// Shares the session's Metal device and queue.
    gpu: GpuContext,
    backend: InferenceBackend,
    preprocess: MetalPreprocessPipeline,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
    source_sizes: Vec<(u32, u32)>,
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
    /// `source_sizes` declares the raw dimensions of each camera in projection order.
    pub fn try_new(
        model_path: impl AsRef<Path>,
        gpu: &GpuContext,
        source_sizes: &[(u32, u32)],
        confidence_threshold: f32,
        labels: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if source_sizes.is_empty() || source_sizes.iter().any(|&(w, h)| w == 0 || h == 0) {
            return Err("Metal detector requires nonempty camera dimensions".into());
        }
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

        for &size in source_sizes {
            crate::letterbox::Letterbox::new(size, input_size)?;
        }
        let preprocess = MetalPreprocessPipeline::new(gpu, input_size)?;
        log::info!(
            "MetalYoloDetector ready: model={}x{}, cameras={source_sizes:?}",
            input_size,
            input_size
        );

        Ok(Self {
            gpu: gpu.clone(),
            backend,
            preprocess,
            input_size,
            confidence_threshold,
            labels,
            source_sizes: source_sizes.to_vec(),
            frame_counter: 0,
        })
    }
}

impl InferenceBackend {
    /// Borrows shared Metal tensor storage for synchronous inference.
    fn run(&mut self, tensor_data: &mut [f32]) -> Result<(usize, Vec<f32>), DetectorError> {
        let failed = |e: String| DetectorError::InferenceFailed(e);
        match self {
            Self::CoreMlNative(coreml) => {
                reco_core::profile_scope!("coreml_native_inference");
                coreml
                    .predict(tensor_data.as_mut_ptr(), tensor_data.len())
                    .map_err(|e| failed(e.to_string()))
            }
            Self::OrtSession {
                session,
                input_size,
            } => {
                reco_core::profile_scope!("metal_ort_inference");
                let sz = *input_size as i64;
                let tensor =
                    ort::value::TensorRef::from_array_view(([1i64, 3, sz, sz], &*tensor_data))
                        .map_err(|e| failed(e.to_string()))?;
                let outputs = session
                    .run(ort::inputs![tensor])
                    .map_err(|e| failed(e.to_string()))?;
                let (shape, slice) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| failed(e.to_string()))?;
                let count = shape
                    .get(1)
                    .and_then(|&n| usize::try_from(n).ok())
                    .ok_or_else(|| failed("invalid Metal model output shape".into()))?;
                Ok((count, slice.to_vec()))
            }
        }
    }
}

impl Detector for MetalYoloDetector {
    fn name(&self) -> &'static str {
        "metal-coreml"
    }

    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let (cv_pixel_buffer, width, height) = match frame {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            DetectorFrame::Metal {
                cv_pixel_buffer,
                width,
                height,
            } => (*cv_pixel_buffer, *width, *height),
            _ => return Err(DetectorError::UnsupportedFrameKind),
        };

        if self.source_sizes.get(camera) != Some(&(width, height)) {
            return Err(DetectorError::InferenceFailed(
                "Metal camera or dimensions do not match the detector".into(),
            ));
        }
        let layout = crate::letterbox::Letterbox::new((width, height), self.input_size)?;
        reco_core::profile_scope!("metal_yolo_detect");

        // Step 1: Metal compute preprocess (NV12 -> CHW f32 tensor).
        let tensor_data = {
            reco_core::profile_scope!("metal_preprocess");
            // SAFETY: caller guarantees cv_pixel_buffer is valid
            // (session wraps it as a RetainedCVPixelBuffer).
            unsafe {
                self.preprocess
                    .preprocess(cv_pixel_buffer, (width, height), &self.gpu)
            }
            .map_err(|e| DetectorError::InferenceFailed(format!("metal preprocess: {e}")))?
        };

        // Step 2: Run inference (CoreML native or ORT).
        let (n, data) = self.backend.run(tensor_data)?;

        // Step 3: Postprocess on CPU (shared between all backends).
        let detections = postprocess(
            &data,
            n,
            camera,
            self.confidence_threshold,
            layout.scale,
            layout.pad_x as f32,
            layout.pad_y as f32,
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
        .filter(|&s| (320..=2048).contains(&s))
}
