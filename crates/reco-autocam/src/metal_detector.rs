//! Metal-resident YOLO detector for the macOS zero-copy pipeline.
//!
//! Runs the entire detection pipeline on GPU via Metal compute shaders:
//! NV12 color conversion, resize with letterbox, normalize + CHW transpose,
//! then inference via ORT with CoreML EP (or CPU fallback). Only the small
//! detection output (~7KB for `[1, 300, 6]`) is processed on CPU.
//!
//! The Metal output buffer uses shared storage mode (Apple Silicon unified
//! memory), so the preprocessed tensor is CPU-accessible without an
//! explicit GPU-to-CPU copy.

use std::path::Path;

use ort::session::Session;
use reco_core::detector::{CameraId, Detection, MetalDetector};
use reco_core::gpu::GpuContext;
use reco_core::metal_compute::MetalPreprocessPipeline;
use reco_core::metal_interop::CVPixelBufferRef;

use crate::detector::postprocess;

/// YOLO detector that operates on Metal-resident NV12 frames.
///
/// Uses a Metal compute shader for preprocessing and ORT with CoreML EP
/// for inference. Pre-allocates the compute pipeline and shared buffers,
/// reusing them across frames.
///
/// Created via [`MetalYoloDetector::try_new`].
pub struct MetalYoloDetector {
    session: Session,
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
    /// Try to create a Metal YOLO detector.
    ///
    /// Returns `Err` for real failures like missing model file, ORT errors,
    /// or Metal pipeline creation failures.
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
        // Load ORT session with CoreML EP (or CPU fallback).
        #[allow(unused_mut)]
        let mut builder = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        #[cfg(feature = "coreml")]
        let mut builder = {
            match builder.with_execution_providers([ort::ep::CoreML::default()
                .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                .with_model_cache_dir("/tmp/reco-coreml-cache")
                .build()])
            {
                Ok(b) => {
                    log::info!("MetalYoloDetector: CoreML EP enabled");
                    b
                }
                Err(e) => {
                    log::warn!("MetalYoloDetector: CoreML EP failed ({e}), falling back to CPU");
                    e.recover()
                }
            }
        };

        let session = builder.commit_from_file(model_path.as_ref())?;

        // Extract input size from model metadata.
        let input_size = match session.inputs()[0].dtype() {
            ort::value::ValueType::Tensor { shape, .. } => {
                let h = shape[2];
                if h > 0 { h as u32 } else { 1280 }
            }
            _ => 1280,
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
            session,
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
}

impl MetalDetector for MetalYoloDetector {
    fn detect_metal(
        &mut self,
        camera: CameraId,
        cv_pixel_buffer: CVPixelBufferRef,
        width: u32,
        height: u32,
        gpu: &GpuContext,
    ) -> Vec<Detection> {
        reco_core::profile_scope!("metal_yolo_detect");

        // Step 1: Metal compute preprocess (NV12 -> CHW f32 tensor).
        let tensor_data = {
            reco_core::profile_scope!("metal_preprocess");
            // SAFETY: caller guarantees cv_pixel_buffer is valid (from RetainedCVPixelBuffer).
            match unsafe { self.preprocess.preprocess(cv_pixel_buffer, gpu) } {
                Ok(data) => data,
                Err(e) => {
                    log::error!("Metal preprocess failed: {e}");
                    return Vec::new();
                }
            }
        };

        // Step 2: Create ORT tensor from the unified-memory buffer and run inference.
        let outputs = {
            reco_core::profile_scope!("metal_ort_inference");

            let sz = self.input_size as i64;
            let tensor =
                match ort::value::TensorRef::from_array_view(([1i64, 3, sz, sz], tensor_data)) {
                    Ok(t) => t,
                    Err(e) => {
                        log::error!("Failed to create ORT tensor: {e}");
                        return Vec::new();
                    }
                };

            match self.session.run(ort::inputs![tensor]) {
                Ok(o) => o,
                Err(e) => {
                    log::error!("Metal YOLO inference failed: {e}");
                    return Vec::new();
                }
            }
        };

        // Step 3: Extract output and postprocess on CPU.
        let (n, data) = match outputs[0].try_extract_tensor::<f32>() {
            Ok((shape, slice)) => (shape[1] as usize, slice.to_vec()),
            Err(e) => {
                log::error!("Failed to extract YOLO output: {e}");
                return Vec::new();
            }
        };
        drop(outputs);

        let detections = postprocess(
            &data,
            n,
            camera,
            self.confidence_threshold,
            &self.labels,
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

        // Periodic texture cache flush.
        self.frame_counter += 1;
        if self.frame_counter.is_multiple_of(60) {
            self.preprocess.flush_cache();
        }

        detections
    }
}
