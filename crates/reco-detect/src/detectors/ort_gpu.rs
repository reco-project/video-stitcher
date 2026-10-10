//! GPU-resident YOLO detector for the zero-copy pipeline.
//!
//! Runs the entire detection pipeline on GPU: NV12 conversion, letterboxing
//! and CHW normalization in one CUDA kernel,
//! and inference (ORT TensorRT/CUDA EP). Only the small detection output
//! (~7KB for `[1, 300, 6]`) is read back to CPU.

/// `MemoryInfo` wrapper with manual `Send` impl.
///
/// ORT's `MemoryInfo` holds `*mut OrtMemoryInfo` and is therefore not
/// `Send` by default. But a `MemoryInfo` is immutable descriptor state
/// (device / allocator type / memtype) — ORT's C layer treats it as
/// read-only config. Detector requires `Send`, so wrap with an
/// opt-in unsafe impl for the "we only read it" use case.
struct SendMemoryInfo(ort::memory::MemoryInfo);
// SAFETY: MemoryInfo is read-only descriptor state; no interior mutation
// crosses thread boundaries.
unsafe impl Send for SendMemoryInfo {}

use std::ffi::c_void;
use std::path::Path;

use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{Shape, TensorRefMut};
use reco_core::detect::detector::{
    CudaNv12Frame, Detection, Detector, DetectorError, DetectorFrame,
};
use reco_core::geometry::CameraIndex;
use reco_core::interop::cuda::{
    CUdeviceptr, CudaInteropError, cuda_ensure_context, cuda_mem_alloc, cuda_mem_free,
};

use super::postprocess;

struct Allocation(CUdeviceptr);

impl Allocation {
    fn new(bytes: usize) -> Result<Self, CudaInteropError> {
        cuda_mem_alloc(bytes).map(Self)
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Err(error) = cuda_ensure_context().and_then(|()| cuda_mem_free(self.0)) {
            log::warn!("Failed to free detector CUDA allocation: {error}");
        }
    }
}

struct InputLayout {
    size: (u32, u32),
    scale: f32,
    pad_x: u32,
    pad_y: u32,
}

impl InputLayout {
    fn new(size: (u32, u32), input_size: u32) -> Self {
        let scale = (input_size as f32 / size.0 as f32).min(input_size as f32 / size.1 as f32);
        Self {
            size,
            scale,
            pad_x: (input_size - (size.0 as f32 * scale).round() as u32) / 2,
            pad_y: (input_size - (size.1 as f32 * scale).round() as u32) / 2,
        }
    }
}

/// YOLO detector that operates on GPU-resident NV12 frames via ORT.
///
/// Pre-allocates GPU scratch buffers for the preprocessing pipeline and
/// reuses them across frames. The ORT session runs with TensorRT or CUDA EP
/// for GPU-side inference.
///
/// Created via [`OrtGpuDetector::try_new`], which requires a registered CUDA-compatible inference provider.
pub struct OrtGpuDetector {
    session: Session,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
    inputs: Vec<InputLayout>,
    tensor_f32: Allocation,
    // One conversion allocation sized for the largest declared input. The offset locates UV.
    p010_scratch: Option<(Allocation, usize)>,
    // Cached CUDA device MemoryInfo. Constant for the detector's
    // lifetime; constructing one per inference is measurable
    // per-frame allocation churn.
    cuda_memory_info: SendMemoryInfo,
}

impl OrtGpuDetector {
    /// Try to create a GPU YOLO detector.
    ///
    /// Returns `Ok(None)` if no CUDA-compatible ORT provider is compiled in.
    /// Returns `Err` if runtime provider registration, CUDA initialization or model loading fails.
    ///
    /// `source_sizes` declares each camera's decoded dimensions in camera-index order. Each frame must match its camera's declared size. One inference session and tensor are reused across cameras.
    ///
    /// When `is_10bit` is true, one reusable allocation covers P010-to-NV12 conversion for the largest declared input.
    pub fn try_new(
        model_path: impl AsRef<Path>,
        source_sizes: &[(u32, u32)],
        confidence_threshold: f32,
        labels: Vec<String>,
        is_10bit: bool,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if source_sizes.is_empty() {
            return Err("at least one source size is required".into());
        }
        let mut max_pixels = 0;
        for &(width, height) in source_sizes {
            if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                return Err("NV12 frame dimensions must be positive and even".into());
            }
            let pixels = (width as usize)
                .checked_mul(height as usize)
                .ok_or("frame dimensions overflow")?;
            max_pixels = max_pixels.max(pixels);
        }
        // GPU detection needs a CUDA-capable EP (TensorRT or CUDA) to
        // process device pointers. Without one, ORT falls back to CPU EP
        // which segfaults on CUDA memory.
        if !cfg!(feature = "tensorrt") && !cfg!(feature = "cuda") {
            log::warn!(
                "OrtGpuDetector: no GPU EP available (need --features tensorrt or --features cuda)"
            );
            return Ok(None);
        }

        cuda_ensure_context()?;

        let (session, input_size, labels) =
            crate::ort_session::create_cuda_ort_session(model_path.as_ref(), labels)?;

        let inputs = source_sizes
            .iter()
            .map(|&size| InputLayout::new(size, input_size))
            .collect();

        // Allocate GPU scratch buffers (checked arithmetic to prevent overflow).
        let tensor_size = (input_size as usize)
            .checked_mul(input_size as usize)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(4))
            .ok_or("input dimensions overflow for tensor_size")?;

        let tensor_f32 = Allocation::new(tensor_size)?;

        // Allocate P010 conversion scratch buffers if needed.
        let p010_scratch = if is_10bit {
            let bytes = max_pixels
                .checked_add(max_pixels / 2)
                .ok_or("P010 scratch dimensions overflow")?;
            Some((Allocation::new(bytes)?, max_pixels))
        } else {
            None
        };

        log::info!(
            "OrtGpuDetector ready: input={}x{}, cameras={}, \
             tensor={:.1}MB, 10bit={}",
            input_size,
            input_size,
            source_sizes.len(),
            tensor_size as f64 / 1024.0 / 1024.0,
            is_10bit,
        );

        let cuda_memory_info = SendMemoryInfo(
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(|e| format!("CUDA MemoryInfo: {e}"))?,
        );

        let mut detector = Self {
            session,
            input_size,
            confidence_threshold,
            labels,
            inputs,
            tensor_f32,
            p010_scratch,
            cuda_memory_info,
        };

        // Warmup: force TRT EP to eagerly build the engine and initialize
        // CUDA resources. Without this, the first real inference triggers
        // lazy init which can conflict with NVDEC decode thread contexts.
        {
            let sz = input_size as usize;
            let warmup_data = vec![0.0f32; 3 * sz * sz];
            let tensor = ort::value::Tensor::from_array(([1, 3, sz, sz], warmup_data))?;
            detector.session.run(ort::inputs![tensor])?;
            log::info!("OrtGpuDetector: warmup inference complete");
        }

        Ok(Some(detector))
    }
}

impl OrtGpuDetector {
    /// CUDA NV12/P010 preprocessing and inference with stage-specific errors.
    fn detect_gpu_raw(
        &mut self,
        camera: CameraIndex,
        frame: &CudaNv12Frame,
    ) -> Result<Vec<Detection>, DetectorError> {
        let CudaNv12Frame {
            y_ptr,
            uv_ptr,
            y_pitch,
            uv_pitch,
            width,
            height,
            rotation,
            is_10bit,
        } = *frame;
        reco_core::profile_scope!("gpu_yolo_detect");

        let input = self.inputs.get(camera).ok_or_else(|| {
            DetectorError::InferenceFailed(format!("undeclared camera index {camera}"))
        })?;
        if input.size != (width, height) {
            return Err(DetectorError::InferenceFailed(format!(
                "camera {camera} dimensions changed: expected {:?}, got {width}x{height}",
                input.size
            )));
        }
        let row_bytes = width as usize * if is_10bit { 2 } else { 1 };
        if y_ptr == 0
            || uv_ptr == 0
            || y_pitch < row_bytes
            || uv_pitch < row_bytes
            || y_pitch > u32::MAX as usize
            || uv_pitch > u32::MAX as usize
            || y_pitch
                .checked_mul(height as usize)
                .is_none_or(|n| n > u32::MAX as usize)
            || uv_pitch
                .checked_mul(height as usize / 2)
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(DetectorError::InferenceFailed(
                "invalid CUDA NV12 pointers or strides".into(),
            ));
        }

        // Ensure a CUDA context is current on this thread. The zero-copy
        // frame loop may not have one after NVDEC decode pushes/pops its
        // own context.
        reco_core::interop::cuda::cuda_ensure_context()
            .map_err(|e| DetectorError::InferenceFailed(format!("cuda_ensure_context: {e}")))?;

        // Step 0: Convert P010 (10-bit) to 8-bit NV12 if needed.
        // The fused conversion reads 8-bit samples, so extract the high byte first.
        let (nv12_y, nv12_y_pitch, nv12_uv, nv12_uv_pitch) = if is_10bit {
            reco_core::profile_scope!("p010_to_nv12");
            let (scratch, uv_offset) = self.p010_scratch.as_ref().ok_or_else(|| {
                DetectorError::InferenceFailed(
                    "P010 frame received but no conversion buffers allocated".into(),
                )
            })?;
            let nv12_8bit_y = scratch.0;
            let nv12_8bit_uv = scratch.0 + *uv_offset as u64;
            // Convert Y plane: width * height samples.
            crate::cuda_kernels::p010_plane_to_nv12(y_ptr, y_pitch, nv12_8bit_y, width, height)
                .map_err(|e| {
                    DetectorError::InferenceFailed(format!("P010->NV12 Y conversion: {e}"))
                })?;
            // Convert UV plane: width * (height/2) samples.
            // UV plane has width/2 pixel pairs, each 2 u16 values = width u16 samples per row.
            crate::cuda_kernels::p010_plane_to_nv12(
                uv_ptr,
                uv_pitch,
                nv12_8bit_uv,
                width,
                height / 2,
            )
            .map_err(|e| {
                DetectorError::InferenceFailed(format!("P010->NV12 UV conversion: {e}"))
            })?;
            // The 8-bit buffers are tightly packed (no pitch padding).
            (nv12_8bit_y, width as usize, nv12_8bit_uv, width as usize)
        } else {
            (y_ptr, y_pitch, uv_ptr, uv_pitch)
        };

        // Combined NV12 -> float32 RGB CHW in one kernel pass.
        // Uses BT.709 full-range coefficients matching GoPro/action-cam
        // yuvj420p output. Replaces the NPP NV12-to-RGB (BT.601
        // video-range) + NPP resize + normalize pipeline that was
        // producing a 10x detection count difference vs CPU ORT (#284).
        {
            reco_core::profile_scope!("nv12_to_rgb_chw");
            crate::cuda_kernels::nv12_to_rgb_chw_fullrange(
                nv12_y,
                nv12_uv,
                self.tensor_f32.0,
                nv12_y_pitch as u32,
                nv12_uv_pitch as u32,
                width,
                height,
                self.input_size,
                self.input_size,
                input.pad_x,
                input.pad_y,
                input.scale,
                rotation,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("NV12->RGB CHW: {e}")))?;
        }

        // Step 4: Wrap GPU buffer as ORT tensor and run inference.
        let outputs = {
            reco_core::profile_scope!("gpu_ort_inference");

            let sz = self.input_size as i64;
            let tensor: TensorRefMut<'_, f32> = unsafe {
                TensorRefMut::from_raw(
                    self.cuda_memory_info.0.clone(),
                    self.tensor_f32.0 as *mut c_void,
                    Shape::new([1i64, 3, sz, sz]),
                )
            }
            .map_err(|e| DetectorError::InferenceFailed(format!("GPU tensor wrap: {e}")))?;

            self.session
                .run(ort::inputs![tensor])
                .map_err(|e| DetectorError::InferenceFailed(format!("ort run: {e}")))?
        };

        // Step 5: Extract output and postprocess on CPU without
        // materializing an intermediate Vec<f32>. `outputs` owns the
        // backing buffer; `slice` borrows from it. Postprocess runs
        // to completion before we drop `outputs`.
        let (shape, slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("output extract: {e}")))?;
        let n = shape[1] as usize;

        let detections = postprocess(
            slice,
            n,
            camera,
            self.confidence_threshold,
            input.scale,
            input.pad_x as f32,
            input.pad_y as f32,
            width,
            height,
        );
        drop(outputs);

        if !detections.is_empty() {
            log::debug!(
                "GPU camera {:?}: {} detection(s) - {}",
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

impl Detector for OrtGpuDetector {
    fn name(&self) -> &'static str {
        "ort-cuda"
    }

    fn prefers_cuda_nv12(&self) -> bool {
        true
    }

    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        match frame {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            DetectorFrame::CudaNv12(gpu_frame) => self.detect_gpu_raw(camera, gpu_frame),
            _ => Err(DetectorError::UnsupportedFrameKind),
        }
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

#[cfg(test)]
mod tests;
