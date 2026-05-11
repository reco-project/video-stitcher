//! GPU-resident YOLO detector for the zero-copy pipeline.
//!
//! Runs the entire detection pipeline on GPU: NV12 color conversion (NPP),
//! resize with letterbox padding (NPP), normalize + CHW transpose (CUDA kernel),
//! and inference (ORT TensorRT/CUDA EP). Only the small detection output
//! (~7KB for `[1, 300, 6]`) is read back to CPU.

/// `MemoryInfo` wrapper with manual `Send` impl.
///
/// ORT's `MemoryInfo` holds `*mut OrtMemoryInfo` and is therefore not
/// `Send` by default. But a `MemoryInfo` is immutable descriptor state
/// (device / allocator type / memtype) — ORT's C layer treats it as
/// read-only config. UnifiedDetector requires `Send`, so wrap with an
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
    CameraId, Detection, DetectorError, DetectorFrame, GpuNv12Frame, UnifiedDetector,
};
use reco_core::interop::cuda::{
    CUdeviceptr, cuda_ensure_context, cuda_mem_alloc, cuda_mem_free, cuda_memset_d8,
};

use super::postprocess;

/// YOLO detector that operates on GPU-resident NV12 frames via ORT.
///
/// Pre-allocates GPU scratch buffers for the preprocessing pipeline and
/// reuses them across frames. The ORT session runs with TensorRT or CUDA EP
/// for GPU-side inference.
///
/// Created via [`OrtGpuDetector::try_new`], which returns `None` if NPP
/// is not available on the system.
pub struct OrtGpuDetector {
    session: Session,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
    // Pre-computed letterbox parameters (constant for fixed frame dimensions).
    scale: f32,
    #[allow(dead_code)]
    new_w: u32,
    #[allow(dead_code)]
    new_h: u32,
    pad_x: f32,
    pad_y: f32,
    // Pre-allocated GPU scratch buffers.
    rgb_u8: CUdeviceptr,
    /// Separate destination for the 180-degree mirror step. NPP's
    /// `nppiMirror_8u_C3R` with `NPPI_AXIS_BOTH` is *not* safe in-place
    /// (the top half gets overwritten before the bottom half is read),
    /// so a distinct scratch is required. Same size as `rgb_u8`.
    rgb_scratch: CUdeviceptr,
    resized_u8: CUdeviceptr,
    tensor_f32: CUdeviceptr,
    // P010 (10-bit NV12) conversion scratch buffers.
    // Allocated only when the source produces P010 frames.
    // Y plane: width * height bytes, UV plane: width * height/2 bytes.
    nv12_8bit_y: CUdeviceptr,
    nv12_8bit_uv: CUdeviceptr,
    // Cached CUDA device MemoryInfo. Constant for the detector's
    // lifetime; constructing one per inference showed up on the
    // per-frame alloc audit (plan M7 item 5).
    cuda_memory_info: SendMemoryInfo,
}

impl OrtGpuDetector {
    /// Try to create a GPU YOLO detector.
    ///
    /// Returns `Ok(None)` if NPP libraries are not available (e.g. on systems
    /// without NVIDIA GPU or without CUDA toolkit). Returns `Err` for real
    /// failures like missing model file or ORT initialization errors.
    ///
    /// `frame_width`/`frame_height` are the raw camera frame dimensions
    /// (e.g. 3840x2160 for 4K). These must match what the decode pipeline
    /// produces. Letterbox parameters are pre-computed from these dimensions.
    ///
    /// When `is_10bit` is true, additional scratch buffers are allocated for
    /// converting P010 (10-bit NV12) frames to 8-bit before NPP color
    /// conversion.
    pub fn try_new(
        model_path: impl AsRef<Path>,
        frame_width: u32,
        frame_height: u32,
        confidence_threshold: f32,
        labels: Vec<String>,
        is_10bit: bool,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
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
            crate::ort_session::create_ort_session(model_path.as_ref(), labels)?;

        // Pre-compute letterbox parameters.
        let (fw, fh) = (frame_width as f32, frame_height as f32);
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (input_size - new_w) as f32 / 2.0;
        let pad_y = (input_size - new_h) as f32 / 2.0;

        // Allocate GPU scratch buffers (checked arithmetic to prevent overflow).
        let rgb_size = (frame_width as usize)
            .checked_mul(frame_height as usize)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| ort::Error::new("frame dimensions overflow for rgb_size"))?;
        let resized_size = (input_size as usize)
            .checked_mul(input_size as usize)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| ort::Error::new("input dimensions overflow for resized_size"))?;
        let tensor_size = (input_size as usize)
            .checked_mul(input_size as usize)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| ort::Error::new("input dimensions overflow for tensor_size"))?;

        let rgb_u8 = cuda_mem_alloc(rgb_size)?;
        let rgb_scratch = cuda_mem_alloc(rgb_size)?;
        let resized_u8 = cuda_mem_alloc(resized_size)?;
        let tensor_f32 = cuda_mem_alloc(tensor_size)?;

        // Allocate P010 conversion scratch buffers if needed.
        let (nv12_8bit_y, nv12_8bit_uv) = if is_10bit {
            let y_size = frame_width as usize * frame_height as usize;
            let uv_size = frame_width as usize * (frame_height as usize / 2);
            let y = cuda_mem_alloc(y_size)?;
            let uv = cuda_mem_alloc(uv_size)?;
            log::info!(
                "OrtGpuDetector: allocated P010 conversion buffers ({:.1}MB)",
                (y_size + uv_size) as f64 / 1024.0 / 1024.0,
            );
            (y, uv)
        } else {
            (0, 0)
        };

        // Fill resized buffer with grey (114) for letterbox padding.
        cuda_memset_d8(resized_u8, 114, resized_size)?;

        log::info!(
            "OrtGpuDetector ready: input={}x{}, frame={}x{}, scale={:.3}, pad=({:.1},{:.1}), \
             GPU scratch={:.1}MB, 10bit={}",
            input_size,
            input_size,
            frame_width,
            frame_height,
            scale,
            pad_x,
            pad_y,
            (rgb_size + resized_size + tensor_size) as f64 / 1024.0 / 1024.0,
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
            scale,
            new_w,
            new_h,
            pad_x,
            pad_y,
            rgb_u8,
            rgb_scratch,
            resized_u8,
            tensor_f32,
            nv12_8bit_y,
            nv12_8bit_uv,
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
    /// Core inference path shared by the legacy [`GpuDetector`] impl
    /// and the new [`UnifiedDetector`] impl. Returns a typed
    /// [`DetectorError`] so unified-trait consumers can distinguish
    /// "no CUDA context" from "inference failed"; the legacy impl
    /// collapses the error to a log + empty vector for backward
    /// compatibility.
    ///
    /// Each CUDA / NPP / ORT step that previously logged and returned
    /// an empty vec now returns
    /// `Err(DetectorError::InferenceFailed(msg))` preserving the
    /// original error text verbatim.
    fn detect_gpu_raw(
        &mut self,
        camera: CameraId,
        frame: &GpuNv12Frame,
    ) -> Result<Vec<Detection>, DetectorError> {
        let GpuNv12Frame {
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

        // Ensure a CUDA context is current on this thread. The zero-copy
        // frame loop may not have one after NVDEC decode pushes/pops its
        // own context.
        reco_core::interop::cuda::cuda_ensure_context()
            .map_err(|e| DetectorError::InferenceFailed(format!("cuda_ensure_context: {e}")))?;

        // Step 0: Convert P010 (10-bit) to 8-bit NV12 if needed.
        // NPP's NV12->RGB expects 8-bit samples, so we must down-convert
        // first by extracting the high byte of each u16 sample.
        let (nv12_y, nv12_y_pitch, nv12_uv, _nv12_uv_pitch) = if is_10bit {
            reco_core::profile_scope!("p010_to_nv12");
            if self.nv12_8bit_y == 0 || self.nv12_8bit_uv == 0 {
                return Err(DetectorError::InferenceFailed(
                    "P010 frame received but no conversion buffers allocated".into(),
                ));
            }
            // Convert Y plane: width * height samples.
            crate::cuda_kernels::p010_plane_to_nv12(
                y_ptr,
                y_pitch,
                self.nv12_8bit_y,
                width,
                height,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("P010->NV12 Y conversion: {e}")))?;
            // Convert UV plane: width * (height/2) samples.
            // UV plane has width/2 pixel pairs, each 2 u16 values = width u16 samples per row.
            crate::cuda_kernels::p010_plane_to_nv12(
                uv_ptr,
                uv_pitch,
                self.nv12_8bit_uv,
                width,
                height / 2,
            )
            .map_err(|e| {
                DetectorError::InferenceFailed(format!("P010->NV12 UV conversion: {e}"))
            })?;
            // The 8-bit buffers are tightly packed (no pitch padding).
            (
                self.nv12_8bit_y,
                width as usize,
                self.nv12_8bit_uv,
                width as usize,
            )
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
                self.tensor_f32,
                nv12_y_pitch as u32,
                width,
                height,
                self.input_size,
                self.input_size,
                self.pad_x as u32,
                self.pad_y as u32,
                self.scale,
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
                    self.tensor_f32 as *mut c_void,
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
            self.scale,
            self.pad_x,
            self.pad_y,
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

impl UnifiedDetector for OrtGpuDetector {
    fn name(&self) -> &'static str {
        "ort-cuda"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        // CUDA-residency backend: accept `Cuda(GpuNv12Frame)` and
        // route everything else to `UnsupportedFrameKind` so the
        // dispatcher can fall back to a CPU backend for `Cpu(_)`.
        // The wildcard arm keeps this stable against future
        // `#[non_exhaustive]` additions to `DetectorFrame`.
        match frame {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            DetectorFrame::Cuda(gpu_frame) => self.detect_gpu_raw(camera, gpu_frame),
            _ => Err(DetectorError::UnsupportedFrameKind),
        }
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

impl Drop for OrtGpuDetector {
    fn drop(&mut self) {
        // Ensure a CUDA context is current before freeing GPU memory.
        // Drop may run on a different thread than the one that allocated.
        if let Err(e) = cuda_ensure_context() {
            log::warn!("OrtGpuDetector drop: failed to set CUDA context: {e}");
            return;
        }
        // Free GPU scratch buffers. Log errors but don't panic in Drop.
        for (name, ptr) in [
            ("rgb_u8", self.rgb_u8),
            ("rgb_scratch", self.rgb_scratch),
            ("resized_u8", self.resized_u8),
            ("tensor_f32", self.tensor_f32),
            ("nv12_8bit_y", self.nv12_8bit_y),
            ("nv12_8bit_uv", self.nv12_8bit_uv),
        ] {
            if ptr != 0
                && let Err(e) = cuda_mem_free(ptr)
            {
                log::warn!("Failed to free GPU buffer {name}: {e}");
            }
        }
    }
}
