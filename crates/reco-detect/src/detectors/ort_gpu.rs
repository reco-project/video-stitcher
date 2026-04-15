//! GPU-resident YOLO detector for the zero-copy pipeline.
//!
//! Runs the entire detection pipeline on GPU: NV12 color conversion (NPP),
//! resize with letterbox padding (NPP), normalize + CHW transpose (CUDA kernel),
//! and inference (ORT TensorRT/CUDA EP). Only the small detection output
//! (~7KB for `[1, 300, 6]`) is read back to CPU.

use std::ffi::c_void;
use std::path::Path;

use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{Shape, TensorRefMut};
use reco_core::cuda_interop::{
    CUdeviceptr, cuda_ensure_context, cuda_mem_alloc, cuda_mem_free, cuda_memset_d8,
};
use reco_core::cuda_kernels::normalize_hwc_to_chw;
use reco_core::detector::{CameraId, Detection, GpuDetector, GpuNv12Frame};
use reco_core::npp_interop::{NppiRect, npp_mirror_c3, npp_nv12_to_rgb, npp_resize_c3};

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
    new_w: u32,
    new_h: u32,
    pad_x: f32,
    pad_y: f32,
    // Pre-allocated GPU scratch buffers.
    rgb_u8: CUdeviceptr,
    resized_u8: CUdeviceptr,
    tensor_f32: CUdeviceptr,
    // P010 (10-bit NV12) conversion scratch buffers.
    // Allocated only when the source produces P010 frames.
    // Y plane: width * height bytes, UV plane: width * height/2 bytes.
    nv12_8bit_y: CUdeviceptr,
    nv12_8bit_uv: CUdeviceptr,
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
        if !reco_core::npp_interop::is_npp_available() {
            log::warn!("OrtGpuDetector: NPP not available, GPU detection disabled");
            return Ok(None);
        }

        // GPU detection requires TensorRT EP to handle CUDA device pointers.
        // Without it, ORT falls back to CPU EP which segfaults on GPU memory.
        if !cfg!(feature = "tensorrt") {
            log::warn!(
                "OrtGpuDetector: TensorRT feature not enabled, GPU detection disabled. \
                 Build with --features tensorrt for zero-copy GPU inference."
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
            resized_u8,
            tensor_f32,
            nv12_8bit_y,
            nv12_8bit_uv,
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

impl GpuDetector for OrtGpuDetector {
    fn detect_gpu(&mut self, camera: CameraId, frame: &GpuNv12Frame) -> Vec<Detection> {
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
        if let Err(e) = reco_core::cuda_interop::cuda_ensure_context() {
            log::error!("GPU detect: failed to set CUDA context: {e}");
            return Vec::new();
        }

        // Step 0: Convert P010 (10-bit) to 8-bit NV12 if needed.
        // NPP's NV12->RGB expects 8-bit samples, so we must down-convert
        // first by extracting the high byte of each u16 sample.
        let (nv12_y, nv12_y_pitch, nv12_uv, nv12_uv_pitch) = if is_10bit {
            reco_core::profile_scope!("p010_to_nv12");
            if self.nv12_8bit_y == 0 || self.nv12_8bit_uv == 0 {
                log::error!("P010 frame received but no conversion buffers allocated");
                return Vec::new();
            }
            // Convert Y plane: width * height samples.
            if let Err(e) = reco_core::cuda_kernels::p010_plane_to_nv12(
                y_ptr,
                y_pitch,
                self.nv12_8bit_y,
                width,
                height,
            ) {
                log::error!("P010->NV12 Y conversion failed: {e}");
                return Vec::new();
            }
            // Convert UV plane: width * (height/2) samples.
            // UV plane has width/2 pixel pairs, each 2 u16 values = width u16 samples per row.
            if let Err(e) = reco_core::cuda_kernels::p010_plane_to_nv12(
                uv_ptr,
                uv_pitch,
                self.nv12_8bit_uv,
                width,
                height / 2,
            ) {
                log::error!("P010->NV12 UV conversion failed: {e}");
                return Vec::new();
            }
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

        // Step 1: NV12 -> packed RGB u8 via NPP.
        {
            reco_core::profile_scope!("npp_nv12_to_rgb");
            if let Err(e) = npp_nv12_to_rgb(
                nv12_y,
                nv12_y_pitch,
                nv12_uv,
                nv12_uv_pitch,
                self.rgb_u8,
                width,
                height,
            ) {
                log::error!("NPP NV12->RGB failed: {e}");
                return Vec::new();
            }
        }

        // Step 1b: Flip 180 degrees if the source has rotation metadata.
        // NVDEC decodes without applying rotation; the render shader flips
        // UV for display, but the detector sees raw upside-down frames.
        // Mirror the RGB buffer in-place via NPP before resize.
        if rotation == 180 {
            reco_core::profile_scope!("npp_mirror_180");
            if let Err(e) = npp_mirror_c3(self.rgb_u8, self.rgb_u8, width, height) {
                log::error!("NPP mirror (rotation=180) failed: {e}");
                return Vec::new();
            }
        }

        // Step 2: Resize to letterboxed region within the pre-filled grey buffer.
        // Re-fill grey padding each frame (NPP resize only writes the dst_roi region).
        {
            reco_core::profile_scope!("npp_resize");
            let is = self.input_size;
            let resized_size = (is as usize) * (is as usize) * 3;
            if let Err(e) = cuda_memset_d8(self.resized_u8, 114, resized_size) {
                log::error!("Grey fill failed: {e}");
                return Vec::new();
            }

            let pad_x_i = self.pad_x as u32;
            let pad_y_i = self.pad_y as u32;
            let dst_roi = NppiRect {
                x: pad_x_i as i32,
                y: pad_y_i as i32,
                width: self.new_w as i32,
                height: self.new_h as i32,
            };

            if let Err(e) =
                npp_resize_c3(self.rgb_u8, width, height, self.resized_u8, is, is, dst_roi)
            {
                log::error!("NPP resize failed: {e}");
                return Vec::new();
            }
        }

        // Step 3: Normalize u8 HWC -> f32 CHW with /255.0 via CUDA kernel.
        {
            reco_core::profile_scope!("cuda_normalize");
            if let Err(e) = normalize_hwc_to_chw(
                self.resized_u8,
                self.tensor_f32,
                self.input_size,
                self.input_size,
            ) {
                log::error!("CUDA normalize kernel failed: {e}");
                return Vec::new();
            }
        }

        // Step 4: Wrap GPU buffer as ORT tensor and run inference.
        let outputs = {
            reco_core::profile_scope!("gpu_ort_inference");

            let sz = self.input_size as i64;
            let memory_info = match MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            ) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Failed to create CUDA MemoryInfo: {e}");
                    return Vec::new();
                }
            };

            let tensor: TensorRefMut<'_, f32> = match unsafe {
                TensorRefMut::from_raw(
                    memory_info,
                    self.tensor_f32 as *mut c_void,
                    Shape::new([1i64, 3, sz, sz]),
                )
            } {
                Ok(t) => t,
                Err(e) => {
                    log::error!("Failed to create GPU tensor: {e}");
                    return Vec::new();
                }
            };

            match self.session.run(ort::inputs![tensor]) {
                Ok(o) => o,
                Err(e) => {
                    log::error!("GPU YOLO inference failed: {e}");
                    return Vec::new();
                }
            }
        };

        // Step 5: Extract output and postprocess on CPU.
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
            self.scale,
            self.pad_x,
            self.pad_y,
            width,
            height,
        );

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

        detections
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
