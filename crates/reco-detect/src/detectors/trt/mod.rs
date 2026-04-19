//! Native TensorRT YOLO detector for NVIDIA GPUs.
//!
//! Runs the complete detection pipeline on GPU without ONNX Runtime:
//! NV12 color conversion (NPP), resize with letterbox (NPP), normalize
//! (CUDA kernel), and inference (TensorRT). Only the small detection
//! output (~7KB for `[1, 300, 6]`) is copied back to CPU.
//!
//! Requires a pre-built `.engine` file (architecture-specific). Build one
//! on the target device with:
//!
//! ```bash
//! # Via Ultralytics (recommended):
//! python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt').export(format='engine', half=True)"
//!
//! # Or via trtexec:
//! trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
//! ```

pub mod cuda;
pub mod engine;
mod sys;

use std::ffi::c_void;
use std::path::Path;

use crate::cuda_kernels::normalize_hwc_to_chw;
use crate::npp_interop::{NppiRect, npp_mirror_c3, npp_nv12_to_rgb, npp_resize_c3};
use reco_core::cuda_interop::{
    CUdeviceptr, cuda_ensure_context, cuda_mem_alloc, cuda_mem_free, cuda_memcpy_dtoh,
    cuda_memcpy_htod_2d, cuda_memset_d8, cuda_synchronize,
};
use reco_core::detector::ChromaFormat;
use reco_core::detector::{
    CameraId, Detection, DetectorError, DetectorFrame, GpuNv12Frame, UnifiedDetector,
};

use self::cuda::{CudaBuffer, CudaStream};
use self::engine::{TrtContext, TrtEngine, TrtError};
use super::postprocess;

/// YOLO detector using native TensorRT inference.
///
/// Implements [`GpuDetector`] for the zero-copy pipeline. Reuses the
/// same NPP preprocessing as [`OrtGpuDetector`](super::ort_gpu::OrtGpuDetector)
/// but replaces ORT inference with direct TensorRT API calls.
///
/// # Drop order
///
/// Fields are declared so that GPU resources drop before the TRT
/// context, and the context drops before the engine (Rust drops
/// fields in declaration order).
pub struct TrtGpuDetector {
    // GPU scratch buffers (drop first).
    rgb_u8: CUdeviceptr,
    resized_u8: CUdeviceptr,
    tensor_f32: CUdeviceptr,
    // P010 (10-bit NV12) conversion scratch buffers.
    nv12_8bit_y: CUdeviceptr,
    nv12_8bit_uv: CUdeviceptr,
    // Live-camera NV12 upload buffers. Lazy-allocated on the
    // first `DetectorFrame::Cpu` call. nvarguscamerasrc +
    // appsink delivers CPU-resident NV12; a ~1-2 ms H2D memcpy
    // per frame lets that flow reuse the same TRT inference
    // path as NVDEC zero-copy. Zero-cost when the caller only
    // sends Cuda frames (buffers stay 0).
    cpu_upload_y: CUdeviceptr,
    cpu_upload_uv: CUdeviceptr,
    // TRT output buffer (drop before context).
    output_buf: CudaBuffer,
    output_host: Vec<u8>,
    output_floats: usize,
    // TRT inference (context drops before engine).
    stream: CudaStream,
    context: TrtContext,
    engine: TrtEngine,
    // Binding indices.
    input_idx: usize,
    output_idx: usize,
    num_bindings: usize,
    // Preprocessing parameters (identical to OrtGpuDetector).
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
    scale: f32,
    new_w: u32,
    new_h: u32,
    pad_x: f32,
    pad_y: f32,
    frame_width: u32,
    frame_height: u32,
}

/// SAFETY: The detector's GPU resources are accessed only from one
/// thread at a time via the GpuDetector trait (which takes &mut self).
/// CUDA context management via cuda_ensure_context handles thread safety.
unsafe impl Send for TrtGpuDetector {}

impl TrtGpuDetector {
    /// Create a TensorRT YOLO detector from a pre-built `.engine` file.
    ///
    /// Returns `Ok(None)` if NPP is not available. Returns `Err` for
    /// real failures (missing engine file, TRT init errors, etc.).
    ///
    /// `labels` are class names for the model. If empty, detections
    /// will have numeric class IDs only. Pass labels from a `.labels`
    /// sidecar file or from the original ONNX model metadata.
    ///
    /// When `is_10bit` is true, additional scratch buffers are allocated
    /// for converting P010 (10-bit NV12) frames to 8-bit before NPP
    /// color conversion.
    pub fn try_new(
        engine_path: impl AsRef<Path>,
        frame_width: u32,
        frame_height: u32,
        confidence_threshold: f32,
        labels: Vec<String>,
        is_10bit: bool,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if !crate::npp_interop::is_npp_available() {
            log::warn!("TrtGpuDetector: NPP not available, GPU detection disabled");
            return Ok(None);
        }

        cuda_ensure_context()?;
        log::info!("TrtGpuDetector: CUDA context ready, loading engine...");

        // Load TRT engine.
        let engine_path = engine_path.as_ref();
        let engine = TrtEngine::from_file(
            engine_path
                .to_str()
                .ok_or_else(|| TrtError::Runtime("invalid engine path".into()))?,
        )?;

        let bindings = engine.bindings()?;
        log::info!(
            "TRT engine loaded: {} bindings from {}",
            bindings.len(),
            engine_path.display()
        );

        // Find input and output bindings.
        let input_idx = bindings
            .iter()
            .position(|b| b.is_input)
            .ok_or_else(|| TrtError::Runtime("no input binding found".into()))?;
        let output_idx = bindings
            .iter()
            .position(|b| !b.is_input)
            .ok_or_else(|| TrtError::Runtime("no output binding found".into()))?;

        // Extract model input size from input dims [1, 3, H, W].
        let input_dims = &bindings[input_idx].dims;
        let input_size = if input_dims.len() == 4 {
            input_dims[2] as u32 // H dimension
        } else {
            return Err(TrtError::Runtime(format!(
                "unexpected input dims: {input_dims:?}, expected [1, 3, H, W]"
            ))
            .into());
        };

        // Output shape: [1, N, 6] for end-to-end NMS YOLO.
        let output_dims = &bindings[output_idx].dims;
        let output_floats = output_dims
            .iter()
            .map(|&d| d.max(1) as usize)
            .product::<usize>();
        let output_byte_size = bindings[output_idx].byte_size;

        log::info!(
            "TRT input: {}x{} (binding '{}'), output: {:?} ({} bytes, binding '{}')",
            input_size,
            input_size,
            bindings[input_idx].name,
            output_dims,
            output_byte_size,
            bindings[output_idx].name,
        );

        // Pre-compute letterbox parameters.
        let (fw, fh) = (frame_width as f32, frame_height as f32);
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (input_size - new_w) as f32 / 2.0;
        let pad_y = (input_size - new_h) as f32 / 2.0;

        // Allocate GPU scratch buffers (same as OrtGpuDetector).
        let rgb_size = (frame_width as usize)
            .checked_mul(frame_height as usize)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| TrtError::Runtime("frame dimensions overflow".into()))?;
        let resized_size = (input_size as usize)
            .checked_mul(input_size as usize)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| TrtError::Runtime("input dimensions overflow".into()))?;
        let tensor_size = resized_size * 4; // f32

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
                "TrtGpuDetector: allocated P010 conversion buffers ({:.1}MB)",
                (y_size + uv_size) as f64 / 1024.0 / 1024.0,
            );
            (y, uv)
        } else {
            (0, 0)
        };

        // Fill resized buffer with grey (114) for letterbox padding.
        cuda_memset_d8(resized_u8, 114, resized_size)?;

        // Allocate TRT output buffer and host-side copy.
        let output_buf = CudaBuffer::new(output_byte_size)?;
        let output_host = vec![0u8; output_byte_size];

        // Create execution context and stream.
        let context = engine.create_context()?;
        let stream = CudaStream::new()?;

        let num_bindings = bindings.len();

        log::info!(
            "TrtGpuDetector ready: input={}x{}, frame={}x{}, scale={:.3}, \
             pad=({:.1},{:.1}), GPU scratch={:.1}MB, 10bit={}",
            input_size,
            input_size,
            frame_width,
            frame_height,
            scale,
            pad_x,
            pad_y,
            (rgb_size + resized_size + tensor_size + output_byte_size) as f64 / 1024.0 / 1024.0,
            is_10bit,
        );

        let mut detector = Self {
            rgb_u8,
            resized_u8,
            tensor_f32,
            nv12_8bit_y,
            nv12_8bit_uv,
            cpu_upload_y: 0,
            cpu_upload_uv: 0,
            output_buf,
            output_host,
            output_floats,
            stream,
            context,
            engine,
            input_idx,
            output_idx,
            num_bindings,
            input_size,
            confidence_threshold,
            labels,
            scale,
            new_w,
            new_h,
            pad_x,
            pad_y,
            frame_width,
            frame_height,
        };

        // Warmup inference to trigger TRT engine optimization.
        log::info!("TrtGpuDetector: starting warmup...");
        detector.warmup()?;
        log::info!("TrtGpuDetector: warmup done");

        Ok(Some(detector))
    }

    /// Run a warmup inference with zero-filled input.
    fn warmup(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let sz = self.input_size as usize;
        let tensor_size = 3 * sz * sz * 4; // f32 bytes
        cuda_memset_d8(self.tensor_f32, 0, tensor_size)?;

        let mut binding_ptrs = self.build_binding_ptrs();
        cuda_synchronize()?;
        self.context.enqueue(&mut binding_ptrs, &self.stream)?;
        self.stream.synchronize()?;

        log::info!("TrtGpuDetector: warmup inference complete");
        Ok(())
    }

    /// Build the binding pointer array for enqueue.
    fn build_binding_ptrs(&self) -> Vec<*mut c_void> {
        let mut ptrs = vec![std::ptr::null_mut(); self.num_bindings];
        // Driver API CUdeviceptr (u64) -> runtime API *mut c_void.
        // These are numerically identical in CUDA's unified address space.
        ptrs[self.input_idx] = self.tensor_f32 as *mut c_void;
        ptrs[self.output_idx] = self.output_buf.as_ptr();
        ptrs
    }

    /// Access class labels for external label resolution.
    pub fn class_names(&self) -> &[String] {
        &self.labels
    }
}

impl TrtGpuDetector {
    /// Core inference path shared by the legacy [`GpuDetector`] impl
    /// and the new [`UnifiedDetector`] impl. Returns a typed
    /// [`DetectorError`] so unified-trait consumers can react to CUDA
    /// / NPP / TensorRT failures; the legacy impl collapses the error
    /// to a log + empty vector for backward compatibility.
    ///
    /// Each CUDA / NPP / TRT step that previously used
    /// `log::error!; return Vec::new()` now returns
    /// `Err(DetectorError::InferenceFailed("stage: {err}"))` so
    /// telemetry dashboards can still break down failures by origin.
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
        reco_core::profile_scope!("trt_yolo_detect");

        // Ensure CUDA context is current on this thread.
        cuda_ensure_context()
            .map_err(|e| DetectorError::InferenceFailed(format!("cuda_ensure_context: {e}")))?;

        // Step 0: Convert P010 (10-bit) to 8-bit NV12 if needed.
        let (nv12_y, nv12_y_pitch, nv12_uv, nv12_uv_pitch) = if is_10bit {
            reco_core::profile_scope!("p010_to_nv12");
            if self.nv12_8bit_y == 0 || self.nv12_8bit_uv == 0 {
                return Err(DetectorError::InferenceFailed(
                    "P010 frame received but no conversion buffers allocated".into(),
                ));
            }
            crate::cuda_kernels::p010_plane_to_nv12(
                y_ptr,
                y_pitch,
                self.nv12_8bit_y,
                width,
                height,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("P010->NV12 Y conversion: {e}")))?;
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
            (
                self.nv12_8bit_y,
                width as usize,
                self.nv12_8bit_uv,
                width as usize,
            )
        } else {
            (y_ptr, y_pitch, uv_ptr, uv_pitch)
        };

        // Step 1: NV12 -> packed RGB u8 via NPP (identical to OrtGpuDetector).
        {
            reco_core::profile_scope!("npp_nv12_to_rgb");
            npp_nv12_to_rgb(
                nv12_y,
                nv12_y_pitch,
                nv12_uv,
                nv12_uv_pitch,
                self.rgb_u8,
                width,
                height,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("NPP NV12->RGB: {e}")))?;
        }

        // Step 1b: Flip 180 degrees if the source has rotation metadata.
        if rotation == 180 {
            reco_core::profile_scope!("npp_mirror_180");
            npp_mirror_c3(self.rgb_u8, self.rgb_u8, width, height).map_err(|e| {
                DetectorError::InferenceFailed(format!("NPP mirror (rotation=180): {e}"))
            })?;
        }

        // Step 2: Resize to letterboxed region (identical to OrtGpuDetector).
        {
            reco_core::profile_scope!("npp_resize");
            let is = self.input_size;
            let resized_size = (is as usize) * (is as usize) * 3;
            cuda_memset_d8(self.resized_u8, 114, resized_size)
                .map_err(|e| DetectorError::InferenceFailed(format!("grey fill: {e}")))?;

            let dst_roi = NppiRect {
                x: self.pad_x as i32,
                y: self.pad_y as i32,
                width: self.new_w as i32,
                height: self.new_h as i32,
            };

            npp_resize_c3(self.rgb_u8, width, height, self.resized_u8, is, is, dst_roi)
                .map_err(|e| DetectorError::InferenceFailed(format!("NPP resize: {e}")))?;
        }

        // Step 3: Normalize u8 HWC -> f32 CHW (identical to OrtGpuDetector).
        {
            reco_core::profile_scope!("cuda_normalize");
            normalize_hwc_to_chw(
                self.resized_u8,
                self.tensor_f32,
                self.input_size,
                self.input_size,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("CUDA normalize: {e}")))?;
        }

        // Step 4: TensorRT inference (replaces ORT).
        {
            reco_core::profile_scope!("trt_inference");

            // Synchronize default stream (preprocessing runs on NULL stream)
            // before enqueuing TRT on our named stream.
            cuda_synchronize()
                .map_err(|e| DetectorError::InferenceFailed(format!("CUDA sync pre-TRT: {e}")))?;

            let mut binding_ptrs = self.build_binding_ptrs();
            self.context
                .enqueue(&mut binding_ptrs, &self.stream)
                .map_err(|e| DetectorError::InferenceFailed(format!("TRT enqueue: {e}")))?;

            self.stream
                .synchronize()
                .map_err(|e| DetectorError::InferenceFailed(format!("TRT stream sync: {e}")))?;
        }

        // Step 5: Copy output to host and postprocess.
        {
            // Copy the small output buffer (~7KB) to CPU.
            self.output_buf
                .copy_to_host(&mut self.output_host, &self.stream)
                .map_err(|e| DetectorError::InferenceFailed(format!("TRT output D2H: {e}")))?;
            self.stream
                .synchronize()
                .map_err(|e| DetectorError::InferenceFailed(format!("TRT output sync: {e}")))?;
        }

        // Reinterpret output bytes as f32 slice.
        // SAFETY: output_host is properly aligned (Vec<u8> from vec![0u8; ...])
        // and output_floats * 4 == output_host.len().
        let output_data: &[f32] = unsafe {
            std::slice::from_raw_parts(self.output_host.as_ptr() as *const f32, self.output_floats)
        };

        // Output is [1, N, 6] - extract N from the dims.
        let n = self.output_floats / 6;

        let detections = postprocess(
            output_data,
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
                "TRT camera {:?}: {} detection(s) - {}",
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

impl UnifiedDetector for TrtGpuDetector {
    fn name(&self) -> &'static str {
        "tensorrt-native"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        // CUDA-residency backend: accept `Cuda(GpuNv12Frame)` for
        // the zero-copy fast path, AND accept `Cpu(RawFrame)` with
        // NV12 chroma via a transparent host→device upload so live
        // camera sources (nvarguscamerasrc + appsink delivering
        // CPU-resident NV12) can drive detection without a parallel
        // CPU backend. Yuv420p Cpu frames and other variants fall
        // through to `UnsupportedFrameKind` for the dispatcher to
        // route elsewhere.
        match frame {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            DetectorFrame::Cuda(gpu_frame) => self.detect_gpu_raw(camera, gpu_frame),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            DetectorFrame::Cpu(raw) => match &raw.chroma {
                ChromaFormat::Nv12 { uv } => {
                    self.detect_cpu_nv12_upload(camera, raw.y, uv, raw.width, raw.height)
                }
                ChromaFormat::Yuv420p { .. } => Err(DetectorError::UnsupportedFrameKind),
            },
            _ => Err(DetectorError::UnsupportedFrameKind),
        }
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl TrtGpuDetector {
    /// Live-camera NV12 detection path: copy a host-resident NV12
    /// frame to CUDA buffers, then reuse [`Self::detect_gpu_raw`].
    /// Buffers are allocated lazily on first call and reused for
    /// the lifetime of the detector; subsequent calls pay only the
    /// host→device memcpy cost (~1-2 ms for a 1080p frame on Orin
    /// Nano). This is the primary path for CSI-origin frames
    /// (nvarguscamerasrc + appsink); NVDEC-sourced frames continue
    /// to use the zero-copy [`DetectorFrame::Cuda`] route.
    fn detect_cpu_nv12_upload(
        &mut self,
        camera: CameraId,
        y_host: &[u8],
        uv_host: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Detection>, DetectorError> {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let y_bytes = width_usize * height_usize;
        let uv_bytes = width_usize * (height_usize / 2);

        if y_host.len() < y_bytes || uv_host.len() < uv_bytes {
            return Err(DetectorError::InferenceFailed(format!(
                "CPU NV12 upload: plane sizes too small (Y {}<{}, UV {}<{})",
                y_host.len(),
                y_bytes,
                uv_host.len(),
                uv_bytes
            )));
        }

        cuda_ensure_context()
            .map_err(|e| DetectorError::InferenceFailed(format!("cuda_ensure_context: {e}")))?;

        // Lazy-allocate upload buffers. Tight pitch (pitch == width).
        if self.cpu_upload_y == 0 {
            self.cpu_upload_y = cuda_mem_alloc(y_bytes)
                .map_err(|e| DetectorError::InferenceFailed(format!("cpu_upload_y alloc: {e}")))?;
            log::info!(
                "TrtGpuDetector: allocated CPU-upload buffers ({} B Y + {} B UV) for live-camera NV12 path",
                y_bytes,
                uv_bytes
            );
        }
        if self.cpu_upload_uv == 0 {
            self.cpu_upload_uv = cuda_mem_alloc(uv_bytes)
                .map_err(|e| DetectorError::InferenceFailed(format!("cpu_upload_uv alloc: {e}")))?;
        }

        cuda_memcpy_htod_2d(
            self.cpu_upload_y,
            width_usize,
            y_host.as_ptr(),
            width_usize,
            width_usize,
            height_usize,
        )
        .map_err(|e| DetectorError::InferenceFailed(format!("Y H2D memcpy: {e}")))?;
        cuda_memcpy_htod_2d(
            self.cpu_upload_uv,
            width_usize,
            uv_host.as_ptr(),
            width_usize,
            width_usize,
            height_usize / 2,
        )
        .map_err(|e| DetectorError::InferenceFailed(format!("UV H2D memcpy: {e}")))?;

        let gpu_frame = GpuNv12Frame {
            y_ptr: self.cpu_upload_y,
            uv_ptr: self.cpu_upload_uv,
            y_pitch: width_usize,
            uv_pitch: width_usize,
            width,
            height,
            rotation: 0,
            is_10bit: false,
        };
        self.detect_gpu_raw(camera, &gpu_frame)
    }
}

impl Drop for TrtGpuDetector {
    fn drop(&mut self) {
        // Ensure a CUDA context is current before freeing GPU memory.
        // Drop may run on a different thread than the one that allocated.
        if let Err(e) = cuda_ensure_context() {
            log::warn!("TrtGpuDetector drop: failed to set CUDA context: {e}");
            return;
        }
        for (name, ptr) in [
            ("rgb_u8", self.rgb_u8),
            ("resized_u8", self.resized_u8),
            ("tensor_f32", self.tensor_f32),
            ("nv12_8bit_y", self.nv12_8bit_y),
            ("nv12_8bit_uv", self.nv12_8bit_uv),
            ("cpu_upload_y", self.cpu_upload_y),
            ("cpu_upload_uv", self.cpu_upload_uv),
        ] {
            if ptr != 0 {
                if let Err(e) = cuda_mem_free(ptr) {
                    log::warn!("Failed to free GPU buffer {name}: {e}");
                }
            }
        }
    }
}
