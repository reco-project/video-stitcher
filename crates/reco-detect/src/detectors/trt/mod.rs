//! Native TensorRT detection with NPP preprocessing.
//!
//! Engines must expose static, linear FP32 tensors: `[1, 3, S, S]` input
//! and `[1, N, 6]` end-to-end detection output. Internal engine precision
//! may differ. CUDA inputs stay on device; only detections return to CPU.

mod cuda;
mod engine;
mod sys;
#[cfg(test)]
mod tests;

use std::{ffi::c_void, path::Path};

use reco_core::detect::detector::{
    ChromaPlanes, CudaNv12Frame, Detection, Detector, DetectorError, DetectorFrame,
};
use reco_core::geometry::CameraIndex;
use reco_core::interop::cuda::{CUdeviceptr, cuda_ensure_context, cuda_memcpy_htod_2d};

use self::cuda::{CudaBuffer, CudaStream};
use self::engine::{ModelLayout, TrtContext, TrtEngine};
use crate::{cuda_kernels, npp_interop as npp};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
struct InputLayout {
    size: (u32, u32),
    scale: f32,
    resized: (u32, u32),
    padding: (u32, u32),
}

impl InputLayout {
    fn new(size: (u32, u32), model_size: u32) -> Self {
        let scale = (model_size as f32 / size.0 as f32).min(model_size as f32 / size.1 as f32);
        let resized = (
            (size.0 as f32 * scale).round().max(1.0) as u32,
            (size.1 as f32 * scale).round().max(1.0) as u32,
        );
        Self {
            size,
            scale,
            resized,
            padding: ((model_size - resized.0) / 2, (model_size - resized.1) / 2),
        }
    }

    fn roi(self) -> npp::NppiRect {
        npp::NppiRect {
            x: self.padding.0 as i32,
            y: self.padding.1 as i32,
            width: self.resized.0 as i32,
            height: self.resized.1 as i32,
        }
    }
}

struct Nv12Scratch {
    rgb: CudaBuffer,
    mirrored: CudaBuffer,
    resized: CudaBuffer,
}

/// Native TensorRT detector with per-camera geometry and shared inference buffers.
///
/// Scratch storage is allocated lazily for the largest declared source and reused
/// across cameras. Pre-letterboxed RGBA skips source-sized preprocessing storage.
pub struct TrtGpuDetector {
    // Drain inference before releasing any referenced allocations or context.
    stream: CudaStream,
    context: TrtContext,
    tensor: CudaBuffer,
    output: CudaBuffer,
    output_host: Vec<f32>,
    model: ModelLayout,
    inputs: Vec<InputLayout>,
    max_pixels: usize,
    nv12: Option<Nv12Scratch>,
    p010: Option<CudaBuffer>,
    upload_nv12: Option<CudaBuffer>,
    upload_rgba: Option<CudaBuffer>,
    resized_rgba: Option<CudaBuffer>,
    confidence_threshold: f32,
    labels: Vec<String>,
}

fn max_pixels(sizes: &[(u32, u32)]) -> Result<usize> {
    if sizes.is_empty()
        || sizes.iter().any(|&(w, h)| {
            w == 0
                || h == 0
                || w > reco_core::calibration::MAX_DIM
                || h > reco_core::calibration::MAX_DIM
        })
    {
        return Err(
            "TensorRT requires nonempty source sizes within the supported dimensions".into(),
        );
    }
    Ok(sizes
        .iter()
        .map(|&(w, h)| w as usize * h as usize)
        .max()
        .unwrap())
}

fn validate_plane(ptr: CUdeviceptr, pitch: usize, row_bytes: usize, rows: u32) -> Result<()> {
    if ptr == 0
        || pitch < row_bytes
        || pitch > i32::MAX as usize
        || pitch
            .checked_mul(rows as usize)
            .is_none_or(|bytes| bytes > u32::MAX as usize)
    {
        return Err("invalid TensorRT input pointer or plane pitch".into());
    }
    Ok(())
}

fn validate_nv12(size: (u32, u32)) -> Result<()> {
    if !size.0.is_multiple_of(2) || !size.1.is_multiple_of(2) {
        return Err("NV12 detection requires even source dimensions".into());
    }
    Ok(())
}

// Always drain NPP, including partial submission failures, before normalization or reuse.
fn npp_work(work: impl FnOnce(*mut c_void) -> Result<()>) -> Result<()> {
    let stream = npp::stream()?;
    let submitted = work(stream);
    let completed = cuda::synchronize(stream);
    submitted.and(completed)
}

impl TrtGpuDetector {
    /// Load a static end-to-end `.engine` detector for the declared camera sizes.
    ///
    /// Returns `Ok(None)` if NPP is unavailable. Model bindings are validated before
    /// inference buffers are allocated. Labels may be empty for numeric class IDs.
    pub fn try_new(
        engine_path: impl AsRef<Path>,
        source_sizes: &[(u32, u32)],
        confidence_threshold: f32,
        labels: Vec<String>,
    ) -> Result<Option<Self>> {
        let max_pixels = max_pixels(source_sizes)?;
        if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
            return Err("TensorRT confidence threshold must be between 0 and 1".into());
        }
        if !npp::is_npp_available() {
            return Ok(None);
        }
        cuda_ensure_context()?;
        let engine = TrtEngine::from_file(engine_path.as_ref())?;
        let model = ModelLayout::new(&engine.bindings()?)?;
        let inputs = source_sizes
            .iter()
            .map(|&size| InputLayout::new(size, model.size))
            .collect();
        let mut detector = Self {
            stream: CudaStream::new()?,
            context: engine.into_context()?,
            tensor: CudaBuffer::new(model.input_bytes)?,
            output: CudaBuffer::new(model.output_elements * size_of::<f32>())?,
            output_host: vec![0.0; model.output_elements],
            model,
            inputs,
            max_pixels,
            nv12: None,
            p010: None,
            upload_nv12: None,
            upload_rgba: None,
            resized_rgba: None,
            confidence_threshold,
            labels,
        };
        detector.tensor.fill(0, detector.stream.as_ptr())?;
        detector.infer()?;
        Ok(Some(detector))
    }

    fn layout(&self, camera: CameraIndex, size: (u32, u32)) -> Result<InputLayout> {
        let layout = self
            .inputs
            .get(camera)
            .ok_or("TensorRT camera index is not configured")?;
        if layout.size != size {
            return Err(format!(
                "TensorRT camera {camera} expects {:?}, got {size:?}",
                layout.size
            )
            .into());
        }
        Ok(*layout)
    }

    fn infer(&mut self) -> Result<()> {
        reco_core::profile_scope!("trt_inference");
        let mut bindings = [
            self.tensor.ptr() as *mut c_void,
            self.output.ptr() as *mut c_void,
        ];
        if self.model.input_index == 1 {
            bindings.swap(0, 1);
        }
        // SAFETY: Validated shapes match these owned buffers. Stream drains before drop.
        if let Err(error) = unsafe { self.context.enqueue(&mut bindings, &self.stream) } {
            self.stream.synchronize()?;
            return Err(error);
        }
        self.output
            .copy_to_host(&mut self.output_host, &self.stream)
    }

    fn preprocess_nv12(&mut self, layout: InputLayout, frame: &CudaNv12Frame) -> Result<()> {
        validate_nv12(layout.size)?;
        let (width, height) = layout.size;
        let row_bytes = width as usize * if frame.is_10bit { 2 } else { 1 };
        validate_plane(frame.y_ptr, frame.y_pitch, row_bytes, height)?;
        validate_plane(frame.uv_ptr, frame.uv_pitch, row_bytes, height / 2)?;
        if frame.is_10bit
            && [
                frame.y_ptr,
                frame.uv_ptr,
                frame.y_pitch as u64,
                frame.uv_pitch as u64,
            ]
            .iter()
            .any(|value| !value.is_multiple_of(2))
        {
            return Err("P010 input pointers and pitches must be aligned to 16-bit samples".into());
        }
        if !matches!(frame.rotation, 0 | 180) {
            return Err("TensorRT NV12 rotation must be 0 or 180 degrees".into());
        }
        if !frame.is_10bit && frame.y_pitch != frame.uv_pitch {
            return Err("NPP NV12 conversion requires equal Y and UV pitches".into());
        }
        if self.nv12.is_none() {
            self.nv12 = Some(Nv12Scratch {
                rgb: CudaBuffer::new(self.max_pixels * 3)?,
                mirrored: CudaBuffer::new(self.max_pixels * 3)?,
                resized: CudaBuffer::new(self.model.input_bytes / size_of::<f32>())?,
            });
        }
        let (y, uv, pitch) = if frame.is_10bit {
            if self.p010.is_none() {
                self.p010 = Some(CudaBuffer::new(self.max_pixels * 3 / 2)?);
            }
            let y = self.p010.as_ref().unwrap().ptr();
            let uv = y + self.max_pixels as u64;
            cuda_kernels::p010_plane_to_nv12(frame.y_ptr, frame.y_pitch, y, width, height)?;
            cuda_kernels::p010_plane_to_nv12(frame.uv_ptr, frame.uv_pitch, uv, width, height / 2)?;
            (y, uv, width as usize)
        } else {
            (frame.y_ptr, frame.uv_ptr, frame.y_pitch)
        };
        let scratch = self.nv12.as_ref().unwrap();
        npp_work(|stream| {
            scratch.resized.fill(114, stream)?;
            npp::npp_nv12_to_rgb(y, pitch, uv, pitch, scratch.rgb.ptr(), width, height)?;
            let rgb = if frame.rotation == 180 {
                npp::npp_mirror_c3(scratch.rgb.ptr(), scratch.mirrored.ptr(), width, height)?;
                scratch.mirrored.ptr()
            } else {
                scratch.rgb.ptr()
            };
            npp::npp_resize_c3(
                rgb,
                width,
                height,
                scratch.resized.ptr(),
                self.model.size,
                self.model.size,
                layout.roi(),
            )?;
            Ok(())
        })?;
        cuda_kernels::normalize_hwc_to_chw(
            scratch.resized.ptr(),
            self.tensor.ptr(),
            self.model.size,
            self.model.size,
        )?;
        Ok(())
    }

    fn preprocess_rgba(
        &mut self,
        layout: InputLayout,
        ptr: CUdeviceptr,
        pitch: usize,
    ) -> Result<()> {
        validate_plane(ptr, pitch, layout.size.0 as usize * 4, layout.size.1)?;
        if self.resized_rgba.is_none() {
            self.resized_rgba = Some(CudaBuffer::new(
                self.model.size as usize * self.model.size as usize * 4,
            )?);
        }
        let resized = self.resized_rgba.as_ref().unwrap();
        npp_work(|stream| {
            resized.fill(114, stream)?;
            npp::npp_resize_c4(
                ptr,
                pitch,
                layout.size,
                resized.ptr(),
                self.model.size,
                self.model.size,
                layout.roi(),
            )?;
            Ok(())
        })?;
        cuda_kernels::normalize_rgba_to_chw(
            resized.ptr(),
            self.model.size as usize * 4,
            self.tensor.ptr(),
            self.model.size,
            self.model.size,
        )?;
        Ok(())
    }

    fn preprocess(&mut self, layout: InputLayout, frame: &DetectorFrame<'_>) -> Result<()> {
        let (width, height) = layout.size;
        match frame {
            DetectorFrame::CudaNv12(frame) => self.preprocess_nv12(layout, frame),
            DetectorFrame::Yuv(raw) => {
                let ChromaPlanes::Nv12 { uv } = raw.chroma else {
                    unreachable!("checked by detect")
                };
                validate_nv12(layout.size)?;
                let pixels = width as usize * height as usize;
                if raw.y.len() < pixels || uv.len() < pixels / 2 {
                    return Err("TensorRT NV12 input planes are too small".into());
                }
                if self.upload_nv12.is_none() {
                    self.upload_nv12 = Some(CudaBuffer::new(self.max_pixels * 3 / 2)?);
                }
                let y_ptr = self.upload_nv12.as_ref().unwrap().ptr();
                let uv_ptr = y_ptr + self.max_pixels as u64;
                cuda_memcpy_htod_2d(
                    y_ptr,
                    width as usize,
                    raw.y.as_ptr(),
                    width as usize,
                    width as usize,
                    height as usize,
                )?;
                cuda_memcpy_htod_2d(
                    uv_ptr,
                    width as usize,
                    uv.as_ptr(),
                    width as usize,
                    width as usize,
                    height as usize / 2,
                )?;
                // Pageable uploads may return before DMA completes. NPP's nonblocking
                // stream does not wait for this default-stream transfer implicitly.
                cuda::synchronize(std::ptr::null_mut())?;
                self.preprocess_nv12(
                    layout,
                    &CudaNv12Frame {
                        y_ptr,
                        uv_ptr,
                        y_pitch: width as usize,
                        uv_pitch: width as usize,
                        width,
                        height,
                        rotation: 0,
                        is_10bit: false,
                    },
                )
            }
            DetectorFrame::Rgba { data, .. } => {
                let pitch = width as usize * 4;
                if data.len() < pitch * height as usize {
                    return Err("TensorRT RGBA input is too small".into());
                }
                if self.upload_rgba.is_none() {
                    self.upload_rgba = Some(CudaBuffer::new(self.max_pixels * 4)?);
                }
                let ptr = self.upload_rgba.as_ref().unwrap().ptr();
                cuda_memcpy_htod_2d(ptr, pitch, data.as_ptr(), pitch, pitch, height as usize)?;
                cuda::synchronize(std::ptr::null_mut())?;
                self.preprocess_rgba(layout, ptr, pitch)
            }
            DetectorFrame::CudaRgba { ptr, pitch, .. } => {
                self.preprocess_rgba(layout, *ptr, *pitch)
            }
            DetectorFrame::CudaRgbaLetterboxed {
                ptr,
                input_size,
                pitch,
                ..
            } => {
                if *input_size != self.model.size {
                    return Err(format!(
                        "TensorRT letterboxed input size {input_size} does not match model size {}",
                        self.model.size
                    )
                    .into());
                }
                validate_plane(*ptr, *pitch, *input_size as usize * 4, *input_size)?;
                cuda_kernels::normalize_rgba_to_chw(
                    *ptr,
                    *pitch,
                    self.tensor.ptr(),
                    self.model.size,
                    self.model.size,
                )?;
                Ok(())
            }
            _ => unreachable!("checked by detect"),
        }
    }
}

impl Detector for TrtGpuDetector {
    fn name(&self) -> &'static str {
        "tensorrt-native"
    }

    fn prefers_cuda_nv12(&self) -> bool {
        true
    }

    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> std::result::Result<Vec<Detection>, DetectorError> {
        reco_core::profile_scope!("trt_yolo_detect");
        let size = match frame {
            DetectorFrame::Yuv(raw) if matches!(raw.chroma, ChromaPlanes::Nv12 { .. }) => {
                (raw.width, raw.height)
            }
            DetectorFrame::CudaNv12(raw) => (raw.width, raw.height),
            DetectorFrame::Rgba { width, height, .. }
            | DetectorFrame::CudaRgba { width, height, .. } => (*width, *height),
            DetectorFrame::CudaRgbaLetterboxed {
                src_width,
                src_height,
                ..
            } => (*src_width, *src_height),
            _ => return Err(DetectorError::UnsupportedFrameKind),
        };
        let mut run = || -> Result<Vec<Detection>> {
            let layout = self.layout(camera, size)?;
            cuda_ensure_context()?;
            self.preprocess(layout, frame)?;
            self.infer()?;
            Ok(super::postprocess(
                &self.output_host,
                self.model.output_elements / 6,
                camera,
                self.confidence_threshold,
                layout.scale,
                layout.padding.0 as f32,
                layout.padding.1 as f32,
                size.0,
                size.1,
            ))
        };
        run().map_err(|error| DetectorError::InferenceFailed(format!("TensorRT: {error}")))
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}
