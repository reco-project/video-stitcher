//! NCNN CPU inference for Ultralytics YOLO exports.
//!
//! Preprocessing preserves each source frame's aspect ratio using the shared integer letterbox layout. NCNN performs RGB resizing and normalization.

use std::ffi::{CString, c_char, c_void};
use std::path::Path;

use reco_core::detect::detector::{
    ChromaPlanes, Detection, Detector, DetectorError, DetectorFrame, YuvView,
};
use reco_core::geometry::CameraIndex;

use crate::letterbox::Letterbox;

unsafe extern "C" {
    fn ncnn_net_create() -> *mut c_void;
    fn ncnn_net_destroy(net: *mut c_void);
    fn ncnn_net_get_option(net: *mut c_void) -> *mut c_void;
    fn ncnn_net_load_param(net: *mut c_void, path: *const c_char) -> i32;
    fn ncnn_net_load_model(net: *mut c_void, path: *const c_char) -> i32;
    fn ncnn_option_set_num_threads(opt: *mut c_void, num_threads: i32);
    fn ncnn_option_set_use_vulkan_compute(opt: *mut c_void, use_vulkan: i32);

    fn ncnn_extractor_create(net: *mut c_void) -> *mut c_void;
    fn ncnn_extractor_destroy(ex: *mut c_void);
    fn ncnn_extractor_input(ex: *mut c_void, name: *const c_char, mat: *const c_void) -> i32;
    fn ncnn_extractor_extract(ex: *mut c_void, name: *const c_char, mat: *mut *mut c_void) -> i32;

    fn ncnn_mat_create() -> *mut c_void;
    fn ncnn_mat_destroy(mat: *mut c_void);
    fn ncnn_mat_get_dims(mat: *const c_void) -> i32;
    fn ncnn_mat_get_w(mat: *const c_void) -> i32;
    fn ncnn_mat_get_h(mat: *const c_void) -> i32;
    fn ncnn_mat_get_elemsize(mat: *const c_void) -> usize;
    fn ncnn_mat_get_elempack(mat: *const c_void) -> i32;
    fn ncnn_mat_get_data(mat: *const c_void) -> *const f32;
    fn ncnn_mat_from_pixels_resize(
        pixels: *const u8,
        pixel_type: i32,
        w: i32,
        h: i32,
        stride: i32,
        target_width: i32,
        target_height: i32,
        allocator: *mut c_void,
    ) -> *mut c_void;
    fn ncnn_copy_make_border(
        src: *const c_void,
        dst: *mut c_void,
        top: i32,
        bottom: i32,
        left: i32,
        right: i32,
        border_type: i32,
        value: f32,
        opt: *const c_void,
    );
    fn ncnn_mat_substract_mean_normalize(
        mat: *mut c_void,
        mean_vals: *const f32,
        norm_vals: *const f32,
    );
}

// Handles own the C API objects, including an extracted Mat on an error return.
struct Mat(*mut c_void);

impl Mat {
    fn require_data(&self) -> Result<(), DetectorError> {
        if self.0.is_null() || unsafe { ncnn_mat_get_data(self.0) }.is_null() {
            return Err(failed("NCNN returned an empty tensor"));
        }
        Ok(())
    }
}

impl Drop for Mat {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ncnn_mat_destroy(self.0) };
        }
    }
}

struct Extractor(*mut c_void);

impl Drop for Extractor {
    fn drop(&mut self) {
        unsafe { ncnn_extractor_destroy(self.0) };
    }
}

fn failed(message: impl Into<String>) -> DetectorError {
    DetectorError::InferenceFailed(message.into())
}

/// CPU YOLO detector using one NCNN model for all source cameras.
pub struct NcnnYoloDetector {
    net: *mut c_void,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
    rgb: Vec<u8>,
}

// SAFETY: The owned net and scratch buffer are accessed exclusively through &mut self.
unsafe impl Send for NcnnYoloDetector {}

impl NcnnYoloDetector {
    /// Load `model.ncnn.param` and `model.ncnn.bin` from an Ultralytics export.
    ///
    /// The model must accept a square RGB input named `in0` and return unpacked float32 `out0` with shape `[4 + classes, proposals]`: pixel-space center X/Y, width/height, then class probabilities. Each YUV frame is letterboxed to `input_size`; source dimensions may differ between calls.
    /// Frames must have nonzero, even dimensions and tightly packed planes.
    pub fn new(
        model_dir: impl AsRef<Path>,
        input_size: u32,
        confidence_threshold: f32,
        labels: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if input_size == 0
            || input_size
                .checked_mul(input_size)
                .and_then(|n| n.checked_mul(3))
                .is_none_or(|n| n > i32::MAX as u32)
        {
            return Err("NCNN input tensor dimensions exceed C integer indexing".into());
        }
        if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
            return Err("NCNN confidence threshold must be finite and in [0, 1]".into());
        }
        let param_path = model_dir.as_ref().join("model.ncnn.param");
        let bin_path = model_dir.as_ref().join("model.ncnn.bin");
        for path in [&param_path, &bin_path] {
            if !path.is_file() {
                return Err(format!("NCNN model file not found: {}", path.display()).into());
            }
        }
        let param_cstr = CString::new(param_path.to_str().ok_or("invalid NCNN model path")?)?;
        let bin_cstr = CString::new(bin_path.to_str().ok_or("invalid NCNN model path")?)?;
        let detector = Self {
            net: unsafe { ncnn_net_create() },
            input_size,
            confidence_threshold,
            labels,
            rgb: Vec::new(),
        };
        unsafe {
            let opt = ncnn_net_get_option(detector.net);
            ncnn_option_set_num_threads(opt, 4);
            ncnn_option_set_use_vulkan_compute(opt, 0);
            let ret = ncnn_net_load_param(detector.net, param_cstr.as_ptr());
            if ret != 0 {
                return Err(format!("Failed to load NCNN param: error {ret}").into());
            }
            let ret = ncnn_net_load_model(detector.net, bin_cstr.as_ptr());
            if ret != 0 {
                return Err(format!("Failed to load NCNN model: error {ret}").into());
            }
        }
        Ok(detector)
    }

    fn preprocess(&mut self, frame: &YuvView<'_>) -> Result<(Mat, Letterbox), DetectorError> {
        let layout = Letterbox::new((frame.width, frame.height), self.input_size)?;
        if !frame.width.is_multiple_of(2)
            || !frame.height.is_multiple_of(2)
            || frame.width > i32::MAX as u32 / 3
            || frame.height > i32::MAX as u32
        {
            return Err(failed(
                "NCNN requires even YUV420 dimensions with a representable RGB stride",
            ));
        }
        let (width, height) = (frame.width as usize, frame.height as usize);
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| failed("NCNN frame size overflow"))?;
        let rgb_len = pixels
            .checked_mul(3)
            .ok_or_else(|| failed("NCNN RGB size overflow"))?;
        if rgb_len > i32::MAX as usize {
            return Err(failed("NCNN RGB dimensions exceed C integer indexing"));
        }
        let chroma_valid = match &frame.chroma {
            ChromaPlanes::Yuv420p { u, v } => u.len() >= pixels / 4 && v.len() >= pixels / 4,
            ChromaPlanes::Nv12 { uv } => uv.len() >= pixels / 2,
        };
        if frame.y.len() < pixels || !chroma_valid {
            return Err(failed(
                "NCNN frame planes are shorter than their dimensions",
            ));
        }
        self.rgb
            .try_reserve(rgb_len.saturating_sub(self.rgb.len()))
            .map_err(|e| failed(format!("NCNN RGB allocation: {e}")))?;
        self.rgb.resize(rgb_len, 0);
        for y in 0..height {
            for x in 0..width {
                let luma = frame.y[y * width + x] as f32;
                let (u, v) = match &frame.chroma {
                    ChromaPlanes::Yuv420p { u, v } => {
                        let index = (y / 2) * (width / 2) + x / 2;
                        (u[index] as f32 - 128.0, v[index] as f32 - 128.0)
                    }
                    ChromaPlanes::Nv12 { uv } => {
                        let index = (y / 2) * width + (x / 2) * 2;
                        (uv[index] as f32 - 128.0, uv[index + 1] as f32 - 128.0)
                    }
                };
                let rgb = &mut self.rgb[(y * width + x) * 3..][..3];
                // Full-range BT.709 YUV -> RGB.
                rgb[0] = (luma + 1.5748 * v).clamp(0.0, 255.0) as u8;
                rgb[1] = (luma - 0.1873 * u - 0.4681 * v).clamp(0.0, 255.0) as u8;
                rgb[2] = (luma + 1.8556 * u).clamp(0.0, 255.0) as u8;
            }
        }
        let resized = Mat(unsafe {
            ncnn_mat_from_pixels_resize(
                self.rgb.as_ptr(),
                1, // NCNN_MAT_PIXEL_RGB
                frame.width as i32,
                frame.height as i32,
                (frame.width * 3) as i32,
                layout.width as i32,
                layout.height as i32,
                std::ptr::null_mut(),
            )
        });
        resized.require_data()?;
        let padded = Mat(unsafe { ncnn_mat_create() });
        unsafe {
            ncnn_copy_make_border(
                resized.0,
                padded.0,
                layout.pad_y as i32,
                (self.input_size - layout.height - layout.pad_y) as i32,
                layout.pad_x as i32,
                (self.input_size - layout.width - layout.pad_x) as i32,
                0, // BORDER_CONSTANT
                114.0,
                ncnn_net_get_option(self.net),
            );
        }
        padded.require_data()?;
        unsafe {
            ncnn_mat_substract_mean_normalize(
                padded.0,
                std::ptr::null(),
                [1.0 / 255.0; 3].as_ptr(),
            );
        }
        Ok((padded, layout))
    }

    fn postprocess(
        &self,
        output: &Mat,
        camera: CameraIndex,
        frame: &YuvView<'_>,
        layout: &Letterbox,
    ) -> Result<Vec<Detection>, DetectorError> {
        output.require_data()?;
        let (proposals, features) = unsafe {
            if ncnn_mat_get_dims(output.0) != 2
                || ncnn_mat_get_elemsize(output.0) != size_of::<f32>()
                || ncnn_mat_get_elempack(output.0) != 1
            {
                return Err(failed(
                    "NCNN output must be a two-dimensional unpacked float32 tensor",
                ));
            }
            (ncnn_mat_get_w(output.0), ncnn_mat_get_h(output.0))
        };
        if proposals <= 0 || !(5..=4 + u16::MAX as i32 + 1).contains(&features) {
            return Err(failed(
                "NCNN output must have proposals and 1..=65536 classes",
            ));
        }
        let (proposals, features) = (proposals as usize, features as usize);
        let len = proposals
            .checked_mul(features)
            .filter(|len| *len <= isize::MAX as usize / size_of::<f32>())
            .ok_or_else(|| failed("NCNN output size overflow"))?;
        // NCNN owns a contiguous, unpacked 2D Mat for this validated layout.
        let data = unsafe { std::slice::from_raw_parts(ncnn_mat_get_data(output.0), len) };
        let mut candidates = Vec::new();
        for i in 0..proposals {
            let mut class_id = 0;
            let mut confidence = 0.0_f32;
            for class in 0..features - 4 {
                let score = data[(4 + class) * proposals + i];
                if score.is_finite() && score > confidence && score <= 1.0 {
                    confidence = score;
                    class_id = class as u16;
                }
            }
            if confidence < self.confidence_threshold || confidence == 0.0 {
                continue;
            }
            let center_x = (data[i] - layout.pad_x as f32) / layout.scale / frame.width as f32;
            let center_y =
                (data[proposals + i] - layout.pad_y as f32) / layout.scale / frame.height as f32;
            let width = data[2 * proposals + i] / layout.scale / frame.width as f32;
            let height = data[3 * proposals + i] / layout.scale / frame.height as f32;
            if ![center_x, center_y, width, height]
                .iter()
                .all(|v| v.is_finite())
                || !(0.0..=1.0).contains(&center_x)
                || !(0.0..=1.0).contains(&center_y)
                || width <= 0.0
                || height <= 0.0
            {
                continue;
            }
            candidates.push(Detection {
                camera,
                class_id,
                confidence,
                center_x,
                center_y,
                width: width.min(1.0),
                height: height.min(1.0),
            });
        }
        candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let mut keep: Vec<Detection> = Vec::new();
        for candidate in candidates {
            if keep.iter().all(|other| {
                other.class_id != candidate.class_id || super::box_iou(other, &candidate) <= 0.45
            }) {
                keep.push(candidate);
            }
        }
        Ok(keep)
    }
}

impl Detector for NcnnYoloDetector {
    fn name(&self) -> &'static str {
        "ncnn"
    }

    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        reco_core::profile_scope!("ncnn_yolo_detect");
        let DetectorFrame::Yuv(frame) = frame else {
            return Err(DetectorError::UnsupportedFrameKind);
        };
        let (input, layout) = self.preprocess(frame)?;
        let ex = Extractor(unsafe { ncnn_extractor_create(self.net) });
        let ret = unsafe { ncnn_extractor_input(ex.0, c"in0".as_ptr(), input.0) };
        if ret != 0 {
            return Err(failed(format!("NCNN input code {ret}")));
        }
        let mut output = Mat(std::ptr::null_mut());
        let ret = unsafe { ncnn_extractor_extract(ex.0, c"out0".as_ptr(), &mut output.0) };
        if ret != 0 {
            return Err(failed(format!("NCNN extract code {ret}")));
        }
        self.postprocess(&output, camera, frame, &layout)
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

impl Drop for NcnnYoloDetector {
    fn drop(&mut self) {
        unsafe { ncnn_net_destroy(self.net) };
    }
}

#[cfg(test)]
mod tests;
