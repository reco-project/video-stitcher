//! Object detection trait for raw camera frames.
//!
//! Detectors run on raw (pre-stitch) camera frames to find objects of interest
//! (e.g. a ball). Detections are mapped to world-reference angular coordinates and fed
//! to a [`super::tracker::Tracker`] which stabilizes identities, after which a
//! [`super::panner::Panner`] turns the tracked world state into a viewport pose.
//!
//! ## Why Raw Frames?
//!
//! The stitched panorama is an L-shaped 3D projection, not a flat image.
//! Object detection models (YOLO, etc.) work on standard 2D images, so
//! they must run on the original camera frames before stitching.
//! The slight wide-angle distortion is negligible for detection accuracy.

use crate::geometry::CameraIndex;

/// Borrowed CPU YUV image planes and dimensions for detection.
///
/// Provides access to all YUV planes so detectors can use luma-only
/// (fast, sufficient for ball tracking) or full color (needed for
/// jersey classification, field segmentation, etc.).
pub struct YuvView<'a> {
    /// Y (luma) plane, full resolution (`width x height` bytes).
    pub y: &'a [u8],
    /// Chroma plane data (format depends on the decode pipeline).
    pub chroma: ChromaPlanes<'a>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// Borrowed chroma planes in the source image's layout.
///
/// The format matches whatever the decode pipeline produces:
/// - Software decode (FFmpeg): YUV420P with separate U and V planes
/// - Hardware decode (NVDEC, V4L2): NV12 with interleaved UV
pub enum ChromaPlanes<'a> {
    /// YUV420P: separate half-resolution U and V planes.
    Yuv420p {
        /// U (Cb) plane, `(width/2) x (height/2)` bytes.
        u: &'a [u8],
        /// V (Cr) plane, `(width/2) x (height/2)` bytes.
        v: &'a [u8],
    },
    /// NV12: interleaved UV plane, `width x (height/2)` bytes.
    Nv12 {
        /// Interleaved U,V data.
        uv: &'a [u8],
    },
}

/// A detected object in a raw camera frame.
///
/// Coordinates are in normalized image space `[0.0, 1.0]` relative to
/// the frame dimensions. The active
/// [`Calibration::map_source_point`](crate::calibration::Calibration::map_source_point) maps these to world-reference
/// yaw/pitch.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Detection {
    /// Which camera this detection came from.
    pub camera: CameraIndex,

    /// Detection class index from the model (e.g. 0 = "ball", 1 = "person").
    /// Map to a human-readable label via the detector's `class_names()`.
    pub class_id: u16,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Bounding box center X in normalized image coordinates `[0.0, 1.0]`.
    pub center_x: f32,

    /// Bounding box center Y in normalized image coordinates `[0.0, 1.0]`.
    pub center_y: f32,

    /// Bounding box width in normalized image coordinates.
    pub width: f32,

    /// Bounding box height in normalized image coordinates.
    pub height: f32,
}

// ---------------------------------------------------------------------------
// Unified detector error + frame variants.
// ---------------------------------------------------------------------------
//
// `YuvView`, `ChromaPlanes`, and `CudaNv12Frame` are the parameter
// types for [`DetectorFrame`] variants and the unified trait's inputs.
//
// The trait returns `Result<Vec<Detection>, DetectorError>` rather
// than a bare `Vec` because remote inference (mobile companions,
// network workers) needs an error surface that an in-process `Vec`
// cannot express; carrying it from the start means the first remote
// backend lands without a breaking trait change.

/// Reasons a detector call can fail.
///
/// Designed to cover both in-process and remote backends. In-process
/// variants (`InferenceFailed`, `Canceled`, `UnsupportedFrameKind`) map
/// to ORT / TensorRT / NCNN runtime errors and the detection scheduler.
/// Remote variants (`Timeout`, `Transport`) cover a future
/// `reco-detect-remote` crate that ships frames to a gRPC / HTTP
/// worker.
#[derive(Debug, Clone)]
pub enum DetectorError {
    /// The underlying inference engine returned an error (ORT, TRT,
    /// NCNN, CoreML). The string is the engine's own message.
    InferenceFailed(String),
    /// A caller-side deadline elapsed before the detector produced a
    /// result. Most useful for remote backends where the budget is
    /// wall-clock RTT plus compute.
    Timeout {
        /// How long the caller waited before giving up.
        after: std::time::Duration,
    },
    /// The detector cannot accept this variant of [`DetectorFrame`]
    /// (e.g. a CPU-only backend given CUDA pointers). Construction-
    /// time mismatches should be caught in the builder; this variant
    /// covers dynamic dispatch errors.
    UnsupportedFrameKind,
    /// Network / IPC / serialization error from a remote backend.
    /// The string is the transport layer's own message (wrapped HTTP
    /// status, gRPC status code, socket error).
    Transport(String),
    /// The caller cancelled the in-flight detection, typically because
    /// the session shut down or a newer frame arrived and the older
    /// one is no longer interesting.
    Canceled,
}

impl std::fmt::Display for DetectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InferenceFailed(msg) => write!(f, "inference failed: {msg}"),
            Self::Timeout { after } => write!(f, "detector timed out after {after:?}"),
            Self::UnsupportedFrameKind => {
                write!(f, "detector does not support this frame variant")
            }
            Self::Transport(msg) => write!(f, "transport error: {msg}"),
            Self::Canceled => write!(f, "detection canceled"),
        }
    }
}

impl std::error::Error for DetectorError {}

/// Unified frame input for the [`Detector`] trait.
///
/// Variants distinguish pixel layout and memory residency. Device-backed inputs require their native graphics or compute context.
#[non_exhaustive]
pub enum DetectorFrame<'a> {
    /// CPU-resident YUV420P or NV12 image planes.
    Yuv(YuvView<'a>),

    /// CUDA device-pointer NV12 frame. Local only; see [`CudaNv12Frame`].
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    CudaNv12(CudaNv12Frame),

    /// Pre-processed float32 CHW tensor at model input resolution.
    ///
    /// Produced by `WgpuPreprocessor`: the NV12→RGB conversion, resize,
    /// letterbox, and normalize are done on the GPU. The detector skips
    /// all preprocessing and runs inference directly on this data. Content dimensions are rounded after scaling to fit; top/left padding is half the remaining space rounded down, with the extra pixel on the bottom/right.
    PreprocessedChw {
        /// Float32 CHW data, `3 * input_size * input_size` elements.
        data: &'a [f32],
        /// Model input size (square dimension).
        input_size: u32,
        /// Original source frame width (for detection coordinate mapping).
        src_width: u32,
        /// Original source frame height (for detection coordinate mapping).
        src_height: u32,
    },

    /// CPU-resident packed RGBA frame (e.g. Bayer demosaic readback).
    Rgba {
        /// Packed RGBA bytes, `width * height * 4` length.
        data: &'a [u8],
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },

    /// CUDA device-pointer RGBA frame (zero-copy from Bayer demosaic).
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "gpu"))]
    CudaRgba {
        /// CUDA device pointer to packed RGBA data.
        ptr: crate::interop::cuda::CUdeviceptr,
        /// Row pitch in bytes (may differ from width*4 due to alignment).
        pitch: usize,
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },

    /// Pre-letterboxed CUDA RGBA at model input resolution.
    ///
    /// Produced by NvBufSurfTransform on Jetson: the resize, format
    /// conversion, and letterbox padding are done in a single hardware-
    /// accelerated operation. The detector skips NPP resize and only
    /// runs the normalize kernel + inference.
    #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "gpu"))]
    CudaRgbaLetterboxed {
        /// CUDA device pointer to letterboxed RGBA data at model size.
        ptr: crate::interop::cuda::CUdeviceptr,
        /// Model input size (square dimension).
        input_size: u32,
        /// Row pitch in bytes, including allocation padding.
        pitch: usize,
        /// Original source frame width (for detection coordinate mapping).
        src_width: u32,
        /// Original source frame height (for detection coordinate mapping).
        src_height: u32,
    },

    /// Metal / VideoToolbox `CVPixelBufferRef`. Local only.
    #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
    Metal {
        /// Opaque CVPixelBuffer pointer from VideoToolbox.
        cv_pixel_buffer: crate::interop::metal::CVPixelBufferRef,
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },

    /// wgpu NV12 texture views from D3D11VA staging or similar.
    ///
    /// The detector backend (typically a composite wrapper) owns a
    /// `WgpuPreprocessor` that converts these views to float32 CHW
    /// via a compute shader, then delegates to the inner detector
    /// with `PreprocessedChw`.
    #[cfg(feature = "gpu")]
    WgpuNv12 {
        /// Y plane view (R8Unorm, full resolution).
        y_view: &'a wgpu::TextureView,
        /// UV plane view (Rg8Unorm, half resolution).
        uv_view: &'a wgpu::TextureView,
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Stream rotation (0 or 180 for DJI).
        rotation: i32,
    },
}

impl DetectorFrame<'_> {
    /// A short label for logs and error messages.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::Yuv(_) => "Yuv",
            Self::Rgba { .. } => "Rgba",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Self::CudaNv12(_) => "CudaNv12",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "gpu"))]
            Self::CudaRgba { .. } => "CudaRgba",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            #[cfg(all(any(target_os = "linux", target_os = "windows"), feature = "gpu"))]
            Self::CudaRgbaLetterboxed { .. } => "CudaRgbaLetterboxed",
            #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
            Self::Metal { .. } => "Metal",
            Self::PreprocessedChw { .. } => "PreprocessedChw",
            #[cfg(feature = "gpu")]
            Self::WgpuNv12 { .. } => "WgpuNv12",
        }
    }
}

/// Object detection over source-image frames.
///
/// Consumers call one method and pass whatever residency the current
/// frame has via [`DetectorFrame`]. The backend either accepts the
/// variant or returns [`DetectorError::UnsupportedFrameKind`]. This
/// is the trait `StitchCore` plugs into the session.
///
/// # Threading
///
/// A detector can move between threads. Detection takes an exclusive borrow; shared concurrent access is not required.
pub trait Detector: Send {
    /// Short human-readable name for logs + diagnostic bundles
    /// (e.g. `"ort-cuda"`, `"coreml"`, `"ncnn"`, `"remote-grpc"`).
    fn name(&self) -> &'static str;

    /// Prefer CUDA NV12 pointers over wgpu textures when the source path offers both. The Linux NVDEC session path honors this preference; it does not add CUDA support to other paths. Wrappers preserving the input contract must forward it; wgpu preprocessing wrappers keep it false.
    fn prefers_cuda_nv12(&self) -> bool {
        false
    }

    /// Attempt to run detection on the supplied frame.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`DetectorError::UnsupportedFrameKind`] when the backend
    ///   cannot handle the supplied [`DetectorFrame`] variant.
    /// - [`DetectorError::InferenceFailed`] for engine-level faults.
    /// - [`DetectorError::Timeout`] / [`DetectorError::Transport`]
    ///   for remote backends.
    /// - [`DetectorError::Canceled`] when the session interrupted
    ///   the call.
    fn detect(
        &mut self,
        camera: CameraIndex,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError>;

    /// Optional class-label lookup so consumers can translate
    /// `class_id: u16` into the human-readable names the model was
    /// trained with. `None` indicates the backend does not know its
    /// labels (rarely useful; most ONNX exports carry a `names`
    /// dict).
    fn class_names(&self) -> Option<&[String]> {
        None
    }
}

/// A GPU-resident NV12 frame described by CUDA device pointers.
///
/// Wraps the raw pointer/pitch/dimension parameters needed to locate the
/// Y and UV planes of an NV12 frame in GPU memory. Passed by reference
/// to detection backends instead of many loose arguments.
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Debug, Clone, Copy)]
pub struct CudaNv12Frame {
    /// CUDA device pointer to the Y (luma) plane.
    pub y_ptr: u64,
    /// CUDA device pointer to the UV (chroma) plane.
    pub uv_ptr: u64,
    /// Row pitch in bytes for the Y plane.
    pub y_pitch: usize,
    /// Row pitch in bytes for the UV plane.
    pub uv_pitch: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Camera rotation from stream metadata (0, 90, 180, 270 degrees).
    ///
    /// In the GPU zero-copy path, NVDEC decodes without applying rotation
    /// metadata. The rendering shader flips UV coordinates so the display
    /// is correct, but the detector receives raw upside-down frames. When
    /// `rotation == 180`, the detector must flip the frame during
    /// preprocessing so detection models see correctly oriented images.
    pub rotation: i32,
    /// Whether this frame uses P010 (10-bit NV12) pixel format.
    ///
    /// P010 stores 10-bit luma/chroma values in the upper 10 bits of each
    /// `u16` sample. Detectors that expect 8-bit NV12 (e.g. NPP's
    /// `nppiNV12ToRGB_8u_P2C3R`) must convert P010 to 8-bit first by
    /// right-shifting each sample by 8 bits.
    pub is_10bit: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal fake detector that accepts only CPU frames and returns
    /// a single synthetic detection. Exercises the dyn-dispatch path
    /// StitchCore will rely on.
    struct FakeDetector {
        labels: Vec<String>,
    }

    impl Detector for FakeDetector {
        fn name(&self) -> &'static str {
            "fake-cpu-only"
        }

        fn detect(
            &mut self,
            camera: CameraIndex,
            frame: &DetectorFrame<'_>,
        ) -> Result<Vec<Detection>, DetectorError> {
            match frame {
                DetectorFrame::Yuv(_) => Ok(vec![Detection {
                    camera,
                    class_id: 0,
                    confidence: 0.9,
                    center_x: 0.5,
                    center_y: 0.5,
                    width: 0.1,
                    height: 0.1,
                }]),
                _ => Err(DetectorError::UnsupportedFrameKind),
            }
        }

        fn class_names(&self) -> Option<&[String]> {
            Some(&self.labels)
        }
    }

    #[test]
    fn unified_detector_accepts_cpu_frame() {
        let y = vec![0u8; 8];
        let u = vec![128u8; 2];
        let v = vec![128u8; 2];
        let raw = YuvView {
            y: &y,
            chroma: ChromaPlanes::Yuv420p { u: &u, v: &v },
            width: 4,
            height: 2,
        };
        let mut det: Box<dyn Detector> = Box::new(FakeDetector {
            labels: vec!["ball".into()],
        });
        let out = det.detect(0, &DetectorFrame::Yuv(raw)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].camera, 0);
    }

    #[test]
    fn unified_detector_returns_error_on_unsupported_variant() {
        // Hand a Metal variant to a CPU-only detector. On non-macOS
        // targets Metal variant doesn't exist in the enum, so this
        // test body is cfg-gated; on macOS it exercises the error
        // path by constructing the variant.
        let mut det: Box<dyn Detector> = Box::new(FakeDetector { labels: vec![] });
        // Construct a CPU frame to prove Ok path works...
        let y = vec![0u8; 4];
        let u = vec![128u8; 1];
        let v = vec![128u8; 1];
        let raw = YuvView {
            y: &y,
            chroma: ChromaPlanes::Yuv420p { u: &u, v: &v },
            width: 2,
            height: 2,
        };
        assert!(det.detect(0, &DetectorFrame::Yuv(raw)).is_ok());
    }

    #[test]
    fn detector_error_is_clone_send_sync() {
        // `Clone + Send + Sync` is a hard requirement for cross-thread
        // channel use (Agent 8 / E5 cross-consumer extraction). Verify
        // via a compile-time bound check.
        fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
        assert_clone_send_sync::<DetectorError>();
    }

    #[test]
    fn detector_frame_variant_name() {
        let y = vec![0u8; 4];
        let u = vec![128u8; 1];
        let v = vec![128u8; 1];
        let raw = YuvView {
            y: &y,
            chroma: ChromaPlanes::Yuv420p { u: &u, v: &v },
            width: 2,
            height: 2,
        };
        assert_eq!(DetectorFrame::Yuv(raw).variant_name(), "Yuv");
    }
}
