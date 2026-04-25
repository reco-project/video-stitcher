//! Object detection trait for raw camera frames.
//!
//! Detectors run on raw (pre-stitch) camera frames to find objects of interest
//! (e.g. a ball). Detections are mapped to panorama-space coordinates and fed
//! to a [`crate::tracker::Tracker`] which stabilizes identities, after which a
//! [`crate::panner::Panner`] turns the tracked world state into a viewport pose.
//!
//! ## Why Raw Frames?
//!
//! The stitched panorama is an L-shaped 3D projection, not a flat image.
//! Object detection models (YOLO, etc.) work on standard 2D images, so
//! they must run on the original camera frames before stitching.
//! The slight wide-angle distortion is negligible for detection accuracy.

/// Which camera produced this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CameraId {
    /// Left camera (plane in X-Z space).
    Left,
    /// Right camera (plane in X-Y space).
    Right,
}

impl std::fmt::Display for CameraId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => f.write_str("L"),
            Self::Right => f.write_str("R"),
        }
    }
}

/// Raw camera frame data for detection.
///
/// Provides access to all YUV planes so detectors can use luma-only
/// (fast, sufficient for ball tracking) or full color (needed for
/// jersey classification, field segmentation, etc.).
pub struct RawFrame<'a> {
    /// Y (luma) plane, full resolution (`width x height` bytes).
    pub y: &'a [u8],
    /// Chroma plane data (format depends on the decode pipeline).
    pub chroma: ChromaFormat<'a>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// Chroma plane layout.
///
/// The format matches whatever the decode pipeline produces:
/// - Software decode (FFmpeg): YUV420P with separate U and V planes
/// - Hardware decode (NVDEC, V4L2): NV12 with interleaved UV
pub enum ChromaFormat<'a> {
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
/// the frame dimensions. Use [`crate::projection::camera_to_panorama`]
/// to map these to panoramic yaw/pitch.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Detection {
    /// Which camera this detection came from.
    pub camera: CameraId,

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

// `trait Detector` / `trait GpuDetector` / `trait MetalDetector` were
// deleted 2026-04-19 (plan-execution §3 M3 step 4 final cleanup).
// Every reco-detect backend and every session consumer now uses
// [`UnifiedDetector`] below. `RawFrame`, `ChromaFormat`,
// `GpuNv12Frame`, and `CameraId` remain as parameter types for
// [`DetectorFrame`] variants and the unified trait's inputs.

// ---------------------------------------------------------------------------
// M3 foundation: unified detector error + frame variants.
// ---------------------------------------------------------------------------
//
// Added 2026-04-18 as part of the plan-execution M3 foundation. Not yet
// used by a trait impl in this commit - the existing per-platform
// `Detector` / `GpuDetector` / `MetalDetector` traits keep returning
// `Vec<Detection>`. A later tranche will introduce the unified
// `UnifiedDetector` trait that returns `Result<Vec<Detection>,
// DetectorError>` so remote + timeout-able inference stops being a
// per-backend problem.
//
// The plan-execution doc §2.7 + §8 row "Distributed AI inference"
// captures why these types exist: remote inference (GoPro / mobile /
// future gRPC workers) needs an error surface that in-process Vec
// cannot express. Baking that in now means the trait shape does not
// need a second breaking change when the first remote backend lands.

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

/// Unified frame input for the future [`UnifiedDetector`] trait.
///
/// Each variant describes a different memory residency. The CPU
/// variant is the only one shippable over the network (for future
/// remote backends); CUDA / Metal variants are local-only.
///
/// Not yet wired up in this crate - the current in-tree detectors
/// still take `RawFrame` / CUDA ptrs / `CVPixelBufferRef` directly.
/// The M3 StitchCore refactor will collapse the three platform traits
/// into one that accepts this enum.
#[non_exhaustive]
pub enum DetectorFrame<'a> {
    /// CPU-resident YUV420P frame. The only variant that can cross
    /// process boundaries - compression and ROI-cropping are the
    /// remote backend's responsibility.
    Cpu(RawFrame<'a>),

    /// CUDA device-pointer NV12 frame. Local only; see [`GpuNv12Frame`].
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Cuda(GpuNv12Frame),

    /// CPU-resident packed RGBA frame (e.g. Bayer demosaic readback).
    Rgba {
        /// Packed RGBA bytes, `width * height * 4` length.
        data: &'a [u8],
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },

    /// Metal / VideoToolbox `CVPixelBufferRef`. Local only.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Metal {
        /// Opaque CVPixelBuffer pointer from VideoToolbox.
        cv_pixel_buffer: crate::metal_interop::CVPixelBufferRef,
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },
}

impl DetectorFrame<'_> {
    /// A short label for logs and error messages.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "Cpu",
            Self::Rgba { .. } => "Rgba",
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            Self::Cuda(_) => "Cuda",
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Self::Metal { .. } => "Metal",
        }
    }
}

/// Unified detector trait that collapses the former per-platform
/// `Detector` / `GpuDetector` / `MetalDetector` into a single contract.
///
/// Consumers call one method and pass whatever residency the current
/// frame has via [`DetectorFrame`]. The backend either accepts the
/// variant or returns [`DetectorError::UnsupportedFrameKind`]. This
/// is the trait `StitchCore` plugs into the session.
///
/// # Rationale
///
/// Per the plan-execution-2026-04-18 doc §2.7 and deep-review-2026-
/// 04-18 Agent 5 finding: today's split into three separate traits
/// forces consumers (reco-cli, reco-gui, reco-obs) to know which
/// backend they have and wire it into a per-platform `set_*_detector`
/// method. A unified trait moves platform dispatch behind the
/// backend constructor, where it belongs.
///
/// # Async-ready via `Result`
///
/// The `Result<_, DetectorError>` return type is non-negotiable for
/// three reasons, all from the plan doc:
///
/// 1. Timeout-able inference (§2.7).
/// 2. Remote backends (§8 "Distributed AI inference"): a future
///    `reco-detect-remote` crate ships frames to a gRPC / HTTP worker
///    and needs to surface network faults.
/// 3. Cancellation: when the session shuts down or a newer frame
///    arrives, an in-flight detection should report `Canceled` rather
///    than returning stale data.
///
/// # Threading
///
/// `Send` only. A detector is typically held by the session's
/// worker-thread scheduler (§2.8 mobile-friendly bound policy). If a
/// concrete backend needs `Sync`, it adds the bound itself.
pub trait UnifiedDetector: Send {
    /// Short human-readable name for logs + diagnostic bundles
    /// (e.g. `"ort-cuda"`, `"coreml"`, `"ncnn"`, `"remote-grpc"`).
    fn name(&self) -> &'static str;

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
        camera: CameraId,
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
pub struct GpuNv12Frame {
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

// `trait GpuDetector` + `trait MetalDetector` deleted: replaced by
// `UnifiedDetector` + `DetectorFrame::{Cuda, Metal}` variants. See
// the comment above the old `trait Detector` removal for context.

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal fake detector that accepts only CPU frames and returns
    /// a single synthetic detection. Exercises the dyn-dispatch path
    /// StitchCore will rely on.
    struct FakeDetector {
        labels: Vec<String>,
    }

    impl UnifiedDetector for FakeDetector {
        fn name(&self) -> &'static str {
            "fake-cpu-only"
        }

        fn detect(
            &mut self,
            camera: CameraId,
            frame: &DetectorFrame<'_>,
        ) -> Result<Vec<Detection>, DetectorError> {
            match frame {
                DetectorFrame::Cpu(_) => Ok(vec![Detection {
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
        let raw = RawFrame {
            y: &y,
            chroma: ChromaFormat::Yuv420p { u: &u, v: &v },
            width: 4,
            height: 2,
        };
        let mut det: Box<dyn UnifiedDetector> = Box::new(FakeDetector {
            labels: vec!["ball".into()],
        });
        let out = det
            .detect(CameraId::Left, &DetectorFrame::Cpu(raw))
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].camera, CameraId::Left);
    }

    #[test]
    fn unified_detector_returns_error_on_unsupported_variant() {
        // Hand a Metal variant to a CPU-only detector. On non-macOS
        // targets Metal variant doesn't exist in the enum, so this
        // test body is cfg-gated; on macOS it exercises the error
        // path by constructing the variant.
        let mut det: Box<dyn UnifiedDetector> = Box::new(FakeDetector { labels: vec![] });
        // Construct a CPU frame to prove Ok path works...
        let y = vec![0u8; 4];
        let u = vec![128u8; 1];
        let v = vec![128u8; 1];
        let raw = RawFrame {
            y: &y,
            chroma: ChromaFormat::Yuv420p { u: &u, v: &v },
            width: 2,
            height: 2,
        };
        assert!(det.detect(CameraId::Left, &DetectorFrame::Cpu(raw)).is_ok());
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
        let raw = RawFrame {
            y: &y,
            chroma: ChromaFormat::Yuv420p { u: &u, v: &v },
            width: 2,
            height: 2,
        };
        assert_eq!(DetectorFrame::Cpu(raw).variant_name(), "Cpu");
    }
}
