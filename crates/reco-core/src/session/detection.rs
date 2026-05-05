//! Detection pipeline used by [`StitchSession`](crate::session::StitchSession).
//!
//! Owns the [`UnifiedDetector`](crate::detector::UnifiedDetector) backend,
//! detection interval, sink, and cached panorama-mapped detections. Exposes
//! CPU and CUDA run paths so the session's decode loop (CPU-resident file
//! sources) and zero-copy loop (GPU-resident shared textures) share one
//! detection state.
//!
//! After the M3 trait-collapse commit there is **one** detector slot
//! here: `Box<dyn UnifiedDetector>`. The backend is responsible for
//! declaring which [`DetectorFrame`](crate::detector::DetectorFrame)
//! residencies it accepts; calling a backend with the wrong variant
//! yields [`DetectorError::UnsupportedFrameKind`](crate::detector::DetectorError::UnsupportedFrameKind)
//! which this module logs at `warn!` and drops (so a flaky frame does
//! not abort the render loop; typed error propagation lives at the
//! StitchCore boundary).
//!
//! Also available as a standalone component for consumers that want
//! detection without the full stitch+encode pipeline (e.g. the
//! analyze CLI path, Python SDKs, analytics workers).

use crate::detector::{CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector};
use crate::director::MappedDetection;

use super::{DetectionSink, DetectionSinkError};

/// Detection pipeline owning a unified-trait detector, interval,
/// sink, and cached detections.
pub struct DetectionPipeline {
    pub(super) detector: Option<Box<dyn UnifiedDetector>>,
    detection_interval: u64,
    sink: Option<Box<dyn DetectionSink>>,
    pub(super) last_detections: Vec<MappedDetection>,
}

impl Default for DetectionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl DetectionPipeline {
    /// Create a new detection pipeline with default settings (no detector, interval 1).
    pub fn new() -> Self {
        Self {
            detector: None,
            detection_interval: 1,
            sink: None,
            last_detections: Vec::new(),
        }
    }

    /// Whether detection should run on the given frame.
    pub(crate) fn should_detect(&self, frame_count: u64) -> bool {
        frame_count.is_multiple_of(self.detection_interval)
    }

    /// Whether a detector is attached.
    pub fn has_detector(&self) -> bool {
        self.detector.is_some()
    }

    /// Attach a detector. Replaces any existing one.
    pub fn set_detector(&mut self, detector: Box<dyn UnifiedDetector>) {
        self.detector = Some(detector);
    }

    /// Set the detection interval (run detection every N frames).
    ///
    /// Clamped to a minimum of 1 (every frame).
    pub fn set_detection_interval(&mut self, interval: u64) {
        self.detection_interval = interval.max(1);
    }

    /// Set the sink that receives tracked detection data each frame.
    ///
    /// Replaces any sink set previously. Sink errors returned from
    /// [`DetectionSink::on_detections`] abort the current session call
    /// with [`SessionError::DetectionSink`](super::SessionError::DetectionSink).
    pub fn set_sink(&mut self, sink: Box<dyn DetectionSink>) {
        self.sink = Some(sink);
    }

    /// The most recent panorama-mapped detections.
    ///
    /// Owned by the pipeline so both [`StitchSession`](super::StitchSession)
    /// and standalone callers (analyze path) can read or replace this
    /// buffer without duplicating state.
    pub fn last_detections(&self) -> &[MappedDetection] {
        &self.last_detections
    }

    /// Replace the cached detections (e.g. after the caller has mapped
    /// raw detector output to panorama coordinates).
    pub fn set_last_detections(&mut self, dets: Vec<MappedDetection>) {
        self.last_detections = dets;
    }

    /// Run detection on a CPU-resident stereo frame (YUV420P / NV12).
    ///
    /// Wraps each camera's planes as `DetectorFrame::Cpu(RawFrame)` and
    /// dispatches through the unified trait, once per camera. GPU-resident
    /// variants return an empty vec (see `run_gpu_detection` for the CUDA
    /// zero-copy path).
    pub fn run_detection(
        &mut self,
        frame: &crate::source::StereoFrame,
        source_width: u32,
        source_height: u32,
    ) -> Vec<Detection> {
        use crate::detector::{ChromaFormat, RawFrame};

        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };

        let run = |det: &mut Box<dyn UnifiedDetector>,
                   camera: CameraId,
                   raw: RawFrame<'_>|
         -> Vec<Detection> {
            match det.detect(camera, &DetectorFrame::Cpu(raw)) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("detector '{}' {camera:?}: {e}", det.name());
                    Vec::new()
                }
            }
        };

        let mut detections = Vec::new();
        match frame {
            crate::source::StereoFrame::Yuv420p(pair) => {
                let left = RawFrame {
                    y: &pair.left.y,
                    chroma: ChromaFormat::Yuv420p {
                        u: &pair.left.u,
                        v: &pair.left.v,
                    },
                    width: source_width,
                    height: source_height,
                };
                let right = RawFrame {
                    y: &pair.right.y,
                    chroma: ChromaFormat::Yuv420p {
                        u: &pair.right.u,
                        v: &pair.right.v,
                    },
                    width: source_width,
                    height: source_height,
                };
                detections.extend(run(detector, CameraId::Left, left));
                detections.extend(run(detector, CameraId::Right, right));
            }
            crate::source::StereoFrame::Nv12(pair) => {
                let left = RawFrame {
                    y: &pair.left.y,
                    chroma: ChromaFormat::Nv12 { uv: &pair.left.uv },
                    width: source_width,
                    height: source_height,
                };
                let right = RawFrame {
                    y: &pair.right.y,
                    chroma: ChromaFormat::Nv12 { uv: &pair.right.uv },
                    width: source_width,
                    height: source_height,
                };
                detections.extend(run(detector, CameraId::Left, left));
                detections.extend(run(detector, CameraId::Right, right));
            }
            // GPU/Metal-resident frames: use `run_gpu_detection` /
            // the Metal equivalent. CPU RawFrame construction is
            // impossible here (no CPU-accessible pixels).
            _ => {}
        }
        detections
    }

    /// Run detection on CPU-resident RGBA stereo frames.
    ///
    /// Used by the Bayer/V4L2 path where the demosaiced RGBA is read
    /// back from the GPU periodically for detection.
    pub fn run_detection_rgba(
        &mut self,
        left_rgba: &[u8],
        right_rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<Detection> {
        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };

        let mut detections = Vec::new();
        for (camera, rgba) in [(CameraId::Left, left_rgba), (CameraId::Right, right_rgba)] {
            let frame = DetectorFrame::Rgba {
                data: rgba,
                width,
                height,
            };
            match detector.detect(camera, &frame) {
                Ok(v) => detections.extend(v),
                Err(DetectorError::UnsupportedFrameKind) => {
                    log::debug!(
                        "detector '{}' does not support RGBA frames",
                        detector.name()
                    );
                }
                Err(e) => {
                    log::warn!("detector '{}' {camera:?}: {e}", detector.name());
                }
            }
        }
        detections
    }

    /// Run detection on CUDA-resident RGBA frames (zero-copy from Bayer demosaic).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn run_detection_cuda_rgba(
        &mut self,
        left_ptr: crate::cuda_interop::CUdeviceptr,
        left_pitch: usize,
        right_ptr: crate::cuda_interop::CUdeviceptr,
        right_pitch: usize,
        width: u32,
        height: u32,
    ) -> Vec<Detection> {
        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };

        let mut detections = Vec::new();
        for (camera, ptr, pitch) in [
            (CameraId::Left, left_ptr, left_pitch),
            (CameraId::Right, right_ptr, right_pitch),
        ] {
            let frame = DetectorFrame::CudaRgba {
                ptr,
                pitch,
                width,
                height,
            };
            match detector.detect(camera, &frame) {
                Ok(v) => detections.extend(v),
                Err(DetectorError::UnsupportedFrameKind) => {
                    log::debug!(
                        "detector '{}' does not support CudaRgba frames",
                        detector.name()
                    );
                }
                Err(e) => {
                    log::warn!("detector '{}' {camera:?}: {e}", detector.name());
                }
            }
        }
        detections
    }

    /// Run detection on pre-letterboxed CUDA RGBA buffers (NvBufSurfTransform output).
    ///
    /// The buffers are already at model input resolution with letterbox padding.
    /// The detector skips NPP resize and only runs normalize + inference.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn run_detection_preletterboxed(
        &mut self,
        left_ptr: crate::cuda_interop::CUdeviceptr,
        left_src_width: u32,
        left_src_height: u32,
        right_ptr: crate::cuda_interop::CUdeviceptr,
        right_src_width: u32,
        right_src_height: u32,
    ) -> Vec<Detection> {
        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };

        let mut detections = Vec::new();
        for (camera, ptr, sw, sh) in [
            (CameraId::Left, left_ptr, left_src_width, left_src_height),
            (CameraId::Right, right_ptr, right_src_width, right_src_height),
        ] {
            let frame = DetectorFrame::CudaRgbaLetterboxed {
                ptr,
                src_width: sw,
                src_height: sh,
            };
            match detector.detect(camera, &frame) {
                Ok(v) => detections.extend(v),
                Err(DetectorError::UnsupportedFrameKind) => {
                    log::debug!(
                        "detector '{}' does not support CudaRgbaLetterboxed",
                        detector.name()
                    );
                }
                Err(e) => {
                    log::warn!("detector '{}' {camera:?}: {e}", detector.name());
                }
            }
        }
        detections
    }

    /// Run detection on a CUDA-resident stereo NV12 frame.
    ///
    /// Builds [`DetectorFrame::Cuda(GpuNv12Frame)`] for each camera
    /// from the shared-texture slot pointers + pitches and dispatches
    /// through the unified trait. Returns raw detections from both
    /// cameras; the caller maps them to panorama coordinates.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(super) fn run_gpu_detection(
        &mut self,
        left_buf: &crate::zero_copy::GpuBufInfo,
        right_buf: &crate::zero_copy::GpuBufInfo,
        left_slot: u8,
        right_slot: u8,
        left_rotation: i32,
        right_rotation: i32,
    ) -> Vec<Detection> {
        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };
        crate::profile_scope!("gpu_detect_total");

        let ls = left_slot as usize;
        let rs = right_slot as usize;
        let is_10bit = left_buf.pixel_format == crate::renderer::GpuPixelFormat::P010;

        let left_frame = crate::detector::GpuNv12Frame {
            y_ptr: left_buf.y_ptr[ls],
            uv_ptr: left_buf.uv_ptr[ls],
            y_pitch: left_buf.y_pitch[ls],
            uv_pitch: left_buf.uv_pitch[ls],
            width: left_buf.width,
            height: left_buf.height,
            rotation: left_rotation,
            is_10bit,
        };
        let right_frame = crate::detector::GpuNv12Frame {
            y_ptr: right_buf.y_ptr[rs],
            uv_ptr: right_buf.uv_ptr[rs],
            y_pitch: right_buf.y_pitch[rs],
            uv_pitch: right_buf.uv_pitch[rs],
            width: right_buf.width,
            height: right_buf.height,
            rotation: right_rotation,
            is_10bit,
        };

        let run = |det: &mut Box<dyn UnifiedDetector>,
                   camera: CameraId,
                   gpu_frame: crate::detector::GpuNv12Frame|
         -> Vec<Detection> {
            match det.detect(camera, &DetectorFrame::Cuda(gpu_frame)) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("detector '{}' {camera:?}: {e}", det.name());
                    Vec::new()
                }
            }
        };

        let mut detections = Vec::new();
        detections.extend(run(detector, CameraId::Left, left_frame));
        detections.extend(run(detector, CameraId::Right, right_frame));
        detections
    }

    /// Fire the detection sink with the current cached detections.
    ///
    /// Returns `Ok(())` when no sink is attached or the sink succeeds.
    /// Sink errors propagate so `run` / `step` can abort. Callers that
    /// compute panorama-mapped detections externally should first call
    /// [`set_last_detections`](Self::set_last_detections).
    pub fn fire_sink(
        &mut self,
        frame_index: u64,
        timestamp_ms: f64,
    ) -> Result<(), DetectionSinkError> {
        if let Some(ref mut sink) = self.sink {
            sink.on_detections(&self.last_detections, frame_index, timestamp_ms)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{Detection, DetectorError};

    #[test]
    fn should_detect_respects_interval() {
        let mut pipeline = DetectionPipeline::new();
        assert!(pipeline.should_detect(0));
        assert!(pipeline.should_detect(1));
        assert!(pipeline.should_detect(99));

        pipeline.set_detection_interval(3);
        assert!(pipeline.should_detect(0));
        assert!(!pipeline.should_detect(1));
        assert!(!pipeline.should_detect(2));
        assert!(pipeline.should_detect(3));
        assert!(pipeline.should_detect(6));
    }

    #[test]
    fn interval_clamped_to_minimum_1() {
        let mut pipeline = DetectionPipeline::new();
        pipeline.set_detection_interval(0);
        assert!(pipeline.should_detect(0));
        assert!(pipeline.should_detect(1));
    }

    #[test]
    fn has_detector_false_by_default() {
        let pipeline = DetectionPipeline::new();
        assert!(!pipeline.has_detector());
    }

    /// Synthetic `UnifiedDetector` returning one canned detection per
    /// CPU call. Exercises the unified-only dispatch path.
    struct RecordingDetector;

    impl UnifiedDetector for RecordingDetector {
        fn name(&self) -> &'static str {
            "recording"
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
    }

    #[test]
    fn run_detection_dispatches_both_cameras_on_yuv() {
        use crate::source::{FramePair, StereoFrame, YuvData};

        let mut pipeline = DetectionPipeline::new();
        pipeline.set_detector(Box::new(RecordingDetector));

        let pair = FramePair {
            left: YuvData {
                y: vec![0u8; 8],
                u: vec![128u8; 2],
                v: vec![128u8; 2],
            },
            right: YuvData {
                y: vec![0u8; 8],
                u: vec![128u8; 2],
                v: vec![128u8; 2],
            },
        };
        let frame = StereoFrame::Yuv420p(pair);

        let detections = pipeline.run_detection(&frame, 4, 2);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].camera, CameraId::Left);
        assert_eq!(detections[1].camera, CameraId::Right);
    }
}
