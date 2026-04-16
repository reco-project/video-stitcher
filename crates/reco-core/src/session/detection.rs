//! Detection pipeline extracted from [`StitchSession`](crate::session::StitchSession).
//!
//! Owns the detector backends (CPU, GPU/CUDA, Metal), detection interval,
//! sink, and cached detections. Separates detection concerns from the
//! rendering/encoding pipeline in `StitchSession`, and is reused by
//! [`AnalyzePipeline`](crate::analyze::AnalyzePipeline) for detection-only
//! consumers.

use crate::detector::{CameraId, Detection, Detector};
use crate::director::MappedDetection;

use super::{DetectionSink, DetectionSinkError};

/// Detection pipeline owning detector backends, interval, sink,
/// and cached detections.
///
/// Used internally by [`StitchSession`](crate::session::StitchSession) and also
/// available as a standalone component for consumers who want detection
/// without the full stitch+encode pipeline (e.g. Python SDKs, analytics).
pub struct DetectionPipeline {
    pub(super) detector: Option<Box<dyn Detector>>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(super) gpu_detector: Option<Box<dyn crate::detector::GpuDetector>>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(super) metal_detector: Option<Box<dyn crate::detector::MetalDetector>>,
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
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            gpu_detector: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal_detector: None,
            detection_interval: 1,
            sink: None,
            last_detections: Vec::new(),
        }
    }

    /// Whether detection should run on the given frame.
    pub(crate) fn should_detect(&self, frame_count: u64) -> bool {
        frame_count.is_multiple_of(self.detection_interval)
    }

    /// Whether a CPU detector is attached.
    #[allow(
        dead_code,
        reason = "helper for future callers and platform-specific paths"
    )]
    pub(crate) fn has_detector(&self) -> bool {
        self.detector.is_some()
    }

    /// Whether a GPU (CUDA) detector is attached.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) fn has_gpu_detector(&self) -> bool {
        self.gpu_detector.is_some()
    }

    /// Whether a Metal detector is attached.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn has_metal_detector(&self) -> bool {
        self.metal_detector.is_some()
    }

    /// Attach a CPU detector for object detection on raw frames.
    pub fn set_detector(&mut self, detector: Box<dyn Detector>) {
        self.detector = Some(detector);
    }

    /// Attach a GPU detector for zero-copy detection on CUDA device pointers.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn set_gpu_detector(&mut self, detector: Box<dyn crate::detector::GpuDetector>) {
        self.gpu_detector = Some(detector);
    }

    /// Attach a Metal detector for zero-copy detection on CVPixelBuffers.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn set_metal_detector(&mut self, detector: Box<dyn crate::detector::MetalDetector>) {
        self.metal_detector = Some(detector);
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

    /// Run the CPU detector on a stereo frame's raw data.
    ///
    /// Returns an empty vec if no CPU detector is attached. GPU-resident
    /// frames (no CPU-accessible pixels) also return an empty vec.
    ///
    /// The caller is responsible for mapping raw detections to panorama
    /// coordinates if needed (see
    /// [`projection::camera_to_panorama`](crate::projection::camera_to_panorama)).
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
                detections.extend(detector.detect(CameraId::Left, &left));
                detections.extend(detector.detect(CameraId::Right, &right));
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
                detections.extend(detector.detect(CameraId::Left, &left));
                detections.extend(detector.detect(CameraId::Right, &right));
            }
            crate::source::StereoFrame::GpuResident { .. } => {
                // GPU-resident frames have no CPU-accessible data for CPU detection.
                // Use gpu_detector or metal_detector instead.
            }
            #[allow(unreachable_patterns)]
            _ => {
                // Future frame variants (e.g. MetalResident) handled by platform-specific detectors
            }
        }
        detections
    }

    /// Run GPU-resident detection via the CUDA [`GpuDetector`](crate::detector::GpuDetector).
    ///
    /// Returns raw detections from both cameras. The caller maps them to
    /// panorama coordinates.
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
        let Some(ref mut gpu_det) = self.gpu_detector else {
            return Vec::new();
        };
        crate::profile_scope!("gpu_detect_total");

        let ls = left_slot as usize;
        let rs = right_slot as usize;
        let mut detections = Vec::new();

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
        detections.extend(gpu_det.detect_gpu(CameraId::Left, &left_frame));
        detections.extend(gpu_det.detect_gpu(CameraId::Right, &right_frame));

        detections
    }

    /// Run Metal-resident detection via [`MetalDetector`](crate::detector::MetalDetector).
    ///
    /// Returns raw detections from both cameras. The caller maps them to
    /// panorama coordinates.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(super) fn run_metal_detection(
        &mut self,
        left_cvpb: crate::metal_interop::CVPixelBufferRef,
        right_cvpb: crate::metal_interop::CVPixelBufferRef,
        width: u32,
        height: u32,
        gpu: &crate::gpu::GpuContext,
    ) -> Vec<Detection> {
        let Some(ref mut metal_det) = self.metal_detector else {
            return Vec::new();
        };
        crate::profile_scope!("metal_detect_total");

        let mut detections = Vec::new();

        detections.extend(metal_det.detect_metal(CameraId::Left, left_cvpb, width, height, gpu));
        detections.extend(metal_det.detect_metal(CameraId::Right, right_cvpb, width, height, gpu));

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

    #[test]
    fn should_detect_respects_interval() {
        let mut pipeline = DetectionPipeline::new();
        // Default interval is 1 - detect every frame.
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
        // 0 is clamped to 1.
        assert!(pipeline.should_detect(0));
        assert!(pipeline.should_detect(1));
    }

    #[test]
    fn has_detector_false_by_default() {
        let pipeline = DetectionPipeline::new();
        assert!(!pipeline.has_detector());
    }
}
