//! Detection entry points for [`StitchSession`].
//!
//! Each `detect_and_update_director_*` variant delegates to the shared
//! [`detect_and_update_director_with`](StitchSession::detect_and_update_director_with)
//! skeleton, passing a closure that calls the right detection backend.
//! The skeleton handles interval gating, coordinate mapping, and the
//! tracker/panner chain.

use super::StitchSession;
use crate::detect::detector::Detection;
use crate::detect::director::MappedDetection;
use crate::projection;
use crate::session::detection::DetectionPipeline;
use crate::session::types::SessionError;
use crate::source::StereoFrame;

impl StitchSession {
    /// Shared detection skeleton: gate by interval, run the backend,
    /// map to panorama, drive trackers/panners.
    ///
    /// Every `detect_and_update_director_*` variant is a one-liner
    /// wrapper that passes a closure here. Adding a new detection
    /// backend means writing one closure, not copying 15 lines.
    fn detect_and_update_director_with(
        &mut self,
        elapsed: std::time::Duration,
        detect_fn: impl FnOnce(&mut DetectionPipeline) -> Vec<Detection>,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);
        if should_detect {
            let detections = detect_fn(&mut self.detection);
            self.detection.last_detections = self.map_detections(detections);
        }
        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Run detection on a CPU-resident stereo frame (YUV420P / NV12).
    pub fn detect_and_update_director(
        &mut self,
        frame: &StereoFrame,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let (w, h) = self.core.pipeline().source_info();
        self.detect_and_update_director_with(elapsed, |det| det.run_detection(frame, w, h))
    }

    /// Whether detection should run on the current frame.
    pub fn detection_should_run(&self) -> bool {
        self.detection.has_detector() && self.detection.should_detect(self.frame_count)
    }

    /// Run detection on CPU-resident RGBA frames.
    pub fn detect_and_update_director_rgba(
        &mut self,
        left_rgba: &[u8],
        right_rgba: &[u8],
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |det| {
            det.run_detection_rgba(left_rgba, right_rgba, width, height)
        })
    }

    /// Run detection on CUDA-resident RGBA frames (Bayer zero-copy).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn detect_and_update_director_cuda_rgba(
        &mut self,
        left_ptr: crate::interop::cuda::CUdeviceptr,
        left_pitch: usize,
        right_ptr: crate::interop::cuda::CUdeviceptr,
        right_pitch: usize,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |det| {
            det.run_detection_cuda_rgba(left_ptr, left_pitch, right_ptr, right_pitch, width, height)
        })
    }

    /// Detect on pre-letterboxed CUDA RGBA (NvBufSurfTransform output).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn detect_and_update_director_preletterboxed(
        &mut self,
        left_ptr: crate::interop::cuda::CUdeviceptr,
        right_ptr: crate::interop::cuda::CUdeviceptr,
        src_width: u32,
        src_height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |det| {
            det.run_detection_preletterboxed(
                left_ptr, src_width, src_height, right_ptr, src_width, src_height,
            )
        })
    }

    /// Run detection via CUDA-imported D3D11VA staging textures.
    #[cfg(target_os = "windows")]
    pub(crate) fn detect_and_update_director_d3d11(
        &mut self,
        left_y: u64,
        left_uv: u64,
        right_y: u64,
        right_uv: u64,
        pitch: usize,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let lr = self.left_rotation;
        let rr = self.right_rotation;
        self.detect_and_update_director_with(elapsed, |det| {
            det.run_gpu_detection_raw(
                left_y, left_uv, right_y, right_uv, pitch, width, height, lr, rr,
            )
        })
    }

    /// Update the director without detection.
    ///
    /// Advances the panner/tracker state without running object
    /// detection. Used when the frame residency has no detection
    /// backend (e.g. D3D11VA without CUDA).
    pub fn update_director(&mut self, elapsed: std::time::Duration) -> Result<(), SessionError> {
        self.fire_sink_and_update_director(elapsed, false)
    }

    /// Run GPU-resident detection from CUDA NV12 shared textures.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) fn detect_and_update_director_gpu(
        &mut self,
        left_buf: &crate::interop::zero_copy::GpuBufInfo,
        right_buf: &crate::interop::zero_copy::GpuBufInfo,
        left_slot: u8,
        right_slot: u8,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let lr = self.left_rotation;
        let rr = self.right_rotation;
        self.detect_and_update_director_with(elapsed, |det| {
            det.run_gpu_detection(left_buf, right_buf, left_slot, right_slot, lr, rr)
        })
    }

    /// Run Metal-resident detection from CVPixelBuffers.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn detect_and_update_director_metal(
        &mut self,
        left_cvpb: crate::interop::metal::CVPixelBufferRef,
        right_cvpb: crate::interop::metal::CVPixelBufferRef,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |det| {
            det.run_detection_metal(left_cvpb, right_cvpb, width, height)
        })
    }

    /// Drive the tracker/panner chain after detection.
    ///
    /// Shared tail for all detection paths (CPU, GPU, Metal, no-detection).
    /// Emits DetectionsRaw events, runs detection filters, drives every
    /// registered tracker to build a
    /// [`WorldState`](crate::detect::tracker::WorldState), then lets the
    /// panner decide the next pose. Viewport constraining is handled
    /// separately by [`director_position`](Self::director_position).
    pub(crate) fn fire_sink_and_update_director(
        &mut self,
        elapsed: std::time::Duration,
        fresh_detection: bool,
    ) -> Result<(), SessionError> {
        let timestamp_ms = elapsed.as_secs_f64() * 1000.0;

        // Trace: DetectionsRaw. Only clones when an event sink is attached.
        if let Some(sink) = self.event_sink.as_deref_mut() {
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::DetectionsRaw {
                    frame_index: self.frame_count,
                    detections: self.detection.last_detections.clone(),
                },
            );
        }

        if !self.detection_filters.is_empty() {
            let calibration = self.core.pipeline().calibration();
            let filter_ctx = crate::detect::filter::FilterContext {
                frame_index: self.frame_count,
                timestamp_ms,
                calibration,
            };
            let trace_enabled = self.event_sink.is_some();
            for filter in self.detection_filters.iter_mut() {
                let before = if trace_enabled {
                    Some(self.detection.last_detections.clone())
                } else {
                    None
                };
                filter.filter(&mut self.detection.last_detections, &filter_ctx);
                if let (Some(before), Some(sink)) = (before, self.event_sink.as_deref_mut()) {
                    sink.emit(
                        crate::detect::pipeline_event::PipelineEvent::DetectionFilter {
                            frame_index: self.frame_count,
                            filter_name: filter.name(),
                            before,
                            after: self.detection.last_detections.clone(),
                        },
                    );
                }
            }
        }

        let _ = fresh_detection;
        let calibration = self.core.pipeline().calibration();
        let dispatch_result = crate::detect::panner::dispatch(
            self.panner.as_mut(),
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            &mut self.previous_panner_pose,
            self.event_sink.as_deref_mut(),
            crate::detect::panner::DispatchContext {
                detections: &self.detection.last_detections,
                calibration,
                frame_index: self.frame_count,
                timestamp_ms,
                caller: "StitchSession",
            },
        );

        self.telemetry.record_detections(
            self.detection.last_detections.len() as u32,
            dispatch_result.as_ref().map_or(0, |r| r.active_tracks),
            dispatch_result.as_ref().is_some_and(|r| r.ball_present),
        );

        Ok(())
    }

    /// Map raw detections to panorama coordinates.
    pub(crate) fn map_detections(&self, detections: Vec<Detection>) -> Vec<MappedDetection> {
        let calibration = self.core.pipeline().calibration();
        let scene = &self.core.pipeline().scene;

        detections
            .iter()
            .map(|d| {
                let position = projection::camera_to_panorama(
                    d.camera,
                    d.center_x,
                    d.center_y,
                    calibration,
                    scene,
                );
                MappedDetection {
                    camera: d.camera,
                    class_id: d.class_id,
                    confidence: d.confidence,
                    camera_center: (d.center_x, d.center_y),
                    camera_size: (d.width, d.height),
                    position,
                }
            })
            .collect()
    }
}
