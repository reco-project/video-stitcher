//! Detection entry points for [`StitchSession`].
//!
//! All `detect_and_update_director_*` variants, the shared
//! `fire_sink_and_update_director` tail, coordinate mapping, and the
//! no-detection `update_director` path live here. These methods have the
//! most `#[cfg]` platform branches in the session.

use super::StitchSession;
use crate::detect::detector::Detection;
use crate::detect::director::MappedDetection;
use crate::projection;
use crate::session::types::SessionError;
use crate::source::StereoFrame;

impl StitchSession {
    /// Run detection on a stereo frame, track, map to panorama, and update the director.
    ///
    /// Detection only runs every `detection_interval` frames. On skipped
    /// frames, the last tracked objects are reused so the director still
    /// has context.
    pub fn detect_and_update_director(
        &mut self,
        frame: &StereoFrame,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect {
            let (width, height) = self.core.pipeline().source_info();
            let detections = self.detection.run_detection(frame, width, height);
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Whether detection should run on the current frame.
    /// Returns false if no detector is attached.
    pub fn detection_should_run(&self) -> bool {
        self.detection.has_detector() && self.detection.should_detect(self.frame_count)
    }

    /// Run detection on CPU-resident RGBA frames and update the director.
    ///
    /// Used by the Bayer/V4L2 path. Detection runs only when
    /// `should_detect` returns true (respects detection_interval).
    /// On non-detection frames, the director still advances.
    pub fn detect_and_update_director_rgba(
        &mut self,
        left_rgba: &[u8],
        right_rgba: &[u8],
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect {
            let detections = self
                .detection
                .run_detection_rgba(left_rgba, right_rgba, width, height);
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Run detection on CUDA-resident RGBA frames and update the director.
    ///
    /// Zero-copy path for Bayer cameras: the RGBA data is already on
    /// CUDA via Vulkan shared memory. No CPU readback needed.
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
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect {
            let detections = self.detection.run_detection_cuda_rgba(
                left_ptr,
                left_pitch,
                right_ptr,
                right_pitch,
                width,
                height,
            );
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Detect on pre-letterboxed CUDA RGBA from NvBufSurfTransform and update director.
    ///
    /// The buffers are already at model size with letterbox padding applied.
    /// Skips NPP resize - only normalize + TRT inference.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn detect_and_update_director_preletterboxed(
        &mut self,
        left_ptr: crate::interop::cuda::CUdeviceptr,
        right_ptr: crate::interop::cuda::CUdeviceptr,
        src_width: u32,
        src_height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect {
            let detections = self.detection.run_detection_preletterboxed(
                left_ptr, src_width, src_height, right_ptr, src_width, src_height,
            );
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Update the director without detection.
    ///
    /// Advances the director state (e.g. sweep position) without running
    /// object detection. Used by zero-copy paths and raw Bayer capture
    /// where no CPU-accessible StereoFrame is available.
    pub fn update_director(&mut self, elapsed: std::time::Duration) -> Result<(), SessionError> {
        self.fire_sink_and_update_director(elapsed, false)
    }

    /// Run GPU-resident detection and update the director.
    ///
    /// Detects objects directly from CUDA device pointers (NV12 shared textures),
    /// avoiding any GPU-to-CPU frame readback. Only the small detection
    /// output is transferred to CPU for tracking and director updates.
    ///
    /// Falls back to [`update_director`](Self::update_director) if no
    /// GPU detector is attached.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) fn detect_and_update_director_gpu(
        &mut self,
        left_buf: &crate::interop::zero_copy::GpuBufInfo,
        right_buf: &crate::interop::zero_copy::GpuBufInfo,
        left_slot: u8,
        right_slot: u8,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect && self.detection.has_detector() {
            let detections = self.detection.run_gpu_detection(
                left_buf,
                right_buf,
                left_slot,
                right_slot,
                self.left_rotation,
                self.right_rotation,
            );
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
    }

    /// Run Metal-resident detection and update the director.
    ///
    /// Dispatches to the attached unified detector through
    /// [`DetectorFrame::Metal`](crate::detect::detector::DetectorFrame::Metal).
    /// Falls back to [`update_director`](Self::update_director) if no
    /// detector is attached or the backend doesn't support Metal frames.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn detect_and_update_director_metal(
        &mut self,
        left_cvpb: crate::interop::metal::CVPixelBufferRef,
        right_cvpb: crate::interop::metal::CVPixelBufferRef,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let should_detect = self.detection.should_detect(self.frame_count);

        if should_detect && self.detection.has_detector() {
            let detections = self
                .detection
                .run_detection_metal(left_cvpb, right_cvpb, width, height);
            self.detection.last_detections = self.map_detections(detections);
        }

        self.fire_sink_and_update_director(elapsed, should_detect)
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

        // Pre-tracker detection-filter chain. Each stage mutates
        // `last_detections` in place; with a sink attached, the
        // before/after snapshot is emitted so a user can audit what
        // each filter changed.
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

        // Drive pose resolution via the shared panner dispatch.
        // Runs even when the detections list is empty so trackers get
        // their coast / loss ticks.
        //
        // `fresh_detection` is unused by panner decisions today -
        // trackers manage their own freshness via detection cadence.
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
    ///
    /// Each detection's camera-space center is projected to panorama
    /// yaw/pitch via [`camera_to_panorama`](projection::camera_to_panorama).
    ///
    /// ROI filtering (discarding detections outside the playing field) is
    /// handled at the detector level by `reco-autocam`'s `RoiFilteredDetector`
    /// decorators, so this method is pure coordinate mapping.
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
