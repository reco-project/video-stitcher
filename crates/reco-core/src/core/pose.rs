//! Detection scheduling, source-to-world mapping and world-pose requests for [`StitchCore`](super::StitchCore).

use crate::detect::detector::{ChromaPlanes, Detection, DetectorError, DetectorFrame, YuvView};
use crate::detect::mapped_detection::MappedDetection;
use crate::detect::tracker::WorldState;
use crate::geometry::CameraIndex;
use crate::geometry::WorldPose;

/// Tracking statistics from one [`super::StitchCore::dispatch_pose`] tick.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) reads these
pub(crate) struct DispatchStats {
    /// Mapped detections the dispatch consumed.
    pub detections: u32,
    /// Player tracks active this frame.
    pub active_tracks: u32,
    /// Whether a non-lost ball track exists.
    pub ball_present: bool,
}

impl super::StitchCore {
    pub(super) fn anchor_session_start(&mut self) {
        if self.session_start.is_none() {
            self.session_start = Some(std::time::Instant::now());
        }
    }

    pub(super) fn request_current_pose(&mut self) -> WorldPose {
        // Pose resolution applies rig orientation and coverage after the panner request.
        let timestamp_ms = self
            .session_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        self.dispatch_pose(self.frame_count, timestamp_ms, "StitchCore");
        self.requested_world_pose
    }

    /// Whether the detection schedule fires at `index` - true when a
    /// detector is attached and the index lands on the interval.
    /// Index 0 always fires (interval defaults to 1), which covers the
    /// "run on the very first frame" case.
    ///
    /// The index is explicit because the pull session gates its
    /// immediate loop on rendered-frame count but its lookahead
    /// produce phase on produce count; the engine's own submit paths
    /// pass [`Self::frame_count`].
    pub fn detection_due(&self, index: u64) -> bool {
        self.detector.is_some()
            && self.detection_interval > 0
            && index.is_multiple_of(self.detection_interval)
    }

    /// One frame of the tracker -> panner chain at an explicit index
    /// and timestamp: emits `MappedDetections`, runs the shared
    /// [`panner::dispatch`](crate::detect::panner) (which emits
    /// `WorldState`/`PanDecision` through the engine's sink), advances
    /// `requested_world_pose`, and reports the dispatch stats.
    pub(crate) fn dispatch_pose(
        &mut self,
        index: u64,
        timestamp_ms: f64,
        caller: &'static str,
    ) -> DispatchStats {
        if let Some(sink) = self.event_sink.as_deref_mut() {
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::MappedDetections {
                    frame_index: index,
                    detections: self.last_detections.clone(),
                },
            );
        }
        let result = crate::detect::panner::dispatch(
            self.panner.as_mut(),
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            &mut self.requested_world_pose,
            self.event_sink.as_deref_mut(),
            crate::detect::panner::DispatchContext {
                detections: &self.last_detections,
                fresh_cameras: &self.fresh_cameras,
                calibration: self.executor.calibration(),
                frame_index: index,
                timestamp_ms,
                caller,
            },
        );
        self.fresh_cameras.clear();
        DispatchStats {
            detections: self.last_detections.len() as u32,
            active_tracks: result.as_ref().map_or(0, |r| r.active_tracks),
            ball_present: result.as_ref().is_some_and(|r| r.ball_present),
        }
    }

    /// Run trackers only (no panner advance) - the lookahead produce
    /// phase. Consumes measurement freshness once while retaining the cache; returns the assembled
    /// [`WorldState`] for buffering.
    pub fn update_trackers(&mut self, index: u64, timestamp_ms: f64) -> WorldState {
        let world = crate::detect::panner::run_trackers(
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            crate::detect::panner::DispatchContext {
                detections: &self.last_detections,
                fresh_cameras: &self.fresh_cameras,
                calibration: self.executor.calibration(),
                frame_index: index,
                timestamp_ms,
                caller: "lookahead_produce",
            },
        );
        self.fresh_cameras.clear();
        world
    }

    /// The buffered (lookahead) panner step: decide the pose for
    /// `world` while seeing `futures`, advancing `requested_world_pose`.
    /// Without a panner the pose stays at its latest request.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) drives this
    pub(crate) fn decide_pose_with_lookahead(
        &mut self,
        world: &WorldState,
        futures: &[WorldState],
        index: u64,
        timestamp_ms: f64,
    ) -> WorldPose {
        let Some(panner) = self.panner.as_mut() else {
            return self.requested_world_pose;
        };
        let ctx = crate::detect::panner::PanContext {
            frame_index: index,
            timestamp_ms,
            previous_pose: self.requested_world_pose,
            calibration: self.executor.calibration(),
        };
        let pose = panner.decide_with_lookahead(world, futures, &ctx);
        self.requested_world_pose = pose;
        pose
    }

    /// Whether the attached detector prefers native CUDA NV12 input. The Linux NVDEC session path uses this to select CUDA pointers or wgpu preprocessing; other source paths select their own frame residency.
    pub fn detector_prefers_cuda_nv12(&self) -> bool {
        self.detector
            .as_ref()
            .is_some_and(|d| d.prefers_cuda_nv12())
    }

    /// Run the attached detector over prebuilt per-camera frames (any
    /// residency), map the raw detections to world-reference angles, and
    /// replace those cameras' cached observations for tracking. Successful empty
    /// results clear that camera's observations. Omitted cameras retain their
    /// observations. Failed inference clears older observations, but preserves a
    /// successful measurement not yet consumed by tracking. No inference without a detector.
    ///
    /// Temporary inference errors are warned-and-dropped. Unsupported frame kinds return a configuration error identifying the detector, camera and supplied input. Successful measurements from other cameras are retained, including when this call returns an error. See `unsupported_detector_input_is_reported_in_immediate_and_lookahead_sessions`.
    ///
    /// `capture_times_us` is empty for unidentified images, or contains one nonnegative capture time per calibration camera. An already successfully analysed image is not inferred again; failed inference can be retried. Replacing the detector or invalidating calibration clears this history.
    ///
    /// Invalid camera indices, timestamp shape or regressing capture times return an error before inference. See `capture_identity_handles_retry_empty_results_and_invalidation`.
    pub fn run_detection_frames(
        &mut self,
        frames: &[(CameraIndex, DetectorFrame<'_>)],
        capture_times_us: &[i64],
    ) -> Result<(), super::StitchCoreError> {
        let count = self.calibration().projection.camera_count();
        if (!capture_times_us.is_empty() && capture_times_us.len() != count)
            || capture_times_us.iter().any(|&time| time < 0)
            || frames.iter().any(|(camera, _)| *camera >= count)
        {
            return Err(super::StitchCoreError::Config(
                "detection cameras and capture times must match the calibration".into(),
            ));
        }
        for (camera, _) in frames {
            if let (Some(&time), Some(Some(last))) = (
                capture_times_us.get(*camera),
                self.detected_capture_times.get(*camera),
            ) && time < *last
            {
                return Err(super::StitchCoreError::Config(
                    "capture time precedes an already analysed image".into(),
                ));
            }
        }
        let Some(ref mut detector) = self.detector else {
            return Ok(());
        };
        self.detected_capture_times.resize(count, None);
        let mut out = Vec::new();
        let mut unsupported_input = None;
        for (camera, frame) in frames {
            let capture_time = capture_times_us.get(*camera).copied();
            if capture_time.is_some() && self.detected_capture_times[*camera] == capture_time {
                continue;
            }
            match detector.detect(*camera, frame) {
                Ok(v) => {
                    self.detected_capture_times[*camera] = capture_time;
                    if !self.fresh_cameras.contains(camera) {
                        self.fresh_cameras.push(*camera);
                    }
                    self.last_detections
                        .retain(|detection| detection.camera != *camera);
                    out.retain(|detection: &Detection| detection.camera != *camera);
                    out.extend(v);
                }
                Err(error) => {
                    if !self.fresh_cameras.contains(camera) {
                        self.last_detections
                            .retain(|detection| detection.camera != *camera);
                    }
                    match error {
                        DetectorError::UnsupportedFrameKind => {
                            unsupported_input.get_or_insert_with(|| {
                                super::StitchCoreError::Config(format!(
                                    "detector '{}' does not support {} input for camera {camera}",
                                    detector.name(),
                                    frame.variant_name()
                                ))
                            });
                        }
                        error => log::warn!(
                            "StitchCore detector '{}' {camera:?}: {error}",
                            detector.name()
                        ),
                    }
                }
            }
        }
        if !out.is_empty() {
            let mapped = match self.map_detections(out) {
                Ok(mapped) => mapped,
                Err(error) => {
                    self.clear_detection_cache();
                    return Err(error);
                }
            };
            self.last_detections.extend(mapped);
            self.last_detections
                .sort_by_key(|detection| detection.camera);
        }
        unsupported_input.map_or(Ok(()), Err)
    }

    /// Expire held measurements before a source-frame tracking tick, including skipped inference ticks.
    pub(crate) fn expire_observations(
        &mut self,
        capture: Option<&crate::source::CaptureTiming>,
        frame_seconds: f32,
    ) -> Result<(), super::StitchCoreError> {
        let Some(capture) = capture else {
            return Ok(());
        };
        capture
            .validate(self.calibration().projection.camera_count())
            .map_err(super::StitchCoreError::Config)?;
        let Some(max_age) = capture.max_age_us else {
            return Ok(());
        };
        let interval_us =
            (f64::from(frame_seconds) * 1_000_000.0 * self.detection_interval as f64).ceil();
        if !interval_us.is_finite() || interval_us <= 0.0 || interval_us >= u64::MAX as f64 {
            return Err(super::StitchCoreError::Config(
                "invalid detection interval duration".into(),
            ));
        }
        let limit = max_age.checked_add(interval_us as u64).ok_or_else(|| {
            super::StitchCoreError::Config("observation retention duration overflow".into())
        })?;
        let retained = |camera: usize| {
            self.detected_capture_times
                .get(camera)
                .copied()
                .flatten()
                .is_some_and(|time| time <= capture.at_us && (capture.at_us - time) as u64 <= limit)
        };
        self.last_detections
            .retain(|detection| retained(detection.camera));
        self.fresh_cameras.retain(|&camera| retained(camera));
        // Identity survives expiry so re-presenting an old image cannot make it new evidence.
        Ok(())
    }

    /// Wrap CPU-resident cameras in projection order for the shared detector.
    pub(crate) fn run_cpu_detection(
        &mut self,
        frame: &crate::source::FrameSet,
        capture_times_us: &[i64],
    ) -> Result<(), super::StitchCoreError> {
        use crate::source::FrameSet;
        frame
            .validate_host_images(self.source_sizes())
            .map_err(super::StitchCoreError::Config)?;
        let frames: Vec<_> = match frame {
            FrameSet::Yuv420p(cameras) => cameras
                .iter()
                .enumerate()
                .map(|(camera, data)| {
                    (
                        camera,
                        DetectorFrame::Yuv(YuvView {
                            y: &data.y,
                            chroma: ChromaPlanes::Yuv420p {
                                u: &data.u,
                                v: &data.v,
                            },
                            width: data.size.0,
                            height: data.size.1,
                        }),
                    )
                })
                .collect(),
            FrameSet::Nv12(cameras) => cameras
                .iter()
                .enumerate()
                .map(|(camera, data)| {
                    (
                        camera,
                        DetectorFrame::Yuv(YuvView {
                            y: &data.y,
                            chroma: ChromaPlanes::Nv12 { uv: &data.uv },
                            width: data.size.0,
                            height: data.size.1,
                        }),
                    )
                })
                .collect(),
            _ => return Ok(()),
        };
        self.run_detection_frames(&frames, capture_times_us)
    }

    /// Enrich raw source-image detections with world-reference angular positions for tracking.
    pub(super) fn map_detections(
        &mut self,
        mut detections: Vec<Detection>,
    ) -> Result<Vec<MappedDetection>, super::StitchCoreError> {
        let calibration = self.executor.calibration();
        if let Some(filter) = &mut self.detection_filter {
            filter.retain(calibration, &mut detections)?;
        }
        Ok(detections
            .into_iter()
            .map(|d| {
                let position = calibration
                    .map_source_point(d.camera, [d.center_x, d.center_y])
                    .ok();
                MappedDetection {
                    camera: d.camera,
                    class_id: d.class_id,
                    confidence: d.confidence,
                    camera_center: (d.center_x, d.center_y),
                    camera_size: (d.width, d.height),
                    position,
                }
            })
            .collect())
    }
}
