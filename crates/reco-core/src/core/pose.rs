//! Detection scheduling, source-to-world mapping and world-pose requests for [`StitchCore`](super::StitchCore).

use crate::detect::detector::{ChromaPlanes, Detection, DetectorError, DetectorFrame, YuvView};
use crate::detect::mapped_detection::MappedDetection;
use crate::detect::tracker::WorldState;
use crate::geometry::CameraIndex;
use crate::geometry::WorldPose;
use crate::render::planes::Yuv420pPlanes;

/// What one [`StitchCore::dispatch_pose`] tick produced - the numbers
/// the session's telemetry records plus the panner's raw decision.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) reads these
pub(crate) struct DispatchStats {
    /// The panner's decided pose (world-space, pre-clamp). `None` when
    /// no panner is attached.
    pub pose: Option<WorldPose>,
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
        self.dispatch_pose(self.frame_count, timestamp_ms, "StitchCore")
            .pose
            .unwrap_or_default()
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
                calibration: self.executor.calibration(),
                frame_index: index,
                timestamp_ms,
                caller,
            },
        );
        DispatchStats {
            pose: result.as_ref().map(|r| r.pose),
            detections: self.last_detections.len() as u32,
            active_tracks: result.as_ref().map_or(0, |r| r.active_tracks),
            ball_present: result.as_ref().is_some_and(|r| r.ball_present),
        }
    }

    /// Run trackers only (no panner advance) - the lookahead produce
    /// phase. Consumes the cached detections; returns the assembled
    /// [`WorldState`] for buffering.
    pub fn update_trackers(&mut self, index: u64, timestamp_ms: f64) -> WorldState {
        crate::detect::panner::run_trackers(
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            crate::detect::panner::DispatchContext {
                detections: &self.last_detections,
                calibration: self.executor.calibration(),
                frame_index: index,
                timestamp_ms,
                caller: "lookahead_produce",
            },
        )
    }

    /// The buffered (lookahead) panner step: decide the pose for
    /// `world` while seeing `futures`, advancing `requested_world_pose`.
    /// Without a panner the pose stays wherever the loop last seeded it
    /// (see [`Self::set_requested_world_pose`]).
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

    /// Seed the pose the next presentation clamp and panner tick see.
    /// The buffered loop writes its post-smoothed pose back through
    /// this before rendering.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) drives this
    pub(crate) fn set_requested_world_pose(&mut self, pose: WorldPose) {
        self.requested_world_pose = pose;
    }

    /// Whether the attached detector prefers native CUDA NV12 input. The Linux NVDEC session path uses this to select CUDA pointers or wgpu preprocessing; other source paths select their own frame residency.
    pub fn detector_prefers_cuda_nv12(&self) -> bool {
        self.detector
            .as_ref()
            .is_some_and(|d| d.prefers_cuda_nv12())
    }

    /// Run the attached detector over prebuilt per-camera frames (any
    /// residency), map the raw detections to world-reference angles, and
    /// cache them for tracking. No-op without a detector.
    ///
    /// Errors are warned-and-dropped: a flaky inference call must not
    /// crash the render loop. `UnsupportedFrameKind` logs at debug -
    /// it is the expected answer while a caller probes backends.
    pub fn run_detection_frames(&mut self, frames: &[(CameraIndex, DetectorFrame<'_>)]) {
        let Some(ref mut detector) = self.detector else {
            return;
        };
        let mut out = Vec::new();
        for (camera, frame) in frames {
            match detector.detect(*camera, frame) {
                Ok(v) => out.extend(v),
                Err(DetectorError::UnsupportedFrameKind) => log::debug!(
                    "StitchCore detector '{}' does not support this frame residency ({camera:?})",
                    detector.name()
                ),
                Err(e) => log::warn!("StitchCore detector '{}' {camera:?}: {e}", detector.name()),
            }
        }
        self.last_detections = self.map_detections(out);
    }

    /// Run detection on a stereo YUV420P frame pair - wraps each
    /// camera's planes as [`YuvView`] + [`DetectorFrame::Yuv`] and
    /// feeds [`Self::run_detection_frames`].
    pub(super) fn run_yuv_detection(
        &mut self,
        left: &Yuv420pPlanes<'_>,
        right: &Yuv420pPlanes<'_>,
        source_width: u32,
        source_height: u32,
    ) {
        self.run_detection_frames(&[
            (
                0,
                DetectorFrame::Yuv(YuvView {
                    y: left.y,
                    chroma: ChromaPlanes::Yuv420p {
                        u: left.u,
                        v: left.v,
                    },
                    width: source_width,
                    height: source_height,
                }),
            ),
            (
                1,
                DetectorFrame::Yuv(YuvView {
                    y: right.y,
                    chroma: ChromaPlanes::Yuv420p {
                        u: right.u,
                        v: right.v,
                    },
                    width: source_width,
                    height: source_height,
                }),
            ),
        ]);
    }

    /// NV12 sibling of [`Self::run_yuv_detection`] - same dispatch,
    /// interleaved-chroma frames (camera and X5 sources are NV12-native).
    pub(super) fn run_nv12_detection(
        &mut self,
        left: &crate::render::planes::Nv12Planes<'_>,
        right: &crate::render::planes::Nv12Planes<'_>,
        source_width: u32,
        source_height: u32,
    ) {
        self.run_detection_frames(&[
            (
                0,
                DetectorFrame::Yuv(YuvView {
                    y: left.y,
                    chroma: ChromaPlanes::Nv12 { uv: left.uv },
                    width: source_width,
                    height: source_height,
                }),
            ),
            (
                1,
                DetectorFrame::Yuv(YuvView {
                    y: right.y,
                    chroma: ChromaPlanes::Nv12 { uv: right.uv },
                    width: source_width,
                    height: source_height,
                }),
            ),
        ]);
    }

    /// Enrich raw source-image detections with world-reference angular positions for tracking.
    pub(super) fn map_detections(&self, detections: Vec<Detection>) -> Vec<MappedDetection> {
        let calibration = self.executor.calibration();
        detections
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
            .collect()
    }
}
