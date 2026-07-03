//! Pose resolution and detection internals for
//! [`StitchCore`](super::StitchCore).
//!
//! Contains the internal pose-resolution logic (`resolve_current_pose`),
//! the detection schedule (`should_run_detection`), YUV detection
//! dispatch, panorama-coordinate mapping, and session-start anchoring.

use crate::detect::detector::{ChromaFormat, Detection, DetectorError, DetectorFrame, RawFrame};
use crate::detect::director::MappedDetection;
use crate::detect::tracker::WorldState;
use crate::geometry::CameraId;
use crate::geometry::ViewportPosition;
use crate::projection;
use crate::render::planes::YuvPlanes;

/// What one [`StitchCore::dispatch_pose`] tick produced - the numbers
/// the session's telemetry records plus the panner's raw decision.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) reads these
pub(crate) struct DispatchStats {
    /// The panner's decided pose (world-space, pre-clamp). `None` when
    /// no panner is attached.
    pub pose: Option<ViewportPosition>,
    /// Panorama-mapped detections the dispatch consumed.
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

    pub(super) fn resolve_current_pose(&mut self, fresh_detection: bool) -> ViewportPosition {
        // Pull raw director output (or default) and clamp through
        // coverage. Then write the resolved FOV back onto the pipeline
        // so the upcoming render uses it.
        //
        // `fresh_detection` is the ACTUAL run decision for this frame,
        // not the schedule-would-fire predicate. The BGRA submit path
        // deliberately skips detection (no BGRA-aware backend exists
        // today) so it must pass `false` even when the interval would
        // have fired - otherwise directors over-count hysteresis on
        // stale cached detections.
        let timestamp_ms = self
            .session_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let _ = fresh_detection; // reserved for future freshness-aware panners
        let raw = self
            .dispatch_pose(self.frame_count, timestamp_ms, "StitchCore")
            .pose
            .unwrap_or_default();
        // The toggle gates only the coverage clamp; the tilt/roll basis
        // inversion is unconditional - a tilted rig must render level
        // whether or not the viewport is constrained.
        let clamped = if self.constrained_look {
            self.safe_clamp(raw)
        } else {
            self.orient_pose(raw)
        };
        if let Some(fov) = clamped.fov_degrees {
            self.executor.set_fov(fov);
        }
        clamped
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
    /// and timestamp: emits `DetectionsRaw`, runs the shared
    /// [`panner::dispatch`](crate::detect::panner) (which emits
    /// `WorldState`/`PanDecision` through the engine's sink), advances
    /// `previous_panner_pose`, and reports the dispatch stats.
    pub(crate) fn dispatch_pose(
        &mut self,
        index: u64,
        timestamp_ms: f64,
        caller: &'static str,
    ) -> DispatchStats {
        if let Some(sink) = self.event_sink.as_deref_mut() {
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::DetectionsRaw {
                    frame_index: index,
                    detections: self.last_detections.clone(),
                },
            );
        }
        let result = crate::detect::panner::dispatch(
            self.panner.as_mut(),
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            &mut self.previous_panner_pose,
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
    pub fn track_only(&mut self, index: u64, timestamp_ms: f64) -> WorldState {
        crate::detect::panner::dispatch_detect_only(
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
    /// `world` while seeing `futures`, advancing `previous_panner_pose`.
    /// Without a panner the pose stays wherever the loop last seeded it
    /// (see [`Self::set_previous_panner_pose`]).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) drives this
    pub(crate) fn decide_pose_with_lookahead(
        &mut self,
        world: &WorldState,
        futures: &[WorldState],
        index: u64,
        timestamp_ms: f64,
    ) -> ViewportPosition {
        let Some(panner) = self.panner.as_mut() else {
            return self.previous_panner_pose;
        };
        let ctx = crate::detect::panner::PanContext {
            frame_index: index,
            timestamp_ms,
            previous_position: self.previous_panner_pose,
            calibration: self.executor.calibration(),
        };
        let pose = panner.decide_with_lookahead(world, futures, &ctx);
        self.previous_panner_pose = pose;
        pose
    }

    /// Seed the pose the next presentation clamp and panner tick see.
    /// The buffered loop writes its post-smoothed pose back through
    /// this before rendering.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) drives this
    pub(crate) fn set_previous_panner_pose(&mut self, pose: ViewportPosition) {
        self.previous_panner_pose = pose;
    }

    /// The pull session's presentation pose: always-clamped resolve of
    /// the panner's latest decision (batch export never reveals black
    /// edges, independent of the constrained-look toggle), with the
    /// `PosePresented` trace and the FOV write-back.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // session (gpu-gated) drives this
    pub(crate) fn presented_clamped_pose(&mut self, index: u64) -> ViewportPosition {
        let pos = self.safe_clamp(self.previous_panner_pose);
        if let Some(sink) = self.event_sink.as_deref_mut() {
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::PosePresented {
                    frame_index: index,
                    pose: pos,
                },
            );
        }
        if let Some(fov) = pos.fov_degrees {
            self.executor.set_fov(fov);
        }
        pos
    }

    /// Whether the attached detector consumes CUDA device pointers
    /// (the ORT CUDA backend). When false, CUDA texture imports are
    /// skipped so wgpu compute preprocessing keeps ownership of the
    /// D3D11 textures.
    pub fn detector_needs_cuda_frames(&self) -> bool {
        self.detector
            .as_ref()
            .is_some_and(|d| d.name().contains("ort-cuda"))
    }

    /// Run the attached detector over prebuilt per-camera frames (any
    /// residency), map the raw detections to panorama coordinates, and
    /// cache them for the director. No-op without a detector.
    ///
    /// Errors are warned-and-dropped: a flaky inference call must not
    /// crash the render loop. `UnsupportedFrameKind` logs at debug -
    /// it is the expected answer while a caller probes backends.
    pub fn run_detection_frames(&mut self, frames: &[(CameraId, DetectorFrame<'_>)]) {
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
        self.last_detections = self.map_detections_to_panorama(out);
    }

    /// Run detection on a stereo YUV420P frame pair - wraps each
    /// camera's planes as [`RawFrame`] + [`DetectorFrame::Cpu`] and
    /// feeds [`Self::run_detection_frames`].
    pub(super) fn run_yuv_detection(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        source_width: u32,
        source_height: u32,
    ) {
        self.run_detection_frames(&[
            (
                CameraId::Left,
                DetectorFrame::Cpu(RawFrame {
                    y: left.y,
                    chroma: ChromaFormat::Yuv420p {
                        u: left.u,
                        v: left.v,
                    },
                    width: source_width,
                    height: source_height,
                }),
            ),
            (
                CameraId::Right,
                DetectorFrame::Cpu(RawFrame {
                    y: right.y,
                    chroma: ChromaFormat::Yuv420p {
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
                CameraId::Left,
                DetectorFrame::Cpu(RawFrame {
                    y: left.y,
                    chroma: ChromaFormat::Nv12 { uv: left.uv },
                    width: source_width,
                    height: source_height,
                }),
            ),
            (
                CameraId::Right,
                DetectorFrame::Cpu(RawFrame {
                    y: right.y,
                    chroma: ChromaFormat::Nv12 { uv: right.uv },
                    width: source_width,
                    height: source_height,
                }),
            ),
        ]);
    }

    /// Map raw camera-space detections to panorama-space
    /// [`MappedDetection`]s the director can consume.
    pub(super) fn map_detections_to_panorama(
        &self,
        detections: Vec<Detection>,
    ) -> Vec<MappedDetection> {
        let calibration = self.executor.calibration();
        let scene = self.executor.scene();
        detections
            .into_iter()
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
