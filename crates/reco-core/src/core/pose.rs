//! Pose resolution and detection internals for
//! [`StitchCore`](super::StitchCore).
//!
//! Contains the internal pose-resolution logic (`resolve_current_pose`),
//! the detection schedule (`should_run_detection`), YUV detection
//! dispatch, panorama-coordinate mapping, and session-start anchoring.

use crate::detect::detector::{CameraId, ChromaFormat, Detection, DetectorFrame, RawFrame};
use crate::detect::director::{MappedDetection, ViewportPosition};
use crate::projection;
use crate::render::pipeline::YuvPlanes;

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

        // Pose resolution: run all registered trackers in order, assemble
        // a [`WorldState`], and let the panner decide via the shared
        // `panner::dispatch` helper. When no panner is attached the pose
        // stays at the pipeline default.
        let _ = fresh_detection; // reserved for future freshness-aware panners
        let raw = crate::detect::panner::dispatch(
            self.panner.as_mut(),
            self.player_tracker.as_mut(),
            self.ball_tracker.as_mut(),
            &mut self.previous_panner_pose,
            // StitchCore does not own an event sink. StitchSession
            // does the tracing when it is the active entry point.
            None,
            &[],
            crate::detect::panner::DispatchContext {
                detections: &self.last_detections,
                calibration: &self.pipeline.calibration,
                frame_index: self.frame_count,
                timestamp_ms,
                caller: "StitchCore",
            },
        )
        .map(|r| r.pose)
        .unwrap_or_default();
        let clamped = if self.constrained_look {
            self.safe_clamp(raw)
        } else {
            raw
        };
        if let Some(fov) = clamped.fov_degrees {
            self.pipeline.set_fov(fov);
        }
        clamped
    }

    pub(super) fn should_run_detection(&self) -> bool {
        // Before any submit, frame_count == 0 and interval defaults to
        // 1, which covers the "run on the very first frame" case.
        self.detection_interval > 0 && self.frame_count.is_multiple_of(self.detection_interval)
    }

    /// Run the attached detector against a stereo YUV420P frame pair.
    ///
    /// Wraps each plane as [`RawFrame`] + [`DetectorFrame::Cpu`] and
    /// dispatches through the unified trait, once per camera. Errors
    /// are warned-and-dropped: a flaky inference call must not crash
    /// the render loop.
    pub(super) fn run_yuv_detection(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        source_width: u32,
        source_height: u32,
    ) -> Vec<Detection> {
        let Some(ref mut detector) = self.detector else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (camera, planes) in [(CameraId::Left, left), (CameraId::Right, right)] {
            let raw = RawFrame {
                y: planes.y,
                chroma: ChromaFormat::Yuv420p {
                    u: planes.u,
                    v: planes.v,
                },
                width: source_width,
                height: source_height,
            };
            match detector.detect(camera, &DetectorFrame::Cpu(raw)) {
                Ok(v) => out.extend(v),
                Err(e) => log::warn!("StitchCore detector '{}' {camera:?}: {e}", detector.name()),
            }
        }
        out
    }

    /// Map raw camera-space detections to panorama-space
    /// [`MappedDetection`]s the director can consume.
    pub(super) fn map_detections_to_panorama(
        &self,
        detections: Vec<Detection>,
    ) -> Vec<MappedDetection> {
        let calibration = self.pipeline.calibration();
        let scene = &self.pipeline.scene;
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
