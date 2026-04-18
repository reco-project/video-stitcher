//! Detection-only pipeline for analytics consumers.
//!
//! [`AnalyzePipeline`] is a lightweight sibling of
//! [`StitchSession`](crate::session::StitchSession) for consumers that
//! need detection data but not a stitched video output (heatmaps,
//! highlight reels, coaching stats, offline analytics).
//!
//! Compared to `StitchSession` it omits:
//! - GPU render pipeline
//! - NV12 converter and readback buffers
//! - Encoder and async encode thread
//! - Director and viewport clamping
//!
//! What remains is the detection pipeline plus the calibration+scene
//! needed to map per-camera detections into panorama coordinates. A
//! typical analytics consumer builds a [`FrameSource`] (usually a file
//! decoder), attaches a detector and a [`DetectionSink`], and calls
//! [`AnalyzePipeline::run`] to drive the loop.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_core::analyze::AnalyzePipeline;
//!
//! let mut pipeline = AnalyzePipeline::new(calibration);
//! pipeline.set_detector(Box::new(my_detector));
//! pipeline.set_detection_sink(Box::new(|dets, frame_idx, ts_ms| {
//!     csv_writer.write_row(dets, frame_idx, ts_ms)?;
//!     Ok(())
//! }));
//!
//! let processed = pipeline.run(&mut source, u64::MAX, &interrupted, None)?;
//! ```

use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

use crate::calibration::MatchCalibration;
use crate::detector::Detection;
use crate::director::MappedDetection;
use crate::projection::{self, CoverageBoundary, PanoramaExtent};
use crate::scene::SceneGeometry;
use crate::session::detection::DetectionPipeline;
use crate::session::{DetectionSink, DetectionSinkError, FrameProgress, ProgressCallback};
use crate::source::{FrameSource, SourceError, StereoFrame};

/// Errors from [`AnalyzePipeline::run`].
#[derive(Debug, Error)]
pub enum AnalyzeError {
    /// Frame source error.
    #[error("source: {0}")]
    Source(#[from] SourceError),

    /// Detection sink error.
    #[error("detection sink: {0}")]
    DetectionSink(#[source] DetectionSinkError),
}

/// Detection-only pipeline that decodes frames and fires a sink without
/// running GPU render or encode.
///
/// Owns the detector and [`DetectionSink`]; holds calibration + scene so
/// it can map camera-space detections to panorama yaw/pitch. Intended for
/// offline analytics and standalone detection consumers that would
/// otherwise pay the full stitch+encode cost just to reach the detection
/// callback.
pub struct AnalyzePipeline {
    calibration: MatchCalibration,
    scene: SceneGeometry,
    coverage: CoverageBoundary,
    detection: DetectionPipeline,
    frame_count: u64,
}

impl AnalyzePipeline {
    /// Create an analyze pipeline for the given camera calibration.
    ///
    /// Builds the scene geometry and coverage boundary up-front so
    /// [`panorama_extent`](Self::panorama_extent) is available before the
    /// first frame is processed. No GPU or encoder resources are
    /// allocated.
    pub fn new(calibration: MatchCalibration) -> Self {
        let aspect = calibration.left.width as f32 / calibration.left.height as f32;
        let scene = SceneGeometry::from_layout_with_aspect(&calibration.layout, aspect);
        let coverage = CoverageBoundary::from_calibration(&calibration, &scene);

        Self {
            calibration,
            scene,
            coverage,
            detection: DetectionPipeline::new(),
            frame_count: 0,
        }
    }

    /// Attach a unified-trait detector for object detection on
    /// decoded frames.
    pub fn set_detector(&mut self, detector: Box<dyn crate::detector::UnifiedDetector>) {
        self.detection.set_detector(detector);
    }

    /// Set the detection interval (run detection every N frames).
    pub fn set_detection_interval(&mut self, interval: u64) {
        self.detection.set_detection_interval(interval);
    }

    /// Set a fallible sink for receiving tracked detection data.
    ///
    /// Replaces any previously registered sink. Sink errors abort
    /// [`run`](Self::run) with [`AnalyzeError::DetectionSink`].
    pub fn set_detection_sink(&mut self, sink: Box<dyn DetectionSink>) {
        self.detection.set_sink(sink);
    }

    /// The precomputed coverage boundary derived from the calibration.
    pub fn coverage(&self) -> &CoverageBoundary {
        &self.coverage
    }

    /// Full angular extent of the stitched panorama (yaw/pitch ranges).
    pub fn panorama_extent(&self) -> PanoramaExtent {
        let (yaw_min, yaw_max) = self.coverage.yaw_range();
        let (pitch_min, pitch_max) = self.coverage.pitch_range();
        PanoramaExtent {
            yaw_min,
            yaw_max,
            pitch_min,
            pitch_max,
        }
    }

    /// Number of frames processed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Run the decode → detect → sink loop to completion.
    ///
    /// Blocks until the source is exhausted, the frame limit is reached,
    /// or `interrupted` is set. Returns the number of frames processed.
    ///
    /// GPU-resident frames (`StereoFrame::GpuResident`,
    /// `StereoFrame::MetalResident`) are counted but skipped for
    /// detection since they carry no CPU-accessible pixel data — the
    /// CPU-only decoder path is the intended companion for this method.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "analyze_run")
    )]
    pub fn run(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        mut on_progress: Option<ProgressCallback>,
    ) -> Result<u64, AnalyzeError> {
        let info = source.info();
        let (width, height) = (info.width, info.height);
        let start = std::time::Instant::now();

        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let frame = match source.next_frame()? {
                Some(f) => f,
                None => break,
            };

            let should_detect = self.detection.should_detect(self.frame_count);

            if should_detect && self.is_cpu_frame(&frame) {
                let raw = self.detection.run_detection(&frame, width, height);
                self.detection.set_last_detections(self.map_detections(raw));
            }

            let timestamp_ms = start.elapsed().as_secs_f64() * 1000.0;
            self.detection
                .fire_sink(self.frame_count, timestamp_ms)
                .map_err(AnalyzeError::DetectionSink)?;

            self.frame_count += 1;

            if let Some(cb) = on_progress.as_mut() {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        Ok(self.frame_count)
    }

    /// Whether the frame carries CPU-accessible plane data.
    fn is_cpu_frame(&self, frame: &StereoFrame) -> bool {
        matches!(frame, StereoFrame::Yuv420p(_) | StereoFrame::Nv12(_))
    }

    fn map_detections(&self, detections: Vec<Detection>) -> Vec<MappedDetection> {
        detections
            .iter()
            .map(|d| {
                let position = projection::camera_to_panorama(
                    d.camera,
                    d.center_x,
                    d.center_y,
                    &self.calibration,
                    &self.scene,
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
