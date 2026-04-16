//! One-shot detection-only job (analytics sibling of
//! [`StitchJob`](crate::StitchJob)).
//!
//! [`AnalyzeJob`] decodes one or more paired videos, runs a detector on
//! each CPU-resident frame, and fires a [`DetectionSink`] per frame.
//! There is no GPU stitch pipeline, no NV12 converter, and no encoder —
//! analytics consumers (heatmaps, highlight reels, offline stats) use
//! this to avoid paying for render + encode just to reach the sink.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_io::AnalyzeJob;
//!
//! let interrupted = std::sync::atomic::AtomicBool::new(false);
//! let result = AnalyzeJob::new("left.mp4", "right.mp4", "match.json")
//!     .detector(Box::new(my_detector))
//!     .on_detections(Box::new(|dets, frame_idx, ts_ms| {
//!         csv.write_row(dets, frame_idx, ts_ms)?;
//!         Ok(())
//!     }))
//!     .run(&interrupted)?;
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use reco_core::analyze::{AnalyzeError, AnalyzePipeline};
use reco_core::detector::Detector;
use reco_core::session::{DetectionSink, FrameProgress};
use reco_core::source::FrameSource;

use crate::stitch_job::{InputPath, StitchError};

/// Boxed progress callback type alias to satisfy clippy::type_complexity.
type ProgressCallback = Box<dyn FnMut(&FrameProgress) + Send>;

/// Where to load calibration from.
enum CalibrationSource {
    /// Load from a JSON file path.
    File(PathBuf),
    /// Use an in-memory calibration (no file I/O).
    Memory(Box<reco_core::calibration::MatchCalibration>),
}

/// One-shot detection-only job: video files in, per-frame detections out.
///
/// No GPU, no encoder, no render. Intended for analytics consumers
/// (heatmaps, highlights, stats) that need only detection data mapped
/// into panorama coordinates.
pub struct AnalyzeJob {
    left: InputPath,
    right: InputPath,
    calibration: CalibrationSource,

    // Processing settings
    max_frames: Option<u64>,
    duration: Option<f64>,
    sync_offset: Option<i64>,
    detection_interval: u64,

    // Detector + sink
    detector: Option<Box<dyn Detector>>,
    sink: Option<Box<dyn DetectionSink>>,

    // Callback
    on_progress: Option<ProgressCallback>,
}

/// Result of a completed analyze job.
#[derive(Debug)]
pub struct AnalyzeResult {
    /// Number of frames processed.
    pub frames_processed: u64,
    /// Total wall-clock time.
    pub elapsed: Duration,
    /// Decode mode (e.g. "CPU upload").
    pub decode_mode: String,
}

impl AnalyzeResult {
    /// Average frames per second.
    pub fn fps(&self) -> f64 {
        self.frames_processed as f64 / self.elapsed.as_secs_f64()
    }
}

impl AnalyzeJob {
    /// Create an analyze job from file paths (loads calibration from JSON).
    pub fn new(
        left: impl Into<InputPath>,
        right: impl Into<InputPath>,
        calibration: impl AsRef<Path>,
    ) -> Self {
        Self {
            left: left.into(),
            right: right.into(),
            calibration: CalibrationSource::File(calibration.as_ref().to_path_buf()),
            max_frames: None,
            duration: None,
            sync_offset: None,
            detection_interval: 1,
            detector: None,
            sink: None,
            on_progress: None,
        }
    }

    /// Create an analyze job with an in-memory calibration.
    pub fn with_calibration(
        left: impl Into<InputPath>,
        right: impl Into<InputPath>,
        calibration: reco_core::calibration::MatchCalibration,
    ) -> Self {
        let mut job = Self::new(left, right, Path::new(""));
        job.calibration = CalibrationSource::Memory(Box::new(calibration));
        job
    }

    // ── Processing settings ──

    /// Limit the number of frames to process.
    pub fn max_frames(mut self, n: u64) -> Self {
        self.max_frames = Some(n);
        self
    }

    /// Limit processing to a duration in seconds.
    pub fn duration(mut self, secs: f64) -> Self {
        self.duration = Some(secs);
        self
    }

    /// Override the temporal sync offset between cameras (frames).
    pub fn sync_offset(mut self, frames: i64) -> Self {
        self.sync_offset = Some(frames);
        self
    }

    /// Run detection every N frames (default 1, every frame).
    pub fn detection_interval(mut self, interval: u64) -> Self {
        self.detection_interval = interval.max(1);
        self
    }

    // ── Detector + sink ──

    /// Attach the CPU detector. Required before [`run`](Self::run).
    pub fn detector(mut self, detector: Box<dyn Detector>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Attach the per-frame detection sink. Required before [`run`](Self::run).
    pub fn on_detections(mut self, sink: Box<dyn DetectionSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Attach a progress callback. Called per processed frame.
    pub fn on_progress(mut self, cb: impl FnMut(&FrameProgress) + Send + 'static) -> Self {
        self.on_progress = Some(Box::new(cb));
        self
    }

    // ── Execute ──

    /// Run the analyze job.
    ///
    /// Blocking call. Returns an [`AnalyzeResult`] with statistics, or
    /// an error from calibration loading, source opening, decode, or the
    /// detection sink.
    pub fn run(mut self, interrupted: &AtomicBool) -> Result<AnalyzeResult, StitchError> {
        crate::init();
        let start = std::time::Instant::now();

        // Load calibration
        let cal = match self.calibration {
            CalibrationSource::File(ref path) => {
                reco_core::calibration::MatchCalibration::from_file(path)
                    .map_err(|e| StitchError::Calibration(format!("{e}")))?
            }
            CalibrationSource::Memory(cal) => *cal,
        };
        let effective_sync = self.sync_offset.unwrap_or(cal.sync_offset);

        // Open the CPU decode path. GPU-resident sources won't yield
        // CPU-accessible frames, which is a problem for detection —
        // FfmpegFileSource stays on the CPU upload path.
        let mut source = crate::adapters::FfmpegFileSource::open_with_offset(
            self.left.first_path(),
            self.right.first_path(),
            effective_sync,
        )?;
        let info = source.info();
        let decode_mode = format!("{}", source.decode_backend());

        // Build analyze pipeline.
        let mut pipeline = AnalyzePipeline::new(cal);
        pipeline.set_detection_interval(self.detection_interval);
        if let Some(det) = self.detector.take() {
            pipeline.set_detector(det);
        } else {
            return Err(StitchError::Other(
                "AnalyzeJob: detector is required; call .detector(...)".into(),
            ));
        }
        if let Some(sink) = self.sink.take() {
            pipeline.set_detection_sink(sink);
        } else {
            return Err(StitchError::Other(
                "AnalyzeJob: detection sink is required; call .on_detections(...)".into(),
            ));
        }

        let frame_limit =
            reco_core::session::compute_frame_limit(self.duration, self.max_frames, info.fps);

        let frames_processed = pipeline
            .run(
                &mut source,
                frame_limit,
                interrupted,
                self.on_progress.take(),
            )
            .map_err(|e| match e {
                AnalyzeError::Source(s) => StitchError::Source(s),
                AnalyzeError::DetectionSink(e) => {
                    StitchError::Session(reco_core::session::SessionError::DetectionSink(e))
                }
            })?;

        Ok(AnalyzeResult {
            frames_processed,
            elapsed: start.elapsed(),
            decode_mode,
        })
    }
}
