//! High-level calibration pipeline orchestrator.
//!
//! [`CalibrationPipeline`] handles the full calibration workflow:
//! explicit optics, sync estimation, frame selection, and optimization.
//! The app is only responsible for providing video metadata and decoded
//! frames - all orchestration logic lives here.
//!
//! ```ignore
//! use reco_calibrate::pipeline::{CalibrationPipeline, VideoInfo};
//!
//! // App creates VideoInfo from its decoder
//! let left_info = VideoInfo { path: "left.mp4".into(), width: 3840, height: 2160, fps: 30.0, total_frames: 1800 };
//! let right_info = VideoInfo { path: "right.mp4".into(), width: 3840, height: 2160, fps: 30.0, total_frames: 1800 };
//!
//! let mut pipeline = CalibrationPipeline::new(left_info, right_info, CalibrationConfig::default());
//!
//! // Missing optics are estimated; optionally supply fixed lenses with set_profiles.
//!
//! // Sync: try IMU first, then audio, then manual
//! if pipeline.imu_sync().ok().flatten().is_none() {
//!     let left_audio = my_decoder.extract_audio("left.mp4")?;
//!     let right_audio = my_decoder.extract_audio("right.mp4")?;
//!     pipeline.audio_sync(&left_audio, &right_audio, 44100)?;
//! }
//!
//! // Get which frames to extract (sync already applied)
//! let (left_indices, right_indices) = pipeline.frame_indices();
//!
//! // App extracts frames with its decoder
//! let frames = my_decoder.extract_pairs(&left_indices, &right_indices)?;
//!
//! // Run calibration
//! let result = pipeline.calibrate(&std::sync::atomic::AtomicBool::new(false), &frames)?;
//! ```

use reco_core::calibration::Lens;

use crate::error::CalibrateError;
use crate::telemetry::TelemetryData;
use crate::types::{CalibrationConfig, CalibrationResult, Yuv420pFrame};
use crate::{audio_sync, sampling, telemetry};

/// Video metadata that the app provides from its decoder.
///
/// reco-calibrate never opens video files directly - the app is
/// responsible for decoding. This struct carries the metadata
/// needed for frame selection and sync.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// Path to the video file (used for metadata extraction only -
    /// telemetry, not for frame decoding).
    pub path: std::path::PathBuf,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second.
    pub fps: f64,
    /// Total frame count (estimated from duration * fps).
    pub total_frames: u64,
}

/// Orchestrates the full calibration workflow.
///
/// Manages lens profiles, sync estimation, frame selection, and the
/// final optimization. The app provides video metadata and decoded
/// frames; the pipeline handles everything else.
pub struct CalibrationPipeline {
    left_info: VideoInfo,
    right_info: VideoInfo,
    config: CalibrationConfig,
    left_params: Option<Lens>,
    right_params: Option<Lens>,
    /// Cached parsed telemetry for the left video, lazily populated when IMU data is requested. `Some(Ok(..))` after a successful parse, `Some(Err(()))` after a failure, `None` before first use.
    left_telemetry: Option<Result<TelemetryData, ()>>,
    /// Cached parsed telemetry for the right video. See [`Self::left_telemetry`].
    right_telemetry: Option<Result<TelemetryData, ()>>,
    sync_offset_frames: i64,
    /// Whether the camera has native gyroscope data (queried lazily for sync).
    /// When false, the full telemetry parse is skipped because derived
    /// quaternion data produces unreliable sync and rig tilt.
    has_native_gyro: Option<bool>,
    /// Rig tilt in radians (forward lean from vertical).
    rig_tilt: f64,
    /// Rig roll in radians (lateral lean).
    rig_roll: f64,
}

impl CalibrationPipeline {
    /// Create a new pipeline from video metadata and configuration.
    pub fn new(left_info: VideoInfo, right_info: VideoInfo, config: CalibrationConfig) -> Self {
        Self {
            left_info,
            right_info,
            config,
            left_params: None,
            right_params: None,
            left_telemetry: None,
            right_telemetry: None,
            sync_offset_frames: 0,
            has_native_gyro: None,
            rig_tilt: 0.0,
            rig_roll: 0.0,
        }
    }

    // ---------------------------------------------------------------
    // Step 1: Lens profiles
    // ---------------------------------------------------------------

    /// Set lens profiles manually (loaded by the app from files, UI, etc.).
    pub fn set_profiles(&mut self, left: Option<Lens>, right: Option<Lens>) {
        self.left_params = left;
        self.right_params = right;
    }

    /// Lazily populate the left-video telemetry cache for IMU processing; subsequent calls are no-ops.
    ///
    /// Parse failures are recorded as `Err(())` so we don't retry on
    /// every call - telemetry extraction is deterministic per-file.
    fn ensure_left_telemetry(&mut self) {
        if self.left_telemetry.is_none() {
            self.left_telemetry = Some(match telemetry::extract(&self.left_info.path) {
                Ok(t) => Ok(t),
                Err(e) => {
                    log::warn!(
                        "left telemetry extraction failed for {}: {e}",
                        self.left_info.path.display()
                    );
                    Err(())
                }
            });
        }
    }

    /// Lazily populate the right-video telemetry cache. See
    /// [`Self::ensure_left_telemetry`].
    fn ensure_right_telemetry(&mut self) {
        if self.right_telemetry.is_none() {
            self.right_telemetry = Some(match telemetry::extract(&self.right_info.path) {
                Ok(t) => Ok(t),
                Err(e) => {
                    log::warn!(
                        "right telemetry extraction failed for {}: {e}",
                        self.right_info.path.display()
                    );
                    Err(())
                }
            });
        }
    }

    // ---------------------------------------------------------------
    // Step 2: Sync estimation
    // ---------------------------------------------------------------

    /// Set sync offset manually (in frames).
    pub fn set_sync_offset(&mut self, frames: i64) {
        self.sync_offset_frames = frames;
    }

    /// Current sync offset in frames.
    pub fn sync_offset(&self) -> i64 {
        self.sync_offset_frames
    }

    /// Estimate sync offset and rig framing from IMU telemetry.
    ///
    /// Reads gyroscope data from both video files and cross-correlates
    /// to find the temporal offset. Also estimates rig tilt and roll.
    ///
    /// Returns the sync offset in frames, or `None` if telemetry is
    /// unavailable or cross-correlation fails.
    pub fn imu_sync(&mut self) -> Result<Option<i64>, CalibrateError> {
        if !*self.has_native_gyro.get_or_insert_with(|| {
            telemetry::extract_metadata(&self.left_info.path).is_ok_and(|m| m.has_native_gyro)
        }) {
            log::info!(
                "IMU sync skipped: camera has no native gyroscope (quaternion-derived \
                 gyro produces unreliable sync). Falling back to audio sync."
            );
            return Ok(None);
        }

        self.ensure_left_telemetry();
        self.ensure_right_telemetry();

        let left_telem = self
            .left_telemetry
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .ok_or_else(|| CalibrateError::InvalidConfig("left IMU extraction failed".into()))?;
        let right_telem = self
            .right_telemetry
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .ok_or_else(|| CalibrateError::InvalidConfig("right IMU extraction failed".into()))?;

        let skip = self.config.skip_start_secs;
        let fps = self.left_info.fps;

        // Compute all telemetry-derived values into locals first so we can
        // release the telemetry borrows before mutating self.
        let sync_computed = telemetry::estimate_sync_offset(left_telem, right_telem).map(|off| {
            let frames = (-off * fps).round() as i64;
            log::info!("IMU sync offset: {off:.3}s = {frames} frames");
            frames
        });
        let rig_tilt_new = telemetry::rig_tilt(left_telem, skip);
        let left_roll = telemetry::gravity_vector(left_telem, skip).map(|g| g[2].atan2(g[0]));
        let right_roll = telemetry::gravity_vector(right_telem, skip).map(|g| g[2].atan2(g[0]));

        // Telemetry borrows released here; safe to mutate self.
        if let Some(frames) = sync_computed {
            self.sync_offset_frames = frames;
        }

        // Rig tilt. Cap at 25 deg - beyond that the IMU reading is likely
        // accelerometer noise rather than real forward lean.
        if let Some(tilt) = rig_tilt_new {
            log::info!("rig tilt: {:.1} deg", tilt.to_degrees());
            if tilt.abs() < 25.0_f64.to_radians() {
                self.rig_tilt = tilt;
            } else {
                log::warn!(
                    "rig tilt {:.1} deg exceeds 25 deg threshold, ignoring",
                    tilt.to_degrees()
                );
            }
        }

        // Rig roll: (left_roll - right_roll) / 2 cancels common IMU bias.
        if let (Some(lr), Some(rr)) = (left_roll, right_roll) {
            let avg = (lr - rr) / 2.0;
            log::info!(
                "rig roll: left={:.1} deg, right={:.1} deg, avg={:.1} deg",
                lr.to_degrees(),
                rr.to_degrees(),
                avg.to_degrees()
            );
            if avg.abs() < 25.0_f64.to_radians() {
                self.rig_roll = avg;
            } else {
                log::warn!(
                    "rig roll {:.1} deg exceeds 25 deg threshold, ignoring",
                    avg.to_degrees()
                );
            }
        }

        Ok(sync_computed)
    }

    /// Estimate sync offset from audio cross-correlation.
    ///
    /// The app extracts mono PCM samples from both videos and passes
    /// them here. Returns the sync offset in frames.
    pub fn audio_sync(
        &mut self,
        left_samples: &[i16],
        right_samples: &[i16],
        sample_rate: u32,
    ) -> Result<i64, CalibrateError> {
        let result = audio_sync::correlate(left_samples, right_samples, sample_rate, 30.0)
            .map_err(|e| CalibrateError::InvalidConfig(format!("audio sync failed: {e}")))?;

        let frames = result.offset_frames_rounded(self.left_info.fps);
        log::info!(
            "audio sync: {:.4}s = {} frames (confidence={:.0})",
            result.offset_secs,
            frames,
            result.confidence,
        );
        self.sync_offset_frames = frames;
        Ok(frames)
    }

    // ---------------------------------------------------------------
    // Step 3: Frame selection
    // ---------------------------------------------------------------

    /// Compute frame indices for both cameras with sync offset applied.
    ///
    /// Returns `(left_indices, right_indices)` that the app should use
    /// to extract frame pairs from its video decoder. Sampling and start/end skips apply to the shared interval after synchronization. Returns empty vectors when the clips do not overlap.
    pub fn frame_indices(&self) -> (Vec<u64>, Vec<u64>) {
        let offset = self.sync_offset_frames.unsigned_abs();
        let (left_start, right_start) = if self.sync_offset_frames < 0 {
            (offset, 0)
        } else {
            (0, offset)
        };
        let total = self
            .left_info
            .total_frames
            .saturating_sub(left_start)
            .min(self.right_info.total_frames.saturating_sub(right_start));
        let base_indices = sampling::select_frame_indices(
            total,
            self.left_info.fps,
            self.config.num_frames,
            self.config.skip_start_secs,
            self.config.skip_end_secs,
        );

        base_indices
            .into_iter()
            .map(|index| (index + left_start, index + right_start))
            .unzip()
    }

    // ---------------------------------------------------------------
    // Step 4: Calibration
    // ---------------------------------------------------------------

    /// Run the calibration pipeline on extracted frame pairs.
    ///
    /// The app must extract frames at the indices returned by
    /// [`CalibrationPipeline::frame_indices`] and pass them here as `(left, right)` pairs.
    pub fn calibrate(
        &self,
        interrupted: &std::sync::atomic::AtomicBool,
        frames: &[(Yuv420pFrame, Yuv420pFrame)],
    ) -> Result<CalibrationResult, CalibrateError> {
        self.config.validate()?;

        let mut result = crate::direct::calibrate(
            frames,
            [self.left_params.as_ref(), self.right_params.as_ref()],
            interrupted,
        )?;
        result.calibration.framing.tilt = self.rig_tilt;
        result.calibration.framing.roll = self.rig_roll;
        result.calibration.sync_offset = self.sync_offset_frames;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires RECO_TEST_FOOTAGE with GoPro HERO10 source videos"]
    fn manually_supplied_optics_do_not_disable_native_imu_sync() {
        let footage = std::env::var_os("RECO_TEST_FOOTAGE").expect("set RECO_TEST_FOOTAGE");
        let video = |side: &str| VideoInfo {
            path: std::path::Path::new(&footage)
                .join("gopro-hero10")
                .join(format!("{side}_4k.mp4")),
            width: 5312,
            height: 2988,
            fps: 30.0,
            total_frames: 7500,
        };
        let mut pipeline =
            CalibrationPipeline::new(video("left"), video("right"), CalibrationConfig::default());
        assert!(pipeline.left_info.path.is_file() && pipeline.right_info.path.is_file());
        let lens = Lens::fisheye(5312, 2988, 2500.0, 2500.0, 2656.0, 1494.0, [0.0; 4]);
        pipeline.set_profiles(Some(lens.clone()), Some(lens));
        pipeline.imu_sync().unwrap();
        assert_eq!(pipeline.has_native_gyro, Some(true));
        assert!(matches!(pipeline.left_telemetry, Some(Ok(_))));
        assert!(matches!(pipeline.right_telemetry, Some(Ok(_))));
    }

    #[test]
    fn explicit_optics_are_independent() {
        let info = VideoInfo {
            path: "unused.mp4".into(),
            width: 640,
            height: 480,
            fps: 30.0,
            total_frames: 120,
        };
        let mut pipeline =
            CalibrationPipeline::new(info.clone(), info, CalibrationConfig::default());
        let lens = Lens::flat(640, 480);
        for fixed in 0..2 {
            let mut lenses = [None, None];
            lenses[fixed] = Some(lens.clone());
            let [left, right] = lenses;
            pipeline.set_profiles(left, right);
            assert_eq!(pipeline.left_params.as_ref().is_some(), fixed == 0);
            assert_eq!(pipeline.right_params.as_ref().is_some(), fixed == 1);
            assert_eq!(
                pipeline
                    .left_params
                    .as_ref()
                    .or(pipeline.right_params.as_ref()),
                Some(&lens)
            );
        }
        pipeline.set_profiles(None, None);
        assert!(
            pipeline.left_params.as_ref().is_none() && pipeline.right_params.as_ref().is_none()
        );
    }
}
