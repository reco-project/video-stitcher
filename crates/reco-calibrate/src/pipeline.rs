//! High-level calibration pipeline orchestrator.
//!
//! [`CalibrationPipeline`] handles the full calibration workflow:
//! profile detection, sync estimation, frame selection, and optimization.
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
//! // Auto-detect lens profiles from embedded database + video metadata
//! pipeline.detect_profiles()?;
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
//! let result = pipeline.calibrate(&gpu, &frames)?;
//! ```

use std::path::Path;

use reco_core::calibration::CameraParams;
use reco_core::gpu::GpuContext;

use crate::error::CalibrateError;
use crate::telemetry::TelemetryData;
use crate::types::{CalibrationConfig, CalibrationResult, LensProfileInfo, YuvFrame};
use crate::{audio_sync, lens_database, sampling, telemetry};

/// Video metadata that the app provides from its decoder.
///
/// reco-calibrate never opens video files directly - the app is
/// responsible for decoding. This struct carries the metadata
/// needed for frame selection, sync, and profile detection.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// Path to the video file (used for metadata extraction only -
    /// telemetry, lens profiles - not for frame decoding).
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
    left_params: Option<CameraParams>,
    right_params: Option<CameraParams>,
    /// Lens profile metadata for the left camera (populated by detect_profiles).
    left_profile_info: Option<LensProfileInfo>,
    /// Lens profile metadata for the right camera.
    right_profile_info: Option<LensProfileInfo>,
    /// Cached parsed telemetry for the left video, lazily populated on
    /// first use (detect_profiles or imu_sync) so the expensive parse
    /// runs at most once per video. `Some(Ok(..))` after a successful
    /// parse, `Some(Err(()))` after a failed one (so we don't retry on
    /// every call), `None` before first use.
    left_telemetry: Option<Result<TelemetryData, ()>>,
    /// Cached parsed telemetry for the right video. See [`Self::left_telemetry`].
    right_telemetry: Option<Result<TelemetryData, ()>>,
    sync_offset_frames: i64,
    /// IMU seeds extracted during imu_sync
    imu_xrz_seed: Option<f64>,
    imu_xrx_seed: Option<f64>,
    imu_zrx_seed: Option<f64>,
    enable_x_rx: bool,
    /// Whether the camera has native gyroscope data (set by detect_profiles).
    /// When false, the full telemetry parse is skipped because derived
    /// quaternion data produces unreliable sync and rig tilt.
    has_native_gyro: bool,
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
            left_profile_info: None,
            right_profile_info: None,
            left_telemetry: None,
            right_telemetry: None,
            sync_offset_frames: 0,
            imu_xrz_seed: None,
            imu_xrx_seed: None,
            imu_zrx_seed: None,
            enable_x_rx: false,
            has_native_gyro: false,
            rig_tilt: 0.0,
            rig_roll: 0.0,
        }
    }

    // ---------------------------------------------------------------
    // Step 1: Lens profiles
    // ---------------------------------------------------------------

    /// Set lens profiles manually (loaded by the app from files, UI, etc.).
    pub fn set_profiles(&mut self, left: CameraParams, right: CameraParams) {
        self.left_params = Some(left);
        self.right_params = Some(right);
    }

    /// Auto-detect lens profiles from video metadata and the embedded database.
    ///
    /// Reads telemetry from the video files to identify camera model,
    /// then looks up the matching profile in the embedded Gyroflow database.
    /// Falls back to the left profile for the right camera if not found.
    ///
    /// The parsed telemetry is cached on the pipeline so
    /// [`imu_sync`](Self::imu_sync) reuses it instead of parsing the same
    /// files a second time. This matters on heavy sources (DJI Action 4,
    /// newer GoPro) where each parse is 30-60s.
    ///
    /// Returns the detected profiles for logging/display.
    pub fn detect_profiles(&mut self) -> Result<(CameraParams, CameraParams), CalibrateError> {
        let db = lens_database::LensDatabase::load_embedded();

        // Use the fast metadata-only path for lens detection. The full
        // telemetry parse (quaternion derivation, IMU normalization) is
        // deferred to ensure_*_telemetry which only runs when IMU data
        // is actually needed (sync offset, rig tilt). On DJI Action 4
        // this saves ~75s of I/O on cold HDD reads.
        let left_meta = telemetry::extract_metadata(&self.left_info.path).ok();
        let right_meta = telemetry::extract_metadata(&self.right_info.path).ok();

        self.has_native_gyro = left_meta.as_ref().is_some_and(|m| m.has_native_gyro);

        // Build a lightweight TelemetryData with just the metadata fields
        // that detect_profile needs (camera_type, camera_model, lens_profile, lens_info).
        let to_stub = |m: &telemetry::CameraMetadata| telemetry::TelemetryData {
            camera_type: m.camera_type.clone(),
            camera_model: m.camera_model.clone(),
            lens_profile: m.lens_profile.clone(),
            lens_info: m.lens_info.clone(),
            gyro: Vec::new(),
            accel: Vec::new(),
            quaternions: Vec::new(),
        };
        let left_tel_stub = left_meta.as_ref().map(to_stub);
        let right_tel_stub = right_meta.as_ref().map(to_stub);
        let left_tel = left_tel_stub.as_ref();
        let right_tel = right_tel_stub.as_ref();

        let (left_p, left_info) = lens_database::detect_profile(
            &self.left_info.path,
            self.left_info.width,
            self.left_info.height,
            &db,
            left_tel,
        )
        .ok_or_else(|| {
            CalibrateError::InvalidConfig(format!(
                "no lens profile matched the left camera at {}x{}: the video carries no \
                 usable camera telemetry and no database profile shares this resolution. \
                 Load a lens profile or record at a supported resolution.",
                self.left_info.width, self.left_info.height
            ))
        })?;

        let (right_p, right_info) = lens_database::detect_profile(
            &self.right_info.path,
            self.right_info.width,
            self.right_info.height,
            &db,
            right_tel,
        )
        .unwrap_or_else(|| {
            log::info!("right camera: no profile found, using left camera profile");
            (left_p.clone(), left_info.clone())
        });

        self.left_params = Some(left_p.clone());
        self.right_params = Some(right_p.clone());
        self.left_profile_info = Some(left_info);
        self.right_profile_info = Some(right_info);
        Ok((left_p, right_p))
    }

    /// Lazily populate the left-video telemetry cache. Called from
    /// `detect_profiles` and `imu_sync`; subsequent calls are no-ops.
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

    /// Load lens profiles from file paths.
    pub fn load_profiles(
        &mut self,
        left_path: &Path,
        right_path: Option<&Path>,
    ) -> Result<(CameraParams, CameraParams), CalibrateError> {
        let left_p = lens_database::load_from_file(left_path).map_err(|e| {
            CalibrateError::InvalidConfig(format!("failed to load left profile: {e}"))
        })?;
        let right_p = if let Some(rp) = right_path {
            lens_database::load_from_file(rp).map_err(|e| {
                CalibrateError::InvalidConfig(format!("failed to load right profile: {e}"))
            })?
        } else {
            left_p.clone()
        };

        self.left_params = Some(left_p.clone());
        self.right_params = Some(right_p.clone());
        Ok((left_p, right_p))
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

    /// Estimate sync offset and rotation seeds from IMU telemetry.
    ///
    /// Reads gyroscope data from both video files and cross-correlates
    /// to find the temporal offset. Also extracts differential orientation
    /// (roll, pitch, tilt) for optimizer seeding.
    ///
    /// Returns the sync offset in frames, or `None` if telemetry is
    /// unavailable or cross-correlation fails.
    pub fn imu_sync(&mut self) -> Result<Option<i64>, CalibrateError> {
        if !self.has_native_gyro {
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
        let diff_orientation = telemetry::differential_orientation(left_telem, right_telem, skip);
        let rig_tilt_new = telemetry::rig_tilt(left_telem, skip);
        let left_roll = telemetry::gravity_vector(left_telem, skip).map(|g| g[2].atan2(g[0]));
        let right_roll = telemetry::gravity_vector(right_telem, skip).map(|g| g[2].atan2(g[0]));

        // Telemetry borrows released here; safe to mutate self.
        if let Some(frames) = sync_computed {
            self.sync_offset_frames = frames;
        }

        if let Some((roll, pitch, tilt)) = diff_orientation {
            log::info!(
                "differential roll: {:.2} deg, pitch: {:.2} deg, tilt: {:.2} deg",
                roll.to_degrees(),
                pitch.to_degrees(),
                tilt.to_degrees(),
            );
            self.imu_xrz_seed = Some(roll);
            self.imu_zrx_seed = Some(tilt);
            if pitch.abs() > 2.0_f64.to_radians() {
                log::info!("pitch > 2 deg, enabling x_rx seeded at {pitch:.4} rad");
                self.enable_x_rx = true;
                self.imu_xrx_seed = Some(pitch);
            }
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
    /// to extract frame pairs from its video decoder.
    pub fn frame_indices(&self) -> (Vec<u64>, Vec<u64>) {
        let total = self
            .left_info
            .total_frames
            .min(self.right_info.total_frames);
        let base_indices = sampling::select_frame_indices(
            total,
            self.left_info.fps,
            self.config.num_frames,
            self.config.skip_start_secs,
            self.config.skip_end_secs,
        );

        let sync = self.sync_offset_frames;
        if sync >= 0 {
            let offset = sync as u64;
            let right = base_indices.iter().map(|&i| i + offset).collect();
            (base_indices, right)
        } else {
            let offset = (-sync) as u64;
            let left = base_indices.iter().map(|&i| i + offset).collect();
            (left, base_indices)
        }
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
        gpu: &GpuContext,
        frames: &[(YuvFrame, YuvFrame)],
    ) -> Result<CalibrationResult, CalibrateError> {
        let left_params = self.left_params.as_ref().ok_or_else(|| {
            CalibrateError::InvalidConfig(
                "lens profiles not set - call detect_profiles() or set_profiles() first".into(),
            )
        })?;
        let right_params = self.right_params.as_ref().ok_or_else(|| {
            CalibrateError::InvalidConfig(
                "lens profiles not set - call detect_profiles() or set_profiles() first".into(),
            )
        })?;

        // Merge IMU seeds into config only when the toggle is on.
        // When off, the optimizer uses its built-in start grid instead
        // of IMU-derived orientation. Sync offset, rig_tilt, and
        // rig_roll are always applied regardless of this flag.
        let mut config = self.config.clone();
        if config.use_imu_rotation_seeds {
            if self.imu_xrz_seed.is_some() {
                config.imu_xrz_seed = self.imu_xrz_seed;
            }
            if self.imu_xrx_seed.is_some() {
                config.imu_xrx_seed = self.imu_xrx_seed;
            }
            if self.imu_zrx_seed.is_some() {
                config.imu_zrx_seed = self.imu_zrx_seed;
            }
            if self.enable_x_rx {
                config.optimizer.enable_x_rx = true;
            }
        }

        let mut result = crate::calibrate(gpu, frames, left_params, right_params, &config)?;
        result.calibration.rig_tilt = self.rig_tilt;
        result.calibration.rig_roll = self.rig_roll;
        result.calibration.sync_offset = self.sync_offset_frames;
        result.left_lens_profile = self.left_profile_info.clone();
        result.right_lens_profile = self.right_profile_info.clone();
        Ok(result)
    }

    /// Get the left camera lens profile (if set).
    pub fn left_params(&self) -> Option<&CameraParams> {
        self.left_params.as_ref()
    }

    /// Get the right camera lens profile (if set).
    pub fn right_params(&self) -> Option<&CameraParams> {
        self.right_params.as_ref()
    }

    /// Get left video info.
    pub fn left_info(&self) -> &VideoInfo {
        &self.left_info
    }

    /// Get right video info.
    pub fn right_info(&self) -> &VideoInfo {
        &self.right_info
    }
}
