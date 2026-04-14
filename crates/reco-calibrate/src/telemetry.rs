//! IMU telemetry extraction for calibration.
//!
//! Uses the [`telemetry_parser`] crate (MIT, by AdrianEddy) to extract
//! IMU data from video files. Supports 30+ camera brands including
//! GoPro, DJI, Insta360, Sony, Canon, and Blackmagic.
//!
//! ## What's available per camera
//!
//! | Camera | Raw Gyro | Raw Accel | Quaternions | Embedded Lens |
//! |--------|----------|-----------|-------------|---------------|
//! | GoPro  | Yes      | Yes       | Yes (CORI)  | No            |
//! | DJI    | No (model-dependent) | No (model-dependent) | Yes | Yes |
//! | Insta360 | Yes    | Yes       | No          | Yes           |
//! | Sony   | Yes      | Yes       | No          | Yes (mesh)    |
//!
//! ## Usage for calibration
//!
//! ```ignore
//! use reco_calibrate::telemetry;
//!
//! let left = telemetry::extract("left.mp4")?;
//! let right = telemetry::extract("right.mp4")?;
//! let sync = telemetry::estimate_sync_offset(&left, &right);
//! let (roll, pitch) = telemetry::differential_orientation(&left, &right);
//! ```

use reco_core::calibration::CameraParams;
use std::path::Path;

/// Extracted telemetry data (gyro, accel, lens profile) from a video file.
#[derive(Debug, Clone)]
pub struct TelemetryData {
    /// Camera brand (e.g. "GoPro", "DJI").
    pub camera_type: String,
    /// Camera model if available.
    pub camera_model: Option<String>,
    /// Gyroscope samples (angular velocity in rad/s, timestamped).
    pub gyro: Vec<ImuSample>,
    /// Accelerometer samples (m/s^2, timestamped).
    pub accel: Vec<ImuSample>,
    /// Embedded lens profile if the camera provides one (DJI, Insta360).
    pub lens_profile: Option<CameraParams>,
    /// Orientation quaternions [w, x, y, z] (DJI, GoPro CORI).
    /// Represents rotation from camera frame to gravity-aligned world frame.
    pub quaternions: Vec<(f64, [f64; 4])>,
    /// FOV mode detected from GPMF metadata (e.g. "Wide", "Narrow", "Linear").
    ///
    /// Used to match the correct Gyroflow lens profile. GoPro cameras embed
    /// a VFOV tag in the GPMF stream indicating the FOV mode at recording time.
    pub lens_info: Option<String>,
    /// Pre-computed gravity vector from sensor fusion (GoPro GRAV tags).
    /// More accurate than raw accelerometer for tilt/roll because GoPro's
    /// internal fusion compensates for IMU chip misalignment.
    pub fused_gravity: Option<[f64; 3]>,
}

/// A single 3-axis IMU sample with timestamp.
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    /// Timestamp in seconds from video start.
    pub t: f64,
    /// X-axis value.
    pub x: f64,
    /// Y-axis value.
    pub y: f64,
    /// Z-axis value.
    pub z: f64,
}

impl ImuSample {
    /// Magnitude of the 3-axis vector.
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Extract telemetry data from a video file.
///
/// Auto-detects the camera type and extracts all available IMU data
/// using telemetry-parser's built-in normalization pipeline. This
/// handles scaling (SCAL tags), unit conversion (rad/s to deg/s, g to
/// m/s^2), orientation mapping (per-camera axis conventions), and
/// timestamp interpolation for all supported cameras.
///
/// Returns an error if the file format is unsupported or contains no
/// telemetry.
pub fn extract(path: &Path) -> Result<TelemetryData, TelemetryError> {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let filesize = std::fs::metadata(path)
        .map_err(|e| TelemetryError::Io(e.to_string()))?
        .len() as usize;

    let mut file = std::fs::File::open(path).map_err(|e| TelemetryError::Io(e.to_string()))?;

    let cancel = Arc::new(AtomicBool::new(false));
    let input = telemetry_parser::Input::from_stream(&mut file, filesize, path, |_| {}, cancel)
        .map_err(|e| TelemetryError::Parse(e.to_string()))?;

    let camera_type = input.camera_type();
    let camera_model = input.camera_model().cloned();

    // Get raw sensor data without orientation mapping. We pass Some("XYZ")
    // to override telemetry-parser's MTRX/ORIN mapping (same as Gyroflow).
    // The orientation is extracted separately and applied explicitly so we
    // know the exact axis convention after mapping.
    let raw_imu = telemetry_parser::util::normalized_imu_interpolated(&input, Some("XYZ".into()))
        .unwrap_or_default();

    // Extract the camera's IMU orientation string from metadata.
    // GoPro embeds this as ORIN/ORIO in the Gyroscope group; the
    // telemetry-parser normalizes it via MTRX. Result is a 3-char
    // string like "YxZ" (uppercase = positive, lowercase = negated).
    let imu_orientation = {
        use telemetry_parser::tags_impl::{GetWithType, GroupId, TagId};
        let mut io = String::from("XYZ"); // identity default
        if let Some(ref samples) = input.samples {
            for sample in samples {
                if let Some(ref tag_map) = sample.tag_map
                    && let Some(map) = tag_map.get(&GroupId::Gyroscope)
                {
                    if let Some(v) = map.get_t(TagId::Orientation) as Option<&String>
                        && v.len() == 3
                    {
                        io = v.clone();
                    }
                    io = input.normalize_imu_orientation(io);
                    break;
                }
            }
        }
        io
    };
    log::debug!("IMU orientation: {imu_orientation}");

    /// Apply a 3-character orientation mapping (e.g., "YxZ") to a 3D vector.
    /// Uppercase = positive axis, lowercase = negated.
    fn orient(v: [f64; 3], io: &str) -> [f64; 3] {
        let map = |c: u8| -> f64 {
            match c as char {
                'X' => v[0],
                'x' => -v[0],
                'Y' => v[1],
                'y' => -v[1],
                'Z' => v[2],
                'z' => -v[2],
                _ => 0.0,
            }
        };
        let b = io.as_bytes();
        [map(b[0]), map(b[1]), map(b[2])]
    }

    let mut gyro = Vec::with_capacity(raw_imu.len());
    let mut accel = Vec::with_capacity(raw_imu.len());

    for sample in &raw_imu {
        let t = sample.timestamp_ms / 1000.0; // ms to seconds
        if let Some(g) = sample.gyro {
            let [x, y, z] = orient(g, &imu_orientation);
            // telemetry-parser outputs deg/s; convert to rad/s
            gyro.push(ImuSample {
                t,
                x: x.to_radians(),
                y: y.to_radians(),
                z: z.to_radians(),
            });
        }
        if let Some(a) = sample.accl {
            let [x, y, z] = orient(a, &imu_orientation);
            accel.push(ImuSample { t, x, y, z });
        }
    }

    // Extract embedded lens profile, FOV mode, quaternions, and fused gravity
    let mut lens_profile = None;
    let mut lens_info: Option<String> = None;
    let mut quaternions: Vec<(f64, [f64; 4])> = Vec::new(); // (timestamp_s, [w, x, y, z])
    let mut fused_gravity: Option<[f64; 3]> = None;

    if let Some(ref samples) = input.samples {
        use telemetry_parser::tags_impl::{GetWithType, GroupId, TagId, TimeQuaternion};

        for sample in samples {
            if let Some(ref tag_map) = sample.tag_map {
                // Try Lens/Data tag first, then ClipMeta JSON fallback
                if lens_profile.is_none() {
                    lens_profile = extract_lens_from_tags(tag_map)
                        .or_else(|| extract_lens_from_clip_meta(tag_map));
                }

                // Extract GoPro FOV mode from GPMF tags (VFOV, ZFOV, PRJT)
                if lens_info.is_none()
                    && let Some(map) = tag_map.get(&GroupId::Default)
                {
                    // VFOV tag: primary FOV mode indicator
                    if let Some(v) = map.get_t(TagId::Unknown(0x56464f56)) as Option<&String> {
                        lens_info = Some(
                            match v.as_str() {
                                "X" => "Max",
                                "W" => "Wide",
                                "S" => "Super",
                                "H" => "Hyper",
                                "L" => "Linear",
                                "N" => "Narrow",
                                "M" => "Medium",
                                other => other,
                            }
                            .to_string(),
                        );
                    }

                    // ZFOV: actual FOV degrees - reclassify Linear < 80 as Narrow
                    if let Some(&v) = map.get_t(TagId::Unknown(0x5a464f56)) as Option<&f32>
                        && lens_info.as_deref() == Some("Linear")
                        && v < 80.0
                    {
                        lens_info = Some("Narrow".to_string());
                    }

                    // PRJT: projection override (GPMW = Max Wide)
                    if let Some(v) = map.get_t(TagId::Unknown(0x50524a54)) as Option<&String>
                        && v.as_str() == "GPMW"
                    {
                        lens_info = Some("Max Wide".to_string());
                    }
                }

                // Extract quaternions (DJI cameras provide fused orientation)
                if let Some(arr) = tag_map
                    .get(&GroupId::Quaternion)
                    .and_then(|map| map.get_t(TagId::Data) as Option<&Vec<TimeQuaternion<f64>>>)
                {
                    for v in arr {
                        quaternions.push((v.t, [v.v.w, v.v.x, v.v.y, v.v.z]));
                    }
                }

                // Extract GoPro fused gravity vectors (GRAV tags).
                // These are sensor-fusion-corrected and account for IMU
                // chip misalignment, unlike raw accelerometer data.
                if fused_gravity.is_none()
                    && let Some(map) = tag_map.get(&GroupId::GravityVector)
                {
                    use telemetry_parser::tags_impl::Vector3 as TpVec3;
                    let scale = *(map.get_t(TagId::Scale) as Option<&i16>).unwrap_or(&32767) as f64;
                    if scale > 0.0
                        && let Some(arr) = map.get_t(TagId::Data) as Option<&Vec<TpVec3<i16>>>
                    {
                        let n = arr.len().min(200);
                        let mut sx = 0.0f64;
                        let mut sy = 0.0f64;
                        let mut sz = 0.0f64;
                        let mut count = 0;
                        for v in &arr[..n] {
                            if v.x != 0 || v.y != 0 || v.z != 0 {
                                sx += v.x as f64 / scale;
                                sy += v.y as f64 / scale;
                                sz += v.z as f64 / scale;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            let inv = 1.0 / count as f64;
                            fused_gravity = Some([sx * inv, sy * inv, sz * inv]);
                        }
                    }
                }
            }
        }
    }

    // If no raw gyro but quaternions are available, derive angular velocity
    // by finite-differencing the quaternion signal.
    if gyro.is_empty() && quaternions.len() >= 2 {
        quaternions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        log::info!(
            "no raw gyro, deriving from {} quaternions",
            quaternions.len()
        );

        for i in 1..quaternions.len() {
            let dt = quaternions[i].0 - quaternions[i - 1].0;
            if dt <= 0.0 || dt > 1.0 {
                continue; // skip bad timestamps
            }

            let [w0, x0, y0, z0] = quaternions[i - 1].1;
            let [w1, x1, y1, z1] = quaternions[i].1;

            // q_delta = q1 * q0^-1 (conjugate for unit quaternion)
            let _dw = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0;
            let dx = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0;
            let dy = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0;
            let dz = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0;

            // Convert to angular velocity: omega = 2 * vec(q_delta) / dt
            // For small rotations, vec(q_delta) ≈ half the rotation vector
            let scale = 2.0 / dt;
            gyro.push(ImuSample {
                t: (quaternions[i - 1].0 + quaternions[i].0) / 2.0,
                x: dx * scale,
                y: dy * scale,
                z: dz * scale,
            });
        }
        log::info!("derived {} gyro samples from quaternions", gyro.len());
    }

    log::info!(
        "telemetry: {} {} - {} gyro, {} accel samples, {} quaternions, lens profile: {}, FOV: {}",
        camera_type,
        camera_model.as_deref().unwrap_or("unknown"),
        gyro.len(),
        accel.len(),
        quaternions.len(),
        if lens_profile.is_some() {
            "embedded"
        } else {
            "none"
        },
        lens_info.as_deref().unwrap_or("unknown")
    );

    if let Some(ref g) = fused_gravity {
        log::info!(
            "fused gravity (GRAV): [{:.4}, {:.4}, {:.4}]",
            g[0],
            g[1],
            g[2]
        );
    }

    Ok(TelemetryData {
        camera_type,
        camera_model,
        gyro,
        accel,
        quaternions,
        lens_profile,
        lens_info,
        fused_gravity,
    })
}

/// Estimate temporal sync offset between two cameras using gyro
/// cross-correlation.
///
/// Resamples both gyro signals to a common rate, correlates their
/// magnitude signals, and returns the lag in seconds. A positive
/// return value means the right camera's recording started that many
/// seconds after the left camera (equivalently, the left camera
/// started earlier).
///
/// Returns `None` if either camera lacks gyro data.
pub fn estimate_sync_offset(left: &TelemetryData, right: &TelemetryData) -> Option<f64> {
    if left.gyro.len() < 100 || right.gyro.len() < 100 {
        log::warn!("insufficient gyro samples for sync estimation");
        return None;
    }

    // Compute magnitude signal for each camera
    let left_mag: Vec<(f64, f64)> = left.gyro.iter().map(|s| (s.t, s.magnitude())).collect();
    let right_mag: Vec<(f64, f64)> = right.gyro.iter().map(|s| (s.t, s.magnitude())).collect();

    // Resample both to 200 Hz on a common time range
    let sample_rate = 200.0;
    let left_duration = left_mag.last()?.0 - left_mag.first()?.0;
    let right_duration = right_mag.last()?.0 - right_mag.first()?.0;
    let duration = left_duration.min(right_duration).min(30.0); // cap at 30s for speed

    let n = (duration * sample_rate) as usize;
    if n < 100 {
        return None;
    }

    let left_resampled = resample_signal(&left_mag, left_mag.first()?.0, sample_rate, n);
    let right_resampled = resample_signal(&right_mag, right_mag.first()?.0, sample_rate, n);

    // Normalize signals (subtract mean, divide by std dev) for Pearson correlation.
    // Without normalization, small-magnitude signals (e.g. derived from quaternion
    // differentiation) produce near-zero correlation regardless of match quality.
    let left_norm = normalize_signal(&left_resampled);
    let right_norm = normalize_signal(&right_resampled);

    if left_norm.is_empty() || right_norm.is_empty() {
        log::warn!("gyro sync: constant signal, cannot correlate");
        return None;
    }

    // Cross-correlate with up to +/-5 second search window
    let max_lag = (5.0 * sample_rate) as i64;
    let max_lag = max_lag.min(n as i64 / 2);

    let mut best_corr = f64::NEG_INFINITY;
    let mut best_lag: i64 = 0;

    for lag in -max_lag..=max_lag {
        let mut sum = 0.0;
        let mut count = 0;
        for (i, &left_val) in left_norm.iter().enumerate() {
            let j = i as i64 + lag;
            if j >= 0 && (j as usize) < right_norm.len() {
                sum += left_val * right_norm[j as usize];
                count += 1;
            }
        }
        if count > 0 {
            let corr = sum / count as f64;
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }

    // Pearson correlation: 1.0 = perfect match, 0.0 = no correlation.
    // Reject if the best match is weak (cameras too still, or bad data).
    if best_corr < 0.1 {
        log::warn!("gyro sync: correlation too low ({best_corr:.4}), rejecting offset");
        return None;
    }

    // Positive lag means right camera started earlier (negate to get seconds)
    let offset_secs = -(best_lag as f64 / sample_rate);
    log::info!("gyro sync: lag={best_lag} samples ({offset_secs:.3}s), correlation={best_corr:.4}");
    Some(offset_secs)
}

/// Compute the average gravity vector from accelerometer data.
///
/// Uses the mean of all accelerometer samples, assuming the camera is
/// mostly stationary during recording. The gravity vector points
/// "down" in the camera's coordinate frame.
///
/// Returns `None` if no accelerometer data is available.
pub fn gravity_vector(data: &TelemetryData) -> Option<[f64; 3]> {
    if data.accel.is_empty() {
        return None;
    }

    // Only average the first ~1 second of samples. The camera should be
    // stationary at the start of recording. Averaging the entire stream
    // includes dynamic acceleration from camera sway during the match,
    // which corrupts the roll estimate (106k samples over 9 minutes of
    // GoPro footage gave 12 degrees of false roll).
    let max_samples = 200; // ~1s at 200Hz (GoPro), ~2s at 100Hz (DJI)
    let samples = &data.accel[..data.accel.len().min(max_samples)];
    let n = samples.len() as f64;
    let mut gx = 0.0;
    let mut gy = 0.0;
    let mut gz = 0.0;
    for s in samples {
        gx += s.x;
        gy += s.y;
        gz += s.z;
    }

    let result = [gx / n, gy / n, gz / n];
    log::debug!(
        "gravity vector for {}: [{:.3}, {:.3}, {:.3}] (mag={:.3})",
        data.camera_type,
        result[0],
        result[1],
        result[2],
        (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt()
    );
    Some(result)
}

/// Differential orientation between two cameras from gravity vectors.
///
/// Returns `(roll_diff, pitch_diff, tilt_diff)` in radians:
/// - `roll_diff`: differential roll (seeds x_rz)
/// - `pitch_diff`: differential pitch (seeds x_rx when > 2 deg)
/// - `tilt_diff`: differential tilt with rig tilt removed (seeds z_rx).
///   Computed by subtracting the average tilt (rig tilt) from each
///   camera's individual tilt, then taking the left camera's residual.
///
/// Returns `None` if either camera lacks accelerometer data.
pub fn differential_orientation(
    left: &TelemetryData,
    right: &TelemetryData,
) -> Option<(f64, f64, f64)> {
    let lg = gravity_vector(left)?;
    let rg = gravity_vector(right)?;

    // Normalized IMU convention (after telemetry-parser's MTRX mapping):
    //   X = down (gravity), Y = forward (optical axis), Z = right
    // This holds for GoPro (HERO5+) after ORIN/ORIO matrix application.
    //
    // Roll = rotation around Y (optical axis), measured as gravity's
    //   lateral component: atan2(gz, gx). Zero when camera is upright.
    //   Seeds x_rz (right camera roll relative to left).
    let left_roll = lg[2].atan2((lg[0] * lg[0] + lg[1] * lg[1]).sqrt());
    let right_roll = rg[2].atan2((rg[0] * rg[0] + rg[1] * rg[1]).sqrt());
    let roll_diff = right_roll - left_roll;

    // Pitch = rotation around Z (lateral axis), measured as gravity's
    //   forward component: atan2(gy, gx). Zero when camera faces level.
    //   Seeds x_rx (right camera pitch relative to left).
    //   Note: stereo rigs have cameras facing opposite directions, so
    //   the sign of gy is flipped between left and right cameras.
    let left_pitch = lg[1].atan2((lg[0] * lg[0] + lg[2] * lg[2]).sqrt());
    let right_pitch = rg[1].atan2((rg[0] * rg[0] + rg[2] * rg[2]).sqrt());
    let pitch_diff = right_pitch - left_pitch;

    // Tilt: each camera's roll from vertical (lateral lean).
    // atan2(gz, gx) gives the roll from the gravity axis.
    // z_rx captures the left camera's deviation from the rig average.
    let left_tilt = lg[2].atan2((lg[0] * lg[0] + lg[1] * lg[1]).sqrt());
    let right_tilt = rg[2].atan2((rg[0] * rg[0] + rg[1] * rg[1]).sqrt());
    let rig_tilt_avg = (left_tilt + right_tilt) / 2.0;
    let tilt_diff = left_tilt - rig_tilt_avg;

    log::info!(
        "differential orientation: roll={roll_diff:.4} rad ({:.1} deg), \
         pitch={pitch_diff:.4} rad ({:.1} deg), \
         tilt_diff={tilt_diff:.4} rad ({:.1} deg)",
        roll_diff.to_degrees(),
        pitch_diff.to_degrees(),
        tilt_diff.to_degrees(),
    );

    Some((roll_diff, pitch_diff, tilt_diff))
}

/// Compute the rig tilt angle from the average gravity vector.
///
/// The rig tilt is the angle between the gravity vector and the
/// camera's "down" axis, measured in the plane perpendicular to the
/// camera's optical axis. This is used to tilt the virtual camera's
/// reference frame in the renderer.
///
/// Rig tilt and roll extracted from IMU data.
#[derive(Debug, Clone, Copy)]
pub struct RigOrientation {
    /// Forward lean from vertical in radians (rotation around right axis).
    pub tilt: f64,
    /// Lateral lean in radians (rotation around forward axis).
    pub roll: f64,
}

/// Extract rig tilt and roll from IMU data.
///
/// Tries accelerometer data first (direct gravity measurement), then
/// falls back to quaternions (DJI cameras provide orientation but
/// no raw accelerometer). Returns `None` if neither is available.
pub fn rig_orientation(data: &TelemetryData) -> Option<RigOrientation> {
    // Priority 1: fused gravity vectors (GoPro GRAV tags).
    // Sensor-fusion-corrected with known axis convention.
    // GRAV convention: Y=down (gravity), X=forward (optical axis), Z=right.
    // Validated: tilt from GRAV[0]/GRAV[1] matches accelerometer tilt (17.3°≈17.4°).
    if let Some(g) = data.fused_gravity {
        // tilt = forward component / perpendicular plane
        let tilt = g[0].atan2((g[1] * g[1] + g[2] * g[2]).sqrt());
        // roll = right component / perpendicular plane
        let roll = g[2].atan2((g[0] * g[0] + g[1] * g[1]).sqrt());
        log::info!(
            "rig orientation (fused GRAV): tilt={tilt:.4} rad ({:.1} deg), roll={roll:.4} rad ({:.1} deg)",
            tilt.to_degrees(),
            roll.to_degrees()
        );
        return Some(RigOrientation { tilt, roll });
    }

    // Priority 2: raw accelerometer (direct gravity measurement).
    if let Some(g) = gravity_vector(data) {
        // In the normalized IMU frame: X=down, Y=forward (optical axis), Z=right.
        // Decompose gravity into tilt (forward lean) and roll (lateral lean).
        // Tilt: angle between gravity projection on XY plane and the X axis.
        // Roll: angle between gravity projection on XZ plane and the X axis.
        // Using sqrt(other^2) as the denominator isolates each angle from
        // the other, unlike atan2(gz, gx) which underestimates the denominator
        // when tilt is large (gx is reduced by both tilt and roll).
        let tilt = g[1].atan2((g[0] * g[0] + g[2] * g[2]).sqrt());
        let roll = g[2].atan2((g[0] * g[0] + g[1] * g[1]).sqrt());
        log::info!(
            "rig orientation (accel): tilt={tilt:.4} rad ({:.1} deg), roll={roll:.4} rad ({:.1} deg)",
            tilt.to_degrees(),
            roll.to_degrees()
        );
        return Some(RigOrientation { tilt, roll });
    }

    // Priority 3: quaternion fallback (DJI cameras without accel/GRAV)
    if let Some(ori) = rig_orientation_from_quaternions(data) {
        log::info!(
            "rig orientation (quaternion): tilt={:.4} rad ({:.1} deg), roll={:.4} rad ({:.1} deg)",
            ori.tilt,
            ori.tilt.to_degrees(),
            ori.roll,
            ori.roll.to_degrees()
        );
        return Some(ori);
    }

    None
}

/// Backward-compatible wrapper that returns only the tilt angle.
pub fn rig_tilt(data: &TelemetryData) -> Option<f64> {
    rig_orientation(data).map(|o| o.tilt)
}

/// Compute rig orientation from the average orientation quaternion.
///
/// Rotates the world gravity vector [0, -1, 0] into camera space using
/// the average quaternion, then computes tilt and roll from the gravity
/// components.
fn rig_orientation_from_quaternions(data: &TelemetryData) -> Option<RigOrientation> {
    if data.quaternions.len() < 10 {
        return None;
    }

    // Average the first 100 quaternions (camera should be stationary at start)
    let n = data.quaternions.len().min(100);
    let mut aw = 0.0;
    let mut ax = 0.0;
    let mut ay = 0.0;
    let mut az = 0.0;
    for &(_, [w, x, y, z]) in &data.quaternions[..n] {
        // Flip sign if dot product with first quat is negative (hemisphere consistency)
        let [w0, x0, y0, z0] = data.quaternions[0].1;
        let dot = w * w0 + x * x0 + y * y0 + z * z0;
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        aw += w * sign;
        ax += x * sign;
        ay += y * sign;
        az += z * sign;
    }
    let inv_n = 1.0 / n as f64;
    aw *= inv_n;
    ax *= inv_n;
    ay *= inv_n;
    az *= inv_n;

    // Normalize
    let len = (aw * aw + ax * ax + ay * ay + az * az).sqrt();
    if len < 1e-10 {
        return None;
    }
    let (w, x, y, z) = (aw / len, ax / len, ay / len, az / len);
    log::debug!("quaternion-based tilt: avg quat=[{w:.4}, {x:.4}, {y:.4}, {z:.4}] (n={n})");

    // Rotate world gravity into camera frame: g_cam = q^-1 * g_world * q
    // For unit quaternion q^-1 = conjugate [w, -x, -y, -z].
    //
    // Try both Z-down [0,0,-1] (ENU, typical for GoPro CORI) and Y-down
    // [0,-1,0] conventions. Pick whichever produces a gravity vector whose
    // largest component aligns with the accelerometer data (if available).
    // Z-down: g_cam = q* [0,0,-1] q
    let gx_z = 2.0 * (x * z + w * y);
    let gy_z = 2.0 * (y * z - w * x);
    let gz_z = 1.0 - 2.0 * (x * x + y * y);
    // Negate because gravity = -Z in Z-up frame
    let (gx_z, gy_z, gz_z) = (-gx_z, -gy_z, -gz_z);

    // Y-down: g_cam = q* [0,-1,0] q
    let gx_y = -2.0 * (x * y - w * z);
    let gy_y = -(1.0 - 2.0 * (x * x + z * z));
    let gz_y = -2.0 * (y * z + w * x);

    // Heuristic: pick the convention where the result looks more like gravity
    // (largest component should dominate, matching accelerometer behavior).
    // For a camera tilted ~20 degrees, the "down" component should be > 0.9.
    let max_z = gx_z.abs().max(gy_z.abs()).max(gz_z.abs());
    let max_y = gx_y.abs().max(gy_y.abs()).max(gz_y.abs());
    let (gx, gy, gz) = if max_z > max_y {
        log::debug!("quaternion gravity convention: Z-down (ENU)");
        (gx_z, gy_z, gz_z)
    } else {
        log::debug!("quaternion gravity convention: Y-down");
        (gx_y, gy_y, gz_y)
    };
    log::debug!("gravity in camera frame: [{gx:.4}, {gy:.4}, {gz:.4}]");

    let tilt = gy.atan2((gx * gx + gz * gz).sqrt());
    let roll = gz.atan2((gx * gx + gy * gy).sqrt());
    Some(RigOrientation { tilt, roll })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract lens profile from ClipMeta JSON (DJI Action 4 fallback).
///
/// When the parser doesn't insert a Lens/Data tag, we read focal_length
/// and distortion_coefficients directly from the ClipMeta JSON stored
/// in the Default group.
fn extract_lens_from_clip_meta(
    tag_map: &telemetry_parser::tags_impl::GroupedTagMap,
) -> Option<CameraParams> {
    use telemetry_parser::tags_impl::*;
    // The ClipMeta JSON is stored in GroupId::Default / TagId::Metadata
    let default_map = tag_map.get(&GroupId::Default)?;
    let tag = default_map.get(&TagId::Metadata)?;
    let json = if let TagValue::Json(ref json_val) = tag.value {
        json_val.get()
    } else {
        return None;
    };

    let focal = json
        .get("digital_focal_length")
        .and_then(|v| v.get("focal_length"))
        .and_then(|v| v.as_f64())?;

    let coeffs = json
        .get("distortion_coefficients")
        .and_then(|v| v.get("coeffients")) // note: typo in DJI proto
        .and_then(|v| v.as_array())?;

    if coeffs.len() < 4 {
        return None;
    }

    // Need resolution - not in ClipMeta, use a reasonable default for DJI Action 4
    // The caller should override width/height from the video decoder
    let width = 3840;
    let height = 2880;
    let half_w = width as f64 / 2.0;
    let half_h = height as f64 / 2.0;

    log::info!(
        "embedded lens (from ClipMeta): focal={focal:.2}, d=[{:.4}, {:.4}, {:.4}, {:.4}]",
        coeffs[0].as_f64().unwrap_or(0.0),
        coeffs[1].as_f64().unwrap_or(0.0),
        coeffs[2].as_f64().unwrap_or(0.0),
        coeffs[3].as_f64().unwrap_or(0.0),
    );

    Some(CameraParams {
        width,
        height,
        fx: focal,
        fy: focal,
        cx: half_w,
        cy: half_h,
        d: [
            coeffs[0].as_f64().unwrap_or(0.0),
            coeffs[1].as_f64().unwrap_or(0.0),
            coeffs[2].as_f64().unwrap_or(0.0),
            coeffs[3].as_f64().unwrap_or(0.0),
        ],
    })
}

/// Extract an embedded lens profile from a tag map (DJI, Insta360).
fn extract_lens_from_tags(
    tag_map: &telemetry_parser::tags_impl::GroupedTagMap,
) -> Option<CameraParams> {
    use telemetry_parser::tags_impl::*;
    let lens_map = tag_map.get(&GroupId::Lens)?;
    let tag = lens_map.get(&TagId::Data)?;
    if let TagValue::Json(ref json_val) = tag.value {
        parse_embedded_lens_profile(json_val.get())
    } else {
        None
    }
}

/// Parse an embedded lens profile JSON from telemetry-parser.
///
/// DJI cameras embed focal_length + distortion_coeffs as JSON in the
/// `Lens/Data` tag. This function converts to our `CameraParams` format.
fn parse_embedded_lens_profile(json: &serde_json::Value) -> Option<CameraParams> {
    let cm = json.get("camera_matrix")?;
    let fx = cm.get("fx")?.as_f64()?;
    let fy = cm.get("fy")?.as_f64()?;
    let cx = cm.get("cx")?.as_f64()?;
    let cy = cm.get("cy")?.as_f64()?;

    let dc = json.get("distortion_coeffs")?.as_array()?;
    if dc.len() < 4 {
        return None;
    }
    let d = [
        dc[0].as_f64().unwrap_or(0.0),
        dc[1].as_f64().unwrap_or(0.0),
        dc[2].as_f64().unwrap_or(0.0),
        dc[3].as_f64().unwrap_or(0.0),
    ];

    // Resolution from the lens profile
    let res = json
        .get("resolution")
        .or_else(|| json.get("calib_dimension"))?;
    let width = res.get("width").or_else(|| res.get("w"))?.as_u64()? as u32;
    let height = res.get("height").or_else(|| res.get("h"))?.as_u64()? as u32;

    log::info!("embedded lens profile: {width}x{height}, fx={fx:.2}, fy={fy:.2}");

    Some(CameraParams {
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        d,
    })
}

/// Linearly resample a timestamped signal to a uniform sample rate.
/// Normalize a signal to zero mean, unit variance (for Pearson correlation).
/// Returns empty vec if the signal has zero variance (constant).
fn normalize_signal(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    let var = signal.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    if std < 1e-15 {
        return Vec::new(); // constant signal
    }
    signal.iter().map(|&x| (x - mean) / std).collect()
}

fn resample_signal(signal: &[(f64, f64)], t_start: f64, rate: f64, n: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(n);
    let mut src_idx = 0;

    for i in 0..n {
        let t = t_start + i as f64 / rate;

        // Advance source index to bracket the target time
        while src_idx + 1 < signal.len() && signal[src_idx + 1].0 < t {
            src_idx += 1;
        }

        if src_idx + 1 >= signal.len() {
            result.push(signal.last().map_or(0.0, |s| s.1));
            continue;
        }

        // Linear interpolation
        let (t0, v0) = signal[src_idx];
        let (t1, v1) = signal[src_idx + 1];
        let dt = t1 - t0;
        if dt > 0.0 {
            let frac = (t - t0) / dt;
            result.push(v0 + frac * (v1 - v0));
        } else {
            result.push(v0);
        }
    }

    result
}

/// Errors from telemetry extraction.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// File I/O error.
    #[error("telemetry I/O error: {0}")]
    Io(String),
    /// Unsupported or unparseable telemetry format.
    #[error("telemetry parse error: {0}")]
    Parse(String),
}
