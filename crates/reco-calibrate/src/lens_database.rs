//! Lens profile database for automatic camera detection.
//!
//! Loads Gyroflow-format lens profiles from the bundled CBOR database
//! (4200+ profiles across 50+ camera brands) and matches them against
//! camera metadata extracted from video files.
//!
//! ## Data source
//!
//! The primary database is the [Gyroflow lens_profiles](https://github.com/gyroflow/lens_profiles)
//! bundle (CC0-1.0, public domain), converted to `profiles.cbor.gz` and
//! embedded at compile time. The JSON format is defined by the Gyroflow
//! project. Additional profiles can be loaded from a directory at runtime.
//!
//! ## Matching strategy
//!
//! 1. Exact: brand + model + resolution
//! 2. Aspect ratio: brand + model + same aspect ratio, closest resolution
//! 3. Any: brand + model, any resolution (with scaling)

use reco_core::calibration::CameraParams;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Embedded Gyroflow lens profile database (profiles.cbor.gz).
static PROFILES_CBOR_GZ: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../resources/profiles.cbor.gz"
));

/// A loaded lens profile entry.
#[derive(Debug, Clone)]
struct ProfileEntry {
    /// Source identifier for diagnostics.
    source: String,
    /// Camera brand (lowercase).
    brand: String,
    /// Camera model (lowercase).
    model: String,
    /// FOV mode from the profile (e.g. "Wide", "Narrow", "Linear").
    lens_model: String,
    /// Non-empty for third-party lens attachments (e.g. "neewer anamorphic wide").
    camera_setting: String,
    /// Calibration width.
    width: u32,
    /// Calibration height.
    height: u32,
    /// Parsed camera parameters.
    params: CameraParams,
}

/// Lens profile database.
///
/// Loads profiles from the embedded CBOR bundle and optional directory
/// overrides, then provides lookup by camera brand, model, and resolution.
pub struct LensDatabase {
    profiles: Vec<ProfileEntry>,
    /// Index: normalized "brand/model" -> list of profile indices.
    by_camera: HashMap<String, Vec<usize>>,
}

impl LensDatabase {
    /// Load the embedded Gyroflow profile database.
    ///
    /// Decompresses and parses the bundled `profiles.cbor.gz` (1.5MB
    /// compressed, 4200+ profiles). Takes ~50ms on first call.
    pub fn load_embedded() -> Self {
        let mut db = Self {
            profiles: Vec::new(),
            by_camera: HashMap::new(),
        };

        // Decompress gzip
        let mut decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(PROFILES_CBOR_GZ));
        let mut decompressed = Vec::new();
        if decoder.read_to_end(&mut decompressed).is_err() {
            log::error!("failed to decompress embedded lens profiles");
            return db;
        }

        // Parse CBOR: Vec<(filename, serde_json::Value)>
        let entries: Vec<(String, serde_json::Value)> =
            match ciborium::from_reader(std::io::Cursor::new(decompressed)) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("failed to parse embedded lens profiles: {e}");
                    return db;
                }
            };

        for (filename, value) in entries {
            if filename.starts_with("__") {
                continue; // skip metadata entries
            }
            if let Some(entry) = parse_profile_value(&value, &filename) {
                let key = normalize_camera_key(&entry.brand, &entry.model);
                let idx = db.profiles.len();
                db.by_camera.entry(key).or_default().push(idx);
                db.profiles.push(entry);
            }
        }

        log::info!(
            "lens database: {} profiles from {} cameras (embedded)",
            db.profiles.len(),
            db.by_camera.len()
        );
        db
    }

    /// Load additional profiles from a directory (custom/override profiles).
    ///
    /// Profiles loaded this way take priority over embedded ones for
    /// the same camera/resolution combination.
    pub fn load_directory(&mut self, dir: &Path) -> Result<usize, std::io::Error> {
        let before = self.profiles.len();
        load_dir_recursive(dir, &mut self.profiles, &mut self.by_camera)?;
        let added = self.profiles.len() - before;
        if added > 0 {
            log::info!(
                "lens database: loaded {added} additional profiles from {}",
                dir.display()
            );
        }
        Ok(added)
    }

    /// Find the best matching profile for a camera.
    ///
    /// Returns `None` if no profile matches the brand/model.
    pub fn find(
        &self,
        brand: &str,
        model: &str,
        width: u32,
        height: u32,
        lens_info: Option<&str>,
    ) -> Option<CameraParams> {
        let key = normalize_camera_key(brand, model);
        // Try exact model first, then strip variant suffixes to find the
        // parent model (e.g. "HERO11 Black Mini" -> "HERO11 Black").
        // Gyroflow's profiles often cover the base model but not every variant.
        let indices = if let Some(idx) = self.by_camera.get(&key) {
            idx
        } else {
            // Try parent model by stripping common suffixes
            let parent = strip_model_variant(&key);
            if let Some(idx) = parent.as_ref().and_then(|p| self.by_camera.get(p)) {
                log::info!(
                    "lens auto-detect: no profiles for {key}, using parent {}",
                    parent.unwrap()
                );
                idx
            } else {
                return None;
            }
        };

        // Filter candidates by FOV mode when available.
        // If lens_info is provided (e.g. "Wide"), only consider profiles
        // whose lens_model matches (case-insensitive). If no FOV-filtered
        // profiles match, fall back to all candidates.
        let fov_filtered: Vec<usize> = if let Some(info) = lens_info {
            let info_lower = info.to_ascii_lowercase();
            indices
                .iter()
                .copied()
                .filter(|&idx| {
                    let p = &self.profiles[idx];
                    // Match FOV mode exactly, exclude third-party lens attachments
                    p.lens_model.to_ascii_lowercase() == info_lower && p.camera_setting.is_empty()
                })
                .collect()
        } else {
            Vec::new()
        };

        // If no FOV match on the exact model, try the parent model's
        // profiles for the same FOV before falling back to any profile.
        // E.g. "HERO11 Black Mini" has no Wide profile, but "HERO11 Black" does.
        let parent_fov_filtered: Vec<usize> = if fov_filtered.is_empty() && lens_info.is_some() {
            strip_model_variant(&key)
                .and_then(|parent| self.by_camera.get(&parent))
                .map(|parent_indices| {
                    let info_lower = lens_info.unwrap().to_ascii_lowercase();
                    let matches: Vec<usize> = parent_indices
                        .iter()
                        .copied()
                        .filter(|&idx| {
                            let p = &self.profiles[idx];
                            p.lens_model.to_ascii_lowercase() == info_lower
                                && p.camera_setting.is_empty()
                        })
                        .collect();
                    if !matches.is_empty() {
                        log::info!(
                            "lens auto-detect: using parent model for FOV '{}'",
                            lens_info.unwrap()
                        );
                    }
                    matches
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let candidates = if !fov_filtered.is_empty() {
            fov_filtered.as_slice()
        } else if !parent_fov_filtered.is_empty() {
            parent_fov_filtered.as_slice()
        } else {
            if lens_info.is_some() {
                log::debug!(
                    "lens auto-detect: no FOV '{}' match for {key} or parent, using all",
                    lens_info.unwrap_or("?")
                );
            }
            indices.as_slice()
        };

        // 1. Exact resolution match
        for &idx in candidates {
            let p = &self.profiles[idx];
            if p.width == width && p.height == height {
                log::info!(
                    "lens auto-detect: exact match {key} {width}x{height} FOV={} ({})",
                    p.lens_model,
                    p.source
                );
                return Some(p.params.clone());
            }
        }

        // 2. Same aspect ratio, closest resolution - scale intrinsics
        let target_aspect = width as f64 / height as f64;
        let mut best: Option<(usize, f64)> = None;
        for &idx in candidates {
            let p = &self.profiles[idx];
            let aspect = p.width as f64 / p.height as f64;
            if (aspect - target_aspect).abs() < 0.05 {
                let scale_diff = (p.width as f64 - width as f64).abs();
                if best.is_none() || scale_diff < best.unwrap().1 {
                    best = Some((idx, scale_diff));
                }
            }
        }

        if let Some((idx, _)) = best {
            let p = &self.profiles[idx];
            let scale = width as f64 / p.width as f64;
            log::info!(
                "lens auto-detect: scaling {key} {}x{} -> {width}x{height} FOV={} (scale={scale:.3}, {})",
                p.width,
                p.height,
                p.lens_model,
                p.source
            );
            return Some(CameraParams {
                width,
                height,
                fx: p.params.fx * scale,
                fy: p.params.fy * scale,
                cx: p.params.cx * scale,
                cy: p.params.cy * scale,
                d: p.params.d, // distortion coeffs are scale-invariant
            });
        }

        log::warn!(
            "lens auto-detect: no match for {key} {width}x{height} FOV={}",
            lens_info.unwrap_or("?")
        );
        None
    }

    /// Find a profile using telemetry-parser camera identification.
    ///
    /// Convenience wrapper that uses the camera_type and camera_model
    /// strings from telemetry extraction.
    pub fn find_from_telemetry(
        &self,
        camera_type: &str,
        camera_model: Option<&str>,
        width: u32,
        height: u32,
        lens_info: Option<&str>,
    ) -> Option<CameraParams> {
        let model = camera_model.unwrap_or(camera_type);
        self.find(camera_type, model, width, height, lens_info)
    }

    /// Find a profile by resolution only (no camera identification).
    ///
    /// Searches all profiles for an exact resolution match. If multiple
    /// cameras share the same resolution, returns the first match.
    /// This is the last-resort fallback when telemetry extraction fails.
    pub fn find_by_resolution(&self, width: u32, height: u32) -> Option<CameraParams> {
        for p in &self.profiles {
            if p.width == width && p.height == height {
                log::info!(
                    "lens auto-detect: resolution-only match {}x{} from {} ({})",
                    width,
                    height,
                    p.brand,
                    p.source
                );
                return Some(p.params.clone());
            }
        }
        None
    }

    /// Number of loaded profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

/// Load a camera profile from a JSON file.
///
/// Accepts multiple formats:
/// - v1 uniforms: `{fx, fy, cx, cy, d: [k1..k4], width, height}`
/// - Gyroflow: `{camera_matrix: {fx,fy,cx,cy}, distortion_coeffs: [...], resolution: {w,h}}`
/// - Gyroflow with fisheye_params wrapper
///
/// This is the standard way to load a manually-specified lens profile.
pub fn load_from_file(path: &Path) -> Result<CameraParams, LensLoadError> {
    let json_str = std::fs::read_to_string(path).map_err(|e| LensLoadError::Io(e.to_string()))?;
    load_from_json(&json_str, path.display().to_string().as_str())
}

/// Load a camera profile from a JSON string.
pub fn load_from_json(json_str: &str, source: &str) -> Result<CameraParams, LensLoadError> {
    let v: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| LensLoadError::Parse(e.to_string()))?;

    // v1 uniforms format (flat with fx/fy/cx/cy/d)
    if v.get("fx").is_some() && v.get("d").is_some() {
        return serde_json::from_str::<CameraParams>(json_str)
            .map_err(|e| LensLoadError::Parse(e.to_string()));
    }

    // Gyroflow/reco profile format
    if let Some(entry) = parse_profile_value(&v, source) {
        return Ok(entry.params);
    }

    Err(LensLoadError::UnrecognizedFormat(source.to_string()))
}

/// Auto-detect the camera profile from a video file.
///
/// Tries in order:
/// 1. Embedded lens profile from video metadata (DJI cameras)
/// 2. Camera identification + database lookup (GoPro, etc.)
///
/// Returns `None` if the camera can't be identified or no profile matches.
pub fn detect_profile(
    video_path: &Path,
    video_width: u32,
    video_height: u32,
    db: &LensDatabase,
) -> Option<CameraParams> {
    let tel = match crate::telemetry::extract(video_path) {
        Ok(t) => Some(t),
        Err(e) => {
            log::warn!("lens auto-detect: telemetry extraction failed: {e}");
            None
        }
    };

    if let Some(ref tel) = tel {
        // 1. Embedded lens profile (DJI cameras embed in metadata)
        if let Some(ref profile) = tel.lens_profile {
            log::info!(
                "lens auto-detect: embedded profile from {} {}",
                tel.camera_type,
                tel.camera_model.as_deref().unwrap_or("?")
            );
            return Some(profile.clone());
        }

        // 2. Database lookup by camera identification + FOV mode
        if let Some(params) = db.find_from_telemetry(
            &tel.camera_type,
            tel.camera_model.as_deref(),
            video_width,
            video_height,
            tel.lens_info.as_deref(),
        ) {
            return Some(params);
        }
    }

    // 3. Fallback: resolution-only database search (no telemetry available)
    log::info!(
        "lens auto-detect: trying resolution-only lookup for {}x{}",
        video_width,
        video_height
    );
    db.find_by_resolution(video_width, video_height)
}

/// Errors from lens profile loading.
#[derive(Debug, thiserror::Error)]
/// Errors from lens profile loading.
pub enum LensLoadError {
    /// File I/O error.
    #[error("cannot read lens profile: {0}")]
    Io(String),
    /// JSON parsing error.
    #[error("cannot parse lens profile: {0}")]
    Parse(String),
    /// Unrecognized format.
    #[error(
        "unrecognized lens profile format in '{0}'. Expected v1 uniforms (fx/fy/cx/cy/d) or Gyroflow (camera_matrix/distortion_coeffs/resolution)."
    )]
    UnrecognizedFormat(String),
}

/// Normalize brand/model for lookup key.
fn normalize_camera_key(brand: &str, model: &str) -> String {
    let b = brand.to_lowercase().replace(' ', "-");
    let m = model.to_lowercase().replace(' ', "-").replace("--", "-");
    format!("{b}/{m}")
}

/// Strip variant suffixes to find the parent model key.
/// E.g. "gopro/hero11-black-mini" -> "gopro/hero11-black".
fn strip_model_variant(key: &str) -> Option<String> {
    // Common GoPro/DJI variant suffixes
    for suffix in &["-mini", "-max", "-session", "-bones", "-creator-edition"] {
        if let Some(parent) = key.strip_suffix(suffix) {
            return Some(parent.to_string());
        }
    }
    None
}

/// Parse a profile from a serde_json::Value (CBOR or JSON source).
fn parse_profile_value(v: &serde_json::Value, source: &str) -> Option<ProfileEntry> {
    let brand = v.get("camera_brand")?.as_str()?.to_string();
    let model = v.get("camera_model")?.as_str()?.to_string();

    let res = v.get("resolution").or_else(|| v.get("calib_dimension"))?;
    let width = res.get("width").or_else(|| res.get("w"))?.as_u64()? as u32;
    let height = res.get("height").or_else(|| res.get("h"))?.as_u64()? as u32;

    let cm = v.get("camera_matrix").or_else(|| {
        v.get("fisheye_params")
            .and_then(|fp| fp.get("camera_matrix"))
    })?;

    let (fx, fy, cx, cy) = if let Some(fx_val) = cm.get("fx") {
        (
            fx_val.as_f64()?,
            cm.get("fy")?.as_f64()?,
            cm.get("cx")?.as_f64()?,
            cm.get("cy")?.as_f64()?,
        )
    } else if let Some(arr) = cm.as_array() {
        if arr.len() >= 3 {
            let r0 = arr[0].as_array()?;
            let r1 = arr[1].as_array()?;
            (
                r0.first()?.as_f64()?,
                r1.get(1)?.as_f64()?,
                r0.get(2)?.as_f64()?,
                r1.get(2)?.as_f64()?,
            )
        } else {
            return None;
        }
    } else {
        return None;
    };

    let dc = v
        .get("distortion_coeffs")
        .or_else(|| {
            v.get("fisheye_params")
                .and_then(|fp| fp.get("distortion_coeffs"))
        })?
        .as_array()?;

    if dc.len() < 4 {
        return None;
    }

    let lens_model = v
        .get("lens_model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let camera_setting = v
        .get("camera_setting")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Some(ProfileEntry {
        source: source.to_string(),
        brand,
        model,
        lens_model,
        camera_setting,
        width,
        height,
        params: CameraParams {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            d: [
                dc[0].as_f64().unwrap_or(0.0),
                dc[1].as_f64().unwrap_or(0.0),
                dc[2].as_f64().unwrap_or(0.0),
                dc[3].as_f64().unwrap_or(0.0),
            ],
        },
    })
}

/// Recursively load JSON profiles from a directory.
fn load_dir_recursive(
    dir: &Path,
    profiles: &mut Vec<ProfileEntry>,
    by_camera: &mut HashMap<String, Vec<usize>>,
) -> Result<(), std::io::Error> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            load_dir_recursive(&path, profiles, by_camera)?;
        } else if path.extension().is_some_and(|e| e == "json") {
            let parsed = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| parse_profile_value(&v, &path.display().to_string()));
            if let Some(entry) = parsed {
                let key = normalize_camera_key(&entry.brand, &entry.model);
                let idx = profiles.len();
                by_camera.entry(key).or_default().push(idx);
                profiles.push(entry);
            }
        }
    }
    Ok(())
}
