//! Explicit lens-profile search and file loading.
//!
//! The bundled Gyroflow database is an optional manual picker. Calibration never
//! selects a profile from camera metadata or falls back to a resolution match.

use reco_core::calibration::Lens;
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

use crate::types::LensProfileSummary;

static EMBEDDED_DB: OnceLock<LensDatabase> = OnceLock::new();

/// Embedded Gyroflow lens profile database (profiles.cbor.gz).
static PROFILES_CBOR_GZ: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../resources/profiles.cbor.gz"
));

/// A loaded lens profile entry.
#[derive(Debug, Clone)]
struct ProfileEntry {
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
    params: Lens,
}

/// Lens profile database.
///
/// Loads profiles from the embedded CBOR bundle for explicit user selection.
pub struct LensDatabase {
    profiles: Vec<ProfileEntry>,
}

impl LensDatabase {
    /// Get the shared embedded profile database (cached singleton).
    ///
    /// The first call decompresses and parses the bundled CBOR. Subsequent calls return the same reference.
    pub fn embedded() -> &'static LensDatabase {
        EMBEDDED_DB.get_or_init(Self::load_embedded)
    }

    /// Load the embedded Gyroflow profile database.
    ///
    /// Prefer [`embedded()`](Self::embedded) for the cached singleton.
    /// This allocates a fresh database each call.
    fn load_embedded() -> Self {
        let mut db = Self {
            profiles: Vec::new(),
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
            if let Some(entry) = parse_profile_value(&value) {
                db.profiles.push(entry);
            }
        }

        log::info!("lens database: {} profiles (embedded)", db.profiles.len());
        db
    }

    /// Full-text search across all profiles.
    ///
    /// Splits `query` into lowercase words and returns profiles where
    /// every word appears in the combined "brand model lens_model
    /// camera_setting WxH" string. Results matching `width`x`height`
    /// sort first (exact match), then same aspect ratio, then the rest.
    /// Pass 0 for width/height to skip resolution ranking.
    /// Capped at 100 results.
    pub fn search(&self, query: &str, width: u32, height: u32) -> Vec<LensProfileSummary> {
        let words: Vec<String> = query.split_whitespace().map(|w| w.to_lowercase()).collect();
        if words.is_empty() {
            return Vec::new();
        }

        let target_aspect = if width > 0 && height > 0 {
            Some(width as f64 / height as f64)
        } else {
            None
        };

        let mut hits: Vec<(usize, u8)> = Vec::new();
        for (i, p) in self.profiles.iter().enumerate() {
            let haystack = format!(
                "{} {} {} {} {}x{}",
                p.brand, p.model, p.lens_model, p.camera_setting, p.width, p.height
            )
            .to_lowercase();
            if words.iter().all(|w| haystack.contains(w.as_str())) {
                let priority = if width > 0 && p.width == width && height > 0 && p.height == height
                {
                    0
                } else if let Some(ta) = target_aspect {
                    let pa = p.width as f64 / p.height as f64;
                    if (pa - ta).abs() < 0.05 { 1 } else { 2 }
                } else {
                    2
                };
                hits.push((i, priority));
            }
        }

        hits.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        hits.truncate(100);

        hits.iter().map(|&(i, _)| self.summary_at(i)).collect()
    }

    /// Load the profile explicitly selected by a consumer.
    pub fn load_by_summary(&self, summary: &LensProfileSummary) -> Option<Lens> {
        self.profiles
            .iter()
            .find(|p| {
                format_camera_name(&p.brand, &p.model) == summary.camera
                    && format_lens_name(&p.lens_model, &p.camera_setting) == summary.lens
                    && p.width == summary.width
                    && p.height == summary.height
            })
            .map(|p| p.params.clone())
    }

    fn summary_at(&self, idx: usize) -> LensProfileSummary {
        let p = &self.profiles[idx];
        LensProfileSummary {
            camera: format_camera_name(&p.brand, &p.model),
            lens: format_lens_name(&p.lens_model, &p.camera_setting),
            width: p.width,
            height: p.height,
        }
    }
}

/// Format "Brand Model" with title case from the lowercase internal storage.
fn format_camera_name(brand: &str, model: &str) -> String {
    let mut name = String::with_capacity(brand.len() + 1 + model.len());
    for (i, word) in brand.split_whitespace().enumerate() {
        if i > 0 {
            name.push(' ');
        }
        let mut chars = word.chars();
        if let Some(c) = chars.next() {
            name.extend(c.to_uppercase());
            name.extend(chars);
        }
    }
    name.push(' ');
    for (i, word) in model.split_whitespace().enumerate() {
        if i > 0 {
            name.push(' ');
        }
        let mut chars = word.chars();
        if let Some(c) = chars.next() {
            name.extend(c.to_uppercase());
            name.extend(chars);
        }
    }
    name
}

/// Format lens name from model and setting fields.
fn format_lens_name(lens_model: &str, camera_setting: &str) -> String {
    if camera_setting.is_empty() {
        lens_model.to_string()
    } else {
        format!("{lens_model} ({camera_setting})")
    }
}

/// Load a camera profile from a JSON file.
///
/// Accepts multiple formats:
/// - v1 uniforms: `{fx, fy, cx, cy, d: [k1..k4], width, height}`
/// - Gyroflow KB4: `{camera_matrix: {fx,fy,cx,cy}, distortion_coeffs: [...], resolution: {w,h}}`
/// - Gyroflow with fisheye_params wrapper
///
/// This is the standard way to load a manually-specified lens profile.
pub fn load_from_file(path: &Path) -> Result<Lens, LensLoadError> {
    let json_str = std::fs::read_to_string(path)
        .map_err(|e| LensLoadError::Io(format!("{}: {e}", path.display())))?;
    load_from_json(&json_str, path.display().to_string().as_str())
}

/// Load a camera profile from a JSON string.
pub fn load_from_json(json_str: &str, source: &str) -> Result<Lens, LensLoadError> {
    let v: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| LensLoadError::Parse(format!("{source}: {e}")))?;

    // Flat lens format (fx/fy/cx/cy + distortion). This branch is the
    // designated adapter for the legacy wire shape too: old profile files
    // named the distortion array `d`, so rename that one key here rather
    // than carrying a serde alias on the Lens schema itself.
    if v.get("fx").is_some() && (v.get("distortion").is_some() || v.get("d").is_some()) {
        let mut v = v;
        if let Some(obj) = v.as_object_mut()
            && let Some(d) = obj.remove("d")
        {
            obj.entry("distortion").or_insert(d);
        }
        return serde_json::from_value::<Lens>(v)
            .map_err(|e| LensLoadError::Parse(format!("{source}: {e}")));
    }

    if let Some(lens) = parse_gyroflow_lens(&v) {
        return Ok(lens);
    }

    Err(LensLoadError::UnrecognizedFormat(source.to_string()))
}

/// Errors from lens profile loading.
#[derive(Debug, thiserror::Error)]
pub enum LensLoadError {
    /// File I/O error.
    #[error("cannot read lens profile: {0}")]
    Io(String),
    /// JSON parsing error.
    #[error("cannot parse lens profile: {0}")]
    Parse(String),
    /// Unrecognized format.
    #[error(
        "unsupported or malformed lens profile in '{0}'. Expected a Reco lens or Gyroflow opencv_fisheye profile with four distortion coefficients."
    )]
    UnrecognizedFormat(String),
}

fn parse_profile_value(v: &serde_json::Value) -> Option<ProfileEntry> {
    let brand = v.get("camera_brand")?.as_str()?.to_string();
    let model = v.get("camera_model")?.as_str()?.to_string();
    let params = parse_gyroflow_lens(v)?;
    Some(ProfileEntry {
        brand,
        model,
        lens_model: v
            .get("lens_model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        camera_setting: v
            .get("camera_setting")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        width: params.width,
        height: params.height,
        params,
    })
}

/// Read KB4 optics from a Gyroflow profile, independently of database labels.
pub(crate) fn parse_gyroflow_lens(v: &serde_json::Value) -> Option<Lens> {
    if v.get("distortion_model")
        .filter(|model| !model.is_null())
        .is_some_and(|model| model.as_str() != Some("opencv_fisheye"))
    {
        return None;
    }

    let res = v.get("resolution").or_else(|| v.get("calib_dimension"))?;
    let width = u32::try_from(res.get("width").or_else(|| res.get("w"))?.as_u64()?).ok()?;
    let height = u32::try_from(res.get("height").or_else(|| res.get("h"))?.as_u64()?).ok()?;
    if width == 0 || height == 0 {
        return None;
    }

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
        if arr.len() == 3 {
            let r0 = arr[0].as_array()?;
            let r1 = arr[1].as_array()?;
            let r2 = arr[2].as_array()?;
            if r0.len() != 3
                || r1.len() != 3
                || r2.len() != 3
                || r0[1].as_f64()? != 0.0
                || r1[0].as_f64()? != 0.0
                || r2[0].as_f64()? != 0.0
                || r2[1].as_f64()? != 0.0
                || r2[2].as_f64()? != 1.0
            {
                return None;
            }
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
    if fx <= 0.0 || fy <= 0.0 {
        return None;
    }

    let dc = v
        .get("distortion_coeffs")
        .or_else(|| {
            v.get("fisheye_params")
                .and_then(|fp| fp.get("distortion_coeffs"))
        })?
        .as_array()?;

    if dc.len() != 4 {
        return None;
    }
    Some(Lens {
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        distortion: [
            dc[0].as_f64()?,
            dc[1].as_f64()?,
            dc[2].as_f64()?,
            dc[3].as_f64()?,
        ],
        correction: 1.0,
    })
}

/// Recursively load JSON profiles from a directory.
#[cfg(test)]
mod tests {
    use super::*;

    fn gyroflow_profile() -> serde_json::Value {
        serde_json::json!({
            "calib_dimension": {"w": 1920, "h": 1080},
            "fisheye_params": {
                "camera_matrix": [[900., 0., 955.], [0., 910., 535.], [0., 0., 1.]],
                "distortion_coeffs": [0.1, -0.01, 0.002, -0.0001]
            }
        })
    }

    #[test]
    fn file_and_database_profiles_share_the_embedded_optical_parser() {
        let mut profile = gyroflow_profile();
        let expected = parse_gyroflow_lens(&profile).unwrap();
        assert_eq!(
            load_from_json(&profile.to_string(), "fixture").unwrap(),
            expected
        );
        assert!(parse_profile_value(&profile).is_none());
        profile["camera_brand"] = serde_json::json!("Example");
        profile["camera_model"] = serde_json::json!("Camera");
        profile["distortion_model"] = serde_json::json!("opencv_fisheye");
        assert_eq!(parse_profile_value(&profile).unwrap().params, expected);
    }

    #[test]
    fn malformed_or_incompatible_gyroflow_optics_are_rejected() {
        use serde_json::json;
        for (path, value) in [
            ("/calib_dimension/w", json!(0)),
            ("/calib_dimension/h", json!(u64::from(u32::MAX) + 1)),
            ("/fisheye_params/camera_matrix/0/0", json!(-1.)),
            ("/fisheye_params/camera_matrix/1/1", json!(0.)),
            ("/fisheye_params/camera_matrix/0/1", json!(1.)),
            ("/fisheye_params/camera_matrix/2/2", json!(2.)),
            ("/fisheye_params/camera_matrix/0/2", json!("missing")),
            ("/fisheye_params/distortion_coeffs/2", json!(null)),
            ("/fisheye_params/distortion_coeffs", json!([0., 0., 0.])),
            (
                "/fisheye_params/distortion_coeffs",
                json!([0., 0., 0., 0., 0.]),
            ),
        ] {
            let mut profile = gyroflow_profile();
            *profile.pointer_mut(path).unwrap() = value;
            assert!(parse_gyroflow_lens(&profile).is_none(), "{path}");
            assert!(
                load_from_json(&profile.to_string(), "fixture").is_err(),
                "{path}"
            );
        }
        for model in [json!("insta360"), json!("opencv_standard"), json!(false)] {
            let mut profile = gyroflow_profile();
            profile["distortion_model"] = model;
            assert!(parse_gyroflow_lens(&profile).is_none());
        }
    }

    #[test]
    fn large_reference_profiles_remain_rescalable_before_rendering() {
        let mut profile = gyroflow_profile();
        profile["calib_dimension"] = serde_json::json!({"w": 12000, "h": 9000});
        let lens = parse_gyroflow_lens(&profile).unwrap();
        assert_eq!((lens.width, lens.height), (12000, 9000));
        assert!(lens.rescaled(4000, 3000).is_ok());
    }

    #[test]
    fn flat_profile_loads_current_and_legacy_key() {
        // Current shape names the array `distortion`; legacy profile files
        // named it `d`. Both must load through the flat branch (the GUI
        // lens picker and `reco camera live-calibrate` feed files here).
        let current = r#"{"width":3840,"height":2160,"fx":1796.3,"fy":1797.2,
                          "cx":1919.4,"cy":1063.2,"distortion":[0.03,0.07,-0.07,0.03]}"#;
        let legacy = r#"{"width":3840,"height":2160,"fx":1796.3,"fy":1797.2,
                         "cx":1919.4,"cy":1063.2,"d":[0.03,0.07,-0.07,0.03]}"#;
        for (label, json) in [("current", current), ("legacy", legacy)] {
            let lens = load_from_json(json, label)
                .unwrap_or_else(|e| panic!("{label} flat profile must load: {e}"));
            assert_eq!(lens.width, 3840, "{label}");
            assert!((lens.distortion[0] - 0.03).abs() < 1e-12, "{label}");
        }
    }

    #[test]
    fn embedded_singleton_returns_same_ref() {
        let a = LensDatabase::embedded();
        let b = LensDatabase::embedded();
        assert!(std::ptr::eq(a, b));
        assert!(!a.profiles.is_empty());
    }

    #[test]
    fn search_gopro_returns_results() {
        let db = LensDatabase::embedded();
        let results = db.search("gopro hero10", 0, 0);
        assert!(!results.is_empty());
        assert!(
            results
                .iter()
                .all(|r| r.camera.to_lowercase().contains("gopro")
                    && r.camera.to_lowercase().contains("hero10"))
        );
    }

    #[test]
    fn search_resolution_ranking() {
        let db = LensDatabase::embedded();
        let results = db.search("gopro hero", 3840, 2160);
        if !results.is_empty() {
            assert_eq!(results[0].width, 3840);
            assert_eq!(results[0].height, 2160);
        }
    }

    #[test]
    fn search_multi_word() {
        let db = LensDatabase::embedded();
        let results = db.search("hero10 wide", 0, 0);
        for r in &results {
            let combined = format!("{} {}", r.camera, r.lens).to_lowercase();
            assert!(combined.contains("hero"));
            assert!(combined.contains("wide"));
        }
    }

    #[test]
    fn search_caps_at_100() {
        let db = LensDatabase::embedded();
        let results = db.search("a", 0, 0);
        assert!(results.len() <= 100);
    }

    #[test]
    fn load_by_summary_roundtrip() {
        let db = LensDatabase::embedded();
        let results = db.search("gopro hero10", 0, 0);
        if let Some(summary) = results.first() {
            let params = db.load_by_summary(summary);
            assert!(params.is_some());
            let p = params.unwrap();
            assert!(p.fx > 0.0);
            assert!(p.fy > 0.0);
        }
    }
}
