//! Automatic camera control for reco.
//!
//! The intelligence layer. [`trackers`] turn noisy per-frame detections
//! into a clean [`WorldState`](reco_core::detect::tracker::WorldState) with
//! stable identities and lifecycle flags; [`panners`] turn that world
//! state into a virtual-camera [`WorldPose`](reco_core::geometry::WorldPose).
//! Detector backends live in [`reco_detect`] and are re-exported at
//! crate root for convenience but are not owned here.
//!
//! # What this crate owns
//!
//! - [`trackers::BallTracker`] (stateful singleton) / [`trackers::ClassProvider`]
//!   (stateless per-class projector) - the two shapes of
//!   [`Tracker`](reco_core::detect::tracker::Tracker).
//! - [`panners::FieldPanner`] / [`panners::SweepPanner`] /
//!   [`panners::FilePanner`] - camera-motion policies implementing
//!   [`Panner`](reco_core::detect::panner::Panner).
//! - [`RoiFilteredDetector`] - polygonal-ROI mask wrapper over any
//!   `Detector`, pre-filtering detections before they reach a
//!   tracker.
//! - [`TrackingMode`] + [`AutocamConfig`] + [`setup_autocam`] -
//!   orchestration glue a consumer calls once per session.
//!
//! # Safety policy
//!
//! Zero `unsafe` code by construction. All platform / FFI boundaries
//! live in reco-core (wgpu, zero-copy) or reco-detect (ORT, CUDA), so
//! this crate stays in safe Rust. CI enforces via `#![forbid(unsafe_code)]`;
//! introducing `unsafe` here requires a lint override + an explicit
//! PR discussion.
//!
//! # Usage
//!
//! ```rust,no_run
//! use reco_autocam::{AutocamConfig, TrackingMode};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let mut session: reco_core::session::StitchSession = todo!();
//! let config = AutocamConfig::new("ball_v0.onnx")
//!     .with_tracking_mode(TrackingMode::Field);
//! reco_autocam::setup_autocam(&mut session, &config, 30.0, false)?;
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]

pub mod panners;
mod roi_filter;
pub mod trackers;
mod tracking_mode;

// Re-export detector types from reco-detect for backwards compatibility.
// Ort-backed detectors are only available when the `ort` feature is
// enabled on this crate (passed through to reco-detect). Builds that
// opt out of ort (e.g. Jetson with glibc-incompatible prebuilt ort-sys)
// and rely on tensorrt-native + .engine models don't see these
// re-exports.
#[cfg(feature = "ort")]
pub use reco_detect::CpuYoloDetector;
#[cfg(all(feature = "ort", target_os = "macos"))]
pub use reco_detect::MetalYoloDetector;
#[cfg(all(feature = "ort", any(target_os = "linux", target_os = "windows")))]
pub use reco_detect::OrtGpuDetector;
#[cfg(feature = "tensorrt-native")]
pub use reco_detect::TrtGpuDetector;

pub use roi_filter::{RoiAnchor, RoiFilteredDetector};
pub mod wgpu_detector;
pub use wgpu_detector::WgpuPreprocessingDetector;
// `RoiFilteredGpuDetector` and `RoiFilteredMetalDetector` were
// deleted: the unified `RoiFilteredDetector` covers every residency
// because it wraps `Box<dyn Detector>`.
pub use tracking_mode::TrackingMode;

use std::path::Path;

/// Set up automatic camera control on any [`DetectionTarget`](reco_core::detect::DetectionTarget).
///
/// Configures the appropriate detector (CPU, GPU/CUDA, or Metal) based
/// on the current platform and zero-copy mode, then attaches the
/// tracker(s) and panner chain selected by [`TrackingMode`].
///
/// When `field_roi` is provided, the detector is wrapped in an ROI filter
/// that discards detections outside the playing field polygon before they
/// reach the trackers.
///
/// Configuration for the autocam pipeline.
///
/// All fields have sensible defaults. Only `model_path` is required.
///
/// # Example
///
/// ```rust,ignore
/// use reco_autocam::{AutocamConfig, TrackingMode};
///
/// let config = AutocamConfig::new("model.onnx")
///     .with_tracking_mode(TrackingMode::Field)
///     .with_detection_interval(3);
///
/// reco_autocam::setup_autocam(&mut session, &config, 30.0, false)?;
/// ```
#[derive(Debug, Clone)]
pub struct AutocamConfig {
    /// Path to a YOLO model file (.onnx, .engine, .mlmodelc, or NCNN dir).
    pub model_path: std::path::PathBuf,
    /// Tracking strategy (default: Ball).
    pub tracking_mode: TrackingMode,
    /// Run detection every N frames (default: 1).
    pub detection_interval: u64,
    /// Optional playing field ROI polygons for filtering.
    pub field_roi: Option<reco_core::calibration::FieldRoi>,
    /// Whether the source produces P010 (10-bit NV12) frames.
    pub is_10bit: bool,
    /// Field panner tuning. Only used when tracking_mode is Field.
    pub field_panner_config: Option<crate::panners::FieldPannerConfig>,
    /// Detector confidence threshold override. When set, replaces
    /// the default 0.10 threshold. Ball-only models need 0.25+.
    pub confidence_threshold: Option<f32>,
}

impl AutocamConfig {
    /// Create a new config with the given model path and sensible defaults.
    pub fn new(model_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tracking_mode: TrackingMode::Field,
            detection_interval: 1,
            field_roi: None,
            is_10bit: false,
            field_panner_config: None,
            confidence_threshold: None,
        }
    }

    /// Set the tracking mode.
    pub fn with_tracking_mode(mut self, mode: TrackingMode) -> Self {
        self.tracking_mode = mode;
        self
    }

    /// Set the detection interval.
    pub fn with_detection_interval(mut self, interval: u64) -> Self {
        self.detection_interval = interval;
        self
    }

    /// Set the field ROI for detection filtering.
    pub fn with_field_roi(mut self, roi: reco_core::calibration::FieldRoi) -> Self {
        self.field_roi = Some(roi);
        self
    }

    /// Mark the source as P010 (10-bit NV12).
    ///
    /// When set, GPU detectors allocate scratch buffers to convert 10-bit
    /// samples to 8-bit before NPP color conversion.
    pub fn with_10bit(mut self, is_10bit: bool) -> Self {
        self.is_10bit = is_10bit;
        self
    }
}

/// Set up the autocam pipeline from a config struct.
///
/// Infers input dimensions, fps, and zero-copy mode from the session.
/// Returns `true` if detection was successfully activated.
/// Set up the autocam pipeline (detection + tracking + panning) on a stitch session.
///
/// Infers input dimensions, GPU capabilities, and fps from the session and
/// source. Returns `true` if detection was successfully activated, `false`
/// if no detector backend is available (the session remains usable without
/// autocam).
#[cfg_attr(
    not(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn")),
    allow(unused_variables, unreachable_code)
)]
pub fn setup_autocam(
    target: &mut impl reco_core::detect::DetectionTarget,
    config: &AutocamConfig,
    fps: f32,
    source_is_gpu_resident: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    #[cfg(not(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn")))]
    {
        let _ = (target, config, fps);
        log::warn!(
            "Autocam: no detector backend compiled in (enable ort, tensorrt-native, or ncnn). \
             Session will run without AI camera control."
        );
        return Ok(false);
    }

    let sizes = target.source_sizes();
    let &(input_width, input_height) = sizes.first().ok_or("autocam requires input dimensions")?;
    let mixed_sizes = sizes
        .iter()
        .any(|&size| size != (input_width, input_height));
    if source_is_gpu_resident && mixed_sizes && !cfg!(target_os = "linux") {
        return Err("native autocam preprocessing requires equal camera dimensions".into());
    }
    let use_zero_copy = source_is_gpu_resident;
    let model_path = config.model_path.to_str().unwrap_or("");
    let detection_interval = config.detection_interval;
    let tracking_mode = config.tracking_mode;
    let field_roi = config.field_roi.as_ref();

    let mut detection_active = false;

    let class_names = load_model_labels(model_path);

    // Check if ROI filtering should be applied.
    // A FieldRoi is only meaningful if at least one polygon has >= 3 vertices.
    let effective_roi = field_roi
        .filter(|roi| roi.left.len() >= 3 || roi.right.len() >= 3)
        .cloned();
    let has_effective_roi = effective_roi.is_some();
    if has_effective_roi {
        log::info!("Autocam: field ROI filtering enabled");
    }

    // Only players get the bottom-anchor policy. Known ball-only models keep
    // the center anchor; unlabeled models retain the COCO class fallback.
    let person_id_for_roi = roi_player_class(&class_names);

    // Tiny helper so each backend's "wrap the detector in
    // RoiFilteredDetector if ROI is present" site stays one line.
    let wrap_with_roi = |inner: Box<dyn reco_core::detect::detector::Detector>,
                         roi: reco_core::calibration::FieldRoi|
     -> Box<dyn reco_core::detect::detector::Detector> {
        let filtered = RoiFilteredDetector::new(inner, roi);
        Box::new(match person_id_for_roi {
            Some(person) => filtered.with_class_anchor(person, RoiAnchor::Bottom),
            None => filtered,
        })
    };

    // Native TensorRT path: if the model is a .engine file and the feature
    // is enabled, use TrtGpuDetector directly (no ORT dependency).
    #[cfg(feature = "tensorrt-native")]
    if use_zero_copy && model_path.ends_with(".engine") {
        // `confidence_threshold` is an override: when the caller sets
        // it (e.g. 0.25 for ball-only models) it wins on every backend,
        // not just the CPU fallback; otherwise each backend keeps its
        // own sensible default.
        match reco_detect::TrtGpuDetector::try_new(
            model_path,
            target.source_sizes(),
            config.confidence_threshold.unwrap_or(0.10),
            class_names.clone(),
        ) {
            Ok(Some(trt_det)) => {
                let detector: Box<dyn reco_core::detect::detector::Detector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(trt_det), roi)
                    } else {
                        Box::new(trt_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: native TensorRT tracking enabled (engine: {model_path})");
            }
            Ok(None) => {
                log::warn!("Autocam: NPP not available, TRT detection disabled");
            }
            Err(e) => {
                log::warn!("Autocam: TRT detector init failed ({e})");
            }
        }
    }

    // ORT-based GPU detection (fallback for .onnx models or when tensorrt-native is not enabled).
    // Linux only: OrtGpuDetector accepts DetectorFrame::CudaNv12 (CUDA NV12 pointers from V4L2/VAAPI).
    // On Windows, D3D11VA zero-copy produces WgpuNv12 frames instead — use the wgpu preprocessing
    // path below which wraps CpuYoloDetector (still gets TensorRT EP for inference).
    #[cfg(all(feature = "ort", target_os = "linux"))]
    if !detection_active && use_zero_copy {
        match OrtGpuDetector::try_new(
            model_path,
            target.source_sizes(),
            config.confidence_threshold.unwrap_or(0.10),
            Vec::new(),
            config.is_10bit,
        ) {
            Ok(Some(gpu_det)) => {
                let detector: Box<dyn reco_core::detect::detector::Detector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(gpu_det), roi)
                    } else {
                        Box::new(gpu_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: GPU YOLO ball tracking enabled (model: {model_path})");
            }
            Ok(None) => {
                log::warn!(
                    "Autocam: no CUDA inference provider compiled in, trying host-input inference"
                );
            }
            Err(e) => {
                log::warn!("Autocam: CUDA detector init failed ({e}), trying host-input inference");
            }
        }
    }

    #[cfg(all(feature = "ort", target_os = "macos"))]
    if use_zero_copy && let Some(gpu) = target.gpu() {
        match MetalYoloDetector::try_new(
            model_path,
            gpu,
            target.source_sizes(),
            config.confidence_threshold.unwrap_or(0.10),
            Vec::new(),
        ) {
            Ok(metal_det) => {
                let detector: Box<dyn reco_core::detect::detector::Detector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(metal_det), roi)
                    } else {
                        Box::new(metal_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: Metal YOLO ball tracking enabled (model: {model_path})");
            }
            Err(e) => {
                log::warn!("Autocam: Metal detector init failed ({e}), ball tracking disabled");
            }
        }
    }

    // NCNN backend: use for _ncnn_model directories (Ultralytics NCNN export).
    // Fastest CPU inference on ARM (RPi5: ~67ms vs ORT ~130ms).
    #[cfg(feature = "ncnn")]
    if !detection_active && std::path::Path::new(model_path).is_dir() {
        if mixed_sizes {
            return Err("NCNN preprocessing requires equal camera dimensions".into());
        }
        match reco_detect::NcnnYoloDetector::new(
            model_path,
            640, // default NCNN input size
            input_width,
            input_height,
            config.confidence_threshold.unwrap_or(0.25),
            Vec::new(), // labels loaded from sidecar if needed
        ) {
            Ok(ncnn_det) => {
                let detector: Box<dyn reco_core::detect::detector::Detector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(ncnn_det), roi)
                    } else {
                        Box::new(ncnn_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: NCNN YOLO tracking enabled (model: {model_path})");
            }
            Err(e) => {
                log::warn!("Autocam: NCNN detector init failed ({e}), trying ORT fallback");
            }
        }
    }

    // Fall back through wgpu preprocessing to host-memory model input.
    // Inference can still use a GPU provider if one registered successfully.
    #[cfg(feature = "ort")]
    if !detection_active
        && use_zero_copy
        && let Some(gpu) = target.gpu()
    {
        let yolo = CpuYoloDetector::with_config(
            model_path,
            config.confidence_threshold.unwrap_or(0.10),
            Vec::new(),
        )?;
        let input_size = yolo.input_size();
        let wrapper = WgpuPreprocessingDetector::new(
            Box::new(yolo),
            gpu.device().clone(),
            gpu.queue().clone(),
            input_size,
            target.source_sizes().to_vec(),
        )?;
        let detector: Box<dyn reco_core::detect::detector::Detector> =
            if let Some(roi) = effective_roi.clone() {
                wrap_with_roi(Box::new(wrapper), roi)
            } else {
                Box::new(wrapper)
            };
        target.set_detector(detector);
        detection_active = true;
        log::info!(
            "Autocam: wgpu preprocessing + host-input ORT tracking enabled (model: {model_path})"
        );
    }

    // ORT CPU fallback for .onnx files.
    #[cfg(feature = "ort")]
    if !detection_active && !use_zero_copy {
        let conf = config.confidence_threshold.unwrap_or(0.10);
        let yolo = CpuYoloDetector::with_config(model_path, conf, Vec::new())?;
        let detector: Box<dyn reco_core::detect::detector::Detector> =
            if let Some(roi) = effective_roi {
                wrap_with_roi(Box::new(yolo), roi)
            } else {
                Box::new(yolo)
            };
        target.set_detector(detector);
        detection_active = true;
        log::info!("Autocam: YOLO ball tracking enabled (model: {model_path})");
    }
    // Without ort feature, detection only activates via the
    // tensorrt-native or ncnn branches above. If we still don't
    // have a detector here, log so the user understands why.
    #[cfg(not(feature = "ort"))]
    if !detection_active {
        log::warn!(
            "Autocam: no detector attached. Build has `ort` disabled; only `.engine` \
             (tensorrt-native) and NCNN `_ncnn_model` directories are supported. \
             Received model_path={model_path}"
        );
    }

    // Sweep panner doesn't need detection - attach it regardless.
    if tracking_mode == TrackingMode::Sweep {
        log::info!("Tracking mode: sweep (debug, no AI)");
        let panner =
            Box::new(crate::panners::SweepPanner::new(0.8, 10.0).with_zoom(30.0, 90.0, 7.0));
        target.set_panner(panner);
        return Ok(true);
    }

    if detection_active {
        if detection_interval > 1 {
            target.set_detection_interval(detection_interval);
            log::info!("Detection interval: every {detection_interval} frames");
        }

        // Resolve label names to class IDs from the model's label list.
        // The ball tracker always needs an id (COCO fallback); the player
        // class is left as an Option so field mode can tell "no players
        // in this model" apart from "players at the COCO index".
        let ball_id = resolve_or(&class_names, &["ball", "sports ball"], 32);
        let person = resolve_class_id(&class_names, &["person"]);
        match person {
            Some(p) => log::info!(
                "Resolved class ids from {} model labels: ball={ball_id}, person={p}",
                class_names.len()
            ),
            None => log::info!(
                "Resolved class ids from {} model labels: ball={ball_id}, person=absent",
                class_names.len()
            ),
        }

        match tracking_mode {
            TrackingMode::Field => {
                let ball_tracker =
                    crate::trackers::BallTracker::new(ball_id).with_max_jump_rad(0.8);
                target.set_ball_tracker(Box::new(ball_tracker));

                // Attach the player provider only when the model actually
                // names a player class. With a ball-only model there are no
                // players to cluster, so the panner runs on the ball alone
                // rather than mis-ingesting the ball as a "player" (its id
                // would collide with the COCO person fallback).
                match person {
                    Some(person_id) => {
                        target.set_player_tracker(Box::new(crate::trackers::ClassProvider::new(
                            person_id,
                        )));
                        log::info!(
                            "Tracking mode: field with player cluster \
                             (player_class={person_id}, ball_class={ball_id})"
                        );
                    }
                    None => log::info!(
                        "Tracking mode: field, but the model names no player class - the panner \
                         will follow the ball alone (ball_class={ball_id})"
                    ),
                }

                let fp_config = config.field_panner_config.clone().unwrap_or(
                    crate::panners::FieldPannerConfig {
                        ball_weight: 0.20,
                        ..Default::default()
                    },
                );
                log::info!(
                    "FieldPanner: framing={:?}, confidence_weighted={}, lock_pitch={}",
                    fp_config.framing,
                    fp_config.confidence_weighted,
                    fp_config.lock_pitch,
                );
                let field_panner = crate::panners::FieldPanner::with_config(fps, fp_config);
                target.set_panner(Box::new(field_panner));
            }
            TrackingMode::Ball => {
                let ball_tracker =
                    crate::trackers::BallTracker::new(ball_id).with_max_jump_rad(0.5);
                // No player provider - ball-only mode, even if the model
                // has a player class (the user asked to track the ball).
                target.set_ball_tracker(Box::new(ball_tracker));

                // No player provider means no cluster, so the ball must
                // drive the frame fully: force ball_weight=1.0 even when a
                // config is supplied (there is nothing to blend against).
                // Owning this here is what lets consumers pass a panner
                // config without re-deriving the mode default themselves.
                // All other tuning (dead-zone / reactivity / lead / fov)
                // still comes from the supplied override.
                let mut fp_config = config.field_panner_config.clone().unwrap_or_default();
                fp_config.ball_weight = 1.0;
                let panner = crate::panners::FieldPanner::with_config(fps, fp_config);

                log::info!(
                    "Tracking mode: ball-only (BallTracker + FieldPanner, \
                     ball_class={ball_id})"
                );
                target.set_panner(Box::new(panner));
            }
            TrackingMode::Sweep => unreachable!("handled before detection block"),
        }
    }

    Ok(detection_active)
}

/// Resolve a class label to its ID from the model's label list, by name
/// (case-insensitive), trying each candidate in order.
///
/// Returns `None` when the model names none of the candidates - the
/// caller decides whether to fall back to a default id or to skip the
/// class entirely (e.g. don't attach a player provider for a model with
/// no player class). This is the signal that drives field-mode's
/// adaptive wiring.
fn resolve_class_id(class_names: &[String], candidates: &[&str]) -> Option<u16> {
    candidates.iter().find_map(|candidate| {
        class_names
            .iter()
            .position(|name| name.eq_ignore_ascii_case(candidate))
            .map(|idx| idx as u16)
    })
}

/// [`resolve_class_id`] with a COCO-index fallback for the ball tracker. Logs when the fallback is used so a mislabeled or label-less model is visible.
fn resolve_or(class_names: &[String], candidates: &[&str], default_id: u16) -> u16 {
    resolve_class_id(class_names, candidates).unwrap_or_else(|| {
        log::warn!(
            "Class '{}' not found in model labels; using COCO default id {default_id}",
            candidates[0]
        );
        default_id
    })
}

fn load_model_labels(model_path: &str) -> Vec<String> {
    let is_onnx = model_path.ends_with(".onnx");
    if model_path.ends_with(".engine") {
        reco_detect::read_labels_file(Path::new(model_path).with_extension("labels"))
    } else {
        #[cfg(feature = "ort")]
        {
            if is_onnx {
                match reco_detect::create_ort_session(Path::new(model_path), Vec::new()) {
                    Ok((_, _, names)) => names,
                    Err(e) => {
                        log::warn!("Could not read model labels: {e}, using COCO defaults");
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        }
        #[cfg(not(feature = "ort"))]
        {
            if is_onnx {
                log::warn!(
                    "Autocam: ort feature disabled; can't parse ONNX class names from {model_path}. Using COCO defaults."
                );
            }
            Vec::new()
        }
    }
}

fn roi_player_class(class_names: &[String]) -> Option<u16> {
    resolve_class_id(class_names, &["person"]).or_else(|| class_names.is_empty().then_some(0))
}

#[cfg(test)]
mod setup_tests {
    use super::*;

    #[test]
    fn engine_sidecar_drives_tracker_and_roi_class_ids() {
        let dir = tempfile::tempdir().unwrap();
        let engine = dir.path().join("detector.engine");
        let labels_path = engine.with_extension("labels");
        std::fs::write(&labels_path, "ball\n").unwrap();
        let labels = load_model_labels(engine.to_str().unwrap());
        assert_eq!(labels, ["ball"]);
        assert_eq!(resolve_or(&labels, &["ball", "sports ball"], 32), 0);
        assert_eq!(roi_player_class(&labels), None);
        std::fs::write(&labels_path, "ball\nperson\n").unwrap();
        let labels = load_model_labels(engine.to_str().unwrap());
        assert_eq!(resolve_or(&labels, &["ball", "sports ball"], 32), 0);
        assert_eq!(roi_player_class(&labels), Some(1));
    }

    #[test]
    fn missing_labels_keep_coco_fallback_without_inventing_known_classes() {
        let dir = tempfile::tempdir().unwrap();
        let labels = load_model_labels(dir.path().join("missing.engine").to_str().unwrap());
        assert!(labels.is_empty());
        assert_eq!(resolve_or(&labels, &["ball", "sports ball"], 32), 32);
        assert_eq!(roi_player_class(&labels), Some(0));
        assert_eq!(roi_player_class(&["ball".into(), "other".into()]), None);
    }
}
