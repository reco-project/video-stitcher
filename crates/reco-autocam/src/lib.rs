//! Automatic camera control for reco.
//!
//! The intelligence layer. [`trackers`] turn noisy per-frame detections
//! into a clean [`WorldState`](reco_core::tracker::WorldState) with
//! stable identities and lifecycle flags; [`panners`] turn that world
//! state into a virtual-camera [`ViewportPosition`](reco_core::director::ViewportPosition).
//! Detector backends live in [`reco_detect`] and are re-exported at
//! crate root for convenience but are not owned here.
//!
//! # What this crate owns
//!
//! - [`trackers::BallTracker`] / [`trackers::PlayerTracker`] - per-class
//!   trackers implementing [`Tracker`](reco_core::tracker::Tracker).
//! - [`panners::BallPanner`] / [`panners::FieldPanner`] /
//!   [`panners::SweepPanner`] - camera-motion policies implementing
//!   [`Panner`](reco_core::panner::Panner).
//! - [`panners::Smoother`] / [`panners::Anticipator`] /
//!   [`panners::DeadZone`] - composable panner decorators.
//! - [`RoiFilteredDetector`] - polygonal-ROI mask wrapper over any
//!   `UnifiedDetector`, pre-filtering detections before they reach a
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
//!     .with_tracking_mode(TrackingMode::Ball);
//! reco_autocam::setup_autocam_from_config(&mut session, &config)?;
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]

pub mod detection_filters;
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
// `RoiFilteredGpuDetector` and `RoiFilteredMetalDetector` were
// deleted: the unified `RoiFilteredDetector` covers every residency
// because it wraps `Box<dyn UnifiedDetector>`.
pub use tracking_mode::TrackingMode;

// Path is only used in the ort-backed class-name lookup
// (reco_detect::create_ort_session) and the CpuYoloDetector
// fallback. Both are cfg'd behind `feature = "ort"`, so this
// import follows the same gate.
#[cfg(feature = "ort")]
use std::path::Path;

use reco_core::calibration::FieldRoi;
use reco_core::session::StitchSession;

/// Set up automatic camera control on a [`StitchSession`].
///
/// Configures the appropriate detector (CPU, GPU/CUDA, or Metal) based
/// on the current platform and zero-copy mode, then attaches the
/// tracker(s) and panner chain selected by [`TrackingMode`].
///
/// When `field_roi` is provided, the detector is wrapped in an ROI filter
/// that discards detections outside the playing field polygon before they
/// reach the director.
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
/// reco_autocam::setup_autocam_from_config(&mut session, &config)?;
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
    /// GPU detectors allocate conversion buffers when true.
    pub is_10bit: bool,
}

impl AutocamConfig {
    /// Create a new config with the given model path and sensible defaults.
    pub fn new(model_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tracking_mode: TrackingMode::Ball,
            detection_interval: 1,
            field_roi: None,
            is_10bit: false,
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
pub fn setup_autocam_from_config(
    session: &mut StitchSession,
    config: &AutocamConfig,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (input_width, input_height) = session.pipeline().source_info();
    let use_zero_copy = session.pipeline().gpu().supports_zero_copy();

    setup_autocam(
        session,
        config.model_path.to_str().unwrap_or(""),
        input_width,
        input_height,
        30.0, // default fps when not available from source
        use_zero_copy,
        config.detection_interval,
        0.0, // no lookahead
        config.tracking_mode,
        config.field_roi.as_ref(),
        config.is_10bit,
    )
}

/// Set up the autocam pipeline (detection + direction) on a stitch session.
///
/// For a simpler API with config struct and inferred parameters, use
/// [`setup_autocam_from_config`] with [`AutocamConfig`] instead.
///
/// `is_10bit` should be true when the source produces P010 (10-bit NV12)
/// frames, so GPU detectors allocate conversion buffers.
///
/// Returns `true` if detection was successfully activated, `false` if
/// detection could not be initialized (the session remains usable without
/// autocam in that case).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    not(any(feature = "ort", feature = "tensorrt-native")),
    allow(unused_variables, unused_mut)
)]
pub fn setup_autocam(
    session: &mut StitchSession,
    model_path: &str,
    input_width: u32,
    input_height: u32,
    _fps: f32,
    use_zero_copy: bool,
    detection_interval: u64,
    _lead_time: f64,
    tracking_mode: TrackingMode,
    field_roi: Option<&FieldRoi>,
    is_10bit: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    // No-detector-feature config: ort + tensorrt-native + ncnn all
    // disabled. Every detector path below is cfg-gated out, so the
    // frame loop runs without autocam. Log once and bail so the caller
    // knows why detection stayed off.
    #[cfg(not(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn")))]
    {
        log::warn!(
            "Autocam: no detector backend compiled in (enable ort, tensorrt-native, or ncnn). \
             Session will run without AI camera control."
        );
        return Ok(false);
    }

    #[allow(unreachable_code)]
    let mut detection_active = false;

    // Load class names from the model to resolve label -> class_id for directors.
    // Skip ORT session creation for non-ONNX models (.engine files,
    // NCNN model directories). ORT can only parse .onnx files.
    // When ort is disabled entirely, fall through to empty names —
    // directors use defaults or a sidecar labels file.
    let is_onnx = model_path.ends_with(".onnx");
    #[cfg(feature = "ort")]
    let class_names = if is_onnx {
        match reco_detect::create_ort_session(Path::new(model_path), Vec::new()) {
            Ok((_, _, names)) => names,
            Err(e) => {
                log::warn!("Could not read model labels: {e}, using COCO defaults");
                Vec::new()
            }
        }
    } else {
        Vec::new() // Labels from sidecar file or defaults
    };
    #[cfg(not(feature = "ort"))]
    let class_names: Vec<String> = {
        // Without ort, we can't parse .onnx metadata. Log once, use
        // defaults. TrtGpuDetector pulls labels from a sidecar
        // .labels file (handled in its init path below).
        if is_onnx {
            log::warn!(
                "Autocam: ort feature disabled; can't parse ONNX class names from {model_path}. \
                 Using COCO defaults. For .engine models, place a <name>.labels sidecar."
            );
        }
        Vec::new()
    };

    // Check if ROI filtering should be applied.
    // A FieldRoi is only meaningful if at least one polygon has >= 3 vertices.
    let effective_roi = field_roi
        .filter(|roi| roi.left.len() >= 3 || roi.right.len() >= 3)
        .cloned();
    let has_effective_roi = effective_roi.is_some();
    if has_effective_roi {
        log::info!("Autocam: field ROI filtering enabled");
    }

    // Resolved once up front so each RoiFilteredDetector wrap below
    // can install the Step 7c per-class anchor policy (player = Bottom
    // so feet + 75th-pctile must both lie inside the ROI; ball stays
    // on the Center default).
    let person_id_for_roi = resolve_class_id(&class_names, &["person"], 0);

    // Tiny helper so each backend's "wrap the detector in
    // RoiFilteredDetector if ROI is present" site stays one line.
    let wrap_with_roi = |inner: Box<dyn reco_core::detector::UnifiedDetector>,
                         roi: reco_core::calibration::FieldRoi|
     -> Box<dyn reco_core::detector::UnifiedDetector> {
        Box::new(
            RoiFilteredDetector::new(inner, roi)
                .with_class_anchor(person_id_for_roi, RoiAnchor::Bottom),
        )
    };

    // Native TensorRT path: if the model is a .engine file and the feature
    // is enabled, use TrtGpuDetector directly (no ORT dependency).
    #[cfg(feature = "tensorrt-native")]
    if use_zero_copy && model_path.ends_with(".engine") {
        // Read labels from sidecar file (e.g. model.engine -> model.labels).
        let labels_path = std::path::Path::new(model_path).with_extension("labels");
        let trt_labels = reco_detect::read_labels_file(&labels_path);
        if !trt_labels.is_empty() {
            log::info!(
                "Autocam: loaded {} class labels from {}",
                trt_labels.len(),
                labels_path.display()
            );
        }

        match reco_detect::TrtGpuDetector::try_new(
            model_path,
            input_width,
            input_height,
            0.10,
            trt_labels,
            is_10bit,
        ) {
            Ok(Some(trt_det)) => {
                let detector: Box<dyn reco_core::detector::UnifiedDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(trt_det), roi)
                    } else {
                        Box::new(trt_det)
                    };
                session.set_detector(detector);
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
    #[cfg(all(feature = "ort", any(target_os = "linux", target_os = "windows")))]
    if !detection_active && use_zero_copy {
        match OrtGpuDetector::try_new(
            model_path,
            input_width,
            input_height,
            0.10,
            Vec::new(),
            is_10bit,
        ) {
            Ok(Some(gpu_det)) => {
                let detector: Box<dyn reco_core::detector::UnifiedDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(gpu_det), roi)
                    } else {
                        Box::new(gpu_det)
                    };
                session.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: GPU YOLO ball tracking enabled (model: {model_path})");
            }
            Ok(None) => {
                log::warn!("Autocam: NPP not available, ball tracking disabled in zero-copy mode");
            }
            Err(e) => {
                log::warn!("Autocam: GPU detector init failed ({e}), ball tracking disabled");
            }
        }
    }

    #[cfg(all(feature = "ort", target_os = "macos"))]
    if use_zero_copy {
        match MetalYoloDetector::try_new(
            model_path,
            session.gpu(),
            input_width,
            input_height,
            0.10,
            Vec::new(),
        ) {
            Ok(metal_det) => {
                let detector: Box<dyn reco_core::detector::UnifiedDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(metal_det), roi)
                    } else {
                        Box::new(metal_det)
                    };
                session.set_detector(detector);
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
        match reco_detect::NcnnYoloDetector::new(
            model_path,
            640, // default NCNN input size
            input_width,
            input_height,
            0.25,
            Vec::new(), // labels loaded from sidecar if needed
        ) {
            Ok(ncnn_det) => {
                let detector: Box<dyn reco_core::detector::UnifiedDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(ncnn_det), roi)
                    } else {
                        Box::new(ncnn_det)
                    };
                session.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: NCNN YOLO tracking enabled (model: {model_path})");
            }
            Err(e) => {
                log::warn!("Autocam: NCNN detector init failed ({e}), trying ORT fallback");
            }
        }
    }

    // ORT CPU fallback for .onnx files.
    #[cfg(feature = "ort")]
    if !detection_active && !use_zero_copy {
        let yolo = CpuYoloDetector::from_file(model_path)?;
        let detector: Box<dyn reco_core::detector::UnifiedDetector> =
            if let Some(roi) = effective_roi {
                wrap_with_roi(Box::new(yolo), roi)
            } else {
                Box::new(yolo)
            };
        session.set_detector(detector);
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
        let panner = Box::new(crate::panners::SweepPanner::new(0.8, 10.0));
        session.set_panner(panner);
        return Ok(true);
    }

    if detection_active {
        if detection_interval > 1 {
            session.set_detection_interval(detection_interval);
            log::info!("Detection interval: every {detection_interval} frames");
        }

        // Resolve label names to class IDs from the model's label list.
        let ball_id = resolve_class_id(&class_names, &["ball", "sports ball"], 32);
        let person_id = resolve_class_id(&class_names, &["person"], 0);
        log::info!(
            "Class IDs: ball={ball_id}, person={person_id} (from {} model labels)",
            class_names.len()
        );

        // Pre-tracker flicker rejection: class-keyed bucketed-spatial
        // histogram that drops recurrent static mimics (line
        // intersections, logos, corner flags). Skipped in field mode
        // because the ROI already filters non-pitch detections, and
        // stationary players (pre-kickoff, set pieces) are legitimate
        // signal that the flicker filter aggressively removes.
        if tracking_mode != TrackingMode::Field {
            session.add_detection_filter(Box::new(
                crate::detection_filters::FlickerDetectionFilter::with_defaults(),
            ));
        }

        match tracking_mode {
            TrackingMode::Ball => {
                // Ball tracker picks one detection per frame with
                // plausibility + cross-cam handoff. ROI presence widens
                // the max-jump gate since the ROI has already removed
                // off-pitch false positives that the gate would
                // otherwise guard against.
                // With ROI pre-filtering the off-pitch false positives,
                // the plausibility gate can be permissive enough to
                // accept long passes without rejecting legitimate
                // detections. ~45° / 0.8 rad captures most real ball
                // trajectories between detection intervals.
                let has_roi = has_effective_roi;
                let max_jump = if has_roi {
                    0.8_f32
                } else {
                    crate::trackers::ball::DEFAULT_MAX_JUMP_RAD
                };
                let tracker =
                    crate::trackers::BallTracker::new(ball_id).with_max_jump_rad(max_jump);
                log::info!(
                    "Tracking mode: ball (BallTracker + BallPanner, \
                     max_jump={max_jump:.3}, roi={has_roi})"
                );

                // BallPanner → Smoother → DeadZone. Anticipator was
                // removed after pose-trace analysis showed it overshot
                // every tracker plateau transition (ball tracker
                // output is piecewise-constant between acquisitions,
                // which velocity-lead extrapolation rings on). Heavy
                // smoothing keeps the ball centered without the ring.
                // No Smoother/DeadZone: BallPanner now has its own
                // velocity model with bounded acceleration.
                let ball_panner = crate::panners::BallPanner::new();

                session.set_ball_tracker(Box::new(tracker));
                session.set_panner(Box::new(ball_panner));
            }
            TrackingMode::Field => {
                // Player tracker populates world.players; FieldPanner
                // clusters them and emits the centroid. Same heavy-
                // smoothing chain as Ball mode (BallPanner was the
                // only difference): FieldPanner → Smoother → DeadZone.
                //
                // Class IDs come from the model label list so the same
                // binary works with COCO-indexed (person=0) or custom
                // (person=<whatever>) models.
                let player_tracker = crate::trackers::PlayerTracker::new(person_id);
                let ball_tracker =
                    crate::trackers::BallTracker::new(ball_id).with_max_jump_rad(0.8);
                log::info!(
                    "Tracking mode: field (PlayerTracker + BallTracker + FieldPanner, \
                     player_class={person_id}, ball_class={ball_id})"
                );

                let field_panner =
                    crate::panners::FieldPanner::new().with_ball_weight(0.15);

                session.set_ball_tracker(Box::new(ball_tracker));
                session.set_player_tracker(Box::new(player_tracker));
                session.set_panner(Box::new(field_panner));
            }
            TrackingMode::Sweep => unreachable!("handled before detection block"),
        }
    }

    Ok(detection_active)
}

/// Resolve a class label to its ID from the model's label list.
///
/// Tries each candidate name in order, returning the first match.
/// Falls back to `default_id` if no match is found (e.g. COCO defaults).
fn resolve_class_id(class_names: &[String], candidates: &[&str], default_id: u16) -> u16 {
    for candidate in candidates {
        if let Some(idx) = class_names
            .iter()
            .position(|name| name.eq_ignore_ascii_case(candidate))
        {
            return idx as u16;
        }
    }
    log::warn!(
        "Class '{}' not found in model labels, using COCO default ID {default_id}",
        candidates[0]
    );
    default_id
}
