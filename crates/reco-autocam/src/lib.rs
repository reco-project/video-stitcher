//! Automatic camera control for reco.
//!
//! The intelligence layer. [`trackers`] turn noisy per-frame detections
//! into a clean [`WorldState`](reco_core::detect::tracker::WorldState) with
//! stable identities and lifecycle flags; [`panners`] turn that world
//! state into a virtual-camera [`WorldPose`](reco_core::geometry::WorldPose).
//! Detector backends live in `reco-detect` and are re-exported at
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
//! - [`RoiFilter`] and [`RoiFilteredDetector`] - one polygonal-ROI policy for completed observations or synchronous inference, before tracking.
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

#[cfg(all(test, feature = "ncnn"))]
mod ncnn_tests;
pub mod panners;
mod roi_filter;
pub mod trackers;
mod tracking_mode;

// Backend feature selection controls both setup and the available detector types.
#[cfg(feature = "ort")]
pub use reco_detect::CpuYoloDetector;
#[cfg(all(feature = "ort", target_os = "macos"))]
pub use reco_detect::MetalYoloDetector;
#[cfg(all(feature = "ort", any(target_os = "linux", target_os = "windows")))]
pub use reco_detect::OrtGpuDetector;
#[cfg(feature = "tensorrt-native")]
pub use reco_detect::TrtGpuDetector;

pub use roi_filter::{RoiAnchor, RoiFilter, RoiFilteredDetector};
#[cfg(feature = "detectors")]
pub mod wgpu_detector;
pub use tracking_mode::TrackingMode;
#[cfg(feature = "detectors")]
pub use wgpu_detector::WgpuPreprocessingDetector;

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
    /// Tracking strategy (default: Field).
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

/// Attach the selected camera-control policy and its detector when needed.
///
/// Source dimensions and GPU context come from the target. The caller supplies source FPS and residency. Returns `true` when configured, or `false` when no detector backend is available. Sweep mode requires neither a detector nor a model file.
pub fn setup_autocam(
    target: &mut impl reco_core::detect::DetectionTarget,
    config: &AutocamConfig,
    fps: f32,
    source_is_gpu_resident: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    if config.tracking_mode == TrackingMode::Sweep {
        target.set_panner(Box::new(
            crate::panners::SweepPanner::new(0.8, 10.0)
                .with_zoom(30.0, 90.0, 7.0)
                .with_fps(fps),
        ));
        return Ok(true);
    }
    #[cfg(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn"))]
    {
        setup_detection(target, config, fps, source_is_gpu_resident)
    }
    #[cfg(not(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn")))]
    {
        let _ = (target, config, fps, source_is_gpu_resident);
        log::warn!(
            "Autocam: no detector backend compiled in (enable ort, tensorrt-native, or ncnn). \
             Session will run without AI camera control."
        );
        Ok(false)
    }
}

#[cfg(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn"))]
fn setup_detection(
    target: &mut impl reco_core::detect::DetectionTarget,
    config: &AutocamConfig,
    fps: f32,
    source_is_gpu_resident: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(detector) = select_detector(target, config, source_is_gpu_resident)? else {
        return Ok(false);
    };
    attach_control(target, config, fps, detector);
    Ok(true)
}

#[cfg(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn"))]
fn select_detector(
    target: &impl reco_core::detect::DetectionTarget,
    config: &AutocamConfig,
    source_is_gpu_resident: bool,
) -> Result<Option<Box<dyn reco_core::detect::detector::Detector>>, Box<dyn std::error::Error>> {
    let sizes = target.source_sizes();
    let &first = sizes.first().ok_or("autocam requires input dimensions")?;
    if source_is_gpu_resident
        && sizes.iter().any(|size| *size != first)
        && !cfg!(target_os = "linux")
    {
        return Err("native autocam preprocessing requires equal camera dimensions".into());
    }
    let model_path = config
        .model_path
        .to_str()
        .ok_or("model path is not UTF-8")?;
    #[cfg(any(feature = "ort", feature = "tensorrt-native"))]
    let confidence = config.confidence_threshold.unwrap_or(0.10);

    #[cfg(feature = "tensorrt-native")]
    if source_is_gpu_resident && model_path.ends_with(".engine") {
        let labels = reco_detect::read_labels_file(config.model_path.with_extension("labels"));
        match TrtGpuDetector::try_new(model_path, sizes, confidence, labels) {
            Ok(Some(detector)) => return Ok(Some(Box::new(detector))),
            Ok(None) => log::warn!("Autocam: NPP not available, TRT detection disabled"),
            Err(error) => log::warn!("Autocam: TRT detector init failed ({error})"),
        }
    }

    // CUDA input requires a CUDA-compatible inference provider; wgpu input uses the adapter below.
    #[cfg(all(feature = "ort", target_os = "linux"))]
    if source_is_gpu_resident {
        match OrtGpuDetector::try_new(model_path, sizes, confidence, Vec::new(), config.is_10bit) {
            Ok(Some(detector)) => return Ok(Some(Box::new(detector))),
            Ok(None) => log::warn!(
                "Autocam: no CUDA inference provider compiled in, trying host-input inference"
            ),
            Err(error) => log::warn!(
                "Autocam: CUDA detector init failed ({error}), trying host-input inference"
            ),
        }
    }

    #[cfg(all(feature = "ort", target_os = "macos"))]
    if source_is_gpu_resident && let Some(gpu) = target.gpu() {
        match MetalYoloDetector::try_new(model_path, gpu, sizes, confidence, Vec::new()) {
            Ok(detector) => return Ok(Some(Box::new(detector))),
            Err(error) => log::warn!(
                "Autocam: Metal detector init failed ({error}), trying host-input inference"
            ),
        }
    }

    #[cfg(feature = "ncnn")]
    if config.model_path.is_dir() {
        if source_is_gpu_resident {
            return Err("NCNN requires host YUV frames; select host-memory decoding".into());
        }
        match reco_detect::NcnnYoloDetector::new(
            model_path,
            640,
            config.confidence_threshold.unwrap_or(0.25),
            Vec::new(),
        ) {
            Ok(detector) => return Ok(Some(Box::new(detector))),
            Err(error) => {
                log::warn!("Autocam: NCNN detector init failed ({error}), trying ORT fallback")
            }
        }
    }

    #[cfg(feature = "ort")]
    {
        if source_is_gpu_resident {
            if let Some(gpu) = target.gpu() {
                let detector = CpuYoloDetector::with_config(model_path, confidence, Vec::new())?;
                let input_size = detector.input_size();
                return Ok(Some(Box::new(WgpuPreprocessingDetector::new(
                    Box::new(detector),
                    gpu.device().clone(),
                    gpu.queue().clone(),
                    input_size,
                    sizes.to_vec(),
                )?)));
            }
        } else {
            return Ok(Some(Box::new(CpuYoloDetector::with_config(
                model_path,
                confidence,
                Vec::new(),
            )?)));
        }
    }
    log::warn!(
        "Autocam: no detector available for model {model_path} and selected input residency"
    );
    Ok(None)
}

#[cfg(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn"))]
fn attach_control(
    target: &mut impl reco_core::detect::DetectionTarget,
    config: &AutocamConfig,
    fps: f32,
    detector: Box<dyn reco_core::detect::detector::Detector>,
) {
    // Read metadata from the selected, loaded detector so inference, tracking and ROI use one class interpretation.
    let names = detector.class_names().unwrap_or(&[]);
    let ball = resolve_or(names, &["ball", "sports ball"], 32);
    let person = resolve_class_id(names, &["person"]);
    let roi_player = roi_player_class(names);
    log::info!(
        "Autocam: {} detector, ball={ball}, person={person:?}",
        detector.name()
    );

    let detector: Box<dyn reco_core::detect::detector::Detector> = if let Some(roi) = config
        .field_roi
        .as_ref()
        .filter(|roi| roi.left.len() >= 3 || roi.right.len() >= 3)
    {
        let filter = RoiFilter::new(roi.clone());
        let filter = match roi_player {
            Some(class) => filter.with_class_anchor(class, RoiAnchor::Bottom),
            None => filter,
        };
        Box::new(RoiFilteredDetector::new(detector, filter))
    } else {
        detector
    };
    target.set_detector(detector);
    target.set_detection_interval(config.detection_interval);

    match config.tracking_mode {
        TrackingMode::Field => {
            target.set_ball_tracker(Box::new(
                crate::trackers::BallTracker::new(ball).with_max_jump_rad(0.8),
            ));
            if let Some(person) = person {
                target.set_player_tracker(Box::new(crate::trackers::ClassProvider::new(person)));
            }
            let panner =
                config
                    .field_panner_config
                    .clone()
                    .unwrap_or(crate::panners::FieldPannerConfig {
                        ball_weight: 0.20,
                        ..Default::default()
                    });
            target.set_panner(Box::new(crate::panners::FieldPanner::with_config(
                fps, panner,
            )));
        }
        TrackingMode::Ball => {
            target.set_ball_tracker(Box::new(
                crate::trackers::BallTracker::new(ball).with_max_jump_rad(0.5),
            ));
            let mut panner = config.field_panner_config.clone().unwrap_or_default();
            // Ball-only mode has no player cluster to mix with.
            panner.ball_weight = 1.0;
            target.set_panner(Box::new(crate::panners::FieldPanner::with_config(
                fps, panner,
            )));
        }
        TrackingMode::Sweep => unreachable!("sweep does not initialize detection"),
    }
}

/// Resolve a class label to its ID from the model's label list, by name
/// (case-insensitive), trying each candidate in order.
///
/// Returns `None` when the model names none of the candidates - the
/// caller decides whether to fall back to a default id or to skip the
/// class entirely (e.g. don't attach a player provider for a model with
/// no player class). This is the signal that drives field-mode's
/// adaptive wiring.
#[cfg(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn"))]
fn resolve_class_id(class_names: &[String], candidates: &[&str]) -> Option<u16> {
    candidates.iter().find_map(|candidate| {
        class_names
            .iter()
            .position(|name| name.eq_ignore_ascii_case(candidate))
            .map(|idx| idx as u16)
    })
}

/// [`resolve_class_id`] with a COCO-index fallback for the ball tracker. Logs when the fallback is used so a mislabeled or label-less model is visible.
#[cfg(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn"))]
fn resolve_or(class_names: &[String], candidates: &[&str], default_id: u16) -> u16 {
    resolve_class_id(class_names, candidates).unwrap_or_else(|| {
        log::warn!(
            "Class '{}' not found in model labels; using COCO default id {default_id}",
            candidates[0]
        );
        default_id
    })
}

#[cfg(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn"))]
fn roi_player_class(class_names: &[String]) -> Option<u16> {
    resolve_class_id(class_names, &["person"]).or_else(|| class_names.is_empty().then_some(0))
}

#[cfg(all(
    test,
    any(feature = "ort", feature = "tensorrt-native", feature = "ncnn")
))]
mod setup_tests;
