//! Automatic camera control for reco.
//!
//! # Safety policy
//!
//! This crate contains zero `unsafe` code by construction. All platform
//! / FFI boundaries live in reco-core (wgpu, zero-copy) or reco-detect
//! (ORT, CUDA), so the intelligence layer can stay in safe Rust.
//! CI enforces this via `#![forbid(unsafe_code)]` below; introducing
//! `unsafe` here requires a lint override + an explicit PR discussion.

#![forbid(unsafe_code)]

//!
//! This crate provides detection and direction for sports camera automation:
//!
//! - [`CpuYoloDetector`] - ONNX-based YOLO object detection on raw camera frames
//! - [`BallDirector`] - Ball-following director with plausibility rejection
//! - [`FieldDirector`] - Ball + player tracking for robust football coverage
//! - [`SmoothedDirector`] - Decorator that adds One Euro trajectory smoothing
//! - [`TrackingMode`] - Selects which director to use
//!
//! # Usage
//!
//! ```rust,no_run
//! use reco_autocam::{CpuYoloDetector, BallDirector, SmoothedDirector};
//!
//! let detector = CpuYoloDetector::from_file("ball_v0.onnx")?;
//! let director = BallDirector::new(30.0); // fps
//! let smoothed = SmoothedDirector::new(Box::new(director), 30.0, 15);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod directors;
mod roi_filter;
mod smoother;

// Re-export detector types from reco-detect for backwards compatibility.
pub use reco_detect::CpuYoloDetector;
#[cfg(target_os = "macos")]
pub use reco_detect::MetalYoloDetector;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use reco_detect::OrtGpuDetector;
#[cfg(feature = "tensorrt-native")]
pub use reco_detect::TrtGpuDetector;

pub use directors::{BallDirector, FieldDirector, SweepDirector, TrackingMode};
pub use roi_filter::RoiFilteredDetector;
// `RoiFilteredGpuDetector` and `RoiFilteredMetalDetector` were
// deleted: the unified `RoiFilteredDetector` covers every residency
// because it wraps `Box<dyn UnifiedDetector>`.
pub use smoother::{SmoothedDirector, TrajectorySmoother};

use std::path::Path;

use reco_core::calibration::FieldRoi;
use reco_core::session::StitchSession;

/// Set up automatic camera control on a [`StitchSession`].
///
/// Configures the appropriate detector (CPU, GPU/CUDA, or Metal) based on
/// the current platform and zero-copy mode, then attaches a [`BallDirector`]
/// for ball-following camera automation.
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
pub fn setup_autocam(
    session: &mut StitchSession,
    model_path: &str,
    input_width: u32,
    input_height: u32,
    fps: f32,
    use_zero_copy: bool,
    detection_interval: u64,
    lead_time: f64,
    tracking_mode: TrackingMode,
    field_roi: Option<&FieldRoi>,
    is_10bit: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut detection_active = false;

    // Load class names from the model to resolve label -> class_id for directors.
    // Skip ORT session creation for non-ONNX models (.engine files,
    // NCNN model directories). ORT can only parse .onnx files.
    let is_onnx = model_path.ends_with(".onnx");
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

    // Check if ROI filtering should be applied.
    // A FieldRoi is only meaningful if at least one polygon has >= 3 vertices.
    let effective_roi = field_roi
        .filter(|roi| roi.left.len() >= 3 || roi.right.len() >= 3)
        .cloned();
    if effective_roi.is_some() {
        log::info!("Autocam: field ROI filtering enabled");
    }

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
                        Box::new(RoiFilteredDetector::new(Box::new(trt_det), roi))
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
    #[cfg(any(target_os = "linux", target_os = "windows"))]
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
                        Box::new(RoiFilteredDetector::new(Box::new(gpu_det), roi))
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

    #[cfg(target_os = "macos")]
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
                        Box::new(RoiFilteredDetector::new(Box::new(metal_det), roi))
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
                        Box::new(RoiFilteredDetector::new(Box::new(ncnn_det), roi))
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
    if !detection_active && !use_zero_copy {
        let yolo = CpuYoloDetector::from_file(model_path)?;
        let detector: Box<dyn reco_core::detector::UnifiedDetector> =
            if let Some(roi) = effective_roi {
                Box::new(RoiFilteredDetector::new(Box::new(yolo), roi))
            } else {
                Box::new(yolo)
            };
        session.set_detector(detector);
        detection_active = true;
        log::info!("Autocam: YOLO ball tracking enabled (model: {model_path})");
    }

    // Sweep director doesn't need detection - attach it regardless.
    if tracking_mode == TrackingMode::Sweep {
        log::info!("Tracking mode: sweep (debug, no AI)");
        let director = Box::new(directors::SweepDirector::new(0.8, 10.0));
        session.set_director(director);
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

        let director: Box<dyn reco_core::director::Director> = match tracking_mode {
            TrackingMode::Ball => {
                let mut d = BallDirector::new(fps).with_class_id(ball_id);
                if detection_interval > 1 {
                    d.set_detection_interval(detection_interval as u32);
                }
                log::info!("Tracking mode: ball");
                Box::new(d)
            }
            TrackingMode::Field => {
                let d = FieldDirector::new()
                    .with_ball_class_id(ball_id)
                    .with_player_class_id(person_id);
                log::info!("Tracking mode: field (ball + players)");
                Box::new(d)
            }
            TrackingMode::Sweep => unreachable!("handled above"),
        };

        let lookahead = if lead_time > 0.0 && !use_zero_copy {
            let frames = (fps as f64 * lead_time).round() as usize;
            if frames > 0 {
                session.set_lookahead(frames);
                log::info!("Director lead time: {lead_time:.1}s ({frames} frames)");
            }
            frames
        } else {
            0
        };

        let smoothed = SmoothedDirector::new(director, fps, lookahead);
        session.set_director(Box::new(smoothed));
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
