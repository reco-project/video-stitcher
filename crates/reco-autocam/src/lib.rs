//! Automatic camera control for reco.
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

pub use directors::{BallDirector, FieldDirector, TrackingMode};
pub use roi_filter::RoiFilteredDetector;
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use roi_filter::RoiFilteredGpuDetector;
#[cfg(target_os = "macos")]
pub use roi_filter::RoiFilteredMetalDetector;
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
/// Returns `true` if detection was successfully activated, `false` if
/// detection could not be initialized (the session remains usable without
/// autocam in that case).
///
/// # Arguments
///
/// * `session` - The stitch session to attach detection and direction to.
/// * `model_path` - Path to a YOLO ONNX model (or `.mlmodelc` on macOS).
/// * `input_width`, `input_height` - Raw camera frame dimensions.
/// * `fps` - Video frame rate (used for director timing).
/// * `use_zero_copy` - Whether the pipeline is running in zero-copy mode.
/// * `detection_interval` - Run detection every N frames (1 = every frame).
/// * `lead_time` - Director lookahead in seconds (CPU path only).
/// * `tracking_mode` - Which director to use (Ball or Field).
/// * `field_roi` - Optional playing field ROI polygons for filtering detections.
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
) -> Result<bool, Box<dyn std::error::Error>> {
    let mut detection_active = false;

    // Load class names from the model to resolve label -> class_id for directors.
    // Skip ORT session creation for .engine files (TRT engines aren't ONNX).
    let class_names = if model_path.ends_with(".engine") {
        Vec::new() // Labels come from sidecar .labels file instead
    } else {
        match reco_detect::create_ort_session(Path::new(model_path), Vec::new()) {
            Ok((_, _, names)) => names,
            Err(e) => {
                log::warn!("Could not read model labels: {e}, using COCO defaults");
                Vec::new()
            }
        }
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
        ) {
            Ok(Some(trt_det)) => {
                let detector: Box<dyn reco_core::detector::GpuDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        Box::new(RoiFilteredGpuDetector::new(Box::new(trt_det), roi))
                    } else {
                        Box::new(trt_det)
                    };
                session.set_gpu_detector(detector);
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
        match OrtGpuDetector::try_new(model_path, input_width, input_height, 0.10, Vec::new()) {
            Ok(Some(gpu_det)) => {
                let detector: Box<dyn reco_core::detector::GpuDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        Box::new(RoiFilteredGpuDetector::new(Box::new(gpu_det), roi))
                    } else {
                        Box::new(gpu_det)
                    };
                session.set_gpu_detector(detector);
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
                let detector: Box<dyn reco_core::detector::MetalDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        Box::new(RoiFilteredMetalDetector::new(Box::new(metal_det), roi))
                    } else {
                        Box::new(metal_det)
                    };
                session.set_metal_detector(detector);
                detection_active = true;
                log::info!("Autocam: Metal YOLO ball tracking enabled (model: {model_path})");
            }
            Err(e) => {
                log::warn!("Autocam: Metal detector init failed ({e}), ball tracking disabled");
            }
        }
    }

    if !use_zero_copy {
        let yolo = CpuYoloDetector::from_file(model_path)?;
        let detector: Box<dyn reco_core::detector::Detector> = if let Some(roi) = effective_roi {
            Box::new(RoiFilteredDetector::new(Box::new(yolo), roi))
        } else {
            Box::new(yolo)
        };
        session.set_detector(detector);
        detection_active = true;
        log::info!("Autocam: YOLO ball tracking enabled (model: {model_path})");
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
                let d = FieldDirector::new(fps)
                    .with_ball_class_id(ball_id)
                    .with_player_class_id(person_id);
                log::info!("Tracking mode: field (ball + players)");
                Box::new(d)
            }
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
    default_id
}
