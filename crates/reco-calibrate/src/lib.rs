//! Stereo camera calibration for reco.
//!
//! Computes the relative positioning of two camera planes by detecting
//! features in overlapping footage and optimizing placement parameters
//! to minimize reprojection error between matched points.
//!
//! ## Pipeline
//!
//! ```text
//! Frame pairs -> GPU Undistort -> AKAZE Detect -> Descriptor Match
//!   -> Spatial + RANSAC Filter -> Nelder-Mead Optimizer -> PlaneLayout
//! ```
//!
//! Each stage also has a trait interface ([`traits`]) with default
//! implementations in [`defaults`] for standalone use.
//!
//! ## Full pipeline usage
//!
//! ```ignore
//! use reco_calibrate::{calibrate, CalibrationConfig};
//! use reco_core::calibration::CameraParams;
//!
//! let result = calibrate(&gpu, &frames, &left_params, &right_params, &CalibrationConfig::default())?;
//! println!("Confidence: {:.1}%", result.confidence * 100.0);
//! ```

/// Create a tracing span guard (no-op when `profiling` feature is disabled).
#[cfg(feature = "profiling")]
macro_rules! profile_scope {
    ($name:expr) => {
        let _span = tracing::info_span!($name).entered();
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! profile_scope {
    ($name:expr) => {};
}

pub(crate) mod akaze;
pub mod audio_sync;
pub mod defaults;
pub mod error;
pub mod features;
pub mod filter;
pub mod geometry;
pub mod lens_database;
pub mod optimizer;
pub mod sampling;
pub mod telemetry;
pub mod traits;
pub mod types;

pub use defaults::{
    AkazeDetector, HammingMatcher, RawReprojectionCost, SeamWeightedCost, YDisparityFilter,
};
pub use error::CalibrateError;
pub use traits::{
    CalibrationOptimizer, CostFunction, FeatureDetector, FeatureMatcher, PointFilter,
};
pub use types::{CalibrationConfig, CalibrationResult, GrayFrame, YuvFrame};

use reco_core::calibration::{CameraParams, MatchCalibration};
use reco_core::gpu::GpuContext;
use reco_core::undistort::GpuUndistort;

use types::{FrameMatches, MatchedPoint};

/// Process an undistorted RGBA frame pair through the feature matching pipeline.
///
/// Takes pre-undistorted RGBA data (from GPU phase) and runs AKAZE
/// detection, matching, and filtering. This function is thread-safe
/// and called in parallel via rayon.
#[allow(clippy::too_many_arguments)]
fn process_undistorted_pair(
    left_rgba: &[u8],
    right_rgba: &[u8],
    lw: u32,
    lh: u32,
    rw: u32,
    rh: u32,
    frame_idx: usize,
    config: &CalibrationConfig,
) -> Option<FrameMatches> {
    profile_scope!("process_frame");
    let inner = config.spatial_x_inner as f32;
    let y_min = config.detect_y_min as f32;
    let y_max = config.detect_y_max as f32;
    let left_region = features::DetectRegion {
        x_min: config.spatial_x_threshold as f32,
        x_max: 1.0 - inner,
        y_min,
        y_max,
    };
    let right_region = features::DetectRegion {
        x_min: inner,
        x_max: 1.0 - config.spatial_x_threshold as f32,
        y_min,
        y_max,
    };

    // Detect features on left and right concurrently (independent CPU work)
    let ((kp_left, desc_left), (kp_right, desc_right)) = rayon::join(
        || {
            profile_scope!("akaze_detect_left");
            features::detect(
                left_rgba,
                lw,
                lh,
                Some(left_region),
                config.max_keypoints,
                config.akaze_threshold,
            )
        },
        || {
            profile_scope!("akaze_detect_right");
            features::detect(
                right_rgba,
                rw,
                rh,
                Some(right_region),
                config.max_keypoints,
                config.akaze_threshold,
            )
        },
    );

    log::debug!(
        "frame {frame_idx}: {} left keypoints, {} right keypoints",
        kp_left.len(),
        kp_right.len()
    );

    if kp_left.is_empty() || kp_right.is_empty() {
        log::warn!("frame {frame_idx}: no keypoints in one or both images");
        return None;
    }

    // Match descriptors with Lowe's ratio test
    let raw_matches = features::match_descriptors(&desc_left, &desc_right, config.lowe_ratio);
    let post_ratio_test = raw_matches.len();

    if raw_matches.len() < config.min_matches {
        log::debug!(
            "frame {frame_idx}: only {} matches after ratio test (need {})",
            raw_matches.len(),
            config.min_matches
        );
        return None;
    }

    // Spatial overlap filter
    let spatial_matches =
        filter::spatial_filter(&raw_matches, &kp_left, &kp_right, lw, lh, rw, rh, config);
    let post_spatial_filter = spatial_matches.len();

    // RANSAC outlier rejection
    let inlier_indices = match filter::ransac_filter(&spatial_matches, &kp_left, &kp_right, config)
    {
        Ok(indices) => indices,
        Err(e) => {
            log::debug!("frame {frame_idx}: RANSAC failed: {e}");
            return None;
        }
    };
    let post_ransac = inlier_indices.len();

    if post_ransac < config.min_matches {
        log::debug!(
            "frame {frame_idx}: only {} inliers after RANSAC (need {})",
            post_ransac,
            config.min_matches
        );
        return None;
    }

    // Normalize surviving matches to plane coordinates.
    //
    // Features were detected on the undistorted image (using original
    // intrinsics), which maps 1:1 to the GPU shader's plane UV space.
    // Linear normalization to [-0.5, 0.5] gives plane coordinates
    // directly - no KB4 remapping needed.
    //
    // CRITICAL: Apply the left/right swap from v1 (processing.py:693).
    // Right camera points -> left plane (x-plane) in optimizer space.
    // Left camera points -> right plane (z-plane) in optimizer space.
    let points: Vec<MatchedPoint> = inlier_indices
        .iter()
        .map(|&i| {
            let m = &spatial_matches[i];
            let lp = &kp_left[m.left_idx];
            let rp = &kp_right[m.right_idx];

            // Swap: right pixel -> left plane (x-plane), left pixel -> right plane (z-plane)
            MatchedPoint {
                left: geometry::normalize_to_plane(rp.x as f64, rp.y as f64, rw, rh),
                right: geometry::normalize_to_plane(lp.x as f64, lp.y as f64, lw, lh),
                // Store normalized pixel x for seam-proximity weighting
                left_pixel_nx: rp.x as f64 / rw as f64,
                right_pixel_nx: lp.x as f64 / lw as f64,
            }
        })
        .collect();

    Some(FrameMatches {
        points,
        keypoints_left: kp_left.len(),
        keypoints_right: kp_right.len(),
        raw_matches: desc_left.len().min(desc_right.len()),
        post_ratio_test,
        post_spatial_filter,
        post_ransac,
    })
}

/// Run the full calibration pipeline.
///
/// Takes pre-extracted YUV frame pairs (left, right) along with camera
/// intrinsics from lens profiles. GPU-undistorts each frame to rectilinear
/// RGBA before detecting AKAZE features.
///
/// # Errors
///
/// Returns [`CalibrateError::NoUsableFrames`] if no frame pairs produce
/// enough matches, or [`CalibrateError::OptimizerFailed`] if all
/// optimization iterations fail.
pub fn calibrate(
    gpu: &GpuContext,
    frames: &[(YuvFrame, YuvFrame)],
    left_params: &CameraParams,
    right_params: &CameraParams,
    config: &CalibrationConfig,
) -> Result<CalibrationResult, CalibrateError> {
    // Create GPU undistort pipelines for each camera's resolution
    let (lw, lh) = if let Some((left, _)) = frames.first() {
        (left.width, left.height)
    } else {
        return Err(CalibrateError::NoUsableFrames);
    };
    let (rw, rh) = if let Some((_, right)) = frames.first() {
        (right.width, right.height)
    } else {
        return Err(CalibrateError::NoUsableFrames);
    };
    let left_undistort = GpuUndistort::new(gpu, lw, lh);
    let right_undistort = GpuUndistort::new(gpu, rw, rh);

    // Phase 1: GPU undistort all frames (sequential - shared GPU state)
    let undistorted: Vec<(Vec<u8>, Vec<u8>)> = {
        profile_scope!("gpu_undistort");
        frames
            .iter()
            .map(|(left, right)| {
                let l_rgba = left_undistort.undistort(gpu, &left.y, &left.u, &left.v, left_params);
                let r_rgba =
                    right_undistort.undistort(gpu, &right.y, &right.u, &right.v, right_params);
                (l_rgba, r_rgba)
            })
            .collect()
    };

    // Phase 2: AKAZE detect + match + filter (parallel - CPU bound)
    let per_frame: Vec<Option<FrameMatches>> = {
        profile_scope!("akaze_parallel");
        use rayon::prelude::*;
        undistorted
            .par_iter()
            .enumerate()
            .map(|(i, (left_rgba, right_rgba))| {
                process_undistorted_pair(left_rgba, right_rgba, lw, lh, rw, rh, i, config)
            })
            .collect()
    };

    // Collect all successful frame matches
    let successful_frames: Vec<FrameMatches> = per_frame.into_iter().flatten().collect();

    if successful_frames.is_empty() {
        return Err(CalibrateError::NoUsableFrames);
    }

    let frames_used = successful_frames.len();
    log::info!(
        "{frames_used}/{} frame pairs produced matches",
        frames.len()
    );

    // Accumulate all matched points across frames
    let all_points: Vec<MatchedPoint> = successful_frames
        .iter()
        .flat_map(|fm| fm.points.iter().copied())
        .collect();

    let total_matches = all_points.len();
    log::info!("{total_matches} total matched points");

    // Log spatial distribution of matches for diagnostics
    if !all_points.is_empty() {
        let lx: Vec<f64> = all_points.iter().map(|p| p.left[0]).collect();
        let ly: Vec<f64> = all_points.iter().map(|p| p.left[1]).collect();
        let rx: Vec<f64> = all_points.iter().map(|p| p.right[0]).collect();
        let ry: Vec<f64> = all_points.iter().map(|p| p.right[1]).collect();
        log::info!(
            "x-plane range: x=[{:.3}, {:.3}] y=[{:.3}, {:.3}]",
            lx.iter().cloned().fold(f64::INFINITY, f64::min),
            lx.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ly.iter().cloned().fold(f64::INFINITY, f64::min),
            ly.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        log::info!(
            "z-plane range: x=[{:.3}, {:.3}] y=[{:.3}, {:.3}]",
            rx.iter().cloned().fold(f64::INFINITY, f64::min),
            rx.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ry.iter().cloned().fold(f64::INFINITY, f64::min),
            ry.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
    }

    if total_matches < config.min_matches {
        return Err(CalibrateError::InsufficientMatches {
            got: total_matches,
            min: config.min_matches,
        });
    }

    // Single-pass optimization on all points with trimmed cost.
    let (best_layout, best_residual) = {
        profile_scope!("optimizer");
        optimizer::optimize(&all_points, config)
    }
    .map_err(|e| {
        log::error!("optimization failed: {e}");
        e
    })?;

    let confidence = (total_matches as f64 / 50.0).min(1.0);

    // Log both metrics for diagnostic comparison
    let best_params = geometry::OptParams {
        x_ty: best_layout.x_ty,
        intersect: best_layout.intersect,
        cam_d: best_layout.camera_axis_offset,
        x_rz: best_layout.x_rz,
        z_rx: best_layout.z_rx,
        z_rz: None,
    };
    let total_reproj = geometry::reprojection_error(&all_points, &best_params);
    let angular_err = geometry::angular_error(&all_points, &best_params);
    let trimmed_err = geometry::trimmed_reprojection_error(&all_points, &best_params, 0.2);
    log::info!(
        "calibration complete: median_error={best_residual:.6}, trimmed={trimmed_err:.6}, \
         total_reproj={total_reproj:.6}, angular_error={angular_err:.6}, \
         confidence={confidence:.2}, z_rz={:.4}",
        best_layout.z_rz
    );

    let calibration = MatchCalibration {
        left: left_params.clone(),
        right: right_params.clone(),
        layout: best_layout,
    };

    Ok(CalibrationResult {
        calibration,
        total_matches,
        frames_used,
        residual_error: best_residual,
        confidence,
        per_frame: successful_frames,
    })
}
