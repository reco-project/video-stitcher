//! RANSAC fundamental matrix estimation.
//!
//! Implements the 8-point algorithm with Hartley normalization for
//! fundamental matrix estimation, wrapped in a RANSAC loop with
//! Sampson error scoring. Replaces the `inlier` crate dependency
//! to eliminate 301 transitive dependencies (including C++ FFI via
//! `usearch` that doesn't compile on Windows).
//!
//! ## Algorithm
//!
//! 1. **RANSAC loop**: randomly sample 8 point correspondences
//! 2. **8-point algorithm**: solve for F via SVD with Hartley normalization
//! 3. **Sampson error**: score each model by symmetric transfer error
//! 4. **Inlier selection**: keep points with Sampson error < threshold

use nalgebra::{Matrix3, SVD};
use rand::SeedableRng;
use rand::prelude::IndexedRandom;

/// Estimate the fundamental matrix and return inlier indices.
///
/// Uses RANSAC with the normalized 8-point algorithm and Sampson error.
///
/// - `pts1`, `pts2`: matched 2D point coordinates (same length)
/// - `threshold`: Sampson error threshold for inlier classification
/// - `max_iterations`: maximum RANSAC iterations (0 = auto from confidence)
///
/// Returns inlier indices, or an error if estimation fails.
pub fn ransac_fundamental(
    pts1: &[[f64; 2]],
    pts2: &[[f64; 2]],
    threshold: f64,
    max_iterations: usize,
) -> Result<Vec<usize>, &'static str> {
    let n = pts1.len();
    if n != pts2.len() {
        return Err("point arrays must have equal length");
    }
    if n < 8 {
        return Err("need at least 8 point pairs");
    }

    let threshold_sq = threshold * threshold;
    let max_iters = if max_iterations == 0 {
        2000
    } else {
        max_iterations
    };

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let indices: Vec<usize> = (0..n).collect();

    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_score = 0usize;

    for _ in 0..max_iters {
        // Sample 8 random indices
        let sample: Vec<usize> = indices.choose_multiple(&mut rng, 8).copied().collect();

        // Estimate F from the 8-point sample
        let f = match estimate_fundamental_8pt(pts1, pts2, &sample) {
            Some(f) => f,
            None => continue,
        };

        // Score: count inliers using Sampson error
        let mut inliers = Vec::new();
        for i in 0..n {
            let err = sampson_error(&f, &pts1[i], &pts2[i]);
            if err < threshold_sq {
                inliers.push(i);
            }
        }

        if inliers.len() > best_score {
            best_score = inliers.len();
            best_inliers = inliers;

            // Early termination: if we have >80% inliers, good enough
            if best_score * 5 > n * 4 {
                break;
            }
        }
    }

    if best_inliers.is_empty() {
        return Err("RANSAC found no inliers");
    }

    // Refine F using all inliers (least-squares on the full inlier set)
    if let Some(f_refined) = estimate_fundamental_8pt(pts1, pts2, &best_inliers) {
        // Re-evaluate inliers with the refined model
        let mut refined_inliers = Vec::new();
        for i in 0..n {
            let err = sampson_error(&f_refined, &pts1[i], &pts2[i]);
            if err < threshold_sq {
                refined_inliers.push(i);
            }
        }
        if refined_inliers.len() >= best_inliers.len() {
            best_inliers = refined_inliers;
        }
    }

    Ok(best_inliers)
}

/// Normalized 8-point algorithm for fundamental matrix estimation.
///
/// Implements Hartley's normalization (translate to centroid, scale to
/// sqrt(2) mean distance) for numerical stability, then solves via SVD.
fn estimate_fundamental_8pt(
    pts1: &[[f64; 2]],
    pts2: &[[f64; 2]],
    sample: &[usize],
) -> Option<Matrix3<f64>> {
    let n = sample.len();
    if n < 8 {
        return None;
    }

    // Compute centroids
    let (mut cx1, mut cy1, mut cx2, mut cy2) = (0.0, 0.0, 0.0, 0.0);
    for &i in sample {
        cx1 += pts1[i][0];
        cy1 += pts1[i][1];
        cx2 += pts2[i][0];
        cy2 += pts2[i][1];
    }
    let inv_n = 1.0 / n as f64;
    cx1 *= inv_n;
    cy1 *= inv_n;
    cx2 *= inv_n;
    cy2 *= inv_n;

    // Compute mean distances from centroid
    let (mut d1, mut d2) = (0.0, 0.0);
    for &i in sample {
        let dx1 = pts1[i][0] - cx1;
        let dy1 = pts1[i][1] - cy1;
        let dx2 = pts2[i][0] - cx2;
        let dy2 = pts2[i][1] - cy2;
        d1 += (dx1 * dx1 + dy1 * dy1).sqrt();
        d2 += (dx2 * dx2 + dy2 * dy2).sqrt();
    }
    d1 *= inv_n;
    d2 *= inv_n;

    if d1 < 1e-10 || d2 < 1e-10 {
        return None;
    }

    let s1 = std::f64::consts::SQRT_2 / d1;
    let s2 = std::f64::consts::SQRT_2 / d2;

    // Normalization transforms
    let t1 = Matrix3::new(s1, 0.0, -s1 * cx1, 0.0, s1, -s1 * cy1, 0.0, 0.0, 1.0);
    let t2 = Matrix3::new(s2, 0.0, -s2 * cx2, 0.0, s2, -s2 * cy2, 0.0, 0.0, 1.0);

    // Build coefficient matrix A (n x 9)
    let mut a_data = vec![0.0f64; n * 9];
    for (row, &i) in sample.iter().enumerate() {
        let x1 = (pts1[i][0] - cx1) * s1;
        let y1 = (pts1[i][1] - cy1) * s1;
        let x2 = (pts2[i][0] - cx2) * s2;
        let y2 = (pts2[i][1] - cy2) * s2;

        let base = row * 9;
        a_data[base] = x2 * x1;
        a_data[base + 1] = x2 * y1;
        a_data[base + 2] = x2;
        a_data[base + 3] = y2 * x1;
        a_data[base + 4] = y2 * y1;
        a_data[base + 5] = y2;
        a_data[base + 6] = x1;
        a_data[base + 7] = y1;
        a_data[base + 8] = 1.0;
    }

    // A^T A (9x9) - solve for null space
    let mut ata = [0.0f64; 81];
    for i in 0..9 {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a_data[k * 9 + i] * a_data[k * 9 + j];
            }
            ata[i * 9 + j] = sum;
            ata[j * 9 + i] = sum;
        }
    }

    let ata_mat = nalgebra::DMatrix::from_row_slice(9, 9, &ata);
    let svd = SVD::new(ata_mat, false, true);
    let vt = svd.v_t?;
    let v = vt.transpose();

    // Last column = smallest singular value's vector
    let f_vec: Vec<f64> = (0..9).map(|i| v[(i, 8)]).collect();
    if f_vec.iter().any(|x| x.is_nan()) {
        return None;
    }

    // Reshape to 3x3
    let f_norm = Matrix3::new(
        f_vec[0], f_vec[1], f_vec[2], f_vec[3], f_vec[4], f_vec[5], f_vec[6], f_vec[7], f_vec[8],
    );

    // Denormalize: F = T2^T * F_norm * T1
    let f = t2.transpose() * f_norm * t1;

    // Enforce rank-2 constraint via SVD
    let svd_f = SVD::new(f, true, true);
    let u = svd_f.u?;
    let vt = svd_f.v_t?;
    let mut sigma = svd_f.singular_values;
    sigma[2] = 0.0; // force rank 2
    let f_rank2 = u * nalgebra::Matrix3::from_diagonal(&sigma) * vt;

    // Normalize so ||F|| = 1
    let norm = f_rank2.norm();
    if norm < 1e-15 {
        return None;
    }

    Some(f_rank2 / norm)
}

/// Sampson error (first-order approximation of symmetric transfer error).
///
/// For a point correspondence (x1, x2) and fundamental matrix F:
/// error = (x2^T F x1)^2 / (||F x1||^2_2first2 + ||F^T x2||^2_2first2)
///
/// where ||.||^2_2first2 means the sum of squares of the first two elements.
fn sampson_error(f: &Matrix3<f64>, p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let x1 = nalgebra::Vector3::new(p1[0], p1[1], 1.0);
    let x2 = nalgebra::Vector3::new(p2[0], p2[1], 1.0);

    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;

    let x2tfx1 = x2.dot(&fx1);
    let numerator = x2tfx1 * x2tfx1;

    let denominator = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];

    if denominator < 1e-15 {
        return f64::MAX;
    }

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ransac_with_perfect_points() {
        // Create a known fundamental matrix scenario:
        // Two cameras with a horizontal baseline, points on a plane
        let pts1: Vec<[f64; 2]> = vec![
            [100.0, 100.0],
            [200.0, 100.0],
            [300.0, 200.0],
            [150.0, 300.0],
            [400.0, 150.0],
            [250.0, 250.0],
            [350.0, 350.0],
            [450.0, 200.0],
            [180.0, 180.0],
            [320.0, 120.0],
        ];

        // Simulate a slight horizontal shift + perspective change
        let pts2: Vec<[f64; 2]> = pts1
            .iter()
            .map(|p| [p[0] + 20.0 + p[0] * 0.02, p[1] + p[0] * 0.01])
            .collect();

        let result = ransac_fundamental(&pts1, &pts2, 5.0, 1000);
        assert!(result.is_ok());
        let inliers = result.unwrap();
        // With perfect synthetic data, all points should be inliers
        assert!(inliers.len() >= 8, "got {} inliers", inliers.len());
    }

    #[test]
    fn test_ransac_with_outliers() {
        // Create non-coplanar points with perspective distortion
        let pts1: Vec<[f64; 2]> = (0..20)
            .map(|i| {
                let x = 100.0 + (i % 5) as f64 * 80.0;
                let y = 100.0 + (i / 5) as f64 * 80.0;
                [x, y]
            })
            .collect();
        let mut pts2: Vec<[f64; 2]> = pts1
            .iter()
            .map(|p| {
                // Simulate epipolar geometry: horizontal shift + slight perspective
                let depth = 1.0 + p[0] * 0.001 + p[1] * 0.0005;
                [p[0] + 20.0 / depth, p[1] + 5.0 * p[0] * 0.001 / depth]
            })
            .collect();

        // Add 5 gross outliers (25%)
        for item in pts2.iter_mut().take(5) {
            item[0] += 500.0;
            item[1] -= 300.0;
        }

        let result = ransac_fundamental(&pts1, &pts2, 5.0, 2000);
        assert!(result.is_ok());
        let inliers = result.unwrap();
        // Should find most of the 15 good points
        assert!(
            inliers.len() >= 10,
            "expected at least 10 inliers from 15 good points, got {}",
            inliers.len()
        );
    }

    #[test]
    fn test_too_few_points() {
        let pts1 = vec![[1.0, 2.0]; 5];
        let pts2 = vec![[3.0, 4.0]; 5];
        assert!(ransac_fundamental(&pts1, &pts2, 1.0, 100).is_err());
    }
}
