//! KB4 fisheye lens model - CPU-side distortion and undistortion.
//!
//! Implements the Kannala-Brandt 4-coefficient fisheye model used by
//! Gyroflow and OpenCV:
//!
//! ```text
//! θ_d = θ × (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
//! ```
//!
//! The primary use case is CPU-side frame undistortion for the
//! calibration pipeline, which needs to match features in rectilinear
//! (undistorted) space.

use crate::calibration::CameraParams;

/// Apply the forward KB4 distortion model.
///
/// Given an undistorted radius `r` in normalized camera coordinates,
/// returns the distorted radius `r_d` and the scale factor `theta_d / r`.
///
/// Forward formula:
/// ```text
/// theta = atan(r)
/// theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
/// scale = theta_d / r
/// ```
#[inline]
fn kb4_forward_scale(r: f64, d: &[f64; 4]) -> f64 {
    if r < 1e-10 {
        return 1.0;
    }
    let theta = r.atan();
    let t2 = theta * theta;
    let theta_d =
        theta * (1.0 + d[0] * t2 + d[1] * t2 * t2 + d[2] * t2 * t2 * t2 + d[3] * t2 * t2 * t2 * t2);
    theta_d / r
}

/// Undistort a grayscale frame using the KB4 fisheye model.
///
/// For each output (undistorted) pixel, computes the corresponding
/// source pixel in the distorted input using the forward KB4 mapping,
/// then samples with bilinear interpolation. This matches what the
/// GPU shader does at render time.
///
/// The lens profile intrinsics are automatically scaled to the frame's
/// actual resolution (the profile may have been calibrated at a
/// different resolution with the same aspect ratio).
///
/// # Arguments
/// * `data` - Row-major grayscale pixel data (1 byte per pixel)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `params` - Camera intrinsics and KB4 distortion coefficients
///
/// # Returns
/// A new pixel buffer of the same dimensions with the undistorted image.
pub fn undistort_gray(data: &[u8], width: u32, height: u32, params: &CameraParams) -> Vec<u8> {
    let w = width as f64;
    let h = height as f64;

    // Scale original intrinsics for source pixel lookup
    let sx = w / params.width as f64;
    let sy = h / params.height as f64;
    let fx = params.fx * sx;
    let fy = params.fy * sy;
    let cx = params.cx * sx;
    let cy = params.cy * sy;

    // Output intrinsics: make the undistorted viewport match the GPU
    // plane exactly.  The shader applies `uv * 2.0 - 0.5` before KB4,
    // doubling the coordinate range.  Geometrically this is equivalent
    // to a virtual pinhole camera at distance d = fx/(2·w) from a
    // plane of half-width 0.5:
    //
    //   tan(half_fov) = 0.5 / d = w / fx
    //   out_fx = (w/2) / tan(half_fov) = fx / 2
    //   out_cx = (w + 2·cx) / 4   (preserves off-center optical axis)
    //
    // This ensures linear normalization of pixel coords to [-0.5, 0.5]
    // gives correct plane coordinates for the calibration optimizer.
    let out_fx = fx / 2.0;
    let out_fy = fy / 2.0;
    let out_cx = (w + 2.0 * cx) / 4.0;
    let out_cy = (h + 2.0 * cy) / 4.0;

    let mut out = vec![0u8; (width * height) as usize];

    for out_y in 0..height {
        for out_x in 0..width {
            // Ray direction from FOV-fitted output intrinsics
            let x = (out_x as f64 - out_cx) / out_fx;
            let y = (out_y as f64 - out_cy) / out_fy;
            let r = (x * x + y * y).sqrt();

            let scale = kb4_forward_scale(r, &params.d);

            // Source pixel in the distorted image using original intrinsics
            let src_x = fx * x * scale + cx;
            let src_y = fy * y * scale + cy;

            let idx = (out_y * width + out_x) as usize;
            out[idx] = bilinear_sample(data, width, height, src_x, src_y);
        }
    }

    out
}

/// Map a pixel position in the undistorted output image back to
/// the corresponding pixel in the original distorted (fisheye) image.
///
/// This is the same mapping that `undistort_gray` computes per-pixel
/// for image resampling, but exposed for individual point lookups.
/// Useful when features are detected in undistorted space but their
/// positions need to be in distorted (original image) coordinates.
///
/// # Arguments
/// * `out_x`, `out_y` - Pixel position in the undistorted image
/// * `width`, `height` - Frame dimensions
/// * `params` - Camera intrinsics and KB4 distortion coefficients
///
/// # Returns
/// `(src_x, src_y)` - Corresponding pixel in the distorted image.
pub fn undistorted_to_distorted(
    out_x: f64,
    out_y: f64,
    width: u32,
    height: u32,
    params: &CameraParams,
) -> (f64, f64) {
    let w = width as f64;
    let h = height as f64;

    // Scale original intrinsics to frame resolution
    let sx = w / params.width as f64;
    let sy = h / params.height as f64;
    let fx = params.fx * sx;
    let fy = params.fy * sy;
    let cx = params.cx * sx;
    let cy = params.cy * sy;

    // Must match undistort_gray output intrinsics (plane-fitted FOV)
    let out_fx = fx / 2.0;
    let out_fy = fy / 2.0;
    let out_cx = (w + 2.0 * cx) / 4.0;
    let out_cy = (h + 2.0 * cy) / 4.0;

    // Ray direction from FOV-fitted output intrinsics
    let x = (out_x - out_cx) / out_fx;
    let y = (out_y - out_cy) / out_fy;
    let r = (x * x + y * y).sqrt();

    let scale = kb4_forward_scale(r, &params.d);

    // Source pixel in the distorted image
    (fx * x * scale + cx, fy * y * scale + cy)
}

/// Bilinear interpolation sample from a grayscale image.
#[inline]
fn bilinear_sample(data: &[u8], w: u32, h: u32, x: f64, y: f64) -> u8 {
    if x < 0.0 || y < 0.0 || x >= (w - 1) as f64 || y >= (h - 1) as f64 {
        return 0;
    }

    let x0 = x as u32;
    let y0 = y as u32;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let p00 = data[(y0 * w + x0) as usize] as f64;
    let p10 = data[(y0 * w + x0 + 1) as usize] as f64;
    let p01 = data[((y0 + 1) * w + x0) as usize] as f64;
    let p11 = data[((y0 + 1) * w + x0 + 1) as usize] as f64;

    let val = p00 * (1.0 - fx) * (1.0 - fy)
        + p10 * fx * (1.0 - fy)
        + p01 * (1.0 - fx) * fy
        + p11 * fx * fy;

    val.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kb4_forward_scale_zero_radius() {
        let d = [0.034, 0.068, -0.074, 0.030];
        assert!((kb4_forward_scale(0.0, &d) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kb4_forward_scale_less_than_one_for_barrel_distortion() {
        // For typical fisheye lenses (small positive k1), scale < 1 at large
        // radii because atan(r)/r < 1 dominates the polynomial correction.
        let d = [0.1, 0.0, 0.0, 0.0];
        let s = kb4_forward_scale(0.5, &d);
        assert!(s < 1.0, "barrel distortion scale should be < 1: got {s}");
    }

    #[test]
    fn undistort_center_pixel_unchanged() {
        // With plane-fitted intrinsics, out_cx = (w + 2*cx) / 4.
        // For cx = w/2 this gives out_cx = w/2 (exact integer with even dims).
        // At the output center, ray = (0,0), scale = 1, src = (cx, cy).
        let params = CameraParams {
            width: 100,
            height: 100,
            fx: 50.0,
            fy: 50.0,
            cx: 50.0,
            cy: 50.0,
            d: [0.1, 0.05, -0.03, 0.01],
        };

        let mut data = vec![0u8; 100 * 100];
        data[50 * 100 + 50] = 255; // bright pixel at optical center

        let result = undistort_gray(&data, 100, 100, &params);
        assert_eq!(
            result[50 * 100 + 50],
            255,
            "center pixel should be unchanged"
        );
    }
}
