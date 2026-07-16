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

#[cfg(feature = "gpu")]
pub mod preview;
#[cfg(feature = "gpu")]
pub mod undistort;

use crate::calibration::Lens;

pub(crate) use kb4::kb4_forward_scale;

/// Canonical Rust implementation of the KB4 polynomial.
///
/// Callers that need the polynomial (inverse_fisheye's Newton-Raphson
/// iteration, forward_fisheye, `kb4_forward_scale`) all delegate here
/// so the formula lives in exactly one place Rust-side. The WGSL
/// mirror at `shaders/fisheye.wgsl` lines 184-188 can't cross-
/// language-link; the `wgsl_kb4_matches_rust_kb4_on_theta_grid`
/// compute-dispatch test locks the two sides together numerically.
pub(crate) mod kb4 {
    /// Evaluate `θ_d = θ * (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)`.
    ///
    /// SYNC_WITH `shaders/fisheye.wgsl` lines 184-188. Any edit here
    /// must stay numerically in agreement with the WGSL fragment; the
    /// `wgsl_kb4_matches_rust_kb4_on_theta_grid` test will catch
    /// f32-vs-f64 drift within 1e-5.
    #[inline]
    pub fn theta_d(theta: f64, d: &[f64; 4]) -> f64 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        theta * (1.0 + d[0] * t2 + d[1] * t4 + d[2] * t6 + d[3] * t8)
    }

    /// Derivative of [`theta_d`] with respect to `theta`. Used by
    /// the Newton-Raphson step in `projection::inverse_fisheye`:
    /// ```text
    /// d/dθ θ_d = 1 + 3 k₁ θ² + 5 k₂ θ⁴ + 7 k₃ θ⁶ + 9 k₄ θ⁸
    /// ```
    #[inline]
    pub fn theta_d_prime(theta: f64, d: &[f64; 4]) -> f64 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        1.0 + 3.0 * d[0] * t2 + 5.0 * d[1] * t4 + 7.0 * d[2] * t6 + 9.0 * d[3] * t8
    }

    /// Forward KB4 scale factor `θ_d / r`. Returns `1.0` at the
    /// optical center (`r ≈ 0`) where the polynomial degenerates.
    #[inline]
    pub fn kb4_forward_scale(r: f64, d: &[f64; 4]) -> f64 {
        if r < 1e-10 {
            return 1.0;
        }
        theta_d(r.atan(), d) / r
    }

    /// Forward scale blended between the pinhole projection and full KB4
    /// by `correction` in `[0, 1]` - the CPU dual of the shader's
    /// `mix(theta, theta_d, correction)` per-pixel lerp (SYNC_WITH
    /// `fisheye.wgsl`). `0` = pure pinhole, `1` = full KB4.
    #[inline]
    pub fn kb4_forward_scale_with_correction(r: f64, d: &[f64; 4], correction: f64) -> f64 {
        if r < 1e-10 {
            return 1.0;
        }
        let pinhole = r.atan() / r;
        let full = kb4_forward_scale(r, d);
        pinhole + (full - pinhole) * correction
    }
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
pub fn undistort_gray(data: &[u8], width: u32, height: u32, params: &Lens) -> Vec<u8> {
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

            let scale = kb4_forward_scale(r, &params.distortion);

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
    params: &Lens,
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

    let scale = kb4_forward_scale(r, &params.distortion);

    // Source pixel in the distorted image
    (fx * x * scale + cx, fy * y * scale + cy)
}

/// Newton-Raphson iteration cap for [`distorted_to_undistorted`] - mirrors
/// `projection::inverse_fisheye`'s tuning (same KB4 polynomial, same
/// convergence behavior).
const MAX_ITERATIONS: usize = 20;
/// Convergence threshold (radians) for [`distorted_to_undistorted`].
const CONVERGENCE_EPS: f64 = 1e-10;

/// Map a pixel position in the distorted (fisheye) image back to the
/// corresponding pixel in the undistorted output image - the numeric
/// inverse of [`undistorted_to_distorted`].
///
/// The forward KB4 map (`theta -> theta_d`) has no closed-form inverse,
/// so this solves `theta_d(theta) = theta_d_target` via Newton-Raphson
/// (mirrors `projection::inverse_fisheye`'s iteration, but recovers the
/// point using this module's plane-fitted output intrinsics - see
/// [`undistorted_to_distorted`]'s doc comment for why those differ from
/// the stitch-plane convention `inverse_fisheye` uses - so the two
/// aren't interchangeable despite solving the same polynomial).
///
/// Returns `None` if Newton-Raphson fails to converge (a degenerate
/// input, e.g. a point at the lens's blind spot where the derivative
/// vanishes).
///
/// # Arguments
/// * `src_x`, `src_y` - Pixel position in the distorted (original) image
/// * `width`, `height` - Frame dimensions
/// * `params` - Camera intrinsics and KB4 distortion coefficients
///
/// # Returns
/// `(out_x, out_y)` - Corresponding pixel in the undistorted image.
pub fn distorted_to_undistorted(
    src_x: f64,
    src_y: f64,
    width: u32,
    height: u32,
    params: &Lens,
) -> Option<(f64, f64)> {
    let w = width as f64;
    let h = height as f64;

    // Scale original intrinsics to frame resolution (SYNC_WITH
    // undistorted_to_distorted).
    let sx = w / params.width as f64;
    let sy = h / params.height as f64;
    let fx = params.fx * sx;
    let fy = params.fy * sy;
    let cx = params.cx * sx;
    let cy = params.cy * sy;

    // Must match undistorted_to_distorted's output intrinsics
    // (plane-fitted FOV).
    let out_fx = fx / 2.0;
    let out_fy = fy / 2.0;
    let out_cx = (w + 2.0 * cx) / 4.0;
    let out_cy = (h + 2.0 * cy) / 4.0;

    // Normalized distorted-side ray. `undistorted_to_distorted` computes
    // `(src_x, src_y) = (fx*x*scale + cx, fy*y*scale + cy)` where
    // `scale = theta_d/r` and `r = |x, y|` - so `|dx, dy| = r*scale =
    // theta_d` directly, letting us skip straight to solving for theta.
    let dx = (src_x - cx) / fx;
    let dy = (src_y - cy) / fy;
    let theta_d = (dx * dx + dy * dy).sqrt();

    if theta_d < 1e-12 {
        // At the optical center - no distortion.
        return Some((out_cx, out_cy));
    }

    let k = params.distortion;
    let mut theta = theta_d; // initial guess
    let mut converged = false;
    for _ in 0..MAX_ITERATIONS {
        let f = kb4::theta_d(theta, &k) - theta_d;
        let f_prime = kb4::theta_d_prime(theta, &k);

        if f_prime.abs() < 1e-15 {
            return None; // degenerate
        }

        let delta = f / f_prime;
        theta -= delta;

        if delta.abs() < CONVERGENCE_EPS {
            converged = true;
            break;
        }
    }

    // Near frame edges/corners this module's extended-plane convention
    // (out_fx = fx/2, doubling the effective FOV per output pixel vs.
    // `projection::inverse_fisheye`'s stitch-plane convention) reaches
    // large theta values where Newton-Raphson can fail to converge
    // within MAX_ITERATIONS, or - worse - silently land on a theta that
    // never actually satisfies theta_d(theta) == theta_d_target (the
    // loop exits on hitting MAX_ITERATIONS, not on failure, so a caller
    // trusting the raw post-loop `theta` unconditionally can get a wildly
    // wrong point back with no signal anything went wrong). Verify the
    // root actually solves the equation before using it - a physically
    // meaningless theta (e.g. > ~89 degrees, beyond any real fisheye's
    // usable FOV) is rejected too, since it indicates runaway iteration
    // rather than a legitimate extreme-angle solution.
    if !converged
        || theta.abs() > std::f64::consts::FRAC_PI_2 * 0.99
        || (kb4::theta_d(theta, &k) - theta_d).abs() > 1e-6
    {
        return None;
    }

    let r = theta.tan();
    let scale = if theta.abs() < 1e-12 {
        1.0
    } else {
        theta_d / r
    };
    if !scale.is_finite() {
        return None;
    }

    let x = dx / scale;
    let y = dy / scale;

    Some((out_fx * x + out_cx, out_fy * y + out_cy))
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
    fn distorted_to_undistorted_roundtrips_with_undistorted_to_distorted() {
        // DJI Action 4 KB4 coefficients (real user calibration, per the
        // ROI-editor coordinate-space bug investigation this fix comes
        // from).
        let params = Lens::fisheye(
            3840,
            2880,
            1457.07373046875,
            1457.07373046875,
            1920.0,
            1440.0,
            [
                0.15513110160827637,
                0.1371408998966217,
                -0.0938614010810852,
                0.0041704000905156136,
            ],
        );
        let (w, h) = (3840u32, 2880u32);

        // Grid of undistorted (output) pixel positions, forward-mapped
        // to distorted space then back - should recover the original
        // point within numerical tolerance. Stays in [0.3, 0.7] (well
        // inside the frame): this module's extended-plane convention
        // reaches large theta faster than a plain stitch-plane
        // normalization would, so points near the edges/corners are
        // expected to hit the documented "Newton-Raphson doesn't
        // converge at extreme angles" case (returns `None`) - covered
        // separately below instead of treated as a roundtrip failure.
        let mut converged_count = 0;
        for gx in 3..=7 {
            for gy in 3..=7 {
                let out_x = w as f64 * gx as f64 / 10.0;
                let out_y = h as f64 * gy as f64 / 10.0;
                let (src_x, src_y) = undistorted_to_distorted(out_x, out_y, w, h, &params);
                if !(0.0..=w as f64).contains(&src_x) || !(0.0..=h as f64).contains(&src_y) {
                    continue;
                }
                let Some((back_x, back_y)) = distorted_to_undistorted(src_x, src_y, w, h, &params)
                else {
                    continue;
                };
                converged_count += 1;
                assert!(
                    (back_x - out_x).abs() < 0.5,
                    "x roundtrip failed at grid ({gx},{gy}): {out_x} -> {src_x} -> {back_x}"
                );
                assert!(
                    (back_y - out_y).abs() < 0.5,
                    "y roundtrip failed at grid ({gx},{gy}): {out_y} -> {src_y} -> {back_y}"
                );
            }
        }
        assert!(
            converged_count >= 20,
            "expected most of the central grid to converge, got {converged_count}"
        );
    }

    #[test]
    fn distorted_to_undistorted_rejects_extreme_corner_instead_of_returning_garbage() {
        // The exact case that first caught the missing convergence
        // check: a point 10% in from the top-left corner, which this
        // module's extended-plane (halved-focal-length) convention
        // pushes well past Newton-Raphson's safe convergence radius.
        // Before the fix this silently returned a wildly wrong point
        // (hundreds of thousands of pixels off) instead of `None`.
        let params = Lens::fisheye(
            3840,
            2880,
            1457.07373046875,
            1457.07373046875,
            1920.0,
            1440.0,
            [
                0.15513110160827637,
                0.1371408998966217,
                -0.0938614010810852,
                0.0041704000905156136,
            ],
        );
        let (w, h) = (3840u32, 2880u32);
        let (src_x, src_y) = undistorted_to_distorted(384.0, 288.0, w, h, &params);
        let result = distorted_to_undistorted(src_x, src_y, w, h, &params);
        assert!(
            result.is_none_or(|(bx, by)| (bx - 384.0).abs() < 0.5 && (by - 288.0).abs() < 0.5),
            "should either converge to the true point or report failure, not return garbage: {result:?}"
        );
    }

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

    // Step 1f: evaluate the KB4 polynomial on a real GPU via wgpu
    // compute dispatch and compare against the f64 Rust canonical. The
    // polynomial body in the compute shader below is a deliberate
    // SYNC_WITH copy of `shaders/fisheye.wgsl` lines 184-188; Step 3
    // will pull both behind a single `reco_core::lens::kb4` module
    // so the duplication collapses.
    #[test]
    #[cfg(feature = "gpu")]
    fn wgsl_kb4_matches_rust_kb4_on_theta_grid() {
        use wgpu::util::DeviceExt;

        let gpu = match pollster::block_on(crate::gpu::GpuContext::new()) {
            Ok(ctx) => ctx,
            Err(crate::gpu::GpuError::NoAdapter | crate::gpu::GpuError::AdapterRequest(_)) => {
                eprintln!("Skipping 1f: no GPU adapter available");
                return;
            }
            Err(e) => panic!("Unexpected GPU error: {e}"),
        };
        let device = gpu.device();
        let queue = gpu.queue();

        // GoPro HERO10 4K KB4 coefficients, same as the projection-
        // side tests.
        let d: [f32; 4] = [0.0342, 0.0677, -0.0741, 0.0299];

        // Theta grid covering the KB4 domain. Max theta ~1.5 rad
        // (~86 deg) matches the angular coverage of the HERO10's
        // fisheye and stays inside Newton-Raphson convergence.
        let thetas: Vec<f32> = (0..=100).map(|i| (i as f32) / 100.0 * 1.5).collect();
        let count = thetas.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            k: [f32; 4],
            count: u32,
            _pad: [u32; 3],
        }
        let uniforms = Uniforms {
            k: d,
            count,
            _pad: [0; 3],
        };

        let shader_source = r#"
struct Uniforms { k: vec4<f32>, count: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> theta_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> theta_d_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.count) { return; }
    let theta = theta_in[i];
    let theta2 = theta * theta;
    // SYNC_WITH shaders/fisheye.wgsl lines 184-188 (KB4 polynomial body)
    let theta_d = theta * (1.0
        + u.k.x * theta2
        + u.k.y * theta2 * theta2
        + u.k.z * theta2 * theta2 * theta2
        + u.k.w * theta2 * theta2 * theta2 * theta2);
    theta_d_out[i] = theta_d;
}
"#;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kb4_agreement_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kb4_agreement_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kb4_agreement_pl"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kb4_agreement_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kb4_uniform"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kb4_input"),
            contents: bytemuck::cast_slice(&thetas),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (count as u64) * std::mem::size_of::<f32>() as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kb4_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kb4_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kb4_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kb4_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kb4_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let groups = count.div_ceil(64);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll should not fail on a well-formed dispatch");
        rx.recv().unwrap().expect("buffer should map successfully");

        let gpu_theta_d: Vec<f32> = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        staging_buffer.unmap();

        let d_f64 = [d[0] as f64, d[1] as f64, d[2] as f64, d[3] as f64];
        for (i, &theta) in thetas.iter().enumerate() {
            let t2 = (theta as f64) * (theta as f64);
            let theta_d_rust = (theta as f64)
                * (1.0
                    + d_f64[0] * t2
                    + d_f64[1] * t2 * t2
                    + d_f64[2] * t2 * t2 * t2
                    + d_f64[3] * t2 * t2 * t2 * t2);
            let diff = (gpu_theta_d[i] as f64 - theta_d_rust).abs();
            assert!(
                diff < 1e-5,
                "theta={theta}: GPU {} vs Rust {theta_d_rust} (diff {diff})",
                gpu_theta_d[i]
            );
        }
    }

    #[test]
    fn undistort_center_pixel_unchanged() {
        // With plane-fitted intrinsics, out_cx = (w + 2*cx) / 4.
        // For cx = w/2 this gives out_cx = w/2 (exact integer with even dims).
        // At the output center, ray = (0,0), scale = 1, src = (cx, cy).
        let params = Lens::fisheye(100, 100, 50.0, 50.0, 50.0, 50.0, [0.1, 0.05, -0.03, 0.01]);

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
