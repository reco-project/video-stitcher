//! KB4 fisheye lens model - CPU-side distortion and undistortion.
//!
//! Implements the Kannala-Brandt 4-coefficient fisheye model used by
//! Gyroflow and OpenCV:
//!
//! ```text
//! θ_d = θ × (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
//! ```
//!
//! Shared optical mapping for rendering, source correspondence and calibration.

#[cfg(feature = "gpu")]
pub mod preview;
#[cfg(feature = "gpu")]
pub mod undistort;

use crate::calibration::Lens;

pub(crate) mod inverse;

/// Radial capacity remaining beyond the farthest sensor corner on the central KB4 branch.
///
/// A nonnegative margin reserves a radial derivative of at least 0.01 across the sensor.
/// This is an invertibility constraint for fitting, not an optical accuracy measure.
/// Invalid or nonfinite optical parameters return negative infinity.
pub fn sensor_domain_margin(lens: &Lens) -> f64 {
    inverse::sensor_margin(lens).unwrap_or(f64::NEG_INFINITY)
}

pub(crate) use kb4::kb4_forward_scale;

/// Canonical Rust implementation of the KB4 polynomial.
///
/// Callers that need the polynomial (inverse-fisheye Newton-Raphson
/// iteration and `kb4_forward_scale`) all delegate here
/// so the formula lives in exactly one place Rust-side. The WGSL
/// mirror in `shaders/lens_mapping.wgsl` can't cross-
/// language-link; the `wgsl_kb4_matches_rust_kb4_on_theta_grid`
/// compute-dispatch test locks the two sides together numerically.
pub(crate) mod kb4 {
    /// Evaluate `θ_d = θ * (1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)`.
    ///
    /// SYNC_WITH `shaders/lens_mapping.wgsl`. Any edit here
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

    /// Derivative of [`theta_d`] with respect to `theta`, used by the inverse solver:
    /// ```text
    /// d/dθ θ_d = 1 + 3 k₁ θ² + 5 k₂ θ⁴ + 7 k₃ θ⁶ + 9 k₄ θ⁸
    /// ```
    #[inline]
    pub(super) fn theta_d_prime(theta: f64, d: &[f64; 4]) -> f64 {
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

    /// Forward scale blended between equidistant (zero-polynomial) and full KB4 by `correction` in `[0, 1]`. Matches the shader's `mix(theta, theta_d, correction)` in `fisheye.wgsl`; zero correction still applies `atan(r) / r`, not the identity mapping.
    #[inline]
    pub fn kb4_forward_scale_with_correction(r: f64, d: &[f64; 4], correction: f64) -> f64 {
        if r < 1e-10 {
            return 1.0;
        }
        let equidistant = r.atan() / r;
        let full = kb4_forward_scale(r, d);
        equidistant + (full - equidistant) * correction
    }
}

/// Inverse KB4 fisheye: distorted camera pixel `[0,1]` to extended plane UV.
pub(crate) fn inverse_fisheye(dist_x: f64, dist_y: f64, params: &Lens) -> Option<(f64, f64)> {
    let w = params.width as f64;
    let h = params.height as f64;
    let fx = params.fx / w;
    let fy = params.fy / h;
    let cx = params.cx / w;
    let cy = params.cy / h;

    let dx = (dist_x - cx) / fx;
    let dy = (dist_y - cy) / fy;
    let theta_d = dx.hypot(dy);
    let theta = inverse::theta(theta_d, &params.distortion, params.correction as f64)?;

    if theta_d < 1e-12 {
        return Some((cx, cy));
    }

    let scale = theta.tan() / theta_d;
    let u = fx * dx * scale + cx;
    let v = fy * dy * scale + cy;
    (u.is_finite() && v.is_finite()).then_some((u, v))
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
    fn inverse_theta_recovers_forward_angles() {
        for coefficients in [
            [0.0; 4],
            [0.0342, 0.0677, -0.0741, 0.0299],
            [0.4, 0.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0, 0.0],
        ] {
            for correction in [0.0, 0.4, 1.0] {
                for i in 0..=100 {
                    let theta = i as f64 * 1.5 / 100.0;
                    let r = theta.tan();
                    let radius =
                        r * kb4::kb4_forward_scale_with_correction(r, &coefficients, correction);
                    let recovered = inverse::theta(radius, &coefficients, correction).unwrap();
                    assert!(
                        (recovered - theta).abs() < 1e-9,
                        "coefficients={coefficients:?}, correction={correction}, theta={theta}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_theta_recovers_a_wide_angle_without_leaving_the_domain() {
        let coefficients = [-0.25, -0.03, 0.074, -0.011];
        let expected = 1.5;
        let radius = kb4::theta_d(expected, &coefficients);
        // An unconstrained Newton step exceeds pi/2 for this finite forward-facing root.
        let actual = inverse::theta(radius, &coefficients, 1.0).unwrap();
        assert!((actual - expected).abs() < 1e-9);
    }

    #[test]
    fn inverse_theta_rejects_converged_roots_outside_the_ray_domain() {
        // With zero coefficients the polynomial root is exactly the source radius.
        assert!(inverse::theta(2.0, &[0.0; 4], 1.0).is_none());
    }

    #[test]
    fn inverse_theta_checks_residual_for_non_monotonic_lenses() {
        let mut solved = 0;
        let mut rejected = 0;
        for k1 in [-0.3, -0.5, -1.0] {
            let coefficients = [k1, 0.0, 0.0, 0.0];
            for i in 0..=100 {
                let radius = i as f64 * 1.5 / 100.0;
                match inverse::theta(radius, &coefficients, 1.0) {
                    Some(theta) => {
                        assert!((0.0..std::f64::consts::FRAC_PI_2).contains(&theta));
                        assert!((kb4::theta_d(theta, &coefficients) - radius).abs() <= 1e-10);
                        solved += 1;
                    }
                    None => rejected += 1,
                }
            }
        }
        assert!(solved > 0 && rejected > 0);
    }

    // Exercise the actual shared optical shader against the f64 Rust model.
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
    theta_d_out[i] = lens_uv(vec2(tan(theta), 0.0), vec4(1.0, 1.0, 0.0, 0.0), u.k, 1.0).x;
}
"#;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kb4_agreement_shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/lens_mapping.wgsl"),
                    shader_source
                )
                .into(),
            ),
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
