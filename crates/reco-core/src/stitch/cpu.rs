//! Float CPU gather and composite.
//!
//! Two-axis-agnostic: the gather loop ([`stitch_l_shape_with`]) is independent
//! of both the projection (it queries [`SurfaceMap`]s from [`super::geometry`])
//! and the pixel format (it works through a `sample(u, v) -> rgb` closure). It
//! mirrors `fisheye.wgsl`'s fragment stage per output pixel - float bilinear,
//! BT.709 YUV->RGB, and a smoothstep seam - agreeing with the GPU render target
//! to ~1 LSB (the GPU blends through an 8-bit intermediate and uses
//! implementation-defined unorm rounding), so it doubles as the agreement oracle.

use crate::calibration::Calibration;
use crate::geometry::Pose;
use crate::projection::Projection;
use crate::render::planes::{Nv12Planes, YuvPlanes};
use crate::render::viewport::ViewportSize;

use super::executor::StitchError;
use super::{BlendRule, SurfaceMap};

/// Reject degenerate source dimensions that would underflow chroma indexing.
fn check_source_dims(cw: u32, ch: u32) -> Result<(), StitchError> {
    if cw < 2 || ch < 2 {
        return Err(StitchError::InvalidConfig(format!(
            "source dimensions must be >= 2, got {cw}x{ch}"
        )));
    }
    Ok(())
}

/// Reject a plane shorter than the configured frame requires (would otherwise
/// index out of bounds in the gather and panic).
fn check_plane(plane: &[u8], expected: usize) -> Result<(), StitchError> {
    if plane.len() < expected {
        return Err(StitchError::FrameSizeMismatch {
            expected,
            actual: plane.len(),
        });
    }
    Ok(())
}

/// Stitch two NV12 camera frames into an RGBA panorama on the CPU.
///
/// The float reference / oracle for the L-shape projection. Output is
/// `config.width * config.height * 4` bytes in `(R, G, B, A)` order, opaque.
///
/// * `left`, `right` - tightly packed NV12 source frames.
/// * `cam` - source frame dimensions `(width, height)`.
/// * `calib` - stereo calibration (intrinsics + plane layout).
/// * `viewport_size` - output dimensions.
/// * `pose` - per-frame virtual-camera pan (radians) + fov zoom (degrees).
/// * `full_range` - `true` for full-range (0-255) YUV, `false` for limited
///   (16-235) BT.709, matching the source decoder.
///
/// Returns [`StitchError::InvalidConfig`] for degenerate source dimensions and
/// [`StitchError::FrameSizeMismatch`] if a plane is shorter than `cam` requires,
/// rather than panicking - this is the GPU-less render path, so callers (e.g.
/// the X5) get a typed error instead of an out-of-bounds panic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stitch_rgba(
    projection: &dyn Projection,
    planes: &[Nv12Planes<'_>],
    cam: (u32, u32),
    calib: &Calibration,
    viewport_size: &ViewportSize,
    pose: Pose,
    full_range: bool,
) -> Result<Vec<u8>, StitchError> {
    let (cw, ch) = cam;
    check_source_dims(cw, ch)?;
    check_camera_count(projection, planes.len())?;
    let (w, h) = (cw as usize, ch as usize);
    for p in planes {
        check_plane(p.y, w * h)?;
        check_plane(p.uv, w * (h / 2))?;
    }
    let samplers: Vec<_> = planes
        .iter()
        .map(|p| move |u, v| sample_nv12(p, cw, ch, u, v, full_range))
        .collect();
    Ok(stitch_with(
        projection,
        calib,
        viewport_size,
        pose,
        &samplers,
    ))
}

/// Stitch two YUV420p (planar) camera frames into an RGBA panorama on the CPU.
///
/// Identical to [`stitch_rgba`] but for the software-decode planar
/// format (separate Y, U, V planes). Useful as the GPU-less rendering path on
/// desktop/cloud where FFmpeg software decode yields YUV420p.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stitch_rgba_yuv420p(
    projection: &dyn Projection,
    planes: &[YuvPlanes<'_>],
    cam: (u32, u32),
    calib: &Calibration,
    viewport_size: &ViewportSize,
    pose: Pose,
    full_range: bool,
) -> Result<Vec<u8>, StitchError> {
    let (cw, ch) = cam;
    check_source_dims(cw, ch)?;
    check_camera_count(projection, planes.len())?;
    let (w, h) = (cw as usize, ch as usize);
    let chroma = (w / 2) * (h / 2);
    for p in planes {
        check_plane(p.y, w * h)?;
        check_plane(p.u, chroma)?;
        check_plane(p.v, chroma)?;
    }
    let samplers: Vec<_> = planes
        .iter()
        .map(|p| move |u, v| sample_yuv420p(p, cw, ch, u, v, full_range))
        .collect();
    Ok(stitch_with(
        projection,
        calib,
        viewport_size,
        pose,
        &samplers,
    ))
}

/// The supplied frame count must match the camera count the
/// projection consumes; a mismatch would silently sample the wrong camera.
fn check_camera_count(projection: &dyn Projection, planes: usize) -> Result<(), StitchError> {
    if planes != projection.camera_count() {
        return Err(StitchError::InvalidConfig(format!(
            "projection '{}' consumes {} camera(s) but {planes} frame(s) were supplied",
            projection.name(),
            projection.camera_count(),
        )));
    }
    Ok(())
}

/// Format-agnostic gather and composite driven by the projection.
///
/// The surface list comes from [`Projection::surface_maps`] - this is the
/// L1 dispatch made real. `samplers[i]` maps a normalised camera UV to
/// sRGB-domain RGB for source frame `i`; the loop itself knows nothing
/// about the pixel format.
fn stitch_with(
    projection: &dyn Projection,
    calib: &Calibration,
    viewport_size: &ViewportSize,
    pose: Pose,
    samplers: &[impl Fn(f64, f64) -> [f64; 3]],
) -> Vec<u8> {
    let ctx = crate::projection::ProjectionContext {
        calibration: calib,
        viewport_size,
        pose,
    };
    let surfaces = projection.surface_maps(&ctx);
    let sampler_refs: Vec<&dyn Fn(f64, f64) -> [f64; 3]> = samplers
        .iter()
        .map(|s| s as &dyn Fn(f64, f64) -> [f64; 3])
        .collect();
    composite_rgba(
        &surfaces,
        &sampler_refs,
        viewport_size.width,
        viewport_size.height,
    )
}

/// Composite an ordered surface list into an opaque RGBA buffer.
///
/// Surface `i` samples source `i` through `samplers[i]`. Surfaces apply in
/// order: the first covered surface lays the base, later ones blend over it
/// per their [`BlendRule`]. This is the projection-agnostic core the
/// N-surface projections (mono cylinder next) drive.
fn composite_rgba(
    surfaces: &[(Box<dyn SurfaceMap>, BlendRule)],
    samplers: &[&dyn Fn(f64, f64) -> [f64; 3]],
    out_w: u32,
    out_h: u32,
) -> Vec<u8> {
    // A real assert (not debug): a length mismatch would make the zip
    // silently truncate in release and composite a wrong image with no
    // error. Once per frame, so the check is free next to the per-pixel
    // work.
    assert_eq!(
        surfaces.len(),
        samplers.len(),
        "projection emitted {} surfaces but the kernel supplied {} samplers",
        surfaces.len(),
        samplers.len()
    );
    let mut out = vec![0u8; (out_w * out_h * 4) as usize];
    for py in 0..out_h {
        for px in 0..out_w {
            let mut rgb = [0.0f64; 3];
            for ((map, rule), sample) in surfaces.iter().zip(samplers) {
                if let Some(s) = map.sample_uv(px, py) {
                    let alpha = rule.alpha(s.edge);
                    let src = sample(s.u, s.v);
                    rgb = [
                        rgb[0] + (src[0] - rgb[0]) * alpha,
                        rgb[1] + (src[1] - rgb[1]) * alpha,
                        rgb[2] + (src[2] - rgb[2]) * alpha,
                    ];
                }
            }
            let i = ((py * out_w + px) * 4) as usize;
            out[i] = to_u8(rgb[0]);
            out[i + 1] = to_u8(rgb[1]);
            out[i + 2] = to_u8(rgb[2]);
            out[i + 3] = 255;
        }
    }
    out
}

/// Sample an NV12 frame at normalised UV (bilinear Y + interleaved chroma).
#[inline]
fn sample_nv12(
    planes: &Nv12Planes,
    cam_w: u32,
    cam_h: u32,
    u: f64,
    v: f64,
    full_range: bool,
) -> [f64; 3] {
    let (w, h) = (cam_w as usize, cam_h as usize);
    let (cw, ch) = (w / 2, h / 2);
    let y_raw = bilinear_u8(planes.y, w, h, u * w as f64 - 0.5, v * h as f64 - 0.5) / 255.0;
    let (u_raw, v_raw) = bilinear_chroma(
        planes.uv,
        w,
        cw,
        ch,
        u * cw as f64 - 0.5,
        v * ch as f64 - 0.5,
    );
    yuv_to_rgb(y_raw, u_raw / 255.0, v_raw / 255.0, full_range)
}

/// Sample a YUV420p frame at normalised UV (bilinear Y + separate U, V planes).
#[inline]
fn sample_yuv420p(
    planes: &YuvPlanes,
    cam_w: u32,
    cam_h: u32,
    u: f64,
    v: f64,
    full_range: bool,
) -> [f64; 3] {
    let (w, h) = (cam_w as usize, cam_h as usize);
    let (cw, ch) = (w / 2, h / 2);
    let y_raw = bilinear_u8(planes.y, w, h, u * w as f64 - 0.5, v * h as f64 - 0.5) / 255.0;
    let u_raw = bilinear_u8(planes.u, cw, ch, u * cw as f64 - 0.5, v * ch as f64 - 0.5) / 255.0;
    let v_raw = bilinear_u8(planes.v, cw, ch, u * cw as f64 - 0.5, v * ch as f64 - 0.5) / 255.0;
    yuv_to_rgb(y_raw, u_raw, v_raw, full_range)
}

/// BT.709 YCbCr -> sRGB-domain R'G'B', inputs normalised to `[0, 1]`.
///
/// Limited range rescales to full before the matrix. Mirrors
/// `fisheye.wgsl::sample_yuv`.
#[inline]
fn yuv_to_rgb(y_raw: f64, u_raw: f64, v_raw: f64, full_range: bool) -> [f64; 3] {
    let (y, cb, cr) = if full_range {
        (y_raw, u_raw - 0.5, v_raw - 0.5)
    } else {
        (
            (y_raw - 16.0 / 255.0) * (255.0 / 219.0),
            (u_raw - 128.0 / 255.0) * (255.0 / 224.0),
            (v_raw - 128.0 / 255.0) * (255.0 / 224.0),
        )
    };
    [
        (y + 1.5748 * cr).clamp(0.0, 1.0),
        (y - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0),
        (y + 1.8556 * cb).clamp(0.0, 1.0),
    ]
}

/// Bilinear sample of a tightly-packed single-byte plane (row stride = `w`),
/// with clamp-to-edge addressing.
#[inline]
fn bilinear_u8(data: &[u8], w: usize, h: usize, fx: f64, fy: f64) -> f64 {
    let fx = fx.clamp(0.0, (w - 1) as f64);
    let fy = fy.clamp(0.0, (h - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = fx - x0 as f64;
    let dy = fy - y0 as f64;
    let p = |x: usize, y: usize| data[y * w + x] as f64;
    let top = p(x0, y0) * (1.0 - dx) + p(x1, y0) * dx;
    let bot = p(x0, y1) * (1.0 - dx) + p(x1, y1) * dx;
    top * (1.0 - dy) + bot * dy
}

/// Bilinear sample of an interleaved NV12 chroma plane (2 bytes per texel),
/// returning `(U, V)` in raw `[0, 255]` units. Clamp-to-edge addressing.
#[inline]
fn bilinear_chroma(uv: &[u8], stride: usize, cw: usize, ch: usize, fx: f64, fy: f64) -> (f64, f64) {
    let fx = fx.clamp(0.0, (cw - 1) as f64);
    let fy = fy.clamp(0.0, (ch - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(cw - 1);
    let y1 = (y0 + 1).min(ch - 1);
    let dx = fx - x0 as f64;
    let dy = fy - y0 as f64;
    let p = |x: usize, y: usize, off: usize| uv[y * stride + x * 2 + off] as f64;
    let lerp = |off: usize| {
        let top = p(x0, y0, off) * (1.0 - dx) + p(x1, y0, off) * dx;
        let bot = p(x0, y1, off) * (1.0 - dx) + p(x1, y1, off) * dx;
        top * (1.0 - dy) + bot * dy
    };
    (lerp(0), lerp(1))
}

/// Quantise an sRGB-domain channel in `[0, 1]` to a `u8` (round-to-nearest).
/// The GPU's `Rgba8Unorm` unorm rounding is implementation-defined, so this
/// agrees to ~1 LSB rather than bit-exactly.
#[inline]
fn to_u8(v: f64) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        use crate::stitch::smoothstep;
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn to_u8_rounds_and_clamps() {
        assert_eq!(to_u8(0.0), 0);
        assert_eq!(to_u8(1.0), 255);
        assert_eq!(to_u8(0.5), 128); // 127.5 rounds to 128
        assert_eq!(to_u8(2.0), 255);
    }

    #[test]
    fn bilinear_flat_plane_is_constant() {
        let data = vec![200u8; 16 * 16];
        let s = bilinear_u8(&data, 16, 16, 3.3, 7.8);
        assert!((s - 200.0).abs() < 1e-9);
    }

    #[test]
    fn nv12_and_yuv420p_agree_on_grey() {
        // Limited-range Y=128, chroma 128 -> the two formats must agree, since
        // they carry the same samples in different layouts.
        let y = vec![128u8; 8 * 8];
        let nv12_uv = vec![128u8; 8 * 4]; // interleaved, (w) * (h/2)
        let u = vec![128u8; 4 * 4]; // (w/2)*(h/2)
        let v = vec![128u8; 4 * 4];
        let nv = Nv12Planes {
            y: &y,
            uv: &nv12_uv,
        };
        let yuv = YuvPlanes {
            y: &y,
            u: &u,
            v: &v,
        };
        let a = sample_nv12(&nv, 8, 8, 0.5, 0.5, false);
        let b = sample_yuv420p(&yuv, 8, 8, 0.5, 0.5, false);
        for k in 0..3 {
            assert!(
                (a[k] - b[k]).abs() < 1e-9,
                "channel {k}: {} vs {}",
                a[k],
                b[k]
            );
        }
    }
}
