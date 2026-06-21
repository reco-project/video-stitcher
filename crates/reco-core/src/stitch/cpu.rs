//! Float CPU gather and composite.
//!
//! Projection-independent: it queries the L-shape [`SurfaceMap`]s and
//! composites the covered surfaces, mirroring `fisheye.wgsl`'s fragment stage
//! per output pixel - live KB4 geometry (in [`super::geometry`]), float
//! bilinear sampling, BT.709 YUV->RGB, and a smoothstep seam. The output is
//! sRGB-domain RGBA written exactly as the GPU render target would, so it
//! doubles as the agreement oracle.

use crate::calibration::MatchCalibration;
use crate::render::planes::Nv12Planes;
use crate::render::viewport::ViewportConfig;

use super::SurfaceMap;
use super::geometry::l_shape_plane_maps;

/// Stitch two NV12 camera frames into an RGBA panorama on the CPU.
///
/// The float reference / oracle for the L-shape projection. Output is
/// `config.width * config.height * 4` bytes in `(R, G, B, A)` order, opaque.
///
/// * `left`, `right` - tightly packed NV12 source frames.
/// * `cam` - source frame dimensions `(width, height)`.
/// * `calib` - stereo calibration (intrinsics + plane layout).
/// * `config` - output dimensions, FOV, blend width, rig and lens correction.
/// * `yaw`, `pitch` - per-frame virtual-camera pan, in radians.
/// * `full_range` - `true` for full-range (0-255) YUV, `false` for limited
///   (16-235) BT.709, matching the source decoder.
#[allow(clippy::too_many_arguments)]
pub fn stitch_l_shape_rgba(
    left: &Nv12Planes,
    right: &Nv12Planes,
    cam: (u32, u32),
    calib: &MatchCalibration,
    config: &ViewportConfig,
    yaw: f32,
    pitch: f32,
    full_range: bool,
) -> Vec<u8> {
    let (cam_w, cam_h) = cam;
    let (out_w, out_h) = (config.width, config.height);
    let (lmap, rmap) = l_shape_plane_maps(calib, config, yaw, pitch);
    let blend_width = config.blend_width as f64;

    let mut out = vec![0u8; (out_w * out_h * 4) as usize];
    for py in 0..out_h {
        for px in 0..out_w {
            // Left plane is the opaque base; right plane fades in over it.
            let left_rgb = lmap
                .sample_uv(px, py)
                .map(|s| sample_nv12(left, cam_w, cam_h, s.u, s.v, full_range));
            let right_s = rmap.sample_uv(px, py);
            let right_rgb = right_s.map(|s| sample_nv12(right, cam_w, cam_h, s.u, s.v, full_range));
            let right_alpha = match right_s {
                Some(s) if blend_width > 0.0 => smoothstep(0.0, blend_width, s.edge),
                Some(_) => 1.0,
                None => 0.0,
            };

            let base = left_rgb.unwrap_or([0.0; 3]);
            let rgb = match right_rgb {
                Some(r) => [
                    base[0] + (r[0] - base[0]) * right_alpha,
                    base[1] + (r[1] - base[1]) * right_alpha,
                    base[2] + (r[2] - base[2]) * right_alpha,
                ],
                None => base,
            };

            let i = ((py * out_w + px) * 4) as usize;
            out[i] = to_u8(rgb[0]);
            out[i + 1] = to_u8(rgb[1]);
            out[i + 2] = to_u8(rgb[2]);
            out[i + 3] = 255;
        }
    }
    out
}

/// Sample an NV12 frame at normalised UV and convert to sRGB-domain RGB.
///
/// Mirrors `fisheye.wgsl::sample_yuv`: bilinear Y + bilinear interleaved
/// chroma (with the GPU sampler's half-texel convention), then BT.709
/// YCbCr->R'G'B' with limited/full range handling.
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
    // Luma: full-resolution plane, row stride = width.
    let y_raw = bilinear_u8(planes.y, w, w, h, u * w as f64 - 0.5, v * h as f64 - 0.5) / 255.0;
    // Chroma: half-resolution interleaved (U, V) texels, row stride = width bytes.
    let (cw, ch) = (w / 2, h / 2);
    let (u_raw, v_raw) = bilinear_chroma(
        planes.uv,
        w,
        cw,
        ch,
        u * cw as f64 - 0.5,
        v * ch as f64 - 0.5,
    );

    // BT.709 YCbCr -> R'G'B'. Limited range rescales to full before the matrix.
    let (y, cb, cr) = if full_range {
        (y_raw, u_raw / 255.0 - 0.5, v_raw / 255.0 - 0.5)
    } else {
        (
            (y_raw - 16.0 / 255.0) * (255.0 / 219.0),
            (u_raw / 255.0 - 128.0 / 255.0) * (255.0 / 224.0),
            (v_raw / 255.0 - 128.0 / 255.0) * (255.0 / 224.0),
        )
    };
    [
        (y + 1.5748 * cr).clamp(0.0, 1.0),
        (y - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0),
        (y + 1.8556 * cb).clamp(0.0, 1.0),
    ]
}

/// Bilinear sample of a single-byte plane with clamp-to-edge addressing.
#[inline]
fn bilinear_u8(data: &[u8], stride: usize, w: usize, h: usize, fx: f64, fy: f64) -> f64 {
    let fx = fx.clamp(0.0, (w - 1) as f64);
    let fy = fy.clamp(0.0, (h - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = fx - x0 as f64;
    let dy = fy - y0 as f64;
    let p = |x: usize, y: usize| data[y * stride + x] as f64;
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

/// Hermite smoothstep, matching WGSL `smoothstep`.
#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Quantise an sRGB-domain channel in `[0, 1]` to a `u8`, matching the GPU's
/// `Rgba8Unorm` write (round-to-nearest).
#[inline]
fn to_u8(v: f64) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
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
        let s = bilinear_u8(&data, 16, 16, 16, 3.3, 7.8);
        assert!((s - 200.0).abs() < 1e-9);
    }

    #[test]
    fn full_range_grey_is_neutral() {
        // Full-range Y=128, chroma 128 -> mid-grey, near-equal channels.
        let y = vec![128u8; 4 * 4];
        let uv = vec![128u8; 4 * 2];
        let planes = Nv12Planes { y: &y, uv: &uv };
        let rgb = sample_nv12(&planes, 4, 4, 0.5, 0.5, true);
        assert!((rgb[0] - rgb[1]).abs() < 0.02 && (rgb[1] - rgb[2]).abs() < 0.02);
        assert!((rgb[0] - 128.0 / 255.0).abs() < 0.02);
    }
}
