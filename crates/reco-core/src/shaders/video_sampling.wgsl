// ---- YUV-space color transfer ----
//
// BT.709 RGB<->YUV conversion uses only multiply-add operations (no pow/sqrt).
// The CPU computes per-channel scale+offset from source/target statistics once;
// the shader applies them every pixel with ~18 arithmetic ops total.

fn rgb_to_yuv(rgb: vec3<f32>) -> vec3<f32> {
    // BT.709 full-range RGB [0,1] -> YUV (Y [0,1], U/V [-0.5, 0.5])
    let y = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    let u = -0.1146 * rgb.r - 0.3854 * rgb.g + 0.5 * rgb.b;
    let v = 0.5 * rgb.r - 0.4542 * rgb.g - 0.0458 * rgb.b;
    return vec3<f32>(y, u, v);
}

fn yuv_to_rgb(yuv: vec3<f32>) -> vec3<f32> {
    // BT.709 YUV -> full-range RGB [0,1]
    let r = yuv.x + 1.5748 * yuv.z;
    let g = yuv.x - 0.1873 * yuv.y - 0.4681 * yuv.z;
    let b = yuv.x + 1.8556 * yuv.y;
    return vec3<f32>(r, g, b);
}

fn apply_color_transfer(rgb: vec3<f32>, scale: vec3<f32>, offset: vec3<f32>) -> vec3<f32> {
    // Skip if identity transform (scale=1, offset=0)
    if all(scale == vec3<f32>(1.0)) && all(offset == vec3<f32>(0.0)) {
        return rgb;
    }
    var yuv = rgb_to_yuv(rgb);
    yuv = yuv * scale + offset;
    return clamp(yuv_to_rgb(yuv), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- YUV → RGB conversion ----

// The common SDR pipeline compiles out matrix, range and P010 selection.
override extended_video_color: bool = false;

/// Sample the input planes and return nonlinear R'G'B'.
///
/// Supports three input layouts (selected by `flags.y`):
///   0 = YUV420P: separate R8 textures for Y, U, V (software decode).
///   1 = NV12: R8 Y texture + Rg8 UV texture with interleaved U,V (NVDEC).
///   2 = RGBA: t_y is an `Rgba8Unorm` texture holding packed RGB
///       (source already sRGB-domain, skip YUV conversion).
///
fn sample_video(uv: vec2<f32>, flags: vec4<u32>) -> vec4<f32> {
    // Native display rotation, mapped back from displayed to stored UVs.
    var sample_uv = uv;
    switch flags.z {
        case 1u: { sample_uv = vec2<f32>(1.0 - uv.x, 1.0 - uv.y); }
        case 2u: { sample_uv = vec2<f32>(uv.y, 1.0 - uv.x); }
        case 3u: { sample_uv = vec2<f32>(1.0 - uv.y, uv.x); }
        default: {}
    }

    // RGBA packed path: sample the full RGB triple in one fetch
    // and return without YUV conversion. The upload side is responsible
    // for delivering the triple in (R, G, B) order - swizzling BGRA is
    // handled at upload time so the shader only sees R-in-red.
    if flags.y == 2u {
        let rgba = textureSample(t_y, s_video, sample_uv);
        return vec4<f32>(rgba.rgb, 1.0);
    }

    let y_raw = textureSample(t_y, s_video, sample_uv).r;

    var u_raw: f32;
    var v_raw: f32;

    if flags.y == 1u {
        // NV12: t_u is Rg8Unorm (or Rg16Unorm for 10-bit) with interleaved (U, V)
        let uv_sample = textureSample(t_u, s_video, sample_uv);
        u_raw = uv_sample.r;
        v_raw = uv_sample.g;
    } else {
        // YUV420P: separate R8 textures
        u_raw = textureSample(t_u, s_video, sample_uv).r;
        v_raw = textureSample(t_v, s_video, sample_uv).r;
    }

    if extended_video_color {
        return vec4<f32>(clamp(extended_yuv_to_rgb(vec3(y_raw, u_raw, v_raw), flags.w),
            vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }

    let y = (y_raw - 16.0 / 255.0) * (255.0 / 219.0);
    let cb = (u_raw - 128.0 / 255.0) * (255.0 / 224.0);
    let cr = (v_raw - 128.0 / 255.0) * (255.0 / 224.0);
    let r = y + 1.5748 * cr;
    let g = y - 0.1873 * cb - 0.4681 * cr;
    let b = y + 1.8556 * cb;
    let rgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb, 1.0);
}

// flags.w: full range in bit 0, matrix in bits 1-2, P010 in bit 3.
fn extended_yuv_to_rgb(raw: vec3<f32>, color: u32) -> vec3<f32> {
    let full_range = (color & 1u) != 0u;
    var yuv: vec3<f32>;
    if (color & 8u) != 0u {
        // R16Unorm stores (code << 6) / 65535, including interpolated samples.
        let code = raw * (65535.0 / 64.0);
        if full_range {
            yuv = (code - vec3(0.0, 512.0, 512.0)) / 1023.0;
        } else {
            yuv = (code - vec3(64.0, 512.0, 512.0)) / vec3(876.0, 896.0, 896.0);
        }
    } else if full_range {
        yuv = raw - vec3(0.0, 128.0 / 255.0, 128.0 / 255.0);
    } else {
        yuv = (raw - vec3(16.0, 128.0, 128.0) / 255.0)
            * (255.0 / vec3(219.0, 224.0, 224.0));
    }

    let y = yuv.x;
    let cb = yuv.y;
    let cr = yuv.z;
    switch (color >> 1u) & 3u {
        case 1u: {
            return vec3(y + 1.4746 * cr, y - 0.16455313 * cb - 0.57135313 * cr, y + 1.8814 * cb);
        }
        case 2u: {
            return vec3(y + 1.402 * cr, y - 0.34413629 * cb - 0.71413629 * cr, y + 1.772 * cb);
        }
        default: { return yuv_to_rgb(yuv); }
    }
}
