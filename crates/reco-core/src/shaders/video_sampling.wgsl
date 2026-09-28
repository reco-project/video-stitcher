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

/// Sample the input plane(s) and return an sRGB-domain RGB triple.
///
/// Supports three input layouts (selected by `flags.y`):
///   0 = YUV420P: separate R8 textures for Y, U, V (software decode).
///   1 = NV12: R8 Y texture + Rg8 UV texture with interleaved U,V (NVDEC).
///   2 = RGBA: t_y is an `Rgba8Unorm` texture holding packed RGB
///       (source already sRGB-domain, skip YUV conversion).
///
/// Source metadata selects limited or full range. The BT.709 matrix
/// produces display-encoded RGB, written without another transfer
/// function to the `Rgba8Unorm` render target.
fn sample_video(uv: vec2<f32>, flags: vec4<u32>) -> vec4<f32> {
    // Apply 180-degree rotation for the GPU zero-copy path.
    // The CPU path reverses buffers in software; the GPU path flips UV coords instead.
    var sample_uv = uv;
    if flags.z == 1u {
        sample_uv = vec2<f32>(1.0 - uv.x, 1.0 - uv.y);
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

    // BT.709 YCbCr -> R'G'B'. Range scaling depends on flags.w:
    //   0 = limited range (Y: 16-235, Cb/Cr: 16-240)
    //   1 = full range (Y: 0-255, Cb/Cr: 0-255)
    var y: f32;
    var cb: f32;
    var cr: f32;
    if flags.w == 1u {
        y = y_raw;
        cb = u_raw - 0.5;
        cr = v_raw - 0.5;
    } else {
        y = (y_raw - 16.0 / 255.0) * (255.0 / 219.0);
        cb = (u_raw - 128.0 / 255.0) * (255.0 / 224.0);
        cr = (v_raw - 128.0 / 255.0) * (255.0 / 224.0);
    }

    let r = y + 1.5748 * cr;
    let g = y - 0.1873 * cb - 0.4681 * cr;
    let b = y + 1.8556 * cb;

    let rgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
    // BT.709 YCbCr->R'G'B' produces sRGB-domain values directly.
    // Render target is Rgba8Unorm, so we write sRGB values as-is.
    return vec4<f32>(rgb, 1.0);
}
