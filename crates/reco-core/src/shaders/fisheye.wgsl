// Reco v2 -- Fisheye undistortion + YUV-space color transfer
//
// Ported from v1 GLSL (frontend/src/features/viewer/shaders/fisheye.js).
// Applies KB4 fisheye distortion correction on two 3D-positioned planes.
//
// Color transfer uses YUV space (BT.709): RGB->YUV is 3 mul + 3 add,
// apply per-channel scale+offset (3 mul + 3 add), YUV->RGB (3 mul + 3 add).
// ~18 arithmetic ops total, zero transcendentals. Replaces the previous
// Reinhard LAB pipeline (~42 transcendental ops per pixel: pow, sqrt).

struct Uniforms {
    mvp: mat4x4<f32>,
    // Camera intrinsics (normalized: fx/width, fy/height, cx/width, cy/height)
    intrinsics: vec4<f32>,
    // KB4 distortion coefficients (k1, k2, k3, k4)
    dist: vec4<f32>,
    // YUV color transfer: scale.xyz (Y, U, V), pad
    color_scale: vec4<f32>,
    // YUV color transfer: offset.xyz (Y, U, V), blend_width
    color_offset_blend: vec4<f32>,
    // flags.x: is_right (0 or 1)
    // flags.y: input_format (0 = YUV420P: separate U,V textures; 1 = NV12: interleaved UV
    //          in t_u; 2 = BGRA/RGBA: t_y holds packed 4-channel RGB, skip YUV conversion)
    // flags.z: flip_180 (0 or 1) - flip UV coordinates for 180-degree rotation
    //          Used by the GPU zero-copy path where buffer reversal is not possible.
    // flags.w: is_full_range (0 = limited 16-235, 1 = full 0-255)
    flags: vec4<u32>,
    // lens_preview.x: correction_amount (<0 = raw source, 0.0 = no correction, 1.0 = full KB4)
    // lens_preview.y: split_view (> 0.5 = left half uncorrected, right half corrected)
    lens_preview: vec4<f32>,
};

// YUV420P plane textures (Y = full res R8Unorm, U/V = half res R8Unorm)
@group(0) @binding(0) var t_y: texture_2d<f32>;
@group(0) @binding(1) var t_u: texture_2d<f32>;
@group(0) @binding(2) var t_v: texture_2d<f32>;
@group(0) @binding(3) var s_video: sampler;
@group(1) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

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
/// Supports three input layouts (selected by `u.flags.y`):
///   0 = YUV420P: separate R8 textures for Y, U, V (software decode).
///   1 = NV12: R8 Y texture + Rg8 UV texture with interleaved U,V (NVDEC).
///   2 = BGRA/RGBA: t_y is an `Rgba8Unorm` texture holding packed RGB
///       (source already sRGB-domain, skip YUV conversion).
///
/// H.264 uses limited range (Y: 16-235, Cb/Cr: 16-240). After the
/// BT.709 matrix we get sRGB-domain values which we write as-is to
/// the `Rgba8Unorm` render target.
fn sample_yuv(uv: vec2<f32>) -> vec4<f32> {
    // Apply 180-degree rotation for the GPU zero-copy path.
    // The CPU path reverses buffers in software; the GPU path flips UV coords instead.
    var sample_uv = uv;
    if u.flags.z == 1u {
        sample_uv = vec2<f32>(1.0 - uv.x, 1.0 - uv.y);
    }

    // BGRA / RGBA packed path: sample the full RGB triple in one fetch
    // and return without YUV conversion. The upload side is responsible
    // for delivering the triple in (R, G, B) order - swizzling BGRA is
    // handled at upload time so the shader only sees R-in-red.
    if u.flags.y == 2u {
        let rgba = textureSample(t_y, s_video, sample_uv);
        return vec4<f32>(rgba.rgb, 1.0);
    }

    let y_raw = textureSample(t_y, s_video, sample_uv).r;

    var u_raw: f32;
    var v_raw: f32;

    if u.flags.y == 1u {
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
    if u.flags.w == 1u {
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

// ---- Fragment shader ----

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Literal raw mode for ROI editing: sample the input frame in its
    // original source-pixel UV space. This matches the old browser/OpenCV
    // ROI editors, which stored raw normalized image coordinates.
    if u.lens_preview.x < 0.0 {
        let raw = sample_yuv(in.uv);
        return vec4<f32>(raw.rgb, 1.0);
    }

    // Remap UV from [0,1] to [-0.5, 1.5] in the fragment shader.
    // This extends the coordinate space so the undistortion can
    // map points outside the plane back to valid texture coords.
    // (Done here instead of the vertex shader because some embedded
    // GPU drivers pass vertex attributes directly to the fragment
    // stage, ignoring vertex shader output for user-defined varyings.)
    let uv = in.uv * 2.0 - vec2<f32>(0.5);

    let fx = u.intrinsics.x;
    let fy = u.intrinsics.y;
    let cx = u.intrinsics.z;
    let cy = u.intrinsics.w;

    // KB4 fisheye undistortion: map from plane UV to video texture coordinate
    let x = (uv.x - cx) / fx;
    let y = (uv.y - cy) / fy;
    let r = sqrt(x * x + y * y);
    let theta = atan(r);
    let theta2 = theta * theta;
    let theta_d_full = theta * (1.0
        + u.dist.x * theta2
        + u.dist.y * theta2 * theta2
        + u.dist.z * theta2 * theta2 * theta2
        + u.dist.w * theta2 * theta2 * theta2 * theta2);

    // Lens correction amount: 1.0 = full KB4, 0.0 = identity (pinhole).
    // Split view: left half uncorrected, right half fully corrected.
    var correction = u.lens_preview.x;
    if u.lens_preview.y > 0.5 {
        correction = select(0.0, 1.0, uv.x > 0.5);
    }
    let theta_d = mix(theta, theta_d_full, correction);

    var scale = 1.0;
    if r > 0.0 {
        scale = theta_d / r;
    }

    let distorted_uv = vec2<f32>(
        fx * x * scale + cx,
        fy * y * scale + cy,
    );

    // Bounds check — v2 uses separate textures (no stacking)
    if distorted_uv.x < 0.0 || distorted_uv.x > 1.0 ||
       distorted_uv.y < 0.0 || distorted_uv.y > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let tex_color = sample_yuv(distorted_uv);
    var color = tex_color.rgb;

    // Apply YUV-space color transfer
    color = apply_color_transfer(color, u.color_scale.xyz, u.color_offset_blend.xyz);

    // Compute alpha for seam blending (right plane fades in at left edge)
    var alpha = 1.0;
    let blend_width = u.color_offset_blend.w;
    if u.flags.x == 1u && blend_width > 0.0 {
        let edge_dist = uv.x;
        alpha = smoothstep(0.0, blend_width, edge_dist);
    }

    // Split-view separator line (1px white at the midpoint)
    if u.lens_preview.y > 0.5 && abs(uv.x - 0.5) < 0.001 {
        return vec4<f32>(1.0, 1.0, 1.0, alpha);
    }

    return vec4<f32>(color, alpha);
}
