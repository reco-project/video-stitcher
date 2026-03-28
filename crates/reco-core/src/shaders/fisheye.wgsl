// Reco v2 -- Fisheye undistortion + Reinhard LAB color transfer
//
// Ported from v1 GLSL (frontend/src/features/viewer/shaders/fisheye.js).
// Applies KB4 fisheye distortion correction on two 3D-positioned planes.

struct Uniforms {
    mvp: mat4x4<f32>,
    // Camera intrinsics (normalized: fx/width, fy/height, cx/width, cy/height)
    intrinsics: vec4<f32>,
    // KB4 distortion coefficients (k1, k2, k3, k4)
    dist: vec4<f32>,
    // Reinhard LAB color transfer: scale.xyz, pad
    lab_scale: vec4<f32>,
    // Reinhard LAB color transfer: offset.xyz, blend_width
    lab_offset_blend: vec4<f32>,
    // flags.x: is_right (0 or 1)
    // flags.y: use_nv12 (0 = YUV420P: separate U,V textures; 1 = NV12: interleaved UV in t_u)
    // flags.zw: reserved
    flags: vec4<u32>,
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
    // Remap UV from [0,1] to [-0.5, 1.5] — matches v1 vertex shader.
    // This extends the coordinate space so the undistortion can
    // map points outside the plane back to valid texture coords.
    out.uv = in.uv * 2.0 - vec2<f32>(0.5);
    return out;
}

// ---- Color space conversions (Reinhard LAB transfer) ----

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(
        c / 12.92,
        pow((c + 0.055) / 1.055, vec3<f32>(2.4)),
        c >= vec3<f32>(0.04045)
    );
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(
        c * 12.92,
        1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055,
        c >= vec3<f32>(0.0031308)
    );
}

fn rgb_to_xyz(rgb: vec3<f32>) -> vec3<f32> {
    let lin = srgb_to_linear(rgb);
    // sRGB -> XYZ (D65 illuminant), column-major
    let m = mat3x3<f32>(
        vec3<f32>(0.4124564, 0.2126729, 0.0193339),
        vec3<f32>(0.3575761, 0.7151522, 0.1191920),
        vec3<f32>(0.1804375, 0.0721750, 0.9503041),
    );
    return m * lin * 100.0;
}

fn xyz_to_lab(xyz: vec3<f32>) -> vec3<f32> {
    let ref_white = vec3<f32>(95.047, 100.0, 108.883);
    let f = xyz / ref_white;
    let ft = select(
        (903.3 * f + 16.0) / 116.0,
        pow(f, vec3<f32>(1.0 / 3.0)),
        f >= vec3<f32>(0.008856)
    );
    let L = 116.0 * ft.y - 16.0;
    let a = 500.0 * (ft.x - ft.y);
    let b = 200.0 * (ft.y - ft.z);
    // OpenCV LAB range: L: 0-255, a: 0-255, b: 0-255
    return vec3<f32>(L * (255.0 / 100.0), a + 128.0, b + 128.0);
}

fn lab_to_xyz(lab_ocv: vec3<f32>) -> vec3<f32> {
    let L = lab_ocv.x * (100.0 / 255.0);
    let a = lab_ocv.y - 128.0;
    let b = lab_ocv.z - 128.0;
    let fy = (L + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let f = vec3<f32>(fx, fy, fz);
    let f3 = f * f * f;
    let xyz = select(
        (116.0 * f - 16.0) / 903.3,
        f3,
        f3 >= vec3<f32>(0.008856)
    );
    return xyz * vec3<f32>(95.047, 100.0, 108.883);
}

fn xyz_to_rgb(xyz: vec3<f32>) -> vec3<f32> {
    // XYZ -> linear sRGB (D65), column-major
    let m = mat3x3<f32>(
        vec3<f32>( 3.2404542, -0.9692660,  0.0556434),
        vec3<f32>(-1.5371385,  1.8760108, -0.2040259),
        vec3<f32>(-0.4985314,  0.0415560,  1.0572252),
    );
    let lin = m * (xyz / 100.0);
    return linear_to_srgb(lin);
}

fn apply_reinhard_lab(rgb: vec3<f32>, scale: vec3<f32>, offset: vec3<f32>) -> vec3<f32> {
    // Skip if identity transform
    if all(scale == vec3<f32>(1.0)) && all(offset == vec3<f32>(0.0)) {
        return rgb;
    }
    var lab = xyz_to_lab(rgb_to_xyz(rgb));
    lab = lab * scale + offset;
    lab = clamp(lab, vec3<f32>(0.0), vec3<f32>(255.0));
    return clamp(xyz_to_rgb(lab_to_xyz(lab)), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- YUV → RGB conversion ----

/// Sample YUV textures and convert to linear RGB via BT.709.
///
/// Supports two input layouts (selected by `u.flags.y`):
///   0 = YUV420P: separate R8 textures for Y, U, V (software decode)
///   1 = NV12: R8 Y texture + Rg8 UV texture with interleaved U,V (NVDEC)
///
/// H.264 uses limited range (Y: 16–235, Cb/Cr: 16–240).
/// After the BT.709 matrix, we get sRGB-like gamma values, which
/// we linearize with srgb_to_linear to match the Rgba8UnormSrgb
/// auto-decode path (identical visual output to the old RGBA upload).
fn sample_yuv(uv: vec2<f32>) -> vec4<f32> {
    let y_raw = textureSample(t_y, s_video, uv).r;

    var u_raw: f32;
    var v_raw: f32;

    if u.flags.y == 1u {
        // NV12: t_u is Rg8Unorm with interleaved (U, V)
        let uv_sample = textureSample(t_u, s_video, uv);
        u_raw = uv_sample.r;
        v_raw = uv_sample.g;
    } else {
        // YUV420P: separate R8 textures
        u_raw = textureSample(t_u, s_video, uv).r;
        v_raw = textureSample(t_v, s_video, uv).r;
    }

    // BT.709 limited-range YCbCr → full-range R'G'B'
    let y = (y_raw - 16.0 / 255.0) * (255.0 / 219.0);
    let cb = (u_raw - 128.0 / 255.0) * (255.0 / 224.0);
    let cr = (v_raw - 128.0 / 255.0) * (255.0 / 224.0);

    let r = y + 1.5748 * cr;
    let g = y - 0.1873 * cb - 0.4681 * cr;
    let b = y + 1.8556 * cb;

    let rgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
    // BT.709 YCbCr→R'G'B' produces sRGB-domain values directly.
    // Render target is Rgba8Unorm, so we write sRGB values as-is.
    return vec4<f32>(rgb, 1.0);
}

// ---- Fragment shader ----

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let fx = u.intrinsics.x;
    let fy = u.intrinsics.y;
    let cx = u.intrinsics.z;
    let cy = u.intrinsics.w;

    // KB4 fisheye undistortion: map from plane UV to video texture coordinate
    let x = (in.uv.x - cx) / fx;
    let y = (in.uv.y - cy) / fy;
    let r = sqrt(x * x + y * y);
    let theta = atan(r);
    let theta2 = theta * theta;
    let theta_d = theta * (1.0
        + u.dist.x * theta2
        + u.dist.y * theta2 * theta2
        + u.dist.z * theta2 * theta2 * theta2
        + u.dist.w * theta2 * theta2 * theta2 * theta2);

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

    // Apply Reinhard LAB color transfer
    color = apply_reinhard_lab(color, u.lab_scale.xyz, u.lab_offset_blend.xyz);

    // Compute alpha for seam blending (right plane fades in at left edge)
    var alpha = 1.0;
    let blend_width = u.lab_offset_blend.w;
    if u.flags.x == 1u && blend_width > 0.0 {
        let edge_dist = in.uv.x;
        alpha = smoothstep(0.0, blend_width, edge_dist);
    }

    return vec4<f32>(color, alpha);
}
