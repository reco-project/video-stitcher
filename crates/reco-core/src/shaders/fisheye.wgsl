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
    //          in t_u; 2 = RGBA: t_y holds packed 4-channel RGB, skip YUV conversion)
    // flags.z: flip_180 (0 or 1) - flip UV coordinates for 180-degree rotation
    //          Used by the GPU zero-copy path where buffer reversal is not possible.
    // flags.w: YUV range, matrix and sample format (see video_sampling.wgsl)
    flags: vec4<u32>,
    // lens_preview.x: 0.0 = equidistant, 1.0 = full KB4, negative = raw bypass
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

// ---- Fragment shader ----

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Remap UV from [0,1] to [-0.5, 1.5] in the fragment shader.
    // This extends the coordinate space so the undistortion can
    // map points outside the plane back to valid texture coords.
    // (Done here instead of the vertex shader because some embedded
    // GPU drivers pass vertex attributes directly to the fragment
    // stage, ignoring vertex shader output for user-defined varyings.)
    let uv = in.uv * 2.0 - vec2<f32>(0.5);

    // Raw mode: negative correction bypasses all projection math
    // and samples the input texture directly at the fragment UV.
    if u.lens_preview.x < 0.0 {
        if !(uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
        let raw = sample_video(uv, u.flags);
        return vec4<f32>(raw.rgb, 1.0);
    }

    var correction = u.lens_preview.x;
    if u.lens_preview.y > 0.5 {
        correction = select(0.0, 1.0, uv.x > 0.5);
    }
    let distorted_uv = lens_uv(uv, u.intrinsics, u.dist, correction);

    // Bounds check — v2 uses separate textures (no stacking)
    if !(distorted_uv.x >= 0.0 && distorted_uv.x <= 1.0 &&
         distorted_uv.y >= 0.0 && distorted_uv.y <= 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let tex_color = sample_video(distorted_uv, u.flags);
    var color = tex_color.rgb;

    // Apply YUV-space color transfer
    color = apply_color_transfer(color, u.color_scale.xyz, u.color_offset_blend.xyz);

    // Compute alpha for seam blending (right plane fades in at left edge)
    var alpha = 1.0;
    let blend_width = u.color_offset_blend.w;
    if u.flags.x == 1u && blend_width > 0.0 && uv.x < blend_width {
        alpha = smoothstep(0.0, blend_width, uv.x);
    }

    // Split-view separator line (1px white at the midpoint)
    if u.lens_preview.y > 0.5 && abs(uv.x - 0.5) < 0.001 {
        return vec4<f32>(1.0, 1.0, 1.0, alpha);
    }

    return vec4<f32>(color, alpha);
}

// Uncovered-pixel fill for the right plane inside its fade band.
// Blended with OneMinusDstAlpha / DstAlpha and ColorWrites::COLOR so pixels
// without left-base coverage (dst.a == 0) take the right color while
// left-covered pixels (dst.a == 1) stay untouched, without altering dst.a.
@fragment
fn fs_right_uncovered(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv * 2.0 - vec2<f32>(0.5);
    let blend_width = u.color_offset_blend.w;
    if !(uv.x < blend_width) {
        discard;
    }
    let distorted_uv = lens_uv(uv, u.intrinsics, u.dist, u.lens_preview.x);
    if !(distorted_uv.x >= 0.0 && distorted_uv.x <= 1.0 &&
         distorted_uv.y >= 0.0 && distorted_uv.y <= 1.0) {
        discard;
    }
    let tex_color = sample_video(distorted_uv, u.flags);
    let color = apply_color_transfer(
        tex_color.rgb,
        u.color_scale.xyz,
        u.color_offset_blend.xyz,
    );
    return vec4<f32>(color, 1.0);
}

// Fullscreen triangle to restore opaque output alpha (1.0) after DstAlpha
// compositing without touching RGB (paired with ColorWrites::ALPHA).
@vertex
fn vs_opaque_alpha(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[idx], 0.0, 1.0);
}

@fragment
fn fs_opaque_alpha() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
