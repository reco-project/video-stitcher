// Fixed cylindrical output; the matrices come from the calibrated L-shape rig.
struct Uniforms {
    to_plane: mat4x4<f32>,
    // half arc, tan half vertical FOV, blend width, correction
    shape: vec4<f32>,
    intrinsics: vec4<f32>,
    distortion: vec4<f32>,
    flags: vec4<u32>,
    base_to_plane: mat4x4<f32>,
    base_intrinsics: vec4<f32>,
    base_distortion: vec4<f32>,
};
@group(0) @binding(0) var t_y: texture_2d<f32>;
@group(0) @binding(1) var t_u: texture_2d<f32>;
@group(0) @binding(2) var t_v: texture_2d<f32>;
@group(0) @binding(3) var s_video: sampler;
@group(1) @binding(0) var<uniform> u: Uniforms;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VertexOutput;
    out.position = vec4(positions[index], 0.0, 1.0);
    out.ndc = positions[index];
    return out;
}
// w is source support; z is the calibrated seam coordinate.
fn source_sample(chart_ray: vec4<f32>, to_plane: mat4x4<f32>, intrinsics: vec4<f32>, distortion: vec4<f32>, correction: f32) -> vec4<f32> {
    let ray = to_plane * chart_ray;
    let eye = to_plane[3];
    if abs(ray.z) < 1e-12 { return vec4(0.0); }
    let distance = -eye.z / ray.z;
    if distance <= 0.0 { return vec4(0.0); }
    let local = eye.xy + distance * ray.xy;
    if any(abs(local) > vec2(0.5)) { return vec4(0.0); }
    let extended = vec2(2.0 * local.x + 0.5, 0.5 - 2.0 * local.y);
    let uv = lens_uv(extended, intrinsics, distortion, correction);
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) { return vec4(0.0); }
    return vec4(uv, extended.x, 1.0);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let angle = in.ndc.x * u.shape.x;
    let ray = vec4(sin(angle), in.ndc.y * u.shape.y, -cos(angle), 0.0);
    let sample = source_sample(ray, u.to_plane, u.intrinsics, u.distortion, u.shape.w);
    if sample.w == 0.0 { discard; }
    var alpha = 1.0;
    if u.flags.x == 1u && u.shape.z > 0.0 && sample.z < u.shape.z {
        let base = source_sample(ray, u.base_to_plane, u.base_intrinsics, u.base_distortion, 1.0);
        if base.w > 0.0 { alpha = smoothstep(0.0, u.shape.z, sample.z); }
    }
    return vec4(sample_video(sample.xy, u.flags).rgb, alpha);
}
