// Ray basis includes the viewport FOV, aspect and rig rotation.
struct Uniforms {
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // radius / painted height, horizontal arc in radians
    shape: vec4<f32>,
    flags: vec4<u32>,
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray = u.forward.xyz + in.ndc.x * u.right.xyz + in.ndc.y * u.up.xyz;
    let horizontal = length(ray.xz);
    if horizontal < 1e-9 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let uv = vec2(0.5 + atan2(ray.x, -ray.z) / u.shape.y,
                  0.5 - ray.y / horizontal * u.shape.x);
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return sample_video(uv, u.flags);
}
