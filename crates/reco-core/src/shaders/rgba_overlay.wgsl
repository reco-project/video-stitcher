struct OverlayParams {
    reference_size: vec2<f32>,
    output_size: vec2<f32>,
};

@group(0) @binding(0) var overlay_texture: texture_2d<f32>;
@group(0) @binding(1) var overlay_sampler: sampler;
@group(0) @binding(2) var<uniform> params: OverlayParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    let position = positions[vertex_index];
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = vec2<f32>((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let scale = min(
        params.output_size.x / params.reference_size.x,
        params.output_size.y / params.reference_size.y,
    );
    let fitted_size = params.reference_size * scale;
    let fitted_origin = (params.output_size - fitted_size) * 0.5;
    let output_pixel = input.uv * params.output_size;
    let overlay_uv = (output_pixel - fitted_origin) / fitted_size;
    if (overlay_uv.x < 0.0 || overlay_uv.y < 0.0 || overlay_uv.x > 1.0 || overlay_uv.y > 1.0) {
        return vec4<f32>(0.0);
    }
    return textureSample(overlay_texture, overlay_sampler, overlay_uv);
}
