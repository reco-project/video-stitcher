// Universal color grading compute shader.
//
// Input:  Rgba8Unorm texture (linear-light from demosaic or any source)
// Output: Rgba8Unorm texture (graded, gamma-encoded, display-ready)
//
// Applies: brightness, saturation, gamma. Future: 1D LUT, tone mapping.
// Designed to slot into ANY input pipeline (Bayer, NV12, file, OBS).

struct ColorGradeParams {
    brightness: f32,
    saturation: f32,
    gamma: f32,
    _pad: f32,
}

@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ColorGradeParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let rgb = textureLoad(input, vec2<i32>(gid.xy), 0).rgb;

    // Brightness (linear space, before gamma)
    let scaled = rgb * params.brightness;

    // Saturation in YCbCr
    let y_luma = 0.2126 * scaled.x + 0.7152 * scaled.y + 0.0722 * scaled.z;
    let saturated = mix(vec3<f32>(y_luma), scaled, params.saturation);

    // Gamma (0.5 = sqrt, 1.0 = linear, 0.4545 = sRGB approx)
    let clamped = clamp(saturated, vec3<f32>(0.0), vec3<f32>(1.0));
    let graded = pow(clamped, vec3<f32>(params.gamma));

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(graded, 1.0));
}
