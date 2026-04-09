// Reco v2 -- RGBA → Grayscale f32 compute shader
//
// Converts an RGBA8 texture to a flat f32 storage buffer containing
// luminance values in [0, 1]. Used to prepare undistorted frames for
// AKAZE feature detection without CPU readback of the RGBA texture.
//
// Luminance: 0.299R + 0.587G + 0.114B (BT.601)
//
// Dispatch: (ceil(width/16), ceil(height/16), 1)

@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    width: u32,
    height: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let rgb = textureLoad(input, vec2<u32>(gid.x, gid.y), 0).rgb;
    let gray = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    output[gid.y * params.width + gid.x] = gray;
}
