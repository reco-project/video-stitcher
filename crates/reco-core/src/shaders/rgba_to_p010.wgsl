// One invocation owns a 2x2 block. Each output u32 holds two P010LE
// samples in the high ten bits of their respective 16-bit words.
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    width: u32,
    height: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override kr: f32;
override kg: f32;
override kb: f32;
override cb_r: f32;
override cb_g: f32;
override cr_g: f32;
override cr_b: f32;
override y_offset: f32;
override y_scale: f32;
override c_scale: f32;

fn quantize(code: f32) -> u32 {
    return u32(clamp(code + 0.5, 0.0, 1023.0));
}

fn luma(rgb: vec3<f32>) -> u32 {
    return quantize(y_offset + y_scale * dot(vec3<f32>(kr, kg, kb), rgb));
}

fn pack(low: u32, high: u32) -> u32 {
    return (low << 6u) | (high << 22u);
}

@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let xy = gid.xy * 2u;
    if xy.x >= params.width || xy.y >= params.height {
        return;
    }

    let a = textureLoad(input, xy, 0).rgb;
    let b = textureLoad(input, xy + vec2<u32>(1u, 0u), 0).rgb;
    let c = textureLoad(input, xy + vec2<u32>(0u, 1u), 0).rgb;
    let d = textureLoad(input, xy + vec2<u32>(1u, 1u), 0).rgb;

    let stride = params.width / 2u;
    let y = xy.y * stride + gid.x;
    output[y] = pack(luma(a), luma(b));
    output[y + stride] = pack(luma(c), luma(d));

    let rgb = (a + b + c + d) * 0.25;
    let cb = quantize(512.0 + c_scale * dot(vec3<f32>(-cb_r, -cb_g, 0.5), rgb));
    let cr = quantize(512.0 + c_scale * dot(vec3<f32>(0.5, -cr_g, -cr_b), rgb));
    let uv = (params.height + gid.y) * stride + gid.x;
    output[uv] = pack(cb, cr);
}
