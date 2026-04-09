// Reco v2 -- Separable convolution compute shader
//
// Applies a 1D convolution kernel either horizontally or vertically.
// Used for Gaussian blur (two passes: H then V) and Scharr derivatives.
//
// The kernel coefficients are packed into vec4<f32> groups in the uniform.
// Maximum supported kernel radius is 7 (15 taps).
//
// Dispatch: (ceil(width/16), ceil(height/16), 1)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    width: u32,
    height: u32,
    radius: u32,      // kernel half-width (full size = 2*radius + 1)
    direction: u32,   // 0 = horizontal, 1 = vertical
    // Kernel packed as 4 vec4s = 16 f32 slots (enough for 15 taps + 1 padding)
    k0: vec4<f32>,
    k1: vec4<f32>,
    k2: vec4<f32>,
    k3: vec4<f32>,
}
@group(0) @binding(2) var<uniform> params: Params;

// Unpack kernel coefficient by index (0..14)
fn get_kernel(i: u32) -> f32 {
    let vec_idx = i / 4u;
    let comp = i % 4u;
    var v: vec4<f32>;
    if vec_idx == 0u { v = params.k0; }
    else if vec_idx == 1u { v = params.k1; }
    else if vec_idx == 2u { v = params.k2; }
    else { v = params.k3; }
    if comp == 0u { return v.x; }
    else if comp == 1u { return v.y; }
    else if comp == 2u { return v.z; }
    else { return v.w; }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if gid.x >= w || gid.y >= h {
        return;
    }

    let x = i32(gid.x);
    let y = i32(gid.y);
    let r = i32(params.radius);

    var sum = 0.0;
    for (var k = -r; k <= r; k++) {
        let ki = u32(k + r);
        let coeff = get_kernel(ki);

        var sx: i32;
        var sy: i32;
        if params.direction == 0u {
            sx = clamp(x + k, 0, i32(w) - 1);
            sy = y;
        } else {
            sx = x;
            sy = clamp(y + k, 0, i32(h) - 1);
        }
        sum += coeff * input[u32(sy) * w + u32(sx)];
    }

    output[gid.y * w + gid.x] = sum;
}
