// Reco v2 -- Fast Explicit Diffusion (FED) step compute shader
//
// Performs one step of nonlinear diffusion with double-buffering.
// Reads from lt_in, writes to lt_out. The caller swaps buffers between
// successive FED tau steps (ping-pong pattern).
//
// Per-pixel reformulation of the diffusion operator:
//   delta = step_size * sum of neighbor flows
//   lt_out[y,x] = lt_in[y,x] + delta
//
// where each neighbor flow is:
//   flow = conductivity[here] * conductivity[neighbor] * (lt[neighbor] - lt[here])
//
// This is mathematically equivalent to the CPU implementation's
// separate flow-computation + accumulation, but avoids write hazards
// by using double-buffering (Jacobi iteration).
//
// Boundary: clamped (no flow across image edges).
//
// Dispatch: (ceil(width/16), ceil(height/16), 1)

@group(0) @binding(0) var<storage, read> lt_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> lt_out: array<f32>;
@group(0) @binding(2) var<storage, read> lflow: array<f32>;

struct Params {
    width: u32,
    height: u32,
    step_size: f32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if gid.x >= w || gid.y >= h {
        return;
    }

    let x = gid.x;
    let y = gid.y;
    let idx = y * w + x;

    let c = lflow[idx];
    let val = lt_in[idx];
    var delta = 0.0;

    // Right neighbor
    if x + 1u < w {
        let n = idx + 1u;
        delta += c * lflow[n] * (lt_in[n] - val);
    }
    // Left neighbor
    if x > 0u {
        let n = idx - 1u;
        delta += lflow[n] * c * (lt_in[n] - val);
    }
    // Down neighbor
    if y + 1u < h {
        let n = idx + w;
        delta += c * lflow[n] * (lt_in[n] - val);
    }
    // Up neighbor
    if y > 0u {
        let n = idx - w;
        delta += lflow[n] * c * (lt_in[n] - val);
    }

    lt_out[idx] = val + params.step_size * delta;
}
