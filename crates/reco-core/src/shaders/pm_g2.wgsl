// Reco v2 -- Perona-Malik g2 conductivity compute shader
//
// Computes the nonlinear diffusion conductivity from image gradients:
//   c(x,y) = 1 / (1 + (Lx^2 + Ly^2) / k^2)
//
// This controls how much diffusion occurs at each pixel. High gradients
// (edges) get low conductivity (preserved), smooth regions get high
// conductivity (blurred).
//
// Dispatch: (ceil(width/16), ceil(height/16), 1)

@group(0) @binding(0) var<storage, read> lx: array<f32>;
@group(0) @binding(1) var<storage, read> ly: array<f32>;
@group(0) @binding(2) var<storage, read_write> lflow: array<f32>;

struct Params {
    width: u32,
    height: u32,
    inv_k_sq: f32,  // 1 / (k * k), precomputed
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let idx = gid.y * params.width + gid.x;
    let gx = lx[idx];
    let gy = ly[idx];
    lflow[idx] = 1.0 / (1.0 + params.inv_k_sq * (gx * gx + gy * gy));
}
