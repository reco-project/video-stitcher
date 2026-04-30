// NV12 bilinear downscale compute shader.
//
// Samples NV12 plane views (Y as R8Unorm, UV as Rg8Unorm) at reduced
// resolution using hardware bilinear filtering, writes tightly-packed
// NV12 to a storage buffer for CPU readback.
//
// Each thread processes 4 horizontal pixels, producing 1 u32 of Y data
// and (on even rows) 1 u32 of UV data. Same pattern as rgba_to_nv12.wgsl.
//
// Dispatch: (ceil(out_width/4), out_height, 1)

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var uv_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

struct Params {
    out_width: u32,
    out_height: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

fn sample_y(uv: vec2<f32>) -> u32 {
    let y = textureSampleLevel(y_tex, samp, uv, 0.0).r;
    return u32(clamp(y * 255.0 + 0.5, 0.0, 255.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_x = gid.x; // each thread handles 4 pixels
    let row = gid.y;

    let px = thread_x * 4u;
    if px >= params.out_width || row >= params.out_height {
        return;
    }

    let inv_w = 1.0 / f32(params.out_width);
    let inv_h = 1.0 / f32(params.out_height);

    // Sample 4 Y values with bilinear filtering.
    let y0 = sample_y(vec2<f32>((f32(px) + 0.5) * inv_w, (f32(row) + 0.5) * inv_h));
    let y1 = sample_y(vec2<f32>((f32(px + 1u) + 0.5) * inv_w, (f32(row) + 0.5) * inv_h));
    let y2 = sample_y(vec2<f32>((f32(px + 2u) + 0.5) * inv_w, (f32(row) + 0.5) * inv_h));
    let y3 = sample_y(vec2<f32>((f32(px + 3u) + 0.5) * inv_w, (f32(row) + 0.5) * inv_h));

    let y_packed = y0 | (y1 << 8u) | (y2 << 16u) | (y3 << 24u);

    // Y plane index: row * (width/4) + thread_x
    let y_words_per_row = params.out_width / 4u;
    output[row * y_words_per_row + thread_x] = y_packed;

    // UV plane: only on even rows, 2 UV pairs per 4 pixels.
    if row % 2u == 0u {
        let uv_row_center = (f32(row) + 0.5) * inv_h;

        let uv0 = textureSampleLevel(uv_tex, samp,
            vec2<f32>((f32(px) + 1.0) * inv_w, uv_row_center), 0.0);
        let uv1 = textureSampleLevel(uv_tex, samp,
            vec2<f32>((f32(px + 2u) + 1.0) * inv_w, uv_row_center), 0.0);

        let u0 = u32(clamp(uv0.r * 255.0 + 0.5, 0.0, 255.0));
        let v0 = u32(clamp(uv0.g * 255.0 + 0.5, 0.0, 255.0));
        let u1 = u32(clamp(uv1.r * 255.0 + 0.5, 0.0, 255.0));
        let v1 = u32(clamp(uv1.g * 255.0 + 0.5, 0.0, 255.0));

        let uv_packed = u0 | (v0 << 8u) | (u1 << 16u) | (v1 << 24u);

        // UV plane starts after Y plane.
        let uv_base = y_words_per_row * params.out_height;
        let uv_row_idx = row / 2u;
        output[uv_base + uv_row_idx * y_words_per_row + thread_x] = uv_packed;
    }
}
