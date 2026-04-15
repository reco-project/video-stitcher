// Reco v2 -- RGBA → NV12 conversion compute shader
//
// Converts the RGBA render target to NV12 format on the GPU,
// eliminating CPU-side swscale and reducing readback bandwidth by 2.7x.
//
// NV12 layout in the output buffer (packed as array<u32>):
//   Y plane:  width × height bytes  (1 byte per pixel)
//   UV plane: width × (height/2) bytes  (interleaved Cb,Cr at half resolution)
//
// Each thread processes 4 horizontal pixels in one row, producing:
//   - 4 Y values packed into 1 u32 (always written)
//   - On even rows: 2 (Cb,Cr) pairs packed into 1 u32 (UV data)
//
// Dispatch: (ceil(width/4), height, 1)
// Requires: width divisible by 4 (true for all standard video resolutions)
//
// Color space: BT.709 sRGB → limited-range YCbCr
// This is the inverse of the sample_yuv() function in fisheye.wgsl.
// Input texture is Rgba8Unorm (sRGB values stored as-is, no hardware decode).

@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    width: u32,
    height: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

// BT.709 sRGB [0,1] → limited-range Y [16,235]
fn srgb_to_y(srgb: vec3<f32>) -> u32 {
    let y = 16.0 + 219.0 * (0.2126 * srgb.r + 0.7152 * srgb.g + 0.0722 * srgb.b);
    return u32(clamp(y + 0.5, 0.0, 255.0));
}

// BT.709 sRGB [0,1] → limited-range Cb [16,240]
fn srgb_to_cb(srgb: vec3<f32>) -> u32 {
    let cb = 128.0 + 224.0 * (-0.1146 * srgb.r - 0.3854 * srgb.g + 0.5 * srgb.b);
    return u32(clamp(cb + 0.5, 0.0, 255.0));
}

// BT.709 sRGB [0,1] → limited-range Cr [16,240]
fn srgb_to_cr(srgb: vec3<f32>) -> u32 {
    let cr = 128.0 + 224.0 * (0.5 * srgb.r - 0.4542 * srgb.g - 0.0458 * srgb.b);
    return u32(clamp(cr + 0.5, 0.0, 255.0));
}

// Each thread: 4 horizontal pixels → 1 u32 of Y data (+ 1 u32 of UV on even rows)
@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x * 4u;  // First pixel x of this thread's 4-pixel block
    let py = gid.y;        // Row

    let w = params.width;
    let h = params.height;

    if px >= w || py >= h {
        return;
    }

    // Load 4 RGBA pixels from the render target (Rgba8Unorm = sRGB values as-is).
    let rgb0 = textureLoad(input, vec2<u32>(px, py), 0).rgb;
    let rgb1 = textureLoad(input, vec2<u32>(min(px + 1u, w - 1u), py), 0).rgb;
    let rgb2 = textureLoad(input, vec2<u32>(min(px + 2u, w - 1u), py), 0).rgb;
    let rgb3 = textureLoad(input, vec2<u32>(min(px + 3u, w - 1u), py), 0).rgb;

    // Pack 4 Y values into one u32 (little-endian: Y0 in LSB)
    let y0 = srgb_to_y(rgb0);
    let y1 = srgb_to_y(rgb1);
    let y2 = srgb_to_y(rgb2);
    let y3 = srgb_to_y(rgb3);
    let y_packed = y0 | (y1 << 8u) | (y2 << 16u) | (y3 << 24u);

    // Y plane: word index = (py * width + px) / 4
    let y_word_idx = (py * w + px) / 4u;
    output[y_word_idx] = y_packed;

    // UV plane: only on even rows (one chroma sample per 2×2 block)
    if (py & 1u) == 0u {
        // Load pixels from the row below for chroma averaging
        let ny = min(py + 1u, h - 1u);
        let rgb0b = textureLoad(input, vec2<u32>(px, ny), 0).rgb;
        let rgb1b = textureLoad(input, vec2<u32>(min(px + 1u, w - 1u), ny), 0).rgb;
        let rgb2b = textureLoad(input, vec2<u32>(min(px + 2u, w - 1u), ny), 0).rgb;
        let rgb3b = textureLoad(input, vec2<u32>(min(px + 3u, w - 1u), ny), 0).rgb;

        // Average each 2×2 block for chroma
        let avg_left = (rgb0 + rgb1 + rgb0b + rgb1b) * 0.25;
        let avg_right = (rgb2 + rgb3 + rgb2b + rgb3b) * 0.25;

        let cb0 = srgb_to_cb(avg_left);
        let cr0 = srgb_to_cr(avg_left);
        let cb1 = srgb_to_cb(avg_right);
        let cr1 = srgb_to_cr(avg_right);

        // NV12 UV plane: interleaved [Cb0, Cr0, Cb1, Cr1] → 1 u32
        let uv_packed = cb0 | (cr0 << 8u) | (cb1 << 16u) | (cr1 << 24u);

        // UV plane offset: after Y plane (w * h bytes = w * h / 4 u32s)
        // UV row index: py / 2 (half vertical resolution)
        // UV row stride: width bytes = width / 4 u32s (same as Y stride)
        // UV word index within row: px / 4 (4 pixels → 2 chroma pairs → 4 bytes → 1 u32)
        let uv_plane_words = (w * h) / 4u;
        let uv_row = py / 2u;
        let uv_word_idx = uv_plane_words + uv_row * (w / 4u) + gid.x;
        output[uv_word_idx] = uv_packed;
    }
}
