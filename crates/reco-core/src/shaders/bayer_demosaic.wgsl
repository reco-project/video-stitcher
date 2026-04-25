// Bayer RGGB demosaic compute shader.
//
// Input:  raw 10-bit Bayer (R16Uint texture, RGGB pattern)
// Output: Rgba8Unorm texture (linear-light, WB+CCM corrected, no gamma)
//
// Pipeline: black subtract -> LSC -> bilinear demosaic -> WB -> CCM
// Color grading (brightness, saturation, gamma) is handled by the
// separate color_grade.wgsl pass, which works on any input path.
// Designed for IMX477 direct V4L2 capture (bypassing NVIDIA ISP).

struct IspParams {
    width: u32,
    height: u32,
    black_level: f32,
    lsc_strength: f32,
    wb_r: f32,
    wb_g: f32,
    wb_b: f32,
    white_level: f32,
    ccm_row0: vec4<f32>,
    ccm_row1: vec4<f32>,
    ccm_row2: vec4<f32>,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var bayer_in: texture_2d<u32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: IspParams;

fn sample_raw(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    // Raw is 16-bit (10-bit left-shifted by 6). Right-shift to 10-bit range.
    let raw = f32(textureLoad(bayer_in, vec2<i32>(cx, cy), 0).r) / 64.0;
    let corrected = max(raw - params.black_level, 0.0) / params.white_level;
    // LSC at sample position (radial vignette correction)
    let sx = f32(cx) / f32(params.width) - 0.5;
    let sy = f32(cy) / f32(params.height) - 0.5;
    let r2 = 4.0 * (sx * sx + sy * sy);
    return corrected * (1.0 + params.lsc_strength * r2);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    if (x >= i32(params.width) || y >= i32(params.height)) {
        return;
    }

    // RGGB Bayer pattern:
    //   (even_x, even_y) = R
    //   (odd_x,  even_y) = Gr
    //   (even_x, odd_y)  = Gb
    //   (odd_x,  odd_y)  = B
    let bx = u32(x) % 2u;
    let by = u32(y) % 2u;

    var r: f32;
    var g: f32;
    var b: f32;

    if (bx == 0u && by == 0u) {
        // Red pixel: R known, interpolate G and B
        r = sample_raw(x, y);
        g = (sample_raw(x - 1, y) + sample_raw(x + 1, y)
           + sample_raw(x, y - 1) + sample_raw(x, y + 1)) * 0.25;
        b = (sample_raw(x - 1, y - 1) + sample_raw(x + 1, y - 1)
           + sample_raw(x - 1, y + 1) + sample_raw(x + 1, y + 1)) * 0.25;
    } else if (bx == 1u && by == 0u) {
        // Green in red row: G known, R from horizontal, B from vertical
        g = sample_raw(x, y);
        r = (sample_raw(x - 1, y) + sample_raw(x + 1, y)) * 0.5;
        b = (sample_raw(x, y - 1) + sample_raw(x, y + 1)) * 0.5;
    } else if (bx == 0u && by == 1u) {
        // Green in blue row: G known, B from horizontal, R from vertical
        g = sample_raw(x, y);
        b = (sample_raw(x - 1, y) + sample_raw(x + 1, y)) * 0.5;
        r = (sample_raw(x, y - 1) + sample_raw(x, y + 1)) * 0.5;
    } else {
        // Blue pixel: B known, interpolate G and R
        b = sample_raw(x, y);
        g = (sample_raw(x - 1, y) + sample_raw(x + 1, y)
           + sample_raw(x, y - 1) + sample_raw(x, y + 1)) * 0.25;
        r = (sample_raw(x - 1, y - 1) + sample_raw(x + 1, y - 1)
           + sample_raw(x - 1, y + 1) + sample_raw(x + 1, y + 1)) * 0.25;
    }

    // White balance
    r *= params.wb_r;
    g *= params.wb_g;
    b *= params.wb_b;

    // CCM (3x3 color correction matrix)
    let rgb_in = vec3<f32>(r, g, b);
    let corrected = vec3<f32>(
        dot(params.ccm_row0.xyz, rgb_in),
        dot(params.ccm_row1.xyz, rgb_in),
        dot(params.ccm_row2.xyz, rgb_in),
    );

    // Output linear-light WB+CCM corrected RGB (no gamma, no saturation).
    // Color grading is applied by the separate ColorGradePass.
    let clamped = clamp(corrected, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, vec2<i32>(x, y), vec4<f32>(clamped, 1.0));
}
