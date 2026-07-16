// Reco v2 -- lookahead pool bit-depth downconversion (P010 -> NV12)
//
// Copies one plane (Y or UV) of a decoded stereo frame from its source
// format into an 8-bit destination plane, as a GPU-resident render pass.
//
// Why this exists: the lookahead pool buffers N future decoded stereo
// frames so the AI panner/tracker can smooth camera trajectory using
// future world-state (see `session::vram_pool::VramPool` and
// `interop::d3d11::D3d11StagingPool`). Those buffered frames are also
// what the final stitch render consumes when it catches up to them - the
// pool is not an AI-only side buffer. For 10-bit sources (`GpuPixelFormat::
// P010`, e.g. DJI Action 4 HEVC), each pool slot costs 2x the bytes of an
// 8-bit source, which is what makes deep lookahead windows VRAM-expensive
// on lower-VRAM cards (see FRICTION.md "Lookahead pool VRAM cost scales
// with source bit depth"). Downconverting to 8-bit (NV12) before the
// frame enters the long-lived pool roughly halves that cost.
//
// This is a genuine, opt-in quality/memory tradeoff, not a free lunch:
// the same buffered frame feeds the final render, so downconverting here
// discards the extra precision a 10-bit source captured (more banding
// risk in smooth gradients - sky, pitch grass, floodlit surfaces). It
// must stay opt-in and default-off; see `LookaheadBitDepth` in
// `session/vram_pool.rs`.
//
// One shader body handles both planes: the Y plane's source is a
// single-channel texture (`R8Unorm`/`R16Unorm`), the UV plane's source is
// two-channel (`Rg8Unorm`/`Rg16Unorm`). `textureLoad` always returns a
// `vec4<f32>` regardless of the source's channel count (missing channels
// read as 0.0/1.0), and the render target's format (`R8Unorm` for Y,
// `Rg8Unorm` for UV) determines which of the output's components actually
// get written - so no separate entry point is needed per plane. No manual
// bit-shift/rescale math is needed either: wgpu/WebGPU normalizes both
// 8-bit `[0,255]` and 16-bit `[0,65535]` Unorm source formats to `[0.0,
// 1.0]` on load, and quantizes the `[0.0,1.0]` fragment output back down
// to the target format's bit depth on write.
//
// Source and destination are always the same resolution (the Y plane is
// full-res in and out, the UV plane is half-res in and out - this shader
// only changes bit depth, never scale), so a direct `textureLoad` at the
// fragment's own integer pixel coordinate is used instead of a filtered
// `textureSample`. This is both more precise (an exact 1:1 texel copy,
// zero interpolation) and sidesteps needing the source format to support
// linear filtering (16-bit Unorm texture formats are not guaranteed
// filterable on every wgpu backend).
//
// Draws a fullscreen triangle (3 vertices, no vertex/index buffer) per
// invocation - see `render::lookahead_downconvert::LookaheadDownconverter`.

// Standard oversized fullscreen triangle: covers the [-1,1] clip-space
// square with a single triangle (no diagonal seam, no vertex buffer).
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@group(0) @binding(0) var src: texture_2d<f32>;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<i32>(pos.xy), 0);
}
