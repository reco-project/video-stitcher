// GPU pack shader — tile + optional downscale for stacked-video replay.
//
// One dispatch per tile. The dispatcher binds the tile's three source
// textures (Y / U / V as R8Unorm) and a uniform describing where in
// the output atlas this tile lands, then runs three kernels: pack_y,
// pack_u, pack_v. Each kernel samples its source plane with linear
// filtering (bilinear downscale is free when output dims < source
// dims) and writes packed bytes into the shared atlas storage buffer.
//
// Byte packing: each thread handles 4 output columns at once and
// writes one u32 to the atlas. This avoids byte-level atomics which
// WGSL doesn't offer on storage buffers. Each output row fits
// `atlas_width / 4` u32s; the Y plane row stride in u32s is
// `atlas_width_y / 4` and the UV plane stride is `atlas_width_uv / 4`.
//
// Atlas layout in the storage buffer (YUV420P planar, tight, matches
// what `reco_io::stacked_video::pack_yuv420p` produces on the CPU
// path so the encoder side doesn't care which fill path filled the
// buffer):
//
//   offset 0              .. y_size       -> Y plane  (atlas_w * atlas_h bytes)
//   offset y_size         .. y_size + uv  -> U plane  ((atlas_w/2) * (atlas_h/2) bytes)
//   offset y_size + uv    .. y_size + 2uv -> V plane  (same size as U)
//
// All offsets / sizes are in bytes; the storage view is `array<u32>`
// so the byte offset is divided by 4 at binding time (handled in the
// Rust side via the uniform `plane_u32_offset` per-plane).

struct PackParams {
    // Tile placement in the atlas (in output-plane pixels, post-scale).
    // Y plane uses these; U/V planes use them divided by 2.
    tile_y_row_offset: u32,   // where this tile's Y plane starts, in rows
    tile_y_col_offset: u32,   // where this tile's Y plane starts, in columns
    // Output tile dims after downscale. Input dims are implicit in the
    // texture; the sampler handles the scale.
    out_tile_width: u32,      // Y plane cols this tile occupies
    out_tile_height: u32,     // Y plane rows this tile occupies
    // Atlas plane stride (bytes, == plane width because tight).
    atlas_y_stride: u32,
    atlas_uv_stride: u32,
    // Per-plane u32 offsets into the shared atlas buffer.
    y_plane_u32_offset: u32,
    u_plane_u32_offset: u32,
    v_plane_u32_offset: u32,
    // Padding for std140 alignment on some backends.
    _pad: u32,
}

@group(0) @binding(0) var src_y: texture_2d<f32>;
@group(0) @binding(1) var src_u: texture_2d<f32>;
@group(0) @binding(2) var src_v: texture_2d<f32>;
@group(0) @binding(3) var src_sampler: sampler;
@group(0) @binding(4) var<storage, read_write> atlas: array<u32>;
@group(0) @binding(5) var<uniform> params: PackParams;

// Pack four normalized [0, 1] luma samples into one little-endian u32.
// Rounds half-to-even to match CPU memcpy output byte-for-byte when the
// inputs are already discrete R8Unorm (sampler returns the exact byte
// value / 255.0 for aligned 1:1 reads).
fn pack4(s0: f32, s1: f32, s2: f32, s3: f32) -> u32 {
    let b0 = u32(clamp(s0, 0.0, 1.0) * 255.0 + 0.5);
    let b1 = u32(clamp(s1, 0.0, 1.0) * 255.0 + 0.5);
    let b2 = u32(clamp(s2, 0.0, 1.0) * 255.0 + 0.5);
    let b3 = u32(clamp(s3, 0.0, 1.0) * 255.0 + 0.5);
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

// Sample one normalized UV from a single-channel R8 / R16 source.
// Using `textureSampleLevel` (not `textureLoad`) so the sampler's
// linear filter can do the downscale for us when output_tile_dims <
// source texture dims. For 1:1 no-scale cases the filter at a pixel
// center returns the exact source byte / 255.0.
fn sample_y(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(src_y, src_sampler, uv, 0.0).r;
}
fn sample_u(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(src_u, src_sampler, uv, 0.0).r;
}
fn sample_v(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(src_v, src_sampler, uv, 0.0).r;
}

// Y plane: dispatch (out_tile_width / 4, out_tile_height, 1) with wg size (8, 8, 1).
// Each thread writes one u32 covering 4 horizontally-adjacent output pixels.
@compute @workgroup_size(8, 8, 1)
fn pack_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_x = gid.x;
    let row    = gid.y;
    if (row >= params.out_tile_height) { return; }
    let quad_count = params.out_tile_width / 4u;
    if (quad_x >= quad_count) { return; }

    let base_col = quad_x * 4u;
    let tile_w = f32(params.out_tile_width);
    let tile_h = f32(params.out_tile_height);
    let y_norm = (f32(row) + 0.5) / tile_h;
    let s0 = sample_y(vec2<f32>((f32(base_col + 0u) + 0.5) / tile_w, y_norm));
    let s1 = sample_y(vec2<f32>((f32(base_col + 1u) + 0.5) / tile_w, y_norm));
    let s2 = sample_y(vec2<f32>((f32(base_col + 2u) + 0.5) / tile_w, y_norm));
    let s3 = sample_y(vec2<f32>((f32(base_col + 3u) + 0.5) / tile_w, y_norm));
    let packed = pack4(s0, s1, s2, s3);

    // Atlas index: byte offset = (tile_y_row_offset + row) * atlas_y_stride
    //                           + (tile_y_col_offset + base_col)
    // then divided by 4 for u32 index. Since both offset terms are
    // multiples of 4 when tile placement is aligned, the u32 index is
    // (byte_offset >> 2).
    let byte_offset =
          (params.tile_y_row_offset + row) * params.atlas_y_stride
        + (params.tile_y_col_offset + base_col);
    atlas[params.y_plane_u32_offset + (byte_offset >> 2u)] = packed;
}

// U plane: dispatch ((out_tile_width/2) / 4, out_tile_height / 2, 1).
// YUV420P half-resolution chroma.
@compute @workgroup_size(8, 8, 1)
fn pack_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_x = gid.x;
    let row    = gid.y;
    let uv_height = params.out_tile_height / 2u;
    if (row >= uv_height) { return; }
    let uv_width = params.out_tile_width / 2u;
    let quad_count = uv_width / 4u;
    if (quad_x >= quad_count) { return; }

    let base_col = quad_x * 4u;
    let uv_w_f = f32(uv_width);
    let uv_h_f = f32(uv_height);
    let y_norm = (f32(row) + 0.5) / uv_h_f;
    let s0 = sample_u(vec2<f32>((f32(base_col + 0u) + 0.5) / uv_w_f, y_norm));
    let s1 = sample_u(vec2<f32>((f32(base_col + 1u) + 0.5) / uv_w_f, y_norm));
    let s2 = sample_u(vec2<f32>((f32(base_col + 2u) + 0.5) / uv_w_f, y_norm));
    let s3 = sample_u(vec2<f32>((f32(base_col + 3u) + 0.5) / uv_w_f, y_norm));
    let packed = pack4(s0, s1, s2, s3);

    let byte_offset =
          ((params.tile_y_row_offset / 2u) + row) * params.atlas_uv_stride
        + ((params.tile_y_col_offset / 2u) + base_col);
    atlas[params.u_plane_u32_offset + (byte_offset >> 2u)] = packed;
}

// V plane: same dispatch shape as U.
@compute @workgroup_size(8, 8, 1)
fn pack_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quad_x = gid.x;
    let row    = gid.y;
    let uv_height = params.out_tile_height / 2u;
    if (row >= uv_height) { return; }
    let uv_width = params.out_tile_width / 2u;
    let quad_count = uv_width / 4u;
    if (quad_x >= quad_count) { return; }

    let base_col = quad_x * 4u;
    let uv_w_f = f32(uv_width);
    let uv_h_f = f32(uv_height);
    let y_norm = (f32(row) + 0.5) / uv_h_f;
    let s0 = sample_v(vec2<f32>((f32(base_col + 0u) + 0.5) / uv_w_f, y_norm));
    let s1 = sample_v(vec2<f32>((f32(base_col + 1u) + 0.5) / uv_w_f, y_norm));
    let s2 = sample_v(vec2<f32>((f32(base_col + 2u) + 0.5) / uv_w_f, y_norm));
    let s3 = sample_v(vec2<f32>((f32(base_col + 3u) + 0.5) / uv_w_f, y_norm));
    let packed = pack4(s0, s1, s2, s3);

    let byte_offset =
          ((params.tile_y_row_offset / 2u) + row) * params.atlas_uv_stride
        + ((params.tile_y_col_offset / 2u) + base_col);
    atlas[params.v_plane_u32_offset + (byte_offset >> 2u)] = packed;
}
