# reco-core API Friction - OBS Plugin Consumer Perspective

Issues encountered while building `reco-obs` as a plugin consumer of `reco-core`.

## 1. No RGBA readback helper on StitchPipeline

**Problem:** `StitchPipeline::render_to_target()` returns a `CommandBuffer` that
renders to an internal wgpu texture, but there's no way to get the pixel data
back to CPU without manually creating a staging buffer, issuing a copy command,
mapping the buffer, and handling row padding (wgpu's 256-byte alignment).

The `Nv12Converter` does this internally (with triple-buffering) but is coupled
to NV12 output and lives in a separate type.

**Impact:** ~40 lines of boilerplate in `source.rs` (`render_and_readback`) for
what should be a one-liner. Every plugin consumer that needs CPU pixels will
duplicate this.

**Suggestion:** Add a `ReadbackHelper` or a method like
`pipeline.readback_rgba() -> Option<&[u8]>` that handles staging buffer
management, row padding, and async mapping. Could follow the same triple-buffer
pattern as `Nv12Converter` but output RGBA.

## 2. YuvPlanes requires pre-extracted plane pointers

**Problem:** `StitchPipeline::render_to_target()` takes `YuvPlanes<'a>` which is
`{ y: &[u8], u: &[u8], v: &[u8] }` - three separate slice references.

An OBS plugin consumer receives frame data as `obs_source_frame` which has
`data[MAX_AV_PLANES]` (array of plane pointers) + `linesize[MAX_AV_PLANES]`
(stride per plane) + `width`/`height` + `format` enum.

**Impact:** The consumer must:
1. Know which OBS video format maps to which reco input format
2. Handle stride mismatches (OBS may have padding per row, reco expects tight packing)
3. Extract the right plane pointers based on format

**Suggestion:** A `RawFrameView` type that accepts pointer + stride + dimensions
per plane (or a contiguous buffer + format enum) would reduce this impedance
mismatch. Something like:
```rust
pub struct FramePlaneView<'a> {
    pub data: &'a [u8],
    pub stride: u32,  // bytes per row (may include padding)
    pub width: u32,
    pub height: u32,
}
```

## 3. GpuContext::new() is async

**Problem:** `GpuContext::new()` is an `async fn` that must be awaited. OBS
plugin callbacks are synchronous C functions. This requires pulling in `pollster`
(or equivalent) to block on the future.

**Impact:** Minor - just `pollster::block_on(GpuContext::new())`. But it adds a
dependency and is slightly surprising for C FFI consumers.

**Note:** `GpuContext::from_device_queue()` exists as a sync alternative if you
already have a wgpu device, which is the right escape hatch. The async
constructor is the natural choice for standalone use.

## 4. No way to share wgpu device with OBS

**Problem:** OBS has its own graphics context (OpenGL/D3D11) and reco-core has
its own wgpu context. There's no interop path between them, so rendered frames
must be copied through CPU memory (GPU -> staging buffer -> CPU -> OBS texture).

**Impact:** One full-resolution RGBA copy per frame. At 1920x1080 that's ~8MB
per frame, which is significant at 60fps (~480MB/s of memory bandwidth wasted).

**Note:** This is fundamentally an OBS architecture constraint, not a reco-core
bug. The `GpuContext::from_device_queue()` method would help if OBS used wgpu,
but it doesn't. Platform-specific solutions (DMA-BUF on Linux, shared D3D11
textures on Windows) would need new interop code in reco-core.
