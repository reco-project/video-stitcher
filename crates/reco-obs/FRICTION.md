# OBS Plugin Consumer API Friction

Friction points encountered while building `reco-obs` as a plugin
consumer of `reco-core` / `reco-io`. Active items at the top;
resolved items archived at the bottom with the PR that fixed them.

## Active

### A1. YuvPlanes requires tight packing, not stride-aware

**Impact**: Medium. Every OBS-style consumer has to copy or re-pack
incoming frames before handing to reco-core.

`StitchPipeline::render_to_target()` takes `YuvPlanes<'a>` which is
`{ y: &[u8], u: &[u8], v: &[u8] }` - three separate tightly-packed
slice references. An OBS plugin receives frames as `obs_source_frame`
with `data[MAX_AV_PLANES]` pointers, `linesize[MAX_AV_PLANES]`
strides (which may include row padding), `width` / `height`, and a
`format` enum.

**Impact**: The consumer must (1) map OBS format → reco input format,
(2) handle stride mismatches by copying tightly, (3) extract the
right plane pointers based on format. Common code that every
realistic live-input consumer will write.

**Suggested addition**:
```rust
pub struct FramePlaneView<'a> {
    pub data: &'a [u8],
    pub stride: u32,  // bytes per row, may include padding
    pub width: u32,
    pub height: u32,
}

pub struct StridedYuvPlanes<'a> {
    pub y: FramePlaneView<'a>,
    pub u: FramePlaneView<'a>,
    pub v: FramePlaneView<'a>,
}
```
Plus a conversion helper that copies stride → tight for the slow path.

### A2. GpuContext::new() is async

**Impact**: Very minor. Noted across all consumers (also in
reco-gui/FRICTION.md A6). OBS plugin callbacks are synchronous C
functions, so standalone `GpuContext::new` requires pulling in
pollster. `GpuContext::from_device_queue` is the right sync escape
hatch when you already have a wgpu device.

### A3. No OBS-level wgpu interop

**Impact**: Fundamental to OBS architecture, not a reco-core bug.

OBS uses its own graphics context (OpenGL / D3D11) and reco-core uses
wgpu. There's no interop path, so rendered frames must be copied
through CPU (GPU → staging → CPU → OBS texture). At 1080p that's
~8 MB per frame; at 60fps = ~480 MB/s memory bandwidth wasted.

Platform-specific solutions (DMA-BUF on Linux, shared D3D11 textures
on Windows) would need new interop code in reco-core. The
`GpuContext::from_device_queue()` method would help if OBS moved to
wgpu, which it hasn't.

Tracking here as a known limit; not actionable at the reco-core
level without a specific interop target.

### A4. Live camera input has no high-level consumer helper

**Impact**: High for the "Reco as OBS input source" use case.

reco-io has `FfmpegFileSource`, `SmartFileSource`, and zero-copy
CUDA adapters - all oriented toward decoding from files. For OBS
(and future live-camera consumers), the incoming data comes from a
callback with frame pointers + timing, not from a file path.

There is currently no reco-io type that says "I will feed you frames
one at a time, please stitch them" without a backing source file.
The OBS plugin will either need to (a) write a mock `FrameSource`
impl that blocks the OBS callback thread waiting for the next
submission, or (b) call `StitchPipeline::render_to_target` directly
and bypass the higher-level session machinery.

**Suggested direction**: a `LiveStitchSession` in reco-core that
exposes `submit_frame(left, right) -> Result<PixelBuffer>` without
expecting a FrameSource. Could share most of the existing session
logic, just with source pulled out.

## Resolved (archived)

- **R1. No RGBA readback helper on StitchPipeline**
  Resolved by Batch A (#223): `StitchRenderer::render_and_readback_rgba()`
  + `flush_rgba()` with triple-buffered staging, same pattern as
  `Nv12Converter`. Earlier ~40 lines of boilerplate in
  `reco-obs/source.rs` can be replaced.

## Notes on plugin status

reco-obs is still scaffolding PoC - it renders a green test pattern
to verify the OBS plugin hook-up works. Real frame ingestion from
OBS callbacks has not been implemented yet. When that happens, A1
and A4 will become blockers and likely motivate a Batch H in
reco-io.
