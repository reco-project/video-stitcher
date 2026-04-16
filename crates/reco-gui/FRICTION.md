# GUI Consumer API Friction

Friction points encountered while building a Slint GUI consumer of reco-core.

## 1. wgpu Version Mismatch (blocking zero-copy rendering)

**Impact**: Forces CPU readback bridge instead of zero-copy GPU texture sharing.

reco-core uses wgpu 29 (custom fork at `mohamedtahaguelzim/wgpu.git` branch
`fix/shader-draw-parameters-v29.0.1`). Slint 1.15 only supports wgpu 27 and 28
via `unstable-wgpu-{27,28}` features. Since wgpu types from different major
versions are distinct Rust types, the GPU device/queue cannot be shared between
Slint and reco-core.

**Current workaround**: Headless `GpuContext` in reco-core, render to internal
RGBA target, readback via staging buffer, convert to `slint::Image::from_rgba8()`.
This adds ~2-5ms of GPU-to-CPU latency per frame at 1080p.

**Resolution path**: Slint wgpu 29 support is tracked in slint-ui/slint#11378.
Once merged, switch to `GpuContext::from_device_queue()` with the device/queue
from Slint's `set_rendering_notifier` callback, eliminating all readback overhead.

## 2. No RGBA Readback API on StitchRenderer

**Impact**: GUI consumers must manually implement wgpu buffer copy + map.

`StitchRenderer` provides `render_and_readback_nv12()` for encoding, but no
RGBA readback equivalent. GUI consumers need RGBA (or BGRA) pixel data to
display in framework widgets.

The workaround is ~120 lines of double-buffered readback boilerplate in
`preview.rs`: two staging buffers with `MAP_READ | COPY_DST`,
`copy_texture_to_buffer` from `render_target()`, non-blocking poll on the
previous frame's buffer while submitting the next, row-padding strip, and
direct write into `SharedPixelBuffer` to avoid an intermediate allocation.

A single-buffered blocking approach is too slow for real-time playback (adds
3-8ms of stall per frame). The double-buffered approach pipelines GPU work
with CPU readback, trading one frame of latency for smooth playback.

**Suggested API addition**:
```rust
impl StitchRenderer {
    /// Render and read back RGBA pixels for display in GUI frameworks.
    /// Double-buffered: returns the previous frame's data (one frame behind).
    pub fn render_and_readback_rgba(
        &mut self,
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
        yaw: f32,
        pitch: f32,
    ) -> Result<Option<&[u8]>, PipelineError> { ... }
}
```

This would mirror `render_and_readback_nv12()` but output tightly-packed RGBA
instead of NV12. Could reuse the same triple-buffer staging pattern that
`Nv12Converter` already implements.

## 3. GpuContext::new() is Async

**Impact**: Minor friction in synchronous GUI init code.

`GpuContext::new()` and `GpuContext::for_surface()` are async. GUI frameworks
typically initialize synchronously (or at least within a synchronous setup
callback). Every call site wraps with `pollster::block_on()`.

This is a deliberate design choice (wgpu's adapter/device creation is async),
not a bug. The friction is minor since pollster is already a workspace dep.

## 4. StitchRenderer Hardcodes InputFormat::Yuv420p

**Impact**: Low - file decode is Yuv420p anyway.

`StitchRenderer::new()` hardcodes `InputFormat::Yuv420p` (line 81 of
`stitch_renderer.rs`). A GUI consumer can't construct a renderer for NV12 input
via this API - they'd need to use `StitchPipeline::with_gpu()` directly with a
custom `InputFormat`. This matters for live camera preview (Jetson NV12 output).

**Suggested**: Accept `InputFormat` as a parameter, or auto-detect from the
source format.

## 5. No Resize Without Recreating ReadbackBuffer

**Impact**: Low - preview resize is infrequent.

When the preview viewport resizes, `StitchPipeline::resize()` recreates the
internal render target at the new dimensions. But any external staging buffers
(like our `ReadbackBuffer`) also need recreation. The renderer doesn't notify
consumers that its internal texture dimensions changed.

A future `on_resize` callback or returning the new dimensions from `resize()`
would help.

## 6. render_to_target() Returns CommandBuffer, Caller Must Submit

**Impact**: Medium - complicates double-buffered readback.

`StitchPipeline::render_to_target()` returns a `wgpu::CommandBuffer` that the
caller must submit to the queue. For the GUI readback bridge, we need to ALSO
encode a copy-to-staging-buffer command and submit both together. This means
the caller creates a separate `CommandEncoder`, encodes the copy, and submits
`[render_cmd, copy_encoder.finish()]` as a batch.

If `render_to_target` submitted internally (like `render_to_view` does), the
caller would lose the ability to batch the copy. So the current API is correct
for this use case, but it would be cleaner if there were a `render_to_buffer()`
that renders + copies to a caller-provided staging buffer in one call.

## 7. FrameSource::try_next_frame() EOF Ambiguity

**Impact**: Low - workaround is a timeout heuristic.

`FfmpegFileSource::try_next_frame()` returns `Ok(None)` for both "no frame
decoded yet" (decode thread busy) and "end of stream" (channel disconnected).
A GUI consumer polling non-blocking can't distinguish "wait for the next frame"
from "playback is finished" without a timeout heuristic.

**Current workaround**: If `Ok(None)` persists for 30x the frame duration
(~1 second), assume EOF.

**Suggested**: Return a distinct result like `Ok(FrameResult::NotReady)` vs
`Ok(FrameResult::EndOfStream)`, or add a `fn is_exhausted(&self) -> bool`
method to the `FrameSource` trait.

## 8. render_to_view() vs render_to_target() Asymmetry

**Impact**: Low - but confusing for GUI consumers.

`StitchRenderer` has `render_yuv()` which renders directly to a
`wgpu::TextureView` (surface) - this is the fast path the CLI uses.
For GUI readback, you need `pipeline().render_to_target()` instead, which
renders to the internal RGBA texture.

The surface path (render_yuv) submits GPU commands internally. The target path
(render_to_target) returns a CommandBuffer. This asymmetry means the GUI
consumer must understand the pipeline internals to choose the right method.

A unified API like `render(target: RenderTarget)` where `RenderTarget` is
either a surface view or an internal texture would reduce confusion.
