# Phase 3 Cross-Platform Benchmark Report

**Date:** 2026-03-30
**Commit:** 64b03e1 (refactor: async encode thread, session owns encoder lifecycle)
**Input:** GoPro 10 stereo pairs, H.264 1920x1080@29.97fps (video input, not camera)
**Calibration:** match.json (v1 format)

## Methodology

- All tests use `cargo run --release -p reco-cli -- stitch LEFT RIGHT -c MATCH -o OUT --max-frames N`
- CPU upload forced via `RECO_NO_HWACCEL=1`
- Specific encoder forced via `--encoder NAME`
- Profiled runs use `--features profiling`, produce `reco-trace.json` (Chrome trace format, viewable in Perfetto)
- Frame counts verified with `ffprobe -count_frames`
- Target ~10s processing per test, frame count adjusted per device speed

## Results

### Desktop (NVIDIA GeForce RTX 5070, Linux, Vulkan)

| # | Input | Output | Mode | Encoder | FPS | Frames | Verified |
|---|-------|--------|------|---------|-----|--------|----------|
| D1 | 1080p H.264 | 1920x1080 h264 | zero-copy | h264_nvenc | 445 | 900 OK | |
| D2 | 1080p H.264 | 1920x1080 h264 | CPU upload | h264_nvenc | 148 | 900 OK | |
| D3 | 1080p H.264 | 1920x1080 h264 | CPU upload | libx264 | 112 | 900 OK | |
| D4 | 1080p H.264 | 1920x1080 hevc | zero-copy | hevc_nvenc | 427 | 900 OK | |
| D5 | 1080p H.264 | 1920x1080 av1 | zero-copy | av1_nvenc | 459 | 900 OK | |
| D6 | 4K HEVC | 1920x1080 h264 | zero-copy | h264_nvenc | 128 | 300 OK | |
| D7 | 4K HEVC | 3840x2160 h264 | zero-copy | h264_nvenc | 107 | 300 OK | |
| D8 | 1080p H.264 | 1280x720 h264 | zero-copy | h264_nvenc | 591 | 900 OK | |
| D9 | 1080p H.264 | 1920x1080 h264 | zero-copy | h264_nvenc | 298 | 300 OK | profiled |

Desktop profiling (D9, 300 frames, zero-copy):

| Span | Avg | Max |
|------|-----|-----|
| encode_nv12_frame | 1.50ms | 3.0ms |
| session_submit_render | 1.43ms | 3.0ms |
| wait_decode | 1.21ms | 333ms |
| nv12_convert_readback | 0.30ms | 0.9ms |
| nv12_readback | 0.24ms | 0.9ms |
| render_to_target_gpu | 0.06ms | 0.4ms |
| nv12_compute | 0.004ms | 0.03ms |

Bottleneck: encode (1.50ms) and decode thread recv (1.21ms) are close. Render + readback are fast.

### Mac Mini (Apple M4, macOS, Metal)

| # | Input | Output | Mode | Encoder | FPS | Frames | Verified |
|---|-------|--------|------|---------|-----|--------|----------|
| M1 | 1080p H.264 | 1920x1080 h264 | zero-copy | h264_videotoolbox | 211 | 900 OK | |
| M2 | 1080p H.264 | 1920x1080 h264 | CPU upload | h264_videotoolbox | 177 | 900 OK | |
| M3 | 1080p H.264 | 1920x1080 h264 | zero-copy | libx264 | 203 | 900 OK | |
| M4 | 1080p H.264 | 1920x1080 hevc | zero-copy | libx265 | 81 | 900 OK | hevc_vt failed (profile bug) |
| M5 | 1080p H.264 | 1920x1080 hevc | zero-copy | libx265 | 81 | 900 OK | |
| M6 | 1080p H.264 | 1280x720 h264 | zero-copy | h264_videotoolbox | 344 | 900 OK | |
| M7 | 1080p H.264 | 1920x1080 h264 | zero-copy | h264_videotoolbox | 199 | 300 OK | profiled |

**Bug found:** `hevc_videotoolbox` fails with "Error setting option profile to value high." Our encoder config sets `profile=high` which VT HEVC doesn't support (needs `main` or `main10`).

Mac profiling (M7, 300 frames, zero-copy):

| Span | Avg | Max |
|------|-----|-----|
| encode_nv12_frame | 4.45ms | 11.1ms |
| session_submit_render | 3.87ms | 9.3ms |
| send_packet | 3.74ms | 63.3ms |
| nv12_convert_readback | 0.66ms | 4.9ms |
| nv12_readback | 0.49ms | 4.8ms |
| render_to_target_gpu | 0.27ms | 1.2ms |
| nv12_compute | 0.003ms | 0.03ms |

Bottleneck: VT encode at 4.45ms/frame. The async channel (capacity 2) creates backpressure (session_submit_render 3.87ms includes ~3.2ms blocked on channel send).

### Jetson Orin Nano Super (NVIDIA Tegra, Linux, Vulkan)

| # | Input | Output | Mode | Encoder | FPS | Frames | Verified |
|---|-------|--------|------|---------|-----|--------|----------|
| J1 | 1080p H.264 | 1920x1080 h264 | CPU upload | libx264 | 32.9 | 300 OK | |
| J2 | 1080p H.264 | 1920x1080 hevc | CPU upload | libx265 | 3.2 | 300 OK | 10x slower than h264 |
| J3 | 1080p H.264 | 1280x720 h264 | CPU upload | libx264 | 37.0 | 300 OK | |
| J4 | 1080p H.264 | 1920x1080 h264 | CPU upload | libx264 | 32.0 | 300 OK | profiled |

Jetson profiling (J4, 300 frames, CPU upload):

| Span | Avg | Max |
|------|-----|-----|
| encode_nv12_frame | 30.5ms | 65.3ms |
| session_submit_render | 26.1ms | 63.3ms |
| wait_decode | 2.66ms | 85.3ms |
| render_to_target | 2.03ms | 5.0ms |
| nv12_convert_readback | 1.44ms | 4.2ms |
| gpu_upload | 0.81ms | 3.1ms |
| nv12_readback | 0.81ms | 3.5ms |

Bottleneck: libx264 SW encode at 30.5ms/frame on 6-core ARM. Async encode overlaps with decode (27.8ms) so net throughput is ~30ms/frame. libx265 is 10x slower (unusable on ARM).

### Surface Laptop (AMD Radeon iGPU, Windows 11, DX12)

| # | Input | Output | Mode | Encoder | FPS | Frames | Verified |
|---|-------|--------|------|---------|-----|--------|----------|
| S1 | 1080p H.264 | 1920x1080 h264 | CPU upload | h264_amf | 74.5 | 900 OK | |
| S2 | 1080p H.264 | 1920x1080 h264 | CPU upload | libx264 | 73.4 | 900 OK | |
| S3 | 1080p H.264 | 1920x1080 hevc | CPU upload | hevc_amf | 78.0 | 900 OK | |
| S4 | 1080p H.264 | 1920x1080 av1 | CPU upload | libsvtav1 | 48.5 | 900 OK | av1_amf probe failed (no AV1 HW on Renoir) |
| S5 | 1080p H.264 | 1280x720 h264 | CPU upload | h264_amf | 86.4 | 900 OK | |

av1_amf failed: `CreateComponent(AMFVideoEncoderHW_AV1) failed with error 10` (Renoir iGPU lacks AV1 encode HW). Fell back to libsvtav1, output verified correct. No profiled run (no profiling feature build on Windows).

### Raspberry Pi 5 (V3D 7.1.7, Debian, Vulkan/GL)

| # | Input | Output | Mode | Encoder | FPS | Frames | Verified |
|---|-------|--------|------|---------|-----|--------|----------|
| R1 | 1080p H.264 | 1920x1080 h264 | Vulkan | libx264 | 34.0 | 300 OK | check for black output |
| R2 | 1080p H.264 | 1920x1080 h264 | GL | libx264 | 23.6 | 300 OK | correct rendering |
| R3 | 1080p H.264 | 1920x1080 hevc | GL | libx265 | 10.4 | 300 OK | |
| R4 | 1080p H.264 | 1280x720 h264 | GL | libx264 | 35.8 | 300 OK | |
| R5 | 1080p H.264 | 1920x1080 h264 | GL | libx264 | 24.4 | 300 OK | profiled |

RPi5 profiling (R5, 300 frames, GL backend, CPU upload):

| Span | Avg | Max |
|------|-----|-----|
| nv12_convert_readback | 35.96ms | 50.0ms |
| nv12_submit | 32.93ms | 44.1ms |
| encode_nv12_frame | 32.29ms | 86.2ms |
| nv12_readback | 2.96ms | 11.5ms |
| render_to_target | 3.35ms | 10.2ms |
| gpu_upload | 1.90ms | 7.4ms |

Bottleneck: NV12 GPU submit (32.9ms!) dominates - the V3D GPU is very slow at the NV12 compute shader. Encode (32.3ms) overlaps on the async thread. The GPU is the primary bottleneck on RPi5, not CPU encode.

## Observations

1. **Zero-copy vs CPU upload**: 3x speedup on Desktop (445 vs 148). 1.2x on Mac (211 vs 177).
2. **720p is faster everywhere**: 33% faster on Desktop (591 vs 445), 63% on Mac (344 vs 211), 12% on Jetson (37 vs 33), 16% on Surface (86 vs 74), 51% on RPi5 GL (36 vs 24).
3. **HEVC HW encode**: Similar speed to H.264 on NVIDIA (427 vs 445) and AMD (78 vs 74). VT HEVC broken (profile bug).
4. **AV1**: Fast on NVIDIA HW (459 fps), slow on ARM SW (not tested). Surface fell back to libsvtav1 (48.5 fps).
5. **libx265 is very slow on ARM**: 3.2 fps on Jetson, 10.4 fps on RPi5. Not viable for real-time.
6. **RPi5 Vulkan vs GL**: Vulkan is 44% faster (34 vs 24 fps) but may produce black output (V3DV driver bug).
7. **RPi5 NV12 shader**: 32.9ms per frame on V3D GPU - this is the primary bottleneck, not CPU encode.

## Bugs Found

1. **`profile=high` breaks multiple HW encoders**: hevc_videotoolbox, h264_v4l2m2m, and hevc_v4l2m2m all fail when our encoder config sets `profile=high`. This causes fallback to software encoders on Jetson (both H.264 and HEVC) and breaks HEVC on Mac entirely. The fix: don't set profile for VT HEVC (use `main`), and handle the v4l2m2m profile format differently.
2. **RPi5 Vulkan produces black output**: R1 output has 30 kb/s bitrate (vs ~1739 kb/s for GL runs) confirming all-black frames. Known V3DV driver bug.

## Visual Verification

All outputs opened and visually confirmed by user.

**Passed (correct stitching, no artifacts):**
- D1-D9: all desktop outputs
- M1, M2, M4, M6: Mac Mini (h264_vt zero-copy, h264_vt CPU upload, libx265 hevc, 720p)
- J1, J2, J3: Jetson (libx264, libx265, 720p)
- R2, R3, R4: RPi5 GL (libx264, libx265 hevc, 720p)
- S1, S2, S3, S5: Surface (h264_amf, libx264, hevc_amf, 720p)

**Failed (black output):**
- R1: RPi5 Vulkan - all-black frames (V3DV driver bug, 30 kb/s bitrate vs ~1739 kb/s for GL)

**Note:** S4 (Surface AV1/libsvtav1) was initially reported as black but frames are correct when extracted with ffmpeg. The video player lacked AV1 decode support.
