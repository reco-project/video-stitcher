# From v1 to v2: A Technical Migration Paper for the Dual-Camera Sports Stitching System

**Author:** reco-project/video-stitcher
**Date:** February 2026
**Version:** 1.0

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [v1 Architecture Deep Dive](#2-v1-architecture-deep-dive)
3. [v1 Limitations](#3-v1-limitations)
4. [Gyroflow Architecture Analysis](#4-gyroflow-architecture-analysis)
5. [Mapping Gyroflow to v2](#5-mapping-gyroflow-to-v2)
6. [v2 Architecture Specification](#6-v2-architecture-specification)
7. [Key Engineering Challenges](#7-key-engineering-challenges)
8. [What to Reuse from Gyroflow vs Build Fresh](#8-what-to-reuse-from-gyroflow-vs-build-fresh)
9. [Recommended Implementation Order](#9-recommended-implementation-order)

---

## 1. Abstract

This system takes two wide-angle camera feeds — physically mounted at approximately 90 degrees to each other with a shared overlap region — and produces a real-time, interactive 180-degree panoramic view. The output behaves like a physical head-turn: the user pans left/right and up/down, and the view rotates through the stitched scene. The geometric model places the two cameras on the bisector of the x=z plane (y=0), with the left camera plane along the z-axis and the right camera plane along the x-axis. The cameras are orthogonal, with configurable overlap and five calibration parameters governing their relative geometry.

**v1** achieves the full pipeline using a Python/FastAPI backend for calibration and transcoding, and a React/Three.js/WebGL frontend for undistortion, projection, and interactive viewing. It supports GoPro-style fisheye lenses (using the OpenCV KB4 fisheye model), audio-based temporal synchronisation, GPU-accelerated encoding (via FFmpeg), and Gyroflow-compatible lens profiles. The output can be recorded as WebM from the browser canvas.

**v2** is being built because v1 has fundamental architectural limits that cannot be patched: the browser→Python frame roundtrip makes headless operation impossible, audio sync is fragile, the canvas recorder is limited to WebM at software speed, and the GPU pipeline is broken by CPU readback at each calibration iteration. v2 targets a full-GPU Rust pipeline (decode → undistort → dual-plane project → encode) with zero CPU frame roundtrip, optical-flow-based temporal sync, CLI-first headless execution, YOLO-based ball tracking with a Kalman Director, and direct reuse of Gyroflow's lens profile database.

---

## 2. v1 Architecture Deep Dive

### 2.1 Overall Structure

v1 is an Electron desktop application wrapping two processes:

- **Frontend**: React + Three.js (WebGL), served as a local static bundle
- **Backend**: Python FastAPI server on `127.0.0.1:8000`

The frontend handles all real-time rendering and interactive viewing. The backend handles all frame I/O, calibration computation, and video transcoding.

### 2.2 Data Model and Storage

**`backend/app/models/match.py`** (inferred from `main.py` and `processing.py`) defines the central `Match` object. A match stores:
- `left_videos`, `right_videos`: lists of video paths per camera
- `params`: calibration parameters (`cameraAxisOffset`, `intersect`, `xTy`, `xRz`, `zRx`)
- `src`: relative path to the stacked video (`videos/{match_id}.mp4`)
- `processing`: status state machine (`pending → transcoding → calibrating → ready`)
- `transcode`: FFmpeg encoding metrics
- `quality_settings`: encoder selection, bitrate, resolution
- `metadata`: freeform dict including `colorCorrection` and `panningRanges`

Matches are persisted as JSON files in `MATCHES_DIR`, with a file-based key-value store (`backend/app/repositories/file_match_store.py`). Lens profiles are stored as JSON in either a SQLite database (bundled official profiles, read-only) or user JSON files, managed by `backend/app/repositories/hybrid_lens_profile_store.py`.

### 2.3 Lens Profile Format

**`backend/app/models/lens_profile.py`** defines the v1 lens profile schema:
```python
{
  "id": "gopro-hero10black-linear-3840x2160",
  "camera_brand": "GoPro",
  "camera_model": "HERO10 Black",
  "resolution": {"width": 3840, "height": 2160},
  "distortion_model": "fisheye_kb4",         # Only model supported in v1
  "camera_matrix": {"fx": 1796.32, "fy": 1797.22, "cx": 1919.37, "cy": 1063.17},
  "distortion_coeffs": [0.03421, 0.06767, -0.07408, 0.02994]  # k1,k2,k3,k4
}
```

This is a simplified subset of Gyroflow's `LensProfile` struct (`gyroflow/src/core/lens_profile.rs`). The `fisheye_kb4` label corresponds directly to Gyroflow's `opencv_fisheye` distortion model (Kannala-Brandt with 4 radial coefficients).

### 2.4 Pipeline Step 1: Temporal Synchronisation

**File:** `backend/app/services/transcoding.py` — `_compute_offset()` (line 401)

The synchronisation extracts audio from both cameras at 16 kHz mono WAV using FFmpeg (`_extract_audio()`, line 376), then computes the cross-correlation using `scipy.signal.correlate()`. The lag at the argmax of the cross-correlation gives the offset in samples, divided by sample rate to get seconds.

```python
corr = correlate(a1, a2, mode="full")
lag = np.argmax(corr) - len(a2)
return lag / sr1
```

This is the classic full-signal cross-correlation approach. It is sensitive to noise, does not handle variable clock drift, and can fail entirely if one camera has very different audio (e.g., different scene position). The offset is then passed to FFmpeg via `-ss` seek flags to trim the leading frames of the earlier-starting camera.

### 2.5 Pipeline Step 2: Video Stacking

**File:** `backend/app/services/transcoding.py` — `_stack_videos()` (line 642)

After offset computation, FFmpeg is invoked with the `vstack=inputs=2` filter to produce a vertically-stacked video where the top half is the left camera and the bottom half is the right camera. The encoder chain is:
1. GPU decoder: `-hwaccel cuda` (NVENC), `-hwaccel qsv` (Intel), or `-hwaccel auto` (AMD)
2. Scale filter (if needed): `scale=1920:1080`
3. vstack: `[0:v][1:v]vstack=inputs=2:shortest=1[vout]`
4. GPU encoder: `h264_nvenc`, `h264_qsv`, `h264_amf`, fallback to `libx264`

The FFmpeg process is spawned with `subprocess.Popen()`, and stdout progress is parsed in real-time for the UI. This is a CPU-side orchestration of FFmpeg — all decoded frames pass through the CPU FFmpeg filter graph; there is no zero-copy GPU path here.

**Critical limitation:** the vstack operation forces both streams through an FFmpeg software filter (`vstack`), even when hardware decoders are used. Hardware-decoded frames are transferred back to CPU memory for the vstack filter, then uploaded again for encoding. This is a fundamental architectural choice of the FFmpeg filter graph, not a bug.

### 2.6 Pipeline Step 3: Frame Extraction and Lens Undistortion

**File:** `frontend/src/features/viewer/shaders/fisheye.js`

The browser frontend holds the stacked video as a WebGL `sampler2D` texture. For each frame, a Three.js `ShaderMaterial` processes each pixel by reversing the fisheye distortion. The fragment shader (`fisheyeShader()`) implements the KB4 forward distortion model:

```glsl
// Fragment shader (fisheye.js, line 258)
float x = (vUv.x - cx) / fx;
float y = (vUv.y - cy) / fy;
float r = sqrt(x*x + y*y);
float theta = atan(r);
float theta_d = theta * (1.0 + d.x*pow(theta,2.0) + d.y*pow(theta,4.0)
                              + d.z*pow(theta,6.0) + d.w*pow(theta,8.0));
float scale = r > 0.0 ? theta_d / r : 1.0;
x *= scale; y *= scale;
distortedUV = vec2(fx * x + cx, fy * y + cy);
```

This is the **forward** KB4 model (undistorted → distorted UV), used as a texture lookup from the original fisheye source. The stacked video occupies UV [0.0, 1.0] vertically; the left (top) half is read from UV.y ∈ [0.5, 1.0] and the right (bottom) half from UV.y ∈ [0.0, 0.5].

**Colour correction** is applied in the same shader pass via Reinhard LAB transfer: `lab = lab * labScale + labOffset`, where the scale and offset are computed server-side by `compute_color_correction()` in `backend/app/services/feature_matching.py` (line 275).

### 2.7 Pipeline Step 4: Frame Upload and Calibration

**File:** `backend/app/routers/processing.py` — `process_match_with_frames()` (line 485)

To trigger calibration, the browser:
1. Seeks the video player to a representative frame
2. Renders that frame through the fisheye shader to an offscreen canvas
3. Captures the canvas as a PNG via `canvas.toDataURL()` or `toBlob()`
4. Uploads the left and right half-frames separately to `POST /api/matches/{id}/process-with-frames`

The server receives two PNG images (typically 1920×1080 each), decodes them with `cv2.imdecode()`, and feeds them to the feature matcher.

**Feature matching** (`backend/app/services/feature_matching.py` — `match_features()`, line 13):
- Resizes to 1920px width
- SIFT with 2000 features (or ORB fallback)
- BFMatcher + Lowe's ratio test (0.7 threshold)
- Spatial filtering: only overlap region (inner 40% horizontally, inner 60% vertically)
- RANSAC via `cv2.findFundamentalMat()` for outlier rejection
- Normalised to plane coordinates (origin at image centre, width=1.0)

### 2.8 Pipeline Step 5: Powell Optimisation

**File:** `backend/app/services/position_optimization.py` — `_minimize_sum_of_angles()` (line 132)

The optimiser finds 5 parameters that minimise the sum of angular differences between corresponding ray pairs, where each matched point becomes a 3D ray from the camera origin. The model:

- **Left plane** lies along the z-axis: each 2D point `(x, y)` maps to 3D `(x, -y, 0)`
- **Right plane** lies along the x-axis: each 2D point `(z, y)` maps to 3D `(0, -y, -z)`
- **Camera** sits at `(cam_d, 0, cam_d)` on the x=z bisector

The 5 optimised parameters are:
| Parameter | Meaning | Bounds |
|-----------|---------|--------|
| `cam_d` | Camera distance from origin | [0.1, 0.35] |
| `intersect` | Plane overlap ratio | [0.0, 1.0] |
| `xTy` | Left plane Y translation | [-1.0, 1.0] |
| `xRz` | Left plane Z rotation | [-π, π] |
| `zRx` | Right plane X rotation | [-π, π] |

SciPy's `minimize(..., method='Powell')` is used with bounds and `maxiter=1000`. **Missing parameter:** there is no left-plane Z rotation around the plane centre (horizon levelling), which is needed when the left camera is not perfectly level.

### 2.9 Pipeline Step 6: Rendering and Projection

**File:** `frontend/src/features/viewer/components/Viewer.jsx`

Two Three.js `PlaneGeometry` meshes render the scene. Their positions and rotations are set from the calibration `params`:

```jsx
// Viewer.jsx, line 72-75
const position = isLeft
  ? [0, 0, (planeWidth/2) * (1 - params.intersect)]
  : [(planeWidth/2) * (1 - params.intersect), params.xTy, 0];
const rotation = isLeft
  ? [params.zRx, THREE.MathUtils.degToRad(90), 0]
  : [0, 0, params.xRz];
```

The camera orbits the origin via mouse/touch controls (yaw ±140°, pitch ±20°), producing the head-turn illusion. A `PerspectiveCamera` with a narrow FOV simulates the equivalent of zooming in on the panorama.

### 2.10 Pipeline Step 7: Canvas Recording

**File:** `frontend/src/features/viewer/hooks/useCanvasRecorder.js`

Export uses `HTMLCanvasElement.captureStream(fps)` and `MediaRecorder`. This produces WebM (VP9 or VP8), is CPU-bound (software encode from GPU-rendered frames), and reintroduces a redundant encode step. Output quality and format are both limited by what the browser's `MediaRecorder` supports.

---

## 3. v1 Limitations

### 3.1 Browser → Python Frame Roundtrip (Critical)

**Location:** `backend/app/routers/processing.py:485` (server side), `frontend/src/features/viewer/components/RecalibratePanel.jsx` (client side)

Every calibration step requires:
1. Render to offscreen canvas (GPU)
2. `canvas.toBlob()` → PNG encode (CPU, ~50–200ms for 1920×1080)
3. HTTP multipart POST to localhost (serialisation + kernel copy)
4. `cv2.imdecode()` on the server (CPU)
5. SIFT feature detection (CPU, ~200–500ms)
6. Powell optimisation (CPU, ~100–500ms per iteration)
7. JSON response → frontend param update

This roundtrip is **unavoidable** in the current architecture. It makes headless CLI operation impossible: the browser must be running and visible for frame extraction. It also adds ~500ms–2s per calibration attempt, preventing real-time re-calibration.

### 3.2 Audio Synchronisation Fragility

**Location:** `backend/app/services/transcoding.py:401`

Cross-correlation on full-length audio signals has O(N²) complexity for exact correlation (SciPy uses FFT-based correlation, so O(N log N), but still operates on the entire file). More critically:

- Silent or near-silent segments produce spurious peaks
- Different camera placements mean different ambient sound → lower correlation SNR
- Clock drift is **ignored**: both cameras are assumed to start together but may drift apart over time at different rates (1–2 ppm is common for consumer cameras)
- No subframe accuracy: the cross-correlation has sample-level precision (1/16000s = 62.5μs), but the actual video is aligned at frame boundaries (1/60s ≈ 16ms), introducing up to half-frame jitter

### 3.3 FFmpeg vstack CPU Bottleneck

**Location:** `backend/app/services/transcoding.py:709`

The FFmpeg `vstack` filter is a software filter. Even with GPU decode (`-hwaccel cuda`), the filter graph forces frames to system memory. On a 4K input pipeline:
- GPU decode → CPU transfer (memory bandwidth limited)
- vstack in CPU memory
- CPU → GPU transfer for NVENC
- NVENC encode

This is the documented behaviour of FFmpeg's hardware acceleration: hardware-decoded frames are in opaque GPU memory; software filters require transfer to system memory. A true zero-copy path would require CUDA filters (`-hwaccel_output_format cuda` + `scale_cuda` / custom CUDA filter), which FFmpeg supports but which v1 does not use.

### 3.4 Canvas Recorder Limitation

**Location:** `frontend/src/features/viewer/hooks/useCanvasRecorder.js`

- **Format**: WebM only (VP9/VP8) — no H.264, no H.265
- **Codec**: browser's software encoder — typically half or less the speed of hardware encoding
- **Quality**: `MediaRecorder` bitrate control is browser-implementation-dependent
- **Redundancy**: the video was already encoded to H.264 by FFmpeg; the canvas recorder re-decodes it (via the `<video>` element), renders it through the WebGL pipeline, and re-encodes it — a three-generation encode chain

### 3.5 Missing Horizon Levelling Parameter

**Location:** `backend/app/services/position_optimization.py:140-146`

The optimiser has no parameter for left-plane rotation around its own z-axis (i.e., roll of the left camera). If the left camera is not perfectly horizontal, all stitched content will appear tilted. The right plane has `zRx` (x-axis rotation) for tilt correction, but the left plane has no equivalent roll parameter. This must be added as a 6th optimisation parameter in v2.

### 3.6 No Drift Correction

The audio sync computes a single scalar offset and applies it uniformly. Over a 90-minute match, a 1ppm clock difference between two cameras produces a 90-millisecond drift — roughly 5–6 frames at 60fps. This produces progressively worsening temporal misalignment that audio cross-correlation cannot detect after the fact.

### 3.7 Calibration Sensitivity to Matching Quality

**Location:** `backend/app/services/feature_matching.py:125-134`

SIFT matching on undistorted frames works well when the overlap region has good texture (e.g., grass, crowd), but fails on:
- Uniform sky regions in the overlap
- Motion blur in the overlap (athletes crossing during calibration frame capture)
- Very small overlap angles

The fallback (`calibration_failed = True`) uses hardcoded defaults rather than propagating the confidence score to the optimiser. There is no retry logic or multi-frame averaging.

---

## 4. Gyroflow Architecture Analysis

Gyroflow is a GPU-accelerated video stabilisation tool that solves several sub-problems directly analogous to v2's needs: lens undistortion on GPU, hardware frame I/O without CPU roundtrip, FFmpeg hardware decode/encode integration, optical-flow-based temporal sync, and a rich lens calibration system. The codebase is split into `gyroflow_core` (a pure Rust library) and a Qt/QML UI shell.

### 4.1 The wgpu GPU Pipeline

**File:** `gyroflow/src/core/gpu/wgpu.rs`

`WgpuWrapper` is the central GPU processing struct. It holds:
```rust
pub struct WgpuWrapper {
    staging_buffer: Option<wgpu::Buffer>,  // CPU readback buffer (CPU path only)
    buf_matrices:   Option<wgpu::Buffer>,  // Per-frame rotation matrices
    buf_params:     Option<wgpu::Buffer>,  // KernelParams uniform
    buf_mesh_data:  Option<wgpu::Buffer>,  // Lens warp mesh
    buf_drawing:    Option<wgpu::Buffer>,  // Overlay drawing
    in_texture:  TextureHolder,            // Input (wraps native handle)
    out_texture: TextureHolder,            // Output (wraps native handle)
    pipeline: PipelineType,               // Render or Compute pipeline
    bind_group: Option<wgpu::BindGroup>,
    device: wgpu::Device,
    queue:  wgpu::Queue,
    ...
}
```

The pipeline is constructed in `WgpuWrapper::new()` (line 146). Key design decision: Gyroflow **selects between render and compute pipelines** depending on whether the frame source is a GPU texture (render pipeline) or a GPU buffer (compute pipeline). For hardware-decoded frames (Metal texture, Vulkan image, CUDA buffer, D3D11 texture), the render pipeline is used. For CPU frames, the compute pipeline is used.

The WGSL shader is loaded as a string at runtime (`include_str!("wgpu_undistort.wgsl")`), and the lens distortion model functions are injected via text substitution before shader compilation:

```rust
// wgpu.rs, line 261-265
let mut kernel = include_str!("wgpu_undistort.wgsl").to_string();
let mut lens_model_functions = distortion_model.wgsl_functions().to_string();
kernel = kernel.replace("LENS_MODEL_FUNCTIONS;", &lens_model_functions);
kernel = kernel.replace("SCALAR", wgpu_format.1);
```

This makes the shader fully runtime-configurable for any lens model without recompilation.

The `undistort_image()` method (line 450) executes one GPU frame:
1. `handle_input_texture()` — copy/import the native frame into wgpu's input texture
2. Upload matrices and params to GPU buffers
3. Execute render pass (`rpass.draw(0..6, 0..1)` — two triangles covering NDC) or compute pass (8×8 workgroups)
4. `handle_output_texture()` — copy/export the wgpu output to the native output handle
5. For CPU output: map staging buffer and copy rows

### 4.2 The Zero-Copy Frame Path (wgpu_interop)

**File:** `gyroflow/src/core/gpu/wgpu_interop.rs`

This is the most architecturally significant file for v2. The `BufferSource` enum in `gpu/mod.rs` (line 31) defines all possible frame origins:

```rust
pub enum BufferSource<'a> {
    None,
    Cpu { buffer: &'a mut [u8] },
    DirectX11 { texture: *mut c_void, device: *mut c_void, device_context: *mut c_void },
    OpenGL { texture: u32, context: *mut c_void },
    Vulkan { texture: u64, device: u64, physical_device: u64, instance: u64 },
    Metal { texture: *mut MTLTexture, command_queue: *mut MTLCommandQueue },
    MetalBuffer { buffer: *mut MTLBuffer, command_queue: *mut MTLCommandQueue },
    CUDABuffer { buffer: *mut c_void },
}
```

`init_texture()` (line 35) constructs a `TextureHolder` from any of these sources. For GPU sources, it uses platform-specific interop:
- **Metal** (macOS): wraps the `MTLTexture` pointer directly into a wgpu texture using `create_texture_from_metal()` — zero copy if stride is aligned
- **Vulkan** (Linux/Windows): wraps the `VkImage` handle via `create_texture_from_vk_image()` — zero copy
- **CUDA** (Windows/Linux): shares memory via Vulkan external memory (`create_vk_image_backed_by_cuda_memory`) or D3D12 shared resource — one device-side copy on CUDA for pitch realignment
- **D3D11** (Windows): cross-API via D3D11/D3D12 shared resource, then into Vulkan

`handle_input_texture()` (line 267) handles the synchronisation: for Metal/Vulkan, it either copies texture-to-texture or does nothing (direct import). For CUDA, it does a device-side 2D copy via `cuda_2d_copy_on_device()` to handle pitch differences.

`handle_output_texture_post()` (line 451) synchronises the output: for CUDA, it does another device-side copy back from the Vulkan image memory into the consumer buffer; for Metal/Vulkan, it just waits for the device.

**The key insight for v2:** when the decoder and encoder share a device (e.g., both on NVIDIA via CUDA), Gyroflow can process a frame entirely on-device. The FFmpeg decoder outputs a `CUdeviceptr`; the GPU shader reads it as a texture; the output writes back to a `CUdeviceptr`; NVENC reads it directly. No bytes cross the PCIe bus.

### 4.3 The WGSL Undistortion Shader

**File:** `gyroflow/src/core/gpu/wgpu_undistort.wgsl`

The `KernelParams` struct (line 9) is a 256-byte uniform that carries:
- `f: vec2<f32>` — focal length in pixels
- `c: vec2<f32>` — principal point
- `k1, k2, k3: vec4<f32>` — distortion coefficients (12 values)
- `fov, r_limit` — FOV scale and radial limit
- `matrix_count` — number of rotation matrices (1 = global shutter, >1 = rolling shutter)
- `translation2d, translation3d` — 2D/3D output offset

The `undistort_coord()` function (line 461) maps each output pixel to its source coordinate via `rotate_and_distort()` (line 370), which applies a 3×3 rotation matrix (from the gyro data) followed by the lens distortion model. For v2, the "rotation matrix" would instead be the dual-plane projection transform.

The `distort_point()` function is injected from the distortion model's WGSL — for `opencv_fisheye`, this is the forward KB4 model `theta_d = theta * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)`. The `undistort_point()` function (the iterative inversion) is used on the CPU side.

The shader supports both render pipeline (vertex/fragment) and compute pipeline (compute shader with 8×8 workgroup). This dual-path design is critical for cross-platform support: platforms where textures cannot be directly imported (e.g., some GPU buffer formats) use the compute path operating on raw `array<SCALAR>` buffers.

**EWA sampling** (line 147–357): Gyroflow implements Elliptical Weighted Average (EWA) CubicBC resampling — a high-quality anisotropic filter that avoids aliasing when the undistortion map causes local magnification or minification. This computes the Jacobian of the warp map numerically (two extra `undistort_coord()` calls at epsilon offsets) and uses it to determine an elliptical filter kernel. For v2's seam region, this is particularly valuable to avoid moiré patterns at the stitch boundary.

### 4.4 The FFmpeg Hardware Integration

**File:** `gyroflow/src/rendering/ffmpeg_hw.rs` and `ffmpeg_processor.rs`

`HWDevice` (line 16) wraps an `AVHWDeviceType` and its `AVBufferRef`. The static `DEVICES` map (line 72) is a process-wide cache of hardware device contexts, preventing repeated initialisation overhead.

`init_device_for_decoding()` (line 120) attaches the hardware device context to the decoder context via `avcodec_get_hw_config()` and `av_hwdevice_ctx_create()`. After decode, each frame carries an opaque `AVHWFramesContext` with the GPU surface.

`FfmpegProcessor` (line 29 in `ffmpeg_processor.rs`) is the main processing struct. Its design:
- `VideoTranscoder` holds the input/output streams
- The frame callback receives `AVFrame*` pointers, which may be in GPU memory
- For hardware frames: `av_hwframe_transfer_data()` downloads to CPU, OR the GPU pointer is extracted and passed directly to `WgpuWrapper` as a `CUDABuffer`/`Metal`/`Vulkan` source

The zero-copy path is: `FFmpeg decoder → AVFrame (GPU surface) → extract native handle → WgpuWrapper (via wgpu_interop) → write back to GPU surface → FFmpeg encoder`. No CPU involvement for pixel data.

`supported_gpu_backends()` (line 88) iterates `av_hwdevice_iterate_types()` to enumerate available hardware acceleration. Gyroflow supports NVDEC/NVENC (CUDA), VideoToolbox (macOS), VAAPI (Linux), QSV (Intel), D3D11VA/DXVA2 (Windows).

### 4.5 The Lens Calibration System

**File:** `gyroflow/src/core/calibration/mod.rs`

`LensCalibrator` (line 36) performs the chessboard-based intrinsic calibration. Key design:
- Processes frames in parallel via Rayon (`crate::run_threaded`)
- `feed_frame()` (line 104): finds chessboard corners using OpenCV `findChessboardCornersSB()`, computes sharpness, stores in `all_matches`
- `calibrate()` (line 205): runs 1000 random Monte Carlo iterations, each picking 10 frames from different temporal segments. Uses parallel iteration over `(0..iterations).into_par_iter()` and reduces to the minimum-RMS solution
- Calls `opencv::calib3d::calibrate()` (Fisheye calibration with `Fisheye_CALIB_RECOMPUTE_EXTRINSIC | Fisheye_CALIB_FIX_SKEW`)

**File:** `gyroflow/src/core/lens_profile.rs`

`LensProfile` (line 25) is the canonical data structure. Critical fields:
- `fisheye_params: CameraParams` — `camera_matrix` (3×3 as three `[f64;3]` rows), `distortion_coeffs` (Vec<f64>, up to 12 values), `RMS_error`
- `distortion_model: Option<String>` — selects model (e.g., `"opencv_fisheye"`)
- `calib_dimension: Dimensions` — resolution at which calibration was performed
- `compatible_settings: Vec<serde_json::Value>` — other resolutions derived from same calibration
- `interpolations: Option<serde_json::Value>` — for zoom lenses with focal-length-dependent distortion

`get_camera_matrix()` (line 290) scales the matrix to the actual video resolution. `get_all_matching_profiles()` (line 321) expands `compatible_settings` into derived profiles. `get_interpolated_lens_at()` (line 495) linearly interpolates between focal-length-indexed profiles for zoom lenses.

`load_from_file()` → `load_from_data()` → `from_json()` is a simple JSON deserde via `serde_json`. The format is a flat JSON object with consistent field names. **Gyroflow's profile database (thousands of cameras) is directly reusable by v2 with zero modification**, because v2 uses the same Rust structs.

### 4.6 Synchronisation System

Gyroflow's sync is designed for aligning gyro data to video, not two video streams, but the mechanism is directly applicable. The `synchronization/` module computes optical flow between consecutive frames to estimate camera motion, then cross-correlates the optical flow magnitude signal against the gyro angular velocity signal.

For v2, the same optical-flow approach applied to *two concurrent video streams* gives temporally robust sync:
1. Compute per-frame optical flow magnitude signal for each camera independently
2. Cross-correlate the two motion signals
3. The lag at the peak correlation is the temporal offset

This is superior to audio cross-correlation because:
- It works even when both cameras are silent
- Motion events (ball kicks, fast panning) create sharp peaks in the flow signal, improving correlation SNR
- Frame-level precision (vs. sample-level for audio, which exceeds frame resolution anyway)

### 4.7 CLI Architecture

**File:** `gyroflow/src/cli.rs`

`Opts` (line 37) uses `argh::FromArgs` for argument parsing. The CLI mode (`run()`, line 132):
1. Parses input files by type (`detect_types()`)
2. Constructs a `RenderQueue` with a `StabilizationManager`
3. Calls `gpu::initialize_contexts()` to set up the GPU backend
4. Adds files to the queue and runs the Qt event loop (`QCoreApplication::exec()`)

The render queue pattern is significant: jobs are processed asynchronously with progress callbacks via Qt signals. For v2's CLI, a simpler `tokio`-based async queue or synchronous loop is preferable (avoiding Qt dependency).

`setup_defaults()` (line 552) shows the configurability: output codec, bitrate, GPU selection, sync parameters, and export mode are all runtime-configurable from CLI flags or JSON presets.

### 4.8 Controller Pattern

**File:** `gyroflow/src/controller.rs`

`Controller` is a QObject-derived struct that bridges the Qt/QML UI and `StabilizationManager`. The pattern is: `Controller` holds an `Arc<StabilizationManager>`, exposes methods as Qt slots, and emits Qt signals for UI updates. For v2's CLI-first design, the equivalent is:

- A `Pipeline` struct (or `StitchingManager`) that holds all state
- Methods callable from both the CLI main function and (in future) a UI layer
- Progress communicated via Rust channels (`tokio::sync::mpsc`) rather than Qt signals

---

## 5. Mapping Gyroflow to v2

### 5.1 Module Mapping Table

| Gyroflow Component | Gyroflow File | v2 Component | Notes |
|---|---|---|---|
| `WgpuWrapper` | `gpu/wgpu.rs` | `DualPlaneRenderer` | Extend to process two textures; add blending pass |
| `wgpu_interop.rs` | `gpu/wgpu_interop.rs` | `FrameImporter` | Reuse verbatim for platform-specific texture import |
| `wgpu_undistort.wgsl` | `gpu/wgpu_undistort.wgsl` | `dual_plane_project.wgsl` | Replace single-view undistortion with dual-plane projection; retain KB4 undistort, EWA sampling |
| `BufferSource` | `gpu/mod.rs` | `FrameSource` | Reuse verbatim; add support for two simultaneous sources |
| `FfmpegProcessor` | `rendering/ffmpeg_processor.rs` | `DualDecoder` + `Encoder` | Adapt to decode two streams simultaneously; share GPU device context |
| `HWDevice` | `rendering/ffmpeg_hw.rs` | `HWDevice` | Reuse verbatim |
| `LensProfile` + `LensProfileDatabase` | `core/lens_profile.rs` + `lens_profile_database.rs` | `LensProfile` | **Direct reuse, zero changes** |
| `LensCalibrator` | `core/calibration/mod.rs` | `LensCalibrator` | Reuse for calibrating cameras; already integrated with `LensProfile` |
| Optical flow sync logic | `core/synchronization/` | `OpticalFlowSync` | Adapt to two-stream cross-correlation |
| `ComputeParams` | `core/stabilization/compute_params.rs` | `StitchParams` | Replace rotation matrix concept with dual-plane 6-parameter geometry |
| `distortion_models/opencv_fisheye.rs` | `gpu/stabilize_spirv/src/` | Same | Reuse WGSL functions verbatim via `wgsl_functions()` injection |
| `KeyframeManager` | `core/keyframes.rs` | `DirectorKeyframes` | Adapt for pan/tilt/zoom output from AI Director |
| CLI (`cli.rs`) | `src/cli.rs` | `src/main.rs` (CLI) | Rewrite without Qt; use `clap` + `tokio` |
| `RenderQueue` | `rendering/render_queue.rs` | `ProcessingQueue` | Rewrite; simpler async queue with `tokio::sync::mpsc` |

### 5.2 Undistortion Shader → Dual-Plane Projection Shader

Gyroflow's `wgpu_undistort.wgsl` applies one undistortion per frame. v2 needs two undistortions (one per camera) plus a projection transform in a single pass. The mapping:

**Gyroflow:** `output_pixel → undistort_coord() → rotate_and_distort() → source_uv → sample_input_at()`

**v2:** `output_pixel → screen_to_ray() → intersect_with_plane() → undistort_uv() → {left_texture or right_texture}`

The v2 shader must:
1. Convert the output pixel from screen space to a world-space ray (given camera position and orientation)
2. Intersect the ray with the left or right plane (determined by which plane the ray hits first)
3. Map the intersection point back to UV coordinates on that plane
4. Apply KB4 undistortion to get the true fisheye UV
5. Sample the appropriate camera texture

This requires binding **two input textures** simultaneously — a layout change from Gyroflow's single input.

### 5.3 wgpu_interop → Dual-Texture Hardware Frame Input

Gyroflow's `init_texture()` creates one `TextureHolder`. v2 needs two: one for the left stream, one for the right stream. Both can be on the same `wgpu::Device` and `wgpu::Queue`. The bind group layout must expose `binding(5)` for the left texture and `binding(6)` for the right texture.

For the zero-copy path with two CUDA streams: NVDEC can decode multiple streams to separate `CUdeviceptr`s; both can be imported into wgpu via separate `wgpu_interop_cuda::create_vk_image_backed_by_cuda_memory()` calls on the same Vulkan device.

### 5.4 ffmpeg_hw → v2 Hardware Decode/Encode Pipeline

v2 runs two simultaneous decoders (one per camera) feeding one renderer and one encoder:

```
[Left video file]  → FFmpeg HWDecoder → CUDA frame A ──┐
                                                         ├→ DualPlaneRenderer → CUDA output → NVENC → output
[Right video file] → FFmpeg HWDecoder → CUDA frame B ──┘
```

Both decoders can share one `HWDevice` (same CUDA context) to avoid cross-device copies. `FfmpegProcessor` must be cloned or restructured to hold two `VideoDecoder` instances that are drained in lockstep with frame timing compensation applied from the optical-flow sync result.

### 5.5 synchronization/opencv.rs → v2 Optical Flow Temporal Sync

Gyroflow's optical flow sync computes a per-frame scalar (e.g., total flow magnitude or principal component of flow direction) and cross-correlates it against the gyro signal. v2 adapts this to compute:

1. For each camera: `flow_magnitude[t]` = mean magnitude of dense optical flow between consecutive frames, computed on the GPU (RAFT or Farneback via wgpu compute)
2. Cross-correlate `flow_L[t]` and `flow_R[t]` over a ±5 second search window
3. Argmax gives the integer-frame offset; subframe refinement via parabolic interpolation

Unlike audio correlation, this detects the same motion events in both cameras (e.g., fast panning during a kick). For drift correction, recompute the offset every N frames using a sliding window and fit a linear drift model.

### 5.6 Gyroflow calibration → v2 Lens Profile Loading (Direct Reuse)

`LensProfileDatabase::load_all()` from `gyroflow_core` can be called directly. The `LensProfile::load_from_file()` / `from_json()` path reads any Gyroflow `.json` profile. For v2:

```rust
let db = LensProfileDatabase::default();
db.load_all(); // loads all .json files from the standard search paths
let profile = db.get_by_id("gopro_hero10_black_4k_60fps").unwrap();
let k = profile.get_camera_matrix((width, height), false);
let d = profile.get_distortion_coeffs(); // [k1, k2, k3, k4, ...]
```

No translation layer is needed. v2 simply links `gyroflow_core` as a library dependency.

### 5.7 Controller.rs Pattern → v2 CLI Architecture

The `StabilizationManager` → `ComputeParams` → `WgpuWrapper` chain in Gyroflow maps to:

```
StitchingManager (Arc<RwLock<StitchState>>)
  ├── left_profile: LensProfile
  ├── right_profile: LensProfile
  ├── sync_offset: AtomicI64 (frames)
  ├── stitch_params: StitchParams (6 calibration values)
  ├── dual_renderer: DualPlaneRenderer (WgpuWrapper variant)
  ├── left_decoder: FfmpegDecoder
  ├── right_decoder: FfmpegDecoder
  └── encoder: FfmpegEncoder
```

The CLI entry point calls methods on `StitchingManager` directly. A future UI layer (Tauri or Qt) exposes the same methods via IPC.

---

## 6. v2 Architecture Specification

### 6.1 Module Structure

```
reco-v2/
├── Cargo.toml                      # workspace
├── crates/
│   ├── reco-core/                  # Library crate (no UI, no CLI)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── gpu/
│   │   │   │   ├── mod.rs          # BufferSource, FrameImporter
│   │   │   │   ├── dual_renderer.rs # DualPlaneRenderer (extends WgpuWrapper)
│   │   │   │   ├── wgpu_interop.rs  # Copied/adapted from Gyroflow
│   │   │   │   ├── wgpu_interop_*.rs # Platform impls (copied from Gyroflow)
│   │   │   │   └── dual_plane.wgsl  # New dual-plane projection shader
│   │   │   ├── pipeline/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── dual_decoder.rs  # Two FfmpegProcessors in lockstep
│   │   │   │   ├── encoder.rs       # HW encode (NVENC/VideoToolbox/VAAPI)
│   │   │   │   └── sync.rs          # Optical flow temporal sync
│   │   │   ├── calibration/
│   │   │   │   ├── mod.rs           # StitchParams, Powell optimiser
│   │   │   │   ├── feature_match.rs # SIFT/SuperPoint matching
│   │   │   │   └── ransac.rs        # Inlier filtering
│   │   │   ├── ai/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── detector.rs      # YOLOv8n inference (ONNX runtime or candle)
│   │   │   │   ├── tracker.rs       # ByteTrack multi-object tracker
│   │   │   │   ├── geometry.rs      # Back-projection to world space
│   │   │   │   └── director.rs      # Kalman filter → pan/tilt/zoom keyframes
│   │   │   ├── lens/
│   │   │   │   └── mod.rs           # Re-export gyroflow_core::LensProfile
│   │   │   └── manager.rs           # StitchingManager (top-level state machine)
│   ├── reco-cli/                    # Binary crate
│   │   └── src/main.rs              # clap CLI → StitchingManager
│   └── reco-ui/                     # (Future) Tauri or Qt UI
│       └── src/main.rs
├── deps/
│   └── gyroflow_core/               # Git submodule (Gyroflow core library)
└── profiles/                        # Gyroflow lens profile database (submodule)
```

### 6.2 The GPU Pipeline

The central pipeline runs fully on-GPU. No CPU touches pixel data between decode and encode:

```
[Left video]   → FFmpeg NVDEC → CUdeviceptr_L ──┐
                                                   │
                                                   ├─ DualPlaneRenderer (wgpu) ─┐
                                                   │   ┌──────────────────────┐  │
[Right video]  → FFmpeg NVDEC → CUdeviceptr_R ──┘   │  dual_plane.wgsl      │  │
                                                      │  (two input textures) │  │
                                                      └──────────────────────┘  │
                                                                                 │
                                                   CUdeviceptr_out ─────────────┘
                                                          │
                                                   FFmpeg NVENC → [Output file]
```

**Frame loop (post-processing mode):**
```rust
// Pseudocode for the per-frame GPU loop
let (left_pts, left_frame) = left_decoder.decode_next()?;
let (right_pts, right_frame) = right_decoder.decode_next_at(left_pts + sync_offset)?;

let out_frame = renderer.process_dual(
    FrameSource::Cuda(left_frame.data_ptr()),
    FrameSource::Cuda(right_frame.data_ptr()),
    &stitch_params,
    &kernel_params,
)?;

encoder.encode(out_frame)?;
```

**Frame loop (live mode):**
Replace `FFmpeg NVDEC` with platform capture APIs (e.g., `v4l2` on Linux, `AVFoundation` on macOS). The render loop is driven by a real-time clock at the target output framerate.

### 6.3 The Dual-Plane Projection Shader (`dual_plane.wgsl`)

The shader must replace Gyroflow's single-view model with a two-plane scene:

```wgsl
// Inputs
@group(0) @binding(0) var<uniform> params: StitchKernelParams;
@group(0) @binding(1) var left_texture:  texture_2d<f32>;
@group(0) @binding(2) var right_texture: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler:   sampler;

struct StitchKernelParams {
    // Camera position (on x=z bisector)
    cam_d:   f32,         // distance from origin

    // Left plane (z-axis plane)
    left_z_pos:    f32,   // z position of plane centre (= plane_width/2 * (1-intersect))
    left_y_offset: f32,   // y translation (xTy)
    left_rz:       f32,   // z rotation around plane centre (horizon levelling - NEW)
    left_rx:       f32,   // x rotation (zRx) - tilt

    // Right plane (x-axis plane)
    right_x_pos:   f32,   // x position of plane centre
    right_rz:       f32,  // z rotation (xRz) - tilt

    // Output viewport
    fov_h: f32, fov_v: f32,
    yaw: f32, pitch: f32,  // camera orientation

    // Left lens params
    left_fx: f32, left_fy: f32, left_cx: f32, left_cy: f32,
    left_k: vec4<f32>,

    // Right lens params
    right_fx: f32, right_fy: f32, right_cx: f32, right_cy: f32,
    right_k: vec4<f32>,
}

@fragment
fn main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // 1. Screen pixel → world ray direction (apply yaw/pitch rotation)
    let ray = screen_to_ray(frag_pos.xy, params.fov_h, params.fov_v,
                             params.yaw, params.pitch);

    // 2. Intersect ray with left plane (z-axis aligned, rotated)
    let left_hit  = intersect_left_plane(ray, params);
    let right_hit = intersect_right_plane(ray, params);

    // 3. Determine which plane to sample from (prefer closer hit)
    if left_hit.valid && (!right_hit.valid || left_hit.t < right_hit.t) {
        let uv = kb4_forward(left_hit.uv, params.left_fx, params.left_fy,
                              params.left_cx, params.left_cy, params.left_k);
        if in_bounds(uv) {
            return textureSample(left_texture, tex_sampler, uv);
        }
    }

    if right_hit.valid {
        let uv = kb4_forward(right_hit.uv, params.right_fx, params.right_fy,
                              params.right_cx, params.right_cy, params.right_k);
        if in_bounds(uv) {
            return textureSample(right_texture, tex_sampler, uv);
        }
    }

    return vec4<f32>(0.0);  // black for out-of-bounds
}

// KB4 forward model (same math as Gyroflow's opencv_fisheye.rs distort_point)
fn kb4_forward(pt_3d: vec3<f32>, fx: f32, fy: f32, cx: f32, cy: f32, k: vec4<f32>) -> vec2<f32> {
    let pt = pt_3d.xy / pt_3d.z;
    let r = length(pt);
    let theta = atan(r);
    let t2 = theta * theta;
    let theta_d = theta * (1.0 + k.x*t2 + k.y*t2*t2 + k.z*t2*t2*t2 + k.w*t2*t2*t2*t2);
    let scale = select(theta_d / r, 1.0, r < 1e-6);
    return vec2<f32>(fx * pt.x * scale + cx, fy * pt.y * scale + cy);
}
```

The blend region in the overlap zone is handled by the priority logic and optionally by a separate blend weight that feathers between left and right samples when both planes are hit within a threshold distance.

### 6.4 The Calibration Pipeline (CPU-side)

Calibration runs headlessly on CPU. The flow:

```
1. Optical flow sync: compute offset between left and right streams
   └── DenseOpticalFlow::compute_magnitude_signal() → cross_correlate() → offset_frames

2. Frame selection: pick N calibration frames spread across the video
   └── avoid uniform-color frames (low texture variance)

3. Per-frame undistortion: apply KB4 undistortion on CPU
   (LensProfile::get_camera_matrix() + LensProfile::get_distortion_coeffs())
   └── cv::fisheye::undistortImage() or wgpu CPU path

4. Feature matching: SIFT on overlap region, RANSAC filtering
   └── SuperPoint (GPU) as optional upgrade for better homogeneous regions

5. Powell optimisation: minimise sum-of-angular-errors over all N frames
   └── 6 parameters: cam_d, intersect, xTy, left_rz (NEW), left_rx, right_rz

6. Confidence scoring: report RMS reprojection error and inlier count
   └── reject and flag frames where optimisation diverges
```

RANSAC pre-filtering is crucial: before Powell, filter matched points by computing the expected disparity from a homography estimate and removing points that deviate from it. This replaces the `cv2.findFundamentalMat()` approach with a more geometrically motivated filter.

### 6.5 The AI Director Pipeline

```
[Left undistorted frame]  → YOLOv8n detect → BoundingBoxes_L ──┐
                                                                  ├→ ByteTrack → TrackID → back_project()
[Right undistorted frame] → YOLOv8n detect → BoundingBoxes_R ──┘         → WorldPoint(x,y,z)
                                                                           → KalmanFilter
                                                                           → pan/tilt/zoom keyframe
```

**Back-projection:** given a pixel position `(u,v)` on the undistorted left plane, the world point is `(u/plane_width * plane_width, y, left_z_pos)` (after applying the plane's calibration transform). The Director converts this to a camera yaw/pitch/FOV command.

**Kalman filter:** a constant-velocity model tracks the predicted ball position. The Director outputs smooth keyframes that avoid jarring camera motion during tracking gaps. Aggressiveness parameters (max slew rate, zoom trigger threshold) are configurable.

**Inference without GPU→CPU copies:** YOLOv8n can be run via ONNX Runtime with CUDA execution provider. The input is a `CUdeviceptr` pointing to the undistorted frame. ONNX Runtime CUDA EP accepts `OrtMemTypeDefault` tensors backed by CUDA memory. This eliminates the GPU→CPU copy for inference.

### 6.6 Live vs Post-Processing Mode

| Aspect | Post-Processing | Live Streaming |
|---|---|---|
| Input source | `FfmpegDecoder` (file) | Platform capture (V4L2/AVFoundation) |
| Sync | Pre-computed offset, applied as seek trim | Real-time clock + PLL to lock drift |
| Output | File (MP4/MKV via NVENC) | RTMP/SRT stream (hardware encode) |
| Director | Keyframes baked into file | Real-time commands to overlay renderer |
| Calibration | Run once, save as `.stitch` project | Load saved project at startup |
| Frame rate | Match source FPS | Match output FPS; drop frames if encoder falls behind |

For live mode, a Phase-Locked Loop maintains sync between the two capture streams. The PLL input is the optical flow cross-correlation score computed on a sliding 5-second window; its output adjusts a hardware timestamp offset applied to one stream's capture buffer.

---

## 7. Key Engineering Challenges

### 7.1 Zero-Copy GPU Frame Path (wgpu ↔ NVENC/VideoToolbox/VAAPI)

**Problem:** The wgpu output texture is in Vulkan/Metal/D3D12 memory; the hardware encoder (NVENC/VideoToolbox) expects its own native surface format. Getting pixels from one to the other without a CPU roundtrip is the hardest single engineering challenge.

**How Gyroflow addresses it:** Gyroflow's `handle_output_texture_post()` (line 451, `wgpu_interop.rs`) for CUDA: after the wgpu render, it does a `cuda_2d_copy_on_device()` from the Vulkan image backing memory to the consumer's CUDA buffer. This is a device-side copy (no PCIe transfer), but it is still a copy. For NVENC, the input would be a `CUdeviceptr`; the CUDA copy brings the data there.

**v2 approach:**
- **NVIDIA (Linux/Windows):** Export the wgpu output Vulkan image as a CUDA external memory handle, then pass the `CUdeviceptr` directly to `nvEncMapInputResource()`. One device-side copy, or zero-copy if the NVENC input format matches the wgpu output format directly.
- **Apple Silicon (macOS):** The wgpu Metal output texture can be passed directly to VideoToolbox's `CVPixelBufferCreateWithPlanarBytes()` or via a `CVPixelBuffer` backed by the same `MTLBuffer`. VideoToolbox then encodes from this IOSurface-backed buffer with no copy.
- **AMD/Intel (Linux, VAAPI):** Use DMA-BUF fd sharing: export the Vulkan image as a DMA-BUF fd, import it into VA-API as an external surface. This is Linux-specific but gives true zero-copy.
- **Fallback:** wgpu texture → staging buffer → CPU map → NVENC CPU input. Viable for ≤1080p but defeats the purpose for 4K.

The wgpu `Texture::as_hal::<wgpu::hal::api::Vulkan>()` API (unstable but available) gives access to the `VkImage` handle, enabling the DMA-BUF or CUDA external memory path.

### 7.2 Dual-Stream Synchronisation and Clock Drift

**Problem:** Two independent cameras have independent crystal oscillators. At 1ppm difference, drift reaches 1 frame per ~8 minutes at 60fps. Over a 90-minute match, drift can exceed 10 frames.

**Gyroflow's approach:** Gyroflow synchronises one video to one gyro signal — not two videos. The optical flow mechanism measures relative motion per frame, which is inherently frame-rate-independent.

**v2 approach:**
1. **Initial offset:** optical flow cross-correlation on first 30 seconds of footage (described in Section 5.5)
2. **Drift model:** recompute offset every 60 seconds using a sliding 10-second window; fit a linear model `offset(t) = a + b*t` where `b` is the drift rate in frames/second
3. **Correction:** at each decoded frame, apply `offset(t)` by advancing or delaying the right stream's presentation timestamp before the renderer receives it
4. **Subframe interpolation:** for sub-frame drift corrections, use temporal interpolation between adjacent right-stream frames (blend `alpha * frame[t] + (1-alpha) * frame[t+1]`). In the GPU pipeline, this is a blend of two textures before the projection pass.

### 7.3 Seam Blending in the Overlap Region

**Problem:** At the boundary between left and right planes, two rendering paths are stitched together. Differences in exposure, white balance, lens vignetting, and chromatic aberration produce a visible seam.

**How Gyroflow addresses it:** Gyroflow renders a single camera; it has no seam. Its `background_mode` parameter handles edge pixels, but no multi-camera blending exists.

**v2 approach:**
1. **Color matching:** Reinhard LAB transfer (already implemented in v1's shader — `applyReinhardLAB()`) corrects global colour differences. Compute per-frame statistics in the overlap region.
2. **Feathered alpha blend:** in the overlap zone, the shader computes a blend weight based on distance to the seam. The weight is `w = smoothstep(0, blend_width, dist_from_seam)` — the left plane contributes `w`, the right `1-w`.
3. **Vignette correction:** estimate the per-camera vignette model (radially symmetric polynomial) from calibration frames and correct it in the shader before blending.
4. **Gradient domain blending (post-processing only):** for export, run a Poisson equation solver over the seam band to achieve gradient-domain blending (no visible seam even at different exposures). This runs as a post-process pass on the GPU via a wgpu compute shader.

### 7.4 Horizon Levelling

**Problem:** The left camera's roll (rotation around the optical axis) maps to a z-rotation of the left plane around its centre. v1 has no parameter for this. The result is a horizon that tilts when the camera is not perfectly level.

**Gyroflow's approach:** Gyroflow has an explicit `HorizonLock` smoothing algorithm (`src/core/smoothing/horizon.rs`) that corrects for gyro-measured roll. It applies an additional rotation to the stabilisation matrix to keep the horizon level.

**v2 approach:**
Add `left_rz` (left plane z-rotation around plane centre) as the 6th Powell parameter. In the shader, apply this rotation to the left plane's basis vectors before the plane-ray intersection. Expose it as a manually adjustable parameter in the UI with a "level horizon" auto-detection button that detects vanishing point tilt from the calibration images.

The Powell bounds: `left_rz ∈ [-π/4, π/4]` (±45°, sufficient for any reasonable mounting angle).

### 7.5 AI Inference Without GPU→CPU Frame Copies

**Problem:** Standard YOLO inference pipelines expect CPU tensors. Copying GPU frames to CPU for inference negates the zero-copy GPU pipeline.

**Gyroflow's approach:** Gyroflow does not include inference; no relevant precedent.

**v2 approach:**
- **ONNX Runtime CUDA EP:** accepts `OrtCUDAProviderOptions` with a pre-allocated CUDA stream. Input tensors can be created from existing `CUdeviceptr` allocations using `OrtApi::CreateTensorWithDataAsOrtValue()` with `OrtMemTypeCUDA`. This bypasses the CPU entirely.
- **Alternative (candle):** the `candle-transformers` crate includes YOLO implementations. Candle tensors can be backed by CUDA memory. However, candle's CUDA support is less mature than ONNX Runtime.
- **Input format:** YOLOv8n expects RGB `float32` tensors normalized to [0, 1]. The wgpu output is typically `Rgba16Float` or `Rgba8Unorm`. A small CUDA kernel (or wgpu compute pass) converts the format before passing to ONNX Runtime.
- **Parallelism:** run inference on a separate CUDA stream from the render/encode stream using `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`. The detector stream and render stream share device ownership but execute concurrently.

### 7.6 Cross-Platform wgpu Backend Differences

**Problem:** wgpu presents a unified API but the underlying backends (Vulkan, Metal, D3D12, OpenGL) have significant capability differences affecting the interop paths.

| Platform | wgpu Backend | Hardware Decode | Hardware Encode | Zero-Copy Path |
|---|---|---|---|---|
| Linux (NVIDIA) | Vulkan | NVDEC (CUDA) | NVENC (CUDA) | CUDA external memory → VkImage |
| Linux (AMD/Intel) | Vulkan | VAAPI | VAAPI | DMA-BUF fd |
| macOS (Apple Silicon) | Metal | VideoToolbox | VideoToolbox | MTLTexture/IOSurface |
| macOS (Intel) | Metal | VideoToolbox (limited) | VideoToolbox | MTLTexture copy |
| Windows (NVIDIA) | Vulkan or D3D12 | NVDEC (CUDA) | NVENC | CUDA ext. mem or D3D12 shared |
| Windows (AMD) | D3D12 | D3D11VA | AMF | D3D11/12 shared resource |
| Windows (Intel) | D3D12 | QSV | QSV | D3D11/12 shared resource |

**Gyroflow's approach:** The `BufferSource` enum handles all cases via compile-time `#[cfg]` guards and runtime backend selection. The `is_buffer_supported()` function (line 557, `wgpu.rs`) gates capabilities per backend.

**v2 approach:** Follow Gyroflow's pattern exactly. Build with the same conditional compilation. Default to the CPU path for any unsupported platform; add GPU paths incrementally. Prioritise Linux+NVIDIA first (primary deployment target for sports broadcast).

For the seam blending Poisson solver, it requires `wgpu::Features::STORAGE_TEXTURE_BINDING_ARRAY` and writeable storage textures — available on Vulkan and Metal but not necessarily on older D3D12 configurations. Gate this feature at runtime and fall back to alpha blending.

---

## 8. What to Reuse from Gyroflow vs Build Fresh

### Reuse Verbatim (zero modification)

| Component | Reason |
|---|---|
| `gyroflow/src/core/gpu/wgpu_interop.rs` (all platform files) | Solves the hardest cross-platform interop problem; battle-tested |
| `gyroflow/src/core/gpu/mod.rs` (`BufferSource`, `BufferDescription`) | Correct abstraction for GPU frame sources |
| `gyroflow/src/rendering/ffmpeg_hw.rs` | `HWDevice` is a clean, correct FFmpeg HW context wrapper |
| `gyroflow/src/core/lens_profile.rs` | Full profile format, matrix scaling, interpolation |
| `gyroflow/src/core/lens_profile_database.rs` | Profile search and loading |
| `gyroflow/src/core/calibration/mod.rs` | `LensCalibrator` for initial per-camera intrinsic calibration |
| `gyroflow/src/core/gpu/stabilize_spirv/src/distortion_models/opencv_fisheye.rs` | KB4 forward/backward math |
| Gyroflow's lens profile JSON files (database) | Thousands of pre-calibrated cameras |

### Adapt (start from Gyroflow, extend significantly)

| Component | What Changes |
|---|---|
| `wgpu.rs` → `dual_renderer.rs` | Add second input texture binding; change pipeline layout; add blend pass |
| `wgpu_undistort.wgsl` → `dual_plane.wgsl` | Replace single-view model with two-plane geometry; add ray-plane intersection; add horizon parameter |
| `ffmpeg_processor.rs` → `dual_decoder.rs` | Add second decoder; synchronise frame presentation timestamps |
| `stabilization/compute_params.rs` → `stitch_params.rs` | Replace gyro-rotation model with 6-parameter plane geometry |
| `calibration/mod.rs` → Use as library; add Powell optimiser around it | The intrinsic calibration is reused; the extrinsic multi-camera optimisation is new |

### Build Fresh

| Component | Reason |
|---|---|
| `dual_plane.wgsl` projection logic | Entirely new geometry (two orthogonal planes, ray-casting, camera model) |
| Optical flow temporal sync | Gyroflow's sync is for video+gyro alignment, not two-video alignment |
| Powell 6-parameter extrinsic optimiser | Custom cost function (sum-of-angles) with new parameter set |
| YOLO detector + ByteTrack tracker | No equivalent in Gyroflow |
| Kalman Director (pan/tilt/zoom keyframes) | Application-specific; no equivalent |
| CLI entry point | Gyroflow's CLI requires Qt; v2 uses `clap` + `tokio`, no Qt dependency |
| Live capture integration (V4L2/AVFoundation) | Gyroflow only processes files |
| Seam blending (Poisson solver) | Not needed in single-camera stabilisation |

---

## 9. Recommended Implementation Order

### Phase 1: CPU Pipeline MVP (weeks 1–4)

**Goal:** prove the calibration geometry is correct end-to-end, headlessly.

1. Set up Cargo workspace; add `gyroflow_core` as a dependency
2. Load a Gyroflow lens profile and undistort a test frame on CPU (verify against v1 shader output)
3. Implement the 6-parameter Powell optimiser (`reco-core/src/calibration/mod.rs`)
4. Add the missing `left_rz` horizon-levelling parameter; validate with a tilted test case
5. Implement frame extraction via `ffmpeg-next` (CPU decode, no HW)
6. Implement SIFT feature matching + RANSAC on CPU (using OpenCV via `opencv-rust` or pure Rust `image` crate)
7. CLI: `reco calibrate --left video_L.mp4 --right video_R.mp4 --left-profile gopro.json --right-profile gopro.json --out calibration.stitch`

**Exit criterion:** the CLI produces a `.stitch` file with calibration parameters that match v1's output on a known test case.

### Phase 2: GPU Render Pipeline (weeks 5–8)

**Goal:** GPU undistortion + dual-plane projection to an output frame.

1. Port Gyroflow's `WgpuWrapper` + `wgpu_interop` as `DualPlaneRenderer`
2. Write `dual_plane.wgsl`: screen → ray → plane intersection → KB4 forward → texture sample
3. CPU texture input first (PNG in, PNG out) to validate geometry
4. Add second texture binding; validate with two test frames
5. Add EWA sampling (copy from `wgpu_undistort.wgsl`)
6. Add feathered alpha blend in overlap zone
7. Add LAB Reinhard color matching as compute pre-pass

**Exit criterion:** `reco render --left frame_L.png --right frame_R.png --stitch calib.stitch --out panorama.png` produces a correctly stitched image.

### Phase 3: Hardware Decode/Encode Pipeline (weeks 9–12)

**Goal:** GPU decode → GPU render → GPU encode, zero CPU pixel roundtrip.

1. Integrate `ffmpeg_hw.rs` from Gyroflow; instantiate NVDEC for left and right streams
2. Extract `CUdeviceptr` from each decoded `AVFrame`; pass to `DualPlaneRenderer` via `BufferSource::CUDABuffer`
3. Verify zero-copy path: add a debug assertion that no `staging_buffer` map operations occur during the render loop
4. Implement NVENC output encoder; receive CUDA buffer from renderer, pass to `nvEncMapInputResource()`
5. Implement frame lockstep: drain both decoders with timestamp-based synchronisation, applying the pre-computed sync offset
6. CLI: `reco stitch --left left.mp4 --right right.mp4 --stitch calib.stitch --out stitched.mp4`

**Exit criterion:** a 4K 60fps stitch runs at ≥60fps on target hardware; `nvidia-smi` shows no PCIe memory transfers during the render loop.

### Phase 4: Optical Flow Sync (weeks 13–14)

**Goal:** replace audio cross-correlation with GPU optical flow temporal sync.

1. Implement dense optical flow on GPU (wgpu compute shader using Lucas-Kanade or integrate RAFT via ONNX Runtime)
2. Compute per-frame motion magnitude signals for both streams
3. FFT-based cross-correlation (wgpu compute or cuFFT)
4. Linear drift model from multiple windows
5. Integrate into `DualDecoder` as a pre-processing step

**Exit criterion:** `reco sync` detects sub-frame offset and drift on test footage with no audio; compare against known ground-truth offset.

### Phase 5: AI Director (weeks 15–18)

**Goal:** ball tracking → pan/tilt/zoom keyframes.

1. Export undistorted left/right frames to CUDA tensors without copy
2. Integrate YOLOv8n via ONNX Runtime CUDA EP; validate detection accuracy
3. Implement ByteTrack tracker; verify track continuity across occlusions
4. Back-project detected positions to world coordinates using calibration geometry
5. Kalman Director: constant-velocity model, smooth keyframe interpolation
6. Output keyframes to a sidecar JSON file and apply to the render pipeline

**Exit criterion:** the Director tracks the ball in a test match clip; output video shows correct camera pan following ball movement.

### Phase 6: Live Mode (weeks 19–22)

**Goal:** real-time capture and streaming.

1. V4L2 capture (Linux) or AVFoundation (macOS) for USB/HDMI cameras
2. PLL-based drift correction (replace file-seek offset with timestamp adjustment)
3. RTMP/SRT output via `librtmp` or `srt-rs`
4. Director → live overlay compositor (player name, score bug, etc.)
5. CLI: `reco live --left /dev/video0 --right /dev/video1 --stitch calib.stitch --out rtmp://...`

**Exit criterion:** live stitch runs at 60fps output with <200ms end-to-end latency; director tracks ball correctly in live footage.

### Phase 7: Polish and Platform Support (weeks 23–26)

1. macOS Metal zero-copy path (VideoToolbox ↔ wgpu Metal)
2. Linux VAAPI DMA-BUF path
3. Windows D3D12 path
4. UI layer (Tauri or Qt) wrapping `reco-core` via IPC
5. Gyroflow lens profile database integration and browsing UI
6. `reco calibrate` wizard with visual feedback on calibration quality

---

*End of paper. All file references use `gyroflow/` and `video-stitcher/` as repository roots within the research directory.*
