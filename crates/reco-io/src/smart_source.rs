//! GPU-aware file source with automatic backend selection.
//!
//! [`SmartFileSource`] is the standard way to open video files for stitching.
//! It probes the input, detects the best decode path (GPU zero-copy or CPU),
//! and delivers frames through the [`FrameSource`] trait.
//!
//! On capable hardware (NVIDIA + Vulkan, Apple VideoToolbox + Metal), frames
//! are GPU-resident - no CPU-GPU transfer needed. On other hardware, frames
//! are decoded to CPU transparently.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_io::SmartFileSource;
//!
//! let gpu = reco_core::gpu::GpuContext::new_blocking()?;
//! let inputs = [reco_io::InputPath::Single("panorama.mp4".into())];
//! let mut source = SmartFileSource::open(&inputs, &gpu, 0)?;
//! // source implements FrameSource - pass to session.run()
//! ```

use reco_core::render::renderer::GpuPixelFormat;
use reco_core::source::{FrameSet, FrameSource, SourceError, SourceInfo};

/// GPU-aware file source that auto-selects the optimal decode path.
///
/// Probes the input files at construction and selects the best available
/// backend: CUDA/Vulkan zero-copy on Linux, D3D11VA/DX12 on Windows, VideoToolbox/Metal
/// on macOS, or CPU software decode as a fallback. The selection is
/// transparent to the consumer.
///
/// For GPU zero-copy paths, this source owns the shared textures and
/// decode threads. The session creates bind groups from the source's
/// textures at the start of `run()`.
pub struct SmartFileSource {
    mode: SourceMode,
    info: SourceInfo,
    pixel_format: GpuPixelFormat,
    full_range: bool,
    left_rotation: i32,
    right_rotation: i32,
    /// Human-readable description of the active decode path.
    decode_mode: &'static str,
    /// True once the active backend has signaled end-of-stream.
    ///
    /// For the CPU mode this tracks the inner `FfmpegFileSource`'s flag;
    /// for zero-copy modes this is set by an explicit EOF message.
    exhausted: bool,
    terminal_error: Option<SourceError>,
}

enum SourceMode {
    Cpu(Box<crate::adapters::FfmpegFileSource>),
    #[cfg(target_os = "linux")]
    GpuZeroCopy(Box<LinuxZeroCopyState>),
    #[cfg(target_os = "macos")]
    MetalZeroCopy(
        std::sync::mpsc::Receiver<
            crate::decode_stream::Message<reco_core::interop::zero_copy::VtFramePair>,
        >,
    ),
    #[cfg(target_os = "windows")]
    D3d11ZeroCopy(Box<WindowsZeroCopyState>),
}

#[cfg(target_os = "windows")]
enum WindowsDecodeState {
    Pending {
        left: crate::stitch_job::InputPath,
        right: crate::stitch_job::InputPath,
        sync_offset: i64,
    },
    Running {
        pair_rx: crate::decode_stream::OwnedPair<crate::ffmpeg::decoder::D3d11Frame>,
    },
    Failed(SourceError),
}

#[cfg(target_os = "windows")]
struct WindowsZeroCopyState {
    /// Holds the D3d11Frame pair from the most recent `next_frame()` call.
    ///
    /// The `_frame_ref` inside each D3d11Frame pins the decode pool slice
    /// in FFmpeg's surface allocator. Without this, the slice is recycled
    /// as soon as the raw pointers are extracted into `FrameSet::D3d11Resident`,
    /// and the decoder thread can overwrite the surface before `stage_frames`
    /// copies it - causing frame reordering glitches on Intel and NVIDIA.
    ///
    /// Dropped when the next pair arrives (or the source is dropped),
    /// which is always after `stage_frames` has completed for this pair.
    live_frame_guard: Option<crate::zero_copy::D3d11FramePair>,
    // Release the displayed surfaces before cancelling and joining workers.
    decode: WindowsDecodeState,
}

#[cfg(target_os = "windows")]
impl WindowsZeroCopyState {
    fn ensure_running(&mut self) {
        if let WindowsDecodeState::Pending {
            left,
            right,
            sync_offset,
        } = &self.decode
        {
            let left = left.clone();
            let right = right.clone();
            let sync_offset = *sync_offset;
            self.decode =
                match crate::zero_copy::spawn_d3d11_decode_pair(&left, &right, sync_offset) {
                    Ok(pair_rx) => WindowsDecodeState::Running { pair_rx },
                    Err(error) => WindowsDecodeState::Failed(error),
                };
        }
    }
}

#[cfg(target_os = "linux")]
struct LinuxZeroCopyState {
    /// Frame-set receiver. Drop takes it early to unblock the collector
    /// before joining decode workers.
    frame_rx: Option<std::sync::mpsc::Receiver<crate::decode_stream::Message<Vec<u8>>>>,
    /// Shared textures (kept alive until Drop).
    shared: reco_core::interop::SharedTextureSet,
    /// Decode thread join handles.
    join_handles: Vec<std::thread::JoinHandle<()>>,
    /// Shutdown flag checked by decode threads for graceful exit.
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl SmartFileSource {
    /// Open a file source with automatic backend selection.
    ///
    /// Probes input metadata and chooses native residency when the decoder,
    /// GPU and input layout support it. CUDA/Vulkan supports one or two
    /// inputs; D3D11 and Metal imports require two.
    ///
    /// `sync_offset` applies temporal alignment between cameras:
    /// positive skips camera 1, negative skips camera 0. Ignored for mono.
    pub fn open(
        inputs: &[crate::stitch_job::InputPath],
        gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
    ) -> Result<Self, SourceError> {
        let Some(left) = inputs.first() else {
            return Err(SourceError::Init {
                path: "(none)".into(),
                reason: "at least one input video is required".into(),
            });
        };
        let left_probe_path = left.first_path();

        let probe = crate::ffmpeg::decoder::VideoDecoder::open(left_probe_path).map_err(|e| {
            SourceError::Init {
                path: left_probe_path.display().to_string(),
                reason: format!("{e}"),
            }
        })?;

        let fps_r = probe.frame_rate();
        let info = SourceInfo {
            width: probe.width(),
            height: probe.height(),
            fps: probe.fps(),
            fps_rational: Some((fps_r.0, fps_r.1)),
            total_frames: None,
        };
        let pixel_format = probe.pixel_format();
        let full_range = probe.is_full_range();
        let left_rotation = probe.rotation();
        let decode_backend = probe.backend();
        drop(probe);

        let mut right_rotation = 0;
        let mut matching_layout = true;
        for (camera, input) in inputs.iter().enumerate().skip(1) {
            let probe =
                crate::ffmpeg::decoder::VideoDecoder::open(input.first_path()).map_err(|e| {
                    SourceError::Init {
                        path: input.first_path().display().to_string(),
                        reason: e.to_string(),
                    }
                })?;
            if probe.is_full_range() != full_range {
                return Err(SourceError::Init { path: input.first_path().display().to_string(), reason: "camera color ranges differ; mixed full-range and limited-range inputs are not supported".into() });
            }
            matching_layout &= probe.width() == info.width
                && probe.height() == info.height
                && probe.pixel_format() == pixel_format
                && is_backend_zero_copy_capable(probe.backend());
            if camera == 1 {
                right_rotation = probe.rotation();
            }
        }
        // The other native importers still consume pairs.
        let native_input_count = if cfg!(target_os = "linux") {
            matches!(inputs.len(), 1 | 2)
        } else {
            inputs.len() == 2
        };

        // Detect zero-copy capability
        let hwaccel_disabled = std::env::var("RECO_NO_HWACCEL").is_ok();
        if hwaccel_disabled {
            log::info!("RECO_NO_HWACCEL set, forcing CPU decode path");
        }
        let backend_capable = is_backend_zero_copy_capable(decode_backend);
        let gpu_capable = gpu.supports_zero_copy();
        let use_zero_copy = !hwaccel_disabled
            && backend_capable
            && gpu_capable
            && matching_layout
            && native_input_count;

        if !use_zero_copy && !hwaccel_disabled {
            log::info!(
                "Zero-copy disabled: decode_backend={decode_backend} (capable={backend_capable}), \
                 gpu_zero_copy={gpu_capable}, matching_layout={matching_layout}, \
                 supported_input_count={native_input_count}. Using CPU upload path."
            );
        }

        if use_zero_copy {
            Self::open_zero_copy(
                inputs,
                gpu,
                sync_offset,
                info,
                pixel_format,
                full_range,
                left_rotation,
                right_rotation,
            )
        } else {
            Self::open_cpu(inputs, sync_offset, false, info, pixel_format, full_range)
        }
    }

    /// Open a CPU-decode source for any number of inputs (projection
    /// order). Format, fps, and color range are probed from input 0;
    /// per-input rotation is probed by the decode layer.
    pub fn open_cpu_only(
        inputs: &[crate::stitch_job::InputPath],
        sync_offset: i64,
        software_decode: bool,
    ) -> Result<Self, SourceError> {
        let [first, ..] = inputs else {
            return Err(SourceError::Init {
                path: "(none)".into(),
                reason: "at least one input video is required".into(),
            });
        };
        let probe_path = first.first_path();

        let open_probe = if software_decode {
            crate::ffmpeg::decoder::VideoDecoder::open_software
        } else {
            crate::ffmpeg::decoder::VideoDecoder::open
        };
        let probe = open_probe(probe_path).map_err(|e| SourceError::Init {
            path: probe_path.display().to_string(),
            reason: format!("{e}"),
        })?;
        let fps_r = probe.frame_rate();
        let info = SourceInfo {
            width: probe.width(),
            height: probe.height(),
            fps: probe.fps(),
            fps_rational: Some((fps_r.0, fps_r.1)),
            total_frames: None,
        };
        let pixel_format = probe.pixel_format();
        let full_range = probe.is_full_range();
        drop(probe);

        Self::open_cpu(
            inputs,
            sync_offset,
            software_decode,
            info,
            pixel_format,
            full_range,
        )
    }

    fn open_cpu(
        inputs: &[crate::stitch_job::InputPath],
        sync_offset: i64,
        software_decode: bool,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        full_range: bool,
    ) -> Result<Self, SourceError> {
        let source =
            crate::adapters::FfmpegFileSource::open_inputs(inputs, sync_offset, software_decode)?;
        log::info!(
            "SmartFileSource: CPU decode ({} camera(s), {}x{}, {pixel_format:?}{})",
            inputs.len(),
            info.width,
            info.height,
            if full_range { ", full-range" } else { "" }
        );
        let left_rotation = source.rotation(0);
        let right_rotation = source.rotation(1);
        Ok(Self {
            mode: SourceMode::Cpu(Box::new(source)),
            info,
            pixel_format,
            full_range,
            left_rotation,
            right_rotation,
            decode_mode: "CPU upload",
            exhausted: false,
            terminal_error: None,
        })
    }

    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        inputs: &[crate::stitch_job::InputPath],
        gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        full_range: bool,
        left_rotation: i32,
        right_rotation: i32,
    ) -> Result<Self, SourceError> {
        use reco_core::interop::SharedTextureSet;
        use reco_core::interop::vulkan::{Nv12Plane, create_nv12_shared_texture};
        use reco_core::interop::zero_copy::GpuBufInfo;

        let map_err = |msg: String| SourceError::Init {
            path: inputs[0].first_path().display().to_string(),
            reason: msg,
        };

        let input_width = info.width;
        let input_height = info.height;

        // NVIDIA driver workaround for R16Unorm: see issue #134
        let _warmup = if pixel_format == GpuPixelFormat::P010 {
            let y = create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Y, pixel_format)
                .map_err(|e| map_err(format!("warmup Y: {e}")))?;
            let uv = create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Uv, pixel_format)
                .map_err(|e| map_err(format!("warmup UV: {e}")))?;
            Some((y, uv))
        } else {
            None
        };

        // Create double-buffered shared textures
        let create_pair = |label: &str| -> Result<_, SourceError> {
            let y = create_nv12_shared_texture(
                gpu,
                input_width,
                input_height,
                Nv12Plane::Y,
                pixel_format,
            )
            .map_err(|e| map_err(format!("{label} Y: {e}")))?;
            let uv = create_nv12_shared_texture(
                gpu,
                input_width,
                input_height,
                Nv12Plane::Uv,
                pixel_format,
            )
            .map_err(|e| map_err(format!("{label} UV: {e}")))?;
            Ok((y, uv))
        };

        let mut cameras = Vec::with_capacity(inputs.len());
        let mut decoders = Vec::with_capacity(inputs.len());
        for input in inputs {
            let (y0, uv0) = create_pair("slot0")?;
            let (y1, uv1) = create_pair("slot1")?;
            let buf = GpuBufInfo {
                y_ptr: [y0.cuda_ptr, y1.cuda_ptr],
                uv_ptr: [uv0.cuda_ptr, uv1.cuda_ptr],
                y_pitch: [y0.pitch, y1.pitch],
                uv_pitch: [uv0.pitch, uv1.pitch],
                width: input_width,
                height: input_height,
                pixel_format,
            };
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            tx.send(0).expect("seed slot");
            tx.send(1).expect("seed slot");
            decoders.push((input.clone(), buf.clone(), rx));
            cameras.push(reco_core::interop::shared_textures::SharedCameraTextures {
                textures: [y0, uv0, y1, uv1],
                buf,
                slot_free_tx: tx,
            });
        }
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let decode_handles =
            crate::zero_copy::spawn_decode_threads_gpu(decoders, sync_offset, shutdown.clone());
        let shared = SharedTextureSet { cameras };
        log::info!(
            "SmartFileSource: GPU zero-copy ({} camera(s), {input_width}x{input_height}, {pixel_format:?})",
            inputs.len()
        );

        Ok(Self {
            mode: SourceMode::GpuZeroCopy(Box::new(LinuxZeroCopyState {
                frame_rx: Some(decode_handles.frame_rx),
                shared,
                join_handles: decode_handles.join_handles,
                shutdown,
            })),
            info,
            pixel_format,
            full_range,
            left_rotation,
            right_rotation,
            decode_mode: "GPU zero-copy (CUDA/Vulkan)",
            exhausted: false,
            terminal_error: None,
        })
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        inputs: &[crate::stitch_job::InputPath],
        _gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        full_range: bool,
        left_rotation: i32,
        right_rotation: i32,
    ) -> Result<Self, SourceError> {
        let pair_rx = crate::zero_copy::spawn_vt_decode_pair(&inputs[0], &inputs[1], sync_offset);

        log::info!(
            "SmartFileSource: Metal zero-copy ({}x{}, VideoToolbox decode{})",
            info.width,
            info.height,
            if full_range { ", full-range" } else { "" }
        );

        Ok(Self {
            mode: SourceMode::MetalZeroCopy(pair_rx),
            info,
            pixel_format,
            full_range,
            left_rotation,
            right_rotation,
            decode_mode: "Metal zero-copy (VideoToolbox)",
            exhausted: false,
            terminal_error: None,
        })
    }

    #[cfg(target_os = "windows")]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        inputs: &[crate::stitch_job::InputPath],
        _gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        full_range: bool,
        left_rotation: i32,
        right_rotation: i32,
    ) -> Result<Self, SourceError> {
        log::info!("SmartFileSource: D3D11VA zero-copy decode enabled");

        Ok(Self {
            mode: SourceMode::D3d11ZeroCopy(Box::new(WindowsZeroCopyState {
                decode: WindowsDecodeState::Pending {
                    left: inputs[0].clone(),
                    right: inputs[1].clone(),
                    sync_offset,
                },
                live_frame_guard: None,
            })),
            info,
            pixel_format,
            full_range,
            left_rotation,
            right_rotation,
            decode_mode: "D3D11VA zero-copy",
            exhausted: false,
            terminal_error: None,
        })
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        inputs: &[crate::stitch_job::InputPath],
        _gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        full_range: bool,
        _left_rotation: i32,
        _right_rotation: i32,
    ) -> Result<Self, SourceError> {
        log::info!("SmartFileSource: zero-copy not yet implemented for this platform, using CPU");
        Self::open_cpu(inputs, sync_offset, false, info, pixel_format, full_range)
    }

    /// Human-readable description of the active decode path.
    pub fn decode_mode(&self) -> &'static str {
        self.decode_mode
    }

    /// Access the shared texture set (GPU zero-copy only).
    ///
    /// The session uses this to create bind groups at the start of `run()`.
    /// Returns `None` for CPU-mode sources.
    #[cfg(target_os = "linux")]
    pub fn shared_texture_set(&self) -> Option<&reco_core::interop::SharedTextureSet> {
        match &self.mode {
            SourceMode::GpuZeroCopy(state) => Some(&state.shared),
            _ => None,
        }
    }
}

impl FrameSource for SmartFileSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<FrameSet>, SourceError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if self.exhausted {
            return Ok(None);
        }
        match &mut self.mode {
            SourceMode::Cpu(source) => {
                let frame = source.next_frame()?;
                if frame.is_none() {
                    self.exhausted = true;
                }
                Ok(frame)
            }
            #[cfg(target_os = "linux")]
            SourceMode::GpuZeroCopy(state) => {
                let rx = state
                    .frame_rx
                    .as_ref()
                    .expect("frame_rx taken during shutdown");
                match crate::decode_stream::receive(rx)
                    .inspect_err(|error| self.terminal_error = Some(error.clone()))?
                {
                    Some(signal) => Ok(Some(FrameSet::GpuResident { slots: signal })),
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                }
            }
            #[cfg(target_os = "macos")]
            SourceMode::MetalZeroCopy(rx) => match crate::decode_stream::receive(rx)
                .inspect_err(|error| self.terminal_error = Some(error.clone()))?
            {
                Some(pair) => Ok(Some(FrameSet::MetalResident([pair.left, pair.right]))),
                None => {
                    self.exhausted = true;
                    Ok(None)
                }
            },
            #[cfg(target_os = "windows")]
            SourceMode::D3d11ZeroCopy(state) => {
                state.ensure_running();
                let result = match &state.decode {
                    WindowsDecodeState::Running { pair_rx } => pair_rx.receive(),
                    WindowsDecodeState::Failed(error) => Err(error.clone()),
                    _ => unreachable!(),
                };
                match result.inspect_err(|error| self.terminal_error = Some(error.clone()))? {
                    Some((left, right)) => {
                        let frames = FrameSet::D3d11Resident([
                            reco_core::source::D3d11CameraFrame {
                                texture: left.texture,
                                slice: left.array_slice,
                            },
                            reco_core::source::D3d11CameraFrame {
                                texture: right.texture,
                                slice: right.array_slice,
                            },
                        ]);
                        // Keep the D3d11Frame pair alive so FFmpeg doesn't
                        // recycle the decode pool slices before stage_frames
                        // copies them. The previous guard is dropped here,
                        // which is safe because its staging copy already
                        // completed in the previous loop iteration.
                        state.live_frame_guard = Some((left, right));
                        Ok(Some(frames))
                    }
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                }
            }
        }
    }

    fn is_gpu_resident(&self) -> bool {
        !matches!(self.mode, SourceMode::Cpu(_))
    }

    fn skip_frames(&mut self, count: u64) -> Result<u64, SourceError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if self.exhausted {
            return Ok(0);
        }
        match &mut self.mode {
            #[cfg(target_os = "linux")]
            SourceMode::GpuZeroCopy(state) => {
                let rx = state
                    .frame_rx
                    .as_ref()
                    .expect("frame_rx taken during shutdown");
                for i in 0..count {
                    match crate::decode_stream::receive(rx)
                        .inspect_err(|error| self.terminal_error = Some(error.clone()))?
                    {
                        Some(signal) => {
                            for (camera, slot) in state.shared.cameras.iter().zip(signal) {
                                let _ = camera.slot_free_tx.send(slot);
                            }
                        }
                        None => {
                            self.exhausted = true;
                            return Ok(i);
                        }
                    }
                }
                log::debug!("GPU zero-copy: skipped {count} frames for start_time");
                Ok(count)
            }
            SourceMode::Cpu(source) => {
                source.seek(count)?;
                log::debug!("CPU seek: jumped to frame {count}");
                Ok(count)
            }
            #[cfg(target_os = "macos")]
            SourceMode::MetalZeroCopy(_) => {
                for i in 0..count {
                    if self.next_frame()?.is_none() {
                        return Ok(i);
                    }
                }
                Ok(count)
            }
            #[cfg(target_os = "windows")]
            SourceMode::D3d11ZeroCopy(_) => {
                for i in 0..count {
                    if self.next_frame()?.is_none() {
                        return Ok(i);
                    }
                }
                Ok(count)
            }
        }
    }

    fn gpu_pixel_format(&self) -> GpuPixelFormat {
        self.pixel_format
    }

    fn is_full_range(&self) -> bool {
        self.full_range
    }

    fn left_rotation(&self) -> i32 {
        self.left_rotation
    }

    fn right_rotation(&self) -> i32 {
        self.right_rotation
    }

    fn is_exhausted(&self) -> bool {
        // CPU mode also honors the inner source's exhaustion state.
        // SmartFileSource exposes blocking reads, not polling.
        match &self.mode {
            SourceMode::Cpu(source) => self.exhausted || source.is_exhausted(),
            #[cfg(target_os = "linux")]
            SourceMode::GpuZeroCopy(_) => self.exhausted,
            #[cfg(target_os = "macos")]
            SourceMode::MetalZeroCopy(_) => self.exhausted,
            #[cfg(target_os = "windows")]
            SourceMode::D3d11ZeroCopy(_) => self.exhausted,
        }
    }

    fn start_decoding(&mut self) {
        #[cfg(target_os = "windows")]
        if let SourceMode::D3d11ZeroCopy(state) = &mut self.mode {
            state.ensure_running();
        }
    }
}

impl Drop for SmartFileSource {
    fn drop(&mut self) {
        // Windows joins through its owned stream. CUDA additionally borrows
        // allocations that must remain mapped until its workers finish.
        #[cfg(target_os = "linux")]
        if let SourceMode::GpuZeroCopy(state) = &mut self.mode {
            // Graceful shutdown ordering to prevent CUDA error 700:
            //
            // 1. Set shutdown flag -> decode threads exit via recv_timeout check
            // 2. Drop frame_rx -> pairing thread's send() returns Err, exits
            // 3. Join all threads -> VideoDecoder::Drop completes CUDA cleanup
            //    while shared CUDA VMM memory is still mapped
            // 4. SharedTextureSet drops naturally -> CUDA memory unmapped

            // Step 1: Signal shutdown to decode threads (they check this
            // flag via recv_timeout, so they exit even if slot senders
            // are still held by the session).
            state
                .shutdown
                .store(true, std::sync::atomic::Ordering::Release);

            // Step 2: Drop frame_rx to unblock pairing thread
            state.frame_rx = None;

            // Step 3: Join decode + pairing threads
            for handle in state.join_handles.drain(..) {
                let _ = handle.join();
            }
            // SharedTextureSet drops naturally after this
        }
    }
}

/// Check if a decode backend supports GPU zero-copy.
fn is_backend_zero_copy_capable(backend: crate::ffmpeg::decoder::DecodeBackend) -> bool {
    use crate::ffmpeg::decoder::DecodeBackend;
    match backend {
        #[cfg(target_os = "linux")]
        DecodeBackend::Cuda => true,
        #[cfg(target_os = "macos")]
        DecodeBackend::VideoToolbox => true,
        #[cfg(target_os = "windows")]
        DecodeBackend::D3d11va => true,
        _ => false,
    }
}
