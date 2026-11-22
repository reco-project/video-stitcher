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

#[cfg(feature = "gpu")]
use reco_core::render::renderer::GpuPixelFormat;
#[cfg(all(
    feature = "gpu",
    any(target_os = "linux", target_os = "windows", target_os = "macos")
))]
use reco_core::source::FrameSet;
use reco_core::source::{FrameSource, SourceError, SourceFrame, SourceInfo};

#[cfg(all(test, feature = "gpu", any(target_os = "windows", target_os = "macos")))]
mod native_tests;

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
    #[cfg(feature = "gpu")]
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
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    GpuZeroCopy(Box<LinuxZeroCopyState>),
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    MetalZeroCopy(
        crate::decode_stream::OwnedFrames<reco_core::interop::metal::RetainedCVPixelBuffer>,
    ),
    #[cfg(all(feature = "gpu", target_os = "windows"))]
    D3d11ZeroCopy(Box<WindowsZeroCopyState>),
}

#[cfg(all(feature = "gpu", target_os = "windows"))]
enum WindowsDecodeState {
    Pending {
        inputs: Vec<crate::stitch_job::InputPath>,
        sync_offset: i64,
    },
    Running {
        frames: crate::decode_stream::OwnedFrames<crate::ffmpeg::decoder::D3d11Frame>,
    },
    Failed(SourceError),
}

#[cfg(all(feature = "gpu", target_os = "windows"))]
struct WindowsZeroCopyState {
    decode: WindowsDecodeState,
}

#[cfg(all(feature = "gpu", target_os = "windows"))]
impl WindowsZeroCopyState {
    fn ensure_running(&mut self) {
        if let WindowsDecodeState::Pending {
            inputs,
            sync_offset,
        } = &self.decode
        {
            let inputs = inputs.clone();
            let sync_offset = *sync_offset;
            self.decode = match crate::zero_copy::spawn_d3d11_decode_frames(&inputs, sync_offset) {
                Ok(frames) => WindowsDecodeState::Running { frames },
                Err(error) => WindowsDecodeState::Failed(error),
            };
        }
    }
}

#[cfg(all(feature = "gpu", target_os = "linux"))]
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
    #[cfg(feature = "gpu")]
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
        let mut info = SourceInfo {
            camera_sizes: vec![(probe.width(), probe.height())],
            fps: probe.fps().unwrap_or(0.0),
            fps_rational: fps_r.map(|r| (r.0, r.1)),
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
            crate::adapters::merge_frame_rate(&mut info, probe.frame_rate(), input.first_path())?;
            matching_layout &= (cfg!(target_os = "linux")
                || (probe.width() == info.camera_sizes[0].0
                    && probe.height() == info.camera_sizes[0].1))
                && probe.pixel_format() == pixel_format
                && is_backend_zero_copy_capable(probe.backend());
            info.camera_sizes.push((probe.width(), probe.height()));
            if camera == 1 {
                right_rotation = probe.rotation();
            }
        }
        // Automatic Windows/Metal native selection is restricted to validated layouts.
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
            info.total_frames = crate::adapters::input_frame_count(inputs, sync_offset);
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
            Self::open_cpu_only(inputs, sync_offset, false)
        }
    }

    /// Open host-backed YUV420P frames for any number of inputs in projection order. Hardware decoding can still be used unless `software_decode` is true. Format, fps and color range are probed from input 0; rotation is probed per input.
    pub fn open_cpu_only(
        inputs: &[crate::stitch_job::InputPath],
        sync_offset: i64,
        software_decode: bool,
    ) -> Result<Self, SourceError> {
        let source =
            crate::adapters::FfmpegFileSource::open_inputs(inputs, sync_offset, software_decode)?;
        let info = source.info();
        #[cfg(feature = "gpu")]
        let pixel_format = source.pixel_format();
        let full_range = source.is_full_range();
        log::info!(
            "SmartFileSource: host-backed frames ({} camera(s), {}x{}{})",
            inputs.len(),
            info.camera_sizes[0].0,
            info.camera_sizes[0].1,
            if full_range { ", full-range" } else { "" }
        );
        let left_rotation = source.rotation(0);
        let right_rotation = source.rotation(1);
        Ok(Self {
            mode: SourceMode::Cpu(Box::new(source)),
            info,
            #[cfg(feature = "gpu")]
            pixel_format,
            full_range,
            left_rotation,
            right_rotation,
            decode_mode: "Host YUV420P",
            exhausted: false,
            terminal_error: None,
        })
    }

    #[cfg(all(feature = "gpu", target_os = "linux"))]
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
        let (shared, receivers) =
            crate::zero_copy::cuda::allocate(gpu, &info.camera_sizes, pixel_format).map_err(
                |error| SourceError::Init {
                    path: inputs[0].first_path().display().to_string(),
                    reason: error.to_string(),
                },
            )?;
        let decoders = inputs
            .iter()
            .zip(&shared.cameras)
            .zip(receivers)
            .map(|((input, camera), free)| (input.clone(), camera.buf.clone(), free))
            .collect();
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let decode_handles =
            crate::zero_copy::spawn_decode_threads_gpu(decoders, sync_offset, shutdown.clone());
        log::info!(
            "SmartFileSource: GPU zero-copy ({:?}, {pixel_format:?})",
            info.camera_sizes
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

    #[cfg(all(feature = "gpu", target_os = "macos"))]
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
        let frames = crate::zero_copy::spawn_vt_decode_frames(inputs, sync_offset);

        log::info!(
            "SmartFileSource: Metal zero-copy ({}x{}, VideoToolbox decode{})",
            info.camera_sizes[0].0,
            info.camera_sizes[0].1,
            if full_range { ", full-range" } else { "" }
        );

        Ok(Self {
            mode: SourceMode::MetalZeroCopy(frames),
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

    #[cfg(all(feature = "gpu", target_os = "windows"))]
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
                    inputs: inputs.to_vec(),
                    sync_offset,
                },
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

    #[cfg(feature = "gpu")]
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        inputs: &[crate::stitch_job::InputPath],
        _gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        _info: SourceInfo,
        _pixel_format: GpuPixelFormat,
        _full_range: bool,
        _left_rotation: i32,
        _right_rotation: i32,
    ) -> Result<Self, SourceError> {
        log::info!("SmartFileSource: zero-copy not yet implemented for this platform, using CPU");
        Self::open_cpu_only(inputs, sync_offset, false)
    }

    /// Human-readable description of the active decode path.
    pub fn decode_mode(&self) -> &'static str {
        self.decode_mode
    }

    /// Access the shared texture set (GPU zero-copy only).
    ///
    /// The session uses this to create bind groups at the start of `run()`.
    /// Returns `None` for CPU-mode sources.
    #[cfg(all(feature = "gpu", target_os = "linux"))]
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

    fn total_frames(&self) -> Option<u64> {
        self.info.total_frames
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
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
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            SourceMode::GpuZeroCopy(state) => {
                let rx = state
                    .frame_rx
                    .as_ref()
                    .expect("frame_rx taken during shutdown");
                match crate::decode_stream::receive(rx)
                    .inspect_err(|error| self.terminal_error = Some(error.clone()))?
                {
                    Some(signal) => Ok(Some(SourceFrame::untimed(FrameSet::GpuResident {
                        slots: signal,
                    }))),
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                }
            }
            #[cfg(all(feature = "gpu", target_os = "macos"))]
            SourceMode::MetalZeroCopy(frames) => match frames
                .receive()
                .inspect_err(|error| self.terminal_error = Some(error.clone()))?
            {
                Some(frames) => Ok(Some(SourceFrame::untimed(FrameSet::MetalResident(frames)))),
                None => {
                    self.exhausted = true;
                    Ok(None)
                }
            },
            #[cfg(all(feature = "gpu", target_os = "windows"))]
            SourceMode::D3d11ZeroCopy(state) => {
                state.ensure_running();
                let result = match &state.decode {
                    WindowsDecodeState::Running { frames } => frames.receive(),
                    WindowsDecodeState::Failed(error) => Err(error.clone()),
                    _ => unreachable!(),
                };
                match result.inspect_err(|error| self.terminal_error = Some(error.clone()))? {
                    Some(decoded) => {
                        let frames = FrameSet::D3d11Resident(
                            decoded
                                .into_iter()
                                .map(|frame| {
                                    // SAFETY: the decoded AVFrame owns the texture and pins this slice.
                                    unsafe {
                                        reco_core::source::D3d11CameraFrame::from_raw(
                                            frame.texture,
                                            frame.array_slice,
                                            frame,
                                        )
                                    }
                                })
                                .collect(),
                        );
                        Ok(Some(SourceFrame::untimed(frames)))
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
            #[cfg(all(feature = "gpu", target_os = "linux"))]
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
            #[cfg(all(feature = "gpu", target_os = "macos"))]
            SourceMode::MetalZeroCopy(_) => {
                for i in 0..count {
                    if self.next_frame()?.is_none() {
                        return Ok(i);
                    }
                }
                Ok(count)
            }
            #[cfg(all(feature = "gpu", target_os = "windows"))]
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

    #[cfg(feature = "gpu")]
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
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            SourceMode::GpuZeroCopy(_) => self.exhausted,
            #[cfg(all(feature = "gpu", target_os = "macos"))]
            SourceMode::MetalZeroCopy(_) => self.exhausted,
            #[cfg(all(feature = "gpu", target_os = "windows"))]
            SourceMode::D3d11ZeroCopy(_) => self.exhausted,
        }
    }

    fn start_decoding(&mut self) {
        #[cfg(all(feature = "gpu", target_os = "windows"))]
        if let SourceMode::D3d11ZeroCopy(state) = &mut self.mode {
            state.ensure_running();
        }
    }
}

impl Drop for SmartFileSource {
    fn drop(&mut self) {
        // Windows joins through its owned stream. CUDA additionally borrows
        // allocations that must remain mapped until its workers finish.
        #[cfg(all(feature = "gpu", target_os = "linux"))]
        if let SourceMode::GpuZeroCopy(state) = &mut self.mode {
            // Graceful shutdown ordering to prevent CUDA error 700:
            //
            // 1. Set shutdown flag -> decode threads exit via recv_timeout check
            // 2. Drop frame_rx -> pairing thread's send() returns Err, exits
            // 3. Join all threads -> VideoDecoder::Drop completes CUDA cleanup
            //    while shared Vulkan memory is still mapped into CUDA
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
#[cfg(feature = "gpu")]
fn is_backend_zero_copy_capable(backend: crate::ffmpeg::decoder::DecodeBackend) -> bool {
    match backend {
        #[cfg(target_os = "linux")]
        crate::ffmpeg::decoder::DecodeBackend::Cuda => true,
        #[cfg(target_os = "macos")]
        crate::ffmpeg::decoder::DecodeBackend::VideoToolbox => true,
        #[cfg(target_os = "windows")]
        crate::ffmpeg::decoder::DecodeBackend::D3d11va => true,
        _ => false,
    }
}
