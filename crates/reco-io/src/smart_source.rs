//! GPU-aware stereo file source with automatic backend selection.
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
//! let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
//! let mut source = SmartFileSource::open("left.mp4", "right.mp4", &gpu, 0)?;
//! // source implements FrameSource - pass to session.run()
//! ```

use std::path::Path;

use reco_core::renderer::GpuPixelFormat;
use reco_core::source::{FrameSource, SourceError, SourceInfo, StereoFrame};

/// GPU-aware stereo file source that auto-selects the optimal decode path.
///
/// Probes the input files at construction and selects the best available
/// backend: CUDA/Vulkan zero-copy on Linux/Windows, VideoToolbox/Metal
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
    left_rotation: i32,
    right_rotation: i32,
    /// Human-readable description of the active decode path.
    decode_mode: &'static str,
}

enum SourceMode {
    Cpu(crate::adapters::FfmpegFileSource),
    #[cfg(target_os = "linux")]
    GpuZeroCopy(Box<LinuxZeroCopyState>),
}

#[cfg(target_os = "linux")]
struct LinuxZeroCopyState {
    /// Paired frame signal receiver. Made `Option` so Drop can take it
    /// early to unblock the pairing thread before joining decode threads.
    frame_rx: Option<std::sync::mpsc::Receiver<reco_core::zero_copy::GpuFrameSignal>>,
    /// Shared textures (kept alive until Drop).
    shared: reco_core::session::SharedTextureSet,
    /// Decode thread join handles.
    join_handles: Vec<std::thread::JoinHandle<()>>,
    /// Shutdown flag checked by decode threads for graceful exit.
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl SmartFileSource {
    /// Open a stereo file source with automatic backend selection.
    ///
    /// Probes the left video for metadata (resolution, codec, rotation,
    /// pixel format) and the right video for rotation. Detects whether
    /// GPU zero-copy is available based on the decode backend and GPU
    /// capabilities.
    ///
    /// `sync_offset` applies temporal alignment between cameras:
    /// positive skips right frames, negative skips left frames.
    pub fn open(
        left: impl AsRef<Path>,
        right: impl AsRef<Path>,
        gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
    ) -> Result<Self, SourceError> {
        let left = left.as_ref();
        let right = right.as_ref();

        // Probe left video for metadata
        let probe =
            crate::ffmpeg::decoder::VideoDecoder::open(left).map_err(|e| SourceError::Init {
                path: left.display().to_string(),
                reason: format!("{e}"),
            })?;

        let fps_r = probe.frame_rate();
        let info = SourceInfo {
            width: probe.width(),
            height: probe.height(),
            fps: probe.fps(),
            fps_rational: Some((fps_r.0, fps_r.1)),
        };
        let pixel_format = probe.pixel_format();
        let left_rotation = probe.rotation();
        let decode_backend = probe.backend();
        drop(probe);

        // Probe right video for rotation
        let right_rotation = crate::ffmpeg::decoder::VideoDecoder::open(right)
            .map(|d| d.rotation())
            .unwrap_or(0);

        // Detect zero-copy capability
        let use_zero_copy = std::env::var("RECO_NO_HWACCEL").is_err()
            && is_backend_zero_copy_capable(decode_backend)
            && gpu.supports_zero_copy();

        if use_zero_copy {
            Self::open_zero_copy(
                left,
                right,
                gpu,
                sync_offset,
                info,
                pixel_format,
                left_rotation,
                right_rotation,
            )
        } else {
            Self::open_cpu(
                left,
                right,
                sync_offset,
                info,
                pixel_format,
                left_rotation,
                right_rotation,
            )
        }
    }

    /// Force CPU-only decode (for testing, benchmarking, or fallback).
    pub fn open_cpu_only(
        left: impl AsRef<Path>,
        right: impl AsRef<Path>,
        sync_offset: i64,
    ) -> Result<Self, SourceError> {
        let left = left.as_ref();
        let right = right.as_ref();

        let probe =
            crate::ffmpeg::decoder::VideoDecoder::open(left).map_err(|e| SourceError::Init {
                path: left.display().to_string(),
                reason: format!("{e}"),
            })?;
        let fps_r = probe.frame_rate();
        let info = SourceInfo {
            width: probe.width(),
            height: probe.height(),
            fps: probe.fps(),
            fps_rational: Some((fps_r.0, fps_r.1)),
        };
        let pixel_format = probe.pixel_format();
        let left_rotation = probe.rotation();
        drop(probe);

        let right_rotation = crate::ffmpeg::decoder::VideoDecoder::open(right)
            .map(|d| d.rotation())
            .unwrap_or(0);

        Self::open_cpu(
            left,
            right,
            sync_offset,
            info,
            pixel_format,
            left_rotation,
            right_rotation,
        )
    }

    fn open_cpu(
        left: &Path,
        right: &Path,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        left_rotation: i32,
        right_rotation: i32,
    ) -> Result<Self, SourceError> {
        let source = crate::adapters::FfmpegFileSource::open_with_offset(left, right, sync_offset)?;
        log::info!(
            "SmartFileSource: CPU decode ({}x{}, {pixel_format:?})",
            info.width,
            info.height
        );
        Ok(Self {
            mode: SourceMode::Cpu(source),
            info,
            pixel_format,
            left_rotation,
            right_rotation,
            decode_mode: "CPU upload",
        })
    }

    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        left: &Path,
        right: &Path,
        gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        left_rotation: i32,
        right_rotation: i32,
    ) -> Result<Self, SourceError> {
        use reco_core::session::SharedTextureSet;
        use reco_core::vulkan_interop::{Nv12Plane, create_nv12_shared_texture};
        use reco_core::zero_copy::GpuBufInfo;

        let map_err = |msg: String| SourceError::Init {
            path: left.display().to_string(),
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

        let (left_y_0, left_uv_0) = create_pair("left[0]")?;
        let (left_y_1, left_uv_1) = create_pair("left[1]")?;
        let (right_y_0, right_uv_0) = create_pair("right[0]")?;
        let (right_y_1, right_uv_1) = create_pair("right[1]")?;

        log::info!(
            "SmartFileSource: GPU zero-copy ({input_width}x{input_height}, {pixel_format:?}), Y pitch={}/{}",
            left_y_0.pitch,
            left_y_1.pitch
        );

        let left_buf = GpuBufInfo {
            y_ptr: [left_y_0.cuda_ptr, left_y_1.cuda_ptr],
            uv_ptr: [left_uv_0.cuda_ptr, left_uv_1.cuda_ptr],
            y_pitch: [left_y_0.pitch, left_y_1.pitch],
            uv_pitch: [left_uv_0.pitch, left_uv_1.pitch],
            width: input_width,
            height: input_height,
            pixel_format,
        };
        let right_buf = GpuBufInfo {
            y_ptr: [right_y_0.cuda_ptr, right_y_1.cuda_ptr],
            uv_ptr: [right_uv_0.cuda_ptr, right_uv_1.cuda_ptr],
            y_pitch: [right_y_0.pitch, right_y_1.pitch],
            uv_pitch: [right_uv_0.pitch, right_uv_1.pitch],
            width: input_width,
            height: input_height,
            pixel_format,
        };

        // Slot-free channels for backpressure
        let (left_slot_free_tx, left_slot_free_rx) = std::sync::mpsc::sync_channel::<u8>(2);
        let (right_slot_free_tx, right_slot_free_rx) = std::sync::mpsc::sync_channel::<u8>(2);
        left_slot_free_tx.send(0).expect("seed slot");
        left_slot_free_tx.send(1).expect("seed slot");
        right_slot_free_tx.send(0).expect("seed slot");
        right_slot_free_tx.send(1).expect("seed slot");

        // Shutdown flag for graceful decode thread exit
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Spawn decode threads
        let decode_handles = crate::zero_copy::spawn_decode_threads_gpu(
            left.to_string_lossy().into_owned(),
            right.to_string_lossy().into_owned(),
            left_buf.clone(),
            right_buf.clone(),
            left_slot_free_rx,
            right_slot_free_rx,
            sync_offset,
            shutdown.clone(),
        );

        // Build bind groups placeholder - session will create them from texture refs
        // For now, we need a dummy bind group. Actually, the bind groups need the
        // pipeline which we don't have. The session will create them via
        // configure_gpu_source() when run() starts.
        //
        // Store textures in SharedTextureSet without bind groups.
        // We'll need to split SharedTextureSet or use a different approach.
        // For now, let's store the textures and bufs directly.

        let shared = SharedTextureSet {
            textures: [
                left_y_0, left_uv_0, left_y_1, left_uv_1, right_y_0, right_uv_0, right_y_1,
                right_uv_1,
            ],
            left_buf,
            right_buf,
            left_slot_free_tx,
            right_slot_free_tx,
            left_slot_free_rx: None,
            right_slot_free_rx: None,
            // Bind groups are created lazily by the session's run_zero_copy_linux()
            // when it sees None. The source doesn't have pipeline access.
            bind_groups: None,
        };

        Ok(Self {
            mode: SourceMode::GpuZeroCopy(Box::new(LinuxZeroCopyState {
                frame_rx: Some(decode_handles.frame_rx),
                shared,
                join_handles: decode_handles.join_handles,
                shutdown,
            })),
            info,
            pixel_format,
            left_rotation,
            right_rotation,
            decode_mode: "GPU zero-copy (CUDA/Vulkan)",
        })
    }

    #[cfg(not(target_os = "linux"))]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        left: &Path,
        right: &Path,
        _gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        left_rotation: i32,
        right_rotation: i32,
    ) -> Result<Self, SourceError> {
        // TODO: macOS zero-copy via VideoToolbox/Metal
        // For now, fall back to CPU
        log::info!("SmartFileSource: zero-copy not yet implemented for this platform, using CPU");
        Self::open_cpu(
            left,
            right,
            sync_offset,
            info,
            pixel_format,
            left_rotation,
            right_rotation,
        )
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
    pub fn shared_texture_set(&self) -> Option<&reco_core::session::SharedTextureSet> {
        match &self.mode {
            SourceMode::GpuZeroCopy(state) => Some(&state.shared),
            _ => None,
        }
    }

    /// Take the slot-free senders for backpressure (GPU zero-copy only).
    ///
    /// The session needs these to release slots back to the decode threads
    /// after rendering each frame. Returns `None` for CPU-mode sources.
    ///
    /// **Important:** The returned senders are clones. The caller MUST drop
    /// them before this source is dropped, otherwise the `Drop` impl cannot
    /// fully signal decode threads to shut down (the cloned senders keep the
    /// channel alive). In practice, the session drops senders when `run()`
    /// returns, which happens before the source is dropped.
    #[cfg(target_os = "linux")]
    pub fn take_slot_senders(
        &mut self,
    ) -> Option<(
        std::sync::mpsc::SyncSender<u8>,
        std::sync::mpsc::SyncSender<u8>,
    )> {
        match &mut self.mode {
            SourceMode::GpuZeroCopy(state) => {
                // Clone senders (they're clonable) for the session
                Some((
                    state.shared.left_slot_free_tx.clone(),
                    state.shared.right_slot_free_tx.clone(),
                ))
            }
            _ => None,
        }
    }

    /// Access the CUDA buffer info for GPU detection (GPU zero-copy only).
    #[cfg(target_os = "linux")]
    pub fn gpu_buf_info(
        &self,
    ) -> Option<(
        &reco_core::zero_copy::GpuBufInfo,
        &reco_core::zero_copy::GpuBufInfo,
    )> {
        match &self.mode {
            SourceMode::GpuZeroCopy(state) => {
                Some((&state.shared.left_buf, &state.shared.right_buf))
            }
            _ => None,
        }
    }
}

impl FrameSource for SmartFileSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        match &mut self.mode {
            SourceMode::Cpu(source) => source.next_frame(),
            #[cfg(target_os = "linux")]
            SourceMode::GpuZeroCopy(state) => {
                let rx = state
                    .frame_rx
                    .as_ref()
                    .expect("frame_rx taken during shutdown");
                match rx.recv() {
                    Ok(signal) => Ok(Some(StereoFrame::GpuResident {
                        left_slot: signal.left_slot,
                        right_slot: signal.right_slot,
                    })),
                    Err(_) => Ok(None),
                }
            }
        }
    }

    fn is_gpu_resident(&self) -> bool {
        !matches!(self.mode, SourceMode::Cpu(_))
    }

    fn gpu_pixel_format(&self) -> GpuPixelFormat {
        self.pixel_format
    }

    fn left_rotation(&self) -> i32 {
        self.left_rotation
    }

    fn right_rotation(&self) -> i32 {
        self.right_rotation
    }
}

impl Drop for SmartFileSource {
    fn drop(&mut self) {
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
            // are still alive from take_slot_senders).
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
        _ => false,
    }
}
