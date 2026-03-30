//! High-level stitching session.
//!
//! [`StitchSession`] bundles the GPU pipeline with the NV12 converter,
//! providing a single entry point for rendering and encoding stitched
//! panoramic frames. This keeps encode orchestration inside `reco-core`
//! so that every consumer (CLI, GUI, OBS plugin, cloud worker) gets the
//! same optimized frame loop without duplicating pipeline plumbing.
//!
//! ## Two-level API
//!
//! - [`StitchSession::process_frame`] - render one frame and submit it
//!   to an encoder. Use this for interactive/GUI applications or when
//!   the caller controls the frame loop (e.g. zero-copy GPU decode).
//!
//! - [`StitchSession::run`] - batch-process an entire [`FrameSource`]
//!   into an encoder, with optional progress reporting and interrupt
//!   support. Use this for CLI batch encoding.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::async_encode::AsyncEncodeThread;
use crate::calibration::MatchCalibration;
use crate::encoder::{EncodeError, Encoder};
use crate::gpu::{GpuContext, GpuError, OutputFormat};
use crate::nv12_converter::{Nv12Converter, Nv12Error};
use crate::pipeline::{PipelineError, StitchPipeline};
use crate::renderer::InputFormat;
use crate::source::{FrameSource, SourceError, StereoFrame};
use crate::viewport::ViewportConfig;

use thiserror::Error;

/// Configuration for creating a [`StitchSession`].
pub struct SessionConfig {
    /// Camera calibration data.
    pub calibration: MatchCalibration,
    /// Output viewport (dimensions, blend width, FOV).
    pub viewport: ViewportConfig,
    /// Input frame width in pixels.
    pub input_width: u32,
    /// Input frame height in pixels.
    pub input_height: u32,
    /// GPU render target format (typically [`OutputFormat::Rgba8Unorm`] for encoding).
    pub output_format: OutputFormat,
    /// Input pixel format (YUV420P or NV12).
    pub input_format: InputFormat,
}

/// Progress information passed to the progress callback.
#[derive(Debug, Clone)]
pub struct FrameProgress {
    /// Number of frames processed so far.
    pub frames_completed: u64,
    /// Elapsed wall-clock time since the run started.
    pub elapsed: std::time::Duration,
}

/// Callback for progress reporting during [`StitchSession::run`].
pub type ProgressCallback = Box<dyn FnMut(&FrameProgress) + Send>;

/// Errors from [`StitchSession`].
#[derive(Debug, Error)]
pub enum SessionError {
    /// GPU initialization error.
    #[error("GPU: {0}")]
    Gpu(#[from] GpuError),

    /// GPU pipeline error.
    #[error("pipeline: {0}")]
    Pipeline(#[from] PipelineError),

    /// NV12 conversion error.
    #[error("NV12 converter: {0}")]
    Nv12(#[from] Nv12Error),

    /// Encoder error.
    #[error("encoder: {0}")]
    Encode(#[from] EncodeError),

    /// Source error.
    #[error("source: {0}")]
    Source(#[from] SourceError),

    /// Zero-copy setup or runtime error.
    #[error("zero-copy: {0}")]
    ZeroCopy(String),
}

/// Bundled shared textures, CUDA buffer info, slot channels, and bind
/// groups for the Linux CUDA/Vulkan zero-copy path.
///
/// Created by [`StitchSession::create_shared_textures`], consumed by
/// [`StitchSession::run_zero_copy_linux`]. The caller must pass
/// `left_buf` / `right_buf` and the slot-free receivers to the decode
/// thread spawner, then pass this struct (minus the receivers) to the
/// session.
#[cfg(target_os = "linux")]
pub struct SharedTextureSet {
    /// The 8 shared textures: [left_y_0, left_uv_0, left_y_1, left_uv_1,
    /// right_y_0, right_uv_0, right_y_1, right_uv_1].
    /// Must be dropped after decode threads are joined.
    pub textures: [crate::vulkan_interop::SharedTexture; 8],
    /// CUDA buffer info for left camera decode thread.
    pub left_buf: crate::zero_copy::GpuBufInfo,
    /// CUDA buffer info for right camera decode thread.
    pub right_buf: crate::zero_copy::GpuBufInfo,
    /// Slot-free sender for left camera (decode backpressure).
    pub left_slot_free_tx: std::sync::mpsc::SyncSender<u8>,
    /// Slot-free sender for right camera (decode backpressure).
    pub right_slot_free_tx: std::sync::mpsc::SyncSender<u8>,
    /// Slot-free receiver for left camera. Taken by decode thread spawner.
    pub left_slot_free_rx: Option<std::sync::mpsc::Receiver<u8>>,
    /// Slot-free receiver for right camera. Taken by decode thread spawner.
    pub right_slot_free_rx: Option<std::sync::mpsc::Receiver<u8>>,
    /// Pre-built bind groups for the shared textures.
    pub bind_groups: crate::pipeline::GpuSourceBindGroups,
}

/// A high-level stitching session that owns the GPU pipeline, NV12
/// converter, and optionally an async encoder.
///
/// Created once per encoding job or application lifetime. Call
/// [`set_encoder`](Self::set_encoder) to attach an encoder before
/// rendering, then use [`submit_render_output`](Self::submit_render_output)
/// for per-frame control or [`run`](Self::run) for batch processing.
/// Call [`finish`](Self::finish) to flush the last frame and finalize
/// encoding.
pub struct StitchSession {
    pipeline: StitchPipeline,
    nv12_converter: Nv12Converter,
    encoder: Option<AsyncEncodeThread>,
    frame_count: u64,
}

impl StitchSession {
    /// Create a new session, initializing the GPU automatically.
    pub async fn new(config: SessionConfig) -> Result<Self, SessionError> {
        let gpu = GpuContext::new().await?;
        Self::with_gpu(gpu, config)
    }

    /// Create a session with an existing GPU context.
    ///
    /// Use this when the caller needs to control GPU selection (e.g.
    /// for zero-copy decode where the GPU must match the CUDA device).
    pub fn with_gpu(gpu: GpuContext, config: SessionConfig) -> Result<Self, SessionError> {
        let output_width = config.viewport.width;
        let output_height = config.viewport.height;

        let pipeline = StitchPipeline::with_gpu(
            gpu,
            config.calibration,
            config.viewport,
            config.input_width,
            config.input_height,
            config.output_format,
            config.input_format,
        )?;

        let nv12_converter = Nv12Converter::new(pipeline.gpu(), output_width, output_height)?;

        Ok(Self {
            pipeline,
            nv12_converter,
            encoder: None,
            frame_count: 0,
        })
    }

    /// Attach an encoder to this session.
    ///
    /// The encoder is moved to a background thread for async encoding.
    /// `buffer_count` controls how many frames can be in-flight between
    /// the render thread and the encode thread (typically 2).
    ///
    /// Must be called before [`Self::submit_render_output`], [`Self::process_frame`],
    /// or [`Self::run`].
    pub fn set_encoder(&mut self, encoder: Box<dyn Encoder + Send>, buffer_count: usize) {
        let width = self.nv12_converter.width();
        let height = self.nv12_converter.height();
        self.encoder = Some(AsyncEncodeThread::new(encoder, width, height, buffer_count));
    }

    /// Render a single CPU-resident stereo frame and submit it to the encoder.
    ///
    /// Handles YUV420P and NV12 input formats. For GPU-resident frames
    /// (zero-copy path), use [`submit_render_output`](Self::submit_render_output)
    /// instead.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_process_frame")
    )]
    pub fn process_frame(
        &mut self,
        frame: &StereoFrame,
        yaw: f32,
        pitch: f32,
    ) -> Result<(), SessionError> {
        let render_buf = self.pipeline.render_stereo_frame(frame, yaw, pitch)?;
        self.submit_render_output(render_buf)
    }

    /// Render from GPU-resident textures and submit to the async encoder.
    ///
    /// Used with the zero-copy path where decode threads write directly
    /// to shared GPU textures. The caller must configure bind groups via
    /// [`pipeline_mut()`](Self::pipeline_mut) and call
    /// [`StitchPipeline::render_gpu_frame`] to get the command buffer,
    /// then pass it here.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_submit_render")
    )]
    pub fn submit_render_output(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<(), SessionError> {
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_commands,
        )?;

        // First call returns None (GPU work submitted, no previous frame yet).
        // From the second call onward, we get the previous frame's data.
        if let Some(data) = nv12_data
            && let Some(ref encoder) = self.encoder
        {
            encoder.submit(data)?;
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Batch-process frames from a source into the encoder.
    ///
    /// Runs the full decode-render-encode loop until the source is
    /// exhausted, the frame limit is reached, or the interrupt flag
    /// is set. Returns the number of frames processed.
    ///
    /// Does NOT call [`Self::finish`] - the caller must do that after this
    /// returns to flush the last frame and finalize encoding.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_run")
    )]
    pub fn run(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        mut on_progress: Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        let start = std::time::Instant::now();
        let yaw = 0.0_f32;
        let pitch = 0.0_f32;

        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let frame = {
                crate::profile_scope!("wait_decode");
                match source.next_frame()? {
                    Some(f) => f,
                    None => break,
                }
            };

            self.process_frame(&frame, yaw, pitch)?;

            if let Some(ref mut cb) = on_progress {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        Ok(self.frame_count)
    }

    // ---- Zero-copy paths ----

    /// Create double-buffered shared textures for CUDA/Vulkan zero-copy.
    ///
    /// Returns 8 shared textures (Y + UV per slot per camera), the
    /// `GpuBufInfo` for each camera (CUDA pointers for decode threads),
    /// and slot-free channels for backpressure.
    ///
    /// Call this once during setup, then pass the results to
    /// [`Self::run_zero_copy_linux`].
    #[cfg(target_os = "linux")]
    pub fn create_shared_textures(
        &mut self,
        input_width: u32,
        input_height: u32,
    ) -> Result<SharedTextureSet, SessionError> {
        use crate::vulkan_interop::{Nv12Plane, create_nv12_shared_texture};

        log::info!("Creating shared textures for zero-copy...");

        let gpu = self.pipeline.gpu();
        let create_pair = |label: &str,
                           slot: usize|
         -> Result<
            (
                crate::vulkan_interop::SharedTexture,
                crate::vulkan_interop::SharedTexture,
            ),
            SessionError,
        > {
            let y = create_nv12_shared_texture(gpu, input_width, input_height, Nv12Plane::Y)
                .map_err(|e| {
                    SessionError::ZeroCopy(format!("{label} Y[{slot}] shared texture: {e}"))
                })?;
            let uv = create_nv12_shared_texture(gpu, input_width, input_height, Nv12Plane::Uv)
                .map_err(|e| {
                    SessionError::ZeroCopy(format!("{label} UV[{slot}] shared texture: {e}"))
                })?;
            Ok((y, uv))
        };

        let (left_y_0, left_uv_0) = create_pair("left", 0)?;
        let (left_y_1, left_uv_1) = create_pair("left", 1)?;
        let (right_y_0, right_uv_0) = create_pair("right", 0)?;
        let (right_y_1, right_uv_1) = create_pair("right", 1)?;

        log::info!(
            "Shared textures created: left Y pitch={}/{}, UV pitch={}/{}",
            left_y_0.pitch,
            left_y_1.pitch,
            left_uv_0.pitch,
            left_uv_1.pitch
        );

        let left_buf = crate::zero_copy::GpuBufInfo {
            y_ptr: [left_y_0.cuda_ptr, left_y_1.cuda_ptr],
            uv_ptr: [left_uv_0.cuda_ptr, left_uv_1.cuda_ptr],
            y_pitch: [left_y_0.pitch, left_y_1.pitch],
            uv_pitch: [left_uv_0.pitch, left_uv_1.pitch],
            width: input_width,
            height: input_height,
        };
        let right_buf = crate::zero_copy::GpuBufInfo {
            y_ptr: [right_y_0.cuda_ptr, right_y_1.cuda_ptr],
            uv_ptr: [right_uv_0.cuda_ptr, right_uv_1.cuda_ptr],
            y_pitch: [right_y_0.pitch, right_y_1.pitch],
            uv_pitch: [right_uv_0.pitch, right_uv_1.pitch],
            width: input_width,
            height: input_height,
        };

        // Slot-free channels: decode threads wait for a slot to be released
        // before writing. Prevents NVDEC from overwriting a shared texture
        // that the GPU render pass is still reading.
        let (left_slot_free_tx, left_slot_free_rx) = std::sync::mpsc::sync_channel::<u8>(2);
        let (right_slot_free_tx, right_slot_free_rx) = std::sync::mpsc::sync_channel::<u8>(2);
        left_slot_free_tx.send(0).expect("seed slot channel");
        left_slot_free_tx.send(1).expect("seed slot channel");
        right_slot_free_tx.send(0).expect("seed slot channel");
        right_slot_free_tx.send(1).expect("seed slot channel");

        // Configure bind groups for GPU-resident shared textures
        let bind_groups = self.pipeline.configure_gpu_source(
            [(&left_y_0, &left_uv_0), (&left_y_1, &left_uv_1)],
            [(&right_y_0, &right_uv_0), (&right_y_1, &right_uv_1)],
        );

        Ok(SharedTextureSet {
            textures: [
                left_y_0, left_uv_0, left_y_1, left_uv_1, right_y_0, right_uv_0, right_y_1,
                right_uv_1,
            ],
            left_buf,
            right_buf,
            left_slot_free_tx,
            right_slot_free_tx,
            left_slot_free_rx: Some(left_slot_free_rx),
            right_slot_free_rx: Some(right_slot_free_rx),
            bind_groups,
        })
    }

    /// Run the zero-copy frame loop on Linux (CUDA/Vulkan).
    ///
    /// Receives decoded frame signals from `decode_handles`, renders
    /// using pre-built bind groups, and submits to the async encoder.
    /// Handles graceful shutdown ordering to prevent CUDA error 700.
    ///
    /// Returns the number of frames processed. The caller must call
    /// [`Self::finish`] after this returns.
    #[cfg(target_os = "linux")]
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_run_zero_copy_linux")
    )]
    pub fn run_zero_copy_linux(
        &mut self,
        shared: SharedTextureSet,
        decode_handles: crate::zero_copy::GpuDecodeHandles,
        frame_limit: u64,
        interrupted: &std::sync::atomic::AtomicBool,
        mut on_progress: Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        let start = std::time::Instant::now();
        let yaw = 0.0_f32;
        let pitch = 0.0_f32;

        // Destructure to control drop ordering precisely.
        let SharedTextureSet {
            textures,
            bind_groups,
            left_slot_free_tx,
            right_slot_free_tx,
            ..
        } = shared;

        let frame_rx = decode_handles.frame_rx;

        loop {
            if self.frame_count >= frame_limit
                || interrupted.load(std::sync::atomic::Ordering::Relaxed)
            {
                break;
            }

            let signal = {
                crate::profile_scope!("wait_decode");
                match frame_rx.recv() {
                    Ok(s) => s,
                    Err(_) => break,
                }
            };

            let render_buf = self.pipeline.render_gpu_frame(
                &bind_groups,
                signal.left_slot,
                signal.right_slot,
                yaw,
                pitch,
            );
            self.submit_render_output(render_buf)?;

            // GPU is done reading these slots - release for decode to reuse.
            let _ = left_slot_free_tx.send(signal.left_slot);
            let _ = right_slot_free_tx.send(signal.right_slot);

            // frame_count already incremented by submit_render_output()
            if let Some(ref mut cb) = on_progress {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        // Graceful shutdown: correct ordering prevents CUDA error 700.
        //
        // 1. Drop slot-free senders -> decode threads' recv() returns Err
        // 2. Drop frame_rx -> pairing thread's send() returns Err
        // 3. Join all threads -> VideoDecoder::Drop completes CUDA cleanup
        //    while shared CUDA VMM memory is still mapped
        // 4. Drop shared textures -> CUDA memory unmapped
        drop(left_slot_free_tx);
        drop(right_slot_free_tx);
        drop(frame_rx);
        for handle in decode_handles.join_handles {
            let _ = handle.join();
        }
        drop(bind_groups);
        drop(textures);

        Ok(self.frame_count)
    }

    /// Run the zero-copy frame loop on macOS (VideoToolbox/Metal).
    ///
    /// Receives retained CVPixelBuffer pairs from decode threads,
    /// imports them as Metal textures, renders, and submits to the
    /// async encoder.
    ///
    /// Returns the number of frames processed. The caller must call
    /// [`Self::finish`] after this returns.
    #[cfg(target_os = "macos")]
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_run_zero_copy_macos")
    )]
    pub fn run_zero_copy_macos(
        &mut self,
        pair_rx: std::sync::mpsc::Receiver<crate::zero_copy::VtFramePair>,
        frame_limit: u64,
        interrupted: &std::sync::atomic::AtomicBool,
        mut on_progress: Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        use crate::metal_interop::MetalTextureCache;

        let start = std::time::Instant::now();
        let cache = MetalTextureCache::new(self.pipeline.gpu())?;
        let yaw = 0.0_f32;
        let pitch = 0.0_f32;

        while !interrupted.load(std::sync::atomic::Ordering::Relaxed)
            && self.frame_count < frame_limit
        {
            let pair = match pair_rx.recv() {
                Ok(p) => p,
                Err(_) => break,
            };

            // Import NV12 planes as Metal textures (zero-copy via IOSurface).
            // SAFETY: RetainedCVPixelBuffer guarantees the pointer is valid.
            let (left_y, left_uv) =
                unsafe { cache.import_nv12(pair.left.as_ptr(), self.pipeline.gpu())? };
            let (right_y, right_uv) =
                unsafe { cache.import_nv12(pair.right.as_ptr(), self.pipeline.gpu())? };

            let render_buf = self.pipeline.render_imported_textures(
                &left_y.texture,
                &left_uv.texture,
                &right_y.texture,
                &right_uv.texture,
                yaw,
                pitch,
            );

            self.submit_render_output(render_buf)?;

            // frame_count already incremented by submit_render_output()
            if let Some(ref mut cb) = on_progress {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }

            if self.frame_count.is_multiple_of(60) {
                cache.flush();
            }
        }

        Ok(self.frame_count)
    }

    /// Flush the NV12 double-buffer and finalize the encoder.
    ///
    /// Submits the last pending frame from the double-buffer pipeline
    /// to the encoder, then shuts down the encode thread and calls
    /// [`Encoder::finish`]. Must be called after the frame loop ends.
    pub fn finish(&mut self) -> Result<(), SessionError> {
        // Flush the last frame from the NV12 double-buffer.
        if let Some(nv12_data) = self.nv12_converter.flush_pending(self.pipeline.gpu())? {
            if let Some(ref encoder) = self.encoder {
                encoder.submit(nv12_data)?;
            }
            self.frame_count += 1;
        }

        // Shut down the async encode thread.
        if let Some(mut encoder) = self.encoder.take() {
            encoder.finish()?;
        }

        Ok(())
    }

    /// Convert a pre-rendered frame to NV12 without encoding.
    ///
    /// Returns the previous frame's NV12 data (or `None` on first call).
    /// Used by the preview path where the caller displays frames directly
    /// instead of encoding them.
    #[cfg_attr(
        feature = "profiling",
        tracing::instrument(skip_all, name = "session_convert_nv12")
    )]
    pub fn convert_to_nv12(
        &mut self,
        render_commands: wgpu::CommandBuffer,
    ) -> Result<Option<&[u8]>, SessionError> {
        let nv12_data = self.nv12_converter.convert_and_readback(
            self.pipeline.gpu(),
            self.pipeline.render_target(),
            render_commands,
        )?;
        self.frame_count += 1;
        Ok(nv12_data)
    }

    /// Number of frames processed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Shared reference to the underlying pipeline.
    pub fn pipeline(&self) -> &StitchPipeline {
        &self.pipeline
    }

    /// Mutable reference to the underlying pipeline.
    ///
    /// Needed for zero-copy setup (configure_gpu_source) and viewport
    /// changes (resize, set_fov).
    pub fn pipeline_mut(&mut self) -> &mut StitchPipeline {
        &mut self.pipeline
    }

    /// Shared reference to the GPU context.
    pub fn gpu(&self) -> &GpuContext {
        self.pipeline.gpu()
    }

    /// The name of the GPU this session is running on.
    pub fn gpu_name(&self) -> &str {
        self.pipeline.gpu_name()
    }
}
