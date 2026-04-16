//! Linux CUDA/Vulkan zero-copy session methods.
//!
//! Separated from the main session module to isolate platform-specific
//! shared-texture orchestration (CUDA VMM, Vulkan external memory).

use super::{FrameProgress, ProgressCallback, SessionError, StitchSession};
use crate::vulkan_interop::{Nv12Plane, SharedTexture, create_nv12_shared_texture};
use crate::zero_copy::GpuBufInfo;

/// Bundled shared textures, CUDA buffer info, slot channels, and bind
/// groups for the Linux CUDA/Vulkan zero-copy path.
///
/// Created by [`StitchSession::create_shared_textures`], consumed by
/// [`StitchSession::run_zero_copy_linux`]. The caller must pass
/// `left_buf` / `right_buf` and the slot-free receivers to the decode
/// thread spawner, then pass this struct (minus the receivers) to the
/// session.
pub struct SharedTextureSet {
    /// The 8 shared textures: [left_y_0, left_uv_0, left_y_1, left_uv_1,
    /// right_y_0, right_uv_0, right_y_1, right_uv_1].
    /// Must be dropped after decode threads are joined.
    pub textures: [SharedTexture; 8],
    /// CUDA buffer info for left camera decode thread.
    pub left_buf: GpuBufInfo,
    /// CUDA buffer info for right camera decode thread.
    pub right_buf: GpuBufInfo,
    /// Slot-free sender for left camera (decode backpressure).
    pub left_slot_free_tx: std::sync::mpsc::SyncSender<u8>,
    /// Slot-free sender for right camera (decode backpressure).
    pub right_slot_free_tx: std::sync::mpsc::SyncSender<u8>,
    /// Slot-free receiver for left camera. Taken by decode thread spawner.
    pub left_slot_free_rx: Option<std::sync::mpsc::Receiver<u8>>,
    /// Slot-free receiver for right camera. Taken by decode thread spawner.
    pub right_slot_free_rx: Option<std::sync::mpsc::Receiver<u8>>,
    /// Pre-built bind groups for the shared textures.
    /// `None` when the source creates textures without pipeline access
    /// (e.g. `SmartFileSource`). The session creates them lazily at
    /// the start of `run()`.
    pub bind_groups: Option<crate::pipeline::GpuSourceBindGroups>,
}

impl StitchSession {
    /// Create double-buffered shared textures for CUDA/Vulkan zero-copy.
    ///
    /// Returns 8 shared textures (Y + UV per slot per camera), the
    /// `GpuBufInfo` for each camera (CUDA pointers for decode threads),
    /// and slot-free channels for backpressure.
    ///
    /// The `pixel_format` selects texture formats: [`GpuPixelFormat::Nv12`](crate::renderer::GpuPixelFormat::Nv12)
    /// uses R8Unorm/Rg8Unorm, [`GpuPixelFormat::P010`](crate::renderer::GpuPixelFormat::P010) uses R16Unorm/Rg16Unorm.
    ///
    /// Call this once during setup, then pass the results to
    /// [`Self::run_zero_copy_linux`].
    pub fn create_shared_textures(
        &mut self,
        input_width: u32,
        input_height: u32,
        pixel_format: crate::renderer::GpuPixelFormat,
    ) -> Result<SharedTextureSet, SessionError> {
        log::info!("Creating shared textures for zero-copy ({pixel_format:?})...");

        let gpu = self.pipeline.gpu();
        let create_pair =
            |label: &str, slot: usize| -> Result<(SharedTexture, SharedTexture), SessionError> {
                let y = create_nv12_shared_texture(
                    gpu,
                    input_width,
                    input_height,
                    Nv12Plane::Y,
                    pixel_format,
                )
                .map_err(|e| {
                    SessionError::ZeroCopy(format!("{label} Y[{slot}] shared texture: {e}"))
                })?;
                let uv = create_nv12_shared_texture(
                    gpu,
                    input_width,
                    input_height,
                    Nv12Plane::Uv,
                    pixel_format,
                )
                .map_err(|e| {
                    SessionError::ZeroCopy(format!("{label} UV[{slot}] shared texture: {e}"))
                })?;
                Ok((y, uv))
            };

        // NVIDIA driver workaround: the first CUDA VMM -> Vulkan external
        // memory import for 16-bit formats (R16Unorm, Rg16Unorm) produces
        // textures whose CUDA writes are invisible to Vulkan reads. Creating
        // (and keeping alive) dummy textures for each format first "primes"
        // the driver's internal state so subsequent imports work correctly.
        // Only needed for P010 (10-bit). Both Y (R16Unorm) and UV (Rg16Unorm)
        // formats need separate warm-ups.
        // Tracked in: https://github.com/reco-project/video-stitcher/issues/134
        let _warmup = if pixel_format == crate::renderer::GpuPixelFormat::P010 {
            let y = create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Y, pixel_format)
                .map_err(|e| SessionError::ZeroCopy(format!("warmup Y texture: {e}")))?;
            let uv = create_nv12_shared_texture(gpu, 16, 16, Nv12Plane::Uv, pixel_format)
                .map_err(|e| SessionError::ZeroCopy(format!("warmup UV texture: {e}")))?;
            Some((y, uv))
        } else {
            None
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
            bind_groups: Some(bind_groups),
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

        // Destructure to control drop ordering precisely.
        let SharedTextureSet {
            textures,
            left_buf,
            right_buf,
            bind_groups: existing_bind_groups,
            left_slot_free_tx,
            right_slot_free_tx,
            ..
        } = shared;

        // Create bind groups lazily if not provided (SmartFileSource path).
        let bind_groups = match existing_bind_groups {
            Some(bg) => bg,
            None => {
                let t = &textures;
                self.pipeline.configure_gpu_source(
                    [(&t[0], &t[1]), (&t[2], &t[3])],
                    [(&t[4], &t[5]), (&t[6], &t[7])],
                )
            }
        };

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

            // CUDA-Vulkan sync (#103): the decode thread called
            // cuCtxSynchronize() before sending this signal, draining
            // all CUDA work. Correct but serializes decode and render
            // on the CPU timeline (no decode/render overlap).
            //
            // The proper fix is VK_KHR_external_semaphore timeline
            // semaphores so CUDA and Vulkan synchronize on the GPU.
            // wgpu has signal support (gfx-rs/wgpu#6813) but wait
            // support is pending (gfx-rs/wgpu#8996, blocked on the
            // multi-queue RFC gfx-rs/wgpu#8844).
            self.detect_and_update_director_gpu(
                &left_buf,
                &right_buf,
                signal.left_slot,
                signal.right_slot,
                start.elapsed(),
            )?;
            let pos = self.director_position();
            let render_buf = self.pipeline.render_gpu_frame(
                &bind_groups,
                signal.left_slot,
                signal.right_slot,
                pos.yaw,
                pos.pitch,
            );
            self.submit_render_output(render_buf)?;

            // Release slots for decode thread reuse.
            //
            // SAFETY CAVEAT (#136): The render pass that reads these shared
            // textures has been submitted to the GPU queue via queue.submit()
            // (inside submit_render_output -> convert_and_readback), but may
            // not have completed yet. The decode thread calls
            // cuCtxSynchronize() before its next write, which drains the
            // CUDA queue but does NOT wait for Vulkan/wgpu work.
            //
            // In practice this is safe because:
            // (a) the decode thread blocks on slot_free_rx.recv() here,
            // (b) then decodes a frame (takes >0.5ms), giving the GPU time,
            // (c) then calls cuCtxSynchronize before the CUDA memcpy.
            //
            // A proper fix requires VK_KHR_external_semaphore wait support
            // in wgpu (gfx-rs/wgpu#8996). Until then, if corrupted frames
            // appear on very fast GPUs, add gpu.device.poll(Wait) here.
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
}
