//! Batch processing entry points for [`StitchSession`].
//!
//! The `run` / `run_immediate` frame loop, source configuration, and
//! GPU zero-copy frame stepping live here.

use std::sync::atomic::{AtomicBool, Ordering};

use super::StitchSession;
use crate::session::types::{FrameProgress, ProgressCallback, SessionError};
use crate::source::{FrameSource, StereoFrame};

impl StitchSession {
    /// Auto-configure the session from source metadata.
    ///
    /// Called at the start of [`run`](Self::run). Applies rotation from
    /// the source's metadata.
    fn configure_from_source(&mut self, source: &dyn FrameSource) {
        // Apply rotation via shader UV flip ONLY for GPU-resident sources.
        // CPU sources handle rotation via buffer reversal in the decoder,
        // so applying the shader flip too would rotate 360 degrees (no-op but wrong).
        if source.is_gpu_resident() {
            let (lr, rr) = (source.left_rotation(), source.right_rotation());
            if lr == 180 || rr == 180 {
                self.core.pipeline_mut().set_flip_180(lr == 180, rr == 180);
                log::info!("Rotation: UV flip left={}, right={}", lr == 180, rr == 180);
            }
            // Store rotation for the GPU detector preprocessing path.
            // The detector needs to flip frames independently of the render shader.
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                self.left_rotation = lr;
                self.right_rotation = rr;
            }
        }
    }

    /// Configure the session for a GPU-resident source.
    ///
    /// Creates bind groups from the source's shared textures and stores
    /// slot-free senders for decode backpressure. Call this before
    /// [`run`](Self::run) when using a GPU-resident [`FrameSource`] like
    /// `SmartFileSource`.
    ///
    /// For the Layer 1 API (`run_zero_copy_linux`), this is handled
    /// internally and you don't need to call it.
    #[cfg(target_os = "linux")]
    pub fn setup_gpu_source(&mut self, shared: &super::SharedTextureSet) {
        let t = &shared.textures;
        let bind_groups = self.core.pipeline_mut().configure_gpu_source(
            [(&t[0], &t[1]), (&t[2], &t[3])],
            [(&t[4], &t[5]), (&t[6], &t[7])],
        );
        self.gpu_bind_groups = Some(bind_groups);
        self.gpu_slot_free_tx = Some((
            shared.left_slot_free_tx.clone(),
            shared.right_slot_free_tx.clone(),
        ));
        self.gpu_buf_info = Some((shared.left_buf.clone(), shared.right_buf.clone()));
        // Pre-build the 8 shared texture views for the GPU
        // stacked-replay pack shader. Same order as `t` above so
        // `step_gpu_with_bufs` can index per slot:
        //   left  y: [ls * 2],     uv: [ls * 2 + 1]
        //   right y: [4 + rs * 2], uv: [4 + rs * 2 + 1]
        // Views hold Arcs to the underlying textures, so the
        // SharedTextureSet still owns the lifetime.
        let desc = wgpu::TextureViewDescriptor::default();
        self.gpu_shared_views = Some([
            t[0].texture.create_view(&desc),
            t[1].texture.create_view(&desc),
            t[2].texture.create_view(&desc),
            t[3].texture.create_view(&desc),
            t[4].texture.create_view(&desc),
            t[5].texture.create_view(&desc),
            t[6].texture.create_view(&desc),
            t[7].texture.create_view(&desc),
        ]);
        log::info!("Session configured for GPU-resident source");
    }

    /// Batch-process frames from a source into the encoder.
    ///
    /// Runs the full decode-render-encode loop until the source is
    /// exhausted, the frame limit is reached, or the interrupt flag
    /// is set. Returns the number of frames processed.
    ///
    /// Automatically handles CPU-resident and GPU-resident frames:
    /// - CPU frames (Yuv420p, Nv12): uploaded to GPU, rendered, encoded
    /// - GPU frames (GpuResident): rendered directly from shared textures
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
        self.configure_from_source(source);

        let result = self.run_immediate(source, frame_limit, interrupted, &mut on_progress);

        // Drop GPU slot senders so decode threads can exit gracefully.
        // Without this, SmartFileSource::drop() deadlocks because the
        // session's cloned senders keep the decode threads' recv() alive.
        #[cfg(target_os = "linux")]
        {
            self.gpu_slot_free_tx = None;
        }

        result
    }

    /// Standard frame loop. Handles CPU-resident and GPU-resident
    /// frames transparently.
    fn run_immediate(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        let start = std::time::Instant::now();

        // Extract GPU buf info once before the loop to avoid per-frame clones.
        // Needed to satisfy the borrow checker (immutable borrow of buf_info
        // vs mutable borrow for detect_and_update_director_gpu).
        #[cfg(target_os = "linux")]
        let gpu_buf_info = self.gpu_buf_info.clone();

        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let frame_t0 = std::time::Instant::now();

            let frame = {
                crate::profile_scope!("wait_decode");
                match source.next_frame()? {
                    Some(f) => f,
                    None => break,
                }
            };
            let decode_time = frame_t0.elapsed();

            if let Some(sink) = self.event_sink.as_deref_mut() {
                sink.emit(crate::detect::pipeline_event::PipelineEvent::FrameStart {
                    frame_index: self.frame_count,
                    timestamp_ms: start.elapsed().as_secs_f64() * 1000.0,
                });
            }

            match &frame {
                #[cfg(target_os = "linux")]
                StereoFrame::GpuResident {
                    left_slot,
                    right_slot,
                } => {
                    self.step_gpu_with_bufs(
                        &gpu_buf_info,
                        *left_slot,
                        *right_slot,
                        start.elapsed(),
                    )?;
                    self.telemetry.record_frame(crate::telemetry::FrameTiming {
                        decode: Some(decode_time),
                        total: Some(frame_t0.elapsed()),
                        ..Default::default()
                    });
                }
                #[cfg(target_os = "windows")]
                StereoFrame::D3d11Resident {
                    left_texture,
                    left_slice,
                    right_texture,
                    right_slice,
                } => {
                    if self.d3d11_staging_pool.is_none() {
                        let (w, h) = self.core.pipeline().source_info();
                        match crate::interop::d3d11::D3d11StagingPool::new(self.core.gpu(), w, h) {
                            Ok(pool) => {
                                log::info!(
                                    "D3D11VA staging pool created: {}x{}, 4 NV12 slots",
                                    w,
                                    h
                                );
                                self.d3d11_staging_pool = Some(pool);
                            }
                            Err(e) => {
                                return Err(SessionError::ZeroCopy(format!(
                                    "D3D11 staging pool: {e}"
                                )));
                            }
                        }
                    }
                    let left_slot = self.frame_count as usize % 2;
                    let right_slot = left_slot + 2;

                    // Stage frames (borrows pool immutably, scoped).
                    {
                        let pool = self.d3d11_staging_pool.as_ref().unwrap();
                        pool.stage_frame(*left_texture, *left_slice, left_slot)?;
                        pool.stage_frame(*right_texture, *right_slice, right_slot)?;
                    }

                    // Director update (borrows self mutably).
                    self.update_director(start.elapsed())?;
                    let pos = self.director_position();

                    // Render from staged views (borrows pool immutably again).
                    let pool = self.d3d11_staging_pool.as_ref().unwrap();
                    let render_buf = self.core.render_imported_views_at_pose(
                        pool.y_view(left_slot),
                        pool.uv_view(left_slot),
                        pool.y_view(right_slot),
                        pool.uv_view(right_slot),
                        pos.yaw,
                        pos.pitch,
                    );
                    self.submit_render_output(render_buf)?;

                    self.telemetry.record_frame(crate::telemetry::FrameTiming {
                        decode: Some(decode_time),
                        total: Some(frame_t0.elapsed()),
                        ..Default::default()
                    });
                }
                _ => {
                    let detect_t0 = std::time::Instant::now();
                    self.detect_and_update_director(&frame, start.elapsed())?;
                    let detect_time = detect_t0.elapsed();

                    let render_t0 = std::time::Instant::now();
                    let pos = self.director_position();
                    self.process_frame(&frame, pos.yaw, pos.pitch)?;
                    let render_time = render_t0.elapsed();

                    self.telemetry.record_frame(crate::telemetry::FrameTiming {
                        decode: Some(decode_time),
                        detection: if self.detection_should_run() {
                            Some(detect_time)
                        } else {
                            None
                        },
                        stitch: Some(render_time),
                        total: Some(frame_t0.elapsed()),
                        ..Default::default()
                    });
                }
            }

            if let Some(cb) = on_progress.as_mut() {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        Ok(self.frame_count)
    }

    /// Process one GPU-resident frame with pre-extracted buffer info.
    ///
    /// The `buf_info` is extracted once before the frame loop to avoid
    /// per-frame clones and satisfy the borrow checker.
    #[cfg(target_os = "linux")]
    fn step_gpu_with_bufs(
        &mut self,
        buf_info: &Option<(
            crate::interop::zero_copy::GpuBufInfo,
            crate::interop::zero_copy::GpuBufInfo,
        )>,
        left_slot: u8,
        right_slot: u8,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        if let Some((left_buf, right_buf)) = buf_info {
            self.detect_and_update_director_gpu(
                left_buf, right_buf, left_slot, right_slot, elapsed,
            )?;
        }
        let pos = self.director_position();

        let bind_groups = self.gpu_bind_groups.as_ref().ok_or_else(|| {
            SessionError::ZeroCopy(
                "GPU bind groups not configured - call setup_gpu_source() before run()".into(),
            )
        })?;
        let render_buf = self.core.render_gpu_frame_at_pose(
            bind_groups,
            left_slot,
            right_slot,
            pos.yaw,
            pos.pitch,
        );
        self.submit_render_output(render_buf)?;

        // GPU stacked-replay pack on zero-copy sources. No-op when
        // the packer isn't enabled. Must complete before slot-free
        // release so the decode thread doesn't overwrite the
        // shared textures mid-pack.
        if let Some(ref views) = self.gpu_shared_views {
            let ls = left_slot as usize;
            let rs = right_slot as usize;
            self.core.pack_gpu_stacked_replay_from_views(
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &views[ls * 2],
                    uv: &views[ls * 2 + 1],
                },
                crate::gpu::yuv_stack_packer::StackedPackSource::Nv12 {
                    y: &views[4 + rs * 2],
                    uv: &views[4 + rs * 2 + 1],
                },
            );
        }

        // Release slots for decode thread to reuse
        if let Some((ref left_tx, ref right_tx)) = self.gpu_slot_free_tx {
            if left_tx.send(left_slot).is_err() {
                log::error!(
                    "Failed to release left GPU slot {left_slot} - decode thread may have died"
                );
            }
            if right_tx.send(right_slot).is_err() {
                log::error!(
                    "Failed to release right GPU slot {right_slot} - decode thread may have died"
                );
            }
        }

        Ok(())
    }
}
