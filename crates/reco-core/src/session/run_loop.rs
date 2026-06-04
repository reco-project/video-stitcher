//! Batch processing entry points for [`StitchSession`].
//!
//! The `run` / `run_immediate` frame loop, source configuration, and
//! GPU zero-copy frame stepping live here.

use std::sync::atomic::{AtomicBool, Ordering};

use super::StitchSession;
use crate::session::types::{FrameProgress, ProgressCallback, SessionError};
use crate::source::FrameSource;

/// Centered moving-average of a pose over the past + current + ahead
/// window. Averages yaw, pitch, AND fov, so the zoom is smoothed the
/// same lag-free way the angles are - otherwise FOV jitter survives the
/// lookahead untouched.
///
/// The window is symmetric only in steady state. At the stream head the
/// `past` side is short (it grows from empty during warm-up) and in the
/// drain tail the `ahead` side shrinks, so the average is computed over a
/// lopsided window at both boundaries - the first/last ~`post_smooth_half`
/// rendered poses are smoothed slightly differently from the middle. On a
/// clip shorter than the lookahead window the entire output is in this
/// boundary regime. This is acceptable (no future data exists past EOF);
/// it is documented so the boundary behavior is not mistaken for a bug.
fn centered_smooth(
    raw_pose: crate::detect::director::ViewportPosition,
    ahead: impl Iterator<Item = crate::detect::director::ViewportPosition>,
    past: impl Iterator<Item = crate::detect::director::ViewportPosition>,
) -> crate::detect::director::ViewportPosition {
    let mut sum_yaw = raw_pose.yaw;
    let mut sum_pitch = raw_pose.pitch;
    let mut sum_fov = raw_pose.fov_degrees.unwrap_or(0.0);
    let mut fov_n = u32::from(raw_pose.fov_degrees.is_some());
    let mut n = 1u32;
    for p in ahead.chain(past) {
        sum_yaw += p.yaw;
        sum_pitch += p.pitch;
        if let Some(f) = p.fov_degrees {
            sum_fov += f;
            fov_n += 1;
        }
        n += 1;
    }
    crate::detect::director::ViewportPosition {
        yaw: sum_yaw / n as f32,
        pitch: sum_pitch / n as f32,
        fov_degrees: (fov_n > 0).then(|| sum_fov / fov_n as f32),
    }
}

impl StitchSession {
    /// Auto-configure the session from source metadata.
    ///
    /// Called at the start of [`run`](Self::run). Applies rotation from
    /// the source's metadata.
    fn configure_from_source(&mut self, source: &dyn FrameSource) {
        self.gpu_pixel_format = source.gpu_pixel_format();
        self.is_full_range = source.is_full_range();
        if self.is_full_range {
            self.core.pipeline_mut().set_full_range(true);
        }
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
    /// The `run` loop uses these bind groups for GPU-resident
    /// `StereoFrame::GpuResident` frames.
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
        self.gpu_shared_textures = Some([
            t[0].texture.clone(),
            t[1].texture.clone(),
            t[2].texture.clone(),
            t[3].texture.clone(),
            t[4].texture.clone(),
            t[5].texture.clone(),
            t[6].texture.clone(),
            t[7].texture.clone(),
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

        let result = if self.lookahead_frames > 0 {
            let fps = source.info().fps.max(1.0);
            log::info!(
                "Lookahead: {} frames ({:.1}s at {:.2} fps)",
                self.lookahead_frames,
                self.lookahead_frames as f64 / fps,
                fps,
            );
            self.run_buffered(source, frame_limit, interrupted, &mut on_progress)
        } else {
            self.run_immediate(source, frame_limit, interrupted, &mut on_progress)
        };

        // Drop GPU slot senders so decode threads can exit gracefully.
        // Without this, SmartFileSource::drop() deadlocks because the
        // session's cloned senders keep the decode threads' recv() alive.
        #[cfg(target_os = "linux")]
        {
            self.gpu_slot_free_tx = None;
        }

        result
    }

    /// Render one buffered frame at the post-smoothed pose.
    ///
    /// Shared tail for the steady-state and drain render sites: emit the
    /// per-frame trace events, set the pose/skip-detection/slot state,
    /// render via `process_frame_any`, release the VRAM-pool slot, and
    /// fire the progress callback. `process_frame_any` increments
    /// `frame_count`, so events are emitted at the pre-increment index.
    fn render_buffered_frame(
        &mut self,
        oldest: super::frame_buffer::BufferedFrame,
        smoothed_pose: crate::detect::director::ViewportPosition,
        start: std::time::Instant,
        ctx: &crate::session::types::FrameLoopContext,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<(), SessionError> {
        if let Some(sink) = self.event_sink.as_deref_mut() {
            sink.emit(crate::detect::pipeline_event::PipelineEvent::FrameStart {
                frame_index: self.frame_count,
                timestamp_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::DetectionsRaw {
                    frame_index: self.frame_count,
                    detections: oldest.detections.clone(),
                },
            );
            sink.emit(crate::detect::pipeline_event::PipelineEvent::WorldState {
                frame_index: self.frame_count,
                timestamp_ms: oldest.elapsed_ms,
                players: oldest.world_state.players.clone(),
                ball: oldest.world_state.ball,
            });
            sink.emit(crate::detect::pipeline_event::PipelineEvent::PanDecision {
                frame_index: self.frame_count,
                pose: smoothed_pose,
            });
        }

        self.previous_panner_pose = smoothed_pose;
        self.skip_detection = true;
        self.current_vram_slot = oldest.vram_slot;

        let frame_t0 = std::time::Instant::now();
        self.process_frame_any(
            &oldest.frame,
            start.elapsed(),
            oldest.decode_time,
            frame_t0,
            ctx,
        )?;

        if let (Some(slot), Some(ref mut pool)) = (oldest.vram_slot, self.vram_pool.as_mut()) {
            pool.release(slot);
        }
        self.current_vram_slot = None;

        if let Some(cb) = on_progress.as_mut() {
            cb(&FrameProgress {
                frames_completed: self.frame_count,
                elapsed: start.elapsed(),
            });
        }
        Ok(())
    }

    /// Buffered frame loop with lookahead.
    ///
    /// Three phases: pre-fill the buffer with N frames (decode + detect),
    /// then steady-state (produce one, consume one), then drain remaining
    /// frames at EOF. The panner sees future WorldStates via
    /// `decide_with_lookahead`.
    fn run_buffered(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        use super::frame_buffer::{BufferedFrame, FrameBuffer};

        let start = std::time::Instant::now();
        let ctx = crate::session::types::FrameLoopContext {
            #[cfg(target_os = "linux")]
            gpu_buf_info: self.gpu_buf_info.clone(),
        };
        let n = self.lookahead_frames;
        let mut buffer = FrameBuffer::new(n + 1);
        let mut produce_count: u64 = 0;

        // Create VRAM pool if the source is GPU-resident.
        if source.is_gpu_resident() && self.vram_pool.is_none() {
            let post_smooth_half = (n / 2).max(1);
            // Keep in sync with the D3D11 staging pool sizing in
            // frame_processing.rs (peak occupancy + slack).
            let pool_size = n + post_smooth_half + 4;
            let (w, h) = self.core.pipeline().source_info();
            let per_slot =
                super::vram_pool::estimate_vram(w, h, 1, self.gpu_pixel_format.bytes_per_sample());
            let required = per_slot * pool_size;

            // Pre-flight VRAM budget check: fail fast with a fit suggestion.
            // available_vram() is None on backends with no budget API; there
            // we skip the check and rely on the allocation-time catch in
            // VramPool::new plus graceful teardown for any slip-through.
            match self.core.pipeline().gpu().available_vram() {
                Some((free, total)) => {
                    const BUDGET_FRACTION: f64 = 0.80;
                    let budget = (free as f64 * BUDGET_FRACTION) as usize;
                    log::info!(
                        "VRAM budget: {:.2} GB free / {:.2} GB total; lookahead pool needs \
                         {:.2} GB ({pool_size} slots @ {w}x{h}), keeping {:.0}% headroom",
                        free as f64 / 1e9,
                        total as f64 / 1e9,
                        required as f64 / 1e9,
                        (1.0 - BUDGET_FRACTION) * 100.0,
                    );
                    if required > budget {
                        let fps = source.info().fps.max(1.0);
                        let max_slots = budget / per_slot.max(1);
                        // pool_size = n + (n/2).max(1) + 2 ~= 1.5n + 2; invert for n.
                        let max_n = max_slots.saturating_sub(2) * 2 / 3;
                        let max_secs = max_n as f64 / fps;
                        let req_secs = n as f64 / fps;
                        return Err(SessionError::Config(format!(
                            "--lookahead {req_secs:.1}s needs ~{:.1} GB VRAM for the frame pool \
                             ({pool_size} slots @ {w}x{h}) but only ~{:.1} GB is free. The pool \
                             stores decoded source-resolution frames, so reduce --lookahead to \
                             <= {max_secs:.1}s, use lower-resolution source footage, or free GPU \
                             memory, then retry. (Output resolution does not affect this pool.)",
                            required as f64 / 1e9,
                            free as f64 / 1e9,
                        )));
                    }
                }
                None => {
                    log::info!(
                        "VRAM budget query unavailable on this backend; relying on \
                         allocation-time OOM handling for the {pool_size}-slot lookahead pool \
                         (~{:.2} GB @ {w}x{h})",
                        required as f64 / 1e9,
                    );
                }
            }

            // The VramPool is only consumed on Linux (CUDA/Vulkan copy) and
            // macOS (Metal CVPixelBuffer import). On Windows the lookahead
            // frames live in the separate D3D11 staging pool, so a VramPool
            // here would be allocated-but-unused VRAM (it roughly doubles the
            // footprint). The pre-flight budget check above still runs on all
            // platforms; on Windows it sizes the D3D11 staging pool, whose
            // total VRAM equals estimate_vram(w,h,1,bps)*pool_size.
            #[cfg(not(target_os = "windows"))]
            {
                let pool = super::vram_pool::VramPool::new(
                    self.core.pipeline().gpu(),
                    self.core.pipeline(),
                    w,
                    h,
                    pool_size,
                    self.gpu_pixel_format,
                )
                .map_err(SessionError::Config)?;
                self.vram_pool = Some(pool);
            }
        }

        let produce_one = |session: &mut StitchSession,
                           source: &mut dyn FrameSource,
                           buffer: &mut FrameBuffer,
                           produce_count: &mut u64,
                           start: std::time::Instant,
                           _ctx: &crate::session::types::FrameLoopContext|
         -> Result<bool, SessionError> {
            let frame_t0 = std::time::Instant::now();
            let frame = match source.next_frame()? {
                Some(f) => f,
                None => return Ok(false),
            };
            let decode_time = frame_t0.elapsed();
            let elapsed = start.elapsed();

            // Stage GPU frames to persistent slots BEFORE detection,
            // so detection can use the staged textures.
            let vram_slot = session.copy_to_vram_pool(&frame, *produce_count)?;

            let world_state = session.detect_and_track_only(&frame, elapsed, *produce_count)?;
            let detections = session.detection.last_detections.clone();

            // Detection has now read the decode slot; it is safe to hand
            // it back to the decode thread for reuse. Releasing earlier
            // (inside copy_to_vram_pool) raced detection's read.
            session.release_gpu_decode_slot(&frame);

            buffer.push(BufferedFrame {
                frame,
                world_state,
                detections,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                decode_time,
                vram_slot,
            });

            *produce_count += 1;
            Ok(true)
        };

        // ── Pre-fill: decode + detect N frames, no rendering ──────
        log::info!("Lookahead: pre-filling {} frames...", n);
        for _ in 0..n {
            if interrupted.load(Ordering::Relaxed) {
                break;
            }
            if !produce_one(self, source, &mut buffer, &mut produce_count, start, &ctx)? {
                break;
            }
        }
        log::info!(
            "Lookahead: pre-filled {} frames, starting render",
            buffer.len()
        );

        // Centered post-smooth: the panner runs AHEAD of rendering
        // so we can average past + future poses symmetrically.
        let post_smooth_half: usize = (n / 2).max(1);

        // frame_count stays at 0 - produce_count drives detection
        // interval, frame_count counts rendered output.

        // Queue of (frame, raw_pose) pairs. The panner fills this
        // ahead of rendering. Rendering consumes from the front,
        // using the centered average of past + current + future poses.
        let mut pose_queue: std::collections::VecDeque<(
            BufferedFrame,
            crate::detect::director::ViewportPosition,
        )> = std::collections::VecDeque::new();
        let mut past_poses: std::collections::VecDeque<crate::detect::director::ViewportPosition> =
            std::collections::VecDeque::new();
        let mut panner_frame_idx: u64 = 0;

        // Helper: run the panner on the oldest buffered frame, push
        // the (frame, pose) pair into the pose queue.
        let run_panner_once = |session: &mut StitchSession,
                               buffer: &mut FrameBuffer,
                               pose_queue: &mut std::collections::VecDeque<(
            BufferedFrame,
            crate::detect::director::ViewportPosition,
        )>,
                               panner_frame_idx: &mut u64| {
            if let Some(frame) = buffer.pop() {
                session.lookahead_world_states = buffer.future_world_states();
                let pose = if let Some(panner) = session.panner.as_mut() {
                    let pan_ctx = crate::detect::panner::PanContext {
                        frame_index: *panner_frame_idx,
                        timestamp_ms: frame.elapsed_ms,
                        previous_position: session.previous_panner_pose,
                        calibration: session.core.pipeline().calibration(),
                    };
                    let p = panner.decide_with_lookahead(
                        &frame.world_state,
                        &session.lookahead_world_states,
                        &pan_ctx,
                    );
                    session.previous_panner_pose = p;
                    p
                } else {
                    session.previous_panner_pose
                };
                *panner_frame_idx += 1;
                pose_queue.push_back((frame, pose));
                true
            } else {
                false
            }
        };

        // ── Panner warm-up: run panner post_smooth_half frames ahead ──
        let mut eof = buffer.len() < n;
        for _ in 0..post_smooth_half {
            if buffer.is_empty() {
                break;
            }
            if !eof
                && !buffer.is_full()
                && !produce_one(self, source, &mut buffer, &mut produce_count, start, &ctx)?
            {
                eof = true;
            }
            run_panner_once(self, &mut buffer, &mut pose_queue, &mut panner_frame_idx);
        }

        // ── Steady state: produce, run panner ahead, render with centered smooth ──
        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            // Produce: decode + detect next frame
            if !eof
                && !buffer.is_full()
                && !produce_one(self, source, &mut buffer, &mut produce_count, start, &ctx)?
            {
                eof = true;
            }

            // Run panner on next buffered frame (stays ahead of rendering)
            if !buffer.is_empty() {
                run_panner_once(self, &mut buffer, &mut pose_queue, &mut panner_frame_idx);
            }

            // Render: consume from pose queue when we have enough context
            if pose_queue.len() > post_smooth_half || (eof && !pose_queue.is_empty()) {
                let (oldest, raw_pose) = pose_queue.pop_front().unwrap();

                // Centered post-smooth: average past + current + future
                // poses (yaw/pitch/fov) so zoom is smoothed like the angles.
                let smoothed_pose = centered_smooth(
                    raw_pose,
                    pose_queue.iter().take(post_smooth_half).map(|(_, p)| *p),
                    past_poses.iter().copied(),
                );
                past_poses.push_back(raw_pose);
                if past_poses.len() > post_smooth_half {
                    past_poses.pop_front();
                }

                self.render_buffered_frame(oldest, smoothed_pose, start, &ctx, on_progress)?;
            } else if eof && buffer.is_empty() && pose_queue.is_empty() {
                break;
            }
        }

        // ── Drain: render remaining pose queue entries ────────────
        while let Some((oldest, raw_pose)) = pose_queue.pop_front() {
            if self.frame_count >= frame_limit || interrupted.load(Ordering::Relaxed) {
                break;
            }
            let smoothed_pose = centered_smooth(
                raw_pose,
                pose_queue.iter().take(post_smooth_half).map(|(_, p)| *p),
                past_poses.iter().copied(),
            );
            past_poses.push_back(raw_pose);
            if past_poses.len() > post_smooth_half {
                past_poses.pop_front();
            }

            self.render_buffered_frame(oldest, smoothed_pose, start, &ctx, on_progress)?;
        }

        self.skip_detection = false;
        self.lookahead_world_states.clear();
        Ok(self.frame_count)
    }

    /// Standard frame loop. Handles CPU-resident and GPU-resident
    /// frames transparently via [`process_frame_any`](Self::process_frame_any).
    fn run_immediate(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<u64, SessionError> {
        let start = std::time::Instant::now();

        let ctx = crate::session::types::FrameLoopContext {
            #[cfg(target_os = "linux")]
            gpu_buf_info: self.gpu_buf_info.clone(),
        };

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

            self.process_frame_any(&frame, start.elapsed(), decode_time, frame_t0, &ctx)?;

            if let Some(cb) = on_progress.as_mut() {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
        }

        Ok(self.frame_count)
    }
}
