//! Batch processing entry points for [`StitchSession`].
//!
//! The `run` / `run_immediate` frame loop, source configuration, and
//! GPU zero-copy frame stepping live here.

use std::sync::atomic::{AtomicBool, Ordering};

use super::StitchSession;
use crate::session::types::{FrameProgress, ProgressCallback, SessionError};
use crate::source::FrameSource;

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
            if source.is_gpu_resident() {
                log::error!(
                    "Lookahead requires CPU decode (--no-zero-copy). \
                     GPU-resident frame buffering is not yet supported."
                );
                return Err(SessionError::Config("lookahead requires CPU decode".into()));
            }
            log::info!(
                "Lookahead: {} frames ({:.1}s at source fps)",
                self.lookahead_frames,
                self.lookahead_frames as f64 / 30.0,
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

        // Helper: decode one frame, run detection, capture WorldState,
        // push into buffer. Returns false on EOF.
        //
        // Suppresses event emission during produce: the JSONL sink
        // should only record events at render time when the final
        // pose (with lookahead context) is decided.
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

            // Run detection + trackers only (no panner, no events).
            // The panner runs at render time with the full lookahead window.
            let world_state = session.detect_and_track_only(&frame, elapsed, *produce_count)?;

            buffer.push(BufferedFrame {
                frame,
                world_state,
                frame_index: *produce_count,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                decode_time,
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

        // frame_count stays at 0 - produce_count drives detection
        // interval, frame_count counts rendered output.

        // ── Steady state: produce one, consume one ────────────────
        let mut eof = buffer.len() < n;

        while self.frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            // Produce: decode + detect next frame (unless EOF)
            if !eof
                && !buffer.is_full()
                && !produce_one(self, source, &mut buffer, &mut produce_count, start, &ctx)?
            {
                eof = true;
            }

            // Consume: render oldest frame with lookahead window
            if let Some(oldest) = buffer.pop() {
                // Set the future WorldStates for the panner
                self.lookahead_world_states = buffer.future_world_states();

                // Emit buffered events at render time.
                if let Some(sink) = self.event_sink.as_deref_mut() {
                    sink.emit(crate::detect::pipeline_event::PipelineEvent::FrameStart {
                        frame_index: self.frame_count,
                        timestamp_ms: start.elapsed().as_secs_f64() * 1000.0,
                    });
                    sink.emit(crate::detect::pipeline_event::PipelineEvent::WorldState {
                        frame_index: self.frame_count,
                        timestamp_ms: oldest.elapsed_ms,
                        players: oldest.world_state.players.clone(),
                        ball: oldest.world_state.ball,
                    });
                }

                // Set the stored WorldState so director_position uses it
                // via dispatch. Then call process_frame_any with detection
                // skipped.
                self.previous_panner_pose = if self.panner.is_some() {
                    let pan_ctx = crate::detect::panner::PanContext {
                        frame_index: self.frame_count,
                        timestamp_ms: oldest.elapsed_ms,
                        previous_position: self.previous_panner_pose,
                        calibration: self.core.pipeline().calibration(),
                    };
                    let pose = self.panner.as_mut().unwrap().decide_with_lookahead(
                        &oldest.world_state,
                        &self.lookahead_world_states,
                        &pan_ctx,
                    );
                    if let Some(sink) = self.event_sink.as_deref_mut() {
                        sink.emit(crate::detect::pipeline_event::PipelineEvent::PanDecision {
                            frame_index: self.frame_count,
                            pose,
                        });
                        if let Some(debug) =
                            self.panner.as_ref().unwrap().debug_event(self.frame_count)
                        {
                            sink.emit(debug);
                        }
                    }
                    pose
                } else {
                    self.previous_panner_pose
                };

                self.skip_detection = true;

                let frame_t0 = std::time::Instant::now();
                self.process_frame_any(
                    &oldest.frame,
                    start.elapsed(),
                    oldest.decode_time,
                    frame_t0,
                    &ctx,
                )?;

                if let Some(cb) = on_progress.as_mut() {
                    cb(&FrameProgress {
                        frames_completed: self.frame_count,
                        elapsed: start.elapsed(),
                    });
                }
            } else {
                break; // Buffer empty and EOF
            }
        }

        // ── Drain: render remaining buffered frames ───────────────
        while let Some(oldest) = buffer.pop() {
            if self.frame_count >= frame_limit || interrupted.load(Ordering::Relaxed) {
                break;
            }
            self.lookahead_world_states = buffer.future_world_states();

            if let Some(sink) = self.event_sink.as_deref_mut() {
                sink.emit(crate::detect::pipeline_event::PipelineEvent::FrameStart {
                    frame_index: self.frame_count,
                    timestamp_ms: start.elapsed().as_secs_f64() * 1000.0,
                });
                sink.emit(crate::detect::pipeline_event::PipelineEvent::WorldState {
                    frame_index: self.frame_count,
                    timestamp_ms: oldest.elapsed_ms,
                    players: oldest.world_state.players.clone(),
                    ball: oldest.world_state.ball,
                });
            }

            self.previous_panner_pose = if self.panner.is_some() {
                let pan_ctx = crate::detect::panner::PanContext {
                    frame_index: self.frame_count,
                    timestamp_ms: oldest.elapsed_ms,
                    previous_position: self.previous_panner_pose,
                    calibration: self.core.pipeline().calibration(),
                };
                let pose = self.panner.as_mut().unwrap().decide_with_lookahead(
                    &oldest.world_state,
                    &self.lookahead_world_states,
                    &pan_ctx,
                );
                if let Some(sink) = self.event_sink.as_deref_mut() {
                    sink.emit(crate::detect::pipeline_event::PipelineEvent::PanDecision {
                        frame_index: self.frame_count,
                        pose,
                    });
                    if let Some(debug) = self.panner.as_ref().unwrap().debug_event(self.frame_count)
                    {
                        sink.emit(debug);
                    }
                }
                pose
            } else {
                self.previous_panner_pose
            };

            self.skip_detection = true;

            let frame_t0 = std::time::Instant::now();
            self.process_frame_any(
                &oldest.frame,
                start.elapsed(),
                oldest.decode_time,
                frame_t0,
                &ctx,
            )?;

            if let Some(cb) = on_progress.as_mut() {
                cb(&FrameProgress {
                    frames_completed: self.frame_count,
                    elapsed: start.elapsed(),
                });
            }
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
