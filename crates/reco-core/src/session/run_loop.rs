//! Batch processing entry points for [`StitchSession`].
//!
//! The `run` / `run_immediate` frame loop, source configuration, and
//! GPU zero-copy frame stepping live here.

use crate::geometry::WorldPose;
use std::sync::atomic::{AtomicBool, Ordering};

use super::StitchSession;
use crate::session::types::{FrameProgress, ProgressCallback, SessionError};
use crate::source::{FrameSource, SourceError, SourceFrame};

fn accept_source_time(
    previous: &mut Option<Option<i64>>,
    next: Option<i64>,
) -> Result<(), SourceError> {
    let valid = match (*previous, next) {
        (None, Some(time)) => time >= 0,
        (Some(Some(last)), Some(time)) => time > last,
        (None | Some(None), None) => true,
        _ => false,
    };
    if !valid {
        return Err(SourceError::Read {
            reason: "source presentation timestamps must be nonnegative, strictly increasing and consistently present or absent".into(),
        });
    }
    *previous = Some(next);
    Ok(())
}

fn frame_seconds(fps: f64) -> Result<f32, SessionError> {
    let seconds = (1.0 / fps) as f32;
    if fps.is_finite() && fps > 0.0 && seconds.is_finite() && seconds > 0.0 {
        Ok(seconds)
    } else {
        Err(SessionError::Config(format!(
            "source frame rate must be finite and positive, got {fps}"
        )))
    }
}

fn accept_capture_timing(
    previous: &mut Option<Option<crate::source::CaptureTiming>>,
    capture: Option<&crate::source::CaptureTiming>,
    cameras: usize,
) -> Result<(), SourceError> {
    if let Some(capture) = capture {
        capture
            .validate(cameras)
            .map_err(|reason| SourceError::Read { reason })?;
    }
    let valid = match (previous.as_ref(), capture) {
        (None, _) | (Some(None), None) => true,
        (Some(Some(last)), Some(next)) => {
            last.max_age_us == next.max_age_us
                && next.at_us >= last.at_us
                && (next.max_age_us.is_none() || next.at_us > last.at_us)
                && last
                    .times_us
                    .iter()
                    .zip(&next.times_us)
                    .all(|(last, next)| next >= last)
        }
        _ => false,
    };
    if !valid {
        return Err(SourceError::Read {
            reason: "capture timing must remain ordered, with consistent availability and age policy; bounded assessment time must advance".into(),
        });
    }
    if let (Some(Some(last)), Some(next)) = (previous.as_mut(), capture) {
        last.times_us.clone_from(&next.times_us);
        last.at_us = next.at_us;
    } else {
        *previous = Some(capture.cloned());
    }
    Ok(())
}

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
    raw_pose: WorldPose,
    ahead: impl Iterator<Item = WorldPose>,
    past: impl Iterator<Item = WorldPose>,
) -> WorldPose {
    let mut sum_yaw = raw_pose.yaw;
    let mut sum_pitch = raw_pose.pitch;
    let mut sum_fov = raw_pose.fov_degrees;
    let mut n = 1u32;
    for p in ahead.chain(past) {
        // Average neighboring directions on the current pose's angular branch.
        let delta = (p.yaw - raw_pose.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        sum_yaw += raw_pose.yaw + delta;
        sum_pitch += p.pitch;
        sum_fov += p.fov_degrees;
        n += 1;
    }
    WorldPose {
        yaw: sum_yaw / n as f32,
        pitch: sum_pitch / n as f32,
        fov_degrees: sum_fov / n as f32,
    }
}

impl StitchSession {
    fn validate_source_frame(&self, frame: &crate::source::FrameSet) -> Result<(), SessionError> {
        frame
            .validate_host_images(self.core.source_sizes())
            .map_err(SessionError::Config)?;
        use crate::source::FrameSet;
        let cameras = match frame {
            FrameSet::Yuv420p(cameras) => cameras.len(),
            FrameSet::Nv12(cameras) => cameras.len(),
            _ if matches!(self.core.executor, crate::stitch::Executor::Cpu(_)) => {
                return Err(SessionError::Config(
                    "native resident frames require a GPU executor".into(),
                ));
            }
            FrameSet::GpuResident { slots } => {
                if slots.iter().any(|&slot| slot > 1) {
                    return Err(SessionError::Config("invalid shared decode slot".into()));
                }
                slots.len()
            }
            #[cfg(target_os = "linux")]
            FrameSet::NvmmResident(images) => {
                for (camera, image) in images.iter().enumerate() {
                    let info = image.info();
                    if self.core.source_sizes().get(camera) != Some(&(info.width, info.height)) {
                        return Err(SessionError::Config("NVMM image dimensions changed".into()));
                    }
                }
                images.len()
            }
            #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
            FrameSet::MetalResident(images) => {
                for (camera, image) in images.iter().enumerate() {
                    if self.core.source_sizes().get(camera)
                        != Some(&(image.width(), image.height()))
                    {
                        return Err(SessionError::Config(
                            "Metal image dimensions changed".into(),
                        ));
                    }
                }
                images.len()
            }
            #[cfg(target_os = "windows")]
            FrameSet::D3d11Resident(images) => images.len(),
        };
        let expected = self.core.calibration().projection.camera_count();
        if cameras != expected {
            return Err(SessionError::Config(format!(
                "expected {expected} source cameras, got {cameras}"
            )));
        }
        Ok(())
    }

    /// Auto-configure the session from source metadata.
    ///
    /// Called at the start of [`run`](Self::run). Applies rotation from
    /// the source's metadata.
    fn configure_from_source(&mut self, source: &dyn FrameSource) {
        #[cfg(feature = "gpu")]
        {
            self.gpu_pixel_format = source.gpu_pixel_format();
        }
        self.source_full_range = source.is_full_range();
        self.core.set_full_range(self.source_full_range);
        // Apply rotation via shader UV flip ONLY for GPU-resident sources.
        // CPU sources handle rotation via buffer reversal in the decoder,
        // so applying the shader flip too would rotate 360 degrees (no-op but wrong).
        let (lr, rr) = if source.is_gpu_resident() {
            (source.left_rotation(), source.right_rotation())
        } else {
            (0, 0)
        };
        self.core.set_flip_180(lr == 180, rr == 180);
        if lr == 180 || rr == 180 {
            log::info!("Rotation: UV flip left={}, right={}", lr == 180, rr == 180);
        }
        // Detector preprocessing consumes the same source orientation independently.
        #[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "windows")))]
        {
            self.left_rotation = lr;
            self.right_rotation = rr;
        }
    }

    /// Configure the session for a GPU-resident source.
    ///
    /// Hands the source's shared textures to the GPU executor, which
    /// builds render bind groups, detection/replay views, and the
    /// decode backpressure channels. Call this before
    /// [`run`](Self::run) when using a GPU-resident [`FrameSource`]
    /// like `SmartFileSource`.
    /// Requires a GPU executor and one shared surface per projection input.
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    pub fn setup_gpu_source(
        &mut self,
        shared: &crate::interop::SharedTextureSet,
    ) -> Result<(), SessionError> {
        if self.core.executor.gpu().is_none()
            || self.core.calibration().projection.camera_count() != shared.cameras.len()
        {
            return Err(SessionError::Config(
                "shared decode textures must match the GPU executor camera count".into(),
            ));
        }
        for (camera, &(width, height)) in shared.cameras.iter().zip(self.core.source_sizes()) {
            if camera.buf.width != width
                || camera.buf.height != height
                || camera.textures.iter().enumerate().any(|(plane, texture)| {
                    let divisor = if plane % 2 == 0 { 1 } else { 2 };
                    (texture.texture.width(), texture.texture.height())
                        != (width / divisor, height / divisor)
                })
            {
                return Err(SessionError::Config(
                    "shared decode texture dimensions do not match the executor".into(),
                ));
            }
        }
        self.gpu_exec().configure_shared_textures(shared);
        log::info!("Session configured for GPU-resident source");
        Ok(())
    }

    /// Batch-process frames from a source into the attached sinks.
    ///
    /// Runs the full decode-render-encode loop until the source is
    /// exhausted, the frame limit is reached, or the interrupt flag
    /// is set. Returns the processed frame count and the observed stop reason.
    /// Reaching the frame limit takes precedence over a concurrent cancellation.
    ///
    /// Automatically handles CPU-resident and GPU-resident frames:
    /// - CPU frames (Yuv420p, Nv12): software-rendered or uploaded to the GPU
    /// - GPU frames (GpuResident): rendered directly from shared textures
    ///
    /// Does NOT call [`Self::finish`] - the caller must do that after this
    /// returns to flush the last frame and finalize encoding.
    /// Each run clears cached detections and image identities because source capture clocks may have different origins. Tracker state is not reset.
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
    ) -> Result<super::types::RunResult, SessionError> {
        if source.info().camera_sizes != self.core.source_sizes() {
            return Err(SessionError::Config(
                "source camera dimensions do not match the executor".into(),
            ));
        }
        if source.is_gpu_resident() && matches!(self.core.executor, crate::stitch::Executor::Cpu(_))
        {
            return Err(SessionError::Config(
                "native resident sources require a GPU executor".into(),
            ));
        }

        self.telemetry.start();
        self.configure_from_source(source);
        // A new source run owns its own capture-clock origin and observations.
        self.core.clear_detection_cache();

        // A host-backed run cannot consume cached native staging images.
        // Release them before starting its retained-image allowance.
        #[cfg(feature = "gpu")]
        if !source.is_gpu_resident()
            && let Some(exec) = self.core.executor.gpu_mut()
        {
            exec.pipeline.clear_imported_bind_groups();
            exec.residency.pool = None;
            #[cfg(target_os = "windows")]
            {
                exec.residency.d3d11_staging = None;
            }
        }

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
        // executor's cloned senders keep the decode threads' recv() alive.
        #[cfg(all(feature = "gpu", target_os = "linux"))]
        if let Some(exec) = self.core.executor.gpu_mut() {
            exec.drop_decode_channels();
            if let Err(error) = exec.finish_nvmm_reads() {
                if result.is_ok() {
                    return Err(SessionError::ZeroCopy(error));
                }
                log::error!("{error}");
            }
        }

        result
    }

    /// Render one buffered frame at the post-smoothed pose.
    ///
    /// Emit per-frame trace events, set the pose/skip-detection/slot state,
    /// render via `process_frame_any`, release the VRAM-pool slot, and
    /// fire the progress callback. `process_frame_any` increments
    /// `frame_count`, so events are emitted at the pre-increment index.
    fn render_buffered_frame(
        &mut self,
        oldest: super::frame_buffer::BufferedFrame,
        smoothed_pose: crate::geometry::WorldPose,
        start: std::time::Instant,
        ctx: &crate::session::types::FrameLoopContext,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<(), SessionError> {
        let frame_t0 = std::time::Instant::now();
        self.telemetry.record_detections(
            oldest.detections.len() as u32,
            oldest.world_state.players.len() as u32,
            oldest
                .world_state
                .ball
                .as_ref()
                .is_some_and(|ball| ball.state != crate::detect::tracker::TrackState::Lost),
        );
        if let Some(sink) = self.core.event_sink.as_deref_mut() {
            sink.emit(crate::detect::pipeline_event::PipelineEvent::FrameStart {
                frame_index: self.frame_count,
                timestamp_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
            sink.emit(
                crate::detect::pipeline_event::PipelineEvent::MappedDetections {
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

        self.core.set_requested_world_pose(smoothed_pose);
        #[cfg(feature = "gpu")]
        {
            self.current_vram_slot = oldest.vram_slot;
        }

        let result = self.process_frame_any(
            oldest.frame.as_ref(),
            start.elapsed(),
            oldest.pts_us,
            oldest.timing,
            frame_t0,
            ctx,
        );

        #[cfg(feature = "gpu")]
        {
            if let Some(slot) = oldest.vram_slot {
                self.gpu_exec().release_pool_slot(slot);
            }
            self.current_vram_slot = None;
        }
        result?;

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
    /// Pre-fill the buffer with N frames (decode + detect), then produce
    /// and render until EOF and the buffered tail are exhausted. The panner
    /// sees future WorldStates via `decide_with_lookahead`.
    fn run_buffered(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<super::types::RunResult, SessionError> {
        use super::frame_buffer::{BufferedFrame, FrameBuffer};

        let mut completion = super::types::RunCompletion::Completed;
        let start = std::time::Instant::now();
        let ctx = crate::session::types::FrameLoopContext {
            tracking_complete: true,
            capture: None,
            frame_seconds: frame_seconds(source.info().fps)?,
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            gpu_buf_info: self.core.executor.gpu().and_then(|g| g.cuda_buf_info()),
        };
        let n = self.lookahead_frames;
        let mut image_budget = crate::lookahead::ImageBudget::new(self.lookahead_memory_bytes)
            .map_err(SessionError::Config)?;
        let capacity = n
            .checked_add(1)
            .ok_or_else(|| SessionError::Config("lookahead capacity overflow".into()))?;
        let mut buffer = FrameBuffer::new(capacity);
        let mut produce_count: u64 = 0;

        // Native decode/capture pools need persistent staging. Metal frames
        // retain CVPixelBuffers and render directly from those buffers.
        #[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "windows")))]
        if source.is_gpu_resident() {
            let pool_size = crate::lookahead::pool_slots(n).map_err(SessionError::Config)?;
            let per_slot = crate::lookahead::image_bytes(
                self.core.source_sizes(),
                self.gpu_pixel_format.bytes_per_sample(),
            )
            .map_err(SessionError::Config)?;
            let required = per_slot
                .checked_mul(pool_size)
                .ok_or_else(|| SessionError::Config("lookahead pool byte size overflow".into()))?;
            if self
                .gpu_exec_ref()
                .residency
                .pool
                .as_ref()
                .is_some_and(|p| {
                    p.capacity() != pool_size || p.pixel_format != self.gpu_pixel_format
                })
            {
                return Err(SessionError::Config(
                    "lookahead pool size or pixel format changed; create a new session".into(),
                ));
            }

            // Pre-flight VRAM budget check: fail fast with a fit suggestion.
            // available_vram() is None on backends with no budget API; there
            // we skip the check and rely on the allocation-time catch in
            // VramPool::new plus graceful teardown for any slip-through.
            #[cfg(target_os = "windows")]
            let needs_allocation = self.gpu_exec_ref().residency.d3d11_staging.is_none();
            #[cfg(not(target_os = "windows"))]
            let needs_allocation = self.gpu_exec_ref().residency.pool.is_none();
            if needs_allocation {
                match self.gpu_exec_ref().pipeline.gpu().available_vram() {
                    Some((free, total)) => {
                        // Budget trusts the driver's "free" figure when it is
                        // believable (it already excludes the compositor and the
                        // rest of the pipeline) and falls back to a fraction of
                        // total when free reads bogus (~0 on a multi-GB card).
                        // Sampled here just before pool creation, after the GUI
                        // preview is released, so it reflects real export-time room.
                        // Over-allocation is caught at pool creation.
                        let budget = crate::gpu::vram_pool::lookahead_budget_bytes(free, total);
                        log::info!(
                            "VRAM budget: {:.2} GB usable of {:.2} GB total ({:.2} GB reported free); \
                         lookahead pool needs {:.2} GB ({pool_size} slots)",
                            budget as f64 / 1e9,
                            total as f64 / 1e9,
                            free as f64 / 1e9,
                            required as f64 / 1e9,
                        );
                        if required > budget {
                            let fps = source.info().fps.max(1.0);
                            let max_n = crate::lookahead::max_frames(per_slot, budget);
                            let max_secs = max_n as f64 / fps;
                            let req_secs = n as f64 / fps;
                            return Err(SessionError::Config(format!(
                                "not enough VRAM for a {req_secs:.1}s lookahead: reduce the lookahead \
                             to <= {max_secs:.1}s, use lower-resolution source footage, or free \
                             GPU memory. The frame pool needs ~{:.1} GB ({pool_size} slots @ \
                             configured camera sizes); usable budget is ~{:.1} GB of {:.1} GB total.",
                                required as f64 / 1e9,
                                budget as f64 / 1e9,
                                total as f64 / 1e9,
                            )));
                        }
                    }
                    None => {
                        log::info!(
                            "VRAM budget query unavailable on this backend; relying on \
                         allocation-time OOM handling for the {pool_size}-slot lookahead pool \
                         (~{:.2} GB)",
                            required as f64 / 1e9,
                        );
                    }
                }
            }

            image_budget.retain(required).map_err(|error| {
                let max_frames = crate::lookahead::max_frames(per_slot, self.lookahead_memory_bytes);
                SessionError::Config(format!(
                    "{error}; requested {n} frames, staging payload fits at most {max_frames} lookahead frames"
                ))
            })?;

            // Windows uses its D3D11 staging pool instead of a second allocation.
            #[cfg(target_os = "linux")]
            if self.gpu_exec_ref().residency.pool.is_none() {
                let pixel_format = self.gpu_pixel_format;
                self.gpu_exec()
                    .create_lookahead_pool(pool_size, pixel_format)
                    .map_err(SessionError::Config)?;
            }
        }

        let mut previous_time = None;
        let mut previous_capture_times = None;
        let mut produce_one = |session: &mut StitchSession,
                               source: &mut dyn FrameSource,
                               buffer: &mut FrameBuffer,
                               image_budget: &mut crate::lookahead::ImageBudget,
                               produce_count: &mut u64,
                               start: std::time::Instant,
                               completion: &mut super::types::RunCompletion|
         -> Result<bool, SessionError> {
            let frame_t0 = std::time::Instant::now();
            let frame = match source.next_frame()? {
                Some(f) => f,
                None => {
                    if interrupted.load(Ordering::Relaxed) {
                        *completion = super::types::RunCompletion::Cancelled;
                    }
                    return Ok(false);
                }
            };
            let decode_time = frame_t0.elapsed();
            let elapsed = start.elapsed();
            let SourceFrame {
                images: frame,
                pts_us,
                capture,
            } = frame;
            accept_source_time(&mut previous_time, pts_us)?;
            accept_capture_timing(
                &mut previous_capture_times,
                capture.as_ref(),
                session.core.calibration().projection.camera_count(),
            )?;
            session.validate_source_frame(&frame)?;
            let retained_bytes = frame.retained_image_bytes().map_err(SessionError::Config)?;
            image_budget
                .retain(retained_bytes)
                .map_err(SessionError::Config)?;

            // Stage GPU frames to persistent slots BEFORE detection,
            // so detection can use the staged textures.
            #[cfg(feature = "gpu")]
            let (vram_slot, staging_time) = {
                let start = std::time::Instant::now();
                let slot = session.copy_to_vram_pool(&frame, *produce_count)?;
                (slot, start.elapsed())
            };
            #[cfg(not(feature = "gpu"))]
            let staging_time = std::time::Duration::ZERO;

            let world_state = session.detect_and_track_only(
                &frame,
                elapsed,
                *produce_count,
                capture.as_ref(),
                ctx.frame_seconds,
            );
            // Detection still reads the original decode surface after staging.
            #[cfg(feature = "gpu")]
            session.release_gpu_decode_slot(&frame);
            let (world_state, mut timing) = match world_state {
                Ok(state) => state,
                Err(error) => {
                    #[cfg(feature = "gpu")]
                    if let Some(slot) = vram_slot {
                        session.gpu_exec().release_pool_slot(slot);
                    }
                    return Err(error);
                }
            };
            let detections = session.core.last_detections().to_vec();

            // Staging has completed and detection no longer uses the capture
            // buffer. Keeping it in the lookahead queue would exhaust the ISP pool.
            let frame = Some(frame);
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            let frame = super::frame_buffer::release_staged_nvmm(frame, vram_slot);

            #[cfg(all(feature = "gpu", target_os = "windows"))]
            let frame = frame.filter(|f| {
                !(vram_slot.is_some() && matches!(f, crate::source::FrameSet::D3d11Resident(_)))
            });
            timing.decode = Some(decode_time);
            timing.upload = Some(staging_time);
            timing.total = Some(frame_t0.elapsed());
            buffer.push(BufferedFrame {
                frame,
                pts_us,
                world_state,
                detections,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                timing,
                #[cfg(feature = "gpu")]
                vram_slot,
                retained_bytes,
            });

            *produce_count += 1;
            Ok(true)
        };

        let mut pose_queue = std::collections::VecDeque::new();
        let result = (|| {
            // ── Pre-fill: decode + detect N frames, no rendering ──────
            log::info!("Lookahead: pre-filling {} frames...", n);
            for _ in 0..n {
                if !continue_run(&mut completion, self.frame_count, frame_limit, interrupted) {
                    break;
                }
                if !produce_one(
                    self,
                    source,
                    &mut buffer,
                    &mut image_budget,
                    &mut produce_count,
                    start,
                    &mut completion,
                )? {
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

            // Production schedules detection independently of rendered output.
            let mut past_poses: std::collections::VecDeque<crate::geometry::WorldPose> =
                std::collections::VecDeque::new();
            let mut panner_frame_idx: u64 = 0;

            // Helper: run the panner on the oldest buffered frame, push
            // the (frame, pose) pair into the pose queue. The engine owns
            // the panner; the buffer supplies the future WorldState window.
            let run_panner_once = |session: &mut StitchSession,
                                   buffer: &mut FrameBuffer,
                                   pose_queue: &mut std::collections::VecDeque<(
                BufferedFrame,
                crate::geometry::WorldPose,
            )>,
                                   panner_frame_idx: &mut u64| {
                if let Some(mut frame) = buffer.pop() {
                    let panner_t0 = std::time::Instant::now();
                    let futures = buffer.future_world_states();
                    let pose = session.core.decide_pose_with_lookahead(
                        &frame.world_state,
                        &futures,
                        *panner_frame_idx,
                        frame.elapsed_ms,
                    );
                    *panner_frame_idx += 1;
                    let panner_time = panner_t0.elapsed();
                    frame.timing.tracking =
                        Some(frame.timing.tracking.unwrap_or_default() + panner_time);
                    frame.timing.total = Some(frame.timing.total.unwrap_or_default() + panner_time);
                    pose_queue.push_back((frame, pose));
                    true
                } else {
                    false
                }
            };

            // ── Panner warm-up: run panner post_smooth_half frames ahead ──
            let mut eof = buffer.len() < n;
            for _ in 0..post_smooth_half {
                if buffer.is_empty()
                    || !continue_run(&mut completion, self.frame_count, frame_limit, interrupted)
                {
                    break;
                }
                if !eof
                    && !buffer.is_full()
                    && !produce_one(
                        self,
                        source,
                        &mut buffer,
                        &mut image_budget,
                        &mut produce_count,
                        start,
                        &mut completion,
                    )?
                {
                    eof = true;
                }
                if !continue_run(&mut completion, self.frame_count, frame_limit, interrupted) {
                    break;
                }
                run_panner_once(self, &mut buffer, &mut pose_queue, &mut panner_frame_idx);
            }

            // ── Steady state: produce, run panner ahead, render with centered smooth ──
            while !(eof && buffer.is_empty() && pose_queue.is_empty())
                && continue_run(&mut completion, self.frame_count, frame_limit, interrupted)
            {
                // Produce: decode + detect next frame
                if !eof
                    && !buffer.is_full()
                    && !produce_one(
                        self,
                        source,
                        &mut buffer,
                        &mut image_budget,
                        &mut produce_count,
                        start,
                        &mut completion,
                    )?
                {
                    eof = true;
                }

                // Run panner on next buffered frame (stays ahead of rendering)
                if !continue_run(&mut completion, self.frame_count, frame_limit, interrupted) {
                    break;
                }
                if !buffer.is_empty() {
                    run_panner_once(self, &mut buffer, &mut pose_queue, &mut panner_frame_idx);
                }

                // Render: consume from pose queue when we have enough context
                if pose_queue.len() > post_smooth_half || (eof && !pose_queue.is_empty()) {
                    let (mut oldest, raw_pose) = pose_queue.pop_front().unwrap();
                    let smooth_t0 = std::time::Instant::now();

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

                    let smooth_time = smooth_t0.elapsed();
                    oldest.timing.tracking =
                        Some(oldest.timing.tracking.unwrap_or_default() + smooth_time);
                    oldest.timing.total =
                        Some(oldest.timing.total.unwrap_or_default() + smooth_time);
                    let retained_bytes = oldest.retained_bytes;
                    let rendered =
                        self.render_buffered_frame(oldest, smoothed_pose, start, &ctx, on_progress);
                    image_budget.release(retained_bytes);
                    rendered?;
                }
            }

            Ok(super::types::RunResult {
                frames_processed: self.frame_count,
                completion,
            })
        })();
        #[cfg(feature = "gpu")]
        while let Some(frame) = buffer.pop() {
            if let Some(slot) = frame.vram_slot {
                self.gpu_exec().release_pool_slot(slot);
            }
        }
        #[cfg(feature = "gpu")]
        for (frame, _) in pose_queue {
            if let Some(slot) = frame.vram_slot {
                self.gpu_exec().release_pool_slot(slot);
            }
        }
        result
    }

    /// Standard frame loop. Handles CPU-resident and GPU-resident
    /// frames transparently via [`process_frame_any`](Self::process_frame_any).
    fn run_immediate(
        &mut self,
        source: &mut dyn FrameSource,
        frame_limit: u64,
        interrupted: &AtomicBool,
        on_progress: &mut Option<ProgressCallback>,
    ) -> Result<super::types::RunResult, SessionError> {
        let mut completion = super::types::RunCompletion::Completed;
        let start = std::time::Instant::now();
        let mut previous_time = None;

        let mut previous_capture_times = None;
        let mut ctx = crate::session::types::FrameLoopContext {
            tracking_complete: false,
            capture: None,
            frame_seconds: frame_seconds(source.info().fps)?,
            #[cfg(all(feature = "gpu", target_os = "linux"))]
            gpu_buf_info: self.core.executor.gpu().and_then(|g| g.cuda_buf_info()),
        };

        while continue_run(&mut completion, self.frame_count, frame_limit, interrupted) {
            let frame_t0 = std::time::Instant::now();

            let frame = {
                crate::profile_scope!("wait_decode");
                match source.next_frame()? {
                    Some(f) => f,
                    None => {
                        if interrupted.load(Ordering::Relaxed) {
                            completion = super::types::RunCompletion::Cancelled;
                        }
                        break;
                    }
                }
            };
            let decode_time = frame_t0.elapsed();

            let SourceFrame {
                images: frame,
                pts_us,
                capture,
            } = frame;
            accept_source_time(&mut previous_time, pts_us)?;
            accept_capture_timing(
                &mut previous_capture_times,
                capture.as_ref(),
                self.core.calibration().projection.camera_count(),
            )?;
            ctx.capture = capture;
            self.validate_source_frame(&frame)?;

            if let Some(sink) = self.core.event_sink.as_deref_mut() {
                sink.emit(crate::detect::pipeline_event::PipelineEvent::FrameStart {
                    frame_index: self.frame_count,
                    timestamp_ms: start.elapsed().as_secs_f64() * 1000.0,
                });
            }

            self.process_frame_any(
                Some(&frame),
                start.elapsed(),
                pts_us,
                crate::telemetry::FrameTiming {
                    decode: Some(decode_time),
                    ..Default::default()
                },
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

        Ok(super::types::RunResult {
            frames_processed: self.frame_count,
            completion,
        })
    }
}

// Record cancellation where it stops processing, not after finalization when
// a late UI request could relabel an already completed run.
fn continue_run(
    completion: &mut super::types::RunCompletion,
    frames: u64,
    limit: u64,
    interrupted: &AtomicBool,
) -> bool {
    use super::types::RunCompletion;
    if frames >= limit || *completion == RunCompletion::Cancelled {
        return false;
    }
    if interrupted.load(Ordering::Relaxed) {
        *completion = RunCompletion::Cancelled;
        return false;
    }
    true
}

#[cfg(test)]
mod frame_rate_tests {
    use super::{centered_smooth, frame_seconds};
    use crate::geometry::WorldPose;

    #[test]
    fn centered_yaw_average_uses_the_local_angular_branch() {
        for sign in [-1.0, 1.0] {
            let raw = WorldPose {
                yaw: sign * 179.0_f32.to_radians(),
                pitch: 0.2,
                fov_degrees: 60.0,
            };
            for turns in [-2.0, 0.0, 3.0] {
                let neighbor = WorldPose {
                    yaw: -raw.yaw + turns * std::f32::consts::TAU,
                    pitch: 0.4,
                    fov_degrees: 80.0,
                };
                let result = centered_smooth(raw, [neighbor].into_iter(), [].into_iter());
                assert!(result.yaw.cos() < -0.9999);
                assert!((result.pitch - 0.3).abs() < 1e-6);
                assert_eq!(result.fov_degrees, 70.0);
            }
        }
    }

    #[test]
    fn session_clock_rejects_unknown_or_unrepresentable_rates() {
        for fps in [0.0, -30.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(frame_seconds(fps).is_err(), "fps={fps}");
        }
        assert_eq!(frame_seconds(30.0).unwrap(), 1.0 / 30.0);
    }
}
