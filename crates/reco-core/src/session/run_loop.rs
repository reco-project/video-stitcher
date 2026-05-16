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
