//! macOS VideoToolbox/Metal zero-copy session methods.
//!
//! Separated from the main session module to isolate platform-specific
//! CVPixelBuffer import and Metal texture cache management.

use super::{FrameProgress, ProgressCallback, SessionError, StitchSession};

impl StitchSession {
    /// Run the zero-copy frame loop on macOS (VideoToolbox/Metal).
    ///
    /// Receives retained CVPixelBuffer pairs from decode threads,
    /// imports them as Metal textures, renders, and submits to the
    /// async encoder.
    ///
    /// Returns the number of frames processed. The caller must call
    /// [`Self::finish`] after this returns.
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

        while !interrupted.load(std::sync::atomic::Ordering::Relaxed)
            && self.frame_count < frame_limit
        {
            let pair = match pair_rx.recv() {
                Ok(p) => p,
                Err(_) => break,
            };

            let left_ptr = pair.left.as_ptr();
            let right_ptr = pair.right.as_ptr();

            // Import NV12 planes as Metal textures (zero-copy via IOSurface).
            // SAFETY: RetainedCVPixelBuffer guarantees the pointer is valid.
            let (left_y, left_uv) = unsafe { cache.import_nv12(left_ptr, self.pipeline.gpu())? };
            let (right_y, right_uv) = unsafe { cache.import_nv12(right_ptr, self.pipeline.gpu())? };

            // Run detection on GPU if a Metal detector is attached,
            // otherwise just update the director with empty state.
            if self.metal_detector.is_some() {
                let width = pair.left.width();
                let height = pair.left.height();
                self.detect_and_update_director_metal(
                    left_ptr,
                    right_ptr,
                    width,
                    height,
                    start.elapsed(),
                );
            } else {
                self.update_director(start.elapsed());
            }
            let pos = self.director_position();
            let render_buf = self.pipeline.render_imported_textures(
                &left_y.texture,
                &left_uv.texture,
                &right_y.texture,
                &right_uv.texture,
                pos.yaw,
                pos.pitch,
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
}
