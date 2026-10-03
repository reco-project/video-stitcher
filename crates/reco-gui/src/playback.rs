//! Video playback controller.
//!
//! Wraps an `FfmpegFileSource` with play/pause/step/seek state and
//! frame timing. Delivers YUV frame data on demand, paced by FPS.

use std::time::{Duration, Instant};

use reco_core::source::{FrameSet, FrameSource, SourceError, SourceInfo};
use reco_io::adapters::FfmpegFileSource;
use reco_io::stitch_job::InputPath;

/// Playback state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    /// No source loaded.
    Empty,
    /// Paused on a frame.
    Paused,
    /// Playing at source FPS.
    Playing,
    /// Reached end of file.
    Finished,
}

/// Controls video file playback for the GUI.
///
/// Uses drift-free scheduling: frame advance timing is computed from a
/// wall-clock anchor set when playback starts, not from the last
/// successful advance. This prevents the cumulative lag that comes
/// from snapping the anchor to actual advance times (which are always
/// slightly late relative to the ideal frame boundary).
pub struct Playback {
    source: Option<FfmpegFileSource>,
    info: Option<SourceInfo>,
    state: PlayState,
    current_frame: Option<FrameSet>,
    frame_index: u64,
    total_frames: Option<u64>,
    frame_duration: Duration,
    /// Wall-clock time when playback started (or resumed) and the
    /// frame index at that moment. Used to compute the ideal target
    /// frame index without drift.
    playback_anchor: Option<(Instant, u64)>,
}

impl Playback {
    /// Range of the decoded source planes.
    pub fn is_full_range(&self) -> bool {
        self.source.as_ref().is_some_and(FrameSource::is_full_range)
    }

    /// Create an empty playback controller (no source loaded).
    pub fn new() -> Self {
        Self {
            source: None,
            info: None,
            state: PlayState::Empty,
            current_frame: None,
            frame_index: 0,
            total_frames: None,
            frame_duration: Duration::from_millis(33), // ~30fps default
            playback_anchor: None,
        }
    }

    /// Open the calibrated camera inputs.
    ///
    /// Takes [`InputPath`] so multi-segment (concat) inputs span every file
    /// in the timeline - total frames, seeking, and the duration range cover
    /// the whole chain, not just the first segment.
    pub fn open(
        &mut self,
        inputs: &[InputPath],
        sync_offset: i64,
        cleanup: &reco_io::DecoderCleanup,
    ) -> Result<(), SourceError> {
        let mut source = FfmpegFileSource::open_inputs(inputs, sync_offset, false)?;
        source.defer_cleanup(cleanup);
        let info = source.info();
        let fps = info.fps;
        if info.fps_rational.is_none() {
            return Err(SourceError::Init {
                path: inputs[0].first_path().display().to_string(),
                reason: "video frame rate is unknown; timed playback is unavailable".into(),
            });
        }
        let mut candidate = Self::new();
        candidate.total_frames = source.total_frames();
        candidate.frame_duration = Duration::from_secs_f64(1.0 / fps);
        candidate.info = Some(info);
        candidate.source = Some(source);
        candidate.state = PlayState::Paused;

        // Keep the working source until the replacement can display a frame.
        if !candidate.step_forward()? {
            return Err(SourceError::Read {
                reason: "no frames available at the requested sync offset".into(),
            });
        }
        *self = candidate;
        Ok(())
    }

    /// Advance one frame. Returns `true` if a new frame is available.
    pub fn step_forward(&mut self) -> Result<bool, SourceError> {
        let source = match self.source.as_mut() {
            Some(s) => s,
            None => return Ok(false),
        };

        match source.next_frame()? {
            Some(frame) => {
                self.current_frame = Some(frame.images);
                self.frame_index += 1;
                Ok(true)
            }
            None => {
                self.state = PlayState::Finished;
                Ok(false)
            }
        }
    }

    /// Non-blocking frame advance for the GUI timer.
    ///
    /// Uses `try_next_frame()` to avoid blocking the UI thread on decode.
    /// Returns `true` if a new frame was consumed.
    ///
    /// Timing is drift-free: the ideal frame index is computed from the
    /// wall-clock elapsed time since playback started, so any single
    /// late tick is caught up on the next tick rather than compounding.
    pub fn tick(&mut self) -> Result<bool, SourceError> {
        if self.state != PlayState::Playing {
            return Ok(false);
        }

        let now = Instant::now();

        // Anchor on the first tick after play/resume/seek.
        let (start, start_frame) = match self.playback_anchor {
            Some(a) => a,
            None => {
                self.playback_anchor = Some((now, self.frame_index));
                (now, self.frame_index)
            }
        };

        // Drift-free target: where the playhead SHOULD be based on wall
        // clock, not based on when the last advance happened.
        let elapsed = now.duration_since(start);
        let frames_since_start = (elapsed.as_secs_f64() / self.frame_duration.as_secs_f64()) as u64;
        let target_frame = start_frame + frames_since_start;

        if self.frame_index >= target_frame {
            // On schedule or ahead — no advance needed this tick.
            return Ok(false);
        }

        let source = match self.source.as_mut() {
            Some(s) => s,
            None => return Ok(false),
        };

        // Non-blocking: returns None if no frame decoded yet.
        match source.try_next_frame()? {
            Some(frame) => {
                self.current_frame = Some(frame.images);
                self.frame_index += 1;
                Ok(true)
            }
            None => {
                // Could be "not ready yet" or "end of stream".
                // `FrameSource::is_exhausted()` answers unambiguously once
                // the decoder channel has disconnected, so no timeout
                // heuristic is needed.
                if source.is_exhausted() {
                    self.state = PlayState::Finished;
                }
                Ok(false)
            }
        }
    }

    /// Toggle play/pause. Returns the new state.
    pub fn pause(&mut self) {
        if self.state == PlayState::Playing {
            self.state = PlayState::Paused;
        }
    }

    pub fn toggle(&mut self) -> PlayState {
        match self.state {
            PlayState::Paused | PlayState::Finished => {
                self.state = PlayState::Playing;
                self.playback_anchor = None;
            }
            PlayState::Playing => {
                self.state = PlayState::Paused;
            }
            PlayState::Empty => {}
        }
        self.state
    }

    /// Seek to a normalized position (0.0 to 1.0).
    pub fn seek(&mut self, fraction: f32) -> Result<(), SourceError> {
        let total = match self.total_frames {
            Some(t) if t > 0 => t,
            _ => return Ok(()),
        };
        let target = ((fraction as f64) * total as f64) as u64;
        let target = target.min(total.saturating_sub(1));

        if let Some(source) = self.source.as_mut() {
            source.seek(target)?;
            self.frame_index = target;
            // Reset pacing anchor so playback resumes cleanly from the
            // new position without trying to "catch up" N seconds of
            // skipped frames.
            self.playback_anchor = None;
            self.step_forward()?;
        }
        Ok(())
    }

    pub fn state(&self) -> PlayState {
        self.state
    }

    pub fn current_frame(&self) -> Option<&FrameSet> {
        self.current_frame.as_ref()
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn total_frames(&self) -> Option<u64> {
        self.total_frames
    }

    pub fn fps(&self) -> f64 {
        self.info.as_ref().map_or(0.0, |i| i.fps)
    }

    pub fn input_dimensions(&self) -> Option<(u32, u32)> {
        self.source_sizes()?.first().copied()
    }

    pub fn source_sizes(&self) -> Option<&[(u32, u32)]> {
        self.info.as_ref().map(|i| i.camera_sizes.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires RECO_PLAYBACK_TEST_VIDEO pointing to a short, multi-frame video"]
    fn replacement_process_exits_without_cuda_cleanup_errors() {
        for _ in 0..3 {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "playback::tests::file_replacement_preserves_playback_on_first_frame_failure",
                    "--ignored",
                    "--test-threads=1",
                ])
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success(), "{stderr}");
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("1 passed;"),
                "child did not run the playback test"
            );
            assert!(
                !stderr.lines().any(|line| {
                    (line.contains("cuCtx") || line.contains("cuMem")) && line.contains("failed")
                }),
                "decoder cleanup failed at process exit: {stderr}"
            );
        }
    }

    #[test]
    #[ignore = "requires RECO_PLAYBACK_TEST_VIDEO pointing to a short, multi-frame video"]
    fn file_replacement_preserves_playback_on_first_frame_failure() {
        fn first_y(playback: &Playback) -> &[u8] {
            let FrameSet::Yuv420p(cameras) = playback.current_frame().unwrap() else {
                panic!("expected YUV420P")
            };
            &cameras[0].y
        }
        let path = std::path::PathBuf::from(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap());
        let input = InputPath::Single(path.clone());
        let missing = path.with_file_name("missing-playback-test-input.mp4");
        assert!(!missing.exists());
        let cleanup = reco_io::DecoderCleanup::default();
        let mut playback = Playback::new();
        playback
            .open(&[input.clone(), input.clone()], 0, &cleanup)
            .unwrap();
        if std::env::var_os("RECO_REQUIRE_CUDA").is_some() {
            assert_eq!(
                playback.source.as_ref().unwrap().decode_backend(),
                reco_io::ffmpeg::decoder::DecodeBackend::Cuda
            );
        }
        for fraction in [0.6, 0.1, 0.8, 0.0] {
            playback.seek(fraction).unwrap();
            assert!(playback.current_frame().is_some());
        }
        let before = first_y(&playback).to_vec();
        let index = playback.frame_index();
        let total = playback.total_frames();
        let past_end = (total.unwrap() * 2 + 1) as i64;
        // Both first segments probe successfully. Advancing the right input
        // into a missing segment fails inside the decode worker.
        let broken = InputPath::Chained(vec![path, missing]);
        let error = playback
            .open(&[input.clone(), broken], past_end, &cleanup)
            .unwrap_err();
        assert!(matches!(error, SourceError::Read { .. }), "{error}");
        assert_eq!(playback.frame_index(), index);
        assert_eq!(playback.total_frames(), total);
        assert_eq!(first_y(&playback), before);
        assert!(
            playback
                .open(&[input.clone(), input.clone()], past_end, &cleanup)
                .is_err()
        );
        assert_eq!(first_y(&playback), before);
        assert!(playback.step_forward().unwrap());
        playback
            .open(&[input.clone(), input.clone()], 1, &cleanup)
            .unwrap();
        assert_eq!(playback.frame_index(), 1);
        assert_eq!(playback.state(), PlayState::Paused);
        playback.open(&[input], 0, &cleanup).unwrap();
        playback.seek(0.5).unwrap();
        assert!(playback.step_forward().unwrap());
        let FrameSet::Yuv420p(cameras) = playback.current_frame().unwrap() else {
            panic!("expected YUV420P")
        };
        assert_eq!(cameras.len(), 1);
    }
}
