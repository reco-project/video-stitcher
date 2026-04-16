//! Video playback controller.
//!
//! Wraps an `FfmpegFileSource` with play/pause/step/seek state and
//! frame timing. Delivers YUV frame data on demand, paced by FPS.

use std::path::Path;
use std::time::{Duration, Instant};

use reco_core::source::{FrameSource, SourceError, SourceInfo, YuvData};
use reco_io::adapters::FfmpegFileSource;

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

/// Owned stereo YUV frame pair.
pub struct StereoYuv {
    pub left: YuvData,
    pub right: YuvData,
}

/// Controls video file playback for the GUI.
pub struct Playback {
    source: Option<FfmpegFileSource>,
    info: Option<SourceInfo>,
    state: PlayState,
    current_frame: Option<StereoYuv>,
    frame_index: u64,
    total_frames: Option<u64>,
    frame_duration: Duration,
    last_frame_time: Option<Instant>,
}

impl Playback {
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
            last_frame_time: None,
        }
    }

    /// Open a stereo video source.
    pub fn open(
        &mut self,
        left_path: &Path,
        right_path: &Path,
        sync_offset: i64,
    ) -> Result<(), SourceError> {
        let source = FfmpegFileSource::open_with_offset(left_path, right_path, sync_offset)?;
        let info = source.info();
        let fps = info.fps;
        self.total_frames = source.total_frames();
        self.frame_duration = if fps > 0.0 {
            Duration::from_secs_f64(1.0 / fps)
        } else {
            Duration::from_millis(33)
        };
        self.info = Some(info);
        self.source = Some(source);
        self.state = PlayState::Paused;
        self.frame_index = 0;
        self.current_frame = None;
        self.last_frame_time = None;

        // Decode the first frame so we have something to display.
        self.step_forward()?;
        Ok(())
    }

    /// Advance one frame. Returns `true` if a new frame is available.
    pub fn step_forward(&mut self) -> Result<bool, SourceError> {
        let source = match self.source.as_mut() {
            Some(s) => s,
            None => return Ok(false),
        };

        match source.next_frame()? {
            Some(stereo) => {
                let (left, right) = match stereo {
                    reco_core::source::StereoFrame::Yuv420p(pair) => (pair.left, pair.right),
                    _ => {
                        return Err(SourceError::Read {
                            reason: "GUI preview expects Yuv420p frames".into(),
                        });
                    }
                };
                self.current_frame = Some(StereoYuv { left, right });
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
    pub fn tick(&mut self) -> Result<bool, SourceError> {
        if self.state != PlayState::Playing {
            return Ok(false);
        }

        let now = Instant::now();
        let should_advance = match self.last_frame_time {
            Some(last) => now.duration_since(last) >= self.frame_duration,
            None => true,
        };

        if !should_advance {
            return Ok(false);
        }

        let source = match self.source.as_mut() {
            Some(s) => s,
            None => return Ok(false),
        };

        // Non-blocking: returns None if no frame decoded yet.
        match source.try_next_frame()? {
            Some(stereo) => {
                let (left, right) = match stereo {
                    reco_core::source::StereoFrame::Yuv420p(pair) => (pair.left, pair.right),
                    _ => {
                        return Err(SourceError::Read {
                            reason: "GUI preview expects Yuv420p frames".into(),
                        });
                    }
                };
                self.current_frame = Some(StereoYuv { left, right });
                self.frame_index += 1;
                self.last_frame_time = Some(now);
                Ok(true)
            }
            None => {
                // Could be "not ready yet" or "end of stream".
                // FfmpegFileSource returns Disconnected for EOF.
                // try_next_frame returns Ok(None) for both cases,
                // but after EOF the channel disconnects and stays None.
                // If we've been getting None for a while with no new
                // frames, assume EOF.
                if self.last_frame_time.is_some()
                    && now.duration_since(self.last_frame_time.unwrap()) > self.frame_duration * 30
                {
                    self.state = PlayState::Finished;
                }
                Ok(false)
            }
        }
    }

    /// Toggle play/pause. Returns the new state.
    pub fn toggle(&mut self) -> PlayState {
        match self.state {
            PlayState::Paused | PlayState::Finished => {
                self.state = PlayState::Playing;
                self.last_frame_time = None;
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
            self.step_forward()?;
        }
        Ok(())
    }

    pub fn state(&self) -> PlayState {
        self.state
    }

    pub fn current_frame(&self) -> Option<&StereoYuv> {
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
        self.info.as_ref().map(|i| (i.width, i.height))
    }

    #[allow(dead_code)]
    pub fn frame_duration(&self) -> Duration {
        self.frame_duration
    }
}
