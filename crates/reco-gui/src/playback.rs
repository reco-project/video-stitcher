//! Video playback controller.
//!
//! File timing and seeking, or non-blocking live frame selection.

use std::time::{Duration, Instant};

use reco_core::source::{FrameSet, FrameSource, SourceError, SourceFrame, SourceInfo};
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

struct PendingSeek {
    target: u64,
    ready: Option<FrameSet>,
}

enum Source {
    File(Box<FfmpegFileSource>),
    Live(Box<reco_io::rtsp::RtspSource>),
}

impl Source {
    fn frames(&self) -> &dyn FrameSource {
        match self {
            Self::File(source) => source.as_ref(),
            Self::Live(source) => source.as_ref(),
        }
    }

    fn frames_mut(&mut self) -> &mut dyn FrameSource {
        match self {
            Self::File(source) => source.as_mut(),
            Self::Live(source) => source.as_mut(),
        }
    }
}

/// Controls video file playback for the GUI.
///
/// Uses drift-free scheduling: frame advance timing is computed from a
/// wall-clock anchor set when playback starts, not from the last
/// successful advance. This prevents the cumulative lag that comes
/// from snapping the anchor to actual advance times (which are always
/// slightly late relative to the ideal frame boundary).
pub struct Playback {
    source: Option<Source>,
    info: Option<SourceInfo>,
    state: PlayState,
    current_frame: Option<FrameSet>,
    frame_index: u64,
    pending_seek: Option<PendingSeek>,
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
        self.source
            .as_ref()
            .is_some_and(|source| source.frames().is_full_range())
    }

    /// Create an empty playback controller (no source loaded).
    pub fn new() -> Self {
        Self {
            source: None,
            info: None,
            state: PlayState::Empty,
            current_frame: None,
            frame_index: 0,
            pending_seek: None,
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
        candidate.source = Some(Source::File(Box::new(source)));
        candidate.state = PlayState::Paused;

        // Keep the working source until the replacement can display a frame.
        if !candidate.advance_blocking()? {
            return Err(SourceError::Read {
                reason: "no frames available at the requested sync offset".into(),
            });
        }
        *self = candidate;
        Ok(())
    }

    /// Adopt an opened live source and its first selected image without waiting on network I/O.
    pub fn from_live(
        mut source: reco_io::rtsp::RtspSource,
        first: SourceFrame,
        cleanup: &reco_io::DecoderCleanup,
    ) -> Self {
        source.defer_cleanup(cleanup);
        Self {
            info: Some(source.info()),
            source: Some(Source::Live(Box::new(source))),
            state: PlayState::Playing,
            current_frame: Some(first.images),
            frame_index: 1,
            ..Self::new()
        }
    }

    /// Whether selection follows a live source clock instead of a seekable file timeline.
    pub fn is_live(&self) -> bool {
        matches!(self.source, Some(Source::Live(_)))
    }

    /// Blocking read for initialization/restoration, completing a pending seek if present.
    /// Returns `true` if a new frame is available; pauses on a source error.
    pub fn advance_blocking(&mut self) -> Result<bool, SourceError> {
        self.advance(true)
    }

    fn advance(&mut self, blocking: bool) -> Result<bool, SourceError> {
        let ready = self
            .pending_seek
            .as_mut()
            .and_then(|seek| seek.ready.take());
        let frame = match ready {
            Some(frame) => Some(frame),
            None => self.read_frame(blocking)?,
        };
        let Some(frame) = frame else { return Ok(false) };
        self.current_frame = Some(frame);
        self.frame_index = self
            .pending_seek
            .take()
            .map_or(self.frame_index + 1, |seek| {
                self.playback_anchor = None;
                seek.target + 1
            });
        Ok(true)
    }

    fn read_frame(&mut self, blocking: bool) -> Result<Option<FrameSet>, SourceError> {
        let source = match self.source.as_mut() {
            Some(s) => s.frames_mut(),
            None => return Ok(None),
        };

        let result = if blocking {
            source.next_frame()
        } else {
            source.try_next_frame()
        };
        match result.inspect_err(|_| {
            self.state = PlayState::Paused;
            self.pending_seek = None;
            self.playback_anchor = None;
        })? {
            Some(frame) => Ok(Some(frame.images)),
            None => {
                if source.is_exhausted() {
                    if self.pending_seek.take().is_some() {
                        self.state = PlayState::Paused;
                        self.playback_anchor = None;
                        return Err(SourceError::Read {
                            reason: "no frame available at the requested position".into(),
                        });
                    }
                    self.state = PlayState::Finished;
                }
                Ok(None)
            }
        }
    }

    /// Prepare at most one pending-seek frame set without changing the image selected for rendering.
    pub fn poll_seek(&mut self) -> Result<bool, SourceError> {
        if self
            .pending_seek
            .as_ref()
            .is_none_or(|seek| seek.ready.is_some())
        {
            return Ok(false);
        }
        let Some(frame) = self.read_frame(false)? else {
            return Ok(false);
        };
        self.pending_seek.as_mut().unwrap().ready = Some(frame);
        Ok(true)
    }

    /// Non-blocking playback or prepared-seek advance for the render callback.
    ///
    /// Uses `try_next_frame()` to avoid blocking the UI thread on decode.
    /// Returns `true` if a new frame was consumed; pauses on a source error.
    ///
    /// Timing is drift-free: the ideal frame index is computed from the
    /// wall-clock elapsed time since playback started, so any single
    /// late tick is caught up on the next tick rather than compounding.
    pub fn tick(&mut self) -> Result<bool, SourceError> {
        if let Some(seek) = &self.pending_seek {
            return if seek.ready.is_some() {
                self.advance(false)
            } else {
                Ok(false)
            };
        }
        if self.state != PlayState::Playing {
            return Ok(false);
        }

        if self.is_live() {
            return self.advance(false);
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

        self.advance(false)
    }

    /// Pause active playback without discarding the displayed frame.
    pub fn pause(&mut self) {
        if self.state == PlayState::Playing {
            self.state = PlayState::Paused;
        }
    }

    /// Toggle play/pause and return the new state.
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

    /// Request a zero-based source frame; [`Self::poll_seek`] prepares it and [`Self::tick`] accepts it during rendering, including while paused.
    /// The current image and its index remain unchanged until that acceptance.
    pub fn seek_frame(&mut self, target: u64) -> Result<(), SourceError> {
        let target = self
            .total_frames
            .map_or(target, |total| target.min(total.saturating_sub(1)));
        if let Some(source) = self.source.as_mut() {
            if self
                .pending_seek
                .as_ref()
                .is_some_and(|seek| seek.target == target)
            {
                return Ok(());
            }
            source.frames_mut().seek(target)?;
            self.pending_seek = Some(PendingSeek {
                target,
                ready: None,
            });
            if self.state == PlayState::Finished {
                self.state = PlayState::Paused;
            }
        }
        Ok(())
    }

    /// Whether a requested frame is awaiting acceptance by the render callback.
    pub fn is_seeking(&self) -> bool {
        self.pending_seek.is_some()
    }

    /// Zero-based target of accumulated seek input, or the displayed frame when idle.
    pub fn requested_frame(&self) -> u64 {
        self.pending_seek
            .as_ref()
            .map_or_else(|| self.frame_index.saturating_sub(1), |seek| seek.target)
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
    use slint::ComponentHandle;
    use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
    use std::{cell::RefCell, rc::Rc};

    fn finish_seek(playback: &mut Playback) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while playback.is_seeking() {
            assert!(Instant::now() < deadline, "seek did not complete");
            playback.poll_seek().unwrap();
            playback.tick().unwrap();
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    #[ignore = "requires RECO_PLAYBACK_TEST_VIDEO pointing to a video with at least 30 frames"]
    fn pending_seeks_preserve_displayed_pixels_and_accept_only_the_latest_target() {
        let input = InputPath::Single(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap().into());
        let cleanup = reco_io::DecoderCleanup::default();
        let mut playback = Playback::new();
        playback
            .open(std::slice::from_ref(&input), 0, &cleanup)
            .unwrap();
        let FrameSet::Yuv420p(before) = playback.current_frame().unwrap() else {
            panic!("YUV420P")
        };
        let before = before[0].clone();
        for target in [20, 8, 14] {
            playback.seek_frame(target).unwrap();
            assert_eq!(playback.frame_index(), 1);
            assert_eq!(playback.requested_frame(), target);
            assert!(playback.is_seeking());
            assert_eq!(playback.state(), PlayState::Paused);
            let FrameSet::Yuv420p(current) = playback.current_frame().unwrap() else {
                panic!("YUV420P")
            };
            assert!(std::sync::Arc::ptr_eq(&before, &current[0]));
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while !playback.poll_seek().unwrap() {
            assert!(Instant::now() < deadline, "seek did not become ready");
            assert_eq!(playback.frame_index(), 1);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(playback.frame_index(), 1, "decoding is not presentation");
        let FrameSet::Yuv420p(current) = playback.current_frame().unwrap() else {
            panic!("YUV420P")
        };
        assert!(std::sync::Arc::ptr_eq(&before, &current[0]));
        assert!(
            !playback.poll_seek().unwrap(),
            "hold only one ready replacement"
        );
        playback.seek_frame(8).unwrap();
        assert!(
            !playback.tick().unwrap(),
            "do not present the superseded ready frame"
        );
        assert_eq!(playback.frame_index(), 1);
        finish_seek(&mut playback);
        assert_eq!(playback.frame_index(), 9);
        playback.seek_frame(14).unwrap();
        finish_seek(&mut playback);
        assert_eq!(playback.frame_index(), 15);
        let mut oracle = FfmpegFileSource::open_inputs(&[input], 0, false).unwrap();
        oracle.seek(14).unwrap();
        let FrameSet::Yuv420p(expected) = oracle.next_frame().unwrap().unwrap().images else {
            panic!("YUV420P")
        };
        let FrameSet::Yuv420p(actual) = playback.current_frame().unwrap() else {
            panic!("YUV420P")
        };
        assert_eq!(actual[0].y, expected[0].y);
        assert_eq!(actual[0].u, expected[0].u);
        assert_eq!(actual[0].v, expected[0].v);

        playback.toggle();
        playback.playback_anchor = Some((Instant::now() - Duration::from_secs(60), 0));
        playback.seek_frame(3).unwrap();
        finish_seek(&mut playback);
        assert_eq!(playback.frame_index(), 4);
        assert_eq!(playback.state(), PlayState::Playing);
        assert!(
            playback.playback_anchor.is_none(),
            "waiting must not create catch-up debt"
        );
        playback.pause();
        playback.seek_frame(3).unwrap();
        finish_seek(&mut playback);

        let old_index = playback.frame_index();
        playback.total_frames = None;
        playback.seek_frame(100_000).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(Instant::now() < deadline, "seek past EOF did not finish");
            if let Err(error) = playback.poll_seek() {
                assert!(error.to_string().contains("requested position"));
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(playback.state(), PlayState::Paused);
        assert!(!playback.is_seeking());
        assert_eq!(
            playback.frame_index(),
            old_index,
            "EOF must not relabel the retained image"
        );
        // The retained image no longer describes the failed source cursor.
        // Selecting it must restore decoding, not treat it as a no-op.
        playback.seek_frame(old_index - 1).unwrap();
        finish_seek(&mut playback);
        assert_eq!(playback.frame_index(), old_index);
        assert!(playback.advance_blocking().unwrap());
        assert_eq!(playback.frame_index(), old_index + 1);
    }

    #[test]
    #[ignore = "requires RECO_PLAYBACK_TEST_VIDEO pointing to a video with at least 30 frames"]
    fn interleaved_seek_readiness_and_presentation_keep_exact_camera_order_and_pixels() {
        use std::hash::{Hash, Hasher};
        fn fingerprint(frame: &FrameSet) -> u64 {
            let FrameSet::Yuv420p(cameras) = frame else {
                panic!("YUV420P")
            };
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            for camera in cameras {
                camera.size.hash(&mut hash);
                camera.y.hash(&mut hash);
                camera.u.hash(&mut hash);
                camera.v.hash(&mut hash);
            }
            hash.finish()
        }

        let input = InputPath::Single(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap().into());
        for (cameras, offset) in [(1, 0), (2, -3), (2, 3), (3, 0)] {
            let inputs = vec![input.clone(); cameras];
            let mut oracle = FfmpegFileSource::open_inputs(&inputs, offset, false).unwrap();
            let expected: Vec<_> = (0..24)
                .map(|_| fingerprint(&oracle.next_frame().unwrap().unwrap().images))
                .collect();
            drop(oracle);
            let cleanup = reco_io::DecoderCleanup::default();
            let mut playback = Playback::new();
            playback.open(&inputs, offset, &cleanup).unwrap();
            let mut sequence = 0x54321u64;
            for cycle in 0..24 {
                for _ in 0..5 {
                    sequence = sequence.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let target = (sequence >> 32) % expected.len() as u64;
                    playback.seek_frame(target).unwrap();
                    let before = fingerprint(playback.current_frame().unwrap());
                    let index = playback.frame_index();
                    playback.poll_seek().unwrap();
                    assert_eq!(playback.frame_index(), index);
                    assert_eq!(fingerprint(playback.current_frame().unwrap()), before);
                    if sequence & 1 == 0 && playback.tick().unwrap() {
                        assert_eq!(playback.frame_index(), target + 1);
                        assert_eq!(
                            fingerprint(playback.current_frame().unwrap()),
                            expected[target as usize]
                        );
                    }
                }
                let target = playback.requested_frame();
                finish_seek(&mut playback);
                assert_eq!(playback.frame_index(), target + 1, "cycle {cycle}");
                assert_eq!(
                    fingerprint(playback.current_frame().unwrap()),
                    expected[target as usize]
                );
                assert_eq!(playback.state(), PlayState::Paused);
            }
        }
    }

    struct HeadlessPlatform;
    impl slint::platform::Platform for HeadlessPlatform {
        fn create_window_adapter(
            &self,
        ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
            Ok(MinimalSoftwareWindow::new(RepaintBufferType::ReusedBuffer))
        }
    }

    #[test]
    fn end_of_playback_stops_ui_redraws_without_a_new_image() {
        slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
        let app = crate::RecoApp::new().unwrap();
        app.set_playing(true);
        let mut state = crate::AppState::new(crate::settings::GuiSettings::default());
        state.playback.state = PlayState::Finished;
        state.preview_dirty = false;
        let state = Rc::new(RefCell::new(state));
        assert!(!crate::vsync_render_tick(&state, &app.as_weak()));
        assert!(!app.get_playing());
        assert_eq!(app.get_status_text(), "Playback finished");
        app.set_status_text("Updated calibration".into());
        assert!(!crate::vsync_render_tick(&state, &app.as_weak()));
        assert_eq!(app.get_status_text(), "Updated calibration");
    }

    #[test]
    #[ignore = "requires RECO_PLAYBACK_TEST_VIDEO pointing to a short, multi-frame video"]
    fn decoded_eof_stops_ui_playback_and_preserves_the_last_image() {
        slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
        let app = crate::RecoApp::new().unwrap();
        let mut state = crate::AppState::new(crate::settings::GuiSettings::default());
        let path = std::path::PathBuf::from(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap());
        state
            .playback
            .open(&[InputPath::Single(path)], 0, &state.decoder_cleanup)
            .unwrap();
        assert_eq!(state.playback.toggle(), PlayState::Playing);
        app.set_playing(true);
        state.preview_dirty = false;
        // Make all frames due, exercising decode and EOF without waiting for their presentation times.
        state.playback.playback_anchor = Some((Instant::now() - Duration::from_secs(60), 0));
        let state = Rc::new(RefCell::new(state));
        let deadline = Instant::now() + Duration::from_secs(10);
        while app.get_playing() {
            assert!(Instant::now() < deadline, "playback did not reach EOF");
            crate::vsync_render_tick(&state, &app.as_weak());
            std::thread::sleep(Duration::from_millis(1));
        }
        let state = state.borrow();
        assert_eq!(state.playback.state(), PlayState::Finished);
        assert!(state.playback.frame_index() > 1);
        assert!(state.playback.current_frame().is_some());
        assert_eq!(app.get_status_text(), "Playback finished");
    }

    #[test]
    #[ignore = "requires RECO_PLAYBACK_TEST_VIDEO pointing to a short, multi-frame video"]
    fn decoded_error_pauses_ui_playback_and_preserves_the_last_image() {
        slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
        let app = crate::RecoApp::new().unwrap();
        let mut state = crate::AppState::new(crate::settings::GuiSettings::default());
        let path = std::path::PathBuf::from(std::env::var("RECO_PLAYBACK_TEST_VIDEO").unwrap());
        let missing = path.with_file_name("missing-playback-error-segment.mkv");
        assert!(!missing.exists());
        state
            .playback
            .open(
                &[InputPath::Chained(vec![path, missing])],
                0,
                &state.decoder_cleanup,
            )
            .unwrap();
        assert_eq!(state.playback.toggle(), PlayState::Playing);
        app.set_playing(true);
        state.preview_dirty = false;
        state.playback.playback_anchor = Some((Instant::now() - Duration::from_secs(60), 0));
        let state = Rc::new(RefCell::new(state));
        let deadline = Instant::now() + Duration::from_secs(10);
        while !app.get_status_text().starts_with("Error:") {
            assert!(
                Instant::now() < deadline,
                "missing segment did not report an error"
            );
            crate::vsync_render_tick(&state, &app.as_weak());
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!app.get_playing());
        assert_eq!(state.borrow().playback.state(), PlayState::Paused);
        let frame = state.borrow().playback.frame_index();
        assert!(frame > 1);
        let status = app.get_status_text();
        for _ in 0..5 {
            assert!(!crate::vsync_render_tick(&state, &app.as_weak()));
        }
        assert_eq!(state.borrow().playback.frame_index(), frame);
        assert!(state.borrow().playback.current_frame().is_some());
        assert_eq!(app.get_status_text(), status);
    }

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
            let Some(Source::File(source)) = &playback.source else {
                panic!("file source")
            };
            assert_eq!(
                source.decode_backend(),
                reco_io::ffmpeg::decoder::DecodeBackend::Cuda
            );
        }
        let total_frames = playback.total_frames().unwrap();
        for target in [
            total_frames * 6 / 10,
            total_frames / 10,
            total_frames * 8 / 10,
            0,
        ] {
            playback.seek_frame(target).unwrap();
            finish_seek(&mut playback);
            assert!(playback.current_frame().is_some());
            assert_eq!(playback.frame_index(), target + 1);
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
        assert!(playback.advance_blocking().unwrap());
        playback
            .open(&[input.clone(), input.clone()], 1, &cleanup)
            .unwrap();
        assert_eq!(playback.frame_index(), 1);
        assert_eq!(playback.state(), PlayState::Paused);
        playback.open(&[input], 0, &cleanup).unwrap();
        playback
            .seek_frame(playback.total_frames().unwrap() / 2)
            .unwrap();
        assert!(playback.advance_blocking().unwrap());
        let FrameSet::Yuv420p(cameras) = playback.current_frame().unwrap() else {
            panic!("expected YUV420P")
        };
        assert_eq!(cameras.len(), 1);
    }
}
