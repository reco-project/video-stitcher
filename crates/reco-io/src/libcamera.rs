//! Live stereo YUV420P capture from `rpicam-vid` stdout.
//!
//! Each child has a bounded frame queue and an owned reader thread. Stereo
//! pairing follows arrival order, not capture timestamps. A single camera
//! index duplicates each frame for the two-input consumer.

use reco_core::source::{FrameSet, SourceError, SourceFrame, SourceInfo, Yuv420pData};
use std::io::{self, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

/// libcamera capture configuration.
#[derive(Debug, Clone)]
pub struct LibcameraConfig {
    /// Capture width in pixels, positive and even.
    pub width: u32,
    /// Capture height in pixels, positive and even.
    pub height: u32,
    /// Positive capture frame rate.
    pub fps: u32,
    /// Left camera index (0, 1, ...).
    pub left_camera: u32,
    /// Right camera index. Equal indices duplicate a single camera's frames.
    pub right_camera: u32,
}

/// Stereo camera source using `rpicam-vid` (installed separately).
///
/// Reads raw YUV420P from one or two child processes. Drop stops and reaps
/// the children and joins their readers, including after partial startup.
pub struct LibcameraCameraSource {
    cameras: Vec<Camera>,
    info: SourceInfo,
    interrupted: Arc<AtomicBool>,
    stopped: bool,
}

struct Camera {
    child: Child,
    frames: Option<mpsc::Receiver<Yuv420pData>>,
    reader: Option<JoinHandle<io::Result<()>>>,
    label: String,
    eof: bool,
}

impl Camera {
    fn start(mut command: Command, label: String, size: (u32, u32)) -> Result<Self, SourceError> {
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| SourceError::Init {
                path: label.clone(),
                reason: format!("start rpicam-vid: {e}"),
            })?;
        let mut camera = Self {
            child,
            frames: None,
            reader: None,
            label,
            eof: false,
        };
        let stdout = camera
            .child
            .stdout
            .take()
            .ok_or_else(|| SourceError::Init {
                path: camera.label.clone(),
                reason: "capture stdout unavailable".into(),
            })?;
        let (tx, rx) = mpsc::sync_channel(2);
        camera.frames = Some(rx);
        camera.reader = Some(
            std::thread::Builder::new()
                .name(format!("libcam_{}", camera.label))
                .spawn(move || read_frames(stdout, size, tx))
                .map_err(|e| SourceError::Init {
                    path: camera.label.clone(),
                    reason: format!("start capture reader: {e}"),
                })?,
        );
        Ok(camera)
    }

    fn error(&self, reason: impl std::fmt::Display) -> SourceError {
        SourceError::Read {
            reason: format!("{}: {reason}", self.label),
        }
    }

    // Check failures independently of frame queues so a silent peer cannot hide them.
    fn check(&mut self) -> Result<bool, SourceError> {
        if self.eof || self.reader.as_ref().is_some_and(JoinHandle::is_finished) {
            self.join_reader(false)?;
        }
        let status = self.child.try_wait().map_err(|e| self.error(e))?;
        if let Some(status) = status
            && !status.success()
        {
            return Err(self.error(format!("rpicam-vid exited with {status}")));
        }
        Ok(status.is_some())
    }

    fn join_reader(&mut self, stopping: bool) -> Result<(), SourceError> {
        if let Some(reader) = self.reader.take() {
            let result = reader
                .join()
                .map_err(|_| self.error("capture reader panicked"))?;
            // Stopping a child can interrupt an otherwise valid partial frame.
            if !stopping {
                result.map_err(|e| self.error(e))?;
            }
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<(), SourceError> {
        // Disconnect blocked sends before stopping a child blocked on stdout.
        self.frames.take();
        let stopped = stop_child(&mut self.child).map_err(|e| self.error(e));
        let joined = self.join_reader(true);
        stopped.and(joined)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            log::error!("{error}");
        }
    }
}

fn read_frames(
    stdout: impl Read,
    size: (u32, u32),
    tx: mpsc::SyncSender<Yuv420pData>,
) -> io::Result<()> {
    let y_size = size.0 as usize * size.1 as usize;
    let uv_size = y_size / 4;
    let frame_size = y_size + 2 * uv_size;
    let mut reader = io::BufReader::with_capacity(frame_size * 2, stdout);
    let mut buf = vec![0; frame_size];
    loop {
        // A clean frame boundary is EOF; an incomplete frame is corruption.
        let mut read = 0;
        while read < frame_size {
            match reader.read(&mut buf[read..]) {
                Ok(0) if read == 0 => return Ok(()),
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("truncated YUV420P frame: {read}/{frame_size} bytes"),
                    ));
                }
                Ok(n) => read += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        let frame = Yuv420pData {
            size,
            y: buf[..y_size].to_vec(),
            u: buf[y_size..y_size + uv_size].to_vec(),
            v: buf[y_size + uv_size..].to_vec(),
        };
        if tx.send(frame).is_err() {
            return Ok(());
        }
    }
}

fn stop_child(child: &mut Child) -> io::Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        // rpicam-vid handles SIGINT by stopping its camera and encoder.
        // SAFETY: this positive PID belongs to an owned, unreaped child.
        let sent = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) };
        if sent == 0 {
            let deadline = std::time::Instant::now() + Duration::from_secs(1);
            while std::time::Instant::now() < deadline {
                if child.try_wait()?.is_some() {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
    // A stalled rpicam child must not survive source teardown. Kill is only
    // a fallback on Unix; always reap afterward, including an exit/kill race.
    if let Err(error) = child.kill()
        && child.try_wait()?.is_none()
    {
        return Err(error);
    }
    child.wait().map(|_| ())
}

impl LibcameraCameraSource {
    /// Start camera processes. Reads observe the caller's cancellation flag.
    ///
    /// Invalid dimensions/rate and process/thread startup failures are returned
    /// directly. Asynchronous camera startup failures are reported on reading.
    pub fn open(
        config: &LibcameraConfig,
        interrupted: Arc<AtomicBool>,
    ) -> Result<Self, SourceError> {
        Self::open_with(config, interrupted, |camera| {
            let mut command = Command::new("rpicam-vid");
            // Inherit diagnostics instead of creating an undrained stderr pipe.
            command.stderr(Stdio::inherit());
            command.args([
                "--camera",
                &camera.to_string(),
                "--width",
                &config.width.to_string(),
                "--height",
                &config.height.to_string(),
                "--framerate",
                &config.fps.to_string(),
                "--codec",
                "yuv420",
                "-t",
                "0",
                "-n",
                "-o",
                "-",
            ]);
            command
        })
    }

    fn open_with(
        config: &LibcameraConfig,
        interrupted: Arc<AtomicBool>,
        mut command: impl FnMut(u32) -> Command,
    ) -> Result<Self, SourceError> {
        let y_size = (config.width as usize).checked_mul(config.height as usize);
        if config.width == 0
            || config.height == 0
            || config.fps == 0
            || config.fps > i32::MAX as u32
            || !config.width.is_multiple_of(2)
            || !config.height.is_multiple_of(2)
            || y_size
                .and_then(|n| n.checked_mul(3))
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(SourceError::Init {
                path: "rpicam-vid".into(),
                reason: "YUV420P capture requires positive even dimensions within addressable buffer size and a frame rate in 1..=i32::MAX".into(),
            });
        }
        let mut source = Self {
            cameras: Vec::with_capacity(2),
            info: SourceInfo {
                camera_sizes: vec![(config.width, config.height); 2],
                fps: config.fps as f64,
                fps_rational: Some((config.fps as i32, 1)),
                total_frames: None,
            },
            interrupted,
            stopped: false,
        };
        for id in [config.left_camera, config.right_camera] {
            source.cameras.push(Camera::start(
                command(id),
                format!("camera {id}"),
                (config.width, config.height),
            )?);
            if config.left_camera == config.right_camera {
                break;
            }
        }
        Ok(source)
    }

    fn next(&mut self) -> Result<Option<FrameSet>, SourceError> {
        let mut pending = [None, None];
        loop {
            let mut ended = false;
            for camera in &mut self.cameras {
                ended |= camera.check()? && camera.eof;
            }
            if ended || self.interrupted.load(Ordering::Relaxed) {
                return Ok(None);
            }
            for (camera, frame) in self.cameras.iter_mut().zip(&mut pending) {
                if frame.is_none() && !camera.eof {
                    match camera
                        .frames
                        .as_ref()
                        .unwrap()
                        .recv_timeout(Duration::from_millis(20))
                    {
                        Ok(next) => *frame = Some(next),
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => camera.eof = true,
                    }
                }
            }
            // Check again before delivering output or accepting an observed EOF.
            for camera in &mut self.cameras {
                ended |= camera.check()? && camera.eof;
            }
            if ended || self.interrupted.load(Ordering::Relaxed) {
                return Ok(None);
            }
            if pending[..self.cameras.len()].iter().all(Option::is_some) {
                let mut frames: Vec<_> = pending
                    .into_iter()
                    .take(self.cameras.len())
                    .map(|frame| std::sync::Arc::new(frame.unwrap()))
                    .collect();
                if frames.len() == 1 {
                    frames.push(frames[0].clone());
                }
                return Ok(Some(FrameSet::Yuv420p(frames)));
            }
            // stdout can close before the child exits. Keep cancellation live
            // without spinning or treating an unknown exit status as success.
            if self.cameras.iter().any(|camera| camera.eof) {
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }

    fn stop(&mut self) -> Result<(), SourceError> {
        self.stopped = true;
        for camera in &mut self.cameras {
            camera.frames.take();
        }
        let mut result = Ok(());
        for camera in &mut self.cameras {
            let stopped = camera.stop();
            if result.is_ok() {
                result = stopped;
            }
        }
        result
    }
}

impl reco_core::source::FrameSource for LibcameraCameraSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        if self.stopped {
            return Ok(None);
        }
        let result = self.next().map(|frame| frame.map(SourceFrame::untimed));
        if !matches!(result, Ok(Some(_))) {
            let stopped = self.stop();
            if result.is_ok() {
                stopped?;
            } else if let Err(error) = stopped {
                log::error!("{error}");
            }
        }
        result
    }

    fn is_exhausted(&self) -> bool {
        self.stopped
    }
}

impl Drop for LibcameraCameraSource {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            log::error!("{error}");
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use reco_core::source::FrameSource;
    use std::time::Instant;

    const FRAME: &str = "printf 'abcdefghUVxy'";
    const IDLE: &str = "exec sleep 60";

    fn config(single: bool) -> LibcameraConfig {
        LibcameraConfig {
            width: 4,
            height: 2,
            fps: 30,
            left_camera: 0,
            right_camera: u32::from(!single),
        }
    }

    fn shell(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    fn source(scripts: &[&str], flag: Arc<AtomicBool>) -> LibcameraCameraSource {
        LibcameraCameraSource::open_with(&config(scripts.len() == 1), flag, |id| {
            shell(scripts[id as usize])
        })
        .unwrap()
    }

    fn stopped(source: &mut LibcameraCameraSource) {
        assert!(source.is_exhausted());
        for camera in &mut source.cameras {
            assert!(camera.child.try_wait().unwrap().is_some());
            assert!(camera.reader.is_none());
            assert!(camera.frames.is_none());
        }
        assert!(source.next_frame().unwrap().is_none());
    }

    #[cfg(all(feature = "ffmpeg", target_os = "linux"))]
    #[test]
    fn supplied_job_reaps_capture_before_output_finalization() {
        use reco_core::calibration::{Calibration, Framing, Lens};
        let interrupted = Arc::new(AtomicBool::new(false));
        let source = source(
            &["while :; do printf 'abcdefghUVxy'; done"; 2],
            interrupted.clone(),
        );
        let children: Vec<_> = source
            .cameras
            .iter()
            .map(|camera| camera.child.id())
            .collect();
        let lens = Lens::fisheye(4, 2, 2.0, 2.0, 2.0, 1.0, [0.0; 4]);
        let calibration = Calibration::new(
            vec![lens.clone(), lens],
            reco_core::projection::LShape {
                intersect: 0.5,
                x_ty: 0.0,
                x_rz: 0.0,
                z_rx: 0.0,
                x_rx: 0.0,
                z_rz: 0.0,
                blend_width: 0.0,
            },
            Framing {
                axis_offset: 0.25,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("capture.mp4");
        let result = crate::StitchJob::from_source(source, calibration, &output)
            .cpu()
            .resolution(64, 32)
            .encoder_name("libx264")
            .max_frames(2)
            .on_session(|session, _| {
                session.core_mut().set_constrained_look(false);
                Ok(())
            })
            .on_finalizing(move || {
                for child in children {
                    assert!(
                        !std::path::Path::new(&format!("/proc/{child}")).exists(),
                        "capture child still alive during finalization"
                    );
                }
            })
            .run(&interrupted)
            .unwrap();
        assert_eq!(result.frames_processed, 2);
        let mut decoder = crate::ffmpeg::decoder::VideoDecoder::open_software(&output).unwrap();
        assert!(decoder.next_frame().unwrap().is_some());
        assert!(decoder.next_frame().unwrap().is_some());
        assert!(decoder.next_frame().unwrap().is_none());
    }

    fn bounded(run: impl FnOnce() + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            run();
            tx.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_secs(5))
            .expect("capture operation stalled");
        worker.join().unwrap();
    }

    #[test]
    fn finite_mono_and_stereo_preserve_planes_and_eof() {
        bounded(|| {
            for single in [true, false] {
                let flag = Arc::new(AtomicBool::new(false));
                let scripts = if single {
                    vec![FRAME]
                } else {
                    vec![FRAME, FRAME]
                };
                let mut source = source(&scripts, flag.clone());
                assert_eq!(source.cameras.len(), if single { 1 } else { 2 });
                let Some(FrameSet::Yuv420p(frames)) =
                    source.next_frame().unwrap().map(|frame| frame.images)
                else {
                    panic!("missing frame");
                };
                assert_eq!(frames.len(), 2);
                assert_eq!(Arc::ptr_eq(&frames[0], &frames[1]), single);
                for frame in frames {
                    assert_eq!(frame.y, b"abcdefgh");
                    assert_eq!(frame.u, b"UV");
                    assert_eq!(frame.v, b"xy");
                }
                assert!(source.next_frame().unwrap().is_none());
                stopped(&mut source);
                assert!(!flag.load(Ordering::Relaxed));
            }
        });
    }

    #[test]
    fn queued_frames_drain_in_order_before_successful_exit() {
        bounded(|| {
            let mut capture = source(
                &["printf 'aaaaaaaaUVxyaaaaaaaaUVxybbbbbbbbUVxyccccccccUVxy'"],
                Arc::new(AtomicBool::new(false)),
            );
            for y in [b'a', b'a', b'b', b'c'] {
                let Some(FrameSet::Yuv420p(frames)) =
                    capture.next_frame().unwrap().map(|frame| frame.images)
                else {
                    panic!("queued frame was lost");
                };
                for frame in frames {
                    assert_eq!(frame.y, vec![y; 8]);
                }
            }
            assert!(capture.next_frame().unwrap().is_none());
            stopped(&mut capture);
        });
    }

    #[test]
    fn exit_and_truncation_errors_are_visible_with_either_peer_stalled() {
        bounded(|| {
            for failure in ["exit 7", "printf 'bad'"] {
                for scripts in [[failure, IDLE], [IDLE, failure]] {
                    let mut source = source(&scripts, Arc::new(AtomicBool::new(false)));
                    let error = source.next_frame().err().unwrap().to_string();
                    assert!(
                        error.contains(if failure == "exit 7" {
                            "exit status: 7"
                        } else {
                            "truncated"
                        }),
                        "{error}"
                    );
                    stopped(&mut source);
                }
            }
        });
    }

    #[test]
    fn shortest_input_eof_stops_a_silent_peer() {
        bounded(|| {
            for scripts in [["exit 0", IDLE], [IDLE, "exit 0"]] {
                let mut source = source(&scripts, Arc::new(AtomicBool::new(false)));
                assert!(source.next_frame().unwrap().is_none());
                stopped(&mut source);
            }
        });
    }

    #[test]
    fn cancellation_unblocks_partial_and_silent_reads() {
        bounded(|| {
            for scripts in [
                ["printf bad; exec sleep 60", IDLE],
                [IDLE, "printf bad; exec sleep 60"],
            ] {
                let flag = Arc::new(AtomicBool::new(false));
                let mut source = source(&scripts, flag.clone());
                let trigger = std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(50));
                    flag.store(true, Ordering::Relaxed);
                });
                assert!(source.next_frame().unwrap().is_none());
                trigger.join().unwrap();
                stopped(&mut source);
            }
        });
    }

    #[test]
    fn closed_stdout_does_not_hide_late_exit_or_block_cancellation() {
        bounded(|| {
            let flag = Arc::new(AtomicBool::new(false));
            let mut capture = source(&["exec 1>&-; exec sleep 60"], flag.clone());
            let trigger = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                flag.store(true, Ordering::Relaxed);
            });
            assert!(capture.next_frame().unwrap().is_none());
            trigger.join().unwrap();
            stopped(&mut capture);
            let mut capture = source(&["exec 1>&-; exit 9"], Arc::new(AtomicBool::new(false)));
            assert!(
                capture
                    .next_frame()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("exit status: 9")
            );
            stopped(&mut capture);
        });
    }

    #[test]
    fn observed_failure_wins_over_cancellation() {
        bounded(|| {
            let flag = Arc::new(AtomicBool::new(false));
            let mut capture = source(&["exit 7"], flag.clone());
            capture.cameras[0].child.wait().unwrap();
            flag.store(true, Ordering::Relaxed);
            assert!(capture.next_frame().is_err());
            stopped(&mut capture);
        });
    }

    #[test]
    fn observed_failure_wins_over_peer_eof() {
        bounded(|| {
            for scripts in [["exit 0", "exit 7"], ["exit 7", "exit 0"]] {
                let mut capture = source(&scripts, Arc::new(AtomicBool::new(false)));
                for camera in &mut capture.cameras {
                    camera.child.wait().unwrap();
                }
                assert!(capture.next_frame().is_err());
                stopped(&mut capture);
            }
        });
    }

    #[test]
    fn full_frame_queues_are_disconnected_before_joining() {
        bounded(|| {
            let script = format!("while :; do {FRAME}; done");
            let mut capture = source(&[&script, &script], Arc::new(AtomicBool::new(false)));
            assert!(capture.next_frame().unwrap().is_some());
            std::thread::sleep(Duration::from_millis(50));
            capture.stop().unwrap();
            stopped(&mut capture);
        });
    }

    #[test]
    fn drop_reaps_children_and_joins_readers() {
        bounded(|| {
            let capture = source(&[IDLE, IDLE], Arc::new(AtomicBool::new(false)));
            let pids: Vec<_> = capture
                .cameras
                .iter()
                .map(|camera| camera.child.id())
                .collect();
            drop(capture);
            for pid in pids {
                // SAFETY: query only our former child; do not signal another process.
                assert_eq!(
                    unsafe {
                        libc::waitpid(pid as libc::pid_t, std::ptr::null_mut(), libc::WNOHANG)
                    },
                    -1
                );
                assert_eq!(
                    io::Error::last_os_error().raw_os_error(),
                    Some(libc::ECHILD)
                );
            }
        });
    }

    #[test]
    fn verbose_child_cannot_fill_an_unread_stderr_pipe() {
        bounded(|| {
            // Exceed a pipe buffer without creating descendants or changing global stdio.
            let script = format!(
                "i=0; while [ \"$i\" -lt 1024 ]; do printf '%s' '{}' >&2; i=$((i+1)); done; {FRAME}",
                "x".repeat(128)
            );
            let diagnostics = tempfile::tempfile().unwrap();
            let mut capture = LibcameraCameraSource::open_with(
                &config(true),
                Arc::new(AtomicBool::new(false)),
                |_| {
                    let mut command = shell(&script);
                    command.stderr(diagnostics.try_clone().unwrap());
                    command
                },
            )
            .unwrap();
            assert!(capture.cameras[0].child.stderr.is_none());
            assert!(capture.next_frame().unwrap().is_some());
            assert!(capture.next_frame().unwrap().is_none());
            stopped(&mut capture);
            assert_eq!(diagnostics.metadata().unwrap().len(), 128 * 1024);
        });
    }

    #[test]
    fn blocked_frame_send_exits_when_receiver_drops() {
        bounded(|| {
            struct Frames {
                count: usize,
                full: mpsc::Sender<()>,
            }
            impl Read for Frames {
                fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                    self.count += 1;
                    if self.count == 3 {
                        self.full.send(()).unwrap();
                    }
                    buf[..12].fill(1);
                    Ok(12)
                }
            }
            let (tx, rx) = mpsc::sync_channel(2);
            let (full, ready) = mpsc::channel();
            let reader =
                std::thread::spawn(move || read_frames(Frames { count: 0, full }, (4, 2), tx));
            ready.recv_timeout(Duration::from_secs(2)).unwrap();
            drop(rx);
            reader.join().unwrap().unwrap();
        });
    }

    fn wait_file(path: &std::path::Path) -> String {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Ok(contents) = std::fs::read_to_string(path)
                && !contents.is_empty()
            {
                return contents;
            }
            assert!(Instant::now() < deadline, "child did not signal readiness");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn second_spawn_failure_reaps_first_child() {
        bounded(|| {
            let dir = tempfile::tempdir().unwrap();
            let pid_file = dir.path().join("pid");
            let mut pid = 0;
            let result = LibcameraCameraSource::open_with(
                &config(false),
                Arc::new(AtomicBool::new(false)),
                |id| {
                    if id == 0 {
                        let mut command = shell("printf '%s' $$ > \"$1\"; exec sleep 60");
                        command.arg("test").arg(&pid_file);
                        command
                    } else {
                        pid = wait_file(&pid_file).parse::<libc::pid_t>().unwrap();
                        Command::new(dir.path().join("missing-executable"))
                    }
                },
            );
            assert!(matches!(result, Err(SourceError::Init { .. })));
            // SAFETY: waitpid checks the specific test child; no signal is sent.
            assert_eq!(
                unsafe { libc::waitpid(pid, std::ptr::null_mut(), libc::WNOHANG) },
                -1
            );
            assert_eq!(
                io::Error::last_os_error().raw_os_error(),
                Some(libc::ECHILD)
            );
        });
    }

    #[test]
    fn graceful_interrupt_and_stubborn_child_fallback_are_reaped() {
        bounded(|| {
            let dir = tempfile::tempdir().unwrap();
            for stubborn in [false, true] {
                let ready = dir.path().join(format!("ready-{stubborn}"));
                let script = if stubborn {
                    "trap '' INT; printf ready > \"$1\"; while :; do :; done"
                } else {
                    "trap 'exit 0' INT; printf ready > \"$1\"; while :; do :; done"
                };
                let mut command = shell(script);
                command.arg("test").arg(&ready);
                let mut camera = Camera::start(command, "test".into(), (4, 2)).unwrap();
                wait_file(&ready);
                camera.stop().unwrap();
                assert_eq!(camera.child.wait().unwrap().success(), !stubborn);
                assert!(camera.reader.is_none());
                assert!(camera.child.stderr.is_none());
            }
        });
    }

    #[test]
    fn invalid_configuration_does_not_spawn() {
        for (width, height, fps) in [
            (0, 2, 30),
            (4, 0, 30),
            (3, 2, 30),
            (4, 3, 30),
            (4, 2, 0),
            (u32::MAX - 1, u32::MAX - 1, 30),
        ] {
            let mut config = config(true);
            config.width = width;
            config.height = height;
            config.fps = fps;
            assert!(
                LibcameraCameraSource::open_with(
                    &config,
                    Arc::new(AtomicBool::new(false)),
                    |_| panic!("invalid input reached spawn")
                )
                .is_err()
            );
        }
    }

    #[test]
    fn reader_preserves_io_errors_and_retries_interrupted_reads() {
        struct Interrupted(bool);
        impl Read for Interrupted {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                if !self.0 {
                    self.0 = true;
                    Err(io::ErrorKind::Interrupted.into())
                } else {
                    Err(io::ErrorKind::BrokenPipe.into())
                }
            }
        }
        let (tx, _rx) = mpsc::sync_channel(2);
        assert_eq!(
            read_frames(Interrupted(false), (4, 2), tx)
                .unwrap_err()
                .kind(),
            io::ErrorKind::BrokenPipe
        );
    }
}
