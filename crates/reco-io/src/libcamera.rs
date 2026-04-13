//! Live stereo camera capture via libcamera (`rpicam-vid`).
//!
//! Spawns `rpicam-vid` processes for each camera and reads raw YUV420P
//! frames from their stdout pipes. This is the lowest-latency capture
//! path for Raspberry Pi CSI cameras - no GStreamer or FFmpeg decode
//! overhead, just the ISP pipeline feeding directly into the stitcher.
//!
//! Requires `rpicam-vid` to be installed (ships with Raspberry Pi OS).
//!
//! # Architecture
//!
//! ```text
//! rpicam-vid --camera 0 --codec yuv420 -o - ──> [stdout pipe] ──> capture_left thread
//! rpicam-vid --camera 1 --codec yuv420 -o - ──> [stdout pipe] ──> capture_right thread
//!                                                                         │
//!                                                     pairing thread <────┘
//!                                                          │
//!                                                     FrameSource::next_frame()
//! ```

use reco_core::source::{FramePair, SourceError, SourceInfo, StereoFrame, YuvData};
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;

/// libcamera capture configuration.
#[derive(Debug, Clone)]
pub struct LibcameraConfig {
    /// Capture width in pixels.
    pub width: u32,
    /// Capture height in pixels.
    pub height: u32,
    /// Capture frame rate.
    pub fps: u32,
    /// Left camera index (0, 1, ...).
    pub left_camera: u32,
    /// Right camera index.
    pub right_camera: u32,
}

/// Stereo camera source using libcamera (`rpicam-vid`).
///
/// Each camera runs as a separate `rpicam-vid` subprocess outputting
/// raw YUV420P frames to stdout. Capture threads read fixed-size
/// frames and pair them for the stitch pipeline.
///
/// When both cameras use the same index (single-camera mode), only
/// one `rpicam-vid` process is spawned and each frame is duplicated
/// for both left and right. Useful for testing with a single CSI camera.
///
/// This is the recommended capture backend for Raspberry Pi CSI cameras.
/// It bypasses GStreamer and FFmpeg entirely, providing the lowest
/// possible latency path from the camera ISP to the GPU renderer.
pub struct LibcameraCameraSource {
    rx: mpsc::Receiver<FramePair>,
    info: SourceInfo,
    /// rpicam-vid child processes. One in single-camera mode, two in stereo.
    children: Vec<Child>,
}

/// Spawn `rpicam-vid` for a single camera, returning the child process
/// and a receiver for parsed YUV420P frames.
///
/// The subprocess outputs raw I420 (YUV420P) frames to stdout,
/// tightly packed with no stride padding. A reader thread splits
/// each frame into Y, U, V planes.
fn spawn_rpicam_vid(
    camera_id: u32,
    label: &'static str,
    width: u32,
    height: u32,
    fps: u32,
) -> Result<(Child, mpsc::Receiver<YuvData>), SourceError> {
    let mut child = Command::new("rpicam-vid")
        .args([
            "--camera",
            &camera_id.to_string(),
            "--width",
            &width.to_string(),
            "--height",
            &height.to_string(),
            "--framerate",
            &fps.to_string(),
            "--codec",
            "yuv420",
            "-t",
            "0",  // run indefinitely
            "-n", // no preview window
            "-o",
            "-", // output to stdout
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| SourceError::Init {
            path: format!("rpicam-vid --camera {camera_id}"),
            reason: format!("Failed to spawn rpicam-vid: {e}. Is rpicam-vid installed?"),
        })?;

    let stdout = child.stdout.take().ok_or_else(|| SourceError::Init {
        path: format!("rpicam-vid --camera {camera_id}"),
        reason: "Failed to capture stdout".into(),
    })?;

    let (tx, rx) = mpsc::sync_channel::<YuvData>(2);

    let frame_size = (width * height * 3 / 2) as usize;
    let y_size = (width * height) as usize;
    let uv_size = ((width / 2) * (height / 2)) as usize;

    std::thread::Builder::new()
        .name(format!("libcam_{label}"))
        .spawn(move || {
            let mut reader = std::io::BufReader::with_capacity(frame_size * 2, stdout);
            let mut buf = vec![0u8; frame_size];

            loop {
                if let Err(e) = reader.read_exact(&mut buf) {
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        log::error!("{label}: read error: {e}");
                    }
                    break;
                }

                let yuv = YuvData {
                    y: buf[..y_size].to_vec(),
                    u: buf[y_size..y_size + uv_size].to_vec(),
                    v: buf[y_size + uv_size..].to_vec(),
                };

                if tx.send(yuv).is_err() {
                    break;
                }
            }
            log::info!("{label}: capture thread finished");
        })
        .expect("spawn libcamera capture thread");

    Ok((child, rx))
}

impl LibcameraCameraSource {
    /// Open a stereo camera source using `rpicam-vid`.
    ///
    /// Spawns `rpicam-vid` processes and starts reading YUV420P frames
    /// from their stdout pipes. When `left_camera == right_camera`,
    /// uses a single process and duplicates each frame for both sides.
    ///
    /// # Errors
    ///
    /// Returns an error if `rpicam-vid` cannot be spawned (not installed,
    /// camera not connected, etc.).
    pub fn open(config: &LibcameraConfig) -> Result<Self, SourceError> {
        let single_camera = config.left_camera == config.right_camera;

        log::info!(
            "Opening libcamera source: cam{}+cam{} @ {}x{} {}fps{}",
            config.left_camera,
            config.right_camera,
            config.width,
            config.height,
            config.fps,
            if single_camera {
                " (single-camera mode: duplicating frames)"
            } else {
                ""
            },
        );

        let (tx, rx) = mpsc::sync_channel::<FramePair>(2);

        let children = if single_camera {
            // Single-camera mode: one rpicam-vid, duplicate each frame.
            let (child, cam_rx) = spawn_rpicam_vid(
                config.left_camera,
                "cam",
                config.width,
                config.height,
                config.fps,
            )?;

            std::thread::Builder::new()
                .name("libcam_dup".into())
                .spawn(move || {
                    while let Ok(frame) = cam_rx.recv() {
                        let pair = FramePair {
                            left: frame.clone(),
                            right: frame,
                        };
                        if tx.send(pair).is_err() {
                            break;
                        }
                    }
                    log::info!("libcamera duplication thread finished");
                })
                .expect("spawn libcamera duplication thread");

            vec![child]
        } else {
            // Stereo mode: two rpicam-vid processes, pair their frames.
            let (left_child, left_rx) = spawn_rpicam_vid(
                config.left_camera,
                "left",
                config.width,
                config.height,
                config.fps,
            )?;

            let (right_child, right_rx) = spawn_rpicam_vid(
                config.right_camera,
                "right",
                config.width,
                config.height,
                config.fps,
            )?;

            std::thread::Builder::new()
                .name("libcam_pair".into())
                .spawn(move || {
                    while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                        if tx.send(FramePair { left, right }).is_err() {
                            break;
                        }
                    }
                    log::info!("libcamera pairing thread finished");
                })
                .expect("spawn libcamera pairing thread");

            vec![left_child, right_child]
        };

        let info = SourceInfo {
            width: config.width,
            height: config.height,
            fps: config.fps as f64,
            fps_rational: None,
        };

        let mode = if single_camera {
            "single-camera duplicate"
        } else {
            "stereo"
        };
        log::info!(
            "libcamera source ready: {}x{} @ {} fps (YUV420P, {mode}, rpicam-vid)",
            config.width,
            config.height,
            config.fps,
        );

        Ok(Self {
            rx,
            info,
            children,
        })
    }
}

impl reco_core::source::FrameSource for LibcameraCameraSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        match self.rx.recv() {
            Ok(pair) => Ok(Some(StereoFrame::Yuv420p(pair))),
            Err(_) => Ok(None),
        }
    }
}

impl Drop for LibcameraCameraSource {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        // Drain pending frames to unblock capture threads
        while self.rx.try_recv().is_ok() {}
        for child in &mut self.children {
            let _ = child.wait();
        }
        log::info!("libcamera source shut down");
    }
}
