//! Live stereo camera capture via GStreamer.
//!
//! Builds a GStreamer pipeline per camera, pulls I420 frames from
//! `appsink` in dedicated threads, and pairs them into stereo
//! `FramePair`s via a bounded channel.

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use reco_core::source::{FramePair, SourceError, SourceInfo, YuvData};
use std::path::Path;
use std::sync::mpsc;

/// Camera capture configuration.
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Capture width in pixels.
    pub width: u32,
    /// Capture height in pixels.
    pub height: u32,
    /// Capture frame rate.
    pub fps: u32,
    /// Left camera device or sensor ID.
    ///
    /// - Jetson: sensor index as string ("0", "1")
    /// - Linux: V4L2 device path ("/dev/video0")
    pub left_device: String,
    /// Right camera device or sensor ID.
    pub right_device: String,
}

/// Build the platform-appropriate GStreamer pipeline string.
fn build_pipeline_string(device: &str, width: u32, height: u32, fps: u32) -> String {
    if is_jetson() {
        // Jetson: nvarguscamerasrc runs the full NVIDIA ISP
        // (debayer, AWB, AE, denoise). Output is NV12 in NVMM;
        // nvvidconv copies to system memory as I420.
        format!(
            "nvarguscamerasrc sensor-id={device} ! \
             video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! \
             nvvidconv ! \
             video/x-raw,format=I420 ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else if cfg!(target_os = "macos") {
        format!(
            "avfvideosrc device-index={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format=I420 ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else if cfg!(target_os = "windows") {
        format!(
            "mfvideosrc device-index={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format=I420 ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else {
        // Linux: generic V4L2
        format!(
            "v4l2src device={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format=I420 ! \
             appsink name=sink emit-signals=false sync=false"
        )
    }
}

/// Detect if we're running on a Jetson (L4T/Tegra).
fn is_jetson() -> bool {
    Path::new("/etc/nv_tegra_release").exists()
        || std::fs::read_to_string("/proc/device-tree/compatible")
            .unwrap_or_default()
            .contains("nvidia,tegra")
}

/// Extract I420 planes from a GStreamer buffer.
fn extract_i420(data: &[u8], width: u32, height: u32) -> Result<YuvData, SourceError> {
    let y_size = (width * height) as usize;
    let uv_size = ((width / 2) * (height / 2)) as usize;

    if data.len() < y_size + 2 * uv_size {
        return Err(SourceError::Read(format!(
            "buffer too small: {} < {}",
            data.len(),
            y_size + 2 * uv_size
        )));
    }

    Ok(YuvData {
        y: data[..y_size].to_vec(),
        u: data[y_size..y_size + uv_size].to_vec(),
        v: data[y_size + uv_size..y_size + 2 * uv_size].to_vec(),
    })
}

/// Spawn a GStreamer capture thread for one camera.
///
/// The thread builds and runs the pipeline internally, pulling I420
/// frames from appsink and sending them through a bounded channel.
/// This keeps GStreamer's pipeline and thread model self-contained.
fn spawn_capture_thread(
    device: String,
    label: &'static str,
    width: u32,
    height: u32,
    fps: u32,
) -> mpsc::Receiver<YuvData> {
    let (tx, rx) = mpsc::sync_channel::<YuvData>(2);

    std::thread::Builder::new()
        .name(format!("capture_{label}"))
        .spawn(move || {
            if let Err(e) = gst::init() {
                log::error!("{label} GStreamer init failed: {e}");
                return;
            }

            let pipeline_str = build_pipeline_string(&device, width, height, fps);
            log::info!("{label} pipeline: {pipeline_str}");

            let pipeline = match gst::parse::launch(&pipeline_str) {
                Ok(p) => match p.downcast::<gst::Pipeline>() {
                    Ok(p) => p,
                    Err(_) => {
                        log::error!("{label}: not a pipeline");
                        return;
                    }
                },
                Err(e) => {
                    log::error!("{label} pipeline parse: {e}");
                    return;
                }
            };

            let appsink = match pipeline.by_name("sink") {
                Some(elem) => match elem.downcast::<gst_app::AppSink>() {
                    Ok(s) => s,
                    Err(_) => {
                        log::error!("{label}: element is not an AppSink");
                        return;
                    }
                },
                None => {
                    log::error!("{label}: appsink not found");
                    return;
                }
            };

            appsink.set_max_buffers(2);
            appsink.set_drop(true);

            if let Err(e) = pipeline.set_state(gst::State::Playing) {
                log::error!("{label} set_state(Playing): {e}");
                return;
            }

            loop {
                let sample = match appsink.pull_sample() {
                    Ok(s) => s,
                    Err(_) => break, // EOS
                };

                let Some(buffer) = sample.buffer() else {
                    log::error!("{label}: sample has no buffer");
                    break;
                };

                let Ok(map) = buffer.map_readable() else {
                    log::error!("{label}: buffer map failed");
                    break;
                };

                match extract_i420(map.as_slice(), width, height) {
                    Ok(yuv) => {
                        if tx.send(yuv).is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        log::error!("{label}: {e}");
                        break;
                    }
                }
            }

            let _ = pipeline.set_state(gst::State::Null);
        })
        .expect("spawn capture thread");

    rx
}

/// Stereo camera source using GStreamer.
///
/// Each camera runs in its own thread pulling frames from appsink.
/// A pairing thread zips left+right into `FramePair`s and sends
/// them through a bounded channel. This ensures both cameras are
/// captured in parallel rather than sequentially.
pub struct GstreamerCameraSource {
    rx: mpsc::Receiver<FramePair>,
    info: SourceInfo,
}

impl GstreamerCameraSource {
    /// Open a stereo camera source with threaded capture.
    pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
        gst::init().map_err(|e| SourceError::Init(format!("GStreamer init: {e}")))?;

        let left_rx = spawn_capture_thread(
            config.left_device.clone(),
            "left",
            config.width,
            config.height,
            config.fps,
        );
        let right_rx = spawn_capture_thread(
            config.right_device.clone(),
            "right",
            config.width,
            config.height,
            config.fps,
        );

        // Pairing thread: zip left + right by arrival order
        let (tx, rx) = mpsc::sync_channel::<FramePair>(2);

        std::thread::Builder::new()
            .name("capture_pair".into())
            .spawn(move || {
                while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                    if tx.send(FramePair { left, right }).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn pairing thread");

        let info = SourceInfo {
            width: config.width,
            height: config.height,
            fps: config.fps as f64,
        };

        log::info!(
            "Camera source ready: {}x{} @ {} fps (threaded)",
            config.width,
            config.height,
            config.fps
        );

        Ok(Self { rx, info })
    }
}

impl reco_core::source::FrameSource for GstreamerCameraSource {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            width: self.info.width,
            height: self.info.height,
            fps: self.info.fps,
        }
    }

    fn next_pair(&mut self) -> Result<Option<FramePair>, SourceError> {
        match self.rx.recv() {
            Ok(pair) => Ok(Some(pair)),
            Err(_) => Ok(None),
        }
    }
}
