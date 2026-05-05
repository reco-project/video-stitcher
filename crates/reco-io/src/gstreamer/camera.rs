//! Live stereo camera capture via GStreamer.
//!
//! Builds a GStreamer pipeline per camera, pulls frames from
//! `appsink` in dedicated threads, and pairs them into stereo
//! frame pairs via a bounded channel.
//!
//! Supports two capture modes:
//! - **I420** (YUV420P): three separate planes, works everywhere
//! - **NV12**: Y + interleaved UV, native NVIDIA ISP output - skips
//!   format conversion on Jetson for lower latency

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use reco_core::source::{
    FramePair, Nv12Data, Nv12FramePair, SourceError, SourceInfo, StereoFrame, YuvData,
};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Capture pixel format for the GStreamer pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureFormat {
    /// YUV420P: three separate planes (Y, U, V). Works everywhere.
    I420,
    /// NV12: Y + interleaved UV. Native NVIDIA ISP output - avoids
    /// a format conversion step on Jetson.
    Nv12,
}

/// Validate a device string before interpolating it into a GStreamer pipeline.
///
/// Accepted formats per platform:
/// - Jetson (nvarguscamerasrc): numeric sensor ID only, e.g. `"0"`, `"1"`
/// - macOS (avfvideosrc): numeric device index only
/// - Windows (mfvideosrc): numeric device index only
/// - Linux V4L2: path matching `/dev/video<digits>`, e.g. `/dev/video0`
///
/// Returns `Err` with a descriptive message if the device string does not
/// match the expected pattern, preventing injection of arbitrary GStreamer
/// elements or shell metacharacters into the pipeline description.
fn validate_device_string(device: &str) -> Result<(), String> {
    if is_tegra() || cfg!(target_os = "macos") || cfg!(target_os = "windows") {
        // Numeric index only (one or more digits, nothing else)
        if device.chars().all(|c| c.is_ascii_digit()) && !device.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "invalid device string {device:?}: expected a numeric index (e.g. \"0\")"
            ))
        }
    } else {
        // Linux V4L2: must be exactly /dev/video<digits>
        let valid = device.starts_with("/dev/video")
            && device["/dev/video".len()..]
                .chars()
                .all(|c| c.is_ascii_digit())
            && device.len() > "/dev/video".len();
        if valid {
            Ok(())
        } else {
            Err(format!(
                "invalid device string {device:?}: expected a V4L2 path like \"/dev/video0\""
            ))
        }
    }
}

/// Build the platform-appropriate GStreamer pipeline string.
///
/// Returns an error if `device` fails [`validate_device_string`].
fn build_pipeline_string(
    device: &str,
    width: u32,
    height: u32,
    fps: u32,
    format: CaptureFormat,
) -> Result<String, String> {
    validate_device_string(device)?;

    let fmt_str = match format {
        CaptureFormat::I420 => "I420",
        CaptureFormat::Nv12 => "NV12",
    };

    let pipeline = if is_tegra() {
        // Jetson: nvarguscamerasrc runs the full NVIDIA ISP
        // (debayer, AWB, AE, denoise). Output is NV12 in NVMM;
        // nvvidconv copies to system memory (and converts format if needed).
        format!(
            "nvarguscamerasrc sensor-id={device} ! \
             video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! \
             nvvidconv ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else if cfg!(target_os = "macos") {
        format!(
            "avfvideosrc device-index={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else if cfg!(target_os = "windows") {
        format!(
            "mfvideosrc device-index={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    } else {
        // Linux: generic V4L2
        format!(
            "v4l2src device={device} ! \
             video/x-raw,width={width},height={height},framerate={fps}/1 ! \
             videoconvert ! \
             video/x-raw,format={fmt_str} ! \
             appsink name=sink emit-signals=false sync=false"
        )
    };

    Ok(pipeline)
}

/// Detect if we're running on NVIDIA Tegra (Jetson platform).
fn is_tegra() -> bool {
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
        return Err(SourceError::Read {
            reason: format!(
                "buffer too small: {} < {}",
                data.len(),
                y_size + 2 * uv_size
            ),
        });
    }

    Ok(YuvData {
        y: data[..y_size].to_vec(),
        u: data[y_size..y_size + uv_size].to_vec(),
        v: data[y_size + uv_size..y_size + 2 * uv_size].to_vec(),
    })
}

/// Extract NV12 planes from a GStreamer buffer.
///
/// NV12 layout: Y plane (width*height), then interleaved UV plane (width*height/2).
fn extract_nv12(data: &[u8], width: u32, height: u32) -> Result<Nv12Data, SourceError> {
    let y_size = (width * height) as usize;
    let uv_size = (width * (height / 2)) as usize;

    if data.len() < y_size + uv_size {
        return Err(SourceError::Read {
            reason: format!(
                "NV12 buffer too small: {} < {}",
                data.len(),
                y_size + uv_size
            ),
        });
    }

    Ok(Nv12Data {
        y: data[..y_size].to_vec(),
        uv: data[y_size..y_size + uv_size].to_vec(),
    })
}

/// Build and start a GStreamer pipeline, returning the pipeline and appsink.
fn build_capture_pipeline(
    device: &str,
    label: &'static str,
    width: u32,
    height: u32,
    fps: u32,
    format: CaptureFormat,
) -> Result<(gst::Pipeline, gst_app::AppSink), String> {
    if let Err(e) = gst::init() {
        return Err(format!("{label} GStreamer init failed: {e}"));
    }

    let pipeline_str = build_pipeline_string(device, width, height, fps, format)?;
    log::info!("{label} pipeline: {pipeline_str}");

    let pipeline = gst::parse::launch(&pipeline_str)
        .map_err(|e| format!("{label} pipeline parse: {e}"))?
        .downcast::<gst::Pipeline>()
        .map_err(|_| format!("{label}: not a pipeline"))?;

    let appsink = pipeline
        .by_name("sink")
        .ok_or_else(|| format!("{label}: appsink not found"))?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| format!("{label}: element is not an AppSink"))?;

    appsink.set_max_buffers(2);
    appsink.set_drop(true);

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| format!("{label} set_state(Playing): {e}"))?;

    Ok((pipeline, appsink))
}

/// Spawn a GStreamer capture thread for one camera (I420 output).
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
            let (pipeline, appsink) = match build_capture_pipeline(
                &device,
                label,
                width,
                height,
                fps,
                CaptureFormat::I420,
            ) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("{e}");
                    return;
                }
            };

            loop {
                let sample = match appsink.pull_sample() {
                    Ok(s) => s,
                    Err(_) => break,
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
                            break;
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

/// Spawn a GStreamer capture thread for one camera (NV12 output).
///
/// The stop signal allows graceful shutdown: send EOS, wait for pipeline
/// to reach Null state, then exit. This prevents Argus teardown crashes.
fn spawn_nv12_capture_thread(
    device: String,
    label: &'static str,
    width: u32,
    height: u32,
    fps: u32,
    stop: Arc<AtomicBool>,
) -> mpsc::Receiver<Nv12Data> {
    let (tx, rx) = mpsc::sync_channel::<Nv12Data>(2);

    std::thread::Builder::new()
        .name(format!("capture_{label}"))
        .spawn(move || {
            let (pipeline, appsink) = match build_capture_pipeline(
                &device,
                label,
                width,
                height,
                fps,
                CaptureFormat::Nv12,
            ) {
                Ok(p) => p,
                Err(e) => {
                    log::error!("{e}");
                    return;
                }
            };

            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let sample = match appsink.pull_sample() {
                    Ok(s) => s,
                    Err(_) => break,
                };

                let Some(buffer) = sample.buffer() else {
                    log::error!("{label}: sample has no buffer");
                    break;
                };

                let Ok(map) = buffer.map_readable() else {
                    log::error!("{label}: buffer map failed");
                    break;
                };

                match extract_nv12(map.as_slice(), width, height) {
                    Ok(nv12) => {
                        if tx.send(nv12).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        log::error!("{label}: {e}");
                        break;
                    }
                }
            }

            // Graceful shutdown: send EOS, then transition to Null
            log::info!("{label}: sending EOS for graceful shutdown");
            pipeline.send_event(gst::event::Eos::new());
            let _ = pipeline.set_state(gst::State::Null);
            // Wait for state change to complete
            let _ = pipeline.state(gst::ClockTime::from_seconds(2));
            log::info!("{label}: pipeline stopped");
        })
        .expect("spawn capture thread");

    rx
}

/// Stereo camera source using GStreamer (I420 output).
///
/// Each camera runs in its own thread pulling frames from appsink.
/// A pairing thread zips left+right into `FramePair`s and sends
/// them through a bounded channel.
pub struct GstreamerCameraSource {
    rx: mpsc::Receiver<FramePair>,
    info: SourceInfo,
}

impl GstreamerCameraSource {
    /// Open a stereo camera source with threaded capture.
    pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
        gst::init().map_err(|e| SourceError::Init {
            path: format!("{} + {}", config.left_device, config.right_device),
            reason: format!("GStreamer init: {e}"),
        })?;

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
            fps_rational: None,
            total_frames: None,
        };

        log::info!(
            "Camera source ready: {}x{} @ {} fps (I420, threaded)",
            config.width,
            config.height,
            config.fps
        );

        Ok(Self { rx, info })
    }
}

impl reco_core::source::FrameSource for GstreamerCameraSource {
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

/// Stereo camera source using GStreamer (NV12 output).
///
/// Like `GstreamerCameraSource` but captures in NV12 format, which is
/// the native output of NVIDIA's ISP. This avoids the NV12->I420
/// conversion that nvvidconv would otherwise perform, saving CPU time
/// and memory bandwidth on Jetson.
///
/// Implements graceful shutdown via `Drop` to avoid Argus teardown crashes.
pub struct GstreamerNv12CameraSource {
    rx: mpsc::Receiver<Nv12FramePair>,
    info: SourceInfo,
    stop: Arc<AtomicBool>,
}

impl GstreamerNv12CameraSource {
    /// Open a stereo NV12 camera source with threaded capture.
    pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
        gst::init().map_err(|e| SourceError::Init {
            path: format!("{} + {}", config.left_device, config.right_device),
            reason: format!("GStreamer init: {e}"),
        })?;

        let stop = Arc::new(AtomicBool::new(false));

        let left_rx = spawn_nv12_capture_thread(
            config.left_device.clone(),
            "left",
            config.width,
            config.height,
            config.fps,
            stop.clone(),
        );
        let right_rx = spawn_nv12_capture_thread(
            config.right_device.clone(),
            "right",
            config.width,
            config.height,
            config.fps,
            stop.clone(),
        );

        let (tx, rx) = mpsc::sync_channel::<Nv12FramePair>(2);

        std::thread::Builder::new()
            .name("capture_pair".into())
            .spawn(move || {
                while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                    if tx.send(Nv12FramePair { left, right }).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn pairing thread");

        let info = SourceInfo {
            width: config.width,
            height: config.height,
            fps: config.fps as f64,
            fps_rational: None,
            total_frames: None,
        };

        log::info!(
            "Camera source ready: {}x{} @ {} fps (NV12, threaded)",
            config.width,
            config.height,
            config.fps
        );

        Ok(Self { rx, info, stop })
    }

    /// Signal capture threads to stop.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Source metadata.
    pub fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    /// Get the next stereo NV12 frame pair, or `None` if the source is exhausted.
    pub fn next_pair(&mut self) -> Result<Option<Nv12FramePair>, SourceError> {
        match self.rx.recv() {
            Ok(pair) => Ok(Some(pair)),
            Err(_) => Ok(None),
        }
    }
}

impl reco_core::source::FrameSource for GstreamerNv12CameraSource {
    fn info(&self) -> reco_core::source::SourceInfo {
        self.info.clone()
    }

    fn next_frame(
        &mut self,
    ) -> Result<Option<reco_core::source::StereoFrame>, reco_core::source::SourceError> {
        self.next_pair()
            .map(|opt| opt.map(reco_core::source::StereoFrame::Nv12))
    }
}

impl Drop for GstreamerNv12CameraSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Drain any pending frames to unblock capture threads
        while self.rx.try_recv().is_ok() {}
        // Give capture threads time to send EOS and reach Null state
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

// ── NVMM zero-copy camera source (Jetson only) ─────────────────────

#[cfg(target_os = "linux")]
pub use nvmm_source::GstreamerNvmmCameraSource;
#[cfg(target_os = "linux")]
pub use nvmm_source::NvmmFramePair;

#[cfg(target_os = "linux")]
mod nvmm_source {
    use super::*;
    use crate::gstreamer::nvmm::{self, NvmmFrameInfo};

    /// Stereo pair of NVMM frame metadata (DMA-buf fds + surface ptrs).
    pub struct NvmmFramePair {
        pub left: NvmmFrameInfo,
        pub right: NvmmFrameInfo,
    }

    fn build_nvmm_pipeline_string(device: &str, config: &CameraConfig) -> Result<String, String> {
        let width = config.width;
        let height = config.height;
        let fps = config.fps;

        // Test override: RECO_NVMM_LEFT / RECO_NVMM_RIGHT env vars pipe a video
        // file through nvv4l2decoder to produce identical NVMM NV12 output.
        let env_key = if device == "0" {
            "RECO_NVMM_LEFT"
        } else {
            "RECO_NVMM_RIGHT"
        };
        if let Ok(path) = std::env::var(env_key) {
            log::info!("[nvmm] {env_key}={path} (file override, bypassing camera)");
            return Ok(format!(
                "filesrc location={path} ! qtdemux ! h264parse ! nvv4l2decoder ! \
                 video/x-raw(memory:NVMM),format=NV12 ! \
                 appsink name=sink emit-signals=false sync=false max-buffers=4 drop=true"
            ));
        }

        validate_device_string(device)?;
        Ok(format!(
            "nvarguscamerasrc sensor-id={device} ! \
             video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate={fps}/1 ! \
             appsink name=sink emit-signals=false sync=false max-buffers=4 drop=true"
        ))
    }

    fn spawn_nvmm_capture_thread(
        device: String,
        label: &'static str,
        config: CameraConfig,
        stop: Arc<AtomicBool>,
    ) -> (mpsc::Receiver<NvmmFrameInfo>, mpsc::SyncSender<()>) {
        let (info_tx, info_rx) = mpsc::sync_channel::<NvmmFrameInfo>(2);
        let (release_tx, release_rx) = mpsc::sync_channel::<()>(4);

        std::thread::Builder::new()
            .name(format!("nvmm_{label}"))
            .spawn(move || {
                let pipeline_str = match build_nvmm_pipeline_string(&device, &config) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("[{label}] NVMM pipeline build: {e}");
                        return;
                    }
                };

                log::info!("[{label}] NVMM pipeline: {pipeline_str}");

                let pipeline = gst::parse::launch(&pipeline_str)
                    .expect("NVMM pipeline parse")
                    .downcast::<gst::Pipeline>()
                    .expect("not a pipeline");
                let appsink = pipeline
                    .by_name("sink")
                    .expect("appsink not found")
                    .downcast::<gst_app::AppSink>()
                    .expect("not an AppSink");
                appsink.set_max_buffers(4);
                appsink.set_drop(true);
                pipeline
                    .set_state(gst::State::Playing)
                    .expect("NVMM pipeline start");

                log::info!("[{label}] NVMM capture started (sensor={device})");

                let mut prev_sample: Option<gst::Sample> = None;

                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    let sample = match appsink.pull_sample() {
                        Ok(s) => s,
                        Err(_) => break,
                    };

                    let buffer = match sample.buffer() {
                        Some(b) => b,
                        None => {
                            log::error!("[{label}] NVMM sample has no buffer");
                            break;
                        }
                    };

                    let map = match buffer.map_readable() {
                        Ok(m) => m,
                        Err(e) => {
                            log::error!("[{label}] NVMM buffer map failed: {e}");
                            break;
                        }
                    };

                    let frame_info =
                        match unsafe { nvmm::extract_nvmm_frame_info(map.as_slice().as_ptr()) } {
                            Ok(info) => info,
                            Err(e) => {
                                log::error!("[{label}] NVMM extract failed: {e}");
                                break;
                            }
                        };

                    drop(map);

                    // Release the previous sample now that the render thread
                    // has had a full frame to use the prior DMA-buf.
                    if let Some(prev) = prev_sample.take() {
                        drop(prev);
                    }

                    if info_tx.send(frame_info).is_err() {
                        break;
                    }

                    prev_sample = Some(sample);

                    // Wait for the render thread to signal it's done importing
                    // this frame's DMA-buf before we let the next pull_sample
                    // potentially recycle the buffer.
                    let _ = release_rx.recv();
                }

                let _ = pipeline.set_state(gst::State::Null);
                log::info!("[{label}] NVMM capture stopped");
            })
            .expect("spawn NVMM capture thread");

        (info_rx, release_tx)
    }

    /// Stereo NVMM camera source for zero-copy Vulkan import + NvBufSurfTransform.
    ///
    /// Each call to `next_pair()` returns DMA-buf fds and NvBufSurface pointers
    /// for both cameras. The caller must call `release_previous()` after GPU
    /// work (rendering + detection) is submitted, before calling `next_pair()`
    /// again - this signals the capture threads to release the old GstSample
    /// and pull the next frame.
    pub struct GstreamerNvmmCameraSource {
        left_rx: mpsc::Receiver<NvmmFrameInfo>,
        right_rx: mpsc::Receiver<NvmmFrameInfo>,
        left_release: mpsc::SyncSender<()>,
        right_release: mpsc::SyncSender<()>,
        info: SourceInfo,
        stop: Arc<AtomicBool>,
    }

    impl GstreamerNvmmCameraSource {
        /// Open a stereo NVMM camera source.
        pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
            gst::init().map_err(|e| SourceError::Init {
                path: format!("{} + {}", config.left_device, config.right_device),
                reason: format!("GStreamer init: {e}"),
            })?;

            let stop = Arc::new(AtomicBool::new(false));

            let (left_rx, left_release) = spawn_nvmm_capture_thread(
                config.left_device.clone(),
                "left",
                config.clone(),
                stop.clone(),
            );
            let (right_rx, right_release) = spawn_nvmm_capture_thread(
                config.right_device.clone(),
                "right",
                config.clone(),
                stop.clone(),
            );

            let info = SourceInfo {
                width: config.width,
                height: config.height,
                fps: config.fps as f64,
                fps_rational: None,
                total_frames: None,
            };

            log::info!(
                "NVMM stereo source: {}x{} @ {}fps, zero-copy DMA-buf + NvBufSurfTransform",
                config.width,
                config.height,
                config.fps,
            );

            Ok(Self {
                left_rx,
                right_rx,
                left_release,
                right_release,
                info,
                stop,
            })
        }

        /// Get the next stereo NVMM frame pair.
        ///
        /// Blocks until both cameras deliver a frame. Returns `None` on EOS.
        pub fn next_pair(&mut self) -> Result<Option<NvmmFramePair>, SourceError> {
            let left = match self.left_rx.recv() {
                Ok(info) => info,
                Err(_) => return Ok(None),
            };
            let right = match self.right_rx.recv() {
                Ok(info) => info,
                Err(_) => return Ok(None),
            };
            Ok(Some(NvmmFramePair { left, right }))
        }

        /// Signal capture threads that GPU work on the current frame is done.
        ///
        /// Must be called after rendering + detection are submitted for the
        /// frame returned by the most recent `next_pair()`.
        pub fn release_previous(&self) {
            let _ = self.left_release.send(());
            let _ = self.right_release.send(());
        }

        /// Source metadata.
        pub fn info(&self) -> SourceInfo {
            self.info.clone()
        }

        /// Stop capture gracefully.
        pub fn stop(&self) {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    impl Drop for GstreamerNvmmCameraSource {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            // Drain pending frames and send release signals to unblock threads
            while self.left_rx.try_recv().is_ok() {}
            while self.right_rx.try_recv().is_ok() {}
            let _ = self.left_release.send(());
            let _ = self.right_release.send(());
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate_device_string;

    // Helper: on non-Jetson Linux builds (the CI environment), V4L2 rules apply.
    // On macOS/Windows CI the numeric rules apply. We test the logic that is
    // actually compiled in, plus explicitly call the Tegra/numeric branch via
    // the shared predicate (numeric-only) which is the same for all three
    // non-V4L2 platforms.

    /// Numeric strings must be accepted on every non-V4L2 platform.
    #[test]
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn numeric_index_accepted() {
        assert!(validate_device_string("0").is_ok());
        assert!(validate_device_string("1").is_ok());
        assert!(validate_device_string("12").is_ok());
    }

    /// Non-numeric strings must be rejected on macOS/Windows.
    #[test]
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn non_numeric_rejected_on_non_linux() {
        assert!(validate_device_string("").is_err());
        assert!(validate_device_string("cam0").is_err());
        assert!(validate_device_string("0 ! fakesrc").is_err());
        assert!(validate_device_string("/dev/video0").is_err());
    }

    /// Valid V4L2 paths must be accepted on Linux (non-Tegra).
    #[test]
    #[cfg(target_os = "linux")]
    fn v4l2_path_accepted() {
        assert!(validate_device_string("/dev/video0").is_ok());
        assert!(validate_device_string("/dev/video1").is_ok());
        assert!(validate_device_string("/dev/video10").is_ok());
    }

    /// Injection attempts and malformed paths must be rejected on Linux.
    #[test]
    #[cfg(target_os = "linux")]
    fn injection_rejected_on_linux() {
        assert!(validate_device_string("").is_err());
        assert!(validate_device_string("/dev/video").is_err()); // no trailing digit
        assert!(validate_device_string("/dev/video0 ! fakesrc").is_err());
        assert!(validate_device_string("0").is_err()); // numeric-only not valid for V4L2
        assert!(validate_device_string("/dev/video0a").is_err());
        assert!(validate_device_string("/dev/../etc/passwd").is_err());
        assert!(validate_device_string("video0").is_err());
    }
}
