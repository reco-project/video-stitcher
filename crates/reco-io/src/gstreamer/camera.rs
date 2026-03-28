//! Live stereo camera capture via GStreamer.
//!
//! Builds a GStreamer pipeline per camera, pulls I420 frames from
//! `appsink`, and pairs them into stereo `FramePair`s.

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use reco_core::source::{FramePair, SourceError, SourceInfo, YuvData};
use std::path::Path;

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

/// A single-camera GStreamer capture pipeline.
struct CameraPipeline {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    width: u32,
    height: u32,
}

impl CameraPipeline {
    /// Build and start a capture pipeline for one camera.
    fn new(device: &str, width: u32, height: u32, fps: u32) -> Result<Self, SourceError> {
        gst::init().map_err(|e| SourceError::Init(format!("GStreamer init: {e}")))?;

        let pipeline_str = Self::build_pipeline_string(device, width, height, fps);
        log::info!("GStreamer pipeline: {pipeline_str}");

        let pipeline = gst::parse::launch(&pipeline_str)
            .map_err(|e| SourceError::Init(format!("pipeline parse: {e}")))?
            .downcast::<gst::Pipeline>()
            .map_err(|_| SourceError::Init("not a pipeline".into()))?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| SourceError::Init("appsink 'sink' not found".into()))?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| SourceError::Init("element is not an AppSink".into()))?;

        // Drop old frames, keep buffer small for low latency
        appsink.set_max_buffers(2);
        appsink.set_drop(true);

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| SourceError::Init(format!("set_state(Playing): {e}")))?;

        Ok(Self {
            pipeline,
            appsink,
            width,
            height,
        })
    }

    /// Build the platform-appropriate pipeline string.
    fn build_pipeline_string(device: &str, width: u32, height: u32, fps: u32) -> String {
        if Self::is_jetson() {
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

    /// Pull one I420 frame from the appsink. Blocks until a frame arrives.
    fn pull_frame(&self) -> Result<Option<YuvData>, SourceError> {
        let sample = match self.appsink.pull_sample() {
            Ok(s) => s,
            Err(_) => return Ok(None), // EOS or pipeline stopped
        };

        let buffer = sample
            .buffer()
            .ok_or_else(|| SourceError::Read("sample has no buffer".into()))?;

        let map = buffer
            .map_readable()
            .map_err(|e| SourceError::Read(format!("buffer map: {e}")))?;

        let data = map.as_slice();

        // I420 layout: Y plane (w*h), U plane (w/2 * h/2), V plane (w/2 * h/2)
        let y_size = (self.width * self.height) as usize;
        let uv_size = ((self.width / 2) * (self.height / 2)) as usize;

        if data.len() < y_size + 2 * uv_size {
            return Err(SourceError::Read(format!(
                "buffer too small: {} < {}",
                data.len(),
                y_size + 2 * uv_size
            )));
        }

        Ok(Some(YuvData {
            y: data[..y_size].to_vec(),
            u: data[y_size..y_size + uv_size].to_vec(),
            v: data[y_size + uv_size..y_size + 2 * uv_size].to_vec(),
        }))
    }
}

impl Drop for CameraPipeline {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Stereo camera source using GStreamer.
///
/// Captures from two cameras simultaneously and delivers paired I420
/// frames. On Jetson, uses `nvarguscamerasrc` which runs the full
/// NVIDIA ISP pipeline (debayer, AWB, AE, denoise). On desktop Linux,
/// uses `v4l2src`.
pub struct GstreamerCameraSource {
    left: CameraPipeline,
    right: CameraPipeline,
    info: SourceInfo,
}

impl GstreamerCameraSource {
    /// Open a stereo camera source.
    pub fn open(config: &CameraConfig) -> Result<Self, SourceError> {
        let left =
            CameraPipeline::new(&config.left_device, config.width, config.height, config.fps)?;
        let right = CameraPipeline::new(
            &config.right_device,
            config.width,
            config.height,
            config.fps,
        )?;

        let info = SourceInfo {
            width: config.width,
            height: config.height,
            fps: config.fps as f64,
        };

        log::info!(
            "Camera source ready: {}x{} @ {} fps",
            config.width,
            config.height,
            config.fps
        );

        Ok(Self { left, right, info })
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
        let left = match self.left.pull_frame()? {
            Some(f) => f,
            None => return Ok(None),
        };
        let right = match self.right.pull_frame()? {
            Some(f) => f,
            None => return Ok(None),
        };
        Ok(Some(FramePair { left, right }))
    }
}
