//! Stereo NVMM capture with frame-owned native buffers.

use super::camera::{CameraConfig, validate_device_string};
use super::capture::CapturePair;
use reco_core::source::{FrameSet, FrameSource, NvmmFrame, SourceError, SourceFrame, SourceInfo};
use std::sync::{Arc, atomic::AtomicBool};

fn pipeline_string(
    device: &str,
    config: &CameraConfig,
    file: Option<&str>,
) -> Result<String, String> {
    if let Some(path) = file {
        if config.synchronization.is_some() {
            return Err("recorded NVMM inputs cannot use live synchronization".into());
        }
        let path = path.replace('\\', "\\\\").replace('"', "\\\"");
        return Ok(format!(
            "filesrc location=\"{path}\" ! qtdemux ! h264parse ! nvv4l2decoder ! \
             video/x-raw(memory:NVMM),format=NV12 ! appsink name=sink sync=false"
        ));
    }
    validate_device_string(device)?;
    let opts = std::env::var("RECO_NVARGUS_OPTS").unwrap_or_default();
    Ok(format!(
        "nvarguscamerasrc sensor-id={device} {opts} ! \
         video/x-raw(memory:NVMM),width={},height={},format=NV12,framerate={}/1 ! \
         appsink name=sink sync=false",
        config.width, config.height, config.fps
    ))
}

/// Stereo Jetson source. Each returned frame retains its mapped NVMM buffer
/// independently of subsequent capture calls and source shutdown.
pub struct GstreamerNvmmCameraSource {
    capture: CapturePair<NvmmFrame>,
    info: SourceInfo,
    interrupted: Arc<AtomicBool>,
}

impl GstreamerNvmmCameraSource {
    /// Open both cameras; failures stop any partially initialized pipeline.
    /// Recorded-input overrides cannot use live synchronization (`tests::recorded_nvmm_input_rejects_live_synchronization`).
    pub fn open(config: &CameraConfig, interrupted: Arc<AtomicBool>) -> Result<Self, SourceError> {
        let build = |device: &str, key: &str| {
            pipeline_string(device, config, std::env::var(key).ok().as_deref()).map_err(|reason| {
                SourceError::Init {
                    path: device.into(),
                    reason,
                }
            })
        };
        let left = build(&config.left_device, "RECO_NVMM_LEFT")?;
        let right = build(&config.right_device, "RECO_NVMM_RIGHT")?;
        let capture = CapturePair::start(
            [&left, &right],
            super::nvmm::extract_sample,
            config.synchronization.map(|policy| (policy, config.fps)),
        )?;
        Ok(Self {
            capture,
            info: SourceInfo {
                width: config.width,
                height: config.height,
                fps: f64::from(config.fps),
                fps_rational: None,
                total_frames: None,
            },
            interrupted,
        })
    }

    /// Stop the capture pipeline and join both capture workers.
    pub fn stop(&mut self) -> Result<(), SourceError> {
        self.capture.stop()
    }
}

impl FrameSource for GstreamerNvmmCameraSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }
    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        self.capture.next(&self.interrupted).map(|pair| {
            pair.map(|pair| SourceFrame {
                images: FrameSet::NvmmResident(pair.images),
                pts_us: Some(pair.pts_us),
                capture: Some(pair.capture),
            })
        })
    }
    fn is_exhausted(&self) -> bool {
        self.capture.is_stopped()
    }
    fn is_gpu_resident(&self) -> bool {
        true
    }
    fn gpu_pixel_format(&self) -> reco_core::render::renderer::GpuPixelFormat {
        reco_core::render::renderer::GpuPixelFormat::Nv12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_nvmm_input_rejects_live_synchronization() {
        let mut config = CameraConfig {
            width: 64,
            height: 32,
            fps: 30,
            left_device: "0".into(),
            right_device: "1".into(),
            synchronization: Some(crate::frame_sync::Config {
                wait_us: 20_000,
                skew_us: 20_000,
                max_age_us: 150_000,
                capacity: 2,
            }),
        };
        for device in [&config.left_device, &config.right_device] {
            let error = pipeline_string(device, &config, Some("recorded.mp4"))
                .expect_err("recorded PTS must not be assessed against the live capture clock");
            assert!(error.contains("live synchronization"), "{error}");
        }
        config.synchronization = None;
        assert!(pipeline_string("0", &config, Some("recorded.mp4")).is_ok());
    }

    #[test]
    fn nvmm_extraction_error_is_not_eof_even_with_stalled_peer() {
        let finite = "videotestsrc num-buffers=1 ! video/x-raw,format=NV12,width=64,height=32 ! appsink name=sink sync=false";
        let idle = "appsrc is-live=true ! appsink name=sink sync=false";
        for pipelines in [[finite, idle], [idle, finite]] {
            let capture =
                CapturePair::start(pipelines, super::super::nvmm::extract_sample, None).unwrap();
            let mut source = GstreamerNvmmCameraSource {
                capture,
                interrupted: Arc::new(AtomicBool::new(false)),
                info: SourceInfo {
                    width: 64,
                    height: 32,
                    fps: 30.0,
                    fps_rational: None,
                    total_frames: None,
                },
            };
            // Deliberate system-memory sample cannot masquerade as native metadata.
            assert!(source.next_frame().is_err());
            assert!(source.is_exhausted());
            assert!(source.next_frame().unwrap().is_none());
            source.stop().unwrap();
        }
    }

    #[test]
    fn nvmm_cancel_stalled_capture_and_drop_are_joined() {
        let idle = "appsrc is-live=true ! appsink name=sink sync=false";
        let capture =
            CapturePair::start([idle, idle], super::super::nvmm::extract_sample, None).unwrap();
        let interrupted = Arc::new(AtomicBool::new(false));
        let mut source = GstreamerNvmmCameraSource {
            capture,
            interrupted: interrupted.clone(),
            info: SourceInfo {
                width: 64,
                height: 32,
                fps: 30.0,
                fps_rational: None,
                total_frames: None,
            },
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            assert!(source.next_frame().unwrap().is_none());
            assert!(source.is_exhausted());
            drop(source);
            tx.send(()).unwrap();
        });
        interrupted.store(true, std::sync::atomic::Ordering::Relaxed);
        rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
    }
}
