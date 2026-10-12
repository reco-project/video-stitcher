//! Live stereo camera capture via GStreamer.
//!
//! Both cameras share a pipeline clock and base time, with separate bounded appsink queues.
//! Host capture owns parallel extraction workers; NVMM retains samples in capture workers.
//! Without a synchronization policy, presentation follows the left camera's segment-adjusted sample clock. With a policy, bounded timestamp matching and reuse follow the configured cadence. Both normalize presentation time to the first delivered pair while preserving original capture times. Missing/non-advancing timestamps, clock loss and exceeded reuse limits stop capture. Neither mode synchronizes camera exposures or independently recorded file timelines. The `capture::sync::tests` and `tests::pipeline::bounded_live_reuse_survives_inference_lookahead_and_encoding` regressions exercise matching, retention and downstream image identity.
//!
//! Supports two capture modes:
//! - **I420** (YUV420P): three separate planes, works everywhere
//! - **NV12**: Y + interleaved UV, native NVIDIA ISP output - skips
//!   format conversion on Jetson for lower latency

use gstreamer as gst;
use reco_core::source::{FrameSet, Nv12Data, SourceError, SourceFrame, SourceInfo, Yuv420pData};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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
    /// Optional bounded matching/reuse policy. None preserves arrival-order capture without an age guarantee.
    pub synchronization: Option<crate::frame_sync::Config>,
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
pub(super) fn validate_device_string(device: &str) -> Result<(), String> {
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
    if fps == 0 || fps > i32::MAX as u32 {
        return Err("capture frame rate must be in 1..=i32::MAX".into());
    }

    let fmt_str = match format {
        CaptureFormat::I420 => "I420",
        CaptureFormat::Nv12 => "NV12",
    };

    let pipeline = if is_tegra() {
        // Jetson: nvarguscamerasrc runs the full NVIDIA ISP
        // (debayer, AWB, AE, denoise). Output is NV12 in NVMM;
        // nvvidconv copies to system memory (and converts format if needed).
        // RECO_NVARGUS_OPTS injects extra source properties (e.g. aelock,
        // exposuretimerange, gainrange, wbmode) for seam-matched capture.
        let nvargus_opts = std::env::var("RECO_NVARGUS_OPTS").unwrap_or_default();
        format!(
            "nvarguscamerasrc sensor-id={device} {nvargus_opts} ! \
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
fn extract_i420(data: &[u8], width: u32, height: u32) -> Result<Yuv420pData, SourceError> {
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

    Ok(Yuv420pData {
        size: (width, height),
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
        size: (width, height),
        y: data[..y_size].to_vec(),
        uv: data[y_size..y_size + uv_size].to_vec(),
    })
}

struct HostCapture<T> {
    pair: super::capture::CapturePair<Arc<T>>,
    info: SourceInfo,
    interrupted: Arc<AtomicBool>,
}

impl<T: Send + Sync + 'static> HostCapture<T> {
    fn open(
        config: &CameraConfig,
        format: CaptureFormat,
        interrupted: Arc<AtomicBool>,
        extract: fn(&[u8], u32, u32) -> Result<T, SourceError>,
    ) -> Result<Self, SourceError> {
        let descriptions = [&config.left_device, &config.right_device].map(|device| {
            build_pipeline_string(device, config.width, config.height, config.fps, format).map_err(
                |reason| SourceError::Init {
                    path: device.clone(),
                    reason,
                },
            )
        });
        let [left, right] = descriptions;
        let (width, height) = (config.width, config.height);
        let pair = super::capture::CapturePair::start(
            [&left?, &right?],
            move |sample| extract_sample(sample, width, height, extract).map(Arc::new),
            config.synchronization.map(|policy| (policy, config.fps)),
        )?;
        Ok(Self {
            pair,
            info: SourceInfo {
                camera_sizes: vec![(config.width, config.height); 2],
                fps: config.fps as f64,
                fps_rational: Some((config.fps as i32, 1)),
                total_frames: None,
            },
            interrupted,
        })
    }

    fn next(&mut self) -> Result<Option<super::capture::CapturedPair<Arc<T>>>, SourceError> {
        self.pair.next(&self.interrupted)
    }
}

fn extract_sample<T>(
    sample: &gst::Sample,
    width: u32,
    height: u32,
    extract: fn(&[u8], u32, u32) -> Result<T, SourceError>,
) -> Result<T, SourceError> {
    let buffer = sample.buffer().ok_or_else(|| SourceError::Read {
        reason: "capture sample has no buffer".into(),
    })?;
    let map = buffer.map_readable().map_err(|e| SourceError::Read {
        reason: format!("capture buffer map failed: {e}"),
    })?;
    extract(map.as_slice(), width, height)
}

/// Stereo I420 camera capture using GStreamer's bounded appsink queues.
/// Pairing follows arrival order, not capture timestamps.
pub struct GstreamerCameraSource {
    capture: HostCapture<Yuv420pData>,
}

impl GstreamerCameraSource {
    /// Start both cameras. Capture waits observe the supplied cancellation flag.
    /// Pipeline initialization failures are returned directly.
    pub fn open(config: &CameraConfig, interrupted: Arc<AtomicBool>) -> Result<Self, SourceError> {
        Ok(Self {
            capture: HostCapture::open(config, CaptureFormat::I420, interrupted, extract_i420)?,
        })
    }
}

impl reco_core::source::FrameSource for GstreamerCameraSource {
    fn info(&self) -> SourceInfo {
        self.capture.info.clone()
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        self.capture.next().map(|pair| {
            pair.map(|pair| SourceFrame {
                images: FrameSet::Yuv420p(pair.images.into_iter().collect()),
                pts_us: Some(pair.pts_us),
                capture: Some(pair.capture),
            })
        })
    }

    fn is_exhausted(&self) -> bool {
        self.capture.pair.is_stopped()
    }
}

/// Stereo host-memory NV12 capture using GStreamer's bounded appsink queues.
/// NV12 avoids the ISP's NV12-to-I420 conversion on Jetson.
/// Pairing follows arrival order, not capture timestamps.
pub struct GstreamerNv12CameraSource {
    capture: HostCapture<Nv12Data>,
}

impl GstreamerNv12CameraSource {
    /// Start both cameras. Capture waits observe the supplied cancellation flag.
    /// Pipeline initialization failures are returned directly.
    pub fn open(config: &CameraConfig, interrupted: Arc<AtomicBool>) -> Result<Self, SourceError> {
        Ok(Self {
            capture: HostCapture::open(config, CaptureFormat::Nv12, interrupted, extract_nv12)?,
        })
    }

    /// Stop the capture pipeline and join both extraction workers.
    pub fn stop(&mut self) -> Result<(), SourceError> {
        self.capture.pair.stop()
    }

    /// Source metadata.
    pub fn info(&self) -> SourceInfo {
        self.capture.info.clone()
    }

    /// Get the next pair, or `None` after EOS or cancellation.
    /// For stereo calibration; pipeline consumers use `FrameSource::next_frame`.
    pub fn next_pair(&mut self) -> Result<Option<[Arc<Nv12Data>; 2]>, SourceError> {
        self.capture.next().map(|pair| pair.map(|pair| pair.images))
    }
}

impl reco_core::source::FrameSource for GstreamerNv12CameraSource {
    fn info(&self) -> SourceInfo {
        self.capture.info.clone()
    }

    #[cfg(feature = "gpu")]
    fn input_format(&self) -> reco_core::render::renderer::InputFormat {
        reco_core::render::renderer::InputFormat::Nv12
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        self.capture.next().map(|pair| {
            pair.map(|pair| SourceFrame {
                images: FrameSet::Nv12(pair.images.into_iter().collect()),
                pts_us: Some(pair.pts_us),
                capture: Some(pair.capture),
            })
        })
    }

    fn is_exhausted(&self) -> bool {
        self.capture.pair.is_stopped()
    }
}

// ── NVMM zero-copy camera source (Jetson only) ─────────────────────
#[cfg(all(feature = "gpu", target_os = "linux"))]
pub use super::nvmm_camera::GstreamerNvmmCameraSource;

#[cfg(test)]
mod tests {
    use super::validate_device_string;

    #[cfg(feature = "ffmpeg")]
    mod pipeline;

    fn synthetic_capture<T: Send + Sync + 'static>(
        format: &str,
        extract: fn(&[u8], u32, u32) -> Result<T, reco_core::source::SourceError>,
    ) -> super::HostCapture<T> {
        use super::*;
        let description = format!(
            "videotestsrc num-buffers=2 pattern=black ! video/x-raw,format={format},width=64,height=32,framerate=10/1 ! appsink name=sink sync=false"
        );
        HostCapture {
            pair: super::super::capture::CapturePair::start(
                [&description, &description],
                move |sample| extract_sample(sample, 64, 32, extract).map(Arc::new),
                None,
            )
            .unwrap(),
            info: SourceInfo {
                camera_sizes: vec![(64, 32); 2],
                fps: 30.0,
                fps_rational: None,
                total_frames: None,
            },
            interrupted: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn host_sources_deliver_both_formats_and_report_terminal_exhaustion() {
        use super::*;
        use reco_core::source::FrameSource;
        let sources: [Box<dyn FrameSource>; 2] = [
            Box::new(GstreamerCameraSource {
                capture: synthetic_capture("I420", extract_i420),
            }),
            Box::new(GstreamerNv12CameraSource {
                capture: synthetic_capture("NV12", extract_nv12),
            }),
        ];
        for mut source in sources {
            assert!(!source.is_exhausted());
            for index in 0..2 {
                let frame = source.next_frame().unwrap().unwrap();
                assert_eq!(frame.pts_us, Some(index * 100_000));
                match frame.images {
                    FrameSet::Yuv420p(frames) => {
                        assert_eq!(frames.len(), 2);
                        for frame in frames {
                            assert_eq!(frame.y, vec![16; 64 * 32]);
                            assert_eq!(frame.u, vec![128; 32 * 16]);
                            assert_eq!(frame.v, vec![128; 32 * 16]);
                        }
                    }
                    FrameSet::Nv12(frames) => {
                        assert_eq!(frames.len(), 2);
                        for frame in frames {
                            assert_eq!(frame.y, vec![16; 64 * 32]);
                            assert_eq!(frame.uv, vec![128; 64 * 16]);
                        }
                    }
                    _ => panic!("expected owned host planes"),
                }
            }
            assert!(source.next_frame().unwrap().is_none());
            assert!(source.is_exhausted());
        }
    }

    #[test]
    fn extraction_failure_stops_capture_and_explicit_stop_preserves_cancellation_flag() {
        use super::*;
        use reco_core::source::FrameSource;
        use std::sync::atomic::Ordering;
        let mut source = GstreamerNv12CameraSource {
            capture: synthetic_capture("NV12", |data, width, height| {
                extract_nv12(data, width * 2, height)
            }),
        };
        // Force a truncated-plane error through the production extraction path.
        assert!(source.next_frame().is_err());
        assert!(source.is_exhausted());
        source.stop().unwrap();
        source.stop().unwrap();
        assert!(!source.capture.interrupted.load(Ordering::Relaxed));
    }

    #[test]
    fn invalid_capture_input_is_error_not_eof() {
        use super::*;
        use reco_core::source::FrameSource;
        let config = CameraConfig {
            width: 64,
            height: 32,
            fps: 30,
            left_device: "invalid-device".into(),
            right_device: "invalid-device".into(),
            synchronization: None,
        };
        let result = GstreamerCameraSource::open(&config, Arc::new(AtomicBool::new(false)))
            .and_then(|mut source| source.next_frame());
        assert!(
            result.is_err(),
            "capture failure must not be successful EOF"
        );
        let result = GstreamerNv12CameraSource::open(&config, Arc::new(AtomicBool::new(false)))
            .and_then(|mut source| source.next_pair());
        assert!(
            result.is_err(),
            "capture failure must not be successful EOF"
        );
    }

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
