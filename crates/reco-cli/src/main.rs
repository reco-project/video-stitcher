//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

mod calibrate;
#[cfg(feature = "gstreamer")]
mod camera;
mod helpers;
#[cfg(feature = "libcamera")]
mod libcamera_cmd;
mod preview;
mod stitch;

use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Initialize the tracing profiler. Returns a guard that must be held
/// until the end of `main()` — the trace file is written on drop.
#[cfg(feature = "profiling")]
fn init_profiling() -> tracing_chrome::FlushGuard {
    use tracing_subscriber::prelude::*;
    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file("reco-trace.json")
        .include_args(true)
        .build();
    tracing_subscriber::registry().with(chrome_layer).init();
    eprintln!("Profiling enabled — trace will be written to reco-trace.json");
    guard
}

/// Install the standard tracing subscriber for the non-profiling path.
///
/// Routes legacy `log::*` macro calls from reco-core / reco-io /
/// reco-calibrate into the tracing pipeline so a single structured
/// source of truth carries every event. Reads `RUST_LOG` for level
/// filtering; defaults to `info` if unset.
///
/// M2 migration: replaces the previous `env_logger::init()`.
#[cfg(not(feature = "profiling"))]
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    // Bridge legacy `log::*` calls into tracing. Ignores "already set"
    // errors so tests that construct multiple CLIs don't panic.
    let _ = tracing_log::LogTracer::init();

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = fmt::layer().with_target(true).with_level(true);

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init();
}

/// Install a panic hook that captures the panic context (location,
/// payload) as a structured `tracing::error!` event before the default
/// hook runs. When a user reports a bug post-deployment with a log
/// file, the panic context is immediately searchable alongside regular
/// log lines.
///
/// M2 addition: required for the post-deployment diagnostic story the
/// user flagged during the plan iteration on 2026-04-18.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".into()
        };
        tracing::error!(
            target: "panic",
            location = %location,
            payload = %payload,
            "panic caught by tracing panic hook"
        );
        default_hook(info);
    }));
}

#[derive(Parser)]
#[command(
    name = "reco",
    version,
    about = "GPU-accelerated panoramic video stitching",
    long_about = "Reco stitches two camera feeds into a seamless panoramic sports view.\n\
                  Designed for sports filming with open-source hardware flexibility."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Stitch two video files into a panoramic output.
    Stitch {
        /// Path to the left camera video file.
        left: String,

        /// Path to the right camera video file.
        right: String,

        /// Path to the calibration JSON file (v1-compatible match format).
        #[arg(short, long)]
        calibration: String,

        /// Output file path.
        #[arg(short, long, default_value = "output.mp4")]
        output: String,

        /// Output width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,

        /// Output height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,

        /// Maximum number of seconds to process.
        #[arg(long)]
        duration: Option<f64>,

        /// Maximum number of frames to process.
        #[arg(long)]
        max_frames: Option<u64>,

        /// Force a specific encoder (e.g., h264_nvenc, hevc_nvenc, libx264). Auto-detects by default.
        #[arg(long)]
        encoder: Option<String>,

        /// Output codec: h264, hevc, av1. Default: h264.
        #[arg(long, default_value = "h264")]
        codec: String,

        /// Quality preset: fast, balanced, high.
        #[arg(long, default_value = "balanced")]
        quality: String,

        /// Seam blend width (0.0–1.0). Controls how much the two camera views
        /// are blended at the seam. 0 = hard edge, 0.15 = smooth transition.
        #[arg(long, default_value_t = 0.15, value_parser = parse_blend)]
        blend: f32,

        /// Frame offset for temporal sync between cameras.
        /// Positive: skip N right frames (right started first).
        /// Negative: skip N left frames (left started first).
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        sync_offset: i64,

        /// Path to a YOLO ONNX model for ball detection and auto-panning.
        /// When provided, enables automatic camera direction that follows the ball.
        #[arg(long)]
        model: Option<String>,

        /// Run detection every N frames (default: 1 = every frame).
        /// Higher values reduce detection overhead. The director uses
        /// the last known detections on skipped frames.
        #[arg(long, default_value_t = 1)]
        detection_interval: u64,

        /// Director lead time in seconds. Buffers decoded frames and runs
        /// detection ahead of rendering so the camera anticipates action.
        /// Typical value: 0.5 (half a second). Only works with CPU path.
        #[arg(long, default_value_t = 0.0)]
        lead_time: f64,

        /// Tracking mode: "ball" (single ball), "field" (ball + players).
        /// Use "field" with a COCO model for robust football tracking.
        #[arg(long, default_value = "ball")]
        tracking: String,

        /// Override encoder CRF/quality value (lower = better, typical 18-28).
        #[arg(long)]
        crf: Option<u8>,

        /// Override encoder preset (e.g. ultrafast, veryfast, fast for x264; p1-p7 for NVENC).
        #[arg(long)]
        preset: Option<String>,

        /// Output container format. One of: `mp4` (default,
        /// finalized at close), `fmp4` (fragmented MP4, readable
        /// mid-write), `mkv` (Matroska, crash-safe + streamable).
        /// Pick `mkv` or `fmp4` if you plan to tee the output to
        /// RTMP via a second ffmpeg with `-c copy` while the
        /// stitch is still running.
        #[arg(long)]
        container: Option<String>,

        /// Record pre-stitch source frames to this path as a
        /// stacked-video file for later replay or cloud upload.
        /// Requires building with `--features replay`.
        #[arg(long)]
        replay: Option<String>,

        /// Downscale replay tiles to `WIDTHxHEIGHT` (e.g.
        /// `1280x720`, `854x480`). Reduces replay file size and
        /// CPU encode cost (libx264 at 1080p stacked crawls on
        /// ARM / Jetson). GPU pack path only: CPU-resident
        /// sources log a warn and record at source dims. Width
        /// must be divisible by 4; height must be even. Requires
        /// `--replay`.
        #[arg(long, value_parser = parse_wxh)]
        replay_scale: Option<(u32, u32)>,
    },

    /// Open an interactive preview window to debug the stitch.
    Preview {
        /// Path to the left camera video file.
        left: String,

        /// Path to the right camera video file.
        right: String,

        /// Path to the calibration JSON file (v1-compatible match format).
        #[arg(short, long)]
        calibration: String,

        /// Window width in pixels.
        #[arg(long, default_value_t = 1280)]
        width: u32,

        /// Window height in pixels.
        #[arg(long, default_value_t = 720)]
        height: u32,

        /// Frame offset to sync left/right videos.
        /// Positive: skip N right frames (right started first).
        /// Negative: skip N left frames (left started first).
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        sync_offset: i64,

        /// Seam blend width (0.0 = hard cut, 0.15 = default smooth blend).
        #[arg(long, default_value_t = 0.15)]
        blend: f32,

        /// Rig tilt in degrees. Rotates the entire scene to compensate for
        /// a tilted camera rig, straightening vertical lines at the edges.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        rig_tilt: f32,
    },

    /// Stitch live camera feeds in real time.
    #[cfg(feature = "gstreamer")]
    Camera {
        /// Left camera device (sensor ID on Jetson, e.g. "0"; device path on Linux, e.g. "/dev/video0").
        #[arg(long)]
        left_device: String,

        /// Right camera device.
        #[arg(long)]
        right_device: String,

        /// Path to the calibration JSON file.
        #[arg(short, long)]
        calibration: String,

        /// Output file path.
        #[arg(short, long)]
        output: String,

        /// Capture width in pixels.
        #[arg(long, default_value_t = 3840)]
        capture_width: u32,

        /// Capture height in pixels.
        #[arg(long, default_value_t = 2160)]
        capture_height: u32,

        /// Capture frame rate.
        #[arg(long, default_value_t = 30)]
        capture_fps: u32,

        /// Output width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,

        /// Output height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,

        /// Force a specific encoder (e.g. "libx264", "h264_nvenc").
        #[arg(long)]
        encoder: Option<String>,

        /// Output codec: h264, hevc, av1.
        #[arg(long, default_value = "h264")]
        codec: String,

        /// Quality preset: fast, balanced, high.
        #[arg(long, default_value = "fast")]
        quality: String,

        /// Seam blend width (0.0-1.0).
        #[arg(long, default_value_t = 0.15, value_parser = parse_blend)]
        blend: f32,

        /// Maximum number of frames to capture.
        #[arg(long)]
        max_frames: Option<u64>,

        /// Duration in seconds to capture.
        #[arg(long)]
        duration: Option<f64>,

        /// Path to a YOLO ONNX model for ball detection and auto-tracking.
        #[arg(long)]
        model: Option<String>,

        /// Detection interval: run detection every N frames (default: 1).
        #[arg(long, default_value_t = 1)]
        detection_interval: u64,

        /// Override encoder CRF/quality value (lower = better, typical 18-28).
        #[arg(long)]
        crf: Option<u8>,

        /// Override encoder preset (e.g. ultrafast, veryfast, fast for x264; p1-p7 for NVENC).
        #[arg(long)]
        preset: Option<String>,

        /// Output container format. One of: `mp4` (default, needs
        /// close-time finalize), `fmp4` (fragmented MP4, streamable
        /// mid-write), `mkv` (Matroska, crash-safe + streamable).
        /// Use `mkv` or `fmp4` if you plan to stream the output
        /// via an external `ffmpeg -c copy -f flv rtmp://...`
        /// tee while the capture is still running.
        #[arg(long)]
        container: Option<String>,

        /// Tracking director mode. `ball` (default): YOLO ball
        /// tracking via BallDirector. `field`: ball + players
        /// (FieldDirector, for multi-class models). `sweep`:
        /// no AI, slow left-right pan across the full coverage
        /// (debug / demo). Sweep mode doesn't require `--model`.
        #[arg(long, default_value = "ball")]
        tracking: String,

        /// Disable the constrained-look coverage clamp
        /// (FRICTION A13). When off, the viewport can pan into
        /// black panorama edges — useful for sweeping the full
        /// coverage or debugging the coverage boundary itself.
        #[arg(long, default_value_t = false)]
        unconstrained: bool,

        /// Record pre-stitch source frames to this path as a
        /// stacked-video file. Same M7 replay feature as
        /// `stitch --replay`. Requires `--features replay`.
        #[arg(long)]
        replay: Option<String>,

        /// Downscale replay tiles to `WIDTHxHEIGHT` (e.g.
        /// `1280x720`, `856x480`). GPU pack path only; width
        /// divisible by 4, height even. Requires `--replay`.
        #[arg(long, value_parser = parse_wxh)]
        replay_scale: Option<(u32, u32)>,
    },

    /// Stitch live RPi CSI camera feeds via libcamera (rpicam-vid).
    #[cfg(feature = "libcamera")]
    Libcamera {
        /// Left camera index (e.g. 0).
        #[arg(long, default_value_t = 0)]
        left_camera: u32,

        /// Right camera index (e.g. 1).
        #[arg(long, default_value_t = 1)]
        right_camera: u32,

        /// Path to the calibration JSON file.
        #[arg(short, long)]
        calibration: String,

        /// Output file path.
        #[arg(short, long)]
        output: String,

        /// Capture width in pixels.
        #[arg(long, default_value_t = 1920)]
        capture_width: u32,

        /// Capture height in pixels.
        #[arg(long, default_value_t = 1080)]
        capture_height: u32,

        /// Capture frame rate.
        #[arg(long, default_value_t = 30)]
        capture_fps: u32,

        /// Output width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,

        /// Output height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,

        /// Force a specific encoder (e.g. "libx264", "h264_v4l2m2m").
        #[arg(long)]
        encoder: Option<String>,

        /// Output codec: h264, hevc, av1.
        #[arg(long, default_value = "h264")]
        codec: String,

        /// Quality preset: fast, balanced, high.
        #[arg(long, default_value = "fast")]
        quality: String,

        /// Seam blend width (0.0-1.0).
        #[arg(long, default_value_t = 0.15, value_parser = parse_blend)]
        blend: f32,

        /// Maximum number of frames to capture.
        #[arg(long)]
        max_frames: Option<u64>,

        /// Duration in seconds to capture.
        #[arg(long)]
        duration: Option<f64>,

        /// Path to a YOLO model for ball detection and auto-tracking.
        #[arg(long)]
        model: Option<String>,

        /// Detection interval: run detection every N frames (default: 1).
        #[arg(long, default_value_t = 1)]
        detection_interval: u64,

        /// Override encoder CRF/quality value (lower = better, typical 18-28).
        #[arg(long)]
        crf: Option<u8>,

        /// Override encoder preset (e.g. ultrafast, veryfast, fast for x264).
        #[arg(long)]
        preset: Option<String>,
    },

    /// Calibrate two cameras: detect features and compute placement parameters.
    Calibrate {
        /// Path to the left camera video file.
        left: String,

        /// Path to the right camera video file.
        right: String,

        /// Path to the left camera lens profile JSON.
        /// If omitted, auto-detects from video metadata using the
        /// bundled Gyroflow lens profile database (4200+ profiles).
        #[arg(long)]
        left_profile: Option<String>,

        /// Path to the right camera lens profile JSON.
        /// If omitted, uses the same profile as --left-profile, or
        /// auto-detects from video metadata.
        #[arg(long)]
        right_profile: Option<String>,

        /// Number of frame pairs to sample from the video.
        #[arg(long, default_value_t = 2)]
        frames: usize,

        /// Disable IMU telemetry extraction (sync offset, rig tilt/roll,
        /// rotation seeds). Use when IMU data is unavailable or unreliable.
        #[arg(long, default_value_t = false)]
        no_auto_imu: bool,

        /// Auto-detect sync offset from audio cross-correlation.
        /// Used as fallback when IMU sync fails. Enabled by default;
        /// use --no-auto-sync to disable.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        auto_sync: bool,

        /// Frame offset for temporal sync between cameras.
        /// Positive: right video is ahead by N frames.
        /// Negative: left video is ahead by N frames.
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        sync_offset: i64,

        /// Seconds to skip from the start of the video (e.g. camera setup).
        #[arg(long, default_value_t = 0.0)]
        skip_start: f64,

        /// Seconds to skip from the end of the video (e.g. teardown).
        #[arg(long, default_value_t = 0.0)]
        skip_end: f64,

        /// AKAZE detector threshold. Lower = more features.
        /// Default 0.0001 (sensitive). Try 0.001 for fewer but stronger features.
        #[arg(long, default_value_t = 0.0001)]
        akaze_threshold: f64,

        /// Lowe's ratio test threshold. Higher = more matches pass.
        /// Default 0.75. Try 0.6 for stricter matching.
        #[arg(long, default_value_t = 0.75)]
        lowe_ratio: f64,

        /// Detection region x threshold (fraction). Left detects in [x, 1.0],
        /// right in [0.0, 1-x]. Lower = wider detection. Default 0.5.
        #[arg(long, default_value_t = 0.5)]
        detect_x: f64,

        /// Detection region y minimum (fraction). Skip top N% of image.
        /// Default 0.25 (skip top 25% to avoid sky and undistortion edges).
        #[arg(long, default_value_t = 0.25)]
        detect_y_min: f64,

        /// Detection region y maximum (fraction). Skip bottom N% of image.
        /// Default 0.85 (skip bottom 15% to avoid ground and undistortion edges).
        #[arg(long, default_value_t = 0.85)]
        detect_y_max: f64,

        /// Lock cam_d = half_offset (0.5 * (1 - intersect)).
        /// Reduces optimization to 4 parameters.
        #[arg(long, default_value_t = false)]
        lock_cam_d: bool,

        /// Lock z_rx = 0 (z-plane stays static, only translates).
        /// Reduces optimization by 1 parameter.
        #[arg(long, default_value_t = false)]
        lock_z_rx: bool,

        /// Drop the worst N% of points during optimization (0.0-1.0).
        /// E.g. 0.3 = ignore worst 30%. Makes optimizer robust to outliers.
        #[arg(long, default_value_t = 0.3)]
        trim: f64,

        /// Seam proximity weighting sigma. Lower = focus more on seam center.
        /// Default 0.08. Try 0.12 for wider weighting.
        #[arg(long, default_value_t = 0.08)]
        seam_sigma: f64,

        /// Directory to write debug data (keypoints, matches as JSON + images).
        #[arg(long)]
        debug_dir: Option<String>,

        /// Output calibration JSON file path.
        #[arg(short, long, default_value = "match.json")]
        output: String,
    },

    /// Display information about the GPU and system capabilities.
    Info,

    /// Query a connected GoPro camera via USB or WiFi.
    #[cfg(feature = "gopro")]
    Gopro {
        /// GoPro serial number suffix (last 3 digits) for USB connection.
        /// Omit to use WiFi AP mode (10.5.5.9).
        #[arg(long)]
        serial: Option<String>,

        /// Connect via a custom URL instead of the default USB/WiFi address.
        #[arg(long)]
        url: Option<String>,

        /// Start recording on the camera.
        #[arg(long)]
        start: bool,

        /// Stop recording on the camera.
        #[arg(long)]
        stop: bool,

        /// Apply the sports stereo preset (disables HyperSmooth + horizon leveling).
        #[arg(long)]
        sports_preset: bool,
    },
}

/// Parse and validate blend width to [0.0, 1.0].
fn parse_blend(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{v} is not in 0.0..=1.0"))
    }
}

/// Parse a `WIDTHxHEIGHT` string (e.g. `1280x720`, `854x480`) into
/// `(u32, u32)`. Used by `--replay-scale`. Validates YUV420P
/// alignment: width divisible by 4, height even.
fn parse_wxh(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got {s:?}"))?;
    let w: u32 = w.parse().map_err(|e| format!("invalid width {w:?}: {e}"))?;
    let h: u32 = h
        .parse()
        .map_err(|e| format!("invalid height {h:?}: {e}"))?;
    if w == 0 || h == 0 {
        return Err(format!("dimensions must be > 0, got {w}x{h}"));
    }
    if !w.is_multiple_of(4) {
        return Err(format!(
            "width must be divisible by 4 (pack shader packs 4 pixels per u32 write), got {w}"
        ));
    }
    if !h.is_multiple_of(2) {
        return Err(format!(
            "height must be even (YUV420P chroma subsampling), got {h}"
        ));
    }
    Ok((w, h))
}

fn main() -> anyhow::Result<()> {
    // When profiling, tracing-subscriber is owned by the chrome layer
    // (one global subscriber per process). Otherwise, install our fmt
    // subscriber + log bridge for normal structured output.
    #[cfg(feature = "profiling")]
    let _profiling_guard = init_profiling();
    #[cfg(not(feature = "profiling"))]
    init_tracing();
    install_panic_hook();

    // Set up Ctrl-C handler so stitch can finalize the output file
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = interrupted.clone();
        ctrlc::set_handler(move || {
            if interrupted.load(Ordering::Relaxed) {
                // Second Ctrl-C — force exit
                std::process::exit(1);
            }
            interrupted.store(true, Ordering::Relaxed);
            eprintln!("\nInterrupted — finishing output file...");
        })
        .expect("set Ctrl-C handler");
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Stitch {
            left,
            right,
            calibration,
            output,
            width,
            height,
            duration,
            max_frames,
            encoder,
            codec,
            quality,
            blend,
            sync_offset,
            model,
            detection_interval,
            lead_time,
            tracking,
            crf,
            preset,
            container,
            replay,
            replay_scale,
        } => stitch::run_stitch(
            stitch::StitchArgs {
                left: &left,
                right: &right,
                calibration: &calibration,
                output: &output,
                width,
                height,
                blend,
                duration,
                max_frames,
                encoder_name: encoder,
                codec: &codec,
                quality: &quality,
                sync_offset,
                model_path: model.as_deref(),
                detection_interval,
                lead_time,
                tracking_mode: &tracking,
                crf,
                preset,
                container: container.as_deref(),
                replay_path: replay.as_deref(),
                replay_scale,
            },
            &interrupted,
        ),

        Commands::Preview {
            left,
            right,
            calibration,
            width,
            height,
            sync_offset,
            blend,
            rig_tilt,
        } => {
            const MAX_DIM: u32 = 8192;
            anyhow::ensure!(
                width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
                "Preview dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
                width,
                height,
            );
            preview::run_preview(
                &preview::PreviewConfig {
                    left_path: &left,
                    right_path: &right,
                    calibration_path: &calibration,
                    width,
                    height,
                    sync_offset,
                    blend_width: blend,
                    rig_tilt_degrees: rig_tilt,
                },
                &interrupted,
            )
        }

        #[cfg(feature = "gstreamer")]
        Commands::Camera {
            left_device,
            right_device,
            calibration,
            output,
            capture_width,
            capture_height,
            capture_fps,
            width,
            height,
            encoder,
            codec,
            quality,
            blend,
            max_frames,
            duration,
            model,
            detection_interval,
            crf,
            preset,
            container,
            tracking,
            unconstrained,
            replay,
            replay_scale,
        } => {
            use reco_io::gstreamer::camera::CameraConfig;

            const MAX_DIM: u32 = 8192;
            anyhow::ensure!(
                width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
                "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
                width,
                height,
            );

            log::info!("Camera capture: {left_device} + {right_device} -> {output}");

            let cam_config = CameraConfig {
                width: capture_width,
                height: capture_height,
                fps: capture_fps,
                left_device,
                right_device,
            };

            camera::run_camera(
                camera::CameraRunConfig {
                    cam_config,
                    calibration: &calibration,
                    output: &output,
                    width,
                    height,
                    blend,
                    encoder_name: encoder,
                    codec: &codec,
                    quality: &quality,
                    duration,
                    max_frames,
                    capture_fps,
                    model_path: model.as_deref(),
                    detection_interval,
                    crf,
                    preset,
                    container: container.as_deref(),
                    tracking: &tracking,
                    unconstrained,
                    replay_path: replay.as_deref(),
                    replay_scale,
                },
                &interrupted,
            )
        }

        #[cfg(feature = "libcamera")]
        Commands::Libcamera {
            left_camera,
            right_camera,
            calibration,
            output,
            capture_width,
            capture_height,
            capture_fps,
            width,
            height,
            encoder,
            codec,
            quality,
            blend,
            max_frames,
            duration,
            model,
            detection_interval,
            crf,
            preset,
        } => {
            use reco_io::libcamera::LibcameraConfig;

            const MAX_DIM: u32 = 8192;
            anyhow::ensure!(
                width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
                "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
                width,
                height,
            );

            log::info!("libcamera capture: cam{left_camera} + cam{right_camera} -> {output}");

            let cam_config = LibcameraConfig {
                width: capture_width,
                height: capture_height,
                fps: capture_fps,
                left_camera,
                right_camera,
            };

            libcamera_cmd::run_libcamera(
                libcamera_cmd::LibcameraRunConfig {
                    cam_config,
                    calibration: &calibration,
                    output: &output,
                    width,
                    height,
                    blend,
                    encoder_name: encoder,
                    codec: &codec,
                    quality: &quality,
                    duration,
                    max_frames,
                    capture_fps,
                    model_path: model.as_deref(),
                    detection_interval,
                    crf,
                    preset,
                },
                &interrupted,
            )
        }

        Commands::Calibrate {
            left,
            right,
            left_profile,
            right_profile,
            frames,
            no_auto_imu,
            auto_sync,
            sync_offset,
            skip_start,
            skip_end,
            akaze_threshold,
            lowe_ratio,
            detect_x,
            detect_y_min,
            detect_y_max,
            lock_cam_d,
            lock_z_rx,
            trim,
            seam_sigma,
            debug_dir,
            output,
        } => calibrate::run_calibrate(
            &left,
            &right,
            left_profile.as_deref(),
            right_profile.as_deref(),
            frames,
            no_auto_imu,
            auto_sync,
            sync_offset,
            skip_start,
            skip_end,
            akaze_threshold,
            lowe_ratio,
            detect_x,
            detect_y_min,
            detect_y_max,
            lock_cam_d,
            lock_z_rx,
            trim,
            seam_sigma,
            debug_dir.as_deref(),
            &output,
        ),

        Commands::Info => {
            let gpu = reco_core::gpu::GpuContext::new_blocking()?;
            println!("GPU: {}", gpu.gpu_name());
            println!("Backend: {}", gpu.backend_name());
            println!("Driver: {}", gpu.driver_info());

            println!("\nH.264 encoders:");
            let encoders = reco_io::ffmpeg::encoder::available_h264_encoders();
            if encoders.is_empty() {
                println!("  (none found)");
            } else {
                for enc in &encoders {
                    let tag = if enc.is_hardware { "HW" } else { "SW" };
                    println!("  {} [{}] — {}", enc.name, tag, enc.description);
                }
            }
            Ok(())
        }
        #[cfg(feature = "gopro")]
        Commands::Gopro {
            serial,
            url,
            start,
            stop,
            sports_preset,
        } => {
            use reco_control::gopro::GoProCamera;

            let cam = if let Some(url) = url {
                GoProCamera::connect_url(&url)?
            } else if let Some(serial) = serial {
                GoProCamera::connect_usb(&serial)?
            } else {
                println!("Trying WiFi AP mode (10.5.5.9)...");
                GoProCamera::connect_wifi()?
            };

            if let Some(info) = cam.info() {
                println!("Model:    {}", info.model_name.as_deref().unwrap_or("unknown"));
                println!("Firmware: {}", info.firmware_version.as_deref().unwrap_or("unknown"));
                println!("Serial:   {}", info.serial_number.as_deref().unwrap_or("unknown"));
                println!("AP SSID:  {}", info.ap_ssid.as_deref().unwrap_or("unknown"));
            }

            let status = cam.status()?;
            println!("\nStatus:");
            if let Some(pct) = status.battery_percent {
                println!("  Battery:   {}%", pct);
            }
            if let Some(enc) = status.encoding {
                println!("  Recording: {}", if enc { "yes" } else { "no" });
            }
            if let Some(hot) = status.overheating {
                println!("  Overheat:  {}", if hot { "YES" } else { "no" });
            }

            if sports_preset {
                println!("\nApplying sports stereo preset...");
                cam.apply_sports_preset(
                    reco_control::gopro::VideoResolution::Res1080p,
                    reco_control::gopro::Fps::Fps30,
                    reco_control::gopro::VideoLens::Linear,
                )?;
                println!("Done.");
            }

            if start {
                cam.start_recording()?;
                println!("Recording started.");
            }

            if stop {
                cam.stop_recording()?;
                println!("Recording stopped.");
            }

            Ok(())
        }
    }
}
