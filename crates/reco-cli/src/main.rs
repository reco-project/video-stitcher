//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

mod calibrate;
#[cfg(feature = "gstreamer")]
mod camera;
mod helpers;
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
    },

    /// Calibrate two cameras: detect features and compute placement parameters.
    Calibrate {
        /// Path to the left camera video file.
        left: String,

        /// Path to the right camera video file.
        right: String,

        /// Path to the left camera lens profile JSON.
        #[arg(long)]
        left_profile: String,

        /// Path to the right camera lens profile JSON.
        #[arg(long)]
        right_profile: String,

        /// Number of frame pairs to sample from the video.
        #[arg(long, default_value_t = 15)]
        frames: usize,

        /// Number of random-subset optimization iterations.
        #[arg(long, default_value_t = 200)]
        iterations: usize,

        /// Disable the 6th parameter (left plane roll correction).
        #[arg(long, default_value_t = false)]
        no_left_roll: bool,

        /// Frame offset for temporal sync between cameras.
        /// Positive: right video is ahead by N frames.
        /// Negative: left video is ahead by N frames.
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        sync_offset: i64,

        /// Directory to write debug data (keypoints, matches as JSON + images).
        #[arg(long)]
        debug_dir: Option<String>,

        /// Output calibration JSON file path.
        #[arg(short, long, default_value = "match.json")]
        output: String,
    },

    /// Display information about the GPU and system capabilities.
    Info,
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

fn main() -> anyhow::Result<()> {
    // When profiling, tracing-subscriber owns the global logger;
    // otherwise, use env_logger for RUST_LOG filtering.
    #[cfg(feature = "profiling")]
    let _profiling_guard = init_profiling();
    #[cfg(not(feature = "profiling"))]
    env_logger::init();

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
        } => stitch::run_stitch(
            &left,
            &right,
            &calibration,
            &output,
            width,
            height,
            blend,
            duration,
            max_frames,
            encoder,
            &codec,
            &quality,
            sync_offset,
            model.as_deref(),
            detection_interval,
            lead_time,
            &interrupted,
        ),

        Commands::Preview {
            left,
            right,
            calibration,
            width,
            height,
        } => {
            const MAX_DIM: u32 = 8192;
            anyhow::ensure!(
                width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
                "Preview dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
                width,
                height,
            );
            preview::run_preview(&left, &right, &calibration, width, height)
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
                cam_config,
                &calibration,
                &output,
                width,
                height,
                blend,
                encoder,
                &codec,
                &quality,
                duration,
                max_frames,
                capture_fps,
                model.as_deref(),
                detection_interval,
                &interrupted,
            )
        }

        Commands::Calibrate {
            left,
            right,
            left_profile,
            right_profile,
            frames,
            iterations,
            no_left_roll,
            sync_offset,
            debug_dir,
            output,
        } => calibrate::run_calibrate(
            &left,
            &right,
            &left_profile,
            &right_profile,
            frames,
            iterations,
            no_left_roll,
            sync_offset,
            debug_dir.as_deref(),
            &output,
        ),

        Commands::Info => {
            let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
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
    }
}
