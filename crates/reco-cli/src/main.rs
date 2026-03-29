//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

mod helpers;
mod preview;
mod stitch;

use clap::{Parser, Subcommand};
#[cfg(feature = "gstreamer")]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "gstreamer")]
use std::time::Instant;

/// Detect if we're running on NVIDIA Tegra (Jetson platform).
///
/// Checks for L4T (Linux for Tegra) release file or Tegra device-tree entry.
/// Used to enable platform-specific optimizations:
/// - NV12 native capture (NVIDIA ISP outputs NV12 directly)
/// - Thread-count tuning for the lower core count
#[cfg(feature = "gstreamer")]
fn is_tegra() -> bool {
    Path::new("/etc/nv_tegra_release").exists()
        || std::fs::read_to_string("/proc/device-tree/compatible")
            .unwrap_or_default()
            .contains("nvidia,tegra")
}

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

            {
                const MAX_CAL_BYTES: u64 = 1024 * 1024; // 1 MiB
                let size = std::fs::metadata(&calibration)
                    .map_err(|e| anyhow::anyhow!("cannot stat calibration file: {e}"))?
                    .len();
                anyhow::ensure!(
                    size <= MAX_CAL_BYTES,
                    "calibration file is too large ({size} bytes, max {MAX_CAL_BYTES})"
                );
            }
            let json = std::fs::read_to_string(&calibration)
                .map_err(|e| anyhow::anyhow!("cannot read calibration: {e}"))?;
            let cal: reco_core::calibration::MatchCalibration = serde_json::from_str(&json)
                .map_err(|e| anyhow::anyhow!("invalid calibration JSON: {e}"))?;
            cal.validate()
                .map_err(|e| anyhow::anyhow!("calibration validation failed: {e}"))?;

            let viewport = reco_core::viewport::ViewportConfig {
                width,
                height,
                blend_width: blend,
                ..Default::default()
            };

            let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;

            // Use NV12 capture on Jetson to skip the NV12->I420 conversion
            // in nvvidconv. The NVIDIA ISP natively outputs NV12.
            let use_nv12_capture = is_tegra();
            let input_format = if use_nv12_capture {
                reco_core::renderer::InputFormat::Nv12
            } else {
                reco_core::renderer::InputFormat::Yuv420p
            };

            let pipeline = reco_core::pipeline::StitchPipeline::with_gpu(
                gpu,
                cal,
                viewport,
                capture_width,
                capture_height,
                wgpu::TextureFormat::Rgba8Unorm,
                input_format,
            )?;

            let mut nv12_converter =
                reco_core::nv12_converter::Nv12Converter::new(pipeline.gpu(), width, height)?;

            let mode_str = if use_nv12_capture { "NV12" } else { "I420" };
            println!(
                "Pipeline ready: GPU = {}, capture = {}x{}@{}fps ({}), output = {}x{}",
                pipeline.gpu_name(),
                capture_width,
                capture_height,
                capture_fps,
                mode_str,
                width,
                height
            );

            reco_io::init();
            let quality = match quality.as_str() {
                "fast" => reco_io::ffmpeg::encoder::Quality::Fast,
                "high" => reco_io::ffmpeg::encoder::Quality::High,
                _ => reco_io::ffmpeg::encoder::Quality::Balanced,
            };
            let video_codec = reco_io::ffmpeg::encoder::VideoCodec::from_str_loose(&codec)
                .unwrap_or_else(|| {
                    eprintln!("Unknown codec '{codec}', defaulting to H.264");
                    reco_io::ffmpeg::encoder::VideoCodec::H264
                });
            let enc_config = reco_io::ffmpeg::encoder::EncoderConfig {
                encoder_name: encoder,
                codec: video_codec,
                quality,
            };

            let fps_rational = reco_io::ffmpeg::Rational::new(capture_fps as i32, 1);
            let mut enc = reco_io::ffmpeg::encoder::VideoEncoder::new(
                Path::new(&output),
                width,
                height,
                fps_rational,
                &enc_config,
            )?;
            println!("Encoder: {}", enc.encoder_name());

            let capture_fps_f64 = capture_fps as f64;
            let frame_limit: u64 = match (duration, max_frames) {
                (Some(dur), Some(mf)) => ((dur * capture_fps_f64) as u64).min(mf),
                (Some(dur), None) => (dur * capture_fps_f64) as u64,
                (None, Some(mf)) => mf,
                (None, None) => u64::MAX,
            };

            if frame_limit < u64::MAX {
                println!("Capturing up to {frame_limit} frames");
            }

            // Async encode: send NV12 data to a background thread so
            // encoding overlaps with the next frame's capture + render.
            let (encode_tx, encode_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(2);
            let encode_thread = std::thread::Builder::new()
                .name("encode".into())
                .spawn(move || -> Result<(), anyhow::Error> {
                    while let Ok(nv12_data) = encode_rx.recv() {
                        enc.write_nv12_frame(&nv12_data)?;
                    }
                    enc.finish()?;
                    Ok(())
                })
                .expect("spawn encode thread");

            let mut frame_count: u64 = 0;
            let yaw = 0.0_f32;
            let pitch = 0.0_f32;

            if use_nv12_capture {
                // NV12 path: skip nvvidconv format conversion, upload 2 planes
                let mut source =
                    reco_io::gstreamer::camera::GstreamerNv12CameraSource::open(&cam_config)?;

                // Warm up: discard first frame (camera ISP + pipeline init)
                if let Some(pair) = source.next_pair()? {
                    let left_planes = reco_core::pipeline::Nv12Planes {
                        y: &pair.left.y,
                        uv: &pair.left.uv,
                    };
                    let right_planes = reco_core::pipeline::Nv12Planes {
                        y: &pair.right.y,
                        uv: &pair.right.uv,
                    };
                    let render_buf =
                        pipeline.render_to_target_nv12(&left_planes, &right_planes, yaw, pitch);
                    let nv12_data = nv12_converter.convert_and_readback(
                        pipeline.gpu(),
                        pipeline.render_target(),
                        render_buf,
                    )?;
                    if encode_tx.send(nv12_data).is_err() {
                        anyhow::bail!("encoder thread died during warmup");
                    }
                    println!("Warmup complete, starting capture...");
                }

                let start = Instant::now();

                while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
                    let pair = {
                        reco_core::profile_scope!("wait_capture");
                        match source.next_pair()? {
                            Some(p) => p,
                            None => break,
                        }
                    };

                    let left_planes = reco_core::pipeline::Nv12Planes {
                        y: &pair.left.y,
                        uv: &pair.left.uv,
                    };
                    let right_planes = reco_core::pipeline::Nv12Planes {
                        y: &pair.right.y,
                        uv: &pair.right.uv,
                    };

                    let render_buf =
                        pipeline.render_to_target_nv12(&left_planes, &right_planes, yaw, pitch);
                    let nv12_data = nv12_converter.convert_and_readback(
                        pipeline.gpu(),
                        pipeline.render_target(),
                        render_buf,
                    )?;
                    if encode_tx.send(nv12_data).is_err() {
                        break;
                    }
                    frame_count += 1;

                    if frame_count.is_multiple_of(30) {
                        let elapsed = start.elapsed().as_secs_f64();
                        let fps_actual = frame_count as f64 / elapsed;
                        print!("\rProcessed {frame_count} frames ({fps_actual:.1} fps)");
                        let _ = std::io::stdout().flush();
                    }
                }

                // Stop cameras gracefully before finishing encoder
                source.stop();
                drop(encode_tx);
                encode_thread.join().expect("encode thread panicked")?;

                let elapsed = start.elapsed().as_secs_f64();
                let fps_actual = frame_count as f64 / elapsed;
                println!(
                    "\nDone: {frame_count} frames in {elapsed:.1}s ({fps_actual:.1} fps) -> {output}"
                );

                // Drop source explicitly to allow graceful GStreamer/Argus teardown
                drop(source);
            } else {
                // I420 path: standard YUV420P upload with 3 planes
                use reco_core::source::FrameSource;
                let mut source =
                    reco_io::gstreamer::camera::GstreamerCameraSource::open(&cam_config)?;

                let start = Instant::now();

                while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
                    let frame = {
                        reco_core::profile_scope!("wait_capture");
                        match source.next_frame()? {
                            Some(f) => f,
                            None => break,
                        }
                    };
                    let pair = match frame {
                        reco_core::source::StereoFrame::Yuv420p(p) => p,
                        _ => anyhow::bail!("expected Yuv420p frame from I420 camera source"),
                    };

                    let left_planes = reco_core::pipeline::YuvPlanes {
                        y: &pair.left.y,
                        u: &pair.left.u,
                        v: &pair.left.v,
                    };
                    let right_planes = reco_core::pipeline::YuvPlanes {
                        y: &pair.right.y,
                        u: &pair.right.u,
                        v: &pair.right.v,
                    };

                    let render_buf =
                        pipeline.render_to_target(&left_planes, &right_planes, yaw, pitch);
                    let nv12_data = nv12_converter.convert_and_readback(
                        pipeline.gpu(),
                        pipeline.render_target(),
                        render_buf,
                    )?;
                    if encode_tx.send(nv12_data).is_err() {
                        break;
                    }
                    frame_count += 1;

                    if frame_count.is_multiple_of(30) {
                        let elapsed = start.elapsed().as_secs_f64();
                        let fps_actual = frame_count as f64 / elapsed;
                        print!("\rProcessed {frame_count} frames ({fps_actual:.1} fps)");
                        let _ = std::io::stdout().flush();
                    }
                }

                drop(encode_tx);
                encode_thread.join().expect("encode thread panicked")?;

                let elapsed = start.elapsed().as_secs_f64();
                let fps_actual = frame_count as f64 / elapsed;
                println!(
                    "\nDone: {frame_count} frames in {elapsed:.1}s ({fps_actual:.1} fps) -> {output}"
                );
            }

            Ok(())
        }

        Commands::Info => {
            let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
            println!("GPU: {}", gpu.adapter_info.name);
            println!("Backend: {:?}", gpu.adapter_info.backend);
            println!("Driver: {}", gpu.adapter_info.driver);

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
