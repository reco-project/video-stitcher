//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

use clap::{Parser, Subcommand};
use std::path::Path;
use std::time::Instant;

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
    },

    /// Display information about the GPU and system capabilities.
    Info,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Stitch {
            left,
            right,
            calibration,
            output,
            width,
            height,
        } => {
            log::info!("Stitching: {left} + {right} → {output}");

            let json = std::fs::read_to_string(&calibration)?;
            let cal: reco_core::calibration::MatchCalibration = serde_json::from_str(&json)?;

            let viewport = reco_core::viewport::ViewportConfig {
                width,
                height,
                ..Default::default()
            };

            let pipeline =
                pollster::block_on(reco_core::pipeline::StitchPipeline::new(cal, viewport))?;

            println!(
                "Pipeline ready: GPU = {}, output = {width}x{height}",
                pipeline.gpu.adapter_info.name
            );

            // Open video decoders
            let mut left_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&left))?;
            let mut right_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&right))?;

            log::info!(
                "Left video: {}x{} @ {:.1} fps",
                left_dec.width(),
                left_dec.height(),
                left_dec.fps()
            );
            log::info!(
                "Right video: {}x{} @ {:.1} fps",
                right_dec.width(),
                right_dec.height(),
                right_dec.fps()
            );

            // Create encoder using the left video's frame rate
            let fps = left_dec.frame_rate();
            let mut encoder =
                reco_ffmpeg::encoder::VideoEncoder::new(Path::new(&output), width, height, fps)?;

            let start = Instant::now();
            let mut frame_count: u64 = 0;

            // Static camera: yaw=0, pitch=0 (no pan/tilt)
            let yaw = 0.0_f32;
            let pitch = 0.0_f32;

            loop {
                let left_frame = left_dec.next_frame()?;
                let right_frame = right_dec.next_frame()?;

                let (left_frame, right_frame) = match (left_frame, right_frame) {
                    (Some(l), Some(r)) => (l, r),
                    _ => break, // Either stream ended
                };

                let stitched =
                    pipeline.process_frame(&left_frame.data, &right_frame.data, yaw, pitch)?;

                encoder.write_frame(&stitched)?;
                frame_count += 1;

                if frame_count.is_multiple_of(30) {
                    let elapsed = start.elapsed().as_secs_f64();
                    let fps_actual = frame_count as f64 / elapsed;
                    print!("\rProcessed {frame_count} frames ({fps_actual:.1} fps)");
                }
            }

            encoder.finish()?;

            let elapsed = start.elapsed().as_secs_f64();
            let fps_actual = frame_count as f64 / elapsed;
            println!(
                "\nDone: {frame_count} frames in {elapsed:.1}s ({fps_actual:.1} fps) → {output}"
            );

            Ok(())
        }

        Commands::Info => {
            let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
            println!("GPU: {}", gpu.adapter_info.name);
            println!("Backend: {:?}", gpu.adapter_info.backend);
            println!("Driver: {}", gpu.adapter_info.driver);
            Ok(())
        }
    }
}
