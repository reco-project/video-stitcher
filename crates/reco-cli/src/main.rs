//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

use clap::{Parser, Subcommand};

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

            // TODO: decode frames, run pipeline, encode output
            println!("Frame processing not yet implemented.");
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
