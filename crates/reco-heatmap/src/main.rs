//! `reco-heatmap` - write a PNG heatmap of ball positions for a recording.
//!
//! Runs a full stitch + encode pass (because `StitchJob` does not have a
//! "detect only" mode) and writes a sidecar PNG heatmap.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;

use reco_autocam::{AutocamConfig, TrackingMode};
use reco_heatmap::{HeatmapAccumulator, HeatmapConfig};
use reco_io::StitchJob;

/// Build a ball-position heatmap from a stereo sports recording.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Left camera video file.
    left: PathBuf,
    /// Right camera video file.
    right: PathBuf,
    /// Calibration JSON file.
    #[arg(short, long)]
    calibration: PathBuf,
    /// Output stitched video file (required by StitchJob).
    #[arg(short = 'o', long)]
    output_video: PathBuf,
    /// Output PNG heatmap path.
    #[arg(short = 'p', long, default_value = "heatmap.png")]
    heatmap: PathBuf,
    /// YOLO model for ball detection.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Heatmap width in cells.
    #[arg(long, default_value_t = 640)]
    grid_width: u32,
    /// Heatmap height in cells.
    #[arg(long, default_value_t = 180)]
    grid_height: u32,
    /// Horizontal half-range in degrees (panorama yaw).
    #[arg(long, default_value_t = 45.0)]
    yaw_degrees: f32,
    /// Vertical half-range in degrees (panorama pitch).
    #[arg(long, default_value_t = 20.0)]
    pitch_degrees: f32,
    /// Minimum confidence.
    #[arg(long, default_value_t = 0.30)]
    min_confidence: f32,
    /// Debug frame cap.
    #[arg(long)]
    max_frames: Option<u64>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let cfg = HeatmapConfig {
        target_class_id: 0,
        min_confidence: args.min_confidence,
        width: args.grid_width,
        height: args.grid_height,
        yaw_min: -args.yaw_degrees.to_radians(),
        yaw_max: args.yaw_degrees.to_radians(),
        pitch_min: -args.pitch_degrees.to_radians(),
        pitch_max: args.pitch_degrees.to_radians(),
    };

    let accumulator = Arc::new(Mutex::new(HeatmapAccumulator::new(cfg)));

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = interrupted.clone();
        ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))
            .context("install ctrl-c handler")?;
    }

    let autocam = AutocamConfig::new(args.model).with_tracking_mode(TrackingMode::Ball);
    let accum_for_session = accumulator.clone();

    let mut job = StitchJob::new(
        &args.left,
        &args.right,
        &args.calibration,
        &args.output_video,
    )
    .on_session(move |session, _source| {
        if let Err(e) = reco_autocam::setup_autocam_from_config(session, &autocam) {
            log::error!("failed to set up autocam: {e}");
            return;
        }
        let sink = accum_for_session.clone();
        session.set_detection_callback(Box::new(move |dets, idx, ts| {
            if let Ok(mut h) = sink.lock() {
                h.push(dets, idx, ts);
            }
        }));
    });

    if let Some(max) = args.max_frames {
        job = job.max_frames(max);
    }

    let result = job.run(&interrupted).context("stitch job failed")?;

    let heatmap = Arc::try_unwrap(accumulator)
        .map_err(|_| anyhow::anyhow!("heatmap still borrowed"))?
        .into_inner()
        .map_err(|e| anyhow::anyhow!("heatmap mutex poisoned: {e}"))?;

    let rgba = heatmap.render();
    let img = image::RgbaImage::from_raw(heatmap.width(), heatmap.height(), rgba)
        .context("heatmap buffer size mismatch")?;
    img.save(&args.heatmap)
        .with_context(|| format!("write heatmap png to {}", args.heatmap.display()))?;

    log::info!(
        "stitched {} frames in {:.1}s on {} ({}) -> {}",
        result.frames_processed,
        result.elapsed.as_secs_f64(),
        result.gpu_name,
        result.encoder_name,
        args.output_video.display(),
    );
    log::info!(
        "wrote heatmap ({} samples) to {}",
        heatmap.samples(),
        args.heatmap.display(),
    );

    Ok(())
}
