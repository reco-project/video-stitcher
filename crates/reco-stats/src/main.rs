//! `reco-stats` - dump detections from a stitch run to a CSV file.
//!
//! Runs the standard stitch + encode pipeline (because `StitchJob` has no
//! detect-only mode) and writes one CSV row per detection to a sidecar file.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;

use reco_autocam::{AutocamConfig, TrackingMode};
use reco_io::StitchJob;
use reco_stats::CsvDetectionSink;

/// Export Reco detections to CSV.
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
    /// Output CSV path.
    #[arg(short = 's', long, default_value = "stats.csv")]
    stats: PathBuf,
    /// YOLO model for detection.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Tracking mode: ball or field.
    #[arg(long, default_value = "ball")]
    tracking: String,
    /// Debug frame cap.
    #[arg(long)]
    max_frames: Option<u64>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let csv_file = std::fs::File::create(&args.stats)
        .with_context(|| format!("create {}", args.stats.display()))?;
    let writer = std::io::BufWriter::new(csv_file);
    let sink = Arc::new(Mutex::new(CsvDetectionSink::new(writer)?));

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = interrupted.clone();
        ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))
            .context("install ctrl-c handler")?;
    }

    let tracking_mode = match args.tracking.to_ascii_lowercase().as_str() {
        "ball" => TrackingMode::Ball,
        "field" => TrackingMode::Field,
        other => anyhow::bail!("unknown --tracking value: {other} (expected ball|field)"),
    };
    let autocam = AutocamConfig::new(args.model).with_tracking_mode(tracking_mode);
    let sink_for_session = sink.clone();

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
        let inner = sink_for_session.clone();
        session.set_detection_callback(Box::new(move |dets, idx, ts| {
            if let Ok(mut s) = inner.lock() {
                if let Err(e) = s.push(dets, idx, ts) {
                    // Can't return errors out of the callback — log and hope.
                    log::error!("csv write failed: {e}");
                }
            }
        }));
    });

    if let Some(max) = args.max_frames {
        job = job.max_frames(max);
    }

    let result = job.run(&interrupted).context("stitch job failed")?;

    // The session (and its clone of our Arc) was dropped inside job.run().
    // try_unwrap should succeed because we are the only remaining holder.
    let sink = Arc::try_unwrap(sink)
        .map_err(|_| anyhow::anyhow!("csv sink still borrowed after run"))?
        .into_inner()
        .map_err(|e| anyhow::anyhow!("sink mutex poisoned: {e}"))?;
    let rows = sink.rows_written();
    // Drop the sink to flush the BufWriter.
    drop(sink);

    log::info!(
        "stitched {} frames in {:.1}s on {} ({}) -> {}",
        result.frames_processed,
        result.elapsed.as_secs_f64(),
        result.gpu_name,
        result.encoder_name,
        args.output_video.display(),
    );
    log::info!("wrote {} rows to {}", rows, args.stats.display());

    Ok(())
}
