//! `reco-highlights` - stitch a video and emit a JSON highlight reel alongside it.
//!
//! Runs the full stitch pipeline (so you also get the encoded panorama as an
//! artifact), and writes a sidecar JSON file describing moments of sustained
//! ball activity that a downstream editor can slice out.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::Parser;

use reco_autocam::{AutocamConfig, TrackingMode};
use reco_highlights::{HighlightConfig, HighlightDetector};
use reco_io::StitchJob;

/// Generate an auto-highlight JSON reel from a stereo sports recording.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Left camera video file.
    left: PathBuf,
    /// Right camera video file.
    right: PathBuf,
    /// Calibration JSON file (produced by `reco calibrate`).
    #[arg(short, long)]
    calibration: PathBuf,
    /// Output stitched video file (always produced - required by StitchJob).
    #[arg(short = 'o', long)]
    output_video: PathBuf,
    /// Output JSON reel path.
    #[arg(short = 'r', long, default_value = "highlights.json")]
    reel: PathBuf,
    /// YOLO model for ball detection (.onnx).
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Confidence threshold for a detection to count toward a highlight.
    #[arg(long, default_value_t = 0.45)]
    min_confidence: f32,
    /// Minimum highlight window length, in seconds.
    #[arg(long, default_value_t = 2.5)]
    min_duration_s: f64,
    /// Maximum gap between active frames that is still bridged, in seconds.
    #[arg(long, default_value_t = 1.0)]
    max_gap_s: f64,
    /// Pre-roll padding in seconds.
    #[arg(long, default_value_t = 1.5)]
    pre_roll_s: f64,
    /// Post-roll padding in seconds.
    #[arg(long, default_value_t = 2.0)]
    post_roll_s: f64,
    /// Cap on frames to process (debugging).
    #[arg(long)]
    max_frames: Option<u64>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let config = HighlightConfig {
        target_class_id: 0,
        min_confidence: args.min_confidence,
        min_duration_ms: args.min_duration_s * 1000.0,
        max_gap_ms: args.max_gap_s * 1000.0,
        pre_roll_ms: args.pre_roll_s * 1000.0,
        post_roll_ms: args.post_roll_s * 1000.0,
    };

    // Shared state between the detection callback and the outer scope.
    // The callback is 'static + FnMut + Send, so we need Arc<Mutex<_>>.
    let detector = Arc::new(Mutex::new(HighlightDetector::new(config)));

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = interrupted.clone();
        ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))
            .context("install ctrl-c handler")?;
    }

    let autocam = AutocamConfig::new(args.model).with_tracking_mode(TrackingMode::Ball);
    let det_handle = detector.clone();

    let mut job = StitchJob::new(
        &args.left,
        &args.right,
        &args.calibration,
        &args.output_video,
    )
    .on_session(move |session, _source| {
        // Attach autocam so the YOLO detector actually runs.
        if let Err(e) = reco_autocam::setup_autocam_from_config(session, &autocam) {
            log::error!("failed to set up autocam: {e}");
            return;
        }
        // Tap the detection stream into our highlight aggregator.
        let sink = det_handle.clone();
        session.set_detection_callback(Box::new(move |dets, idx, ts| {
            if let Ok(mut hl) = sink.lock() {
                hl.push(dets, idx, ts);
            }
        }));
    });

    if let Some(max) = args.max_frames {
        job = job.max_frames(max);
    }

    let result = job.run(&interrupted).context("stitch job failed")?;

    let reel = Arc::try_unwrap(detector)
        .map_err(|_| anyhow::anyhow!("detector still borrowed after run"))?
        .into_inner()
        .map_err(|e| anyhow::anyhow!("detector mutex poisoned: {e}"))?
        .finish();

    std::fs::write(&args.reel, serde_json::to_string_pretty(&reel)?)
        .with_context(|| format!("write reel to {}", args.reel.display()))?;

    log::info!(
        "stitched {} frames in {:.1}s on {} ({}) -> {}",
        result.frames_processed,
        result.elapsed.as_secs_f64(),
        result.gpu_name,
        result.encoder_name,
        args.output_video.display(),
    );
    log::info!(
        "wrote {} highlight window(s) to {}",
        reel.windows.len(),
        args.reel.display(),
    );

    Ok(())
}
