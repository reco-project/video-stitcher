//! Stitch subcommand: encode two video files into a panoramic output.
//!
//! Uses `StitchJob` (Layer 3 API) for all cases, including autocam.
//! The `on_session` callback wires up detection and direction when a
//! YOLO model is provided.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Arguments for the stitch subcommand, collected from CLI parsing.
pub struct StitchArgs<'a> {
    pub left: &'a str,
    pub right: &'a str,
    pub calibration: &'a str,
    pub output: &'a str,
    pub width: u32,
    pub height: u32,
    pub blend: f32,
    pub duration: Option<f64>,
    pub max_frames: Option<u64>,
    pub encoder_name: Option<String>,
    pub codec: &'a str,
    pub quality: &'a str,
    pub sync_offset: i64,
    pub model_path: Option<&'a str>,
    pub detection_interval: u64,
    pub lead_time: f64,
    pub tracking_mode: reco_autocam::TrackingMode,
    pub crf: Option<u8>,
    pub preset: Option<String>,
}

/// Run the stitch subcommand.
pub fn run_stitch(args: StitchArgs<'_>, interrupted: &Arc<AtomicBool>) -> anyhow::Result<()> {
    const MAX_DIM: u32 = 8192;
    anyhow::ensure!(
        args.width > 0 && args.width <= MAX_DIM && args.height > 0 && args.height <= MAX_DIM,
        "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
        args.width,
        args.height,
    );

    let progress = crate::helpers::ProgressReporter::new(30);

    // Load calibration up front so we can extract field_roi for autocam
    // and pass the pre-loaded calibration to StitchJob.
    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(args.calibration))?;
    let field_roi = cal.field_roi.clone();

    let mut job = reco_io::StitchJob::with_calibration(args.left, args.right, cal, args.output)
        .codec(parse_codec(args.codec))
        .quality(parse_quality(args.quality))
        .resolution(args.width, args.height)
        .blend_width(args.blend)
        .on_progress(move |p: &reco_core::session::FrameProgress| {
            progress.report(p.frames_completed);
        });

    if let Some(d) = args.duration {
        job = job.duration(d);
    }
    if let Some(n) = args.max_frames {
        job = job.max_frames(n);
    }
    if args.sync_offset != 0 {
        job = job.sync_offset(args.sync_offset);
    }
    if let Some(ref enc) = args.encoder_name {
        job = job.encoder_name(enc);
    }
    if let Some(crf) = args.crf {
        job = job.crf(crf);
    }
    if let Some(ref preset) = args.preset {
        job = job.preset(preset);
    }

    // Wire up autocam via the on_session callback if a model is provided.
    if let Some(model_path) = args.model_path {
        let model_path = model_path.to_owned();
        let interval = args.detection_interval;
        let lead = args.lead_time;
        let mode = args.tracking_mode;
        job = job.on_session(move |session, source| {
            let info = source.info();
            match reco_autocam::setup_autocam(
                session,
                &model_path,
                info.width,
                info.height,
                info.fps as f32,
                source.is_gpu_resident(),
                interval,
                lead,
                mode,
                field_roi.as_ref(),
            ) {
                Ok(true) => println!("Autocam: tracking enabled (model: {model_path})"),
                Ok(false) => {
                    eprintln!(
                        "Warning: ball tracking unavailable (build with --features tensorrt \
                         for GPU detection, or use CPU decode)"
                    );
                }
                Err(e) => {
                    eprintln!("Warning: autocam setup failed ({e}), continuing without tracking")
                }
            }
        });
    }

    let result = job.run(interrupted)?;
    println!(
        "\nDone: {} frames in {:.1}s ({:.1} fps) -> {}",
        result.frames_processed,
        result.elapsed.as_secs_f64(),
        result.fps(),
        args.output
    );
    Ok(())
}

fn parse_codec(s: &str) -> reco_io::output::Codec {
    match s {
        "hevc" | "h265" => reco_io::output::Codec::HEVC,
        "av1" => reco_io::output::Codec::AV1,
        _ => reco_io::output::Codec::H264,
    }
}

fn parse_quality(s: &str) -> reco_io::output::Quality {
    match s {
        "fast" => reco_io::output::Quality::Fast,
        "high" => reco_io::output::Quality::High,
        _ => reco_io::output::Quality::Balanced,
    }
}
