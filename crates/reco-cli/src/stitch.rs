//! Stitch subcommand: encode two video files into a panoramic output.
//!
//! Uses `StitchJob` (Layer 3 API) for simple cases, or falls back to
//! Layer 2 (`SmartFileSource` + `session.run()`) when autocam is needed.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use reco_core::source::FrameSource;

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

    // If no autocam, use StitchJob (Layer 3) for maximum simplicity.
    if args.model_path.is_none() {
        return run_with_stitch_job(&args, interrupted);
    }

    // With autocam, use Layer 2 (SmartFileSource + session.run()).
    run_with_autocam(&args, interrupted)
}

/// Layer 3 path: use StitchJob for simple stitching without autocam.
fn run_with_stitch_job(args: &StitchArgs<'_>, interrupted: &Arc<AtomicBool>) -> anyhow::Result<()> {
    let progress = crate::helpers::ProgressReporter::new(30);

    let mut job = reco_io::StitchJob::new(args.left, args.right, args.calibration, args.output)
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

/// Layer 2 path: SmartFileSource + session.run() with autocam support.
fn run_with_autocam(args: &StitchArgs<'_>, interrupted: &Arc<AtomicBool>) -> anyhow::Result<()> {
    let model_path = args.model_path.expect("autocam model required");

    // Load calibration
    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(args.calibration))?;
    let effective_sync = if args.sync_offset != 0 {
        args.sync_offset
    } else {
        cal.sync_offset
    };

    // Initialize GPU and open smart source
    let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
    let mut source = reco_io::SmartFileSource::open(args.left, args.right, &gpu, effective_sync)?;
    let info = source.info();

    let input_format = if source.is_gpu_resident() {
        reco_core::renderer::InputFormat::Nv12
    } else {
        reco_core::renderer::InputFormat::Yuv420p
    };

    // Build session
    let viewport = reco_core::viewport::ViewportConfig {
        width: args.width,
        height: args.height,
        blend_width: args.blend,
        rig_tilt: cal.rig_tilt as f32,
        ..Default::default()
    };
    let session_config = reco_core::session::SessionConfig {
        calibration: cal,
        viewport,
        input_width: info.width,
        input_height: info.height,
        output_format: reco_core::gpu::OutputFormat::Rgba8Unorm,
        input_format,
        left_rotation: source.left_rotation(),
        right_rotation: source.right_rotation(),
    };
    let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

    // Configure GPU bind groups if source is GPU-resident
    #[cfg(target_os = "linux")]
    if let Some(shared) = source.shared_texture_set() {
        session.setup_gpu_source(shared);
    }

    println!(
        "Pipeline ready: GPU = {}, output = {}x{}, mode = {}",
        session.gpu_name(),
        args.width,
        args.height,
        source.decode_mode(),
    );

    // Set up encoder
    let fps_rational = info.fps_rational.unwrap_or((30, 1));
    let (encoder, enc_name) = reco_io::adapters::create_encoder(
        Path::new(args.output),
        args.width,
        args.height,
        (fps_rational.0, fps_rational.1),
        args.codec,
        args.quality,
        args.encoder_name.clone(),
    )?;
    println!("Encoder: {enc_name}");
    session.set_encoder(Box::new(encoder), 2);

    // Set up autocam
    match reco_autocam::setup_autocam(
        &mut session,
        model_path,
        info.width,
        info.height,
        info.fps as f32,
        source.is_gpu_resident(),
        args.detection_interval,
        args.lead_time,
    ) {
        Ok(true) => println!("Autocam: ball tracking enabled (model: {model_path})"),
        Ok(false) => {
            eprintln!(
                "Warning: ball tracking unavailable in {} mode (build with --features tensorrt for GPU detection, \
                 or use CPU decode)",
                source.decode_mode(),
            );
        }
        Err(e) => eprintln!("Warning: autocam setup failed ({e}), continuing without tracking"),
    }

    // Run
    let frame_limit =
        reco_core::session::compute_frame_limit(args.duration, args.max_frames, info.fps);
    if frame_limit < u64::MAX {
        println!("Processing up to {frame_limit} frames");
    }

    let progress = crate::helpers::ProgressReporter::new(30);
    let frame_count = session.run(
        &mut source,
        frame_limit,
        interrupted,
        Some(Box::new(move |p: &reco_core::session::FrameProgress| {
            progress.report(p.frames_completed);
        })),
    )?;
    session.finish()?;

    let progress = crate::helpers::ProgressReporter::new(30);
    progress.finish(frame_count, args.output);
    Ok(())
}

fn parse_codec(s: &str) -> reco_core::output::Codec {
    match s {
        "hevc" | "h265" => reco_core::output::Codec::HEVC,
        "av1" => reco_core::output::Codec::AV1,
        _ => reco_core::output::Codec::H264,
    }
}

fn parse_quality(s: &str) -> reco_core::output::Quality {
    match s {
        "fast" => reco_core::output::Quality::Fast,
        "high" => reco_core::output::Quality::High,
        _ => reco_core::output::Quality::Balanced,
    }
}
