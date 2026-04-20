//! Stitch subcommand: encode two video files into a panoramic output.
//!
//! Uses `StitchJob` (Layer 3 API) for all cases, including autocam.
//! The `on_session` callback wires up detection and direction when a
//! YOLO model is provided.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Arguments for the stitch subcommand, collected from CLI parsing.
///
/// `detection_interval`, `lead_time`, and `tracking_mode` are only
/// consumed inside `#[cfg(feature = "autocam")]` blocks below, so
/// `--no-default-features` builds see them as dead. Silence the lint
/// here instead of per-field gating to keep the struct shape stable
/// across features.
#[allow(dead_code)]
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
    pub tracking_mode: &'a str,
    pub crf: Option<u8>,
    pub preset: Option<String>,
    /// Output container selector (`mp4` / `fmp4` / `mkv`). None
    /// means default (plain MP4, finalized at close). `mkv` or
    /// `fmp4` for streamable tee use.
    pub container: Option<&'a str>,
    /// Optional replay-recording output path. When `Some`, the
    /// stitch job writes a stacked-video copy of the source frames
    /// alongside the stitched output (M6.5 feature, `replay`
    /// feature flag on reco-cli).
    pub replay_path: Option<&'a str>,
    /// Optional replay-tile downscale `(width, height)`. When
    /// `Some`, the GPU pack shader produces smaller replay tiles
    /// (FRICTION reco-obs A19). Has no effect without
    /// [`Self::replay_path`]. GPU path only — CPU-resident
    /// sources log a warn and record at source dims.
    pub replay_scale: Option<(u32, u32)>,
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
    // and pass the pre-loaded calibration to StitchJob. `field_roi` is
    // only consumed under the autocam feature; a leading underscore
    // silences the unused-var lint on `--no-default-features` builds.
    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(args.calibration))?;
    #[cfg_attr(not(feature = "autocam"), allow(unused_variables))]
    let field_roi = cal.field_roi.clone();

    let mut job = reco_io::StitchJob::with_calibration(args.left, args.right, cal, args.output)
        .codec(parse_codec(args.codec))
        .quality(parse_quality(args.quality))
        .resolution(args.width, args.height)
        .blend_width(args.blend)
        .on_progress(move |p: &reco_core::session::FrameProgress| {
            // Use the session's own elapsed clock so the reported
            // rate excludes one-time GPU / encoder / ORT init and
            // reflects only the decode → stitch → encode loop.
            progress.report_with_elapsed(p.frames_completed, p.elapsed);
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
    if let Some(container) = args.container {
        let c =
            reco_io::ffmpeg::encoder::Container::from_str_loose(container).ok_or_else(|| {
                anyhow::anyhow!("unknown container '{container}' (expected mp4, fmp4, or mkv)")
            })?;
        job = job.format(match c {
            reco_io::ffmpeg::encoder::Container::Mp4 => reco_io::output::Format::Mp4,
            reco_io::ffmpeg::encoder::Container::Mp4Fragmented => {
                reco_io::output::Format::Mp4Fragmented
            }
            reco_io::ffmpeg::encoder::Container::Matroska => reco_io::output::Format::Mkv,
        });
    }

    // Opt-in replay recording. The builder call is all the
    // consumer needs - StitchJob owns the encoder lifecycle, the
    // per-frame tap, and the finalize.
    #[cfg(feature = "replay")]
    if let Some(replay_path) = args.replay_path {
        job = job.with_replay_recording(replay_path);
        if let Some((w, h)) = args.replay_scale {
            job = job.with_replay_scale(w, h);
            println!("Replay recording: {replay_path} (scaled to {w}x{h} per tile)");
        } else {
            println!("Replay recording: {replay_path}");
        }
    }
    #[cfg(feature = "replay")]
    if args.replay_scale.is_some() && args.replay_path.is_none() {
        log::warn!("--replay-scale specified without --replay; ignoring.");
    }
    #[cfg(not(feature = "replay"))]
    if args.replay_path.is_some() {
        log::warn!(
            "--replay specified but `replay` feature is disabled. \
             Build with --features replay to enable."
        );
    }

    // Sweep panner needs no model - attach it directly.
    #[cfg(feature = "autocam")]
    if args.tracking_mode == "sweep" {
        job = job.on_session(|session, _source| {
            // Use 80% of coverage max FOV so the viewport fits comfortably.
            let max_fov = session.coverage().map_or(50.0, |c| c.max_fov_degrees());
            let sweep_fov = (max_fov * 0.8).clamp(5.0, 50.0);
            let panner =
                Box::new(reco_autocam::panners::SweepPanner::new(0.8, 10.0).with_fov(sweep_fov));
            session.set_panner(panner);
            log::info!("Tracking mode: sweep (debug, FOV={sweep_fov:.1} deg)");
        });
    }

    // Wire up autocam via the on_session callback if a model is provided.
    #[cfg(feature = "autocam")]
    if args.tracking_mode != "sweep"
        && let Some(model_path) = args.model_path
    {
        let model_path = model_path.to_owned();
        let interval = args.detection_interval;
        let lead = args.lead_time;
        let mode_str = args.tracking_mode.to_owned();
        job = job.on_session(move |session, source| {
            let info = source.info();
            let mode = match mode_str.as_str() {
                "field" => reco_autocam::TrackingMode::Field,
                "sweep" => reco_autocam::TrackingMode::Sweep,
                _ => reco_autocam::TrackingMode::Ball,
            };
            let is_10bit = source.gpu_pixel_format() == reco_core::renderer::GpuPixelFormat::P010;
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
                is_10bit,
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
    #[cfg(not(feature = "autocam"))]
    if args.model_path.is_some() {
        log::warn!(
            "--model specified but autocam feature is disabled. Build with --features autocam to enable AI tracking."
        );
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
        "h264" | "avc" => reco_io::output::Codec::H264,
        "hevc" | "h265" => reco_io::output::Codec::HEVC,
        "av1" => reco_io::output::Codec::AV1,
        other => {
            log::warn!("Unknown codec '{other}', defaulting to H.264");
            reco_io::output::Codec::H264
        }
    }
}

fn parse_quality(s: &str) -> reco_io::output::Quality {
    match s {
        "fast" => reco_io::output::Quality::Fast,
        "balanced" => reco_io::output::Quality::Balanced,
        "high" => reco_io::output::Quality::High,
        other => {
            log::warn!("Unknown quality '{other}', defaulting to balanced");
            reco_io::output::Quality::Balanced
        }
    }
}
