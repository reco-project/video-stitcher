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
/// `detection_interval`, `lookahead`, and `tracking_mode` are only
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
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub max_frames: Option<u64>,
    pub encoder_name: Option<String>,
    pub codec: &'a str,
    pub quality: &'a str,
    pub sync_offset: i64,
    pub model_path: Option<&'a str>,
    pub detection_interval: u64,
    pub lookahead: f64,
    pub tracking_mode: &'a str,
    pub quality_value: Option<u8>,
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
    /// When true, silently continue without tracking if detection
    /// cannot run (e.g. zero-copy mode without TensorRT). Default
    /// false: error out so the user knows tracking was requested but
    /// not delivered.
    pub allow_no_tracking: bool,
    /// Force CPU decode to enable ORT CPU detection without TensorRT.
    pub no_zero_copy: bool,
    /// Path for pipeline event JSONL output.
    pub events_path: Option<&'a str>,
    /// Precomputed trajectory CSV (overrides AI panner).
    pub trajectory_path: Option<&'a str>,
    /// FieldPanner tuning JSON (field mode); only present keys override.
    pub panner_config_path: Option<&'a str>,
    /// Named panner preset (base config); JSON overlays on top.
    pub panner_preset: Option<&'a str>,
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

    // Accept `a.mp4;b.mp4;c.mp4` to chain segments via the concat demuxer
    // (mirrors the GUI's multi-segment selection). A single path stays Single.
    let to_input = |s: &str| -> reco_io::stitch_job::InputPath {
        let parts: Vec<std::path::PathBuf> = s
            .split(';')
            .filter(|p| !p.is_empty())
            .map(std::path::PathBuf::from)
            .collect();
        if parts.len() > 1 {
            log::info!(
                "CLI input: {} segments, chaining via concat demuxer",
                parts.len()
            );
            reco_io::stitch_job::InputPath::Chained(parts)
        } else {
            reco_io::stitch_job::InputPath::Single(std::path::PathBuf::from(s))
        }
    };
    let mut job = reco_io::StitchJob::with_calibration(
        to_input(args.left),
        to_input(args.right),
        cal,
        args.output,
    )
    .codec(parse_codec(args.codec))
    .quality(parse_quality(args.quality))
    .resolution(args.width, args.height)
    .blend_width(args.blend)
    .on_progress(move |p: &reco_core::session::types::FrameProgress| {
        // Use the session's own elapsed clock so the reported
        // rate excludes one-time GPU / encoder / ORT init and
        // reflects only the decode → stitch → encode loop.
        progress.report_with_elapsed(p.frames_completed, p.elapsed);
    });

    if let Some(t) = args.start_time {
        job = job.start_time(t);
    }
    if let Some(t) = args.end_time {
        job = job.end_time(t);
    }
    if let Some(n) = args.max_frames {
        job = job.max_frames(n);
    }
    if args.sync_offset != 0 {
        job = job.sync_offset(args.sync_offset);
    }
    if args.no_zero_copy {
        job = job.force_cpu_decode();
    }
    // Lookahead only helps when an AI panner drives the camera: it buffers
    // future frames so the panner can lead and the loop can centered-smooth.
    // For a plain stitch (no model) or sweep mode it would only add latency
    // and VRAM, so skip it and say why.
    let tracking_active = args.model_path.is_some() && args.tracking_mode != "sweep";
    if args.lookahead > 0.0 {
        if tracking_active {
            job = job.lookahead(args.lookahead);
            log::info!(
                "Lookahead: {:.1}s buffer enabled (AI tracking active)",
                args.lookahead
            );
        } else {
            log::debug!(
                "Lookahead {:.1}s ignored: no AI tracking (needs --model, non-sweep); \
                 a plain stitch needs none",
                args.lookahead
            );
        }
    }
    if let Some(path) = args.events_path {
        job = job.events(path);
    }
    if let Some(ref enc) = args.encoder_name {
        job = job.encoder_name(enc);
    }
    if let Some(qv) = args.quality_value {
        job = job.quality_value(qv);
    }
    if let Some(ref preset) = args.preset {
        job = job.preset(preset);
    }
    if let Some(container) = args.container {
        let fmt: reco_io::output::Format = container
            .parse()
            .map_err(|e: String| anyhow::anyhow!("{e} (expected mp4, fmp4, mkv, mov, or flv)"))?;
        job = job.format(fmt);
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

    // Precomputed trajectory file overrides all AI tracking.
    #[cfg(feature = "autocam")]
    if let Some(traj_path) = args.trajectory_path {
        let traj_path = traj_path.to_owned();
        job =
            job.on_session(
                move |session, _source| match reco_autocam::panners::FilePanner::from_csv(
                    std::path::Path::new(&traj_path),
                ) {
                    Ok(panner) => {
                        session.set_panner(Box::new(panner));
                        log::info!("Tracking mode: precomputed trajectory from {traj_path}");
                    }
                    Err(e) => {
                        log::error!("Failed to load trajectory {traj_path}: {e}");
                    }
                },
            );
    }

    // Sweep panner needs no model - attach it directly.
    #[cfg(feature = "autocam")]
    if args.trajectory_path.is_none() && args.tracking_mode == "sweep" {
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

    // Flag set inside the on_session callback when tracking was requested
    // but couldn't be initialized. Checked after job.run() to produce a
    // clean error exit without segfaulting (process::exit inside a GPU
    // callback crashes NVDEC/Vulkan teardown).
    #[cfg(feature = "autocam")]
    let tracking_failed = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Wire up autocam via the on_session callback if a model is provided.
    #[cfg(feature = "autocam")]
    if args.trajectory_path.is_none()
        && args.tracking_mode != "sweep"
        && let Some(model_path) = args.model_path
    {
        let model_path = model_path.to_owned();
        let interval = args.detection_interval;
        let mode_str = args.tracking_mode.to_owned();
        let allow_fallback = args.allow_no_tracking;
        let tracking_failed = Arc::clone(&tracking_failed);
        // Resolve FieldPanner tuning up front so a bad preset/file fails
        // before rendering. Preset is the base; --panner-config overlays.
        let panner_cfg: Option<reco_autocam::panners::FieldPannerConfig> = {
            use reco_autocam::panners::{FieldPannerConfig, PRESET_NAMES};
            let base = match args.panner_preset {
                Some(name) => {
                    let c = FieldPannerConfig::from_preset_name(name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown --panner-preset '{name}' (expected: {})",
                            PRESET_NAMES.join(", ")
                        )
                    })?;
                    log::info!("FieldPanner preset: {name}");
                    Some(c)
                }
                None => None,
            };
            match args.panner_config_path {
                Some(p) => {
                    let contents = std::fs::read_to_string(p)
                        .map_err(|e| anyhow::anyhow!("reading panner config {p}: {e}"))?;
                    let cfg = if let Some(b) = base {
                        let mut v = serde_json::to_value(&b).expect("config serializes");
                        let over: serde_json::Value = serde_json::from_str(&contents)
                            .map_err(|e| anyhow::anyhow!("parsing panner config {p}: {e}"))?;
                        if let (Some(bm), serde_json::Value::Object(om)) = (v.as_object_mut(), over)
                        {
                            bm.extend(om);
                        }
                        serde_json::from_value(v)
                            .map_err(|e| anyhow::anyhow!("applying panner config {p}: {e}"))?
                    } else {
                        serde_json::from_str(&contents)
                            .map_err(|e| anyhow::anyhow!("parsing panner config {p}: {e}"))?
                    };
                    log::info!("FieldPanner config loaded from {p}");
                    Some(cfg)
                }
                None => base,
            }
        };
        job = job.on_session(move |session, source| {
            let info = source.info();
            let mode = match mode_str.as_str() {
                "sweep" => reco_autocam::TrackingMode::Sweep,
                "ball" => reco_autocam::TrackingMode::Ball,
                _ => reco_autocam::TrackingMode::Field,
            };
            let is_10bit =
                source.gpu_pixel_format() == reco_core::render::renderer::GpuPixelFormat::P010;
            let mut autocam_config = reco_autocam::AutocamConfig::new(&model_path)
                .with_tracking_mode(mode)
                .with_detection_interval(interval)
                .with_10bit(is_10bit);
            if mode == reco_autocam::TrackingMode::Ball {
                autocam_config.confidence_threshold = Some(0.25);
            }
            if let Some(ref cfg) = panner_cfg {
                autocam_config.field_panner_config = Some(cfg.clone());
            }
            let autocam_config = if let Some(roi) = field_roi {
                autocam_config.with_field_roi(roi)
            } else {
                autocam_config
            };
            match reco_autocam::setup_autocam(
                session,
                &autocam_config,
                info.fps as f32,
                source.is_gpu_resident(),
            ) {
                Ok(true) => println!("Autocam: tracking enabled (model: {model_path})"),
                Ok(false) => {
                    let msg = "Tracking requested but detection cannot run in zero-copy mode. \
                               Build with --features tensorrt for GPU detection, \
                               or use CPU decode (--no-zero-copy). \
                               Pass --allow-no-tracking to continue without tracking.";
                    if allow_fallback {
                        log::warn!("{msg}");
                    } else {
                        log::error!("{msg}");
                        tracking_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                Err(e) => {
                    let msg = format!(
                        "Autocam setup failed: {e}. \
                                       Pass --allow-no-tracking to continue without tracking."
                    );
                    if allow_fallback {
                        log::warn!("{msg}");
                    } else {
                        log::error!("{msg}");
                        tracking_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
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

    #[cfg(feature = "autocam")]
    if tracking_failed.load(std::sync::atomic::Ordering::Relaxed) {
        anyhow::bail!(
            "Tracking was requested but could not run. \
             Pass --allow-no-tracking to continue without tracking."
        );
    }

    println!(
        "\nDone: {} frames in {:.1}s ({:.1} fps) -> {}",
        result.frames_processed,
        result.elapsed.as_secs_f64(),
        result.fps(),
        args.output
    );

    if let Some(snap) = &result.telemetry {
        let summary = reco_core::telemetry::SessionSummary {
            snapshot: snap.clone(),
        };
        println!("\n{summary}");
    }

    Ok(())
}

fn parse_codec(s: &str) -> reco_io::output::Codec {
    s.parse().unwrap_or_else(|_| {
        log::warn!("Unknown codec '{s}', defaulting to H.264");
        reco_io::output::Codec::H264
    })
}

fn parse_quality(s: &str) -> reco_io::output::Quality {
    s.parse().unwrap_or_else(|_| {
        log::warn!("Unknown quality '{s}', defaulting to balanced");
        reco_io::output::Quality::Balanced
    })
}
