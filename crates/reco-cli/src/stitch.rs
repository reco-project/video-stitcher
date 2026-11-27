//! Stitch subcommand: encode per-camera video files into a panoramic
//! output.
//!
//! Uses `StitchJob` (Layer 3 API) for all cases, including autocam.
//! The `on_session` callback wires up detection and direction when a
//! YOLO model is provided.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub mod input;

/// Arguments for the stitch subcommand, collected from CLI parsing.
///
/// `detection_interval`, `lookahead`, and `tracking_mode` are only
/// consumed inside `#[cfg(feature = "autocam")]` blocks below, so
/// `--no-default-features` builds see them as dead. Silence the lint
/// here instead of per-field gating to keep the struct shape stable
/// across features.
#[allow(dead_code)]
pub struct StitchArgs<'a> {
    /// Input videos, one per camera in projection order.
    pub inputs: &'a [String],
    /// Explicit live input cadence and synchronization controls.
    pub live_input: input::LiveInputArgs,
    pub audio_source: reco_io::output::AudioMode,
    pub calibration: &'a str,
    pub output: &'a str,
    pub stream_url: Option<String>,
    pub width: u32,
    pub height: u32,
    pub blend: Option<f32>,
    pub panorama_arc: Option<f64>,
    pub panorama_vertical_fov: Option<f64>,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub max_frames: Option<u64>,
    pub encoder_name: Option<String>,
    /// Device identifier for the named encoder.
    pub encoder_device: Option<String>,
    pub codec: Option<&'a str>,
    pub quality: &'a str,
    pub sync_offset: Option<i64>,
    pub model_path: Option<&'a str>,
    pub detection_interval: u64,
    pub lookahead: f64,
    pub lookahead_memory_bytes: Option<usize>,
    pub tracking_mode: &'a str,
    pub quality_value: Option<u8>,
    pub preset: Option<String>,
    /// Output container selector (`mp4` / `fmp4` / `mkv`). None
    /// means default (plain MP4, finalized at close). `mkv` or
    /// `fmp4` for streamable tee use.
    pub container: Option<&'a str>,
    /// Optional replay-recording output path. When `Some`, the
    /// stitch job writes a stacked-video copy of the source frames
    /// alongside the stitched output (requires the `replay` feature
    /// flag on reco-cli).
    pub replay_path: Option<&'a str>,
    /// Optional replay-tile downscale `(width, height)`. When
    /// `Some`, the GPU pack shader produces smaller replay tiles.
    /// Has no effect without [`Self::replay_path`]. GPU path only -
    /// CPU-resident sources log a warn and record at source dims.
    pub replay_scale: Option<(u32, u32)>,
    /// Deliver host images for CPU preprocessing while allowing hardware decoding.
    pub no_zero_copy: bool,
    pub cpu: bool,
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
    #[cfg(feature = "autocam")]
    let tracking_mode = crate::helpers::parse_tracking_mode(args.tracking_mode)?;
    const MAX_DIM: u32 = 8192;
    anyhow::ensure!(
        args.width > 0 && args.width <= MAX_DIM && args.height > 0 && args.height <= MAX_DIM,
        "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
        args.width,
        args.height,
    );

    #[cfg(feature = "autocam")]
    if args.trajectory_path.is_none()
        && tracking_mode != reco_autocam::TrackingMode::Sweep
        && let Some(model_path) = args.model_path
    {
        reco_autocam::validate_model_path(Path::new(model_path))?;
    }

    let progress = crate::helpers::ProgressReporter::new(30);
    let stream = Arc::new(std::sync::OnceLock::<reco_io::ffmpeg::encoder::StreamMonitor>::new());
    let stream_ready = stream.clone();
    let stream_progress = stream.clone();
    let mut last_stream_status = None;

    // Load and validate the accepted calibration.
    let mut cal = reco_core::calibration::Calibration::from_file(Path::new(args.calibration))?;
    if let (Some(arc_deg), Some(vertical_fov_deg)) = (args.panorama_arc, args.panorama_vertical_fov)
    {
        let rig = cal
            .projection
            .l_shape()
            .ok_or_else(|| anyhow::anyhow!("--panorama-arc requires an L-shape calibration"))?
            .clone();
        cal.projection = reco_core::projection::Panorama {
            rig,
            arc_deg,
            vertical_fov_deg,
        }
        .into();
        cal.validate()?;
    }
    if matches!(
        cal.projection,
        reco_core::calibration::Projection::Panorama(_)
    ) {
        anyhow::ensure!(
            args.model_path.is_none()
                && args.trajectory_path.is_none()
                && args.tracking_mode != "sweep",
            "fixed panorama output does not support autocam or trajectory pan/zoom"
        );
    }
    let mut job = args.live_input.job(args.inputs, cal, args.output)?;
    for path in [
        Some(args.calibration),
        args.model_path,
        args.trajectory_path,
        args.panner_config_path,
    ]
    .into_iter()
    .flatten()
    {
        job = job.protect_input(path);
    }
    job = job
        .audio(args.audio_source)
        .quality(args.quality.parse().map_err(anyhow::Error::msg)?)
        .resolution(args.width, args.height)
        .on_stream_ready(move |monitor| {
            let _ = stream_ready.set(monitor);
        })
        .on_progress(move |p: &reco_core::session::types::FrameProgress| {
            if let Some(monitor) = stream_progress.get() {
                let status = monitor.status();
                if last_stream_status.as_ref() != Some(&status) {
                    eprintln!("Stream: {status:?}");
                    last_stream_status = Some(status);
                }
            }
            // Use the session's own elapsed clock so the reported
            // rate excludes one-time GPU / encoder / ORT init and
            // reflects only the decode → stitch → encode loop.
            progress.report_with_elapsed(p.frames_completed, p.elapsed);
        });

    if let Some(url) = args.stream_url {
        job = job.stream_to(url);
    }

    if let Some(b) = args.blend {
        job = job.blend_width(b);
    }
    if let Some(t) = args.start_time {
        job = job.start_time(t);
    }
    if let Some(t) = args.end_time {
        job = job.end_time(t);
    }
    if let Some(n) = args.max_frames {
        job = job.max_frames(n);
    }
    if let Some(offset) = args.sync_offset {
        job = job.sync_offset(offset);
    }
    if args.no_zero_copy {
        job = job.force_cpu_decode();
    }
    if args.cpu {
        job = job.cpu();
    }
    let tracking_active = args.trajectory_path.is_none()
        && args.model_path.is_some()
        && args.tracking_mode != "sweep";
    if let Some(bytes) = args.lookahead_memory_bytes {
        job = job.lookahead_memory_limit(bytes);
    }
    if args.lookahead > 0.0 {
        if tracking_active {
            job = job.lookahead(args.lookahead);
            log::info!(
                "Lookahead: {:.1}s buffer enabled (AI tracking active)",
                args.lookahead
            );
        } else {
            log::debug!(
                "Lookahead {:.1}s ignored: no tracking (needs --model with field or ball tracking); \
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
    if let Some(codec) = args.codec {
        job = job.codec(codec.parse().map_err(anyhow::Error::msg)?);
    }
    if let Some(ref device) = args.encoder_device {
        job = job.encoder_device(device);
    }
    if let Some(qv) = args.quality_value {
        job = job.quality(reco_io::output::Quality::Value(qv));
    }
    if let Some(ref preset) = args.preset {
        job = job.preset(preset);
    }
    if let Some(container) = args.container {
        let fmt: reco_io::output::Container = container
            .parse()
            .map_err(|e: String| anyhow::anyhow!("{e} (expected mp4, fmp4, mkv, mov, or flv)"))?;
        job = job.container(fmt);
    }

    // Opt-in replay recording. The builder call is all the
    // consumer needs - StitchJob owns the encoder lifecycle, the
    // per-frame tap, and the finalize.
    if let Some(replay_path) = args.replay_path {
        job = job.with_replay_recording(replay_path);
        if let Some((w, h)) = args.replay_scale {
            job = job.with_replay_scale(w, h);
            println!("Replay recording: {replay_path} (scaled to {w}x{h} per tile)");
        } else {
            println!("Replay recording: {replay_path}");
        }
    }
    if args.replay_scale.is_some() && args.replay_path.is_none() {
        log::warn!("--replay-scale specified without --replay; ignoring.");
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
                        Ok(())
                    }
                    Err(e) => Err(reco_io::stitch_job::StitchError::Other(format!(
                        "Failed to load trajectory {traj_path}: {e}"
                    ))),
                },
            );
    }

    // Sweep panner needs no model - attach it directly.
    #[cfg(feature = "autocam")]
    if args.trajectory_path.is_none() && args.tracking_mode == "sweep" {
        job = job.on_session(|session, _source| {
            let max_fov = session
                .core_mut()
                .prepare_coverage()
                .map_err(|error| reco_io::stitch_job::StitchError::Other(error.to_string()))?;
            let sweep_fov = (max_fov * 0.8).clamp(1.0, 50.0);
            let panner =
                Box::new(reco_autocam::panners::SweepPanner::new(0.8, 10.0).with_fov(sweep_fov));
            session.set_panner(panner);
            log::info!("Tracking mode: sweep (debug, FOV={sweep_fov:.1} deg)");
            Ok(())
        });
    }

    // Wire up autocam via the on_session callback if a model is provided.
    #[cfg(feature = "autocam")]
    if args.trajectory_path.is_none()
        && args.tracking_mode != "sweep"
        && let Some(model_path) = args.model_path
    {
        let model_path = model_path.to_owned();
        let interval = args.detection_interval;
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
            let is_10bit =
                source.gpu_pixel_format() == reco_core::render::renderer::GpuPixelFormat::P010;
            let mut autocam_config = reco_autocam::AutocamConfig::new(&model_path)
                .with_tracking_mode(tracking_mode)
                .with_detection_interval(interval)
                .with_10bit(is_10bit);
            if tracking_mode == reco_autocam::TrackingMode::Ball {
                autocam_config.confidence_threshold = Some(0.25);
            }
            if let Some(ref cfg) = panner_cfg {
                autocam_config.field_panner_config = Some(cfg.clone());
            }
            match reco_autocam::setup_autocam(
                session,
                &autocam_config,
                info.fps as f32,
                source.is_gpu_resident(),
            ) {
                Ok(true) => println!("Autocam: tracking enabled"),
                Ok(false) => {
                    let msg = "Tracking requested but detection cannot run in zero-copy mode. \
                               Build with --features tensorrt for GPU detection, \
                               or use host image delivery (--no-zero-copy).";
                    return Err(reco_io::stitch_job::StitchError::Other(msg.into()));
                }
                Err(e) => {
                    return Err(reco_io::stitch_job::StitchError::Other(format!(
                        "Autocam setup failed: {e}"
                    )));
                }
            }
            Ok(())
        });
    }
    #[cfg(not(feature = "autocam"))]
    if args.model_path.is_some() || args.trajectory_path.is_some() || args.tracking_mode == "sweep"
    {
        anyhow::bail!("AI tracking is unavailable in this build");
    }

    let result = job.run(interrupted);
    if let Some(monitor) = stream.get() {
        eprintln!("Stream: {:?}", monitor.status());
    }
    let result = result?;
    report_result(&result, args.output)
}

fn report_result(result: &reco_io::StitchResult, output: &str) -> anyhow::Result<()> {
    let output = if output.contains("://") {
        "network output"
    } else {
        output
    };
    anyhow::ensure!(
        result.completion != reco_core::session::types::RunCompletion::Cancelled,
        "stitch cancelled after {} frames; finalized output: {}",
        result.frames_processed,
        output
    );
    println!(
        "\nDone: {} frames in {:.1}s ({:.1} fps) -> {}",
        result.frames_processed,
        result.elapsed.as_secs_f64(),
        result.fps(),
        output
    );

    if let Some(snap) = &result.telemetry {
        let summary = reco_core::telemetry::SessionSummary {
            snapshot: snap.clone(),
        };
        println!("\n{summary}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::session::types::RunCompletion;

    #[test]
    fn partial_export_is_not_reported_as_command_success() {
        let mut result = reco_io::StitchResult {
            completion: RunCompletion::Completed,
            frames_processed: 3,
            elapsed: std::time::Duration::from_secs(1),
            gpu_name: "test".into(),
            encoder_name: "test".into(),
            decode_mode: "test".into(),
            telemetry: None,
            stream_status: None,
        };
        assert!(report_result(&result, "output.mp4").is_ok());
        result.completion = RunCompletion::Cancelled;
        let error = report_result(&result, "output.mp4").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cancelled after 3 frames; finalized output: output.mp4")
        );
    }
}
