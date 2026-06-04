//! Export worker thread.
//!
//! Runs a [`StitchJob`](reco_io::StitchJob) on a background thread,
//! pumping progress back to the Slint UI via `invoke_from_event_loop`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use reco_core::calibration::MatchCalibration;

use crate::RecoApp;

/// Result published by the export thread.
#[derive(Debug)]
pub enum ExportOutcome {
    /// Finished successfully - carries (frames, output path).
    Ok(u64, PathBuf),
    /// Export was cancelled by the user.
    Cancelled,
    /// Export failed with the structured error.
    Failed(reco_io::stitch_job::StitchError),
}

/// Autocam + panner settings captured from the GUI export panel.
///
/// Slint properties can only be read on the UI thread, so the caller
/// snapshots them into this struct before spawning the worker thread.
/// String fields (`tracking_mode`, `framing`, `cluster_mode`) carry the
/// Slint combo values and are mapped to the reco-autocam enums inside
/// [`run_export`].
///
/// In an AI-less build (`--no-default-features`, no detection backend) the
/// fields are only snapshotted and never read, so dead-code is allowed for
/// that configuration alone.
#[cfg_attr(not(feature = "autocam"), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct AutocamUiConfig {
    /// Master toggle for the AI tracking pipeline.
    pub enabled: bool,
    /// Path to the YOLO model (ONNX/engine/NCNN dir). Empty = no-op.
    pub model_path: String,
    /// `"field"`, `"ball"`, or `"sweep"`.
    pub tracking_mode: String,
    /// Run the detector every N frames.
    pub detection_interval: u32,
    /// Lookahead buffer depth in seconds (0 = off).
    pub lookahead_secs: f64,
    /// `"action"` or `"frame_all"`.
    pub framing: String,
    /// Horizontal-only pan (hold pitch level).
    pub lock_pitch: bool,
    /// `"density"` or `"trimmed_mean"`.
    pub cluster_mode: String,
    /// Density-peak neighborhood / trim window, radians.
    pub cluster_bandwidth_rad: f32,
    /// Soft dead-zone radius, radians.
    pub dead_zone_rad: f32,
    /// Ball-vs-cluster blend weight (0..1); forced to 1.0 in ball mode.
    pub ball_weight: f32,
    /// Tight / wide / default field-of-view, degrees.
    pub fov_tight: f32,
    pub fov_wide: f32,
    pub fov_default: f32,
}

/// Telemetry sink that forwards snapshots to the Slint UI thread.
struct ExportTelemetrySink {
    window: slint::Weak<RecoApp>,
}

impl reco_core::telemetry::TelemetrySink for ExportTelemetrySink {
    fn on_snapshot(&mut self, snap: &reco_core::telemetry::TelemetrySnapshot) {
        let snap = snap.clone();
        let weak = self.window.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_telem_fps_avg(snap.fps_average);
                app.set_telem_fps_recent(snap.fps_recent);
                app.set_telem_decode_ms(snap.avg_decode_ms);
                app.set_telem_stitch_ms(snap.avg_stitch_ms);
                app.set_telem_readback_ms(snap.avg_readback_ms);
                app.set_telem_encode_ms(snap.avg_encode_ms);
                app.set_telem_total_ms(snap.avg_total_ms);
                app.set_telem_p99_ms(snap.p99_total_ms);
                app.set_telem_detection_ms(snap.avg_detection_ms);
                app.set_telem_active_tracks(snap.active_tracks as i32);
                app.set_telem_ball_pct(snap.ball_presence_pct);
                app.set_telem_det_per_frame(snap.detections_per_frame);
                app.set_telem_gpu_name(snap.gpu_name.clone().into());
                app.set_telem_bottleneck(
                    snap.bottleneck
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                        .into(),
                );
            }
        });
    }
}

/// Run a StitchJob on the worker thread.
#[allow(clippy::too_many_arguments)]
pub fn run_export(
    left: reco_io::stitch_job::InputPath,
    right: reco_io::stitch_job::InputPath,
    cal: MatchCalibration,
    output: PathBuf,
    stream_url: Option<String>,
    replay_enabled: bool,
    events_path: Option<PathBuf>,
    width: u32,
    height: u32,
    codec_str: String,
    quality_str: String,
    blend: f32,
    start_secs: f32,
    end_secs: f32,
    autocam: AutocamUiConfig,
    app_weak: slint::Weak<RecoApp>,
    interrupted: &AtomicBool,
    last_progress_at: Arc<Mutex<Option<Instant>>>,
) -> ExportOutcome {
    use reco_io::output::{Codec, Quality};

    let codec: Codec = codec_str.parse().unwrap_or_default();
    let quality: Quality = quality_str.parse().unwrap_or_default();

    #[cfg(feature = "autocam")]
    let field_roi = cal.field_roi.clone();

    let post_status = |text: String| {
        let weak = app_weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_export_status_text(text.into());
            }
        });
    };

    post_status("Probing source...".into());

    use reco_core::source::FrameSource;
    if let Ok(source) = reco_io::adapters::FfmpegFileSource::open_from_inputs(&left, &right, 0)
        && let Some(full_total) = source.total_frames()
    {
        let fps = reco_io::adapters::FfmpegFileSource::frame_rate(left.first_path())
            .map(|(n, d)| if d != 0 { n as f64 / d as f64 } else { 30.0 })
            .unwrap_or(30.0);

        let start_frames = if start_secs > 0.0 {
            (start_secs as f64 * fps) as u64
        } else {
            0
        };
        let end_frames = if end_secs > 0.0 {
            (end_secs as f64 * fps) as u64
        } else {
            full_total
        };
        let range_total = end_frames.saturating_sub(start_frames);
        let weak = app_weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_export_frames_total(range_total as i32);
            }
        });
    }

    post_status("Opening encoder and decoders...".into());

    let progress_weak = app_weak.clone();
    let progress_start = Instant::now();
    let progress_last_at = Arc::clone(&last_progress_at);
    let effective_output = stream_url
        .as_ref()
        .filter(|u| !u.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| output.clone());
    let format = reco_io::output::Format::for_output(&effective_output.to_string_lossy());
    if format.is_streaming() {
        log::info!("Streaming to {}", effective_output.display());
    }

    let mut job = reco_io::StitchJob::with_calibration(
        left.clone(),
        right.clone(),
        cal,
        effective_output.clone(),
    )
    .codec(codec)
    .quality(quality)
    .format(format)
    .resolution(width, height)
    .blend_width(blend)
    .on_progress(move |p: &reco_core::session::types::FrameProgress| {
        let frames = p.frames_completed;
        let elapsed = progress_start.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            frames as f64 / elapsed
        } else {
            0.0
        };
        *progress_last_at.lock().unwrap() = Some(Instant::now());

        let weak = progress_weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_export_frames_done(frames as i32);
                let total = app.get_export_frames_total();
                if total > 0 {
                    app.set_export_progress(frames as f32 / total as f32);
                }
                let eta = if total > 0 && fps > 0.0 {
                    let remaining = (total as f64 - frames as f64) / fps;
                    let mins = remaining as u64 / 60;
                    let secs = remaining as u64 % 60;
                    format!(" - ~{mins}:{secs:02} remaining")
                } else {
                    String::new()
                };
                app.set_export_status_text(format!("Frame {frames} ({fps:.0} fps){eta}").into());
            }
        });
    });

    if start_secs > 0.0 {
        let skip_frames = (start_secs as f64 * 30.0) as u64;
        post_status(format!(
            "Seeking to {start_secs:.0}s (skipping ~{skip_frames} frames)..."
        ));
        job = job.start_time(start_secs as f64);
    }
    if end_secs > 0.0 {
        job = job.end_time(end_secs as f64);
    }

    if replay_enabled {
        let replay_path = effective_output.with_extension("replay.mkv");
        log::info!("Replay recording: {}", replay_path.display());
        job = job.with_replay_recording(&replay_path);
    }

    if let Some(ref ep) = events_path {
        log::info!("Pipeline events: {}", ep.display());
        job = job.events(ep);
    }

    let finalizing_weak = app_weak.clone();
    job = job.on_finalizing(move || {
        let weak = finalizing_weak;
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_export_status_text("Finalizing output file...".into());
            }
        });
    });

    let telem_weak = app_weak.clone();
    job = job.on_session(move |session, _source| {
        let sink = ExportTelemetrySink { window: telem_weak };
        session.telemetry_mut().set_sink(Box::new(sink), 30);
    });

    // Lookahead buffers N future frames so the panner can centered-smooth
    // its pose stream. It is the dominant quality lever for AI panning, so
    // the GUI defaults it on; the baked dead_zone_rad assumes it is active.
    #[cfg(feature = "autocam")]
    if autocam.enabled && autocam.lookahead_secs > 0.0 {
        job = job.lookahead(autocam.lookahead_secs);
    }

    #[cfg(feature = "autocam")]
    if autocam.enabled && !autocam.model_path.is_empty() {
        let ac = autocam.clone();
        let status_weak = app_weak.clone();
        job = job.on_session(move |session, source| {
            let info = source.info();
            let mode = match ac.tracking_mode.as_str() {
                "ball" => reco_autocam::TrackingMode::Ball,
                "sweep" => reco_autocam::TrackingMode::Sweep,
                _ => reco_autocam::TrackingMode::Field,
            };
            let is_10bit =
                source.gpu_pixel_format() == reco_core::render::renderer::GpuPixelFormat::P010;

            // Map the GUI knobs onto FieldPannerConfig, leaving every other
            // field at its validated default. Ball mode's ball_weight=1.0 is
            // owned by setup_autocam, so we pass the slider value as-is.
            let framing = match ac.framing.as_str() {
                "frame_all" => reco_autocam::panners::FramingMode::FrameAll,
                _ => reco_autocam::panners::FramingMode::Action,
            };
            let cluster_mode = match ac.cluster_mode.as_str() {
                "trimmed_mean" => reco_autocam::panners::ClusterMode::TrimmedMean,
                _ => reco_autocam::panners::ClusterMode::Density,
            };
            let panner_cfg = reco_autocam::panners::FieldPannerConfig {
                framing,
                cluster_mode,
                cluster_bandwidth_rad: ac.cluster_bandwidth_rad,
                dead_zone_rad: ac.dead_zone_rad,
                ball_weight: ac.ball_weight,
                lock_pitch: ac.lock_pitch,
                fov_tight: ac.fov_tight,
                fov_wide: ac.fov_wide,
                fov_default: ac.fov_default,
                ..Default::default()
            };

            let mut autocam_config = reco_autocam::AutocamConfig::new(&ac.model_path)
                .with_tracking_mode(mode)
                .with_detection_interval(ac.detection_interval as u64)
                .with_10bit(is_10bit);
            autocam_config.field_panner_config = Some(panner_cfg);
            // Ball-only models need a higher floor than the 0.10 field
            // default (matches the CLI's ball-mode override).
            if mode == reco_autocam::TrackingMode::Ball {
                autocam_config.confidence_threshold = Some(0.25);
            }
            let autocam_config = if let Some(roi) = field_roi.as_ref() {
                autocam_config.with_field_roi(roi.clone())
            } else {
                autocam_config
            };
            let result = reco_autocam::setup_autocam(
                session,
                &autocam_config,
                info.fps as f32,
                source.is_gpu_resident(),
            );
            let is_failure = !matches!(&result, Ok(true));
            let banner: String = match result {
                Ok(true) => "AI tracking: active".into(),
                Ok(false) => {
                    "AI tracking unavailable - build with --features tensorrt, or use CPU decode"
                        .into()
                }
                Err(e) => format!("AI tracking failed: {e}"),
            };
            log::info!("Export: {banner}");
            let weak = status_weak.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(app) = weak.upgrade() {
                    if is_failure {
                        app.set_export_status_text(banner.clone().into());
                    }
                    app.set_status_text(banner.into());
                }
            });
        });
    }
    #[cfg(not(feature = "autocam"))]
    let _ = &autocam;

    match job.run(interrupted) {
        Ok(r) => ExportOutcome::Ok(r.frames_processed, output),
        Err(e) => {
            if interrupted.load(Ordering::Relaxed) {
                ExportOutcome::Cancelled
            } else {
                ExportOutcome::Failed(e)
            }
        }
    }
}
