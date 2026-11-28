//! Export worker thread.
//!
//! Runs a [`StitchJob`](reco_io::StitchJob) on a background thread,
//! pumping progress back to the Slint UI via `invoke_from_event_loop`.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use reco_core::calibration::Calibration;

use crate::RecoApp;
use reco_io::ffmpeg::encoder::{StreamMonitor, StreamStatus};

/// Choose the initial dialog codec. Publishing uses H.264. HDR file export
/// selects HEVC when offered, and other file exports use the saved preference.
pub(crate) fn initial_codec(
    streaming: bool,
    hdr: bool,
    hevc_available: bool,
    user_default: &str,
) -> &str {
    if streaming {
        "h264"
    } else if hdr && hevc_available {
        "hevc"
    } else {
        user_default
    }
}

/// Settings snapshotted on the UI thread for one processed-output job.
pub struct ExportRequest {
    pub inputs: Inputs,
    pub calibration: Calibration,
    pub calibration_path: Option<PathBuf>,
    pub output: PathBuf,
    pub stream_url: Option<String>,
    pub audio: reco_io::output::AudioMode,
    pub replay_enabled: bool,
    pub events_path: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
    pub codec: String,
    pub quality: String,
    pub start_secs: f32,
    pub end_secs: f32,
    pub autocam: AutocamUiConfig,
}

/// Input opening remains owned by the shared job, not the interactive event loop.
pub enum Inputs {
    Files(Vec<reco_io::InputPath>),
    Network(crate::network::Config),
}

/// Observes optional publishing independently of the output worker's progress.
#[derive(Default)]
pub struct StreamState {
    monitor: Arc<OnceLock<StreamMonitor>>,
    last: Option<StreamStatus>,
}

impl StreamState {
    /// Share the job's eventual monitor without creating another transport worker.
    pub fn monitor(&self) -> Arc<OnceLock<StreamMonitor>> {
        self.monitor.clone()
    }

    /// Refresh only changed transport state, including idle and terminal failures.
    pub fn update(&mut self, app: &RecoApp) {
        let Some(monitor) = self.monitor.get() else {
            return;
        };
        let status = monitor.status();
        if self.last.as_ref() == Some(&status) {
            return;
        }
        let (text, failed) = stream_status(&status);
        app.set_export_stream_status(text.into());
        app.set_export_stream_failed(failed);
        self.last = Some(status);
    }

    /// Finalize visible status even when the job failed before creating an encoder.
    pub fn finish(&mut self, app: &RecoApp) {
        self.update(app);
        if self.monitor.get().is_none() && !app.get_export_stream_status().is_empty() {
            app.set_export_stream_status("Stream: not started".into());
            app.set_export_stream_failed(false);
        }
    }
}

fn stream_status(status: &StreamStatus) -> (String, bool) {
    match status {
        StreamStatus::Connecting => ("Stream: connecting…".into(), false),
        StreamStatus::Connected => (
            "Stream: connected (check playback on your service)".into(),
            false,
        ),
        StreamStatus::Finished => ("Stream: finished".into(), false),
        StreamStatus::Cancelled => ("Stream: stopped".into(), false),
        StreamStatus::Failed(error) => (format!("Stream failed: {error}"), true),
    }
}

/// Validate optional publishing before opening resources, without exposing its URL in errors.
pub fn validate_stream(url: Option<&str>, codec: &str) -> Result<(), String> {
    if let Some(url) = url {
        if !(url.starts_with("rtmp://") || url.starts_with("rtmps://"))
            || url
                .split_once("://")
                .is_none_or(|(_, rest)| rest.is_empty())
            || url
                .bytes()
                .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
        {
            return Err("Enter a complete RTMP or RTMPS publishing URL.".into());
        }
        if codec != "h264" {
            return Err("Recording and streaming share one encode; select H.264 to stream.".into());
        }
    }
    Ok(())
}

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
/// tuning fields are only snapshotted and never read, so dead-code is allowed for
/// that configuration alone.
#[cfg_attr(not(feature = "autocam"), allow(dead_code))]
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(Default))]
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
    /// Preset name used as the config base; visible knobs overlay it.
    pub preset: String,
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
    /// Margin added around detected players in frame-all mode, degrees.
    pub frame_all_margin_deg: f32,
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
                app.set_telem_decode_ms(snap.avg_decode_ms + snap.avg_upload_ms);
                app.set_telem_stitch_ms(snap.avg_stitch_ms);
                app.set_telem_readback_ms(snap.avg_readback_ms);
                app.set_telem_submit_ms(snap.avg_submit_ms);
                app.set_telem_total_ms(snap.avg_total_ms);
                app.set_telem_p99_ms(snap.p99_total_ms);
                app.set_telem_detection_ms(snap.avg_detection_ms);
                app.set_telem_tracking_ms(snap.avg_tracking_ms);
                app.set_telem_encode_ms(snap.avg_encode_worker_ms);
                app.set_telem_backpressure_ms(snap.backpressure_ms);
                app.set_telem_active_tracks(snap.active_tracks as i32);
                app.set_telem_ball_pct(snap.ball_presence_pct);
                app.set_telem_det_per_frame(snap.detections_per_frame);
                app.set_telem_gpu_name(snap.gpu_name.clone().into());
            }
        });
    }
}

/// Run a StitchJob on the worker thread.
pub fn run_export(
    request: ExportRequest,
    stream_monitor: Arc<OnceLock<StreamMonitor>>,
    app_weak: slint::Weak<RecoApp>,
    interrupted: &Arc<AtomicBool>,
) -> ExportOutcome {
    use reco_io::output::{Codec, Quality};

    let ExportRequest {
        inputs,
        calibration: cal,
        calibration_path,
        output,
        stream_url,
        audio,
        replay_enabled,
        events_path,
        width,
        height,
        codec: codec_str,
        quality: quality_str,
        start_secs,
        end_secs,
        autocam,
    } = request;
    if let Err(error) = validate_stream(stream_url.as_deref(), &codec_str) {
        return ExportOutcome::Failed(reco_io::stitch_job::StitchError::Other(error));
    }

    let (codec, quality) = match (codec_str.parse::<Codec>(), quality_str.parse::<Quality>()) {
        (Ok(codec), Ok(quality)) => (codec, quality),
        (Err(reason), _) | (_, Err(reason)) => {
            return ExportOutcome::Failed(reco_io::stitch_job::StitchError::Other(reason));
        }
    };

    let post_status = |text: String| {
        let weak = app_weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_export_status_text(text.into());
            }
        });
    };

    #[cfg(feature = "autocam")]
    if autocam.enabled
        && autocam.tracking_mode != "sweep"
        && !autocam.model_path.is_empty()
        && let Err(error) =
            reco_autocam::validate_model_path(std::path::Path::new(&autocam.model_path))
    {
        return ExportOutcome::Failed(reco_io::stitch_job::StitchError::Other(error.to_string()));
    }

    post_status("Opening encoder and decoders...".into());

    let progress_weak = app_weak.clone();
    let progress_start = Instant::now();
    let format = reco_io::output::Container::for_output(&output.to_string_lossy());

    let live = matches!(inputs, Inputs::Network(_));
    let mut job = match inputs {
        Inputs::Files(inputs) => reco_io::StitchJob::with_calibration(inputs, cal, &output),
        Inputs::Network(config) => {
            reco_io::StitchJob::from_rtsp(config.inputs, config.options, cal, &output)
        }
    }
    .audio(audio)
    .on_stream_ready(move |monitor| {
        let _ = stream_monitor.set(monitor);
    })
    .codec(codec)
    .quality(quality)
    .container(format)
    .resolution(width, height)
    .on_progress(move |p: &reco_core::session::types::FrameProgress| {
        let frames = p.frames_completed;
        let elapsed = progress_start.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            frames as f64 / elapsed
        } else {
            0.0
        };

        let weak = progress_weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_export_frames_done(frames as i32);
                let total = app.get_export_frames_total();
                if total > 0 {
                    app.set_export_progress((frames as f32 / total as f32).min(1.0));
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

    if let Some(path) = calibration_path {
        job = job.protect_input(path);
    }
    if autocam.enabled && !autocam.model_path.is_empty() {
        job = job.protect_input(&autocam.model_path);
    }
    if let Some(url) = stream_url {
        job = job.stream_to(url);
    }

    if start_secs > 0.0 {
        post_status(format!("Seeking to {start_secs:.0}s..."));
        job = job.start_time(start_secs as f64);
    }
    if end_secs > 0.0 {
        job = job.end_time(end_secs as f64);
    }

    if replay_enabled {
        let replay_path = output.with_extension("replay.mkv");
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
    job = job.on_session(move |session, source| {
        let total = estimated_export_frames(&source.info(), start_secs.into(), end_secs.into());
        let weak = telem_weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_export_frames_total(total.unwrap_or(0).min(i32::MAX as u64) as i32);
            }
        });
        let sink = ExportTelemetrySink { window: telem_weak };
        session.set_telemetry_sink(Box::new(sink), 30);
        Ok(())
    });

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
                frame_all_margin_deg: ac.frame_all_margin_deg,
                // Preset is the base; the visible knobs above overlay it
                // (they mirror the preset until the user tweaks them).
                ..reco_autocam::panners::FieldPannerConfig::from_preset_name(&ac.preset)
                    .unwrap_or_default()
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
            let result = reco_autocam::setup_autocam(
                session,
                &autocam_config,
                info.fps as f32,
                source.is_gpu_resident(),
            );
            match result {
                Ok(true) => {}
                Ok(false) => return Err(reco_io::stitch_job::StitchError::Other(
                    "AI tracking unavailable - build with --features tensorrt, or use CPU decode"
                        .into(),
                )),
                Err(e) => {
                    return Err(reco_io::stitch_job::StitchError::Other(format!(
                        "AI tracking failed: {e}"
                    )));
                }
            };
            let banner = "AI tracking: active";
            log::info!("Export: {banner}");
            let weak = status_weak.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(app) = weak.upgrade() {
                    // Banner goes to the export status only, never the main
                    // status bar. This async set can land after the export
                    // completion handler and would otherwise clobber the final
                    // outcome (e.g. the VRAM error) shown there.
                    app.set_export_status_text(banner.into());
                }
            });
            Ok(())
        });
    }
    // Defense in depth: the GUI blocks Start when AI is on with no model,
    // but if that guard is ever bypassed, say so rather than silently
    // producing a tracking-free export.
    #[cfg(feature = "autocam")]
    if autocam.enabled && autocam.model_path.is_empty() {
        return ExportOutcome::Failed(reco_io::stitch_job::StitchError::Other(
            "AI tracking enabled but no model selected".into(),
        ));
    }
    #[cfg(not(feature = "autocam"))]
    if autocam.enabled {
        return ExportOutcome::Failed(reco_io::stitch_job::StitchError::Other(
            "AI tracking is unavailable in this build".into(),
        ));
    }

    match job.run(interrupted) {
        Ok(r) => match r.completion {
            reco_core::session::types::RunCompletion::Completed => {
                ExportOutcome::Ok(r.frames_processed, output)
            }
            reco_core::session::types::RunCompletion::Cancelled
                if live && r.frames_processed > 0 =>
            {
                ExportOutcome::Ok(r.frames_processed, output)
            }
            reco_core::session::types::RunCompletion::Cancelled => ExportOutcome::Cancelled,
        },
        Err(
            reco_io::stitch_job::StitchError::Encoder(reco_core::sink::SinkError::Cancelled)
            | reco_io::stitch_job::StitchError::Session(
                reco_core::session::types::SessionError::Sink(
                    reco_core::sink::SinkError::Cancelled,
                ),
            ),
        ) => ExportOutcome::Cancelled,
        Err(e) => ExportOutcome::Failed(e),
    }
}

// Follow StitchJob's nearest-frame rounding for seek and duration independently.
fn estimated_export_frames(
    info: &reco_core::source::SourceInfo,
    start: f64,
    end: f64,
) -> Option<u64> {
    if !info.fps.is_finite() || info.fps <= 0.0 {
        return None;
    }
    let start = if start.is_finite() && start > 0.0 {
        start
    } else {
        0.0
    };
    let remaining = info
        .total_frames?
        .saturating_sub((start * info.fps).round() as u64);
    Some(if end.is_finite() && end > 0.0 {
        remaining.min(((end - start) * info.fps).round() as u64)
    } else {
        remaining
    })
}

/// Poll a worker without confusing a missing result with ongoing work.
pub fn poll_export(rx: &std::sync::mpsc::Receiver<ExportOutcome>) -> Option<ExportOutcome> {
    match rx.try_recv() {
        Ok(outcome) => Some(outcome),
        Err(std::sync::mpsc::TryRecvError::Empty) => None,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(ExportOutcome::Failed(
            reco_io::stitch_job::StitchError::Other(
                "Export worker stopped without a result".into(),
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_dialog_initial_codec() {
        assert_eq!(initial_codec(false, true, true, "h264"), "hevc");
        assert_eq!(initial_codec(true, true, true, "av1"), "h264");
        assert_eq!(initial_codec(false, false, true, "av1"), "av1");
        assert_eq!(initial_codec(false, true, false, "av1"), "av1");
    }

    #[test]
    #[cfg(feature = "autocam")]
    fn missing_model_fails_before_opening_inputs_or_touching_output() {
        use reco_core::calibration::{Framing, Lens};
        use reco_core::projection::Cylinder;

        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("missing-model.onnx");
        for existing in [false, true] {
            let output = dir
                .path()
                .join(if existing { "existing.mp4" } else { "new.mp4" });
            if existing {
                std::fs::write(&output, b"keep existing output").unwrap();
            }
            let monitor = Arc::new(OnceLock::new());
            let result = run_export(
                ExportRequest {
                    inputs: Inputs::Files(vec![reco_io::InputPath::Single(
                        dir.path().join("unopened-input.mp4"),
                    )]),
                    calibration_path: None,
                    calibration: Calibration::new(
                        vec![Lens::flat(64, 36)],
                        Cylinder::default(),
                        Framing {
                            axis_offset: 0.0,
                            tilt: 0.0,
                            roll: 0.0,
                        },
                    ),
                    output: output.clone(),
                    stream_url: None,
                    audio: reco_io::output::AudioMode::default(),
                    replay_enabled: false,
                    events_path: None,
                    width: 64,
                    height: 36,
                    codec: "h264".into(),
                    quality: "fast".into(),
                    start_secs: 0.0,
                    end_secs: 0.0,
                    autocam: AutocamUiConfig {
                        enabled: true,
                        model_path: model.to_string_lossy().into_owned(),
                        tracking_mode: "field".into(),
                        ..Default::default()
                    },
                },
                monitor.clone(),
                slint::Weak::default(),
                &Arc::new(AtomicBool::new(false)),
            );
            let ExportOutcome::Failed(reco_io::stitch_job::StitchError::Other(error)) = result
            else {
                panic!("Expected model preflight failure, got {result:?}");
            };
            assert!(error.contains("AI model path does not exist"), "{error}");
            assert!(error.contains(model.to_str().unwrap()), "{error}");
            assert!(monitor.get().is_none());
            if existing {
                assert_eq!(std::fs::read(&output).unwrap(), b"keep existing output");
            } else {
                assert!(!output.exists());
            }
        }
    }

    #[test]
    fn publishing_validation_keeps_urls_out_of_errors() {
        assert!(validate_stream(None, "hevc").is_ok());
        assert!(validate_stream(Some("rtmps://127.0.0.1/live/key"), "h264").is_ok());
        for url in [
            "",
            "https://private-key.invalid/",
            "rtmp://",
            "rtmp://private-key.invalid/\n",
        ] {
            let error = validate_stream(Some(url), "h264").unwrap_err();
            assert!(!error.contains("private-key"));
        }
        assert!(validate_stream(Some("rtmp://127.0.0.1/live/key"), "hevc").is_err());
        assert!(!stream_status(&StreamStatus::Connected).1);
        assert!(stream_status(&StreamStatus::Failed("disconnected".into())).1);
        assert!(
            stream_status(&StreamStatus::Connected)
                .0
                .contains("check playback")
        );
    }

    #[test]
    fn export_estimate_uses_aligned_source_count_and_rounded_trim() {
        let mut info = reco_core::source::SourceInfo {
            camera_sizes: vec![(128, 96); 2],
            fps: 30000.0 / 1001.0,
            fps_rational: Some((30000, 1001)),
            total_frames: Some(48),
        };
        for (start, end, expected) in [
            (0.0, 0.0, 48),
            (0.051, 0.149, 3),
            (0.051, 99.0, 46),
            (0.051, 0.0, 46),
            (99.0, 0.0, 0),
            (0.2, 0.1, 0),
        ] {
            assert_eq!(estimated_export_frames(&info, start, end), Some(expected));
        }
        info.total_frames = None;
        assert_eq!(estimated_export_frames(&info, 0.0, 1.0), None);
        info.total_frames = Some(48);
        info.fps = f64::NAN;
        assert_eq!(estimated_export_frames(&info, 0.0, 1.0), None);
    }

    #[test]
    #[ignore = "requires a GPU and RECO_EXPORT_TEST_VIDEO pointing to a readable video"]
    fn export_worker_cancels_optional_handshake_and_keeps_recording() {
        exercise_optional_stream(true);
    }

    #[test]
    #[ignore = "requires a GPU, FFmpeg 8+ and RECO_EXPORT_TEST_VIDEO with at least two seconds of video"]
    fn export_worker_keeps_recording_when_optional_peer_fails() {
        exercise_optional_stream(false);
    }

    fn exercise_optional_stream(cancel_output: bool) {
        use reco_core::calibration::{Framing, Lens};
        use reco_core::projection::Cylinder;
        use std::io::Read;
        use std::net::TcpListener;
        use std::sync::atomic::Ordering;
        use std::time::Duration;

        let input = PathBuf::from(std::env::var("RECO_EXPORT_TEST_VIDEO").unwrap());
        let source = reco_io::ffmpeg::decoder::VideoDecoder::open_software(&input).unwrap();
        let expected_frames = (source.fps().unwrap() * 2.0).round() as u64;
        let cal = Calibration::new(
            vec![Lens::flat(source.width(), source.height())],
            Cylinder {
                surface_height: Some(2400.0),
                ..Default::default()
            },
            Framing {
                axis_offset: 0.0,
                tilt: 0.0,
                roll: 0.0,
            },
        );
        drop(source);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("rtmp://{}/test/cancel", listener.local_addr().unwrap());
        let flag = Arc::new(AtomicBool::new(false));
        let cancel = flag.clone();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let peer = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(30);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if !matches!(
                            release_rx.try_recv(),
                            Err(std::sync::mpsc::TryRecvError::Empty)
                        ) {
                            return Duration::MAX;
                        }
                        assert!(Instant::now() < deadline, "export never connected");
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            };
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            assert!(socket.read(&mut [0; 4096]).unwrap() > 0);
            if !cancel_output {
                return Duration::ZERO;
            }
            let cancelled_at = Instant::now();
            cancel.store(true, Ordering::Relaxed);
            // Keep the unanswered connection open until the export reports cancellation.
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            cancelled_at.elapsed()
        });
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("recording.mp4");
        let monitor = Arc::new(OnceLock::new());
        let outcome = run_export(
            ExportRequest {
                inputs: Inputs::Files(vec![reco_io::InputPath::Single(input)]),
                calibration: cal,
                calibration_path: None,
                output: output.clone(),
                stream_url: Some(url),
                audio: reco_io::output::AudioMode::default(),
                replay_enabled: false,
                events_path: None,
                width: 64,
                height: 36,
                codec: "h264".into(),
                quality: "fast".into(),
                start_secs: 0.0,
                end_secs: 2.0,
                autocam: AutocamUiConfig {
                    enabled: false,
                    model_path: String::new(),
                    tracking_mode: "field".into(),
                    detection_interval: 1,
                    lookahead_secs: 0.0,
                    preset: String::new(),
                    framing: "action".into(),
                    lock_pitch: false,
                    cluster_mode: "density".into(),
                    cluster_bandwidth_rad: 0.0,
                    dead_zone_rad: 0.0,
                    ball_weight: 0.0,
                    fov_tight: 30.0,
                    fov_wide: 90.0,
                    fov_default: 60.0,
                    frame_all_margin_deg: 0.0,
                },
            },
            monitor.clone(),
            slint::Weak::default(),
            &flag,
        );
        let _ = release_tx.send(());
        let cancel_latency = peer.join().unwrap();
        assert!(
            matches!(outcome, ExportOutcome::Cancelled | ExportOutcome::Ok(..)),
            "{outcome:?}"
        );
        if cancel_output {
            assert_eq!(monitor.get().unwrap().status(), StreamStatus::Cancelled);
            assert!(cancel_latency < Duration::from_secs(2));
            assert!(std::fs::metadata(&output).unwrap().len() > 0);
        } else {
            let ExportOutcome::Ok(frames, path) = outcome else {
                panic!("recording failed");
            };
            assert_eq!(path, output);
            assert_eq!(frames, expected_frames);
            assert!(matches!(
                monitor.get().unwrap().status(),
                StreamStatus::Failed(_)
            ));
            let mut recording =
                reco_io::ffmpeg::decoder::VideoDecoder::open_software(&output).unwrap();
            for _ in 0..expected_frames {
                assert!(recording.next_frame().unwrap().is_some());
            }
            assert!(recording.next_frame().unwrap().is_none());
        }
    }

    #[test]
    fn worker_poll_distinguishes_pending_result_and_disconnection() {
        let (tx, rx) = std::sync::mpsc::channel();
        assert!(poll_export(&rx).is_none());
        tx.send(ExportOutcome::Cancelled).unwrap();
        drop(tx);
        assert!(matches!(poll_export(&rx), Some(ExportOutcome::Cancelled)));
        assert!(matches!(poll_export(&rx), Some(ExportOutcome::Failed(_))));
    }
}
