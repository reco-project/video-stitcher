//! Export worker thread.
//!
//! Runs a [`StitchJob`](reco_io::StitchJob) on a background thread,
//! pumping progress back to the Slint UI via `invoke_from_event_loop`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use reco_core::calibration::Calibration;

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
/// tuning fields are only snapshotted and never read, so dead-code is allowed for
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
#[allow(clippy::too_many_arguments)]
pub fn run_export(
    inputs: Vec<reco_io::InputPath>,
    cal: Calibration,
    output: PathBuf,
    stream_url: Option<String>,
    replay_enabled: bool,
    events_path: Option<PathBuf>,
    width: u32,
    height: u32,
    codec_str: String,
    quality_str: String,
    start_secs: f32,
    end_secs: f32,
    autocam: AutocamUiConfig,
    app_weak: slint::Weak<RecoApp>,
    interrupted: &Arc<AtomicBool>,
) -> ExportOutcome {
    use reco_io::output::{Codec, Quality};

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

    post_status("Opening encoder and decoders...".into());

    let progress_weak = app_weak.clone();
    let progress_start = Instant::now();
    let effective_output = stream_url
        .as_ref()
        .filter(|u| !u.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| output.clone());
    let format = reco_io::output::Container::for_output(&effective_output.to_string_lossy());
    if format.is_streaming() {
        log::info!("Streaming to network output");
    }

    let mut job = reco_io::StitchJob::with_calibration(inputs, cal, effective_output.clone())
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
                    app.set_export_status_text(
                        format!("Frame {frames} ({fps:.0} fps){eta}").into(),
                    );
                }
            });
        });

    if start_secs > 0.0 {
        post_status(format!("Seeking to {start_secs:.0}s..."));
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
    fn export_worker_cancels_an_unanswered_network_handshake() {
        use reco_core::calibration::{Framing, Lens};
        use reco_core::projection::Cylinder;
        use std::io::Read;
        use std::net::TcpListener;
        use std::sync::atomic::Ordering;
        use std::time::Duration;

        let input = PathBuf::from(std::env::var("RECO_EXPORT_TEST_VIDEO").unwrap());
        let source = reco_io::ffmpeg::decoder::VideoDecoder::open_software(&input).unwrap();
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
            let cancelled_at = Instant::now();
            cancel.store(true, Ordering::Relaxed);
            // Keep the unanswered connection open until the export reports cancellation.
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            cancelled_at.elapsed()
        });
        let outcome = run_export(
            vec![reco_io::InputPath::Single(input)],
            cal,
            PathBuf::new(),
            Some(url),
            false,
            None,
            64,
            36,
            "h264".into(),
            "fast".into(),
            0.0,
            0.1,
            AutocamUiConfig {
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
            slint::Weak::default(),
            &flag,
        );
        let _ = release_tx.send(());
        let cancel_latency = peer.join().unwrap();
        assert!(matches!(outcome, ExportOutcome::Cancelled), "{outcome:?}");
        assert!(cancel_latency < Duration::from_secs(2));
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
