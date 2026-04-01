//! Live camera stitching via GStreamer.
//!
//! Captures stereo camera feeds, stitches them on the GPU, and encodes
//! the panoramic output to a video file in real time. Supports optional
//! YOLO ball detection and auto-tracking via the director pipeline.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::source::StereoFrame;
use reco_io::gstreamer::camera::CameraConfig;

use crate::helpers;

/// Run live camera stitching.
///
/// Captures frames from two cameras via GStreamer, stitches them into a
/// panoramic view on the GPU, and encodes the result to `output`. Uses
/// NV12 native capture on Jetson (Tegra) and I420 elsewhere.
///
/// When `model_path` is provided, sets up YOLO ball detection with
/// EKF tracking and a ball-following director for automatic panning.
#[allow(clippy::too_many_arguments)]
pub fn run_camera(
    cam_config: CameraConfig,
    calibration: &str,
    output: &str,
    width: u32,
    height: u32,
    blend: f32,
    encoder_name: Option<String>,
    codec: &str,
    quality: &str,
    duration: Option<f64>,
    max_frames: Option<u64>,
    capture_fps: u32,
    model_path: Option<&str>,
    detection_interval: u64,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    // Reject FFmpeg network URLs as output to prevent data exfiltration (#64).
    anyhow::ensure!(
        !output.contains("://"),
        "Output path looks like a network URL ({output}). Only local file paths are supported.",
    );

    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(calibration))?;

    let viewport = reco_core::viewport::ViewportConfig {
        width,
        height,
        blend_width: blend,
        ..Default::default()
    };

    let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;

    // Use NV12 capture on Jetson to skip the NV12->I420 conversion
    // in nvvidconv. The NVIDIA ISP natively outputs NV12.
    let use_nv12_capture = helpers::is_tegra();
    let input_format = if use_nv12_capture {
        reco_core::renderer::InputFormat::Nv12
    } else {
        reco_core::renderer::InputFormat::Yuv420p
    };

    let capture_width = cam_config.width;
    let capture_height = cam_config.height;

    let session_config = reco_core::session::SessionConfig {
        calibration: cal,
        viewport,
        input_width: capture_width,
        input_height: capture_height,
        output_format: reco_core::gpu::OutputFormat::Rgba8Unorm,
        input_format,
    };
    let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

    // Set up autocam (detector + tracker + director) if model provided.
    if let Some(model) = model_path {
        let detector = reco_autocam::YoloDetector::from_file(model)?;
        session.set_detector(Box::new(detector));
        session.set_director(Box::new(reco_autocam::BallDirector::new(
            capture_fps as f32,
        )));
        if detection_interval > 1 {
            session.set_detection_interval(detection_interval);
        }
        println!("Autocam: YOLO ball tracking enabled (model: {model})");
    }

    let mode_str = if use_nv12_capture { "NV12" } else { "I420" };
    println!(
        "Pipeline ready: GPU = {}, capture = {}x{}@{}fps ({}), output = {}x{}",
        session.gpu_name(),
        capture_width,
        capture_height,
        capture_fps,
        mode_str,
        width,
        height
    );

    reco_io::init();
    let quality = match quality {
        "fast" => reco_io::ffmpeg::encoder::Quality::Fast,
        "high" => reco_io::ffmpeg::encoder::Quality::High,
        _ => reco_io::ffmpeg::encoder::Quality::Balanced,
    };
    let video_codec =
        reco_io::ffmpeg::encoder::VideoCodec::from_str_loose(codec).unwrap_or_else(|| {
            eprintln!("Unknown codec '{codec}', defaulting to H.264");
            reco_io::ffmpeg::encoder::VideoCodec::H264
        });
    let enc_config = reco_io::ffmpeg::encoder::EncoderConfig {
        encoder_name,
        codec: video_codec,
        quality,
    };

    let encoder = reco_io::adapters::FfmpegFileEncoder::new(
        Path::new(output),
        width,
        height,
        (capture_fps as i32, 1),
        &enc_config,
    )?;
    println!("Encoder: {}", encoder.encoder_name());

    session.set_encoder(Box::new(encoder), 2);

    let capture_fps_f64 = capture_fps as f64;
    let frame_limit: u64 = match (duration, max_frames) {
        (Some(dur), Some(mf)) => ((dur * capture_fps_f64) as u64).min(mf),
        (Some(dur), None) => (dur * capture_fps_f64) as u64,
        (None, Some(mf)) => mf,
        (None, None) => u64::MAX,
    };

    if frame_limit < u64::MAX {
        println!("Capturing up to {frame_limit} frames");
    }

    let start = std::time::Instant::now();
    let mut frame_count: u64 = 0;

    if use_nv12_capture {
        // NV12 path: skip nvvidconv format conversion, upload 2 planes
        let mut source = reco_io::gstreamer::camera::GstreamerNv12CameraSource::open(&cam_config)?;

        // Warm up: discard first frame (camera ISP + pipeline init)
        if let Some(pair) = source.next_pair()? {
            let stereo = StereoFrame::Nv12(pair);
            session.detect_and_update_director(&stereo, start.elapsed());
            let pos = session.director_position();
            session.process_frame(&stereo, pos.yaw, pos.pitch)?;
            println!("Warmup complete, starting capture...");
        }

        let progress = helpers::ProgressReporter::new(30);

        while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let pair = {
                reco_core::profile_scope!("wait_capture");
                match source.next_pair()? {
                    Some(p) => p,
                    None => break,
                }
            };

            let stereo = StereoFrame::Nv12(pair);
            session.detect_and_update_director(&stereo, start.elapsed());
            let pos = session.director_position();
            session.process_frame(&stereo, pos.yaw, pos.pitch)?;
            frame_count += 1;
            progress.report(frame_count);
        }

        // Stop cameras gracefully before finishing encoder
        source.stop();
        session.finish()?;

        progress.finish(frame_count, output);

        // Drop source explicitly to allow graceful GStreamer/Argus teardown
        drop(source);
    } else {
        // I420 path: standard YUV420P upload with 3 planes
        use reco_core::source::FrameSource;
        let mut source = reco_io::gstreamer::camera::GstreamerCameraSource::open(&cam_config)?;

        let progress = helpers::ProgressReporter::new(30);

        while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let frame = {
                reco_core::profile_scope!("wait_capture");
                match source.next_frame()? {
                    Some(f) => f,
                    None => break,
                }
            };

            session.detect_and_update_director(&frame, start.elapsed());
            let pos = session.director_position();
            session.process_frame(&frame, pos.yaw, pos.pitch)?;
            frame_count += 1;
            progress.report(frame_count);
        }

        session.finish()?;

        progress.finish(frame_count, output);
    }

    Ok(())
}
