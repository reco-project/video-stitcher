//! Live camera stitching via libcamera (rpicam-vid).
//!
//! Captures stereo camera feeds from Raspberry Pi CSI cameras using
//! `rpicam-vid`, stitches them on the GPU, and encodes the panoramic
//! output to a video file in real time.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::source::FrameSource;

use crate::helpers;

/// Run live camera stitching via libcamera.
///
/// Captures frames from two RPi CSI cameras via `rpicam-vid`, stitches
/// them into a panoramic view on the GPU, and encodes the result.
/// Always uses YUV420P (I420) format from the RPi ISP.
/// Configuration for libcamera live stitching.
pub struct LibcameraRunConfig<'a> {
    pub cam_config: reco_io::libcamera::LibcameraConfig,
    pub calibration: &'a str,
    pub output: &'a str,
    pub width: u32,
    pub height: u32,
    pub blend: f32,
    pub encoder_name: Option<String>,
    pub codec: &'a str,
    pub quality: &'a str,
    pub duration: Option<f64>,
    pub max_frames: Option<u64>,
    pub capture_fps: u32,
    pub model_path: Option<&'a str>,
    pub detection_interval: u64,
    pub crf: Option<u8>,
    pub preset: Option<String>,
}

pub fn run_libcamera(
    config: LibcameraRunConfig<'_>,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let LibcameraRunConfig {
        cam_config,
        calibration,
        output,
        width,
        height,
        blend,
        encoder_name,
        codec,
        quality,
        duration,
        max_frames,
        capture_fps,
        model_path,
        detection_interval,
        crf,
        preset,
    } = config;
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

    let gpu = reco_core::gpu::GpuContext::new_blocking()?;

    let capture_width = cam_config.width;
    let capture_height = cam_config.height;

    let session_config = reco_core::session::SessionConfig {
        calibration: cal,
        viewport,
        input_width: capture_width,
        input_height: capture_height,
        output_format: reco_core::gpu::OutputFormat::Rgba8Unorm,
        input_format: reco_core::renderer::InputFormat::Yuv420p,
        left_rotation: 0,
        right_rotation: 0,
    };
    let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

    // Set up autocam (detector + director) if model provided.
    #[cfg(feature = "autocam")]
    if let Some(model) = model_path {
        let autocam_config =
            reco_autocam::AutocamConfig::new(model).with_detection_interval(detection_interval);
        match reco_autocam::setup_autocam(&mut session, &autocam_config, capture_fps as f32) {
            Ok(true) => println!("Autocam: YOLO ball tracking enabled (model: {model})"),
            Ok(false) => eprintln!("Warning: ball tracking unavailable in current capture mode"),
            Err(e) => eprintln!("Warning: autocam setup failed ({e}), continuing without tracking"),
        }
    }
    #[cfg(not(feature = "autocam"))]
    if model_path.is_some() {
        log::warn!("--model specified but autocam feature is disabled");
    }

    println!(
        "Pipeline ready: GPU = {}, capture = {}x{}@{}fps (YUV420P, libcamera), output = {}x{}",
        session.gpu_name(),
        capture_width,
        capture_height,
        capture_fps,
        width,
        height
    );

    reco_io::init();
    let quality = match quality {
        "fast" => reco_io::ffmpeg::encoder::Quality::Fast,
        "balanced" => reco_io::ffmpeg::encoder::Quality::Balanced,
        "high" => reco_io::ffmpeg::encoder::Quality::High,
        other => {
            log::warn!("Unknown quality '{other}', defaulting to balanced");
            reco_io::ffmpeg::encoder::Quality::Balanced
        }
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
        crf,
        preset,
        ..Default::default()
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

    let frame_limit =
        reco_core::session::compute_frame_limit(duration, max_frames, capture_fps as f64);

    if frame_limit < u64::MAX {
        println!("Capturing up to {frame_limit} frames");
    }

    let mut source = reco_io::libcamera::LibcameraCameraSource::open(&cam_config)?;

    let start = std::time::Instant::now();
    let mut frame_count: u64 = 0;

    // Warm up: discard first frame (camera ISP init)
    if let Some(frame) = source.next_frame()? {
        session.detect_and_update_director(&frame, start.elapsed());
        let pos = session.director_position();
        session.process_frame(&frame, pos.yaw, pos.pitch)?;
        println!("Warmup complete, starting capture...");
    }

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

    // Drop source first to kill rpicam-vid processes
    drop(source);
    session.finish()?;

    progress.finish(frame_count, output);

    Ok(())
}
