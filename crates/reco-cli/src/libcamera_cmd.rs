//! Live camera stitching via libcamera (rpicam-vid).
//!
//! Captures stereo camera feeds from Raspberry Pi CSI cameras using
//! `rpicam-vid`, stitches them on the GPU, and encodes the panoramic
//! output to a video file in real time.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::render::viewport::ViewportSize;
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
    pub blend: Option<f32>,
    pub encoder_name: Option<String>,
    /// Device identifier for the named encoder.
    pub encoder_device: Option<String>,
    pub codec: Option<&'a str>,
    pub quality: &'a str,
    pub end_time: Option<f64>,
    pub max_frames: Option<u64>,
    pub capture_fps: u32,
    pub model_path: Option<&'a str>,
    pub detection_interval: u64,
    pub quality_value: Option<u8>,
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
        encoder_device,
        codec,
        quality,
        end_time,
        max_frames,
        capture_fps,
        model_path,
        detection_interval,
        quality_value,
        preset,
    } = config;
    anyhow::ensure!(
        !output.contains("://"),
        "Output path looks like a network URL ({output}). Only local file paths are supported.",
    );

    let mut cal = reco_core::calibration::Calibration::from_file(Path::new(calibration))?;
    if let Some(b) = blend {
        match cal.projection.l_shape_mut() {
            Some(t) => {
                eprintln!(
                    "Seam blend: --blend {b} overrides the calibration's {}",
                    t.blend_width
                );
                t.blend_width = b;
            }
            None => eprintln!("Seam blend: --blend {b} ignored (projection has no seam)"),
        }
    }

    let viewport_size = ViewportSize { width, height };

    let gpu = reco_core::gpu::GpuContext::new_blocking()?;

    let capture_width = cam_config.width;
    let capture_height = cam_config.height;

    let session_config = reco_core::session::types::SessionConfig {
        calibration: cal,
        viewport_size,
        source_sizes: vec![(capture_width, capture_height); 2],
        output_format: reco_core::gpu::OutputFormat::Rgba8Unorm,
        input_format: reco_core::render::renderer::InputFormat::Yuv420p,
        left_rotation: 0,
        right_rotation: 0,
    };
    let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

    // Set up autocam (detector + director) if model provided.
    #[cfg(feature = "autocam")]
    if let Some(model) = model_path {
        let autocam_config =
            reco_autocam::AutocamConfig::new(model).with_detection_interval(detection_interval);
        match reco_autocam::setup_autocam(&mut session, &autocam_config, capture_fps as f32, false)
        {
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
    let out_quality: reco_io::output::Quality = quality.parse().map_err(anyhow::Error::msg)?;
    let out_codec = codec
        .map(str::parse)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let enc_config = reco_io::ffmpeg::encoder::EncoderConfig {
        encoder_name,
        device: encoder_device,
        container: reco_io::output::Container::for_output(output),
        codec: out_codec,
        quality: quality_value.map_or(out_quality, reco_io::output::Quality::Value),
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

    session.add_sink(
        Box::new(encoder),
        reco_core::session::SinkOptions::threaded(2),
    )?;

    let frame_limit =
        reco_core::session::types::compute_frame_limit(end_time, max_frames, capture_fps as f64);

    if frame_limit < u64::MAX {
        println!("Capturing up to {frame_limit} frames");
    }

    let (progress, frame_count) = helpers::with_session_finalization(&mut session, |session| {
        let mut source =
            reco_io::libcamera::LibcameraCameraSource::open(&cam_config, interrupted.clone())?;

        let start = std::time::Instant::now();
        let mut frame_count: u64 = 0;

        // Warm up: discard first frame (camera ISP init)
        if let Some(frame) = source.next_frame()? {
            session.detect_and_update_world_pose(
                &frame.images,
                frame.capture.as_ref(),
                1.0 / capture_fps as f32,
                start.elapsed(),
            )?;
            let pos = session.advance_render_pose(1.0 / capture_fps as f32)?;
            session.render_frame(&frame.images, pos, frame.pts_us)?;
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

            session.detect_and_update_world_pose(
                &frame.images,
                frame.capture.as_ref(),
                1.0 / capture_fps as f32,
                start.elapsed(),
            )?;
            let pos = session.advance_render_pose(1.0 / capture_fps as f32)?;
            session.render_frame(&frame.images, pos, frame.pts_us)?;
            frame_count += 1;
            progress.report(frame_count);
        }

        // Stop capture before finalizing the encoder.
        drop(source);
        Ok((progress, frame_count))
    })?;

    progress.finish(frame_count, output);

    Ok(())
}
