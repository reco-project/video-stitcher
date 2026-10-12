//! Live camera stitching via libcamera (rpicam-vid).
//!
//! Captures stereo camera feeds from Raspberry Pi CSI cameras using
//! `rpicam-vid`, stitches them on the GPU, and encodes the panoramic
//! output to a video file in real time.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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
        model_path,
        detection_interval,
        quality_value,
        preset,
    } = config;
    anyhow::ensure!(
        !output.contains("://"),
        "Output path looks like a network URL ({output}). Only local file paths are supported.",
    );

    let cal = reco_core::calibration::Calibration::from_file(Path::new(calibration))?;
    let out_quality = quality.parse().map_err(anyhow::Error::msg)?;
    let out_codec = codec
        .map(str::parse)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let source = reco_io::libcamera::LibcameraCameraSource::open(&cam_config, interrupted.clone())?;
    let progress = helpers::ProgressReporter::new(30);
    let mut job = reco_io::StitchJob::from_source(source, cal, output)
        .resolution(width, height)
        .quality(quality_value.map_or(out_quality, reco_io::output::Quality::Value))
        .on_progress(move |p| progress.report_with_elapsed(p.frames_completed, p.elapsed));
    if let Some(blend) = blend {
        job = job.blend_width(blend);
    }
    if let Some(codec) = out_codec {
        job = job.codec(codec);
    }
    if let Some(encoder) = encoder_name {
        job = job.encoder_name(encoder);
    }
    if let Some(device) = encoder_device {
        job = job.encoder_device(device);
    }
    if let Some(preset) = preset {
        job = job.preset(preset);
    }
    if let Some(end) = end_time {
        job = job.end_time(end);
    }
    if let Some(limit) = max_frames {
        job = job.max_frames(limit);
    }

    // Set up autocam (detector + director) if model provided.
    #[cfg(feature = "autocam")]
    if let Some(model) = model_path {
        let autocam_config =
            reco_autocam::AutocamConfig::new(model).with_detection_interval(detection_interval);
        job = job.on_session(move |session, source| {
            match reco_autocam::setup_autocam(
                session,
                &autocam_config,
                source.info().fps as f32,
                false,
            ) {
                Ok(true) => println!("Autocam: YOLO ball tracking enabled"),
                Ok(false) => {
                    eprintln!("Warning: ball tracking unavailable in current capture mode")
                }
                Err(e) => {
                    eprintln!("Warning: autocam setup failed ({e}), continuing without tracking")
                }
            }
            Ok(())
        });
    }
    #[cfg(not(feature = "autocam"))]
    {
        let _ = detection_interval;
        if model_path.is_some() {
            log::warn!("--model specified but autocam feature is disabled");
        }
    }

    println!(
        "Capture: {}x{}@{}fps (YUV420P, libcamera)",
        cam_config.width, cam_config.height, cam_config.fps
    );
    let result = job.run(interrupted)?;
    println!(
        "\nDone: {} frames in {:.1}s -> {output}",
        result.frames_processed,
        result.elapsed.as_secs_f64()
    );

    Ok(())
}
