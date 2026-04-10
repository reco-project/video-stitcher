//! Stitch subcommand: encode two video files into a panoramic output.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use reco_core::source::FrameSource;

/// Run the stitch subcommand: decode two video files and encode a stitched panorama.
#[allow(clippy::too_many_arguments)]
pub fn run_stitch(
    left: &str,
    right: &str,
    calibration: &str,
    output: &str,
    width: u32,
    height: u32,
    blend: f32,
    duration: Option<f64>,
    max_frames: Option<u64>,
    encoder_name: Option<String>,
    codec: &str,
    quality: &str,
    sync_offset: i64,
    model_path: Option<&str>,
    detection_interval: u64,
    lead_time: f64,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    const MAX_DIM: u32 = 8192;
    anyhow::ensure!(
        width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
        "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
        width,
        height,
    );
    anyhow::ensure!(
        !output.contains("://"),
        "Output path looks like a network URL ({output}). Only local file paths are supported.",
    );

    log::info!("Stitching: {left} + {right} -> {output}");

    // Load calibration first so we can use its sync_offset and rig_tilt
    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(calibration))?;

    // Use calibration's sync offset unless the user explicitly overrode it
    let effective_sync = if sync_offset != 0 {
        sync_offset
    } else {
        cal.sync_offset
    };

    // Probe input and create source (kept for CPU path, dropped for zero-copy)
    let mut source = Some(reco_io::adapters::FfmpegFileSource::open_with_offset(
        Path::new(left),
        Path::new(right),
        effective_sync,
    )?);
    let fps_rational = reco_io::adapters::FfmpegFileSource::frame_rate(Path::new(left))?;
    let info = source.as_ref().unwrap().info();
    let (input_width, input_height, fps_val) = (info.width, info.height, info.fps);
    log::info!("Input: {input_width}x{input_height} @ {fps_val:.1} fps");
    if effective_sync != 0 {
        println!(
            "Sync offset: {effective_sync} frames (from {})",
            if sync_offset != 0 {
                "CLI"
            } else {
                "calibration"
            }
        );
    }

    let viewport = reco_core::viewport::ViewportConfig {
        width,
        height,
        blend_width: blend,
        rig_tilt: cal.rig_tilt as f32,
        ..Default::default()
    };

    // Detect zero-copy capability
    let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
    let use_zero_copy = std::env::var("RECO_NO_HWACCEL").is_err()
        && source.as_ref().unwrap().supports_zero_copy()
        && gpu.supports_zero_copy();

    let input_format = if use_zero_copy {
        reco_core::renderer::InputFormat::Nv12
    } else {
        reco_core::renderer::InputFormat::Yuv420p
    };

    let session_config = reco_core::session::SessionConfig {
        calibration: cal,
        viewport,
        input_width,
        input_height,
        output_format: reco_core::gpu::OutputFormat::Rgba8Unorm,
        input_format,
    };
    let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

    let mode_str = if use_zero_copy {
        #[cfg(target_os = "linux")]
        {
            "zero-copy (CUDA/Vulkan)"
        }
        #[cfg(target_os = "macos")]
        {
            "zero-copy (VideoToolbox/Metal)"
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            "zero-copy"
        }
    } else {
        "CPU upload"
    };
    println!(
        "Pipeline ready: GPU = {}, output = {width}x{height}, mode = {mode_str}",
        session.gpu_name(),
    );

    // Set up encoder
    let quality_enum = match quality {
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
        quality: quality_enum,
    };
    let encoder = reco_io::adapters::FfmpegFileEncoder::new(
        Path::new(output),
        width,
        height,
        fps_rational,
        &enc_config,
    )?;
    println!("Encoder: {}", encoder.encoder_name());
    session.set_encoder(Box::new(encoder), 2);

    // Set up autocam (detector + director) if model provided
    if let Some(model) = model_path {
        let mut detection_active = false;

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if use_zero_copy {
            // Zero-copy path: use GPU detector (NPP + CUDA kernel + TensorRT).
            match reco_autocam::GpuYoloDetector::try_new(
                model,
                input_width,
                input_height,
                0.10,
                Vec::new(),
            ) {
                Ok(Some(gpu_det)) => {
                    session.set_gpu_detector(Box::new(gpu_det));
                    detection_active = true;
                    println!("Autocam: GPU YOLO ball tracking enabled (model: {model})");
                }
                Ok(None) => {
                    eprintln!(
                        "Warning: NPP not available, ball tracking disabled in zero-copy mode"
                    );
                }
                Err(e) => {
                    eprintln!("Warning: GPU detector init failed ({e}), ball tracking disabled");
                }
            }
        }

        #[cfg(target_os = "macos")]
        if use_zero_copy {
            // Zero-copy path: use Metal detector (compute shaders + CoreML).
            match reco_autocam::MetalYoloDetector::try_new(
                model,
                session.gpu(),
                input_width,
                input_height,
                0.10,
                Vec::new(),
            ) {
                Ok(metal_det) => {
                    session.set_metal_detector(Box::new(metal_det));
                    detection_active = true;
                    println!("Autocam: Metal YOLO ball tracking enabled (model: {model})");
                }
                Err(e) => {
                    eprintln!("Warning: Metal detector init failed ({e}), ball tracking disabled");
                }
            }
        }

        if !use_zero_copy {
            // CPU upload path: use CPU detector.
            let detector = reco_autocam::YoloDetector::from_file(model)?;
            session.set_detector(Box::new(detector));
            detection_active = true;
            println!("Autocam: YOLO ball tracking enabled (model: {model})");
        }

        if detection_active {
            let director = reco_autocam::BallDirector::new(fps_val as f32);
            session.set_director(Box::new(director));
            if detection_interval > 1 {
                session.set_detection_interval(detection_interval);
                println!("Detection interval: every {detection_interval} frames");
            }
            if lead_time > 0.0 && !use_zero_copy {
                let lookahead = (fps_val * lead_time).round() as usize;
                if lookahead > 0 {
                    session.set_lookahead(lookahead);
                    println!("Director lead time: {lead_time:.1}s ({lookahead} frames)");
                }
            }
        }
    }

    // Compute frame limit
    let frame_limit: u64 = match (duration, max_frames) {
        (Some(dur), Some(mf)) => ((dur * fps_val) as u64).min(mf),
        (Some(dur), None) => (dur * fps_val) as u64,
        (None, Some(mf)) => mf,
        (None, None) => u64::MAX,
    };
    if frame_limit < u64::MAX {
        println!("Processing up to {frame_limit} frames");
    }

    let progress = crate::helpers::ProgressReporter::new(30);
    #[allow(clippy::needless_late_init)] // cfg-gated branches each assign independently
    let frame_count;

    // Run the appropriate pipeline
    #[cfg(target_os = "linux")]
    if use_zero_copy {
        source.take(); // Drop CPU source before zero-copy setup
        frame_count = run_zero_copy_linux(
            &mut session,
            left,
            right,
            input_width,
            input_height,
            frame_limit,
            sync_offset,
            interrupted,
            &progress,
        )?;
    } else {
        frame_count = run_cpu_path(
            &mut session,
            &mut source,
            frame_limit,
            interrupted,
            progress,
        )?;
    }

    #[cfg(target_os = "macos")]
    if use_zero_copy {
        source.take();
        frame_count = run_zero_copy_macos(
            &mut session,
            left,
            right,
            frame_limit,
            sync_offset,
            interrupted,
            &progress,
        )?;
    } else {
        frame_count = run_cpu_path(
            &mut session,
            &mut source,
            frame_limit,
            interrupted,
            progress,
        )?;
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        frame_count = run_cpu_path(
            &mut session,
            &mut source,
            frame_limit,
            interrupted,
            progress,
        )?;
    }

    session.finish()?;
    progress.finish(frame_count, output);
    Ok(())
}

/// Run the CPU upload path using `session.run()`.
fn run_cpu_path(
    session: &mut reco_core::session::StitchSession,
    source: &mut Option<reco_io::adapters::FfmpegFileSource>,
    frame_limit: u64,
    interrupted: &Arc<AtomicBool>,
    progress: crate::helpers::ProgressReporter,
) -> anyhow::Result<u64> {
    let source = source.as_mut().expect("source dropped in CPU path");
    let count = session.run(
        source,
        frame_limit,
        interrupted,
        Some(Box::new(move |p: &reco_core::session::FrameProgress| {
            progress.report(p.frames_completed);
        })),
    )?;
    Ok(count)
}

/// Set up and run the CUDA/Vulkan zero-copy pipeline.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_zero_copy_linux(
    session: &mut reco_core::session::StitchSession,
    left: &str,
    right: &str,
    input_width: u32,
    input_height: u32,
    frame_limit: u64,
    sync_offset: i64,
    interrupted: &Arc<AtomicBool>,
    progress: &crate::helpers::ProgressReporter,
) -> anyhow::Result<u64> {
    let mut shared = session.create_shared_textures(input_width, input_height)?;

    let decode_handles = reco_io::zero_copy::spawn_decode_threads_gpu(
        left.to_string(),
        right.to_string(),
        shared.left_buf.clone(),
        shared.right_buf.clone(),
        shared.left_slot_free_rx.take().expect("left slot rx"),
        shared.right_slot_free_rx.take().expect("right slot rx"),
        sync_offset,
    );

    println!("Zero-copy pipeline active: NVDEC -> cuMemcpy2D -> shared texture -> render");

    let progress = *progress;
    let count = session.run_zero_copy_linux(
        shared,
        decode_handles,
        frame_limit,
        interrupted,
        Some(Box::new(move |p: &reco_core::session::FrameProgress| {
            progress.report(p.frames_completed);
        })),
    )?;
    Ok(count)
}

/// Set up and run the VideoToolbox/Metal zero-copy pipeline.
#[cfg(target_os = "macos")]
fn run_zero_copy_macos(
    session: &mut reco_core::session::StitchSession,
    left: &str,
    right: &str,
    frame_limit: u64,
    sync_offset: i64,
    interrupted: &Arc<AtomicBool>,
    progress: &crate::helpers::ProgressReporter,
) -> anyhow::Result<u64> {
    let pair_rx = reco_io::zero_copy::spawn_vt_decode_pair(left, right, sync_offset);

    println!("Zero-copy pipeline active: VideoToolbox -> CVMetalTextureCache -> Metal render");

    let progress = *progress;
    let count = session.run_zero_copy_macos(
        pair_rx,
        frame_limit,
        interrupted,
        Some(Box::new(move |p: &reco_core::session::FrameProgress| {
            progress.report(p.frames_completed);
        })),
    )?;
    Ok(count)
}
