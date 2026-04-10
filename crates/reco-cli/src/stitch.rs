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

    // Load calibration
    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(calibration))?;
    let effective_sync = if sync_offset != 0 {
        sync_offset
    } else {
        cal.sync_offset
    };

    // Probe input
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

    // Detect zero-copy capability
    let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
    let use_zero_copy = reco_io::adapters::detect_zero_copy(source.as_ref().unwrap(), &gpu);

    let input_format = if use_zero_copy {
        reco_core::renderer::InputFormat::Nv12
    } else {
        reco_core::renderer::InputFormat::Yuv420p
    };

    // Build session
    let viewport = reco_core::viewport::ViewportConfig {
        width,
        height,
        blend_width: blend,
        rig_tilt: cal.rig_tilt as f32,
        ..Default::default()
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

    println!(
        "Pipeline ready: GPU = {}, output = {width}x{height}, mode = {}",
        session.gpu_name(),
        zero_copy_mode_label(use_zero_copy),
    );

    // Set up encoder
    let (encoder, enc_name) = reco_io::adapters::create_encoder(
        Path::new(output),
        width,
        height,
        fps_rational,
        codec,
        quality,
        encoder_name,
    )?;
    println!("Encoder: {enc_name}");
    session.set_encoder(Box::new(encoder), 2);

    // Set up autocam if model provided
    if let Some(model) = model_path {
        match reco_autocam::setup_autocam(
            &mut session,
            model,
            input_width,
            input_height,
            fps_val as f32,
            use_zero_copy,
            detection_interval,
            lead_time,
        ) {
            Ok(active) => {
                if active {
                    println!("Autocam: ball tracking enabled (model: {model})");
                }
            }
            Err(e) => {
                eprintln!("Warning: autocam setup failed ({e}), continuing without tracking");
            }
        }
    }

    // Compute frame limit
    let frame_limit = reco_core::session::compute_frame_limit(duration, max_frames, fps_val);
    if frame_limit < u64::MAX {
        println!("Processing up to {frame_limit} frames");
    }

    // Run the appropriate pipeline
    let progress = crate::helpers::ProgressReporter::new(30);
    #[allow(clippy::needless_late_init)] // cfg-gated branches each assign independently
    let frame_count;

    #[cfg(target_os = "linux")]
    if use_zero_copy {
        source.take();
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

/// Human-readable label for the active pipeline mode.
fn zero_copy_mode_label(use_zero_copy: bool) -> &'static str {
    if use_zero_copy {
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
    }
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
