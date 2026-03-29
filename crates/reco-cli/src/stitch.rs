//! Stitch subcommand: encode two video files into a panoramic output.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::encoder::Encoder;
use reco_core::profile_scope;

// ---- GPU decode types for zero-copy path ----

/// CUDA buffer info passed to decode threads for cuMemcpy2D destination.
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Clone)]
struct GpuBufInfo {
    /// CUDA device pointers for double-buffered Y textures.
    y_ptr: [u64; 2],
    /// CUDA device pointers for double-buffered UV textures.
    uv_ptr: [u64; 2],
    /// Row pitch of shared Y textures (may differ from width due to alignment).
    y_pitch: [usize; 2],
    /// Row pitch of shared UV textures.
    uv_pitch: [usize; 2],
    width: u32,
    height: u32,
}

/// A pair of double-buffer slot indices from the decode threads.
#[cfg(any(target_os = "linux", target_os = "windows"))]
struct GpuFrameSignal {
    left_slot: u8,
    right_slot: u8,
}

/// Handles for the GPU decode threads, used for graceful shutdown.
#[cfg(any(target_os = "linux", target_os = "windows"))]
struct GpuDecodeHandles {
    frame_rx: std::sync::mpsc::Receiver<GpuFrameSignal>,
    /// Join handles for the 2 decode threads + 1 pairing thread.
    /// Must be joined before dropping shared textures to ensure FFmpeg's
    /// CUDA context cleanup completes while shared memory is still valid.
    join_handles: Vec<std::thread::JoinHandle<()>>,
}

/// Spawn a single-video GPU decode thread that writes NV12 frames directly
/// to CUDA/Vulkan shared textures via cuMemcpy2D.
///
/// Uses `slot_free_rx` for backpressure: the decode thread waits for a slot
/// to be released by the main thread before writing to it. This prevents
/// NVDEC from overwriting a slot that the GPU render pass is still reading.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn spawn_single_decoder_gpu(
    path: String,
    label: &'static str,
    buf: GpuBufInfo,
    slot_free_rx: std::sync::mpsc::Receiver<u8>,
) -> (std::sync::mpsc::Receiver<u8>, std::thread::JoinHandle<()>) {
    let (tx, rx) = std::sync::mpsc::sync_channel::<u8>(1);

    let handle = std::thread::Builder::new()
        .name(format!("decode_{label}_gpu"))
        .spawn(move || {
            let mut dec = match reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(&path)) {
                Ok(d) => {
                    log::info!(
                        "{label} GPU decoder: {} ({}x{})",
                        d.backend(),
                        d.width(),
                        d.height()
                    );
                    d
                }
                Err(e) => {
                    log::error!("Failed to open {label} video: {e}");
                    return;
                }
            };

            while let Ok(slot) = slot_free_rx.recv() {
                match dec.next_frame_gpu() {
                    Ok(Some(frame)) => {
                        let s = slot as usize;

                        // Ensure CUDA context is current (FFmpeg may have popped it)
                        if let Err(e) = reco_core::cuda_interop::cuda_ensure_context() {
                            log::error!("{label} cuda_ensure_context: {e}");
                            break;
                        }

                        // Copy Y plane: NVDEC -> shared texture
                        if let Err(e) = reco_core::cuda_interop::cuda_2d_copy(
                            buf.y_ptr[s],
                            buf.y_pitch[s],
                            frame.y_ptr,
                            frame.y_pitch,
                            buf.width as usize, // Y: 1 byte/pixel
                            buf.height as usize,
                        ) {
                            log::error!("{label} cuMemcpy2D Y: {e}");
                            break;
                        }

                        // Copy UV plane: NVDEC -> shared texture
                        // NV12 UV: width bytes per row, height/2 rows
                        if let Err(e) = reco_core::cuda_interop::cuda_2d_copy(
                            buf.uv_ptr[s],
                            buf.uv_pitch[s],
                            frame.uv_ptr,
                            frame.uv_pitch,
                            buf.width as usize, // UV: 2 bytes x width/2 = width
                            buf.height as usize / 2,
                        ) {
                            log::error!("{label} cuMemcpy2D UV: {e}");
                            break;
                        }

                        // Synchronize CUDA to ensure copies complete before GPU reads
                        if let Err(e) = reco_core::cuda_interop::cuda_synchronize() {
                            log::error!("{label} cuCtxSynchronize: {e}");
                            break;
                        }

                        if tx.send(slot).is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Ok(None) => {
                        log::error!("{label}: next_frame_gpu returned None (non-CUDA?)");
                        break;
                    }
                    Err(e) => {
                        log::error!("{label} decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn GPU decode thread");

    (rx, handle)
}

/// Spawn parallel GPU decode threads and a pairing thread.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn spawn_decode_thread_gpu(
    left_path: String,
    right_path: String,
    left_buf: GpuBufInfo,
    right_buf: GpuBufInfo,
    left_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    right_slot_free_rx: std::sync::mpsc::Receiver<u8>,
) -> GpuDecodeHandles {
    let (left_rx, left_handle) =
        spawn_single_decoder_gpu(left_path, "left", left_buf, left_slot_free_rx);
    let (right_rx, right_handle) =
        spawn_single_decoder_gpu(right_path, "right", right_buf, right_slot_free_rx);

    let (tx, rx) = std::sync::mpsc::sync_channel::<GpuFrameSignal>(1);

    let pair_handle = std::thread::Builder::new()
        .name("decode_pair_gpu".into())
        .spawn(move || {
            while let (Ok(left_slot), Ok(right_slot)) = (left_rx.recv(), right_rx.recv()) {
                if tx
                    .send(GpuFrameSignal {
                        left_slot,
                        right_slot,
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .expect("spawn GPU pairing thread");

    GpuDecodeHandles {
        frame_rx: rx,
        join_handles: vec![left_handle, right_handle, pair_handle],
    }
}

/// Run the stitch loop using CUDA/Vulkan zero-copy (no CPU upload).
///
/// Creates double-buffered shared textures, spawns GPU decode threads,
/// and renders directly from GPU-resident data.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_stitch_zero_copy(
    session: &mut reco_core::session::StitchSession,
    encoder: &mut dyn reco_core::encoder::Encoder,
    left_path: &str,
    right_path: &str,
    input_width: u32,
    input_height: u32,
    frame_limit: u64,
    yaw: f32,
    pitch: f32,
    interrupted: &Arc<AtomicBool>,
    progress: &crate::helpers::ProgressReporter,
) -> anyhow::Result<u64> {
    use reco_core::vulkan_interop::{Nv12Plane, create_nv12_shared_texture};

    // Create double-buffered shared textures for each camera (Y + UV per slot)
    log::info!("Creating shared textures for zero-copy...");

    let gpu = session.gpu();
    let create_pair = |label: &str,
                       slot: usize|
     -> anyhow::Result<(
        reco_core::vulkan_interop::SharedTexture,
        reco_core::vulkan_interop::SharedTexture,
    )> {
        let y = create_nv12_shared_texture(gpu, input_width, input_height, Nv12Plane::Y)
            .map_err(|e| anyhow::anyhow!("{label} Y[{slot}] shared texture: {e}"))?;

        let uv = create_nv12_shared_texture(gpu, input_width, input_height, Nv12Plane::Uv)
            .map_err(|e| anyhow::anyhow!("{label} UV[{slot}] shared texture: {e}"))?;

        Ok((y, uv))
    };

    let (left_y_0, left_uv_0) = create_pair("left", 0)?;
    let (left_y_1, left_uv_1) = create_pair("left", 1)?;
    let (right_y_0, right_uv_0) = create_pair("right", 0)?;
    let (right_y_1, right_uv_1) = create_pair("right", 1)?;

    log::info!(
        "Shared textures created: left Y pitch={}/{}, UV pitch={}/{}",
        left_y_0.pitch,
        left_y_1.pitch,
        left_uv_0.pitch,
        left_uv_1.pitch
    );

    // Build GPU buffer info for decode threads (just CUDA pointers + pitches)
    let left_buf = GpuBufInfo {
        y_ptr: [left_y_0.cuda_ptr, left_y_1.cuda_ptr],
        uv_ptr: [left_uv_0.cuda_ptr, left_uv_1.cuda_ptr],
        y_pitch: [left_y_0.pitch, left_y_1.pitch],
        uv_pitch: [left_uv_0.pitch, left_uv_1.pitch],
        width: input_width,
        height: input_height,
    };
    let right_buf = GpuBufInfo {
        y_ptr: [right_y_0.cuda_ptr, right_y_1.cuda_ptr],
        uv_ptr: [right_uv_0.cuda_ptr, right_uv_1.cuda_ptr],
        y_pitch: [right_y_0.pitch, right_y_1.pitch],
        uv_pitch: [right_uv_0.pitch, right_uv_1.pitch],
        width: input_width,
        height: input_height,
    };

    // Slot-free channels: decode threads wait for a slot to be released by
    // main before writing to it. This prevents NVDEC from overwriting a
    // shared texture that the GPU render pass is still reading.
    let (left_slot_free_tx, left_slot_free_rx) = std::sync::mpsc::sync_channel::<u8>(2);
    let (right_slot_free_tx, right_slot_free_rx) = std::sync::mpsc::sync_channel::<u8>(2);
    // Both slots start as free
    left_slot_free_tx.send(0).expect("seed slot channel");
    left_slot_free_tx.send(1).expect("seed slot channel");
    right_slot_free_tx.send(0).expect("seed slot channel");
    right_slot_free_tx.send(1).expect("seed slot channel");

    // Spawn GPU decode threads
    let decode = spawn_decode_thread_gpu(
        left_path.to_string(),
        right_path.to_string(),
        left_buf,
        right_buf,
        left_slot_free_rx,
        right_slot_free_rx,
    );
    let frame_rx = decode.frame_rx;

    println!("Zero-copy pipeline active: NVDEC -> cuMemcpy2D -> shared texture -> render");

    // Configure bind groups for GPU-resident shared textures
    let bind_groups = session.pipeline_mut().configure_gpu_source(
        [(&left_y_0, &left_uv_0), (&left_y_1, &left_uv_1)],
        [(&right_y_0, &right_uv_0), (&right_y_1, &right_uv_1)],
    );

    let mut frame_count: u64 = 0;

    loop {
        if frame_count >= frame_limit || interrupted.load(Ordering::Relaxed) {
            break;
        }

        let signal = {
            profile_scope!("wait_decode");
            match frame_rx.recv() {
                Ok(s) => s,
                Err(_) => break,
            }
        };

        let render_buf = session.pipeline_mut().render_gpu_frame(
            &bind_groups,
            signal.left_slot,
            signal.right_slot,
            yaw,
            pitch,
        );
        session.submit_render_output(render_buf, encoder)?;

        // GPU is done reading these slots - release them for decode to reuse.
        // poll(Wait) inside convert_and_readback guarantees the render pass
        // has finished reading from the shared textures.
        let _ = left_slot_free_tx.send(signal.left_slot);
        let _ = right_slot_free_tx.send(signal.right_slot);

        frame_count += 1;
        progress.report(frame_count);
    }

    // Graceful shutdown: correct ordering prevents CUDA error 700.
    //
    // 1. Drop slot-free senders -> decode threads' recv() returns Err -> threads exit
    // 2. Drop frame_rx -> pairing thread's send() returns Err -> thread exits
    // 3. Join all threads -> VideoDecoder::Drop completes FFmpeg CUDA cleanup
    //    (cuMemFree) while the shared CUDA VMM memory is still mapped
    // 4. Drop shared textures -> cuMemUnmap + cuMemAddressFree (safe now that
    //    FFmpeg is done with the CUDA context)
    drop(left_slot_free_tx);
    drop(right_slot_free_tx);
    drop(frame_rx);
    for handle in decode.join_handles {
        let _ = handle.join();
    }

    // Drop shared textures: wgpu texture (VkImage + VkDeviceMemory)
    // must be freed before the CUDA shared memory is unmapped.
    // SharedTexture field order guarantees this (texture before _shared_mem).
    drop(left_y_0);
    drop(left_uv_0);
    drop(left_y_1);
    drop(left_uv_1);
    drop(right_y_0);
    drop(right_uv_0);
    drop(right_y_1);
    drop(right_uv_1);
    Ok(frame_count)
}

/// Run the stitch loop using VideoToolbox/Metal zero-copy (macOS).
///
/// Each frame: VideoToolbox decode -> CVPixelBuffer -> CVMetalTextureCache ->
/// MTLTexture -> wgpu bind group -> render -> NV12 readback -> encode.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_stitch_metal_zero_copy(
    session: &mut reco_core::session::StitchSession,
    encoder: &mut dyn reco_core::encoder::Encoder,
    left_path: &str,
    right_path: &str,
    frame_limit: u64,
    yaw: f32,
    pitch: f32,
    interrupted: &Arc<AtomicBool>,
    progress: &crate::helpers::ProgressReporter,
) -> anyhow::Result<u64> {
    use reco_core::metal_interop::MetalTextureCache;
    use reco_io::ffmpeg::decoder::VideoDecoder;
    use std::path::Path;

    let mut left_dec = VideoDecoder::open(Path::new(left_path))?;
    let mut right_dec = VideoDecoder::open(Path::new(right_path))?;

    log::info!(
        "Metal zero-copy: left={} ({}), right={} ({})",
        left_dec.backend(),
        left_dec.backend(),
        right_dec.backend(),
        right_dec.backend(),
    );

    // Create the Metal texture cache (bridges CVPixelBuffer -> MTLTexture)
    let cache = MetalTextureCache::new(session.gpu())?;

    let mut frame_count: u64 = 0;

    while !interrupted.load(Ordering::Relaxed) && frame_count < frame_limit {
        // Decode one frame from each camera
        let left_vt = match left_dec.next_frame_vt()? {
            Some(f) => f,
            None => break,
        };
        let right_vt = match right_dec.next_frame_vt()? {
            Some(f) => f,
            None => break,
        };

        // Import NV12 planes as Metal textures (zero-copy via IOSurface)
        let (left_y, left_uv) = cache.import_nv12(left_vt.cv_pixel_buffer, session.gpu())?;
        let (right_y, right_uv) = cache.import_nv12(right_vt.cv_pixel_buffer, session.gpu())?;

        // Render using the imported textures
        let render_buf = session.pipeline_mut().render_imported_textures(
            &left_y.texture,
            &left_uv.texture,
            &right_y.texture,
            &right_uv.texture,
            yaw,
            pitch,
        );

        // Convert to NV12 and submit to encoder
        session.submit_render_output(render_buf, encoder)?;

        frame_count += 1;
        progress.report(frame_count);

        // ImportedPlaneTextures drop here, releasing CVMetalTextureRefs
    }

    Ok(frame_count)
}

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
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    use reco_core::source::FrameSource;

    const MAX_DIM: u32 = 8192;
    anyhow::ensure!(
        width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
        "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
        width,
        height,
    );

    // Reject FFmpeg network URLs as output to prevent data exfiltration (#64).
    anyhow::ensure!(
        !output.contains("://"),
        "Output path looks like a network URL ({output}). Only local file paths are supported.",
    );

    log::info!("Stitching: {left} + {right} -> {output}");

    // Probe input dimensions and fps. The source is kept for CPU upload path
    // but dropped early in the zero-copy path (its CUDA decoders would conflict
    // with shared texture cleanup).
    let mut source = Some(reco_io::adapters::FfmpegFileSource::open(
        Path::new(left),
        Path::new(right),
    )?);
    let fps_rational = reco_io::adapters::FfmpegFileSource::frame_rate(Path::new(left))?;
    let info = source.as_ref().unwrap().info();
    let input_width = info.width;
    let input_height = info.height;
    let fps_val = info.fps;

    log::info!(
        "Input: {}x{} @ {:.1} fps",
        input_width,
        input_height,
        fps_val
    );

    let cal = crate::helpers::load_calibration(Path::new(calibration))?;

    let viewport = reco_core::viewport::ViewportConfig {
        width,
        height,
        blend_width: blend,
        ..Default::default()
    };

    // Detect zero-copy capability: CUDA decode + CUDA interop + Vulkan backend + Linux.
    // On Jetson/Tegra, CUDA runtime is available but FFmpeg uses V4L2 NVDEC
    // (not CUVID), so zero-copy shared textures don't work.
    let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;

    #[cfg(target_os = "linux")]
    let use_zero_copy = std::env::var("RECO_NO_HWACCEL").is_err()
        && source.as_ref().unwrap().decode_backend()
            == reco_io::ffmpeg::decoder::DecodeBackend::Cuda
        && reco_core::cuda_interop::is_cuda_available()
        && gpu.is_vulkan();
    #[cfg(target_os = "macos")]
    let use_zero_copy = std::env::var("RECO_NO_HWACCEL").is_err()
        && source.as_ref().unwrap().decode_backend()
            == reco_io::ffmpeg::decoder::DecodeBackend::VideoToolbox
        && gpu.is_metal();
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let use_zero_copy = false;

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

    println!(
        "Pipeline ready: GPU = {}, output = {width}x{height}, mode = {}",
        session.gpu_name(),
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
    );

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

    let mut encoder = reco_io::adapters::FfmpegFileEncoder::new(
        Path::new(output),
        width,
        height,
        fps_rational,
        &enc_config,
    )?;
    println!("Encoder: {}", encoder.encoder_name());

    // Compute frame limit from --duration and --max-frames
    let frame_limit: u64 = match (duration, max_frames) {
        (Some(dur), Some(mf)) => {
            let dur_frames = (dur * fps_val) as u64;
            dur_frames.min(mf)
        }
        (Some(dur), None) => (dur * fps_val) as u64,
        (None, Some(mf)) => mf,
        (None, None) => u64::MAX,
    };

    if frame_limit < u64::MAX {
        println!("Processing up to {frame_limit} frames");
    }

    let progress = crate::helpers::ProgressReporter::new(30);
    let mut frame_count: u64 = 0;
    let yaw = 0.0_f32;
    let pitch = 0.0_f32;

    #[cfg(target_os = "linux")]
    if use_zero_copy {
        // Drop the CPU source - zero-copy spawns its own GPU decode threads.
        // Must drop before shared texture cleanup to avoid CUDA context conflicts
        // (the source's decoders hold CUDA resources from hardware probe).
        source.take();

        frame_count = run_stitch_zero_copy(
            &mut session,
            &mut encoder,
            left,
            right,
            input_width,
            input_height,
            frame_limit,
            yaw,
            pitch,
            interrupted,
            &progress,
        )?;
    }

    #[cfg(target_os = "macos")]
    if use_zero_copy {
        // Drop the CPU source - Metal zero-copy opens its own decoders.
        source.take();

        frame_count = run_stitch_metal_zero_copy(
            &mut session,
            &mut encoder,
            left,
            right,
            frame_limit,
            yaw,
            pitch,
            interrupted,
            &progress,
        )?;
    }

    if !use_zero_copy {
        // CPU upload path: use FfmpegFileSource for decoding
        let source = source.as_mut().expect("source dropped in CPU path");
        frame_count = session.run(
            source,
            &mut encoder,
            frame_limit,
            interrupted,
            Some(Box::new(move |p| {
                progress.report(p.frames_completed);
            })),
        )?;
    }

    log::info!("Finishing encoder...");
    encoder.finish()?;
    log::info!("Encoder finished");

    progress.finish(frame_count, output);

    log::info!("Dropping session...");
    drop(session);
    log::info!("Session dropped, exiting");

    Ok(())
}
