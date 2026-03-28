//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Create a tracing span guard (no-op when `profiling` feature is disabled).
#[cfg(feature = "profiling")]
macro_rules! profile_scope {
    ($name:expr) => {
        let _span = tracing::info_span!($name).entered();
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! profile_scope {
    ($name:expr) => {};
}

/// Initialize the tracing profiler. Returns a guard that must be held
/// until the end of `main()` — the trace file is written on drop.
#[cfg(feature = "profiling")]
fn init_profiling() -> tracing_chrome::FlushGuard {
    use tracing_subscriber::prelude::*;
    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file("reco-trace.json")
        .include_args(true)
        .build();
    tracing_subscriber::registry().with(chrome_layer).init();
    eprintln!("Profiling enabled — trace will be written to reco-trace.json");
    guard
}

#[derive(Parser)]
#[command(
    name = "reco",
    version,
    about = "GPU-accelerated panoramic video stitching",
    long_about = "Reco stitches two camera feeds into a seamless panoramic sports view.\n\
                  Designed for sports filming with open-source hardware flexibility."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Stitch two video files into a panoramic output.
    Stitch {
        /// Path to the left camera video file.
        left: String,

        /// Path to the right camera video file.
        right: String,

        /// Path to the calibration JSON file (v1-compatible match format).
        #[arg(short, long)]
        calibration: String,

        /// Output file path.
        #[arg(short, long, default_value = "output.mp4")]
        output: String,

        /// Output width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,

        /// Output height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,

        /// Maximum number of seconds to process.
        #[arg(long)]
        duration: Option<f64>,

        /// Maximum number of frames to process.
        #[arg(long)]
        max_frames: Option<u64>,

        /// Force a specific encoder (e.g., h264_nvenc, hevc_nvenc, libx264). Auto-detects by default.
        #[arg(long)]
        encoder: Option<String>,

        /// Output codec: h264, hevc, av1. Default: h264.
        #[arg(long, default_value = "h264")]
        codec: String,

        /// Quality preset: fast, balanced, high.
        #[arg(long, default_value = "balanced")]
        quality: String,
    },

    /// Open an interactive preview window to debug the stitch.
    Preview {
        /// Path to the left camera video file.
        left: String,

        /// Path to the right camera video file.
        right: String,

        /// Path to the calibration JSON file (v1-compatible match format).
        #[arg(short, long)]
        calibration: String,

        /// Window width in pixels.
        #[arg(long, default_value_t = 1280)]
        width: u32,

        /// Window height in pixels.
        #[arg(long, default_value_t = 720)]
        height: u32,
    },

    /// Display information about the GPU and system capabilities.
    Info,
}

fn main() -> anyhow::Result<()> {
    // When profiling, tracing-subscriber owns the global logger;
    // otherwise, use env_logger for RUST_LOG filtering.
    #[cfg(feature = "profiling")]
    let _profiling_guard = init_profiling();
    #[cfg(not(feature = "profiling"))]
    env_logger::init();

    // Set up Ctrl-C handler so stitch can finalize the output file
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = interrupted.clone();
        ctrlc::set_handler(move || {
            if interrupted.load(Ordering::Relaxed) {
                // Second Ctrl-C — force exit
                std::process::exit(1);
            }
            interrupted.store(true, Ordering::Relaxed);
            eprintln!("\nInterrupted — finishing output file...");
        })
        .expect("set Ctrl-C handler");
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Stitch {
            left,
            right,
            calibration,
            output,
            width,
            height,
            duration,
            max_frames,
            encoder,
            codec,
            quality,
        } => {
            log::info!("Stitching: {left} + {right} → {output}");

            // Open video decoders to get input dimensions and fps
            let left_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&left))?;
            let right_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&right))?;

            let input_width = left_dec.width();
            let input_height = left_dec.height();

            anyhow::ensure!(
                input_width == right_dec.width() && input_height == right_dec.height(),
                "Video dimension mismatch: left={}x{}, right={}x{}",
                input_width,
                input_height,
                right_dec.width(),
                right_dec.height()
            );

            log::info!(
                "Left video: {}x{} @ {:.1} fps",
                input_width,
                input_height,
                left_dec.fps()
            );
            log::info!(
                "Right video: {}x{} @ {:.1} fps",
                right_dec.width(),
                right_dec.height(),
                right_dec.fps()
            );

            let fps_val = left_dec.fps();
            let fps_rational = left_dec.frame_rate();
            // Drop decoders — the decode thread opens its own
            drop(left_dec);
            drop(right_dec);

            let json = std::fs::read_to_string(&calibration).map_err(|e| {
                anyhow::anyhow!("cannot read calibration file '{calibration}': {e}")
            })?;
            let cal: reco_core::calibration::MatchCalibration = serde_json::from_str(&json)
                .map_err(|e| anyhow::anyhow!("invalid calibration JSON '{calibration}': {e}"))?;

            let viewport = reco_core::viewport::ViewportConfig {
                width,
                height,
                ..Default::default()
            };

            // Detect zero-copy capability: CUDA available + Vulkan backend + Linux
            let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;

            #[cfg(target_os = "linux")]
            let use_zero_copy = std::env::var("RECO_NO_HWACCEL").is_err()
                && reco_core::cuda_interop::is_cuda_available()
                && gpu.adapter_info.backend == wgpu::Backend::Vulkan;
            #[cfg(not(target_os = "linux"))]
            let use_zero_copy = false;

            let input_format = if use_zero_copy {
                reco_core::renderer::InputFormat::Nv12
            } else {
                reco_core::renderer::InputFormat::Yuv420p
            };

            let mut pipeline = reco_core::pipeline::StitchPipeline::with_gpu(
                gpu,
                cal,
                viewport,
                input_width,
                input_height,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                input_format,
            )?;

            // GPU RGBA→NV12 compute shader: eliminates CPU swscale and
            // reduces GPU→CPU readback bandwidth by 2.7×
            let nv12_converter =
                reco_core::nv12_converter::Nv12Converter::new(&pipeline.gpu, width, height);

            println!(
                "Pipeline ready: GPU = {}, output = {width}x{height}, mode = {}",
                pipeline.gpu.adapter_info.name,
                if use_zero_copy {
                    "zero-copy (CUDA/Vulkan)"
                } else {
                    "CPU upload"
                }
            );

            let quality = match quality.as_str() {
                "fast" => reco_ffmpeg::encoder::Quality::Fast,
                "high" => reco_ffmpeg::encoder::Quality::High,
                _ => reco_ffmpeg::encoder::Quality::Balanced,
            };
            let video_codec = reco_ffmpeg::encoder::VideoCodec::from_str_loose(&codec)
                .unwrap_or_else(|| {
                    eprintln!("Unknown codec '{codec}', defaulting to H.264");
                    reco_ffmpeg::encoder::VideoCodec::H264
                });
            let enc_config = reco_ffmpeg::encoder::EncoderConfig {
                encoder_name: encoder,
                codec: video_codec,
                quality,
            };

            let mut encoder = reco_ffmpeg::encoder::VideoEncoder::new(
                Path::new(&output),
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

            let start = Instant::now();
            let mut frame_count: u64 = 0;
            let yaw = 0.0_f32;
            let pitch = 0.0_f32;

            #[cfg(target_os = "linux")]
            if use_zero_copy {
                // Zero-copy path: CUDA/Vulkan shared textures, no CPU upload
                frame_count = run_stitch_zero_copy(
                    &mut pipeline,
                    &mut encoder,
                    &nv12_converter,
                    &left,
                    &right,
                    input_width,
                    input_height,
                    frame_limit,
                    yaw,
                    pitch,
                    &interrupted,
                    &start,
                )?;
            }

            if !use_zero_copy {
                // CPU upload path (existing flow)
                let frame_rx = spawn_decode_thread(left, right);

                loop {
                    if frame_count >= frame_limit || interrupted.load(Ordering::Relaxed) {
                        break;
                    }

                    let pair = {
                        profile_scope!("wait_decode");
                        match frame_rx.recv() {
                            Ok(p) => p,
                            Err(_) => break,
                        }
                    };

                    let left_planes = reco_core::pipeline::YuvPlanes {
                        y: &pair.left.y,
                        u: &pair.left.u,
                        v: &pair.left.v,
                    };
                    let right_planes = reco_core::pipeline::YuvPlanes {
                        y: &pair.right.y,
                        u: &pair.right.u,
                        v: &pair.right.v,
                    };
                    let render_buf =
                        pipeline.render_to_target(&left_planes, &right_planes, yaw, pitch);
                    let nv12_data = nv12_converter.convert_and_readback(
                        &pipeline.gpu,
                        pipeline.render_target(),
                        render_buf,
                    )?;
                    encoder.write_nv12_frame(&nv12_data)?;
                    frame_count += 1;

                    if frame_count.is_multiple_of(30) {
                        let elapsed = start.elapsed().as_secs_f64();
                        let fps_actual = frame_count as f64 / elapsed;
                        print!("\rProcessed {frame_count} frames ({fps_actual:.1} fps)");
                        let _ = std::io::stdout().flush();
                    }
                }
            }

            log::info!("Finishing encoder...");
            encoder.finish()?;
            log::info!("Encoder finished");

            let elapsed = start.elapsed().as_secs_f64();
            let fps_actual = frame_count as f64 / elapsed;
            println!(
                "\nDone: {frame_count} frames in {elapsed:.1}s ({fps_actual:.1} fps) → {output}"
            );

            log::info!("Dropping pipeline...");
            drop(pipeline);
            log::info!("Pipeline dropped, exiting");

            Ok(())
        }

        Commands::Preview {
            left,
            right,
            calibration,
            width,
            height,
        } => run_preview(&left, &right, &calibration, width, height),

        Commands::Info => {
            let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;
            println!("GPU: {}", gpu.adapter_info.name);
            println!("Backend: {:?}", gpu.adapter_info.backend);
            println!("Driver: {}", gpu.adapter_info.driver);

            println!("\nH.264 encoders:");
            let encoders = reco_ffmpeg::encoder::available_h264_encoders();
            if encoders.is_empty() {
                println!("  (none found)");
            } else {
                for enc in &encoders {
                    let tag = if enc.is_hardware { "HW" } else { "SW" };
                    println!("  {} [{}] — {}", enc.name, tag, enc.description);
                }
            }
            Ok(())
        }
    }
}

/// Owned YUV420P plane data for one camera.
struct YuvBuf {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

/// A pair of decoded YUV frames (left + right), sent from the decode thread.
struct FramePair {
    left: YuvBuf,
    right: YuvBuf,
}

/// Spawn a single-video decode thread that sends YUV frames through a channel.
fn spawn_single_decoder(path: String, label: &'static str) -> std::sync::mpsc::Receiver<YuvBuf> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<YuvBuf>(4);

    std::thread::Builder::new()
        .name(format!("decode_{label}"))
        .spawn(move || {
            let mut dec = match reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&path)) {
                Ok(d) => {
                    log::info!(
                        "{label} decoder: {} ({}x{})",
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
            loop {
                match dec.next_frame() {
                    Ok(Some(f)) => {
                        let buf = YuvBuf {
                            y: f.y,
                            u: f.u,
                            v: f.v,
                        };
                        if tx.send(buf).is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::error!("{label} decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn decode thread");

    rx
}

/// Spawn parallel decode threads (one per video) and a pairing thread
/// that zips frames into `FramePair`s through a bounded channel.
fn spawn_decode_thread(
    left_path: String,
    right_path: String,
) -> std::sync::mpsc::Receiver<FramePair> {
    let left_rx = spawn_single_decoder(left_path, "left");
    let right_rx = spawn_single_decoder(right_path, "right");

    let (tx, rx) = std::sync::mpsc::sync_channel::<FramePair>(4);

    std::thread::Builder::new()
        .name("decode_pair".into())
        .spawn(move || {
            while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                if tx.send(FramePair { left, right }).is_err() {
                    break; // Consumer dropped
                }
            }
        })
        .expect("spawn pairing thread");

    rx
}

// ---- Zero-copy GPU decode (CUDA/Vulkan interop) ----

/// CUDA buffer info passed to decode threads for cuMemcpy2D destination.
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

/// Spawn a single-video GPU decode thread that writes NV12 frames directly
/// to CUDA/Vulkan shared textures via cuMemcpy2D.
///
/// Uses `slot_free_rx` for backpressure: the decode thread waits for a slot
/// to be released by the main thread before writing to it. This prevents
/// NVDEC from overwriting a slot that the GPU render pass is still reading.
fn spawn_single_decoder_gpu(
    path: String,
    label: &'static str,
    buf: GpuBufInfo,
    slot_free_rx: std::sync::mpsc::Receiver<u8>,
) -> std::sync::mpsc::Receiver<u8> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<u8>(1);

    std::thread::Builder::new()
        .name(format!("decode_{label}_gpu"))
        .spawn(move || {
            let mut dec = match reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&path)) {
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

                        // Copy Y plane: NVDEC → shared texture
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

                        // Copy UV plane: NVDEC → shared texture
                        // NV12 UV: width bytes per row, height/2 rows
                        if let Err(e) = reco_core::cuda_interop::cuda_2d_copy(
                            buf.uv_ptr[s],
                            buf.uv_pitch[s],
                            frame.uv_ptr,
                            frame.uv_pitch,
                            buf.width as usize, // UV: 2 bytes × width/2 = width
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

    rx
}

/// A pair of double-buffer slot indices from the decode threads.
struct GpuFrameSignal {
    left_slot: u8,
    right_slot: u8,
}

/// Spawn parallel GPU decode threads and a pairing thread.
fn spawn_decode_thread_gpu(
    left_path: String,
    right_path: String,
    left_buf: GpuBufInfo,
    right_buf: GpuBufInfo,
    left_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    right_slot_free_rx: std::sync::mpsc::Receiver<u8>,
) -> std::sync::mpsc::Receiver<GpuFrameSignal> {
    let left_rx = spawn_single_decoder_gpu(left_path, "left", left_buf, left_slot_free_rx);
    let right_rx = spawn_single_decoder_gpu(right_path, "right", right_buf, right_slot_free_rx);

    let (tx, rx) = std::sync::mpsc::sync_channel::<GpuFrameSignal>(1);

    std::thread::Builder::new()
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

    rx
}

/// Run the stitch loop using CUDA/Vulkan zero-copy (no CPU upload).
///
/// Creates double-buffered shared textures, spawns GPU decode threads,
/// and renders directly from GPU-resident data.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_stitch_zero_copy(
    pipeline: &mut reco_core::pipeline::StitchPipeline,
    encoder: &mut reco_ffmpeg::encoder::VideoEncoder,
    nv12_converter: &reco_core::nv12_converter::Nv12Converter,
    left_path: &str,
    right_path: &str,
    input_width: u32,
    input_height: u32,
    frame_limit: u64,
    yaw: f32,
    pitch: f32,
    interrupted: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    start: &Instant,
) -> anyhow::Result<u64> {
    use reco_core::vulkan_interop::create_shared_texture;

    // Create double-buffered shared textures for each camera (Y + UV per slot)
    log::info!("Creating shared textures for zero-copy...");

    let create_pair = |label: &str,
                       slot: usize|
     -> anyhow::Result<(
        reco_core::vulkan_interop::SharedTexture,
        reco_core::vulkan_interop::SharedTexture,
    )> {
        let y = create_shared_texture(
            &pipeline.gpu,
            input_width,
            input_height,
            wgpu::TextureFormat::R8Unorm,
        )
        .map_err(|e| anyhow::anyhow!("{label} Y[{slot}] shared texture: {e}"))?;

        let uv = create_shared_texture(
            &pipeline.gpu,
            input_width / 2,
            input_height / 2,
            wgpu::TextureFormat::Rg8Unorm,
        )
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
    left_slot_free_tx.send(0).unwrap();
    left_slot_free_tx.send(1).unwrap();
    right_slot_free_tx.send(0).unwrap();
    right_slot_free_tx.send(1).unwrap();

    // Spawn GPU decode threads
    let frame_rx = spawn_decode_thread_gpu(
        left_path.to_string(),
        right_path.to_string(),
        left_buf,
        right_buf,
        left_slot_free_rx,
        right_slot_free_rx,
    );

    println!("Zero-copy pipeline active: NVDEC → cuMemcpy2D → shared texture → render");

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

        // Set the appropriate bind groups for this frame's buffer slots
        let renderer = pipeline.renderer_mut();
        let left_idx = signal.left_slot as usize;
        let right_idx = signal.right_slot as usize;

        // Rebuild bind groups for the active double-buffer slots.
        // Bind group creation is cheap (~10µs) vs the 2.56ms we save.
        let left_bg = if left_idx == 0 {
            renderer.create_texture_bind_group(&left_y_0.texture, &left_uv_0.texture, "left_active")
        } else {
            renderer.create_texture_bind_group(&left_y_1.texture, &left_uv_1.texture, "left_active")
        };
        let right_bg = if right_idx == 0 {
            renderer.create_texture_bind_group(
                &right_y_0.texture,
                &right_uv_0.texture,
                "right_active",
            )
        } else {
            renderer.create_texture_bind_group(
                &right_y_1.texture,
                &right_uv_1.texture,
                "right_active",
            )
        };

        renderer.set_left_bind_group(left_bg);
        renderer.set_right_bind_group(right_bg);

        let render_buf = pipeline.render_to_target_gpu(yaw, pitch);
        let nv12_data = nv12_converter.convert_and_readback(
            &pipeline.gpu,
            pipeline.render_target(),
            render_buf,
        )?;
        encoder.write_nv12_frame(&nv12_data)?;

        // GPU is done reading these slots — release them for decode to reuse.
        // poll(Wait) inside convert_and_readback guarantees the render pass
        // has finished reading from the shared textures.
        let _ = left_slot_free_tx.send(signal.left_slot);
        let _ = right_slot_free_tx.send(signal.right_slot);

        frame_count += 1;

        if frame_count.is_multiple_of(30) {
            let elapsed = start.elapsed().as_secs_f64();
            let fps_actual = frame_count as f64 / elapsed;
            print!("\rProcessed {frame_count} frames ({fps_actual:.1} fps)");
            let _ = std::io::stdout().flush();
        }
    }

    // Leak shared textures intentionally — their Vulkan/CUDA cleanup interaction
    // causes a segfault (double-free between wgpu's internal VkImage destruction
    // and our drop_callback). These resources live for the entire pipeline operation
    // and the OS reclaims all GPU memory at process exit. This is standard practice
    // in GPU interop code (cf. Gyroflow).
    log::info!("Leaking shared textures (cleaned up at process exit)");
    std::mem::forget(left_y_0);
    std::mem::forget(left_uv_0);
    std::mem::forget(left_y_1);
    std::mem::forget(left_uv_1);
    std::mem::forget(right_y_0);
    std::mem::forget(right_uv_0);
    std::mem::forget(right_y_1);
    std::mem::forget(right_uv_1);
    Ok(frame_count)
}

fn run_preview(
    left_path: &str,
    right_path: &str,
    calibration_path: &str,
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::window::{Window, WindowAttributes, WindowId};

    let left_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(left_path))?;
    let right_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(right_path))?;

    let input_width = left_dec.width();
    let input_height = left_dec.height();

    anyhow::ensure!(
        input_width == right_dec.width() && input_height == right_dec.height(),
        "Video dimension mismatch: left={}x{}, right={}x{}",
        input_width,
        input_height,
        right_dec.width(),
        right_dec.height()
    );

    let fps = left_dec.fps();
    // Drop the decoders — the thread will open its own
    drop(left_dec);
    drop(right_dec);

    let json = std::fs::read_to_string(calibration_path)
        .map_err(|e| anyhow::anyhow!("cannot read calibration file '{calibration_path}': {e}"))?;
    let cal: reco_core::calibration::MatchCalibration = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("invalid calibration JSON '{calibration_path}': {e}"))?;

    println!(
        "Preview: {}x{} input, {}x{} window",
        input_width, input_height, width, height
    );

    // Spawn decode thread and get the first frame for initial display
    let frame_rx = spawn_decode_thread(left_path.to_string(), right_path.to_string());
    let first = frame_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("videos have no frames"))?;

    struct App {
        window: Option<Window>,
        surface: Option<wgpu::Surface<'static>>,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        pipeline: Option<reco_core::pipeline::StitchPipeline>,
        frame_rx: std::sync::mpsc::Receiver<FramePair>,
        cal: reco_core::calibration::MatchCalibration,
        input_width: u32,
        input_height: u32,
        width: u32,
        height: u32,
        current_left: YuvBuf,
        current_right: YuvBuf,
        yaw: f32,
        pitch: f32,
        frame_count: u64,
        playing: bool,
        needs_redraw: bool,
        frame_duration: std::time::Duration,
        last_frame_time: Instant,
        // Mouse drag state
        mouse_dragging: bool,
        last_mouse_pos: Option<(f64, f64)>,
        // Smoothed camera: target values that yaw/pitch/fov lerp toward
        target_yaw: f32,
        target_pitch: f32,
        target_fov: f32,
    }

    impl App {
        fn advance_frame(&mut self) {
            match self.frame_rx.try_recv() {
                Ok(pair) => {
                    self.apply_pair(pair);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Decode thread hasn't caught up yet — skip this frame
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.playing = false;
                    println!("End of video");
                }
            }
        }

        /// Blocking advance for step mode (N key, P key).
        fn step_frame(&mut self) {
            match self.frame_rx.recv() {
                Ok(pair) => {
                    self.apply_pair(pair);
                }
                Err(_) => {
                    self.playing = false;
                    println!("End of video");
                }
            }
        }

        fn apply_pair(&mut self, pair: FramePair) {
            self.current_left = pair.left;
            self.current_right = pair.right;
            self.frame_count += 1;
            self.needs_redraw = true;
        }

        /// Interpolate yaw/pitch/fov toward their targets for smooth camera.
        /// Returns `true` if any values changed (needs redraw).
        fn smooth_camera(&mut self) -> bool {
            const SMOOTHING: f32 = 0.3;
            const EPSILON: f32 = 0.0001;
            const FOV_EPSILON: f32 = 0.01;

            let dy = self.target_yaw - self.yaw;
            let dp = self.target_pitch - self.pitch;
            let current_fov = self
                .pipeline
                .as_ref()
                .map_or(90.0, |p| p.viewport.fov_degrees);
            let df = self.target_fov - current_fov;

            if dy.abs() < EPSILON && dp.abs() < EPSILON && df.abs() < FOV_EPSILON {
                return false;
            }

            self.yaw += dy * SMOOTHING;
            self.pitch += dp * SMOOTHING;

            if let Some(p) = &mut self.pipeline {
                p.viewport.fov_degrees += df * SMOOTHING;
                if (self.target_fov - p.viewport.fov_degrees).abs() < FOV_EPSILON {
                    p.viewport.fov_degrees = self.target_fov;
                }
            }

            if (self.target_yaw - self.yaw).abs() < EPSILON {
                self.yaw = self.target_yaw;
            }
            if (self.target_pitch - self.pitch).abs() < EPSILON {
                self.pitch = self.target_pitch;
            }
            true
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let attrs = WindowAttributes::default()
                .with_title("Reco Preview")
                .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));

            let window = event_loop.create_window(attrs).expect("create window");

            // Create wgpu surface and GPU context
            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(&window).expect("create surface");

            let (gpu, caps) = pollster::block_on(async {
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: Some(&surface),
                    })
                    .await
                    .expect("request adapter");

                let info = adapter.get_info();
                log::info!("Preview GPU: {} ({:?})", info.name, info.backend);

                let caps = surface.get_capabilities(&adapter);

                let (device, queue) = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("reco_preview"),
                        ..Default::default()
                    })
                    .await
                    .expect("request device");

                (
                    reco_core::gpu::GpuContext {
                        device,
                        queue,
                        adapter_info: info,
                    },
                    caps,
                )
            });

            self.surface_format = caps.formats[0];
            self.alpha_mode = caps.alpha_modes[0];
            let surface_format = self.surface_format;
            log::info!("Surface format: {:?}", surface_format);

            surface.configure(
                &gpu.device,
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: surface_format,
                    width: self.width,
                    height: self.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: self.alpha_mode,
                    view_formats: vec![],
                },
            );

            let viewport = reco_core::viewport::ViewportConfig {
                width: self.width,
                height: self.height,
                ..Default::default()
            };

            let pipeline = reco_core::pipeline::StitchPipeline::with_gpu(
                gpu,
                self.cal.clone(),
                viewport,
                self.input_width,
                self.input_height,
                surface_format,
                reco_core::renderer::InputFormat::Yuv420p,
            )
            .expect("create pipeline");

            println!(
                "Preview ready: GPU = {}, format = {:?}",
                pipeline.gpu.adapter_info.name, surface_format
            );
            println!("Controls: Space = play/pause, N = next frame, P = skip 30 frames");
            println!("          Arrows/drag = pan, +/-/scroll = zoom, Q/Escape = quit");

            // SAFETY: surface lifetime is tied to window which we keep alive
            self.surface = Some(unsafe {
                std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
            });
            self.pipeline = Some(pipeline);
            self.window = Some(window);
            self.needs_redraw = true;
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if size.width > 0 && size.height > 0 {
                        self.width = size.width;
                        self.height = size.height;
                        if let (Some(surface), Some(pipeline)) = (&self.surface, &mut self.pipeline)
                        {
                            surface.configure(
                                &pipeline.gpu.device,
                                &wgpu::SurfaceConfiguration {
                                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                                    format: self.surface_format,
                                    width: self.width,
                                    height: self.height,
                                    present_mode: wgpu::PresentMode::Fifo,
                                    desired_maximum_frame_latency: 2,
                                    alpha_mode: self.alpha_mode,
                                    view_formats: vec![],
                                },
                            );
                            pipeline.viewport.width = self.width;
                            pipeline.viewport.height = self.height;
                            pipeline.resize_depth(self.width, self.height);
                            self.needs_redraw = true;
                        }
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    use winit::keyboard::{KeyCode, PhysicalKey};
                    if event.state == winit::event::ElementState::Pressed {
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::Escape | KeyCode::KeyQ) => {
                                event_loop.exit();
                            }
                            PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                self.target_yaw += 0.05;
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::ArrowRight) => {
                                self.target_yaw -= 0.05;
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::ArrowUp) => {
                                self.target_pitch += 0.05;
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::ArrowDown) => {
                                self.target_pitch -= 0.05;
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::Space) => {
                                self.playing = !self.playing;
                                if self.playing {
                                    event_loop.set_control_flow(ControlFlow::Poll);
                                    println!("Playing");
                                } else {
                                    event_loop.set_control_flow(ControlFlow::Wait);
                                    println!("Paused");
                                }
                            }
                            PhysicalKey::Code(KeyCode::KeyN) => {
                                // Step one frame (blocking — waits for decode)
                                self.playing = false;
                                self.step_frame();
                            }
                            PhysicalKey::Code(KeyCode::KeyP) => {
                                // Skip 30 frames (blocking — waits for decode)
                                for _ in 0..30 {
                                    self.step_frame();
                                }
                            }
                            PhysicalKey::Code(KeyCode::Equal | KeyCode::NumpadAdd) => {
                                self.target_fov = (self.target_fov - 5.0).max(20.0);
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::Minus | KeyCode::NumpadSubtract) => {
                                self.target_fov = (self.target_fov + 5.0).min(150.0);
                                self.needs_redraw = true;
                            }
                            _ => {}
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    use winit::event::ElementState;
                    use winit::event::MouseButton;
                    if button == MouseButton::Left {
                        let pressed = state == ElementState::Pressed;
                        self.mouse_dragging = pressed;
                        if pressed {
                            // Capture start position — first CursorMoved will anchor here
                            self.last_mouse_pos = None;
                        } else {
                            self.last_mouse_pos = None;
                            if !self.playing {
                                event_loop.set_control_flow(ControlFlow::Wait);
                            }
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.mouse_dragging {
                        if let Some((prev_x, prev_y)) = self.last_mouse_pos {
                            let dx = position.x - prev_x;
                            let dy = position.y - prev_y;
                            // Accumulate into smoothing targets (raw deltas)
                            self.target_yaw += dx as f32 * 0.005;
                            self.target_pitch += dy as f32 * 0.005;
                        } else {
                            // First move after click — switch to Poll for smooth updates
                            event_loop.set_control_flow(ControlFlow::Poll);
                        }
                        self.last_mouse_pos = Some((position.x, position.y));
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y / 30.0,
                    };
                    self.target_fov = (self.target_fov - scroll as f32 * 3.0).clamp(20.0, 150.0);
                    self.needs_redraw = true;
                }
                WindowEvent::RedrawRequested => {
                    // Apply camera smoothing before rendering
                    if self.smooth_camera() {
                        self.needs_redraw = true;
                    }
                    if !self.needs_redraw && !self.playing {
                        return;
                    }
                    self.needs_redraw = false;

                    let surface = self.surface.as_ref().unwrap();
                    let pipeline = self.pipeline.as_ref().unwrap();

                    let frame = match surface.get_current_texture() {
                        Ok(f) => f,
                        Err(e) => {
                            log::warn!("Surface error: {e}");
                            return;
                        }
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let left = reco_core::pipeline::YuvPlanes {
                        y: &self.current_left.y,
                        u: &self.current_left.u,
                        v: &self.current_left.v,
                    };
                    let right = reco_core::pipeline::YuvPlanes {
                        y: &self.current_right.y,
                        u: &self.current_right.u,
                        v: &self.current_right.v,
                    };
                    pipeline.render_to_view(&left, &right, self.yaw, self.pitch, &view);

                    frame.present();
                }
                _ => {}
            }

            if self.needs_redraw
                && let Some(w) = &self.window
            {
                w.request_redraw();
            }
        }

        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            // Keep animating while smoothing hasn't converged
            let current_fov = self
                .pipeline
                .as_ref()
                .map_or(self.target_fov, |p| p.viewport.fov_degrees);
            let smoothing_active = (self.target_yaw - self.yaw).abs() > 0.0001
                || (self.target_pitch - self.pitch).abs() > 0.0001
                || (self.target_fov - current_fov).abs() > 0.01;

            if !self.playing && !smoothing_active {
                return;
            }

            if self.playing {
                let elapsed = self.last_frame_time.elapsed();
                if elapsed < self.frame_duration {
                    std::thread::sleep(self.frame_duration - elapsed);
                }

                self.advance_frame();
                self.last_frame_time = Instant::now();

                if !self.playing {
                    event_loop.set_control_flow(ControlFlow::Wait);
                }
            }

            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    let event_loop = EventLoop::new()?;

    let frame_duration = std::time::Duration::from_secs_f64(1.0 / fps);

    let mut app = App {
        window: None,
        surface: None,
        surface_format: wgpu::TextureFormat::Bgra8UnormSrgb, // overwritten in resumed()
        alpha_mode: wgpu::CompositeAlphaMode::Auto,          // overwritten in resumed()
        pipeline: None,
        frame_rx,
        cal,
        input_width,
        input_height,
        width,
        height,
        current_left: first.left,
        current_right: first.right,

        yaw: 0.0,
        pitch: 0.0,
        frame_count: 1,
        playing: false,
        needs_redraw: false,
        frame_duration,
        last_frame_time: Instant::now(),
        mouse_dragging: false,
        last_mouse_pos: None,
        target_yaw: 0.0,
        target_pitch: 0.0,
        target_fov: 75.0,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}
