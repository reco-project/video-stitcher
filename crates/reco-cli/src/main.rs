//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

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

        /// Force a specific encoder (e.g., h264_nvenc, libx264). Auto-detects by default.
        #[arg(long)]
        encoder: Option<String>,

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
    env_logger::init();
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
            quality,
        } => {
            log::info!("Stitching: {left} + {right} → {output}");

            // Open video decoders first to get input dimensions
            let mut left_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&left))?;
            let mut right_dec = reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&right))?;

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

            let json = std::fs::read_to_string(&calibration)?;
            let cal: reco_core::calibration::MatchCalibration = serde_json::from_str(&json)?;

            let viewport = reco_core::viewport::ViewportConfig {
                width,
                height,
                ..Default::default()
            };

            let pipeline = pollster::block_on(reco_core::pipeline::StitchPipeline::new(
                cal,
                viewport,
                input_width,
                input_height,
            ))?;

            println!(
                "Pipeline ready: GPU = {}, output = {width}x{height}",
                pipeline.gpu.adapter_info.name
            );

            // Create encoder using the left video's frame rate
            let fps = left_dec.frame_rate();
            let quality = match quality.as_str() {
                "fast" => reco_ffmpeg::encoder::Quality::Fast,
                "high" => reco_ffmpeg::encoder::Quality::High,
                _ => reco_ffmpeg::encoder::Quality::Balanced,
            };
            let enc_config = reco_ffmpeg::encoder::EncoderConfig {
                encoder_name: encoder,
                quality,
            };
            let mut encoder = reco_ffmpeg::encoder::VideoEncoder::new(
                Path::new(&output),
                width,
                height,
                fps,
                &enc_config,
            )?;
            println!("Encoder: {}", encoder.encoder_name());

            // Compute frame limit from --duration and --max-frames
            let frame_limit: u64 = match (duration, max_frames) {
                (Some(dur), Some(mf)) => {
                    let dur_frames = (dur * left_dec.fps()) as u64;
                    dur_frames.min(mf)
                }
                (Some(dur), None) => (dur * left_dec.fps()) as u64,
                (None, Some(mf)) => mf,
                (None, None) => u64::MAX,
            };

            if frame_limit < u64::MAX {
                println!("Processing up to {frame_limit} frames");
            }

            let start = Instant::now();
            let mut frame_count: u64 = 0;

            // Static camera: yaw=0, pitch=0 (centered on seam)
            let yaw = 0.0_f32;
            let pitch = 0.0_f32;

            loop {
                if frame_count >= frame_limit {
                    break;
                }

                let left_frame = left_dec.next_frame()?;
                let right_frame = right_dec.next_frame()?;

                let (left_frame, right_frame) = match (left_frame, right_frame) {
                    (Some(l), Some(r)) => (l, r),
                    _ => break, // Either stream ended
                };

                let stitched =
                    pipeline.process_frame(&left_frame.data, &right_frame.data, yaw, pitch)?;

                encoder.write_frame(&stitched)?;
                frame_count += 1;

                if frame_count.is_multiple_of(30) {
                    let elapsed = start.elapsed().as_secs_f64();
                    let fps_actual = frame_count as f64 / elapsed;
                    print!("\rProcessed {frame_count} frames ({fps_actual:.1} fps)");
                    let _ = std::io::stdout().flush();
                }
            }

            encoder.finish()?;

            let elapsed = start.elapsed().as_secs_f64();
            let fps_actual = frame_count as f64 / elapsed;
            println!(
                "\nDone: {frame_count} frames in {elapsed:.1}s ({fps_actual:.1} fps) → {output}"
            );

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

/// A pair of decoded RGBA frames (left + right), sent from the decode thread.
struct FramePair {
    left: Vec<u8>,
    right: Vec<u8>,
}

/// Spawn a background thread that decodes both video files in lockstep
/// and sends frame pairs through a bounded channel.
fn spawn_decode_thread(
    left_path: String,
    right_path: String,
) -> std::sync::mpsc::Receiver<FramePair> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<FramePair>(4);

    std::thread::spawn(move || {
        let mut left_dec = match reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&left_path)) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to open left video: {e}");
                return;
            }
        };
        let mut right_dec = match reco_ffmpeg::decoder::VideoDecoder::open(Path::new(&right_path)) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to open right video: {e}");
                return;
            }
        };

        loop {
            let left = match left_dec.next_frame() {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    log::error!("Left decode error: {e}");
                    break;
                }
            };
            let right = match right_dec.next_frame() {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    log::error!("Right decode error: {e}");
                    break;
                }
            };
            if tx
                .send(FramePair {
                    left: left.data,
                    right: right.data,
                })
                .is_err()
            {
                break; // Receiver dropped (window closed)
            }
        }
    });

    rx
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

    let json = std::fs::read_to_string(calibration_path)?;
    let cal: reco_core::calibration::MatchCalibration = serde_json::from_str(&json)?;

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
        current_left: Vec<u8>,
        current_right: Vec<u8>,
        yaw: f32,
        pitch: f32,
        frame_count: u64,
        playing: bool,
        needs_redraw: bool,
        frame_duration: std::time::Duration,
        last_frame_time: Instant,
    }

    impl App {
        fn advance_frame(&mut self) {
            match self.frame_rx.try_recv() {
                Ok(pair) => {
                    self.current_left = pair.left;
                    self.current_right = pair.right;
                    self.frame_count += 1;
                    self.needs_redraw = true;
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
                    self.current_left = pair.left;
                    self.current_right = pair.right;
                    self.frame_count += 1;
                    self.needs_redraw = true;
                }
                Err(_) => {
                    self.playing = false;
                    println!("End of video");
                }
            }
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
            )
            .expect("create pipeline");

            println!(
                "Preview ready: GPU = {}, format = {:?}",
                pipeline.gpu.adapter_info.name, surface_format
            );
            println!("Controls: Space = play/pause, N = next frame, P = skip 30 frames");
            println!("          Arrows = pan, +/- = zoom, Q/Escape = quit");

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
                                self.yaw += 0.05;
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::ArrowRight) => {
                                self.yaw -= 0.05;
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::ArrowUp) => {
                                self.pitch += 0.05;
                                self.needs_redraw = true;
                            }
                            PhysicalKey::Code(KeyCode::ArrowDown) => {
                                self.pitch -= 0.05;
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
                                // Zoom in (decrease FOV)
                                if let Some(p) = &mut self.pipeline {
                                    p.viewport.fov_degrees =
                                        (p.viewport.fov_degrees - 5.0).max(20.0);
                                    println!("FOV: {:.0}°", p.viewport.fov_degrees);
                                    self.needs_redraw = true;
                                }
                            }
                            PhysicalKey::Code(KeyCode::Minus | KeyCode::NumpadSubtract) => {
                                // Zoom out (increase FOV)
                                if let Some(p) = &mut self.pipeline {
                                    p.viewport.fov_degrees =
                                        (p.viewport.fov_degrees + 5.0).min(150.0);
                                    println!("FOV: {:.0}°", p.viewport.fov_degrees);
                                    self.needs_redraw = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
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

                    pipeline.render_to_view(
                        &self.current_left,
                        &self.current_right,
                        self.yaw,
                        self.pitch,
                        &view,
                    );

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
            if !self.playing {
                return;
            }

            let elapsed = self.last_frame_time.elapsed();
            if elapsed < self.frame_duration {
                // Yield CPU until next frame is due instead of busy-spinning
                std::thread::sleep(self.frame_duration - elapsed);
            }

            self.advance_frame();
            self.last_frame_time = Instant::now();

            if !self.playing {
                event_loop.set_control_flow(ControlFlow::Wait);
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
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}
