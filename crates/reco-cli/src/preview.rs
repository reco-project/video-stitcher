//! Interactive preview window for debugging the panoramic stitch.
//!
//! Opens a `winit` window with GPU-rendered preview. Controls:
//! - Space: play/pause
//! - N: step one frame, P: skip 30 frames
//! - Arrows / mouse drag: pan (yaw/pitch)
//! - +/- / scroll: zoom (FOV)
//! - Q / Escape: quit

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use reco_core::source::{FramePair, YuvData};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::helpers;

/// Spawn a single-video decode thread that sends YUV frames through a channel.
fn spawn_single_decoder(path: String, label: &'static str) -> std::sync::mpsc::Receiver<YuvData> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<YuvData>(4);

    std::thread::Builder::new()
        .name(format!("decode_{label}"))
        .spawn(move || {
            let mut dec = match reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(&path)) {
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
                        let buf = YuvData {
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
fn spawn_preview_decode_thread(
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

/// Run the interactive preview window.
pub fn run_preview(
    left_path: &str,
    right_path: &str,
    calibration_path: &str,
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    let left_dec = reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(left_path))?;
    let right_dec = reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(right_path))?;

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
    // Drop the decoders - the thread will open its own
    drop(left_dec);
    drop(right_dec);

    let cal = helpers::load_calibration(Path::new(calibration_path))?;

    println!(
        "Preview: {}x{} input, {}x{} window",
        input_width, input_height, width, height
    );

    // Spawn decode thread and get the first frame for initial display
    let frame_rx = spawn_preview_decode_thread(left_path.to_string(), right_path.to_string());
    let first = frame_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("videos have no frames"))?;

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

struct App {
    surface: Option<wgpu::Surface<'static>>,
    window: Option<Arc<Window>>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    pipeline: Option<reco_core::pipeline::StitchPipeline>,
    frame_rx: std::sync::mpsc::Receiver<FramePair>,
    cal: reco_core::calibration::MatchCalibration,
    input_width: u32,
    input_height: u32,
    width: u32,
    height: u32,
    current_left: YuvData,
    current_right: YuvData,
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
                // Decode thread hasn't caught up yet - skip this frame
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
        let current_fov = self.pipeline.as_ref().map_or(90.0, |p| p.fov());
        let df = self.target_fov - current_fov;

        if dy.abs() < EPSILON && dp.abs() < EPSILON && df.abs() < FOV_EPSILON {
            return false;
        }

        self.yaw += dy * SMOOTHING;
        self.pitch += dp * SMOOTHING;

        if let Some(p) = &mut self.pipeline {
            let new_fov = p.fov() + df * SMOOTHING;
            p.set_fov(new_fov);
            if (self.target_fov - p.fov()).abs() < FOV_EPSILON {
                p.set_fov(self.target_fov);
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

        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        // Create wgpu surface and GPU context
        // Arc<Window> gives Surface<'static> without transmute
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

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
            pipeline.gpu_name(),
            surface_format
        );
        println!("Controls: Space = play/pause, N = next frame, P = skip 30 frames");
        println!("          Arrows/drag = pan, +/-/scroll = zoom, Q/Escape = quit");

        self.surface = Some(surface);
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
                    if let (Some(surface), Some(pipeline)) = (&self.surface, &mut self.pipeline) {
                        surface.configure(
                            &pipeline.gpu().device,
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
                        pipeline.resize(self.width, self.height);
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
                            // Step one frame (blocking - waits for decode)
                            self.playing = false;
                            self.step_frame();
                        }
                        PhysicalKey::Code(KeyCode::KeyP) => {
                            // Skip 30 frames (blocking - waits for decode)
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
                        // Capture start position - first CursorMoved will anchor here
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
                        // First move after click - switch to Poll for smooth updates
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
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    other => {
                        log::warn!("Surface error: {other:?}");
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
        let current_fov = self.pipeline.as_ref().map_or(self.target_fov, |p| p.fov());
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
