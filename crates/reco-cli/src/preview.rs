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

use reco_core::source::{FrameSource, YuvData};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

// ---- Preview constants ----

/// Yaw/pitch increment per arrow key press (radians).
const ARROW_PAN_STEP: f32 = 0.05;
/// Mouse drag sensitivity: radians per pixel of cursor movement.
const MOUSE_SENSITIVITY: f32 = 0.005;
/// FOV change per +/- key press (degrees).
const FOV_KEY_STEP: f32 = 5.0;
/// FOV change per scroll tick (degrees).
const FOV_SCROLL_STEP: f32 = 3.0;
/// Minimum FOV (degrees) - prevents extreme zoom-in.
const FOV_MIN: f32 = 20.0;
/// Maximum FOV (degrees) - prevents extreme zoom-out.
const FOV_MAX: f32 = 150.0;
/// Default FOV at startup (degrees).
const FOV_DEFAULT: f32 = 75.0;
/// Number of frames to skip on P key press.
const FRAME_SKIP_COUNT: usize = 30;

/// Extract a [`YuvData`] pair from a [`StereoFrame`](reco_core::source::StereoFrame).
///
/// Panics if the frame is not `Yuv420p` (preview always uses CPU decode).
fn unwrap_yuv_pair(frame: reco_core::source::StereoFrame) -> (YuvData, YuvData) {
    match frame {
        reco_core::source::StereoFrame::Yuv420p(pair) => (pair.left, pair.right),
        _ => panic!("preview expects Yuv420p frames"),
    }
}

/// Run the interactive preview window.
pub fn run_preview(
    left_path: &str,
    right_path: &str,
    calibration_path: &str,
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    // Probe the right file to verify dimensions match
    let right_dec = reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(right_path))?;
    let right_dims = (right_dec.width(), right_dec.height());
    drop(right_dec);

    let mut source =
        reco_io::adapters::FfmpegFileSource::open(Path::new(left_path), Path::new(right_path))?;
    let info = source.info();

    anyhow::ensure!(
        info.width == right_dims.0 && info.height == right_dims.1,
        "Video dimension mismatch: left={}x{}, right={}x{}",
        info.width,
        info.height,
        right_dims.0,
        right_dims.1
    );

    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(calibration_path))?;

    println!(
        "Preview: {}x{} input, {}x{} window",
        info.width, info.height, width, height
    );

    // Get the first frame for initial display
    let first = source
        .next_frame()?
        .ok_or_else(|| anyhow::anyhow!("videos have no frames"))?;
    let (first_left, first_right) = unwrap_yuv_pair(first);

    let event_loop = EventLoop::new()?;

    let frame_duration = std::time::Duration::from_secs_f64(1.0 / info.fps);

    let mut app = App {
        window: None,
        surface: None,
        surface_format: reco_core::wgpu::TextureFormat::Bgra8UnormSrgb, // overwritten in resumed()
        alpha_mode: reco_core::wgpu::CompositeAlphaMode::Auto,          // overwritten in resumed()
        pipeline: None,
        source,
        cal,
        input_width: info.width,
        input_height: info.height,
        width,
        height,
        current_left: first_left,
        current_right: first_right,

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
        target_fov: FOV_DEFAULT,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    surface: Option<reco_core::wgpu::Surface<'static>>,
    window: Option<Arc<Window>>,
    surface_format: reco_core::wgpu::TextureFormat,
    alpha_mode: reco_core::wgpu::CompositeAlphaMode,
    pipeline: Option<reco_core::pipeline::StitchPipeline>,
    source: reco_io::adapters::FfmpegFileSource,
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
        match self.source.try_next_frame() {
            Ok(Some(frame)) => {
                let (left, right) = unwrap_yuv_pair(frame);
                self.current_left = left;
                self.current_right = right;
                self.frame_count += 1;
                self.needs_redraw = true;
            }
            Ok(None) => {
                // Decode thread hasn't caught up yet, or end of stream
            }
            Err(e) => {
                log::error!("Decode error: {e}");
                self.playing = false;
                println!("End of video");
            }
        }
    }

    /// Blocking advance for step mode (N key, P key).
    fn step_frame(&mut self) {
        match self.source.next_frame() {
            Ok(Some(frame)) => {
                let (left, right) = unwrap_yuv_pair(frame);
                self.current_left = left;
                self.current_right = right;
                self.frame_count += 1;
                self.needs_redraw = true;
            }
            Ok(None) | Err(_) => {
                self.playing = false;
                println!("End of video");
            }
        }
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

        // Create wgpu surface and GPU context via reco-core helper
        // Arc<Window> gives Surface<'static> without transmute
        let instance = reco_core::wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let (gpu, surface_info) =
            pollster::block_on(reco_core::gpu::GpuContext::for_surface(&instance, &surface))
                .expect("create GPU context for surface");

        self.surface_format = surface_info.format;
        self.alpha_mode = surface_info.alpha_modes[0];
        let surface_format = self.surface_format;
        log::info!("Surface format: {:?}", surface_format);

        surface.configure(
            gpu.device(),
            &reco_core::wgpu::SurfaceConfiguration {
                usage: reco_core::wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: self.width,
                height: self.height,
                present_mode: reco_core::wgpu::PresentMode::Fifo,
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
                            pipeline.gpu().device(),
                            &reco_core::wgpu::SurfaceConfiguration {
                                usage: reco_core::wgpu::TextureUsages::RENDER_ATTACHMENT,
                                format: self.surface_format,
                                width: self.width,
                                height: self.height,
                                present_mode: reco_core::wgpu::PresentMode::Fifo,
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
                            self.target_yaw += ARROW_PAN_STEP;
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.target_yaw -= ARROW_PAN_STEP;
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.target_pitch += ARROW_PAN_STEP;
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.target_pitch -= ARROW_PAN_STEP;
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
                            for _ in 0..FRAME_SKIP_COUNT {
                                self.step_frame();
                            }
                        }
                        PhysicalKey::Code(KeyCode::Equal | KeyCode::NumpadAdd) => {
                            self.target_fov = (self.target_fov - FOV_KEY_STEP).max(FOV_MIN);
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::Minus | KeyCode::NumpadSubtract) => {
                            self.target_fov = (self.target_fov + FOV_KEY_STEP).min(FOV_MAX);
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
                        self.target_yaw += dx as f32 * MOUSE_SENSITIVITY;
                        self.target_pitch += dy as f32 * MOUSE_SENSITIVITY;
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
                self.target_fov =
                    (self.target_fov - scroll as f32 * FOV_SCROLL_STEP).clamp(FOV_MIN, FOV_MAX);
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

                let (Some(surface), Some(pipeline)) =
                    (self.surface.as_ref(), self.pipeline.as_ref())
                else {
                    return;
                };

                let frame = match surface.get_current_texture() {
                    reco_core::wgpu::CurrentSurfaceTexture::Success(f)
                    | reco_core::wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    other => {
                        log::warn!("Surface error: {other:?}");
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&reco_core::wgpu::TextureViewDescriptor::default());

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
                if let Err(e) = pipeline.render_to_view(&left, &right, self.yaw, self.pitch, &view)
                {
                    log::error!("Render failed: {e}");
                    return;
                }

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
