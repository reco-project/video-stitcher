//! Interactive preview window for debugging the panoramic stitch.
//!
//! Opens a `winit` window with GPU-rendered preview. Controls:
//! - Space: play/pause
//! - N: step one frame, P: skip 30 frames
//! - Arrows / mouse drag: pan (yaw/pitch)
//! - +/- / scroll: zoom (FOV)
//! - Q / Escape: quit
//!
//! This module is intentionally CLI-only. The rendering is already handled by
//! `StitchRenderer` in reco-core. What remains here is the winit event loop,
//! surface management, input handling, and frame pacing - all tightly coupled
//! to the desktop window environment and not useful to library consumers
//! (GUI, OBS, cloud). A future GUI app would use its own event loop and call
//! the same `StitchRenderer` API directly.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use reco_core::source::{FrameSource, YuvData};
use reco_core::stitch_renderer::StitchRenderer;
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

/// Configuration for the interactive preview window.
pub struct PreviewConfig<'a> {
    pub left_path: &'a str,
    pub right_path: &'a str,
    pub calibration_path: &'a str,
    pub width: u32,
    pub height: u32,
    pub sync_offset: i64,
    pub blend_width: f32,
    pub rig_tilt_degrees: f32,
}

/// Run the interactive preview window.
pub fn run_preview(
    config: &PreviewConfig<'_>,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let PreviewConfig {
        left_path,
        right_path,
        calibration_path,
        width,
        height,
        sync_offset,
        blend_width,
        rig_tilt_degrees,
    } = *config;
    // Load calibration first so we can use its sync_offset and rig_tilt
    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(calibration_path))?;

    // Use calibration's sync offset unless the user explicitly overrode it
    let effective_sync = if sync_offset != 0 {
        sync_offset
    } else {
        cal.sync_offset
    };
    // Use calibration's rig tilt unless the user explicitly overrode it.
    // User provides degrees, calibration stores radians.
    let rig_tilt_degrees = if rig_tilt_degrees.abs() > 1e-6 {
        rig_tilt_degrees
    } else {
        (cal.rig_tilt as f32).to_degrees()
    };

    // Probe the right file to verify dimensions match
    let right_dec = reco_io::ffmpeg::decoder::VideoDecoder::open(Path::new(right_path))?;
    let right_dims = (right_dec.width(), right_dec.height());
    drop(right_dec);

    let mut source = reco_io::adapters::FfmpegFileSource::open_with_offset(
        Path::new(left_path),
        Path::new(right_path),
        effective_sync,
    )?;
    let info = source.info();

    anyhow::ensure!(
        info.width == right_dims.0 && info.height == right_dims.1,
        "Video dimension mismatch: left={}x{}, right={}x{}",
        info.width,
        info.height,
        right_dims.0,
        right_dims.1
    );

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

    // Precompute max FOV from coverage boundary using calibration metadata.
    // The actual CoverageBoundary is computed inside StitchRenderer::new().
    let max_fov = {
        let aspect = info.width as f32 / info.height as f32;
        let scene = reco_core::scene::SceneGeometry::from_layout_with_aspect(&cal.layout, aspect);
        let coverage = reco_core::projection::CoverageBoundary::from_calibration(&cal, &scene);
        coverage.max_fov_degrees().min(FOV_MAX)
    };
    let initial_fov = FOV_DEFAULT.min(max_fov);
    log::info!("Preview: max FOV = {max_fov:.1} degrees (coverage-limited)");

    let mut app = App {
        source: Some(source),
        window: None,
        surface: None,
        surface_format: reco_core::wgpu::TextureFormat::Bgra8UnormSrgb, // overwritten in resumed()
        alpha_mode: reco_core::wgpu::CompositeAlphaMode::Auto,          // overwritten in resumed()
        renderer: None,
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
        target_fov: initial_fov,
        blend_width,
        rig_tilt: rig_tilt_degrees.to_radians(),
        max_fov,
        clamp_enabled: true,
        interrupted: interrupted.clone(),
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    // Drop order matters: source (NVDEC) must drop before renderer (wgpu/Vulkan)
    // to avoid CUDA context teardown race. Option lets us drop explicitly.
    source: Option<reco_io::adapters::FfmpegFileSource>,
    surface: Option<reco_core::wgpu::Surface<'static>>,
    window: Option<Arc<Window>>,
    surface_format: reco_core::wgpu::TextureFormat,
    alpha_mode: reco_core::wgpu::CompositeAlphaMode,
    renderer: Option<StitchRenderer>,
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
    blend_width: f32,
    rig_tilt: f32,
    /// Maximum FOV from coverage (cached from coverage.max_fov_degrees()).
    max_fov: f32,
    /// Whether coverage boundary clamping is active.
    clamp_enabled: bool,
    /// Ctrl-C signal from the main thread.
    interrupted: Arc<AtomicBool>,
}

impl App {
    /// Apply a calibration change: update renderer + refresh max_fov from new coverage.
    fn apply_calibration_change(&mut self) {
        if let Some(ref mut r) = self.renderer {
            r.update_calibration(self.cal.clone());
            self.max_fov = r.coverage().max_fov_degrees().min(FOV_MAX);
            // Clamp current FOV target to new max
            self.target_fov = self.target_fov.min(self.max_fov);
        }
        self.needs_redraw = true;
    }

    fn advance_frame(&mut self) {
        let Some(ref mut source) = self.source else {
            return;
        };
        match source.try_next_frame() {
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
        let Some(ref mut source) = self.source else {
            return;
        };
        match source.next_frame() {
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
        let current_fov = self.renderer.as_ref().map_or(90.0, |r| r.pipeline().fov());
        let df = self.target_fov - current_fov;

        if dy.abs() < EPSILON && dp.abs() < EPSILON && df.abs() < FOV_EPSILON {
            return false;
        }

        self.yaw += dy * SMOOTHING;
        self.pitch += dp * SMOOTHING;

        if let Some(r) = &mut self.renderer {
            let new_fov = r.pipeline().fov() + df * SMOOTHING;
            r.pipeline_mut().set_fov(new_fov.min(self.max_fov));
            if (self.target_fov - r.pipeline().fov()).abs() < FOV_EPSILON {
                r.pipeline_mut().set_fov(self.target_fov);
            }
        }

        if (self.target_yaw - self.yaw).abs() < EPSILON {
            self.yaw = self.target_yaw;
        }
        if (self.target_pitch - self.pitch).abs() < EPSILON {
            self.pitch = self.target_pitch;
        }

        // Clamp to coverage boundary so no black edges appear
        if self.clamp_enabled
            && let Some(ref renderer) = self.renderer
        {
            let coverage = renderer.coverage();
            let current_fov = renderer.pipeline().fov();
            let aspect = self.width as f32 / self.height as f32;
            let clamped =
                coverage.safe_clamp(self.yaw, self.pitch, current_fov, aspect, self.rig_tilt);
            self.yaw = clamped.yaw;
            self.pitch = clamped.pitch;
            let target_clamped = coverage.safe_clamp(
                self.target_yaw,
                self.target_pitch,
                current_fov,
                aspect,
                self.rig_tilt,
            );
            self.target_yaw = target_clamped.yaw;
            self.target_pitch = target_clamped.pitch;
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

        // Configure surface with stripped sRGB view format to avoid double-gamma.
        let render_format = StitchRenderer::strip_srgb(surface_format);
        let view_formats = if render_format != surface_format {
            vec![render_format]
        } else {
            vec![]
        };

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
                view_formats,
            },
        );

        let viewport = reco_core::viewport::ViewportConfig {
            width: self.width,
            height: self.height,
            blend_width: self.blend_width,
            rig_tilt: self.rig_tilt,
            ..Default::default()
        };

        let renderer = StitchRenderer::new(
            self.cal.clone(),
            gpu,
            viewport,
            self.input_width,
            self.input_height,
            surface_format,
        )
        .expect("create renderer");

        println!(
            "Preview ready: GPU = {}, format = {:?}",
            renderer.gpu().gpu_name(),
            surface_format
        );
        println!("Controls: Space = play/pause, N = next, P = skip 30, Q = quit");
        println!("          Arrows/drag = pan, +/-/scroll = zoom");
        println!("          1/2 = intersect, 3/4 = axis offset, 5/6 = vertical align");
        println!("          7/8 = focal length, 9/0 = k1 distortion (barrel/pincushion)");
        println!("          B = cycle blend width, C = toggle clamping, S = save calibration");
        println!("          Arrows/drag = pan, +/-/scroll = zoom, Q/Escape = quit");

        self.surface = Some(surface);
        self.renderer = Some(renderer);
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
            WindowEvent::CloseRequested => {
                // Drop decoder (NVDEC) before GPU pipeline to avoid
                // CUDA context teardown race on exit.
                self.source.take();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.width = size.width;
                    self.height = size.height;
                    if let (Some(surface), Some(renderer)) = (&self.surface, &mut self.renderer) {
                        let render_format = StitchRenderer::strip_srgb(self.surface_format);
                        let view_formats = if render_format != self.surface_format {
                            vec![render_format]
                        } else {
                            vec![]
                        };
                        surface.configure(
                            renderer.gpu().device(),
                            &reco_core::wgpu::SurfaceConfiguration {
                                usage: reco_core::wgpu::TextureUsages::RENDER_ATTACHMENT,
                                format: self.surface_format,
                                width: self.width,
                                height: self.height,
                                present_mode: reco_core::wgpu::PresentMode::Fifo,
                                desired_maximum_frame_latency: 2,
                                alpha_mode: self.alpha_mode,
                                view_formats,
                            },
                        );
                        renderer.pipeline_mut().resize(self.width, self.height);
                        self.needs_redraw = true;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state == winit::event::ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape | KeyCode::KeyQ) => {
                            self.source.take();
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
                            self.target_fov = (self.target_fov + FOV_KEY_STEP).min(self.max_fov);
                            self.needs_redraw = true;
                        }
                        // Calibration adjustment keys
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            self.cal.layout.intersect = (self.cal.layout.intersect + 0.01).min(1.0);
                            self.apply_calibration_change();
                            println!("intersect: {:.4}", self.cal.layout.intersect);
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            self.cal.layout.intersect = (self.cal.layout.intersect - 0.01).max(0.0);
                            self.apply_calibration_change();
                            println!("intersect: {:.4}", self.cal.layout.intersect);
                        }
                        PhysicalKey::Code(KeyCode::Digit3) => {
                            self.cal.layout.camera_axis_offset += 0.005;
                            self.apply_calibration_change();
                            println!(
                                "camera_axis_offset: {:.4}",
                                self.cal.layout.camera_axis_offset
                            );
                        }
                        PhysicalKey::Code(KeyCode::Digit4) => {
                            self.cal.layout.camera_axis_offset -= 0.005;
                            self.apply_calibration_change();
                            println!(
                                "camera_axis_offset: {:.4}",
                                self.cal.layout.camera_axis_offset
                            );
                        }
                        PhysicalKey::Code(KeyCode::Digit5) => {
                            self.cal.layout.x_ty += 0.005;
                            self.apply_calibration_change();
                            println!("x_ty: {:.4}", self.cal.layout.x_ty);
                        }
                        PhysicalKey::Code(KeyCode::Digit6) => {
                            self.cal.layout.x_ty -= 0.005;
                            self.apply_calibration_change();
                            println!("x_ty: {:.4}", self.cal.layout.x_ty);
                        }
                        PhysicalKey::Code(KeyCode::KeyB) => {
                            // Cycle blend width: 0.0 -> 0.05 -> 0.10 -> 0.15 -> 0.20 -> 0.0
                            self.blend_width = if self.blend_width >= 0.19 {
                                0.0
                            } else {
                                self.blend_width + 0.05
                            };
                            if let Some(ref mut r) = self.renderer {
                                r.set_blend_width(self.blend_width);
                            }
                            println!("blend_width: {:.2}", self.blend_width);
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::Digit7) => {
                            // Increase focal length (both cameras) - zoom in effect
                            self.cal.left.fx *= 1.02;
                            self.cal.left.fy *= 1.02;
                            self.cal.right.fx *= 1.02;
                            self.cal.right.fy *= 1.02;
                            self.apply_calibration_change();
                            println!(
                                "focal length: {:.1} / {:.1}",
                                self.cal.left.fx, self.cal.right.fx
                            );
                        }
                        PhysicalKey::Code(KeyCode::Digit8) => {
                            // Decrease focal length - zoom out / wider
                            self.cal.left.fx *= 0.98;
                            self.cal.left.fy *= 0.98;
                            self.cal.right.fx *= 0.98;
                            self.cal.right.fy *= 0.98;
                            self.apply_calibration_change();
                            println!(
                                "focal length: {:.1} / {:.1}",
                                self.cal.left.fx, self.cal.right.fx
                            );
                        }
                        PhysicalKey::Code(KeyCode::Digit9) => {
                            // Increase k1 distortion - more barrel
                            self.cal.left.d[0] += 0.005;
                            self.cal.right.d[0] += 0.005;
                            self.apply_calibration_change();
                            println!(
                                "k1 distortion: {:.4} / {:.4}",
                                self.cal.left.d[0], self.cal.right.d[0]
                            );
                        }
                        PhysicalKey::Code(KeyCode::Digit0) => {
                            // Decrease k1 distortion - less barrel / more pincushion
                            self.cal.left.d[0] -= 0.005;
                            self.cal.right.d[0] -= 0.005;
                            self.apply_calibration_change();
                            println!(
                                "k1 distortion: {:.4} / {:.4}",
                                self.cal.left.d[0], self.cal.right.d[0]
                            );
                        }
                        PhysicalKey::Code(KeyCode::KeyC) => {
                            self.clamp_enabled = !self.clamp_enabled;
                            println!(
                                "Coverage clamping: {}",
                                if self.clamp_enabled { "ON" } else { "OFF" }
                            );
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            // Save calibration
                            let path = std::path::Path::new("calibration_adjusted.json");
                            match self.cal.to_file(path) {
                                Ok(()) => println!("Calibration saved to {}", path.display()),
                                Err(e) => eprintln!("Failed to save calibration: {e}"),
                            }
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
                self.target_fov = (self.target_fov - scroll as f32 * FOV_SCROLL_STEP)
                    .clamp(FOV_MIN, self.max_fov);
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

                let (Some(surface), Some(renderer)) =
                    (self.surface.as_ref(), self.renderer.as_ref())
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
                let render_format = StitchRenderer::strip_srgb(self.surface_format);
                let view = frame
                    .texture
                    .create_view(&reco_core::wgpu::TextureViewDescriptor {
                        format: Some(render_format),
                        ..Default::default()
                    });

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
                // Apply rig tilt yaw-pitch coupling: adjust pitch so the
                // view stays level on the field as yaw changes.
                let render_pitch = self.pitch + self.rig_tilt * (1.0 - self.yaw.cos());
                if let Err(e) = renderer.render_yuv(&left, &right, self.yaw, render_pitch, &view) {
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
        // Check Ctrl-C
        if self.interrupted.load(Ordering::Relaxed) {
            self.source.take();
            event_loop.exit();
            return;
        }

        // Keep animating while smoothing hasn't converged
        let current_fov = self
            .renderer
            .as_ref()
            .map_or(self.target_fov, |r| r.pipeline().fov());
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
