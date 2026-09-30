//! Interactive preview window for debugging the panoramic stitch.
//!
//! Opens a `winit` window with GPU-rendered preview. Controls:
//! - Space: play/pause
//! - N: step one frame, P: skip 30 frames
//! - R: toggle recording to MP4
//! - `[`/`]`: seek backward/forward 5 seconds, Home: restart
//! - Arrows / mouse drag: pan (yaw/pitch)
//! - +/- / scroll: zoom (FOV)
//! - Q / Escape: quit
//!
//! This module is intentionally CLI-only. The rendering is already handled by
//! `StitchCore` in reco-core. What remains here is the winit event loop,
//! surface management, input handling, and frame pacing - all tightly coupled
//! to the desktop window environment and not useful to library consumers
//! (GUI, OBS, cloud). A future GUI app would use its own event loop and call
//! the same `StitchCore` API directly.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use reco_control::ControlIntent;
use reco_core::core::StitchCore;
use reco_core::render::viewport::ViewportSize;
use reco_core::sink::{OutputFrame, OutputSink, PixelFormat};
use reco_core::source::{FrameSet, FrameSource};
use reco_core::stitch::{Executor, GpuExecutor, GpuExecutorConfig};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

// ---- Preview constants ----

/// Yaw/pitch increment per arrow key press (radians). Also the
/// PoseControl hotkey step for the preview window.
const ARROW_PAN_STEP: f32 = 0.05;
/// FOV change per +/- key press (degrees).
const FOV_KEY_STEP: f32 = 5.0;
/// FOV change per scroll tick (degrees).
const FOV_SCROLL_STEP: f32 = 3.0;
/// Minimum FOV (degrees) - prevents extreme zoom-in.
const FOV_MIN: f32 = 1.0;
/// Maximum FOV (degrees) - prevents extreme zoom-out.
const FOV_MAX: f32 = 150.0;
/// Default FOV at startup (degrees).
const FOV_DEFAULT: f32 = 75.0;
/// Number of frames to skip on P key press.
const FRAME_SKIP_COUNT: usize = 30;
/// Number of seconds to seek on `[`/`]` key press.
const SEEK_STEP_SECS: f64 = 5.0;

/// Configuration for the interactive preview window.
pub struct PreviewConfig<'a> {
    pub inputs: &'a [String],
    pub calibration_path: &'a str,
    pub width: u32,
    pub height: u32,
    pub sync_offset: i64,
    /// Overrides the calibration's saved seam blend when set.
    pub blend_width: Option<f32>,
    pub rig_tilt_degrees: f32,
}

/// Run the interactive preview window.
pub fn run_preview(
    config: &PreviewConfig<'_>,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let PreviewConfig {
        inputs,
        calibration_path,
        width,
        height,
        sync_offset,
        blend_width,
        rig_tilt_degrees,
    } = *config;
    // Load calibration first so we can use its sync_offset and rig_tilt
    let mut cal = reco_core::calibration::Calibration::from_file(Path::new(calibration_path))?;
    // Blend lives on the calibration document; --blend is an explicit
    // override, otherwise the saved value is used (and hotkey edits
    // mutate the document directly - no shadow copy).
    anyhow::ensure!(
        inputs.len() == cal.projection.camera_count(),
        "calibration requires {} inputs, got {}",
        cal.projection.camera_count(),
        inputs.len()
    );
    if let Some(b) = blend_width {
        let l_shape = cal
            .projection
            .l_shape_mut()
            .ok_or_else(|| anyhow::anyhow!("--blend requires a projection with a seam"))?;
        eprintln!(
            "Seam blend: --blend {b} overrides the calibration's {}",
            l_shape.blend_width
        );
        l_shape.blend_width = b;
    }

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
        (cal.framing.tilt as f32).to_degrees()
    };

    let decoder_cleanup = reco_io::DecoderCleanup::default();
    let inputs: Vec<_> = inputs
        .iter()
        .map(|p| reco_io::InputPath::parse_segments(p))
        .collect();
    let mut source =
        reco_io::adapters::FfmpegFileSource::open_inputs(&inputs, effective_sync, false)?;
    source.defer_cleanup(&decoder_cleanup);
    let info = source.info();
    let fps_rational = info.fps_rational.ok_or_else(|| {
        anyhow::anyhow!("video frame rate is unknown; timed preview is unavailable")
    })?;

    println!(
        "Preview: {}x{} input, {}x{} window",
        info.width, info.height, width, height
    );

    // Get the first frame for initial display
    let first = source
        .next_frame()?
        .ok_or_else(|| anyhow::anyhow!("videos have no frames"))?
        .images;

    let event_loop = EventLoop::new()?;

    let frame_duration = std::time::Duration::from_secs_f64(1.0 / info.fps);

    let initial_fov = FOV_DEFAULT;

    let total_frames = source.total_frames();
    let rig_roll = cal.framing.roll as f32;

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
        current_frame: first,

        pose: {
            use reco_control::pose_control::{PoseControl, PoseControlConfig};
            use reco_core::geometry::RenderPose;
            PoseControl::new(PoseControlConfig {
                wheel_fov_step_degrees: FOV_SCROLL_STEP,
                fov_min_degrees: FOV_MIN,
                fov_max_degrees: FOV_MAX,
                invert_drag_x: true,
                invert_drag_y: false,
                hotkey_fov_step_degrees: FOV_KEY_STEP,
                rest_pose: RenderPose {
                    yaw: 0.0,
                    pitch: 0.0,
                    fov_degrees: initial_fov,
                },
            })?
        },
        frame_count: 1,
        playing: false,
        needs_redraw: false,
        frame_duration,
        last_frame_time: Instant::now(),
        mouse_dragging: false,
        last_mouse_pos: None,
        rig_tilt: rig_tilt_degrees.to_radians(),
        rig_roll,
        clamp_enabled: true,
        failure: None,
        coverage_error_reported: false,
        interrupted: interrupted.clone(),
        recording: None,
        recording_path: None,
        recording_frames: 0,
        record_frame: false,
        fps: info.fps,
        fps_rational,
        total_frames,
        pending_seek: None,
    };

    event_loop.run_app(&mut app)?;
    if let Some(error) = app.failure {
        anyhow::bail!(error);
    }
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
    renderer: Option<StitchCore>,
    cal: reco_core::calibration::Calibration,
    input_width: u32,
    input_height: u32,
    width: u32,
    height: u32,
    current_frame: FrameSet,
    /// Requested and accepted render-space pose, with no temporal smoothing.
    pose: reco_control::pose_control::PoseControl,
    frame_count: u64,
    playing: bool,
    needs_redraw: bool,
    frame_duration: std::time::Duration,
    last_frame_time: Instant,
    // Mouse drag state
    mouse_dragging: bool,
    last_mouse_pos: Option<(f64, f64)>,
    rig_tilt: f32,
    rig_roll: f32,
    /// Whether coverage boundary clamping is active.
    clamp_enabled: bool,
    failure: Option<String>,
    coverage_error_reported: bool,
    /// Ctrl-C signal from the main thread.
    interrupted: Arc<AtomicBool>,
    // -- Recording state --
    recording: Option<Box<dyn OutputSink>>,
    recording_path: Option<String>,
    recording_frames: u64,
    record_frame: bool,
    fps: f64,
    fps_rational: (i32, i32),
    // -- Seek state --
    total_frames: Option<u64>,
    /// Coalesced seek target. Multiple rapid key presses accumulate here;
    /// only the final value is executed (in about_to_wait).
    pending_seek: Option<u64>,
}

impl App {
    fn advance_playback(&mut self) {
        if (self.playing || self.recording.is_some())
            && let Some(clock) =
                advance_playback_clock(self.last_frame_time, Instant::now(), self.frame_duration)
        {
            self.last_frame_time = clock;
            if self.playing {
                self.advance_frame();
            }
            if self.recording.is_some() {
                self.record_frame = true;
                self.needs_redraw = true;
            }
        }
    }

    fn start_recording(&mut self) {
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let path = format!("reco_recording_{epoch}.mp4");
        // NV12 requires width divisible by 4 and height divisible by 2.
        let Some(renderer) = &self.renderer else {
            return;
        };
        let (width, height) = renderer.output_dims();
        let rec_w = width & !3;
        let rec_h = height & !1;
        let encoder = match reco_io::adapters::FfmpegFileEncoder::new(
            Path::new(&path),
            rec_w,
            rec_h,
            self.fps_rational,
            &reco_io::ffmpeg::encoder::EncoderConfig::default(),
        ) {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("Failed to start recording: {e}");
                return;
            }
        };
        println!("[REC] Recording to {path} ({})", encoder.encoder_name());
        self.recording = Some(Box::new(encoder));
        self.recording_path = Some(path);
        self.recording_frames = 0;
        self.record_frame = true;
        self.needs_redraw = true;
    }

    fn stop_recording(&mut self) {
        self.record_frame = false;
        // Flush remaining NV12 frames from triple-buffer.
        if let Some(ref mut renderer) = self.renderer {
            let (width, height) = renderer.output_dims();
            for _ in 0..2 {
                if let Ok(Some(nv12)) = renderer.flush_nv12()
                    && let Some(ref mut enc) = self.recording
                {
                    let _ = enc.consume(OutputFrame {
                        data: nv12,
                        width: width & !3,
                        height: height & !1,
                        format: PixelFormat::Nv12,
                        pts_us: None,
                    });
                    self.recording_frames += 1;
                }
            }
        }
        if let Some(mut enc) = self.recording.take()
            && let Err(e) = enc.finish()
        {
            eprintln!("Failed to finalize recording: {e}");
        }
        if let Some(path) = self.recording_path.take() {
            println!("[STOP] Recorded {} frames to {path}", self.recording_frames);
        }
    }

    /// Seek and display the first frame at the new position (blocking).
    /// Only called from about_to_wait after coalescing rapid key presses.
    fn seek_to(&mut self, frame: u64) {
        let Some(ref mut source) = self.source else {
            return;
        };
        if let Err(e) = source.seek(frame) {
            eprintln!("Seek failed: {e}");
            return;
        }
        self.frame_count = frame;
        // Blocking: wait for the first decoded frame at the new position.
        self.step_frame();
        let secs = frame as f64 / self.fps;
        println!("Seeked to frame {frame} ({secs:.1}s)");
    }

    /// Accept an edited document before updating preview controls.
    fn edit_calibration(&mut self, edit: impl FnOnce(&mut reco_core::calibration::Calibration)) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let mut draft = renderer.calibration().clone();
        edit(&mut draft);
        if let Err(error) = renderer.update_calibration_preview(draft) {
            eprintln!("Calibration unchanged: {error}");
            return;
        }
        self.cal = renderer.calibration().clone();
        self.needs_redraw = true;
    }

    fn advance_frame(&mut self) {
        let Some(ref mut source) = self.source else {
            return;
        };
        match source.try_next_frame() {
            Ok(Some(frame)) => {
                self.current_frame = frame.images;
                self.frame_count += 1;
                self.needs_redraw = true;
            }
            Ok(None) => {
                if source.is_exhausted() {
                    self.playing = false;
                    println!("End of video");
                }
            }
            Err(e) => {
                self.playing = false;
                eprintln!("Playback failed: {e}");
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
                self.current_frame = frame.images;
                self.frame_count += 1;
                self.needs_redraw = true;
            }
            Ok(None) => {
                self.playing = false;
                println!("End of video");
            }
            Err(error) => {
                self.playing = false;
                eprintln!("Playback failed: {error}");
            }
        }
    }

    /// Resolve each input against the accepted pose, discarding blocked movement.
    fn apply_camera(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            renderer.set_constrained_look(self.clamp_enabled);
            match self.pose.constrain_preview(renderer) {
                Ok(()) => self.coverage_error_reported = false,
                Err(error) => {
                    if !self.coverage_error_reported {
                        eprintln!("Coverage unavailable; preview remains unconstrained: {error}");
                        self.coverage_error_reported = true;
                    }
                    self.pose.set_pose(self.pose.target_pose());
                }
            }
        }
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
        let render_format = reco_core::render::strip_srgb(surface_format);
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

        // Rig tilt/roll live on the calibration now; seed them from the
        // preview's resolved startup state before building the renderer.
        // (Blend was already resolved onto self.cal at load.)
        self.cal.framing.tilt = self.rig_tilt as f64;
        self.cal.framing.roll = self.rig_roll as f64;
        let viewport_size = ViewportSize {
            width: self.width,
            height: self.height,
        };

        let executor = GpuExecutor::new(
            gpu,
            GpuExecutorConfig {
                calibration: self.cal.clone(),
                viewport_size,
                input_width: self.input_width,
                input_height: self.input_height,
                input_format: reco_core::render::renderer::InputFormat::Yuv420p,
                output_format: reco_core::render::strip_srgb(surface_format),
                full_range: false,
            },
        )
        .expect("create executor");
        let renderer = StitchCore::new(Executor::Gpu(Box::new(executor))).expect("create engine");
        let initial_world = reco_core::geometry::WorldPose {
            fov_degrees: self.pose.target_pose().fov_degrees,
            ..Default::default()
        };
        self.pose
            .set_target(renderer.calibration().world_to_render_pose(initial_world));

        println!(
            "Preview ready: GPU = {}, format = {:?}",
            renderer.gpu().gpu_name(),
            surface_format
        );
        println!("Controls: Space = play/pause, N = next, P = skip 30, Q = quit");
        println!("          R = toggle recording, [/] = seek -/+ 5s, Home = restart");
        println!("          Arrows/drag = pan, +/-/scroll = zoom");
        println!("          3/4 = axis offset, C = toggle clamping, S = save calibration");
        if self.cal.projection.l_shape().is_some() {
            println!("          1/2 = intersect, 5/6 = vertical align, B = cycle blend width");
            println!("          7/8 = focal length, 9/0 = k1 distortion (barrel/pincushion)");
        }

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
                if self.recording.is_some() {
                    self.stop_recording();
                }
                // Drop decoder (NVDEC) before GPU pipeline to avoid
                // CUDA context teardown race on exit.
                self.source.take();
                event_loop.exit();
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                self.width = size.width;
                self.height = size.height;
                if let (Some(surface), Some(renderer)) = (&self.surface, &mut self.renderer) {
                    let render_format = reco_core::render::strip_srgb(self.surface_format);
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
                    self.needs_redraw = true;
                }
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state == winit::event::ElementState::Pressed {
                    if let PhysicalKey::Code(key) = event.physical_key
                        && let Some(intent) = pose_key_intent(key)
                    {
                        self.pose.apply_intent(intent);
                        self.needs_redraw = true;
                    } else {
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::Escape | KeyCode::KeyQ) => {
                                if self.recording.is_some() {
                                    self.stop_recording();
                                }
                                self.source.take();
                                event_loop.exit();
                            }
                            PhysicalKey::Code(KeyCode::Space) if !event.repeat => {
                                self.playing = !self.playing;
                                println!("{}", if self.playing { "Playing" } else { "Paused" });
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
                            PhysicalKey::Code(
                                key @ (KeyCode::Digit0
                                | KeyCode::Digit1
                                | KeyCode::Digit2
                                | KeyCode::Digit3
                                | KeyCode::Digit4
                                | KeyCode::Digit5
                                | KeyCode::Digit6
                                | KeyCode::Digit7
                                | KeyCode::Digit8
                                | KeyCode::Digit9
                                | KeyCode::KeyB),
                            ) => {
                                self.edit_calibration(|cal| {
                                    if key == KeyCode::Digit3 {
                                        cal.framing.axis_offset += 0.005;
                                        return;
                                    }
                                    if key == KeyCode::Digit4 {
                                        cal.framing.axis_offset -= 0.005;
                                        return;
                                    }
                                    let Some(layout) = cal.projection.l_shape_mut() else {
                                        return;
                                    };
                                    match key {
                                        KeyCode::Digit1 => {
                                            layout.intersect = (layout.intersect + 0.01).min(1.0)
                                        }
                                        KeyCode::Digit2 => {
                                            layout.intersect = (layout.intersect - 0.01).max(0.0)
                                        }
                                        KeyCode::Digit5 => layout.x_ty += 0.005,
                                        KeyCode::Digit6 => layout.x_ty -= 0.005,
                                        KeyCode::KeyB => {
                                            layout.blend_width = if layout.blend_width >= 0.19 {
                                                0.0
                                            } else {
                                                layout.blend_width + 0.05
                                            };
                                        }
                                        KeyCode::Digit7 | KeyCode::Digit8 => {
                                            let scale =
                                                if key == KeyCode::Digit7 { 1.02 } else { 0.98 };
                                            for lens in &mut cal.lenses {
                                                lens.fx *= scale;
                                                lens.fy *= scale;
                                            }
                                        }
                                        KeyCode::Digit9 | KeyCode::Digit0 => {
                                            let delta = if key == KeyCode::Digit9 {
                                                0.005
                                            } else {
                                                -0.005
                                            };
                                            for lens in &mut cal.lenses {
                                                lens.distortion[0] += delta;
                                            }
                                        }
                                        _ => unreachable!(),
                                    }
                                });
                                println!(
                                    "projection: {:?}, camera_axis_offset: {:.4}",
                                    self.cal.projection, self.cal.framing.axis_offset,
                                );
                            }
                            PhysicalKey::Code(KeyCode::KeyC) if !event.repeat => {
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
                            PhysicalKey::Code(KeyCode::KeyR) if !event.repeat => {
                                if self.recording.is_some() {
                                    self.stop_recording();
                                } else {
                                    self.start_recording();
                                }
                            }
                            PhysicalKey::Code(KeyCode::BracketLeft) => {
                                let step = (SEEK_STEP_SECS * self.fps) as u64;
                                let base = self.pending_seek.unwrap_or(self.frame_count);
                                self.pending_seek = Some(base.saturating_sub(step));
                            }
                            PhysicalKey::Code(KeyCode::BracketRight) => {
                                let step = (SEEK_STEP_SECS * self.fps) as u64;
                                let max = self.total_frames.unwrap_or(u64::MAX);
                                let base = self.pending_seek.unwrap_or(self.frame_count);
                                self.pending_seek = Some((base + step).min(max));
                            }
                            PhysicalKey::Code(KeyCode::Home) => {
                                self.pending_seek = Some(0);
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                use winit::event::ElementState;
                use winit::event::MouseButton;
                if button == MouseButton::Left {
                    let pressed = state == ElementState::Pressed;
                    self.mouse_dragging = pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_dragging
                    && let Some((prev_x, prev_y)) = self.last_mouse_pos
                {
                    let dx = (position.x - prev_x) as f32;
                    let dy = (position.y - prev_y) as f32;
                    self.pose.apply_drag(dx, dy, self.height as f32);
                    self.needs_redraw = true;
                }
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::Focused(false) | WindowEvent::CursorLeft { .. } => {
                self.mouse_dragging = false;
                self.last_mouse_pos = None;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 30.0) as f32,
                };
                self.pose.apply_wheel(scroll);
                self.needs_redraw = true;
            }
            WindowEvent::RedrawRequested => {
                if !self.needs_redraw {
                    return;
                }
                self.apply_camera();
                let Some(surface) = self.surface.as_ref() else {
                    return;
                };

                let frame = match surface.get_current_texture() {
                    Ok(f) => f,
                    Err(e) => {
                        log::warn!("Surface error: {e:?}");
                        return;
                    }
                };
                // Surface acquisition can wait for vsync. Select a due video
                // frame after that wait rather than presenting stale content.
                self.advance_playback();
                self.needs_redraw = false;
                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                let render_format = reco_core::render::strip_srgb(self.surface_format);
                let view = frame
                    .texture
                    .create_view(&reco_core::wgpu::TextureViewDescriptor {
                        format: Some(render_format),
                        ..Default::default()
                    });

                let render = self.pose.current_pose();
                if let Err(e) = renderer.render_to_view(&self.current_frame, render, &view) {
                    log::error!("Render failed: {e}");
                    return;
                }

                // Record: render to internal target + NV12 readback.
                if self.recording.is_some() && self.record_frame {
                    self.record_frame = false;
                    let (width, height) = renderer.output_dims();
                    match renderer.render_and_readback_nv12(&self.current_frame, render) {
                        Ok(Some(nv12)) => {
                            if let Some(ref mut enc) = self.recording
                                && let Err(e) = enc.consume(OutputFrame {
                                    data: nv12,
                                    width: width & !3,
                                    height: height & !1,
                                    format: PixelFormat::Nv12,
                                    pts_us: None,
                                })
                            {
                                eprintln!("Recording encode error: {e}");
                                self.stop_recording();
                            }
                            self.recording_frames += 1;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            eprintln!("Recording readback error: {e}");
                            self.stop_recording();
                        }
                    }
                }

                frame.present();
            }
            _ => {}
        }

        if self.needs_redraw {
            self.apply_camera();
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Check Ctrl-C
        if self.interrupted.load(Ordering::Relaxed) {
            if self.recording.is_some() {
                self.stop_recording();
            }
            self.source.take();
            event_loop.exit();
            return;
        }

        // Execute coalesced seek: rapid key presses accumulate into
        // pending_seek; only the final target runs here (one seek
        // instead of many). Blocking: waits for first frame to render.
        if let Some(target) = self.pending_seek.take() {
            self.seek_to(target);
            self.needs_redraw = true;
        }

        let mut resizing = false;
        if self.recording.is_none()
            && let Some(renderer) = &mut self.renderer
        {
            let before = renderer.output_dims();
            match renderer.resize_preview(self.width, self.height) {
                Ok(()) => {
                    resizing = self.clamp_enabled && renderer.coverage_max_fov_degrees().is_none();
                    if renderer.output_dims() != before {
                        self.needs_redraw = true;
                    }
                }
                Err(error) => {
                    self.failure = Some(error.to_string());
                    self.source.take();
                    event_loop.exit();
                    return;
                }
            }
        }

        if resizing {
            let before = self.pose.current_pose();
            self.apply_camera();
            self.needs_redraw |= self.pose.current_pose() != before;
            resizing = !self.coverage_error_reported;
        }

        if self.playing || self.recording.is_some() {
            self.advance_playback();
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                self.last_frame_time + self.frame_duration,
            ));
        } else if resizing {
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + std::time::Duration::from_millis(16),
            ));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }

        if self.needs_redraw
            && let Some(w) = &self.window
        {
            w.request_redraw();
        }
    }
}

fn pose_key_intent(key: winit::keyboard::KeyCode) -> Option<ControlIntent> {
    use winit::keyboard::KeyCode;
    match key {
        KeyCode::ArrowLeft => Some(ControlIntent::DeltaYawRad(ARROW_PAN_STEP)),
        KeyCode::ArrowRight => Some(ControlIntent::DeltaYawRad(-ARROW_PAN_STEP)),
        KeyCode::ArrowUp => Some(ControlIntent::DeltaPitchRad(ARROW_PAN_STEP)),
        KeyCode::ArrowDown => Some(ControlIntent::DeltaPitchRad(-ARROW_PAN_STEP)),
        KeyCode::Equal | KeyCode::NumpadAdd => Some(ControlIntent::ZoomIn),
        KeyCode::Minus | KeyCode::NumpadSubtract => Some(ControlIntent::ZoomOut),
        _ => None,
    }
}

fn advance_playback_clock(
    previous: Instant,
    now: Instant,
    period: std::time::Duration,
) -> Option<Instant> {
    // Redraw delays must not accumulate into slower playback. Rebase after
    // a missed whole period so pauses and stalls cannot cause catch-up bursts.
    let next = previous + period;
    if now < next {
        return None;
    }
    if now.saturating_duration_since(next) >= period {
        Some(now)
    } else {
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn keyboard_pose_bindings_preserve_signs_and_zoom_steps() {
        use reco_control::pose_control::{PoseControl, PoseControlConfig};
        use winit::keyboard::KeyCode;
        for (key, yaw, pitch, fov) in [
            (KeyCode::ArrowLeft, 0.05, 0.0, 75.0),
            (KeyCode::ArrowRight, -0.05, 0.0, 75.0),
            (KeyCode::ArrowUp, 0.0, 0.05, 75.0),
            (KeyCode::ArrowDown, 0.0, -0.05, 75.0),
            (KeyCode::Equal, 0.0, 0.0, 70.0),
            (KeyCode::NumpadAdd, 0.0, 0.0, 70.0),
            (KeyCode::Minus, 0.0, 0.0, 80.0),
            (KeyCode::NumpadSubtract, 0.0, 0.0, 80.0),
        ] {
            let mut control = PoseControl::new(PoseControlConfig {
                hotkey_fov_step_degrees: FOV_KEY_STEP,
                ..Default::default()
            })
            .unwrap();
            control.apply_intent(pose_key_intent(key).unwrap());
            let pose = control.target_pose();
            assert_eq!((pose.yaw, pose.pitch, pose.fov_degrees), (yaw, pitch, fov));
        }
        for key in [KeyCode::KeyC, KeyCode::KeyR, KeyCode::Space] {
            assert!(pose_key_intent(key).is_none());
        }
    }

    #[test]
    fn redraw_jitter_does_not_accumulate_over_900_frames() {
        let start = Instant::now();
        let period = Duration::from_micros(33_367);
        let mut clock = start;
        for frame in 1..=900 {
            let scheduled = start + period * frame;
            let redraw_delay = Duration::from_millis(u64::from(frame % 17));
            clock = advance_playback_clock(clock, scheduled + redraw_delay, period).unwrap();
            assert_eq!(clock, scheduled);
        }
    }

    #[test]
    fn missed_periods_rebase_without_a_catch_up_burst() {
        let start = Instant::now();
        let period = Duration::from_millis(33);
        for delay in [period * 2, Duration::from_secs(60)] {
            let resumed = start + delay;
            let clock = advance_playback_clock(start, resumed, period).unwrap();
            assert_eq!(clock, resumed);
            let next = advance_playback_clock(clock, resumed + period, period).unwrap();
            assert_eq!(next, resumed + period);
        }
    }

    #[test]
    fn display_wait_crosses_deadline_without_advancing_twice() {
        let start = Instant::now();
        let period = Duration::from_millis(33);
        assert_eq!(
            advance_playback_clock(start, start + Duration::from_millis(25), period),
            None
        );
        let after_display_wait = start + Duration::from_millis(40);
        let clock = advance_playback_clock(start, after_display_wait, period).unwrap();
        assert_eq!(clock, start + period);
        assert_eq!(
            advance_playback_clock(clock, after_display_wait, period),
            None
        );
        assert_eq!(
            advance_playback_clock(clock, start + period * 2, period),
            Some(start + period * 2)
        );
    }
}
