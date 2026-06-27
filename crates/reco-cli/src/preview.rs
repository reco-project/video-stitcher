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
//! `StitchRenderer` in reco-core. What remains here is the winit event loop,
//! surface management, input handling, and frame pacing - all tightly coupled
//! to the desktop window environment and not useful to library consumers
//! (GUI, OBS, cloud). A future GUI app would use its own event loop and call
//! the same `StitchRenderer` API directly.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use reco_control::pose_control::HotkeyIntent;
use reco_control::{ControlIntent, IntentTranslator, PoseIntent};
use reco_core::encoder::{Encoder, OutputFrame, PixelFormat};
use reco_core::render::stitch_renderer::StitchRenderer;
use reco_core::source::{FrameSource, YuvData};
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
const FOV_MIN: f32 = 20.0;
/// Maximum FOV (degrees) - prevents extreme zoom-out.
const FOV_MAX: f32 = 150.0;
/// Default FOV at startup (degrees).
const FOV_DEFAULT: f32 = 75.0;
/// Number of frames to skip on P key press.
const FRAME_SKIP_COUNT: usize = 30;
/// Number of seconds to seek on `[`/`]` key press.
const SEEK_STEP_SECS: f64 = 5.0;

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
    let cal = reco_core::calibration::Calibration::from_file(Path::new(calibration_path))?;

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

    let mut source = reco_io::adapters::FfmpegFileSource::open_with_offset(
        Path::new(left_path),
        Path::new(right_path),
        effective_sync,
    )?;
    let info = source.info();

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
        let scene =
            reco_core::render::scene::SceneGeometry::from_layout_with_aspect(&cal.layout, aspect);
        let coverage = reco_core::projection::CoverageBoundary::from_calibration(&cal, &scene);
        coverage.max_fov_degrees().min(FOV_MAX)
    };
    let initial_fov = FOV_DEFAULT;
    log::info!("Preview: max FOV = {max_fov:.1} degrees (coverage-limited)");

    let fps_rational = info.fps_rational.unwrap_or((30, 1));
    let total_frames = source.total_frames();
    let rig_roll = cal.rig_roll as f32;

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

        pose: {
            use reco_control::pose_control::{PoseControl, PoseControlConfig};
            use reco_core::detect::director::ViewportPosition;
            // Match preview's historical feel: 0.005 rad/px drag,
            // 0.05 rad arrow step, 3.0 deg scroll step, 0.3 smoothing.
            // `invert_drag_x = true` reproduces preview's "drag right
            // increases yaw" convention (opposite of PoseControl default).
            PoseControl::new(PoseControlConfig {
                drag_deg_per_pixel: 0.005_f32.to_degrees(),
                wheel_fov_per_tick: FOV_SCROLL_STEP,
                smoothing: 0.3,
                fov_min_degrees: FOV_MIN,
                fov_max_degrees: max_fov.min(FOV_MAX),
                invert_drag_x: true,
                invert_drag_y: false,
                hotkey_yaw_step_rad: ARROW_PAN_STEP,
                hotkey_pitch_step_rad: ARROW_PAN_STEP,
                hotkey_fov_step_deg: FOV_KEY_STEP,
                rest_pose: ViewportPosition {
                    yaw: 0.0,
                    pitch: 0.0,
                    fov_degrees: Some(initial_fov),
                },
            })
        },
        frame_count: 1,
        playing: false,
        needs_redraw: false,
        frame_duration,
        last_frame_time: Instant::now(),
        mouse_dragging: false,
        last_mouse_pos: None,
        blend_width,
        rig_tilt: rig_tilt_degrees.to_radians(),
        rig_roll,
        max_fov,
        clamp_enabled: false,
        interrupted: interrupted.clone(),
        recording: None,
        recording_path: None,
        recording_frames: 0,
        fps: info.fps,
        fps_rational,
        total_frames,
        pending_seek: None,
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
    cal: reco_core::calibration::Calibration,
    input_width: u32,
    input_height: u32,
    width: u32,
    height: u32,
    current_left: YuvData,
    current_right: YuvData,
    /// Unified pose state (target + current yaw/pitch/FOV with smoothing).
    /// Drives every pan / zoom input; the renderer's pitch gets a
    /// Model 3 compensation via `pose.render_pose(rig_tilt)` so the
    /// horizon stays level as yaw changes. Replaced the hand-rolled
    /// (target_yaw, target_pitch, target_fov, yaw, pitch) state
    /// machine 2026-04-20.
    pose: reco_control::pose_control::PoseControl,
    frame_count: u64,
    playing: bool,
    needs_redraw: bool,
    frame_duration: std::time::Duration,
    last_frame_time: Instant,
    // Mouse drag state
    mouse_dragging: bool,
    last_mouse_pos: Option<(f64, f64)>,
    blend_width: f32,
    rig_tilt: f32,
    rig_roll: f32,
    /// Maximum FOV from coverage (cached from coverage.max_fov_degrees()).
    max_fov: f32,
    /// Whether coverage boundary clamping is active.
    clamp_enabled: bool,
    /// Ctrl-C signal from the main thread.
    interrupted: Arc<AtomicBool>,
    // -- Recording state --
    recording: Option<Box<dyn Encoder>>,
    recording_path: Option<String>,
    recording_frames: u64,
    fps: f64,
    fps_rational: (i32, i32),
    // -- Seek state --
    total_frames: Option<u64>,
    /// Coalesced seek target. Multiple rapid key presses accumulate here;
    /// only the final value is executed (in about_to_wait).
    pending_seek: Option<u64>,
}

impl App {
    fn start_recording(&mut self) {
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let path = format!("reco_recording_{epoch}.mp4");
        // NV12 requires width divisible by 4 and height divisible by 2.
        let rec_w = self.width & !3;
        let rec_h = self.height & !1;
        let (encoder, enc_name) = match reco_io::adapters::create_encoder(
            Path::new(&path),
            rec_w,
            rec_h,
            self.fps_rational,
            "h264",
            "balanced",
            None,
            None,
            None,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("Failed to start recording: {e}");
                return;
            }
        };
        println!("[REC] Recording to {path} ({enc_name})");
        self.recording = Some(Box::new(encoder));
        self.recording_path = Some(path);
        self.recording_frames = 0;
    }

    fn stop_recording(&mut self) {
        // Flush remaining NV12 frames from triple-buffer.
        if let Some(ref mut renderer) = self.renderer {
            for _ in 0..2 {
                if let Ok(Some(nv12)) = renderer.flush_nv12()
                    && let Some(ref mut enc) = self.recording
                {
                    let _ = enc.submit(OutputFrame {
                        data: nv12,
                        width: self.width & !3,
                        height: self.height & !1,
                        format: PixelFormat::Nv12,
                        pts_us: (self.recording_frames as f64 / self.fps * 1_000_000.0) as i64,
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

    /// Apply a calibration change: update renderer + refresh max_fov from new coverage.
    fn apply_calibration_change(&mut self) {
        if let Some(ref mut r) = self.renderer {
            r.update_calibration(self.cal.clone());
            self.max_fov = r.coverage().max_fov_degrees().min(FOV_MAX);
            if self.clamp_enabled {
                // Narrow PoseControl's FOV ceiling so the target FOV
                // can't exceed what coverage allows.
                let mut cfg = *self.pose.config();
                cfg.fov_max_degrees = self.max_fov;
                self.pose.set_config(cfg);
                self.pose.set_target_fov(
                    self.pose
                        .target_pose()
                        .fov_degrees
                        .unwrap_or(self.max_fov)
                        .min(self.max_fov),
                );
            }
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

    /// Step PoseControl one tick + optional coverage clamp + push
    /// the resolved FOV onto the pipeline. Returns `true` if the
    /// pose moved (needs redraw). Replaces the hand-rolled
    /// target/current lerp 2026-04-20.
    fn smooth_camera(&mut self) -> bool {
        const EPSILON: f32 = 0.0001;
        const FOV_EPSILON: f32 = 0.01;

        let before = self.pose.current_pose();
        self.pose.tick();

        if self.clamp_enabled
            && let Some(ref renderer) = self.renderer
        {
            let coverage = renderer.coverage();
            let aspect = self.width as f32 / self.height as f32;
            self.pose
                .clamp_via_coverage(coverage, aspect, self.rig_tilt);
        }

        if let Some(r) = &mut self.renderer {
            let target_fov = self.pose.current_fov_deg();
            r.pipeline_mut().set_fov(target_fov.clamp(1.0, FOV_MAX));
        }

        let after = self.pose.current_pose();
        let dy = (after.yaw - before.yaw).abs();
        let dp = (after.pitch - before.pitch).abs();
        let df = after.fov_degrees.unwrap_or(0.0) - before.fov_degrees.unwrap_or(0.0);
        dy > EPSILON || dp > EPSILON || df.abs() > FOV_EPSILON
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

        let viewport = reco_core::render::viewport::ViewportConfig {
            width: self.width,
            height: self.height,
            blend_width: self.blend_width,
            rig_tilt: self.rig_tilt,
            rig_roll: self.rig_roll,
            ..Default::default()
        };

        let renderer = StitchRenderer::new(
            self.cal.clone(),
            gpu,
            viewport,
            self.input_width,
            self.input_height,
            surface_format,
            reco_core::render::renderer::InputFormat::Yuv420p,
        )
        .expect("create renderer");

        println!(
            "Preview ready: GPU = {}, format = {:?}",
            renderer.gpu().gpu_name(),
            surface_format
        );
        println!("Controls: Space = play/pause, N = next, P = skip 30, Q = quit");
        println!("          R = toggle recording, [/] = seek -/+ 5s, Home = restart");
        println!("          Arrows/drag = pan, +/-/scroll = zoom");
        println!("          1/2 = intersect, 3/4 = axis offset, 5/6 = vertical align");
        println!("          7/8 = focal length, 9/0 = k1 distortion (barrel/pincushion)");
        println!("          B = cycle blend width, C = toggle clamping, S = save calibration");

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
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                if event.state == winit::event::ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape | KeyCode::KeyQ) => {
                            if self.recording.is_some() {
                                self.stop_recording();
                            }
                            self.source.take();
                            event_loop.exit();
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            IntentTranslator::new(&mut self.pose).dispatch(ControlIntent::Pose(
                                PoseIntent::DeltaYawRad(ARROW_PAN_STEP),
                            ));
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            IntentTranslator::new(&mut self.pose).dispatch(ControlIntent::Pose(
                                PoseIntent::DeltaYawRad(-ARROW_PAN_STEP),
                            ));
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            IntentTranslator::new(&mut self.pose).dispatch(ControlIntent::Pose(
                                PoseIntent::DeltaPitchRad(ARROW_PAN_STEP),
                            ));
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            IntentTranslator::new(&mut self.pose).dispatch(ControlIntent::Pose(
                                PoseIntent::DeltaPitchRad(-ARROW_PAN_STEP),
                            ));
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
                            IntentTranslator::new(&mut self.pose)
                                .dispatch(ControlIntent::Hotkey(HotkeyIntent::ZoomIn));
                            self.needs_redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::Minus | KeyCode::NumpadSubtract) => {
                            IntentTranslator::new(&mut self.pose)
                                .dispatch(ControlIntent::Hotkey(HotkeyIntent::ZoomOut));
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
                        PhysicalKey::Code(KeyCode::KeyR) => {
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
            WindowEvent::CursorMoved { position, .. } if self.mouse_dragging => {
                if let Some((prev_x, prev_y)) = self.last_mouse_pos {
                    let dx = (position.x - prev_x) as f32;
                    let dy = (position.y - prev_y) as f32;
                    self.pose.apply_drag(dx, dy);
                } else {
                    // First move after click - switch to Poll for smooth updates
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
                self.last_mouse_pos = Some((position.x, position.y));
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
                // Apply camera smoothing before rendering
                if self.smooth_camera() {
                    self.needs_redraw = true;
                }
                if !self.needs_redraw && !self.playing {
                    return;
                }
                self.needs_redraw = false;

                let (Some(surface), Some(renderer)) =
                    (self.surface.as_ref(), self.renderer.as_mut())
                else {
                    return;
                };

                let frame = match surface.get_current_texture() {
                    Ok(f) => f,
                    Err(e) => {
                        log::warn!("Surface error: {e:?}");
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

                let left = self.current_left.as_planes();
                let right = self.current_right.as_planes();
                // PoseControl::render_pose applies the Model 3
                // rig_tilt yaw-pitch coupling so the horizon stays
                // level as yaw changes. See pose_control.rs docs.
                let render = self.pose.render_pose(self.rig_tilt);
                let (render_yaw, render_pitch) = (render.yaw, render.pitch);
                if let Err(e) = renderer.render_yuv(&left, &right, render_yaw, render_pitch, &view)
                {
                    log::error!("Render failed: {e}");
                    return;
                }

                // Record: render to internal target + NV12 readback.
                if self.recording.is_some() {
                    match renderer.render_and_readback_nv12(&left, &right, render_yaw, render_pitch)
                    {
                        Ok(Some(nv12)) => {
                            let pts_us =
                                (self.recording_frames as f64 / self.fps * 1_000_000.0) as i64;
                            if let Some(ref mut enc) = self.recording
                                && let Err(e) = enc.submit(OutputFrame {
                                    data: nv12,
                                    width: self.width & !3,
                                    height: self.height & !1,
                                    format: PixelFormat::Nv12,
                                    pts_us,
                                })
                            {
                                eprintln!("Recording encode error: {e}");
                                self.stop_recording();
                            }
                            self.recording_frames += 1;
                        }
                        Ok(None) => {
                            // First 2 frames: triple-buffer warming up.
                            self.recording_frames += 1;
                        }
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

        if self.needs_redraw
            && let Some(w) = &self.window
        {
            w.request_redraw();
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

        // Keep animating while PoseControl smoothing hasn't converged.
        let target = self.pose.target_pose();
        let current = self.pose.current_pose();
        let smoothing_active = (target.yaw - current.yaw).abs() > 0.0001
            || (target.pitch - current.pitch).abs() > 0.0001
            || (target.fov_degrees.unwrap_or(0.0) - current.fov_degrees.unwrap_or(0.0)).abs()
                > 0.01;

        // Execute coalesced seek: rapid key presses accumulate into
        // pending_seek; only the final target runs here (one seek
        // instead of many). Blocking: waits for first frame to render.
        if let Some(target) = self.pending_seek.take() {
            self.seek_to(target);
            self.needs_redraw = true;
        }

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
