//! Live camera stitching via GStreamer.
//!
//! Captures stereo camera feeds, stitches them on the GPU, and encodes
//! the panoramic output to a video file in real time. Supports optional
//! YOLO ball detection and auto-tracking via the director pipeline.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_core::source::StereoFrame;
use reco_io::gstreamer::camera::CameraConfig;

use crate::helpers;

/// Run live camera stitching.
///
/// Captures frames from two cameras via GStreamer, stitches them into a
/// panoramic view on the GPU, and encodes the result to `output`. Uses
/// NV12 native capture on Jetson (Tegra) and I420 elsewhere.
///
/// When `model_path` is provided, sets up YOLO ball detection with
/// EKF tracking and a ball-following director for automatic panning.
/// Configuration for live camera stitching.
pub struct CameraRunConfig<'a> {
    pub cam_config: CameraConfig,
    pub calibration: &'a str,
    pub output: &'a str,
    pub width: u32,
    pub height: u32,
    pub blend: f32,
    pub encoder_name: Option<String>,
    pub codec: &'a str,
    pub quality: &'a str,
    pub duration: Option<f64>,
    pub max_frames: Option<u64>,
    pub capture_fps: u32,
    pub model_path: Option<&'a str>,
    pub detection_interval: u64,
    pub crf: Option<u8>,
    pub preset: Option<String>,
    /// Output container (`mp4` / `fmp4` / `mkv`). None -> `mp4`.
    /// `mkv` recommended for live recording (crash-safe, streamable).
    pub container: Option<&'a str>,
    /// Tracking director: `ball`, `field`, `sweep`. Sweep mode
    /// bypasses detection entirely (no --model needed).
    pub tracking: &'a str,
    /// Disable coverage-boundary clamp on the director output.
    /// Useful in sweep mode to cover the full panorama width.
    pub unconstrained: bool,
    /// Optional path for M7 stacked-replay recording. Same feature
    /// as `StitchJob::with_replay_recording`. Requires the `replay`
    /// feature flag on reco-cli.
    pub replay_path: Option<&'a str>,
    /// Optional downscaled replay tile dims `(width, height)`.
    /// GPU-pack path only; no-op when replay_path is None.
    pub replay_scale: Option<(u32, u32)>,
    /// RTMP stream URL for simultaneous file + stream output.
    pub stream_url: Option<&'a str>,
    /// Use NVMM zero-copy capture (DMA-buf + NvBufSurfTransform).
    /// Auto-enabled on Tegra when NvBufSurfTransform is available.
    pub use_nvmm: bool,
    /// Use V4L2 direct capture with raw Bayer + GPU demosaic.
    pub v4l2_direct: bool,
    /// Sensor exposure in microseconds (V4L2 direct only).
    pub exposure: u32,
    /// Sensor analog gain (V4L2 direct only).
    pub sensor_gain: u32,
}

pub fn run_camera(
    config: CameraRunConfig<'_>,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let CameraRunConfig {
        cam_config,
        calibration,
        output,
        width,
        height,
        blend,
        encoder_name,
        codec,
        quality,
        duration,
        max_frames,
        capture_fps,
        model_path,
        detection_interval,
        crf,
        preset,
        container,
        tracking,
        unconstrained,
        replay_path,
        replay_scale,
        stream_url,
        use_nvmm,
        v4l2_direct,
        exposure: _exposure,
        sensor_gain: _sensor_gain,
    } = config;
    // Reject FFmpeg network URLs as output to prevent data exfiltration (#64).
    anyhow::ensure!(
        !output.contains("://"),
        "Output path looks like a network URL ({output}). Only local file paths are supported.",
    );

    let cal = reco_core::calibration::MatchCalibration::from_file(Path::new(calibration))?;
    let field_roi = cal.field_roi.clone();

    let viewport = reco_core::viewport::ViewportConfig {
        width,
        height,
        blend_width: blend,
        rig_tilt: cal.rig_tilt as f32,
        rig_roll: cal.rig_roll as f32,
        ..Default::default()
    };

    let gpu = reco_core::gpu::GpuContext::new_blocking()?;

    let (use_nv12_capture, input_format) = if v4l2_direct {
        // V4L2 direct: raw Bayer -> GPU demosaic -> RGBA -> stitch via BGRA path.
        (true, reco_core::renderer::InputFormat::Bgra)
    } else if use_nvmm || helpers::is_tegra() {
        (true, reco_core::renderer::InputFormat::Nv12)
    } else {
        (false, reco_core::renderer::InputFormat::Yuv420p)
    };

    let capture_width = cam_config.width;
    let capture_height = cam_config.height;

    let session_config = reco_core::session::SessionConfig {
        calibration: cal,
        viewport,
        input_width: capture_width,
        input_height: capture_height,
        output_format: reco_core::gpu::OutputFormat::Rgba8Unorm,
        input_format,
        left_rotation: 0,
        right_rotation: 0,
    };
    let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

    // Constrained-look clamp (FRICTION A13). Default on; `--unconstrained`
    // flips it off so sweep / debug views can pan past the coverage
    // boundary.
    if unconstrained {
        session.core_mut().set_constrained_look(false);
        log::info!("constrained_look: disabled (unconstrained viewport)");
    }

    // Parse tracking mode. Sweep is useful without detection.
    #[cfg(feature = "autocam")]
    let tracking_mode = match tracking {
        "ball" => reco_autocam::TrackingMode::Ball,
        "field" => reco_autocam::TrackingMode::Field,
        "sweep" => reco_autocam::TrackingMode::Sweep,
        other => {
            log::warn!("unknown tracking mode '{other}', defaulting to 'ball'");
            reco_autocam::TrackingMode::Ball
        }
    };

    // Set up autocam (detector + director). Model path is optional in
    // sweep mode — SweepDirector needs no detector.
    #[cfg(feature = "autocam")]
    {
        let effective_model = if tracking_mode == reco_autocam::TrackingMode::Sweep {
            // Sweep bypasses detection entirely.
            model_path.unwrap_or("")
        } else {
            model_path.unwrap_or("")
        };
        if !effective_model.is_empty() || tracking_mode == reco_autocam::TrackingMode::Sweep {
            let autocam_config = reco_autocam::AutocamConfig::new(effective_model)
                .with_tracking_mode(tracking_mode)
                .with_detection_interval(detection_interval);
            let autocam_config = if let Some(roi) = field_roi.as_ref() {
                autocam_config.with_field_roi(roi.clone())
            } else {
                autocam_config
            };
            match reco_autocam::setup_autocam(
                &mut session,
                &autocam_config,
                capture_fps as f32,
                use_nvmm,
            ) {
                Ok(true) => println!("Autocam: {tracking_mode:?} director attached"),
                Ok(false) => {
                    eprintln!("Warning: tracking unavailable in current capture mode")
                }
                Err(e) => {
                    eprintln!("Warning: autocam setup failed ({e}), continuing without tracking")
                }
            }
        }
    }
    #[cfg(not(feature = "autocam"))]
    if model_path.is_some() {
        log::warn!("--model specified but autocam feature is disabled");
    }

    let mode_str = if v4l2_direct {
        "Bayer RGGB (V4L2 direct)"
    } else if use_nvmm {
        "NVMM zero-copy (DMA-buf + NvBufSurfTransform)"
    } else if use_nv12_capture {
        "NV12"
    } else {
        "I420"
    };
    println!(
        "Pipeline ready: GPU = {}, capture = {}x{}@{}fps ({}), output = {}x{}",
        session.gpu_name(),
        capture_width,
        capture_height,
        capture_fps,
        mode_str,
        width,
        height
    );

    // M7 replay recording on live cams (closes #273). Live capture
    // runs through `session.process_frame` → CPU-upload render
    // path; the GPU pack tap in `process_frame` reads from the
    // renderer's internal plane textures (populated by
    // `queue.write_texture` on each frame), so replay works
    // regardless of whether the source is NV12 or I420.
    #[cfg(feature = "replay")]
    let _replay_attached = if let Some(replay_path) = replay_path {
        let (out_w, out_h, is_scaled) = if let Some((w, h)) = replay_scale {
            (w, h, true)
        } else if capture_width > 1920 {
            let scale = 1920.0 / capture_width as f64;
            let h = ((capture_height as f64 * scale) / 2.0).round() as u32 * 2;
            log::info!(
                "Replay auto-downscale: {}x{} -> 1920x{} (--replay-scale overrides)",
                capture_width,
                capture_height,
                h,
            );
            (1920, h, true)
        } else {
            (capture_width, capture_height, false)
        };
        let layout = reco_core::yuv_stack_packer::StackGridLayout::vstack(out_w, out_h, 2)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "replay tile dims {out_w}x{out_h} not YUV420P-aligned \
                     (width divisible by 4, height even)"
                )
            })?;
        let output_size = if is_scaled {
            reco_core::yuv_stack_packer::OutputTileSize::scaled(out_w, out_h)
        } else {
            reco_core::yuv_stack_packer::OutputTileSize::unscaled(out_w, out_h)
        };
        session
            .enable_gpu_stacked_replay(layout, output_size)
            .map_err(|e| anyhow::anyhow!("enable GPU stacked replay: {e}"))?;
        let (atlas_w, atlas_h) = session
            .stacked_atlas_dims()
            .ok_or_else(|| anyhow::anyhow!("stacked_atlas_dims returned None after enable"))?;
        // Jetson has no NVENC; libx264 default is what we have. On
        // non-Jetson the same default applies per session discussion
        // (NVENC + NV12 pack shader combo deferred to #271).
        let encoder_config = reco_io::stacked_video::encoder::StackedEncoderConfig {
            fps: (capture_fps as i32, 1),
            ..Default::default()
        };
        let recorder = reco_io::stacked_video::replay::session_gpu_recorder(
            std::path::Path::new(replay_path),
            encoder_config,
            atlas_w,
            atlas_h,
        )
        .map_err(|e| anyhow::anyhow!("open GPU replay recorder: {e}"))?;
        session.set_stacked_gpu_recorder(recorder);
        let scale_note = if replay_scale.is_some() {
            format!(
                " [A19 downscale: source {}x{} -> tile {}x{}]",
                capture_width, capture_height, out_w, out_h
            )
        } else {
            String::new()
        };
        log::info!(
            "reco-cli: camera replay pack path = GPU (tile {}x{}, N=2, atlas {}x{}){} -> {}",
            out_w,
            out_h,
            atlas_w,
            atlas_h,
            scale_note,
            replay_path,
        );
        println!("Replay recording: {replay_path}");
        true
    } else {
        false
    };
    #[cfg(not(feature = "replay"))]
    {
        let _ = replay_path;
        let _ = replay_scale;
        if replay_path.is_some() {
            log::warn!("--replay specified but `replay` feature is disabled.");
        }
    }

    reco_io::init();
    let out_quality: reco_io::output::Quality = quality.parse().unwrap_or_else(|_| {
        log::warn!("Unknown quality '{quality}', defaulting to balanced");
        reco_io::output::Quality::Balanced
    });
    let out_codec: reco_io::output::Codec = codec.parse().unwrap_or_else(|_| {
        log::warn!("Unknown codec '{codec}', defaulting to H.264");
        reco_io::output::Codec::H264
    });
    let out_format: reco_io::output::Format = if let Some(c) = container {
        c.parse().map_err(|e: String| {
            anyhow::anyhow!("{e} (expected mp4, fmp4, mkv, mov, or flv)")
        })?
    } else {
        reco_io::output::Format::default()
    };
    let enc_config = reco_io::ffmpeg::encoder::EncoderConfig {
        encoder_name,
        codec: out_codec.into(),
        quality: out_quality.into(),
        crf,
        preset,
        container: out_format.into(),
        gop_size: Some(60),
        stream_url: stream_url.map(|s| s.to_string()),
        ..Default::default()
    };

    let encoder = reco_io::adapters::FfmpegFileEncoder::new(
        Path::new(output),
        width,
        height,
        (capture_fps as i32, 1),
        &enc_config,
    )?;
    println!("Encoder: {}", encoder.encoder_name());

    session.set_encoder(Box::new(encoder), 2);

    let frame_limit =
        reco_core::session::compute_frame_limit(duration, max_frames, capture_fps as f64);

    if frame_limit < u64::MAX {
        println!("Capturing up to {frame_limit} frames");
    }

    let start = std::time::Instant::now();
    let mut frame_count: u64 = 0;

    if v4l2_direct {
        #[cfg(not(feature = "v4l2"))]
        anyhow::bail!("--v4l2-direct requires the `v4l2` feature flag");

        #[cfg(feature = "v4l2")]
        {
            use reco_core::bayer::{AeController, AwbController, BayerDemosaic, IspParams};
            use reco_io::v4l2::{V4l2CameraConfig, V4l2StereoCameraSource};

            let make_v4l2_config = |device: String| V4l2CameraConfig {
                device,
                width: capture_width,
                height: capture_height,
                fps: capture_fps,
                exposure: _exposure,
                gain: _sensor_gain,
            };
            let left_v4l2 = make_v4l2_config(cam_config.left_device.clone());
            let right_v4l2 = make_v4l2_config(cam_config.right_device.clone());

            let mut isp = IspParams::imx477_default(capture_width, capture_height);
            let mut demosaic_left =
                BayerDemosaic::new(session.gpu(), capture_width, capture_height, &isp);
            let mut demosaic_right =
                BayerDemosaic::new(session.gpu(), capture_width, capture_height, &isp);

            let mut awb = AwbController::new(isp.wb_r, isp.wb_b, 15);
            let mut ae = AeController::new(
                _exposure,
                _sensor_gain,
                200.0,
                vec![left_v4l2.device.clone(), right_v4l2.device.clone()],
                15,
            );

            println!(
                "GPU demosaic ready ({}x{}, zero-copy, AE+AWB), exposure={}, gain={}",
                capture_width, capture_height, _exposure, _sensor_gain,
            );

            let mut source = V4l2StereoCameraSource::open(&left_v4l2, &right_v4l2)?;
            if source.next_pair()?.is_some() {
                println!("Warmup complete, starting V4L2 Bayer capture...");
            }

            let progress = helpers::ProgressReporter::new(30);

            while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
                let (left_bytes, right_bytes) = {
                    reco_core::profile_scope!("wait_capture");
                    match source.next_pair()? {
                        Some(p) => p,
                        None => break,
                    }
                };

                if let Some((r, b)) = awb.update(&*left_bytes, capture_width, capture_height) {
                    isp.wb_r = r;
                    isp.wb_b = b;
                    demosaic_left.update_params(session.gpu(), &isp);
                    demosaic_right.update_params(session.gpu(), &isp);
                }
                ae.update(&*left_bytes, capture_width, capture_height);

                {
                    reco_core::profile_scope!("demosaic");
                    let gpu = session.gpu();
                    let mut encoder = gpu.device().create_command_encoder(
                        &reco_core::wgpu::CommandEncoderDescriptor {
                            label: Some("bayer_demosaic"),
                        },
                    );
                    demosaic_left.encode_demosaic(gpu, &mut encoder, &*left_bytes);
                    demosaic_right.encode_demosaic(gpu, &mut encoder, &*right_bytes);
                    gpu.queue().submit(std::iter::once(encoder.finish()));
                }

                // Detection: zero-copy CUDA path when available, CPU readback fallback.
                if session.detection_should_run() {
                    #[cfg(target_os = "linux")]
                    {
                        reco_core::profile_scope!("detection_zerocopy");
                        let gpu = session.gpu();
                        let mut det_encoder = gpu.device().create_command_encoder(
                            &reco_core::wgpu::CommandEncoderDescriptor {
                                label: Some("detection_copy"),
                            },
                        );
                        let (l_ptr, l_pitch, w, h) =
                            demosaic_left.copy_to_detection_shared(gpu, &mut det_encoder)?;
                        let (r_ptr, r_pitch, _, _) =
                            demosaic_right.copy_to_detection_shared(gpu, &mut det_encoder)?;
                        gpu.queue().submit(std::iter::once(det_encoder.finish()));
                        session.detect_and_update_director_cuda_rgba(
                            l_ptr,
                            l_pitch,
                            r_ptr,
                            r_pitch,
                            w,
                            h,
                            start.elapsed(),
                        )?;
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        reco_core::profile_scope!("detection_readback");
                        let left_rgba = demosaic_left.readback_rgba(session.gpu())?;
                        let right_rgba = demosaic_right.readback_rgba(session.gpu())?;
                        session.detect_and_update_director_rgba(
                            &left_rgba,
                            &right_rgba,
                            capture_width,
                            capture_height,
                            start.elapsed(),
                        )?;
                    }
                } else {
                    session.update_director(start.elapsed())?;
                }
                let pos = session.director_position();
                session.process_frame_gpu_rgba(
                    demosaic_left.output_texture(),
                    demosaic_right.output_texture(),
                    pos.yaw,
                    pos.pitch,
                )?;
                frame_count += 1;
                progress.report(frame_count);
            }

            source.stop();
            #[cfg(feature = "replay")]
            session.clear_stacked_gpu_recorder();
            session.finish()?;

            progress.finish(frame_count, output);
        }
    } else if use_nvmm {
        // NVMM zero-copy path: DMA-buf Vulkan import for rendering,
        // NvBufSurfTransform for detection. No CPU copies at all.
        #[cfg(target_os = "linux")]
        {
            use reco_core::dmabuf_import::DmaBufTextureCache;
            use reco_core::nvbuf_transform::NvBufDetectionSurface;
            use reco_io::gstreamer::camera::GstreamerNvmmCameraSource;

            let mut source = GstreamerNvmmCameraSource::open(&cam_config)?;

            // Detection surface size must match the TRT engine's input dims.
            // All shipped yolo26 engines use 1280x1280.
            let model_size = 1280u32;
            log::info!("NVMM detection surface size: {model_size}x{model_size}");
            let mut det_left =
                NvBufDetectionSurface::new(model_size, capture_width, capture_height)
                    .map_err(|e| anyhow::anyhow!("left detection surface: {e}"))?;
            let mut det_right =
                NvBufDetectionSurface::new(model_size, capture_width, capture_height)
                    .map_err(|e| anyhow::anyhow!("right detection surface: {e}"))?;

            let det_surface_mb =
                (det_left.pitch as f64 * model_size as f64 * 2.0) / 1024.0 / 1024.0;
            let tensor_mb =
                (model_size as f64 * model_size as f64 * 3.0 * 4.0 * 2.0) / 1024.0 / 1024.0;
            let nvmm_buf_mb =
                (capture_width as f64 * capture_height as f64 * 1.5 * 8.0) / 1024.0 / 1024.0;
            log::info!(
                "NVMM detection surfaces: {}x{} RGBA, letterbox scale={:.4}, pitch={}",
                model_size,
                model_size,
                det_left.scale,
                det_left.pitch,
            );
            log::info!(
                "Memory budget: det_surfaces={:.1}MB, trt_tensors={:.1}MB, nvmm_bufs={:.1}MB",
                det_surface_mb,
                tensor_mb,
                nvmm_buf_mb,
            );

            let mut dmabuf_cache = DmaBufTextureCache::new();

            // Warmup: pull first pair to initialize ISP + Argus
            let warmup = source
                .next_pair()?
                .ok_or_else(|| anyhow::anyhow!("NVMM source returned no frames"))?;
            source.release_previous();
            log::info!(
                "NVMM warmup: {}x{} NV12, fd_left={}, fd_right={}",
                warmup.left.width,
                warmup.left.height,
                warmup.left.dmabuf_fd,
                warmup.right.dmabuf_fd,
            );
            log::info!("Warmup complete, starting NVMM capture");

            let progress = helpers::ProgressReporter::new(30);

            while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
                let pair = {
                    reco_core::profile_scope!("wait_capture");
                    match source.next_pair()? {
                        Some(p) => p,
                        None => break,
                    }
                };

                // Detection: NvBufSurfTransform on the raw NVMM surface pointers
                if session.detection_should_run() {
                    reco_core::profile_scope!("nvbuf_detect");
                    unsafe {
                        det_left
                            .transform_from_nvmm(pair.left.surface_ptr)
                            .map_err(|e| anyhow::anyhow!("left transform: {e}"))?;
                        det_right
                            .transform_from_nvmm(pair.right.surface_ptr)
                            .map_err(|e| anyhow::anyhow!("right transform: {e}"))?;
                    }
                    session.detect_and_update_director_preletterboxed(
                        det_left.data_ptr,
                        det_right.data_ptr,
                        capture_width,
                        capture_height,
                        start.elapsed(),
                    )?;
                } else {
                    session.update_director(start.elapsed())?;
                }

                // Rendering: ensure DMA-buf textures are cached, then borrow both
                {
                    reco_core::profile_scope!("dmabuf_ensure_cached");
                    dmabuf_cache
                        .ensure_imported(
                            session.gpu(),
                            pair.left.dmabuf_fd,
                            pair.left.width,
                            pair.left.height,
                            pair.left.y_offset,
                            pair.left.uv_offset,
                            pair.left.total_size,
                        )
                        .map_err(|e| anyhow::anyhow!("left DMA-buf import: {e}"))?;
                    dmabuf_cache
                        .ensure_imported(
                            session.gpu(),
                            pair.right.dmabuf_fd,
                            pair.right.width,
                            pair.right.height,
                            pair.right.y_offset,
                            pair.right.uv_offset,
                            pair.right.total_size,
                        )
                        .map_err(|e| anyhow::anyhow!("right DMA-buf import: {e}"))?;
                }
                let left_textures = dmabuf_cache.get(pair.left.dmabuf_fd);
                let right_textures = dmabuf_cache.get(pair.right.dmabuf_fd);

                let pos = session.director_position();
                {
                    reco_core::profile_scope!("stitch_and_encode");
                    session.process_frame_imported_nv12(
                        &left_textures.y_texture,
                        &left_textures.uv_texture,
                        &right_textures.y_texture,
                        &right_textures.uv_texture,
                        pos.yaw,
                        pos.pitch,
                    )?;
                }

                source.release_previous();

                frame_count += 1;
                progress.report(frame_count);
            }

            source.stop();
            #[cfg(feature = "replay")]
            session.clear_stacked_gpu_recorder();
            session.finish()?;

            progress.finish(frame_count, output);
        }

        #[cfg(not(target_os = "linux"))]
        anyhow::bail!("NVMM zero-copy is only available on Linux/Jetson");
    } else if use_nv12_capture {
        // NV12 path: skip nvvidconv format conversion, upload 2 planes
        let mut source = reco_io::gstreamer::camera::GstreamerNv12CameraSource::open(&cam_config)?;

        // Warm up: discard first frame (camera ISP + pipeline init)
        if let Some(pair) = source.next_pair()? {
            let stereo = StereoFrame::Nv12(pair);
            session.detect_and_update_director(&stereo, start.elapsed())?;
            let pos = session.director_position();
            session.process_frame(&stereo, pos.yaw, pos.pitch)?;
            println!("Warmup complete, starting capture...");
        }

        let progress = helpers::ProgressReporter::new(30);

        while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let pair = {
                reco_core::profile_scope!("wait_capture");
                match source.next_pair()? {
                    Some(p) => p,
                    None => break,
                }
            };

            let stereo = StereoFrame::Nv12(pair);
            session.detect_and_update_director(&stereo, start.elapsed())?;
            let pos = session.director_position();
            session.process_frame(&stereo, pos.yaw, pos.pitch)?;
            frame_count += 1;
            progress.report(frame_count);
        }

        // Stop cameras gracefully before finishing encoder
        source.stop();
        #[cfg(feature = "replay")]
        session.clear_stacked_gpu_recorder();
        session.finish()?;

        progress.finish(frame_count, output);

        // Drop source explicitly to allow graceful GStreamer/Argus teardown
        drop(source);
    } else {
        // I420 path: standard YUV420P upload with 3 planes
        use reco_core::source::FrameSource;
        let mut source = reco_io::gstreamer::camera::GstreamerCameraSource::open(&cam_config)?;

        let progress = helpers::ProgressReporter::new(30);

        while frame_count < frame_limit && !interrupted.load(Ordering::Relaxed) {
            let frame = {
                reco_core::profile_scope!("wait_capture");
                match source.next_frame()? {
                    Some(f) => f,
                    None => break,
                }
            };

            session.detect_and_update_director(&frame, start.elapsed())?;
            let pos = session.director_position();
            session.process_frame(&frame, pos.yaw, pos.pitch)?;
            frame_count += 1;
            progress.report(frame_count);
        }

        #[cfg(feature = "replay")]
        session.clear_stacked_gpu_recorder();
        session.finish()?;

        progress.finish(frame_count, output);
    }

    Ok(())
}

fn nv12_to_yuv(
    nv12: &reco_core::source::Nv12Data,
    width: u32,
    height: u32,
) -> reco_calibrate::types::YuvFrame {
    let mut u = Vec::with_capacity((width as usize / 2) * (height as usize / 2));
    let mut v = Vec::with_capacity(u.capacity());
    for pair in nv12.uv.chunks_exact(2) {
        u.push(pair[0]);
        v.push(pair[1]);
    }
    reco_calibrate::types::YuvFrame {
        y: nv12.y.clone(),
        u,
        v,
        width,
        height,
        timestamp_us: 0,
    }
}

struct Nv12CameraCalibSource {
    source: reco_io::gstreamer::camera::GstreamerNv12CameraSource,
    width: u32,
    height: u32,
    fps: u32,
    frame_count: u32,
}

impl reco_calibrate::live::LiveFramePairSource for Nv12CameraCalibSource {
    fn next_pair(
        &mut self,
        timeout: std::time::Duration,
    ) -> Option<(
        reco_calibrate::types::YuvFrame,
        reco_calibrate::types::YuvFrame,
    )> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if std::time::Instant::now() >= deadline {
                return None;
            }
            match self.source.next_pair() {
                Ok(Some(pair)) => {
                    self.frame_count += 1;
                    if self.frame_count < self.fps && self.frame_count > 1 {
                        continue;
                    }
                    return Some((
                        nv12_to_yuv(&pair.left, self.width, self.height),
                        nv12_to_yuv(&pair.right, self.width, self.height),
                    ));
                }
                Ok(None) => return None,
                Err(e) => {
                    log::error!("camera capture error: {e}");
                    return None;
                }
            }
        }
    }
}

/// Run live calibration from camera feeds.
pub fn run_live_calibrate(
    cam_config: CameraConfig,
    output_path: &str,
    capture_width: u32,
    capture_height: u32,
    num_pairs: usize,
    left_profile: Option<&str>,
    right_profile: Option<&str>,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    use reco_core::calibration::CameraParams;

    eprintln!(
        "Live calibration: capturing {num_pairs} frame pairs at {capture_width}x{capture_height}",
    );

    let load_or_default = |path: Option<&str>, w: u32, h: u32| -> anyhow::Result<CameraParams> {
        if let Some(p) = path {
            let json = std::fs::read_to_string(p)?;
            let params: CameraParams = serde_json::from_str(&json)?;
            eprintln!("Lens profile: {p}");
            return Ok(params);
        }
        if let Ok(home) = std::env::var("HOME") {
            let convention = std::path::PathBuf::from(home).join("imx477_profile.json");
            if convention.exists() {
                let json = std::fs::read_to_string(&convention)?;
                let params: CameraParams = serde_json::from_str(&json)?;
                eprintln!("Lens profile (auto): {}", convention.display());
                return Ok(params);
            }
        }
        eprintln!("No lens profile found, using synthetic default (wide-angle)");
        let fw = w as f64;
        let fh = h as f64;
        Ok(CameraParams {
            width: w,
            height: h,
            fx: fw * 0.5,
            fy: fw * 0.5,
            cx: fw * 0.5,
            cy: fh * 0.5,
            d: [0.0; 4],
        })
    };
    let left_params = load_or_default(left_profile, capture_width, capture_height)?;
    let right_params = load_or_default(right_profile, capture_width, capture_height)
        .unwrap_or_else(|_| left_params.clone());

    let gpu = reco_core::gpu::GpuContext::new_blocking()?;
    eprintln!("GPU: {}", gpu.gpu_name());

    let source = reco_io::gstreamer::camera::GstreamerNv12CameraSource::open(&cam_config)?;
    eprintln!("Cameras open, warming up ISP...");

    let mut calib_source = Nv12CameraCalibSource {
        source,
        width: capture_width,
        height: capture_height,
        fps: cam_config.fps,
        frame_count: 0,
    };

    let options = reco_calibrate::live::CalibrateFromLiveOptions {
        num_pairs,
        timeout_per_pair: std::time::Duration::from_secs(10),
        config: reco_calibrate::types::CalibrationConfig::default(),
    };

    let result = reco_calibrate::live::calibrate_from_live(
        &gpu,
        &mut calib_source,
        &left_params,
        &right_params,
        &options,
        &mut |progress| {
            eprintln!("  [{}] {}", progress.step, progress.detail);
        },
        interrupted,
    )?;

    calib_source.source.stop();

    let confidence_pct = result.confidence * 100.0;
    eprintln!(
        "Calibration complete: {} matches, confidence {confidence_pct:.0}%",
        result.total_matches,
    );

    if confidence_pct < 50.0 {
        eprintln!(
            "WARNING: Low confidence ({confidence_pct:.0}%). \
             Try adjusting camera overlap or improving lighting.",
        );
    }

    // Preserve field_roi and rig_tilt from existing calibration file
    let mut cal = result.calibration;
    if let Ok(existing) = std::fs::read_to_string(output_path) {
        if let Ok(prev) =
            serde_json::from_str::<reco_core::calibration::MatchCalibration>(&existing)
        {
            if prev.field_roi.is_some() {
                cal.field_roi = prev.field_roi;
                eprintln!("Preserved existing field_roi");
            }
            if prev.rig_tilt.abs() > 1e-6 {
                cal.rig_tilt = prev.rig_tilt;
                eprintln!(
                    "Preserved existing rig_tilt ({:.1} deg)",
                    prev.rig_tilt.to_degrees()
                );
            }
            if prev.rig_roll.abs() > 1e-6 {
                cal.rig_roll = prev.rig_roll;
            }
        }
    }
    let json = serde_json::to_string_pretty(&cal)?;
    std::fs::write(output_path, &json)?;
    eprintln!("Written to {output_path}");

    let summary = serde_json::json!({
        "status": "ok",
        "confidence": result.confidence,
        "matches": result.total_matches,
        "path": output_path,
    });
    println!("{summary}");

    Ok(())
}
