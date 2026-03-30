//! Live camera stitching via GStreamer.
//!
//! Captures stereo camera feeds, stitches them on the GPU, and encodes
//! the panoramic output to a video file in real time.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reco_io::gstreamer::camera::CameraConfig;

use crate::helpers;

/// Run live camera stitching.
///
/// Captures frames from two cameras via GStreamer, stitches them into a
/// panoramic view on the GPU, and encodes the result to `output`. Uses
/// NV12 native capture on Jetson (Tegra) and I420 elsewhere.
#[allow(clippy::too_many_arguments)] // Will be refactored into a config struct in Phase 2
pub fn run_camera(
    cam_config: CameraConfig,
    calibration: &str,
    output: &str,
    width: u32,
    height: u32,
    blend: f32,
    encoder_name: Option<String>,
    codec: &str,
    quality: &str,
    duration: Option<f64>,
    max_frames: Option<u64>,
    capture_fps: u32,
    interrupted: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    // Reject FFmpeg network URLs as output to prevent data exfiltration (#64).
    anyhow::ensure!(
        !output.contains("://"),
        "Output path looks like a network URL ({output}). Only local file paths are supported.",
    );

    let cal = helpers::load_calibration(Path::new(calibration))?;

    let viewport = reco_core::viewport::ViewportConfig {
        width,
        height,
        blend_width: blend,
        ..Default::default()
    };

    let gpu = pollster::block_on(reco_core::gpu::GpuContext::new())?;

    // Use NV12 capture on Jetson to skip the NV12->I420 conversion
    // in nvvidconv. The NVIDIA ISP natively outputs NV12.
    let use_nv12_capture = helpers::is_tegra();
    let input_format = if use_nv12_capture {
        reco_core::renderer::InputFormat::Nv12
    } else {
        reco_core::renderer::InputFormat::Yuv420p
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
    };
    let mut session = reco_core::session::StitchSession::with_gpu(gpu, session_config)?;

    let mode_str = if use_nv12_capture { "NV12" } else { "I420" };
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

    reco_io::init();
    let quality = match quality {
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
        quality,
    };

    let fps_rational = reco_io::ffmpeg::Rational::new(capture_fps as i32, 1);
    let mut enc = reco_io::ffmpeg::encoder::VideoEncoder::new(
        Path::new(output),
        width,
        height,
        fps_rational,
        &enc_config,
    )?;
    println!("Encoder: {}", enc.encoder_name());

    let capture_fps_f64 = capture_fps as f64;
    let frame_limit: u64 = match (duration, max_frames) {
        (Some(dur), Some(mf)) => ((dur * capture_fps_f64) as u64).min(mf),
        (Some(dur), None) => (dur * capture_fps_f64) as u64,
        (None, Some(mf)) => mf,
        (None, None) => u64::MAX,
    };

    if frame_limit < u64::MAX {
        println!("Capturing up to {frame_limit} frames");
    }

    // Async encode: send NV12 data to a background thread so
    // encoding overlaps with the next frame's capture + render.
    let (encode_tx, encode_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(2);
    let encode_thread = std::thread::Builder::new()
        .name("encode".into())
        .spawn(move || -> Result<(), anyhow::Error> {
            while let Ok(nv12_data) = encode_rx.recv() {
                enc.write_nv12_frame(&nv12_data)?;
            }
            enc.finish()?;
            Ok(())
        })
        .expect("spawn encode thread");

    let mut frame_count: u64 = 0;
    let yaw = 0.0_f32;
    let pitch = 0.0_f32;

    if use_nv12_capture {
        // NV12 path: skip nvvidconv format conversion, upload 2 planes
        let mut source = reco_io::gstreamer::camera::GstreamerNv12CameraSource::open(&cam_config)?;

        // Warm up: discard first frame (camera ISP + pipeline init)
        if let Some(pair) = source.next_pair()? {
            let left_planes = reco_core::pipeline::Nv12Planes {
                y: &pair.left.y,
                uv: &pair.left.uv,
            };
            let right_planes = reco_core::pipeline::Nv12Planes {
                y: &pair.right.y,
                uv: &pair.right.uv,
            };
            let render_buf = session.pipeline().render_to_target_nv12(
                &left_planes,
                &right_planes,
                yaw,
                pitch,
            )?;
            let nv12_data = session.convert_to_nv12(render_buf)?;
            if encode_tx.send(nv12_data.to_vec()).is_err() {
                anyhow::bail!("encoder thread died during warmup");
            }
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

            let left_planes = reco_core::pipeline::Nv12Planes {
                y: &pair.left.y,
                uv: &pair.left.uv,
            };
            let right_planes = reco_core::pipeline::Nv12Planes {
                y: &pair.right.y,
                uv: &pair.right.uv,
            };

            let render_buf = session.pipeline().render_to_target_nv12(
                &left_planes,
                &right_planes,
                yaw,
                pitch,
            )?;
            let nv12_data = session.convert_to_nv12(render_buf)?;
            if encode_tx.send(nv12_data.to_vec()).is_err() {
                break;
            }
            frame_count += 1;
            progress.report(frame_count);
        }

        // Flush the last pending frame from the double-buffered NV12 pipeline.
        if let Some(last_frame) = session.flush_pending_nv12()? {
            let _ = encode_tx.send(last_frame.to_vec());
            frame_count += 1;
        }

        // Stop cameras gracefully before finishing encoder
        source.stop();
        drop(encode_tx);
        encode_thread.join().expect("encode thread panicked")?;

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
            let pair = match frame {
                reco_core::source::StereoFrame::Yuv420p(p) => p,
                _ => anyhow::bail!("expected Yuv420p frame from I420 camera source"),
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
                session
                    .pipeline()
                    .render_to_target(&left_planes, &right_planes, yaw, pitch)?;
            let nv12_data = session.convert_to_nv12(render_buf)?;
            if encode_tx.send(nv12_data.to_vec()).is_err() {
                break;
            }
            frame_count += 1;
            progress.report(frame_count);
        }

        // Flush the last pending frame from the double-buffered NV12 pipeline.
        if let Some(last_frame) = session.flush_pending_nv12()? {
            let _ = encode_tx.send(last_frame.to_vec());
            frame_count += 1;
        }

        drop(encode_tx);
        encode_thread.join().expect("encode thread panicked")?;

        progress.finish(frame_count, output);
    }

    Ok(())
}
