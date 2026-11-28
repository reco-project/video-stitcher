//! Decode workers for zero-copy GPU paths.
//!
//! Frame messages are followed by explicit EOF or failure. CUDA copies finish
//! before a slot is published; Metal and D3D11 messages retain their surfaces.

#[cfg(target_os = "linux")]
pub(crate) mod cuda;

#[cfg(all(test, target_os = "windows"))]
mod windows_tests;

#[cfg(target_os = "linux")]
use crate::decode_stream::Message;
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
use crate::decode_stream::spawn;
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
use crate::ffmpeg::decoder::{DecodeError, VideoDecoder};
#[cfg(target_os = "linux")]
use reco_core::interop::cuda_lease::{CudaLease, CudaSlotPool};
#[cfg(target_os = "linux")]
use std::{
    sync::{Arc, atomic::AtomicBool, mpsc::Receiver},
    thread::JoinHandle,
};

/// Spawn a single-video GPU decode worker that copies frames into shared textures.
#[cfg(target_os = "linux")]
fn spawn_single_decoder_gpu(
    input: crate::stitch_job::InputPath,
    label: String,
    pool: CudaSlotPool,
    skip_frames: u64,
    shutdown: Arc<AtomicBool>,
) -> (Receiver<Message<CudaLease>>, JoinHandle<()>) {
    spawn(
        format!("decode_{label}_gpu"),
        1,
        move |tx| -> Result<(), DecodeError> {
            let mut dec = VideoDecoder::open_input(&input)?;
            log::info!(
                "{label} GPU decoder: {} ({}x{})",
                dec.backend(),
                dec.width(),
                dec.height()
            );

            for _ in 0..skip_frames {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed)
                    || dec.next_frame_gpu()?.is_none()
                {
                    return Ok(());
                }
            }

            while let Some(mut lease) = pool.acquire(&shutdown) {
                let Some(frame) = dec.next_frame_gpu()? else {
                    break;
                };
                cuda::stage(&frame, &mut lease)?;
                if tx.send(Ok(Some(lease))).is_err() {
                    break;
                }
            }
            Ok(())
        },
    )
}

/// Spawn one worker per input and collect synchronized resident frame sets.
#[cfg(target_os = "linux")]
pub(crate) fn spawn_decode_threads_gpu(
    decoders: Vec<(crate::stitch_job::InputPath, CudaSlotPool)>,
    sync_offset: i64,
    shutdown: Arc<AtomicBool>,
) -> crate::decode_stream::OwnedFrames<CudaLease> {
    let mut workers = Vec::with_capacity(decoders.len());
    let stereo = decoders.len() == 2;
    for (camera, (input, pool)) in decoders.into_iter().enumerate() {
        let skip =
            if stereo && ((camera == 0 && sync_offset < 0) || (camera == 1 && sync_offset > 0)) {
                sync_offset.unsigned_abs()
            } else {
                0
            };
        let (rx, handle) =
            spawn_single_decoder_gpu(input, camera.to_string(), pool, skip, shutdown.clone());
        workers.push((rx, crate::decoder_cleanup::DecoderTask::new(handle)));
    }
    crate::decode_stream::OwnedFrames::new(workers, 0, shutdown)
}

/// Spawn camera-ordered VideoToolbox workers with cancellation and joined cleanup.
#[cfg(target_os = "macos")]
pub(crate) fn spawn_vt_decode_frames(
    inputs: &[crate::stitch_job::InputPath],
    sync_offset: i64,
) -> crate::decode_stream::OwnedFrames<reco_core::interop::metal::RetainedCVPixelBuffer> {
    use reco_core::interop::metal::RetainedCVPixelBuffer;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let stop = Arc::new(AtomicBool::new(false));
    let workers = inputs
        .iter()
        .enumerate()
        .map(|(camera, input)| {
            let input = input.clone();
            let stop = stop.clone();
            let (rx, worker) = spawn(
                format!("vt_decode_{camera}"),
                4,
                move |tx| -> Result<(), DecodeError> {
                    if stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let mut dec = VideoDecoder::open_input(&input)?;
                    log::info!("VT decode thread {camera}: backend={}", dec.backend());
                    crate::decode_stream::DecodeStream {
                        label: &format!("VT decode thread {camera}"),
                        start_secs: None,
                        stop: &stop,
                        tx,
                    }
                    .run(
                        &mut dec,
                        VideoDecoder::next_frame_vt,
                        |frame| frame.timestamp_us,
                        // SAFETY: `frame.cv_pixel_buffer` stays valid until the next `next_frame_vt` call.
                        |frame| unsafe { RetainedCVPixelBuffer::retain(frame.cv_pixel_buffer) },
                    )
                },
            );
            (rx, crate::decoder_cleanup::DecoderTask::new(worker))
        })
        .collect();
    crate::decode_stream::OwnedFrames::new(workers, sync_offset, stop)
}

#[cfg(target_os = "windows")]
fn spawn_d3d11_decode_thread_shared(
    input: crate::stitch_job::InputPath,
    label: usize,
    shared_device: crate::ffmpeg::decoder::SharedHwDevice,
    start_secs: Option<f64>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> (
    std::sync::mpsc::Receiver<Message<crate::ffmpeg::decoder::D3d11Frame>>,
    crate::decoder_cleanup::DecoderTask,
) {
    let (rx, worker) = spawn(
        format!("d3d11_decode_{label}"),
        4,
        move |tx| -> Result<(), DecodeError> {
            use std::sync::atomic::Ordering;
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let mut dec = VideoDecoder::open_input_with_shared_device(&input, &shared_device)?;
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            log::info!("D3D11VA decode thread {label}: backend={}", dec.backend());
            crate::decode_stream::DecodeStream {
                label: &format!("D3D11VA decode thread {label}"),
                start_secs,
                stop: &stop,
                tx,
            }
            .run(
                &mut dec,
                VideoDecoder::next_frame_d3d11,
                |frame| frame.timestamp_us,
                |frame| frame,
            )
        },
    );
    (rx, crate::decoder_cleanup::DecoderTask::new(worker))
}

/// Spawn camera-ordered D3D11VA decode threads on a shared hardware device.
///
/// All decoders must use the same `ID3D11Device` for staging copies.
/// Positive `sync_offset` skips right frames; negative skips left frames.
/// When `start_secs` is `Some`, each decoder seeks to the nearest keyframe
/// before that timestamp and discards frames until the target PTS on the
/// stream's timestamp grid. Passing an existing `shared_device` keeps
/// respawned decoders on the same `ID3D11Device`.
/// The receiver emits retained frames, then explicit EOF or an error.
#[cfg(target_os = "windows")]
pub(crate) fn spawn_d3d11_decode_frames(
    inputs: &[crate::stitch_job::InputPath],
    sync_offset: i64,
    start_secs: Option<f64>,
    shared_device: Option<&crate::ffmpeg::decoder::SharedHwDevice>,
) -> Result<
    crate::decode_stream::OwnedFrames<crate::ffmpeg::decoder::D3d11Frame>,
    reco_core::source::SourceError,
> {
    let owned_device;
    let shared_device = match shared_device {
        Some(device) => device,
        None => {
            owned_device = crate::ffmpeg::decoder::create_shared_hw_device().ok_or_else(|| {
                reco_core::source::SourceError::Init {
                    path: "D3D11VA".into(),
                    reason: "hardware device creation failed".into(),
                }
            })?;
            &owned_device
        }
    };
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let workers = inputs
        .iter()
        .enumerate()
        .map(|(camera, input)| {
            spawn_d3d11_decode_thread_shared(
                input.clone(),
                camera,
                shared_device.new_ref(),
                start_secs,
                stop.clone(),
            )
        })
        .collect();
    Ok(crate::decode_stream::OwnedFrames::new(
        workers,
        sync_offset,
        stop,
    ))
}
