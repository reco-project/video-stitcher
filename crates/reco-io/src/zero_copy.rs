//! Decode thread spawning for zero-copy GPU paths.
//!
//! These functions spawn FFmpeg decode threads that write directly to
//! GPU-shared memory (CUDA/Vulkan on Linux, VideoToolbox/Metal on macOS,
//! D3D11VA on Windows). The types they consume and produce are defined
//! in [`reco_core::zero_copy`].
//!
//! This module lives in `reco-io` (not `reco-core`) because it needs
//! `VideoDecoder` from the FFmpeg backend. `reco-core` orchestrates
//! the frame loop; `reco-io` handles the decode threads.

pub use reco_core::zero_copy::DecodePauseControl;

#[cfg(target_os = "windows")]
use std::sync::Arc;

/// Spawn a single-video GPU decode thread that writes NV12 frames directly
/// to CUDA/Vulkan shared textures via `cuMemcpy2D`.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn spawn_single_decoder_gpu(
    input: crate::stitch_job::InputPath,
    label: &'static str,
    buf: reco_core::zero_copy::GpuBufInfo,
    slot_free_rx: std::sync::mpsc::Receiver<u8>,
    skip_frames: u64,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> (std::sync::mpsc::Receiver<u8>, std::thread::JoinHandle<()>) {
    use crate::ffmpeg::decoder::VideoDecoder;

    let (tx, rx) = std::sync::mpsc::sync_channel::<u8>(1);

    let handle = std::thread::Builder::new()
        .name(format!("decode_{label}_gpu"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open_input(&input) {
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

            // Skip frames for temporal sync (decode and discard, no GPU write).
            for i in 0..skip_frames {
                match dec.next_frame() {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        log::error!("{label}: EOF after skipping {i}/{skip_frames} frames");
                        return;
                    }
                    Err(e) => {
                        log::error!("{label} skip decode error: {e}");
                        return;
                    }
                }
            }
            if skip_frames > 0 {
                log::info!("{label}: skipped {skip_frames} frames for sync offset");
            }

            loop {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let slot = match slot_free_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(s) => s,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                };
                match dec.next_frame_gpu() {
                    Ok(Some(frame)) => {
                        let s = slot as usize;

                        if let Err(e) = reco_core::cuda_interop::cuda_ensure_context() {
                            log::error!("{label} cuda_ensure_context: {e}");
                            break;
                        }

                        // Byte width for CUDA copy: width * bytes_per_sample.
                        // For UV, half-width but 2 components cancels out.
                        let bps = buf.pixel_format.bytes_per_sample();
                        let y_width_bytes = buf.width as usize * bps;
                        let uv_width_bytes = buf.width as usize * bps;

                        // Copy Y plane: NVDEC -> shared texture
                        if let Err(e) = reco_core::cuda_interop::cuda_2d_copy(
                            buf.y_ptr[s],
                            buf.y_pitch[s],
                            frame.y_ptr,
                            frame.y_pitch,
                            y_width_bytes,
                            buf.height as usize,
                        ) {
                            log::error!("{label} cuMemcpy2D Y: {e}");
                            break;
                        }

                        // Copy UV plane: NVDEC -> shared texture
                        if let Err(e) = reco_core::cuda_interop::cuda_2d_copy(
                            buf.uv_ptr[s],
                            buf.uv_pitch[s],
                            frame.uv_ptr,
                            frame.uv_pitch,
                            uv_width_bytes,
                            buf.height as usize / 2,
                        ) {
                            log::error!("{label} cuMemcpy2D UV: {e}");
                            break;
                        }

                        if let Err(e) = reco_core::cuda_interop::cuda_synchronize() {
                            log::error!("{label} cuCtxSynchronize: {e}");
                            break;
                        }

                        if tx.send(slot).is_err() {
                            break;
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

    (rx, handle)
}

/// Spawn parallel GPU decode threads and a pairing thread.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames (see [`FfmpegFileSource::open_with_offset`](crate::adapters::FfmpegFileSource::open_with_offset)).
///
/// Returns [`GpuDecodeHandles`](reco_core::zero_copy::GpuDecodeHandles) containing the paired frame signal
/// receiver and join handles for graceful shutdown.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn spawn_decode_threads_gpu(
    left_input: crate::stitch_job::InputPath,
    right_input: crate::stitch_job::InputPath,
    left_buf: reco_core::zero_copy::GpuBufInfo,
    right_buf: reco_core::zero_copy::GpuBufInfo,
    left_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    right_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    sync_offset: i64,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> reco_core::zero_copy::GpuDecodeHandles {
    use reco_core::zero_copy::{GpuDecodeHandles, GpuFrameSignal};

    // Compute per-decoder skip counts from the sync offset.
    let (left_skip, right_skip) = if sync_offset > 0 {
        (0, sync_offset as u64)
    } else {
        (sync_offset.unsigned_abs(), 0)
    };

    let (left_rx, left_handle) = spawn_single_decoder_gpu(
        left_input,
        "left",
        left_buf,
        left_slot_free_rx,
        left_skip,
        shutdown.clone(),
    );
    let (right_rx, right_handle) = spawn_single_decoder_gpu(
        right_input,
        "right",
        right_buf,
        right_slot_free_rx,
        right_skip,
        shutdown,
    );

    let (tx, rx) = std::sync::mpsc::sync_channel::<GpuFrameSignal>(1);

    let pair_handle = std::thread::Builder::new()
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

    GpuDecodeHandles {
        frame_rx: rx,
        join_handles: vec![left_handle, right_handle, pair_handle],
    }
}

/// Spawn a VideoToolbox decode thread that sends retained CVPixelBuffers.
#[cfg(target_os = "macos")]
pub fn spawn_vt_decode_thread(
    input: crate::stitch_job::InputPath,
    label: &'static str,
) -> std::sync::mpsc::Receiver<reco_core::metal_interop::RetainedCVPixelBuffer> {
    use crate::ffmpeg::decoder::VideoDecoder;
    use reco_core::metal_interop::RetainedCVPixelBuffer;

    let (tx, rx) = std::sync::mpsc::sync_channel::<RetainedCVPixelBuffer>(4);

    std::thread::Builder::new()
        .name(format!("vt_decode_{label}"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open_input(&input) {
                Ok(d) => d,
                Err(e) => {
                    log::error!("Failed to open {label} video: {e}");
                    return;
                }
            };
            log::info!("VT decode thread {label}: backend={}", dec.backend());

            loop {
                match dec.next_frame_vt() {
                    Ok(Some(vt)) => {
                        let retained = unsafe { RetainedCVPixelBuffer::retain(vt.cv_pixel_buffer) };
                        if tx.send(retained).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::error!("{label} VT decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn VT decode thread");

    rx
}

/// Spawn paired VideoToolbox decode threads and return the pair receiver.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames.
///
/// Spawns two VT decode threads (left + right) and a pairing thread
/// that zips frames into [`VtFramePair`]s.
#[cfg(target_os = "macos")]
pub fn spawn_vt_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    sync_offset: i64,
) -> std::sync::mpsc::Receiver<reco_core::zero_copy::VtFramePair> {
    use reco_core::zero_copy::VtFramePair;

    let left_rx = spawn_vt_decode_thread(left.clone(), "left");
    let right_rx = spawn_vt_decode_thread(right.clone(), "right");

    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<VtFramePair>(4);
    std::thread::Builder::new()
        .name("vt_pair".into())
        .spawn(move || {
            // Apply sync offset.
            if sync_offset > 0 {
                for _ in 0..sync_offset {
                    if right_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("VT sync offset: skipped {sync_offset} right frames");
            } else if sync_offset < 0 {
                let skip = sync_offset.unsigned_abs();
                for _ in 0..skip {
                    if left_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("VT sync offset: skipped {skip} left frames");
            }

            while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                if pair_tx.send(VtFramePair { left, right }).is_err() {
                    break;
                }
            }
        })
        .expect("spawn VT pairing thread");

    pair_rx
}

/// Spawn a D3D11VA decode thread that sends raw texture pointers + slice indices.
#[cfg(target_os = "windows")]
pub fn spawn_d3d11_decode_thread(
    input: crate::stitch_job::InputPath,
    label: &'static str,
    pause_ctl: Arc<DecodePauseControl>,
) -> std::sync::mpsc::Receiver<crate::ffmpeg::decoder::D3d11Frame> {
    use crate::ffmpeg::decoder::{D3d11Frame, VideoDecoder};

    let (tx, rx) = std::sync::mpsc::sync_channel::<D3d11Frame>(4);

    std::thread::Builder::new()
        .name(format!("d3d11_decode_{label}"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open_input(&input) {
                Ok(d) => d,
                Err(e) => {
                    log::error!("Failed to open {label} video: {e}");
                    return;
                }
            };
            log::info!("D3D11VA decode thread {label}: backend={}", dec.backend());

            loop {
                pause_ctl.check_pause();
                match dec.next_frame_d3d11() {
                    Ok(Some(frame)) => {
                        if tx.send(frame).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::error!("{label} D3D11VA decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn D3D11VA decode thread");

    rx
}

/// Spawn paired D3D11VA decode threads and return the pair receiver
/// plus a pause control for power management during AI inference.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames.
#[cfg(target_os = "windows")]
pub fn spawn_d3d11_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    sync_offset: i64,
) -> (
    std::sync::mpsc::Receiver<(
        crate::ffmpeg::decoder::D3d11Frame,
        crate::ffmpeg::decoder::D3d11Frame,
    )>,
    Arc<DecodePauseControl>,
) {
    use crate::ffmpeg::decoder::D3d11Frame;

    let pause_ctl = DecodePauseControl::new(2);

    let left_rx = spawn_d3d11_decode_thread(left.clone(), "left", Arc::clone(&pause_ctl));
    let right_rx = spawn_d3d11_decode_thread(right.clone(), "right", Arc::clone(&pause_ctl));

    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<(D3d11Frame, D3d11Frame)>(4);
    std::thread::Builder::new()
        .name("d3d11_pair".into())
        .spawn(move || {
            if sync_offset > 0 {
                for _ in 0..sync_offset {
                    if right_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("D3D11VA sync offset: skipped {sync_offset} right frames");
            } else if sync_offset < 0 {
                let skip = sync_offset.unsigned_abs();
                for _ in 0..skip {
                    if left_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("D3D11VA sync offset: skipped {skip} left frames");
            }

            while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                if pair_tx.send((left, right)).is_err() {
                    break;
                }
            }
        })
        .expect("spawn D3D11VA pairing thread");

    (pair_rx, pause_ctl)
}
