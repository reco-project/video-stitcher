//! Decode thread spawning for zero-copy GPU paths.
//!
//! These functions spawn FFmpeg decode threads that write directly to
//! GPU-shared memory (CUDA/Vulkan on Linux, VideoToolbox/Metal on macOS).
//! The types they consume and produce are defined in [`reco_core::zero_copy`].
//!
//! This module lives in `reco-io` (not `reco-core`) because it needs
//! `VideoDecoder` from the FFmpeg backend. `reco-core` orchestrates
//! the frame loop; `reco-io` handles the decode threads.

use std::path::Path;

/// Spawn a single-video GPU decode thread that writes NV12 frames directly
/// to CUDA/Vulkan shared textures via `cuMemcpy2D`.
///
/// Uses `slot_free_rx` for backpressure: the decode thread waits for a slot
/// to be released by the main thread before writing to it. This prevents
/// NVDEC from overwriting a slot that the GPU render pass is still reading.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn spawn_single_decoder_gpu(
    path: String,
    label: &'static str,
    buf: reco_core::zero_copy::GpuBufInfo,
    slot_free_rx: std::sync::mpsc::Receiver<u8>,
) -> (std::sync::mpsc::Receiver<u8>, std::thread::JoinHandle<()>) {
    use crate::ffmpeg::decoder::VideoDecoder;

    let (tx, rx) = std::sync::mpsc::sync_channel::<u8>(1);

    let handle = std::thread::Builder::new()
        .name(format!("decode_{label}_gpu"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open(Path::new(&path)) {
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

            while let Ok(slot) = slot_free_rx.recv() {
                match dec.next_frame_gpu() {
                    Ok(Some(frame)) => {
                        let s = slot as usize;

                        if let Err(e) = reco_core::cuda_interop::cuda_ensure_context() {
                            log::error!("{label} cuda_ensure_context: {e}");
                            break;
                        }

                        // Copy Y plane: NVDEC -> shared texture
                        if let Err(e) = reco_core::cuda_interop::cuda_2d_copy(
                            buf.y_ptr[s],
                            buf.y_pitch[s],
                            frame.y_ptr,
                            frame.y_pitch,
                            buf.width as usize,
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
                            buf.width as usize,
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
/// Returns [`GpuDecodeHandles`](reco_core::zero_copy::GpuDecodeHandles) containing the paired frame signal
/// receiver and join handles for graceful shutdown.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn spawn_decode_threads_gpu(
    left_path: String,
    right_path: String,
    left_buf: reco_core::zero_copy::GpuBufInfo,
    right_buf: reco_core::zero_copy::GpuBufInfo,
    left_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    right_slot_free_rx: std::sync::mpsc::Receiver<u8>,
) -> reco_core::zero_copy::GpuDecodeHandles {
    use reco_core::zero_copy::{GpuDecodeHandles, GpuFrameSignal};

    let (left_rx, left_handle) =
        spawn_single_decoder_gpu(left_path, "left", left_buf, left_slot_free_rx);
    let (right_rx, right_handle) =
        spawn_single_decoder_gpu(right_path, "right", right_buf, right_slot_free_rx);

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
    path: std::path::PathBuf,
    label: &'static str,
) -> std::sync::mpsc::Receiver<reco_core::metal_interop::RetainedCVPixelBuffer> {
    use crate::ffmpeg::decoder::VideoDecoder;
    use reco_core::metal_interop::RetainedCVPixelBuffer;

    let (tx, rx) = std::sync::mpsc::sync_channel::<RetainedCVPixelBuffer>(4);

    std::thread::Builder::new()
        .name(format!("vt_decode_{label}"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open(&path) {
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
/// Spawns two VT decode threads (left + right) and a pairing thread
/// that zips frames into [`VtFramePair`]s.
#[cfg(target_os = "macos")]
pub fn spawn_vt_decode_pair(
    left_path: &str,
    right_path: &str,
) -> std::sync::mpsc::Receiver<reco_core::zero_copy::VtFramePair> {
    use reco_core::zero_copy::VtFramePair;

    let left_rx = spawn_vt_decode_thread(std::path::PathBuf::from(left_path), "left");
    let right_rx = spawn_vt_decode_thread(std::path::PathBuf::from(right_path), "right");

    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<VtFramePair>(4);
    std::thread::Builder::new()
        .name("vt_pair".into())
        .spawn(move || {
            while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                if pair_tx.send(VtFramePair { left, right }).is_err() {
                    break;
                }
            }
        })
        .expect("spawn VT pairing thread");

    pair_rx
}
