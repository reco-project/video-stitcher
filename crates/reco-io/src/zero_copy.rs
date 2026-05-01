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
                        let use_array = buf.y_array[s] != 0;

                        // Copy Y plane: NVDEC -> shared texture
                        let y_result = if use_array {
                            reco_core::cuda_interop::cuda_2d_copy_to_array(
                                buf.y_array[s] as *mut std::ffi::c_void,
                                frame.y_ptr,
                                frame.y_pitch,
                                y_width_bytes,
                                buf.height as usize,
                            )
                        } else {
                            reco_core::cuda_interop::cuda_2d_copy(
                                buf.y_ptr[s],
                                buf.y_pitch[s],
                                frame.y_ptr,
                                frame.y_pitch,
                                y_width_bytes,
                                buf.height as usize,
                            )
                        };
                        if let Err(e) = y_result {
                            log::error!("{label} cuMemcpy2D Y: {e}");
                            break;
                        }
                        // Copy UV plane: NVDEC -> shared texture
                        let uv_result = if use_array {
                            reco_core::cuda_interop::cuda_2d_copy_to_array(
                                buf.uv_array[s] as *mut std::ffi::c_void,
                                frame.uv_ptr,
                                frame.uv_pitch,
                                uv_width_bytes,
                                buf.height as usize / 2,
                            )
                        } else {
                            reco_core::cuda_interop::cuda_2d_copy(
                                buf.uv_ptr[s],
                                buf.uv_pitch[s],
                                frame.uv_ptr,
                                frame.uv_pitch,
                                uv_width_bytes,
                                buf.height as usize / 2,
                            )
                        };
                        if let Err(e) = uv_result {
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

/// Spawn paired D3D11VA decode threads and return the pair receiver.
///
/// Both decoders share the same D3D11 device (via hw_device_ref sharing)
/// to ensure `CopySubresourceRegion` operates on same-device resources.
/// Without this, NVIDIA drivers hang on cross-device copies.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames.
#[cfg(target_os = "windows")]
pub fn spawn_d3d11_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    sync_offset: i64,
) -> std::sync::mpsc::Receiver<(
    crate::ffmpeg::decoder::D3d11Frame,
    crate::ffmpeg::decoder::D3d11Frame,
)> {
    use crate::ffmpeg::decoder::{D3d11Frame, VideoDecoder};

    let left_input = left.clone();
    let right_input = right.clone();

    let (left_tx, left_rx) = std::sync::mpsc::sync_channel::<D3d11Frame>(4);
    let (right_tx, right_rx) = std::sync::mpsc::sync_channel::<D3d11Frame>(4);

    // Create left decoder on calling thread to extract hw_device_ref.
    let mut left_dec = match VideoDecoder::open_input(&left_input) {
        Ok(d) => d,
        Err(e) => {
            log::error!("Failed to open left video: {e}");
            let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<(D3d11Frame, D3d11Frame)>(1);
            drop(pair_tx);
            return pair_rx;
        }
    };
    let shared_hw = left_dec.hw_device_ref();
    log::info!("D3D11VA left decoder: backend={}", left_dec.backend());

    // Create right decoder sharing the left's D3D11 device.
    let mut right_dec = if shared_hw.is_null() {
        VideoDecoder::open_input(&right_input)
    } else {
        VideoDecoder::open_input_shared_hw(&right_input, shared_hw)
    };
    let mut right_dec = match right_dec {
        Ok(d) => d,
        Err(e) => {
            log::error!("Failed to open right video: {e}");
            let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<(D3d11Frame, D3d11Frame)>(1);
            drop(pair_tx);
            return pair_rx;
        }
    };
    log::info!("D3D11VA right decoder: backend={}", right_dec.backend());

    // Apply sync offset before spawning threads.
    if sync_offset > 0 {
        for i in 0..sync_offset as u64 {
            match right_dec.next_frame_d3d11() {
                Ok(Some(_)) => {}
                Ok(None) => {
                    log::warn!("D3D11VA sync: right EOF after {i}/{sync_offset} skip frames");
                    break;
                }
                Err(e) => {
                    log::error!("D3D11VA sync skip error: {e}");
                    break;
                }
            }
        }
        log::info!("D3D11VA sync offset: skipped {sync_offset} right frames");
    } else if sync_offset < 0 {
        let skip = sync_offset.unsigned_abs();
        for i in 0..skip {
            match left_dec.next_frame_d3d11() {
                Ok(Some(_)) => {}
                Ok(None) => {
                    log::warn!("D3D11VA sync: left EOF after {i}/{skip} skip frames");
                    break;
                }
                Err(e) => {
                    log::error!("D3D11VA sync skip error: {e}");
                    break;
                }
            }
        }
        log::info!("D3D11VA sync offset: skipped {skip} left frames");
    }

    // Move decoders to their threads.
    std::thread::Builder::new()
        .name("d3d11_decode_left".into())
        .spawn(move || {
            loop {
                match left_dec.next_frame_d3d11() {
                    Ok(Some(frame)) => {
                        if left_tx.send(frame).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::error!("left D3D11VA decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn D3D11VA left decode thread");

    std::thread::Builder::new()
        .name("d3d11_decode_right".into())
        .spawn(move || {
            loop {
                match right_dec.next_frame_d3d11() {
                    Ok(Some(frame)) => {
                        if right_tx.send(frame).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::error!("right D3D11VA decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn D3D11VA right decode thread");

    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<(D3d11Frame, D3d11Frame)>(4);
    std::thread::Builder::new()
        .name("d3d11_pair".into())
        .spawn(move || {
            while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                if pair_tx.send((left, right)).is_err() {
                    break;
                }
            }
        })
        .expect("spawn D3D11VA pairing thread");

    pair_rx
}
