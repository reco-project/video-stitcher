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

/// Result of [`spawn_d3d11_decode_pair`]: paired slot receiver + wgpu views
/// + readback copiers for detection.
#[cfg(target_os = "windows")]
pub struct D3d11DecodeHandles {
    /// Receives paired staging slot indices (left_slot, right_slot).
    pub pair_rx: std::sync::mpsc::Receiver<(usize, usize)>,
    /// Pre-built wgpu NV12 plane views for rendering. Taken by the
    /// session via [`SmartFileSource::take_d3d11_views`].
    pub views: Option<reco_core::d3d11_interop::D3d11WgpuViews>,
    /// Staging copiers for readback on detection frames.
    /// Sent back from decode threads via channel on detection frames.
    pub left_readback_tx: std::sync::mpsc::SyncSender<ReadbackRequest>,
    pub left_readback_rx: std::sync::mpsc::Receiver<ReadbackResult>,
    pub right_readback_tx: std::sync::mpsc::SyncSender<ReadbackRequest>,
    pub right_readback_rx: std::sync::mpsc::Receiver<ReadbackResult>,
}

/// Request a readback from a specific slot.
#[cfg(target_os = "windows")]
pub struct ReadbackRequest {
    pub slot: usize,
}

/// Readback result with NV12 plane data.
#[cfg(target_os = "windows")]
pub struct ReadbackResult {
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
}

/// Readback callback that requests NV12 data from the decode threads
/// via channels. Implements [`D3d11ReadbackFn`] for the session.
#[cfg(target_os = "windows")]
pub struct ChannelReadbackFn {
    left_tx: std::sync::mpsc::SyncSender<ReadbackRequest>,
    left_rx: std::sync::mpsc::Receiver<ReadbackResult>,
    right_tx: std::sync::mpsc::SyncSender<ReadbackRequest>,
    right_rx: std::sync::mpsc::Receiver<ReadbackResult>,
}

#[cfg(target_os = "windows")]
impl ChannelReadbackFn {
    /// Create from a D3d11DecodeHandles, taking ownership of the rx channels.
    pub fn from_handles(handles: &mut D3d11DecodeHandles) -> Self {
        Self {
            left_tx: handles.left_readback_tx.clone(),
            left_rx: std::mem::replace(
                &mut handles.left_readback_rx,
                std::sync::mpsc::sync_channel(0).1,
            ),
            right_tx: handles.right_readback_tx.clone(),
            right_rx: std::mem::replace(
                &mut handles.right_readback_rx,
                std::sync::mpsc::sync_channel(0).1,
            ),
        }
    }
}

#[cfg(target_os = "windows")]
impl reco_core::session::D3d11ReadbackFn for ChannelReadbackFn {
    fn readback(
        &mut self,
        left_slot: usize,
        right_slot: usize,
    ) -> Result<
        (
            reco_core::session::D3d11ReadbackData,
            reco_core::session::D3d11ReadbackData,
        ),
        String,
    > {
        self.left_tx
            .send(ReadbackRequest { slot: left_slot })
            .map_err(|e| format!("left readback send: {e}"))?;
        self.right_tx
            .send(ReadbackRequest { slot: right_slot })
            .map_err(|e| format!("right readback send: {e}"))?;

        let left = self
            .left_rx
            .recv()
            .map_err(|e| format!("left readback recv: {e}"))?;
        let right = self
            .right_rx
            .recv()
            .map_err(|e| format!("right readback recv: {e}"))?;

        Ok((
            reco_core::session::D3d11ReadbackData {
                y: left.y,
                uv: left.uv,
            },
            reco_core::session::D3d11ReadbackData {
                y: right.y,
                uv: right.uv,
            },
        ))
    }
}

/// Spawn paired D3D11VA decode threads with decode-thread staging.
///
/// Decoders and staging copiers are created on the calling thread.
/// The decoders are then moved to decode threads that perform both
/// decode and staging copy, eliminating D3D11 context contention
/// on the main thread.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames.
#[cfg(target_os = "windows")]
pub fn spawn_d3d11_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    gpu: &reco_core::gpu::GpuContext,
    sync_offset: i64,
) -> Result<D3d11DecodeHandles, reco_core::source::SourceError> {
    use crate::ffmpeg::decoder::VideoDecoder;
    use reco_core::d3d11_interop::SLOTS_PER_CAMERA;

    let mut left_dec =
        VideoDecoder::open_input(left).map_err(|e| reco_core::source::SourceError::Init {
            path: left.first_path().display().to_string(),
            reason: format!("{e}"),
        })?;
    // Share the left decoder's D3D11 device with the right decoder.
    // Without this, each decoder creates its own D3D11 device, and
    // CopySubresourceRegion across devices hangs on NVIDIA drivers.
    let shared_hw = left_dec.hw_device_ref();
    let mut right_dec = if shared_hw.is_null() {
        VideoDecoder::open_input(right)
    } else {
        VideoDecoder::open_input_shared_hw(right, shared_hw)
    }
    .map_err(|e| reco_core::source::SourceError::Init {
        path: right.first_path().display().to_string(),
        reason: format!("{e}"),
    })?;

    log::info!(
        "D3D11VA decode: left={} ({}x{}), right={} ({}x{})",
        left_dec.backend(),
        left_dec.width(),
        left_dec.height(),
        right_dec.backend(),
        right_dec.width(),
        right_dec.height(),
    );

    let (dev, ctx) =
        left_dec
            .d3d11_device_ptrs()
            .ok_or_else(|| reco_core::source::SourceError::Init {
                path: left.first_path().display().to_string(),
                reason: "D3D11VA device not available".into(),
            })?;

    let (mut left_copier, mut right_copier, views) = unsafe {
        reco_core::d3d11_interop::create_staging_pair(
            gpu,
            dev,
            ctx,
            left_dec.width(),
            left_dec.height(),
        )
        .map_err(|e| reco_core::source::SourceError::Init {
            path: left.first_path().display().to_string(),
            reason: format!("D3D11 staging pool: {e}"),
        })?
    };

    // Apply sync offset before spawning threads.
    if sync_offset > 0 {
        for i in 0..sync_offset as u64 {
            if right_dec
                .next_frame_d3d11()
                .map_err(|e| reco_core::source::SourceError::Read {
                    reason: format!("sync skip: {e}"),
                })?
                .is_none()
            {
                log::warn!("D3D11VA sync: right EOF after {i}/{sync_offset} skip frames");
                break;
            }
        }
        log::info!("D3D11VA sync offset: skipped {sync_offset} right frames");
    } else if sync_offset < 0 {
        let skip = sync_offset.unsigned_abs();
        for i in 0..skip {
            if left_dec
                .next_frame_d3d11()
                .map_err(|e| reco_core::source::SourceError::Read {
                    reason: format!("sync skip: {e}"),
                })?
                .is_none()
            {
                log::warn!("D3D11VA sync: left EOF after {i}/{skip} skip frames");
                break;
            }
        }
        log::info!("D3D11VA sync offset: skipped {skip} left frames");
    }

    // Channel depth must be < SLOTS_PER_CAMERA to prevent the decode
    // thread from wrapping around and overwriting a slot the main thread
    // hasn't rendered yet. With 3 slots: depth 2 = one being rendered,
    // one in the channel, one being staged.
    let slot_channel_depth = SLOTS_PER_CAMERA - 1;
    let (left_slot_tx, left_slot_rx) = std::sync::mpsc::sync_channel::<usize>(slot_channel_depth);
    let (right_slot_tx, right_slot_rx) = std::sync::mpsc::sync_channel::<usize>(slot_channel_depth);

    // Readback channels for detection.
    let (left_rb_req_tx, left_rb_req_rx) = std::sync::mpsc::sync_channel::<ReadbackRequest>(1);
    let (left_rb_res_tx, left_rb_res_rx) = std::sync::mpsc::sync_channel::<ReadbackResult>(1);
    let (right_rb_req_tx, right_rb_req_rx) = std::sync::mpsc::sync_channel::<ReadbackRequest>(1);
    let (right_rb_res_tx, right_rb_res_rx) = std::sync::mpsc::sync_channel::<ReadbackResult>(1);

    // Spawn left decode thread.
    std::thread::Builder::new()
        .name("d3d11_decode_left".into())
        .spawn(move || {
            let mut frame_count: u64 = 0;
            loop {
                // Check for readback request (non-blocking).
                if let Ok(req) = left_rb_req_rx.try_recv() {
                    match left_copier.readback_nv12(req.slot) {
                        Ok((y, uv)) => {
                            let _ = left_rb_res_tx.send(ReadbackResult { y, uv });
                        }
                        Err(e) => {
                            log::error!("left readback failed: {e}");
                        }
                    }
                }

                match left_dec.next_frame_d3d11() {
                    Ok(Some(frame)) => {
                        let slot = frame_count as usize % SLOTS_PER_CAMERA;
                        if let Err(e) =
                            left_copier.stage_frame(frame.texture, frame.array_slice, slot)
                        {
                            log::error!("left stage_frame: {e}");
                            break;
                        }
                        if left_slot_tx.send(slot).is_err() {
                            break;
                        }
                        frame_count += 1;
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

    // Spawn right decode thread.
    std::thread::Builder::new()
        .name("d3d11_decode_right".into())
        .spawn(move || {
            let mut frame_count: u64 = 0;
            loop {
                if let Ok(req) = right_rb_req_rx.try_recv() {
                    match right_copier.readback_nv12(req.slot) {
                        Ok((y, uv)) => {
                            let _ = right_rb_res_tx.send(ReadbackResult { y, uv });
                        }
                        Err(e) => {
                            log::error!("right readback failed: {e}");
                        }
                    }
                }

                match right_dec.next_frame_d3d11() {
                    Ok(Some(frame)) => {
                        let slot = frame_count as usize % SLOTS_PER_CAMERA;
                        if let Err(e) =
                            right_copier.stage_frame(frame.texture, frame.array_slice, slot)
                        {
                            log::error!("right stage_frame: {e}");
                            break;
                        }
                        if right_slot_tx.send(slot).is_err() {
                            break;
                        }
                        frame_count += 1;
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

    // Pairing thread: zip left + right slot indices.
    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<(usize, usize)>(slot_channel_depth);
    std::thread::Builder::new()
        .name("d3d11_pair".into())
        .spawn(move || {
            while let (Ok(left_slot), Ok(right_slot)) = (left_slot_rx.recv(), right_slot_rx.recv())
            {
                if pair_tx.send((left_slot, right_slot)).is_err() {
                    break;
                }
            }
        })
        .expect("spawn D3D11VA pairing thread");

    Ok(D3d11DecodeHandles {
        pair_rx,
        views: Some(views),
        left_readback_tx: left_rb_req_tx,
        left_readback_rx: left_rb_res_rx,
        right_readback_tx: right_rb_req_tx,
        right_readback_rx: right_rb_res_rx,
    })
}
