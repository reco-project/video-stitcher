//! Decode workers for zero-copy GPU paths.
//!
//! Frame messages are followed by explicit EOF or failure. CUDA copies finish
//! before a slot is published; Metal and D3D11 messages retain their surfaces.

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
use crate::decode_stream::{Message, pair, spawn};
#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
use crate::ffmpeg::decoder::{DecodeError, VideoDecoder};

/// Spawn a single-video GPU decode worker that copies frames into shared textures.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn spawn_single_decoder_gpu(
    input: crate::stitch_job::InputPath,
    label: &'static str,
    buf: reco_core::interop::zero_copy::GpuBufInfo,
    slot_free_rx: std::sync::mpsc::Receiver<u8>,
    skip_frames: u64,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> (
    std::sync::mpsc::Receiver<Message<u8>>,
    std::thread::JoinHandle<()>,
) {
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

            loop {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let slot = match slot_free_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(slot) => slot,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                };
                let Some(frame) = dec.next_frame_gpu()? else {
                    break;
                };
                // Shared textures have a fixed layout. Never copy a changed frame
                // using stale extents or sample widths.
                if frame.width != buf.width
                    || frame.height != buf.height
                    || frame.pixel_format != buf.pixel_format
                {
                    return Err(DecodeError::ConversionError(
                        "decoded frame layout does not match shared CUDA textures".into(),
                    ));
                }
                let s = usize::from(slot);
                reco_core::interop::cuda::cuda_ensure_context()
                    .map_err(|e| DecodeError::ConversionError(e.to_string()))?;
                let width_bytes = buf.width as usize * buf.pixel_format.bytes_per_sample();
                reco_core::interop::cuda::cuda_2d_copy(
                    buf.y_ptr[s],
                    buf.y_pitch[s],
                    frame.y_ptr,
                    frame.y_pitch,
                    width_bytes,
                    buf.height as usize,
                )
                .map_err(|e| DecodeError::ConversionError(format!("CUDA Y copy: {e}")))?;
                reco_core::interop::cuda::cuda_2d_copy(
                    buf.uv_ptr[s],
                    buf.uv_pitch[s],
                    frame.uv_ptr,
                    frame.uv_pitch,
                    width_bytes,
                    buf.height as usize / 2,
                )
                .map_err(|e| DecodeError::ConversionError(format!("CUDA UV copy: {e}")))?;
                reco_core::interop::cuda::cuda_synchronize()
                    .map_err(|e| DecodeError::ConversionError(e.to_string()))?;
                if tx.send(Ok(Some(slot))).is_err() {
                    break;
                }
            }
            Ok(())
        },
    )
}

/// Spawn parallel GPU decode threads and a pairing thread.
///
/// Positive `sync_offset` skips right frames; negative skips left frames.
/// The receiver emits paired slots, then explicit EOF or an error. Join all
/// workers before dropping the shared textures.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn spawn_decode_threads_gpu(
    left_input: crate::stitch_job::InputPath,
    right_input: crate::stitch_job::InputPath,
    left_buf: reco_core::interop::zero_copy::GpuBufInfo,
    right_buf: reco_core::interop::zero_copy::GpuBufInfo,
    left_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    right_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    sync_offset: i64,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> reco_core::interop::zero_copy::GpuDecodeHandles {
    use reco_core::interop::zero_copy::{GpuDecodeHandles, GpuFrameSignal};

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
    let (rx, pair_handle) = pair(left_rx, right_rx, 0, 1, |(left_slot, right_slot)| {
        GpuFrameSignal {
            left_slot,
            right_slot,
        }
    });
    GpuDecodeHandles {
        frame_rx: rx,
        join_handles: vec![left_handle, right_handle, pair_handle],
    }
}

#[cfg(target_os = "macos")]
fn spawn_vt_decode_thread(
    input: crate::stitch_job::InputPath,
    label: &'static str,
) -> std::sync::mpsc::Receiver<Message<reco_core::interop::metal::RetainedCVPixelBuffer>> {
    use reco_core::interop::metal::RetainedCVPixelBuffer;

    spawn(
        format!("vt_decode_{label}"),
        4,
        move |tx| -> Result<(), DecodeError> {
            let mut dec = VideoDecoder::open_input(&input)?;
            log::info!("VT decode thread {label}: backend={}", dec.backend());
            while let Some(vt) = dec.next_frame_vt()? {
                let retained = unsafe { RetainedCVPixelBuffer::retain(vt.cv_pixel_buffer) };
                if tx.send(Ok(Some(retained))).is_err() {
                    break;
                }
            }
            Ok(())
        },
    )
    .0
}

/// Spawn paired VideoToolbox decode threads.
///
/// Positive `sync_offset` skips right frames; negative skips left frames.
/// The receiver emits retained buffer pairs, then explicit EOF or an error.
#[cfg(target_os = "macos")]
pub fn spawn_vt_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    sync_offset: i64,
) -> std::sync::mpsc::Receiver<Message<reco_core::interop::zero_copy::VtFramePair>> {
    use reco_core::interop::zero_copy::VtFramePair;

    let left_rx = spawn_vt_decode_thread(left.clone(), "left");
    let right_rx = spawn_vt_decode_thread(right.clone(), "right");
    pair(left_rx, right_rx, sync_offset, 4, |(left, right)| {
        VtFramePair { left, right }
    })
    .0
}

#[cfg(target_os = "windows")]
fn spawn_d3d11_decode_thread_shared(
    input: crate::stitch_job::InputPath,
    label: &'static str,
    shared_device: crate::ffmpeg::decoder::SharedHwDevice,
) -> std::sync::mpsc::Receiver<Message<crate::ffmpeg::decoder::D3d11Frame>> {
    spawn(
        format!("d3d11_decode_{label}"),
        4,
        move |tx| -> Result<(), DecodeError> {
            let mut dec = VideoDecoder::open_input_with_shared_device(&input, &shared_device)?;
            log::info!("D3D11VA decode thread {label}: backend={}", dec.backend());
            while let Some(frame) = dec.next_frame_d3d11()? {
                if tx.send(Ok(Some(frame))).is_err() {
                    break;
                }
            }
            Ok(())
        },
    )
    .0
}

#[cfg(target_os = "windows")]
pub(crate) type D3d11FramePair = (
    crate::ffmpeg::decoder::D3d11Frame,
    crate::ffmpeg::decoder::D3d11Frame,
);

/// Spawn paired D3D11VA decode threads on a shared hardware device.
///
/// Both decoders must use the same `ID3D11Device` for staging copies.
/// Positive `sync_offset` skips right frames; negative skips left frames.
/// The receiver emits retained texture pairs, then explicit EOF or an error.
#[cfg(target_os = "windows")]
pub fn spawn_d3d11_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    sync_offset: i64,
) -> std::sync::mpsc::Receiver<Message<D3d11FramePair>> {
    let Some(shared_device) = crate::ffmpeg::decoder::create_shared_hw_device() else {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = tx.send(Err(reco_core::source::SourceError::Init {
            path: "D3D11VA".into(),
            reason: "hardware device creation failed".into(),
        }));
        return rx;
    };
    let left_rx = spawn_d3d11_decode_thread_shared(left.clone(), "left", shared_device.new_ref());
    let right_rx =
        spawn_d3d11_decode_thread_shared(right.clone(), "right", shared_device.new_ref());
    pair(left_rx, right_rx, sync_offset, 4, |frames| frames).0
}
