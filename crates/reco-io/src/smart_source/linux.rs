//! Linux CUDA/Vulkan zero-copy native file source.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;

use reco_core::gpu::GpuContext;
use reco_core::interop::SharedTextureSet;
use reco_core::render::renderer::GpuPixelFormat;
use reco_core::source::{FrameSet, SourceError, SourceFrame, SourceInfo};

use super::SkipOutcome;
use crate::decode_stream::Message;
use crate::ffmpeg::decoder::DecodeBackend;
use crate::stitch_job::InputPath;

pub(super) struct NativeSource {
    /// Frame-set receiver. Drop takes it early to unblock the collector
    /// before joining decode workers.
    frame_rx: Option<Receiver<Message<Vec<u8>>>>,
    /// Shared textures (kept alive until Drop).
    shared: SharedTextureSet,
    /// Decode thread join handles.
    join_handles: Vec<JoinHandle<()>>,
    /// Shutdown flag checked by decode threads for graceful exit.
    shutdown: Arc<AtomicBool>,
}

impl NativeSource {
    pub(super) fn supports_backend(backend: DecodeBackend) -> bool {
        matches!(backend, DecodeBackend::Cuda)
    }

    pub(super) fn allows_mixed_camera_sizes() -> bool {
        true
    }

    pub(super) fn supports_input_count(count: usize) -> bool {
        matches!(count, 1 | 2)
    }

    pub(super) fn try_open(
        inputs: &[InputPath],
        gpu: &GpuContext,
        sync_offset: i64,
        info: &SourceInfo,
        pixel_format: GpuPixelFormat,
        _full_range: bool,
    ) -> Result<Option<Self>, SourceError> {
        let (shared, receivers) =
            crate::zero_copy::cuda::allocate(gpu, &info.camera_sizes, pixel_format).map_err(
                |error| SourceError::Init {
                    path: inputs[0].first_path().display().to_string(),
                    reason: error.to_string(),
                },
            )?;
        let decoders = inputs
            .iter()
            .zip(&shared.cameras)
            .zip(receivers)
            .map(|((input, camera), free)| (input.clone(), camera.buf.clone(), free))
            .collect();
        let shutdown = Arc::new(AtomicBool::new(false));
        let decode_handles =
            crate::zero_copy::spawn_decode_threads_gpu(decoders, sync_offset, shutdown.clone());
        log::info!(
            "SmartFileSource: GPU zero-copy ({:?}, {pixel_format:?})",
            info.camera_sizes
        );

        Ok(Some(Self {
            frame_rx: Some(decode_handles.frame_rx),
            shared,
            join_handles: decode_handles.join_handles,
            shutdown,
        }))
    }

    pub(super) fn decode_mode(&self) -> &'static str {
        "GPU zero-copy (CUDA/Vulkan)"
    }

    pub(super) fn shared_texture_set(&self) -> &SharedTextureSet {
        &self.shared
    }

    pub(super) fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        let rx = self
            .frame_rx
            .as_ref()
            .expect("frame_rx taken during shutdown");
        match crate::decode_stream::receive(rx)? {
            Some(signal) => Ok(Some(SourceFrame::untimed(FrameSet::GpuResident {
                slots: signal,
            }))),
            None => Ok(None),
        }
    }

    pub(super) fn skip_frames(
        &mut self,
        count: u64,
        _info: &SourceInfo,
    ) -> Result<SkipOutcome, SourceError> {
        let rx = self
            .frame_rx
            .as_ref()
            .expect("frame_rx taken during shutdown");
        for i in 0..count {
            match crate::decode_stream::receive(rx)? {
                Some(signal) => {
                    for (camera, slot) in self.shared.cameras.iter().zip(signal) {
                        let _ = camera.slot_free_tx.send(slot);
                    }
                }
                None => {
                    return Ok(SkipOutcome {
                        skipped: i,
                        exhausted: true,
                    });
                }
            }
        }
        log::debug!("GPU zero-copy: skipped {count} frames for start_time");
        Ok(SkipOutcome {
            skipped: count,
            exhausted: false,
        })
    }

    pub(super) fn start_decoding(&mut self) {}
}

impl Drop for NativeSource {
    fn drop(&mut self) {
        // Graceful shutdown ordering to prevent CUDA error 700:
        //
        // 1. Set shutdown flag -> decode threads exit via recv_timeout check
        // 2. Drop frame_rx -> pairing thread's send() returns Err, exits
        // 3. Join all threads -> VideoDecoder::Drop completes CUDA cleanup
        //    while shared Vulkan memory is still mapped into CUDA
        // 4. SharedTextureSet drops naturally -> CUDA memory unmapped

        // Step 1: Signal shutdown to decode threads (they check this
        // flag via recv_timeout, so they exit even if slot senders
        // are still held by the session).
        self.shutdown.store(true, Ordering::Release);

        // Step 2: Drop frame_rx to unblock pairing thread
        self.frame_rx = None;

        // Step 3: Join decode + pairing threads
        for handle in self.join_handles.drain(..) {
            let _ = handle.join();
        }
        // SharedTextureSet drops naturally after this
    }
}
