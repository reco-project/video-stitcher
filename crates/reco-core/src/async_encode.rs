//! Async encode thread for pipelined video encoding.
//!
//! Wraps any [`Encoder`] and runs it on a dedicated thread, decoupling
//! the render loop from encoder latency. The caller submits NV12 data
//! via a bounded channel; the encode thread encodes frames in the
//! background. This is critical on Apple M4 where VideoToolbox encode
//! takes ~3.5ms/frame - without async encoding, this stall dominates
//! the frame time even though GPU readback is only ~0.5ms.
//!
//! ## Buffer pool
//!
//! To avoid allocating 3.1 MB (at 1080p) per frame for the channel
//! send, the thread maintains a pool of pre-allocated buffers. After
//! encoding, each buffer is returned to the pool for reuse.

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

use crate::encoder::{EncodeError, Encoder, OutputFrame, PixelFormat};

/// A frame payload sent to the encode thread.
struct EncodeJob {
    /// NV12 pixel data (borrowed from the buffer pool).
    data: Vec<u8>,
}

/// Async encoder that runs on a dedicated thread.
///
/// Created via [`new`](Self::new), which moves the encoder to a
/// background thread. Call [`submit`](Self::submit) to queue frames
/// for encoding, then [`finish`](Self::finish) to flush and join.
pub struct AsyncEncodeThread {
    /// Channel to send frames to the encode thread.
    /// Wrapped in Option so `finish()` can take ownership to drop it.
    tx: Option<SyncSender<EncodeJob>>,
    /// Channel to receive recycled buffers back from the encode thread.
    pool_rx: Option<Receiver<Vec<u8>>>,
    /// The encode thread handle.
    handle: Option<JoinHandle<Result<(), EncodeError>>>,
    /// Output dimensions (needed for OutputFrame construction).
    width: u32,
    height: u32,
}

impl AsyncEncodeThread {
    /// Create an async encode thread.
    ///
    /// Moves `encoder` to a background thread and pre-allocates
    /// `buffer_count + 1` NV12 buffers for zero-allocation submits.
    /// The `buffer_count` parameter controls how many frames can be
    /// in-flight between the render thread and the encode thread
    /// (typically 2).
    pub fn new(
        encoder: Box<dyn Encoder + Send>,
        width: u32,
        height: u32,
        buffer_count: usize,
    ) -> Self {
        let nv12_size = width as usize * height as usize * 3 / 2;
        let (tx, rx) = mpsc::sync_channel::<EncodeJob>(buffer_count);
        let (pool_tx, pool_rx) = mpsc::sync_channel::<Vec<u8>>(buffer_count + 1);

        // Pre-allocate buffer pool. buffer_count go into the pool channel,
        // +1 stays in reserve (the caller might hold one while sending).
        for _ in 0..buffer_count + 1 {
            let _ = pool_tx.try_send(vec![0u8; nv12_size]);
        }

        let handle = thread::Builder::new()
            .name("encode".into())
            .spawn(move || Self::encode_loop(rx, pool_tx, encoder, width, height))
            .expect("spawn encode thread");

        Self {
            tx: Some(tx),
            pool_rx: Some(pool_rx),
            handle: Some(handle),
            width,
            height,
        }
    }

    /// Submit NV12 data for encoding.
    ///
    /// Copies `nv12_data` into a pooled buffer and sends it to the
    /// encode thread. Blocks if the channel is full (backpressure).
    pub fn submit(&self, nv12_data: &[u8]) -> Result<(), EncodeError> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| EncodeError::Frame("encoder already finished".into()))?;
        let pool_rx = self.pool_rx.as_ref();

        // Try to get a recycled buffer from the pool, or allocate if empty.
        let mut buf = pool_rx
            .and_then(|rx| rx.try_recv().ok())
            .unwrap_or_else(|| {
                let nv12_size = self.width as usize * self.height as usize * 3 / 2;
                vec![0u8; nv12_size]
            });

        buf.resize(nv12_data.len(), 0);
        buf.copy_from_slice(nv12_data);

        tx.send(EncodeJob { data: buf })
            .map_err(|_| EncodeError::Frame("encode thread died".into()))
    }

    /// Flush all pending frames and shut down the encode thread.
    ///
    /// Drops the send channel (encode thread sees disconnect and
    /// calls `encoder.finish()`), then joins the thread and
    /// propagates any encoder error.
    pub fn finish(&mut self) -> Result<(), EncodeError> {
        // Drop sender so the encode thread's recv() returns Err and it finishes.
        self.tx.take();
        // Drop pool_rx so the encode thread's pool_tx sends don't block.
        self.pool_rx.take();

        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| EncodeError::Finalize("encode thread panicked".into()))?
        } else {
            Ok(())
        }
    }

    /// The encode thread's main loop.
    fn encode_loop(
        rx: Receiver<EncodeJob>,
        pool_tx: SyncSender<Vec<u8>>,
        mut encoder: Box<dyn Encoder + Send>,
        width: u32,
        height: u32,
    ) -> Result<(), EncodeError> {
        while let Ok(job) = rx.recv() {
            encoder.submit(OutputFrame {
                data: &job.data,
                width,
                height,
                format: PixelFormat::Nv12,
                pts_us: 0,
            })?;

            // Return the buffer to the pool for reuse.
            // If the pool channel is full or disconnected, just drop it.
            let _ = pool_tx.try_send(job.data);
        }

        encoder.finish()
    }
}

impl Drop for AsyncEncodeThread {
    fn drop(&mut self) {
        // If finish() wasn't called explicitly, join the thread on drop.
        // The tx is dropped by Drop order (before this), so the thread
        // will see disconnect and exit.
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
