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

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::encoder::{EncodeError, Encoder, OutputFrame, PixelFormat};

/// A frame payload sent to the encode thread.
struct EncodeJob {
    /// NV12 pixel data (borrowed from the buffer pool).
    data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    pts_us: i64,
}

/// Shared encode-thread counters. The worker records the true (overlapped)
/// encode cost; `submit` records backpressure stalls (the encoder being the
/// real bottleneck). Distinct from the pipeline's per-frame "encode" timing,
/// which only measures the submit memcpy + enqueue.
#[derive(Default)]
struct EncodeStats {
    encode_busy_ns: AtomicU64,
    frames_encoded: AtomicU64,
    backpressure_ns: AtomicU64,
    backpressure_count: AtomicU64,
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
    /// Encode-thread counters (shared with the worker).
    stats: Arc<EncodeStats>,
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

        let stats = Arc::new(EncodeStats::default());
        let worker_stats = Arc::clone(&stats);
        let handle = thread::Builder::new()
            .name("encode".into())
            .spawn(move || Self::encode_loop(rx, pool_tx, encoder, width, height, worker_stats))
            .expect("spawn encode thread");

        Self {
            tx: Some(tx),
            pool_rx: Some(pool_rx),
            handle: Some(handle),
            stats,
            width,
            height,
        }
    }

    /// Submit NV12 data for encoding.
    ///
    /// Copies `nv12_data` into a pooled buffer and sends it to the
    /// encode thread. Blocks if the channel is full (backpressure).
    /// `pts_us` is the presentation timestamp in microseconds.
    pub fn submit(&self, nv12_data: &[u8], pts_us: i64) -> Result<(), EncodeError> {
        profile_scope!("async_encode_submit");
        let tx = self.tx.as_ref().ok_or_else(|| EncodeError::Frame {
            frame_index: None,
            reason: "encoder already finished".into(),
        })?;
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

        let dead = || EncodeError::Frame {
            frame_index: None,
            reason: "encode thread died".into(),
        };
        // Try non-blocking first; only a full channel means the encoder is
        // the bottleneck. Measure that stall.
        match tx.try_send(EncodeJob { data: buf, pts_us }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(job)) => {
                let t0 = Instant::now();
                let r = tx.send(job).map_err(|_| dead());
                self.stats
                    .backpressure_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                self.stats
                    .backpressure_count
                    .fetch_add(1, Ordering::Relaxed);
                r
            }
            Err(TrySendError::Disconnected(_)) => Err(dead()),
        }
    }

    /// Snapshot of encode-thread counters: (frames, avg encode ms,
    /// backpressure stalls, total backpressure ms).
    pub fn stats(&self) -> (u64, f32, u64, f32) {
        let frames = self.stats.frames_encoded.load(Ordering::Relaxed);
        let busy_ns = self.stats.encode_busy_ns.load(Ordering::Relaxed);
        let avg_ms = if frames > 0 {
            (busy_ns as f64 / frames as f64 / 1e6) as f32
        } else {
            0.0
        };
        let bp_count = self.stats.backpressure_count.load(Ordering::Relaxed);
        let bp_ms = (self.stats.backpressure_ns.load(Ordering::Relaxed) as f64 / 1e6) as f32;
        (frames, avg_ms, bp_count, bp_ms)
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

        let result = if let Some(handle) = self.handle.take() {
            handle.join().map_err(|_| EncodeError::Finalize {
                reason: "encode thread panicked".into(),
            })?
        } else {
            Ok(())
        };

        let (frames, avg_ms, bp_count, bp_ms) = self.stats();
        if frames > 0 {
            log::info!(
                "Encode thread: {frames} frames, avg encode {avg_ms:.2}ms (overlapped); \
                 backpressure {bp_count} stalls totaling {bp_ms:.1}ms"
            );
        }
        result
    }

    /// The encode thread's main loop.
    fn encode_loop(
        rx: Receiver<EncodeJob>,
        pool_tx: SyncSender<Vec<u8>>,
        mut encoder: Box<dyn Encoder + Send>,
        width: u32,
        height: u32,
        stats: Arc<EncodeStats>,
    ) -> Result<(), EncodeError> {
        while let Ok(job) = rx.recv() {
            profile_scope!("encode_submit");
            let t0 = Instant::now();
            encoder.submit(OutputFrame {
                data: &job.data,
                width,
                height,
                format: PixelFormat::Nv12,
                pts_us: job.pts_us,
            })?;
            stats
                .encode_busy_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            stats.frames_encoded.fetch_add(1, Ordering::Relaxed);

            // Return the buffer to the pool for reuse.
            // If the pool channel is full or disconnected, just drop it.
            let _ = pool_tx.try_send(job.data);
        }

        encoder.finish()
    }
}

impl Drop for AsyncEncodeThread {
    fn drop(&mut self) {
        // Drop the sender (and pool receiver) BEFORE joining. Struct
        // fields are dropped AFTER this body runs, so `self.tx` is still
        // alive here; if we joined first, the encode thread's `rx.recv()`
        // would never see a disconnect and the join would block forever.
        // This matters on early-error paths (e.g. a pre-flight VRAM budget
        // failure) where `finish()` was never called.
        self.tx.take();
        self.pool_rx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopEncoder;
    impl Encoder for NoopEncoder {
        fn submit(&mut self, _f: OutputFrame<'_>) -> Result<(), EncodeError> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), EncodeError> {
            Ok(())
        }
    }

    #[test]
    fn counts_encoded_frames() {
        let mut t = AsyncEncodeThread::new(Box::new(NoopEncoder), 16, 16, 2);
        let data = vec![0u8; 16 * 16 * 3 / 2];
        for i in 0..8 {
            t.submit(&data, i).unwrap();
        }
        t.finish().unwrap();
        let (frames, avg_ms, _bp, _bp_ms) = t.stats();
        assert_eq!(frames, 8);
        assert!(avg_ms.is_finite() && avg_ms >= 0.0);
    }
}
