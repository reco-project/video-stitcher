//! Dedicated delivery thread for lossless output sinks.
//!
//! Wraps any [`OutputSink`] and runs it on a dedicated thread,
//! decoupling the render loop from sink latency. The caller submits
//! NV12 data via a bounded channel; the thread consumes frames in the
//! background. This is critical on Apple M4 where VideoToolbox encode
//! takes ~3.5ms/frame - without the thread, this stall dominates the
//! frame time even though GPU readback is only ~0.5ms.
//!
//! ## Buffer pool
//!
//! To avoid allocating 3.1 MB (at 1080p) per frame for the channel
//! send, the thread maintains a pool of pre-allocated buffers. After
//! the sink consumes a frame, each buffer is returned to the pool for
//! reuse.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::sink::{OutputFrame, OutputSink, PixelFormat, SinkError};

/// A frame payload sent to the sink thread.
struct SinkJob {
    /// NV12 pixel data (borrowed from the buffer pool).
    data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    pts_us: i64,
}

/// Shared sink-thread counters. The worker records the true
/// (overlapped) consume cost; `submit` records backpressure stalls
/// (the sink being the real bottleneck). Distinct from the pipeline's
/// per-frame "encode" timing, which only measures the submit memcpy +
/// enqueue.
#[derive(Default)]
struct SinkStats {
    consume_busy_ns: AtomicU64,
    frames_consumed: AtomicU64,
    backpressure_ns: AtomicU64,
    backpressure_count: AtomicU64,
}

/// A sink running on a dedicated thread.
///
/// Created via [`new`](Self::new), which moves the sink to a
/// background thread. Call [`submit`](Self::submit) to queue frames,
/// then [`finish`](Self::finish) to flush and join.
pub struct SinkThread {
    /// Channel to send frames to the sink thread.
    /// Wrapped in Option so `finish()` can take ownership to drop it.
    tx: Option<SyncSender<SinkJob>>,
    /// Channel to receive recycled buffers back from the sink thread.
    pool_rx: Option<Receiver<Vec<u8>>>,
    /// The sink thread handle.
    handle: Option<JoinHandle<Result<(), SinkError>>>,
    /// Output dimensions (needed for OutputFrame construction).
    width: u32,
    height: u32,
    /// Sink-thread counters (shared with the worker).
    stats: Arc<SinkStats>,
}

impl SinkThread {
    /// Create a sink thread.
    ///
    /// Moves `sink` to a background thread and pre-allocates
    /// `queue_depth + 1` NV12 buffers for zero-allocation submits.
    /// The `queue_depth` parameter controls how many frames can be
    /// in-flight between the render thread and the sink thread
    /// (typically 2).
    pub fn new(sink: Box<dyn OutputSink>, width: u32, height: u32, queue_depth: usize) -> Self {
        let nv12_size = width as usize * height as usize * 3 / 2;
        let (tx, rx) = mpsc::sync_channel::<SinkJob>(queue_depth);
        let (pool_tx, pool_rx) = mpsc::sync_channel::<Vec<u8>>(queue_depth + 1);

        // Pre-allocate buffer pool. queue_depth go into the pool channel,
        // +1 stays in reserve (the caller might hold one while sending).
        for _ in 0..queue_depth + 1 {
            let _ = pool_tx.try_send(vec![0u8; nv12_size]);
        }

        let stats = Arc::new(SinkStats::default());
        let worker_stats = Arc::clone(&stats);
        let handle = thread::Builder::new()
            .name("sink".into())
            .spawn(move || Self::consume_loop(rx, pool_tx, sink, width, height, worker_stats))
            .expect("spawn sink thread");

        Self {
            tx: Some(tx),
            pool_rx: Some(pool_rx),
            handle: Some(handle),
            stats,
            width,
            height,
        }
    }

    /// Submit NV12 data to the sink.
    ///
    /// Copies `nv12_data` into a pooled buffer and sends it to the
    /// sink thread. Blocks if the channel is full (backpressure).
    /// `pts_us` is the presentation timestamp in microseconds.
    pub fn submit(&mut self, nv12_data: &[u8], pts_us: i64) -> Result<(), SinkError> {
        profile_scope!("sink_thread_submit");
        let tx = self.tx.as_ref().ok_or_else(|| SinkError::Consume {
            frame_index: None,
            reason: "sink already finished".into(),
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

        // Try non-blocking first; only a full channel means the sink is
        // the bottleneck. Measure that stall.
        match tx.try_send(SinkJob { data: buf, pts_us }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(job)) => {
                let t0 = Instant::now();
                let sent = tx.send(job).is_ok();
                self.stats
                    .backpressure_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                self.stats
                    .backpressure_count
                    .fetch_add(1, Ordering::Relaxed);
                if sent {
                    Ok(())
                } else {
                    Err(self.worker_error())
                }
            }
            Err(TrySendError::Disconnected(_)) => Err(self.worker_error()),
        }
    }

    /// The worker exited while frames were still being submitted: join
    /// it and surface the sink's real error (e.g. the codec rejecting a
    /// frame) instead of a generic "thread died". Without the join, the
    /// underlying error only lives in the JoinHandle and is discarded
    /// on Drop when the caller aborts before `finish()`.
    fn worker_error(&mut self) -> SinkError {
        let generic = || SinkError::Consume {
            frame_index: None,
            reason: "sink thread died".into(),
        };
        match self.handle.take() {
            Some(handle) => match handle.join() {
                Ok(Err(e)) => e,
                Ok(Ok(())) => generic(),
                Err(_) => SinkError::Finish {
                    reason: "sink thread panicked".into(),
                },
            },
            None => generic(),
        }
    }

    /// Snapshot of sink-thread counters: (frames, avg consume ms,
    /// backpressure stalls, total backpressure ms).
    pub fn stats(&self) -> (u64, f32, u64, f32) {
        let frames = self.stats.frames_consumed.load(Ordering::Relaxed);
        let busy_ns = self.stats.consume_busy_ns.load(Ordering::Relaxed);
        let avg_ms = if frames > 0 {
            (busy_ns as f64 / frames as f64 / 1e6) as f32
        } else {
            0.0
        };
        let bp_count = self.stats.backpressure_count.load(Ordering::Relaxed);
        let bp_ms = (self.stats.backpressure_ns.load(Ordering::Relaxed) as f64 / 1e6) as f32;
        (frames, avg_ms, bp_count, bp_ms)
    }

    /// Flush all pending frames and shut down the sink thread.
    ///
    /// Drops the send channel (the sink thread sees disconnect and
    /// calls `sink.finish()`), then joins the thread and propagates
    /// any sink error.
    pub fn finish(&mut self) -> Result<(), SinkError> {
        // Drop sender so the sink thread's recv() returns Err and it finishes.
        self.tx.take();
        // Drop pool_rx so the sink thread's pool_tx sends don't block.
        self.pool_rx.take();

        let result = if let Some(handle) = self.handle.take() {
            handle.join().map_err(|_| SinkError::Finish {
                reason: "sink thread panicked".into(),
            })?
        } else {
            Ok(())
        };

        let (frames, avg_ms, bp_count, bp_ms) = self.stats();
        if frames > 0 {
            log::info!(
                "Sink thread: {frames} frames, avg consume {avg_ms:.2}ms (overlapped); \
                 backpressure {bp_count} stalls totaling {bp_ms:.1}ms"
            );
        }
        result
    }

    /// The sink thread's main loop.
    fn consume_loop(
        rx: Receiver<SinkJob>,
        pool_tx: SyncSender<Vec<u8>>,
        mut sink: Box<dyn OutputSink>,
        width: u32,
        height: u32,
        stats: Arc<SinkStats>,
    ) -> Result<(), SinkError> {
        while let Ok(job) = rx.recv() {
            profile_scope!("sink_consume");
            let t0 = Instant::now();
            sink.consume(OutputFrame {
                data: &job.data,
                width,
                height,
                format: PixelFormat::Nv12,
                pts_us: job.pts_us,
            })?;
            stats
                .consume_busy_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            stats.frames_consumed.fetch_add(1, Ordering::Relaxed);

            // Return the buffer to the pool for reuse.
            // If the pool channel is full or disconnected, just drop it.
            let _ = pool_tx.try_send(job.data);
        }

        sink.finish()
    }
}

impl Drop for SinkThread {
    fn drop(&mut self) {
        // Drop the sender (and pool receiver) BEFORE joining. Struct
        // fields are dropped AFTER this body runs, so `self.tx` is still
        // alive here; if we joined first, the sink thread's `rx.recv()`
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
    use crate::sink::SinkInput;

    struct NoopSink;
    impl OutputSink for NoopSink {
        fn name(&self) -> &str {
            "noop"
        }
        fn wants(&self) -> SinkInput {
            SinkInput::CpuBytes(PixelFormat::Nv12)
        }
        fn consume(&mut self, _f: OutputFrame<'_>) -> Result<(), SinkError> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            Ok(())
        }
    }

    #[test]
    fn counts_consumed_frames() {
        let mut t = SinkThread::new(Box::new(NoopSink), 16, 16, 2);
        let data = vec![0u8; 16 * 16 * 3 / 2];
        for i in 0..8 {
            t.submit(&data, i).unwrap();
        }
        t.finish().unwrap();
        let (frames, avg_ms, _bp, _bp_ms) = t.stats();
        assert_eq!(frames, 8);
        assert!(avg_ms.is_finite() && avg_ms >= 0.0);
    }

    /// A dying worker must surface the sink's real error to the
    /// submitting thread, not a generic "thread died".
    #[test]
    fn dead_worker_surfaces_the_real_error() {
        struct FailingSink;
        impl OutputSink for FailingSink {
            fn name(&self) -> &str {
                "failing"
            }
            fn wants(&self) -> SinkInput {
                SinkInput::CpuBytes(PixelFormat::Nv12)
            }
            fn consume(&mut self, _f: OutputFrame<'_>) -> Result<(), SinkError> {
                Err(SinkError::Consume {
                    frame_index: Some(0),
                    reason: "codec rejected the frame".into(),
                })
            }
            fn finish(&mut self) -> Result<(), SinkError> {
                Ok(())
            }
        }

        let mut t = SinkThread::new(Box::new(FailingSink), 16, 16, 1);
        let data = vec![0u8; 16 * 16 * 3 / 2];
        // The first submits may succeed (queued before the worker
        // errors); keep going until the disconnect is observed.
        let err = (0..100)
            .find_map(|i| t.submit(&data, i).err())
            .expect("worker death must surface as a submit error");
        assert!(
            err.to_string().contains("codec rejected the frame"),
            "generic error masked the sink's real failure: {err}"
        );
    }
}
