//! Periodic JPEG snapshot sink for live preview.
//!
//! An [`OutputSink`] that writes a JPEG snapshot of the stitched NV12
//! output to a directory at a configurable frame interval. Designed
//! for the gameday control panel which reads the latest `snapshot.jpg`
//! to show a live preview without waiting for the encoder to finish.
//!
//! Attach with inline delivery
//! ([`SinkOptions::inline_lossy`](reco_core::session::SinkOptions::inline_lossy)):
//! `consume` hands the frame to a background thread over a capacity-1
//! channel so it never blocks the 30fps frame loop. If the JPEG
//! encoder falls behind, old frames are silently dropped.

use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use reco_core::sink::{OutputFrame, OutputSink, PixelFormat, SinkError, SinkInput};

/// A snapshot job sent to the background thread.
struct SnapshotJob {
    nv12_data: Vec<u8>,
    width: u32,
    height: u32,
}

/// Writes periodic JPEG snapshots of the stitched output to disk.
///
/// Created once per camera session and attached via
/// `StitchSession::add_sink`; the session finalizes it with the other
/// sinks.
pub struct SnapshotWriter {
    /// Kept alive to prevent the background thread from exiting early.
    /// Dropped on finish to signal the thread to stop.
    tx: Option<mpsc::SyncSender<SnapshotJob>>,
    handle: Option<JoinHandle<()>>,
    /// Write a snapshot every `interval` frames.
    interval: u64,
    frame_count: u64,
}

impl SnapshotWriter {
    /// Create a snapshot sink writing to `dir/snapshot.jpg`.
    ///
    /// The directory is created if it does not exist.
    pub fn new(dir: &Path, interval: u64) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;

        let snapshot_path = dir.join("snapshot.jpg");
        let tmp_path = dir.join(".snapshot.jpg.tmp");

        // Capacity 1: consume can always try_send without blocking.
        // If the background thread is still encoding the previous
        // snapshot, the new frame is silently dropped.
        let (tx, rx) = mpsc::sync_channel::<SnapshotJob>(1);

        let handle = thread::Builder::new()
            .name("snapshot".into())
            .spawn(move || encode_loop(rx, &snapshot_path, &tmp_path))
            .expect("spawn snapshot thread");

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            interval: interval.max(1),
            frame_count: 0,
        })
    }

    /// Shut down the background thread gracefully.
    fn shutdown(&mut self) {
        // Drop the sender so the background thread's recv() returns Err.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl OutputSink for SnapshotWriter {
    fn name(&self) -> &str {
        "snapshot"
    }

    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(PixelFormat::Nv12)
    }

    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        let count = self.frame_count;
        self.frame_count += 1;

        if !count.is_multiple_of(self.interval) {
            return Ok(());
        }
        let Some(tx) = self.tx.as_ref() else {
            return Ok(());
        };

        let job = SnapshotJob {
            nv12_data: frame.data.to_vec(),
            width: frame.width,
            height: frame.height,
        };
        // try_send: drop the frame if the channel is full.
        let _ = tx.try_send(job);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        self.shutdown();
        Ok(())
    }
}

impl Drop for SnapshotWriter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Background thread loop: receive NV12 frames, convert to RGB, encode
/// JPEG, and write atomically.
fn encode_loop(rx: mpsc::Receiver<SnapshotJob>, snapshot_path: &Path, tmp_path: &Path) {
    while let Ok(job) = rx.recv() {
        if let Err(e) = write_snapshot(&job, snapshot_path, tmp_path) {
            log::warn!("snapshot write failed: {e}");
        }
    }
}

/// Convert NV12 to RGB, encode as JPEG, and write atomically.
fn write_snapshot(job: &SnapshotJob, snapshot_path: &Path, tmp_path: &Path) -> anyhow::Result<()> {
    let w = job.width as usize;
    let h = job.height as usize;
    let y_size = w * h;

    // NV12 layout: Y plane (w*h bytes), then interleaved UV plane (w*h/2 bytes).
    if job.nv12_data.len() < y_size + y_size / 2 {
        anyhow::bail!(
            "NV12 data too short: expected {} bytes, got {}",
            y_size + y_size / 2,
            job.nv12_data.len()
        );
    }

    let y_plane = &job.nv12_data[..y_size];
    let uv_plane = &job.nv12_data[y_size..];

    let mut rgb = vec![0u8; w * h * 3];

    for row in 0..h {
        for col in 0..w {
            let yi = row * w + col;
            let uv_idx = (row / 2) * w + (col & !1);

            let y = y_plane[yi] as f32;
            let u = uv_plane[uv_idx] as f32;
            let v = uv_plane[uv_idx + 1] as f32;

            let r = y + 1.402 * (v - 128.0);
            let g = y - 0.344 * (u - 128.0) - 0.714 * (v - 128.0);
            let b = y + 1.772 * (u - 128.0);

            let pi = yi * 3;
            rgb[pi] = r.clamp(0.0, 255.0) as u8;
            rgb[pi + 1] = g.clamp(0.0, 255.0) as u8;
            rgb[pi + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }

    // Encode JPEG and write atomically (tmp + rename).
    let img = image::RgbImage::from_raw(job.width, job.height, rgb).ok_or_else(|| {
        anyhow::anyhow!(
            "failed to create image buffer ({}x{})",
            job.width,
            job.height
        )
    })?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(tmp_path)?);
    img.write_to(&mut out, image::ImageFormat::Jpeg)?;
    drop(out);
    std::fs::rename(tmp_path, snapshot_path)?;

    Ok(())
}
