//! Periodic JPEG snapshot writer for live preview.
//!
//! Writes a JPEG snapshot of the stitched NV12 output to a directory at
//! a configurable frame interval. Designed for the gameday control panel
//! which reads the latest `snapshot.jpg` to show a live preview without
//! waiting for the encoder to finish.
//!
//! The writer runs on a background thread with a capacity-1 channel so
//! it never blocks the 30fps frame loop. If the JPEG encoder falls
//! behind, old frames are silently dropped.

use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

/// NV12 tap closure handed to `StitchSession::set_nv12_tap`:
/// `(nv12_data, width, height)`.
type Nv12Tap = Box<dyn FnMut(&[u8], u32, u32) + Send>;

/// A snapshot job sent to the background thread.
struct SnapshotJob {
    nv12_data: Vec<u8>,
    width: u32,
    height: u32,
}

/// Writes periodic JPEG snapshots of the stitched output to disk.
///
/// Created once per camera session. Call [`create_nv12_tap`] to get a
/// closure suitable for [`StitchSession::set_nv12_tap`], then call
/// [`shutdown`] when the session ends.
pub struct SnapshotWriter {
    /// Kept alive to prevent the background thread from exiting early.
    /// Dropped on shutdown to signal the thread to stop.
    _tx: Option<mpsc::SyncSender<SnapshotJob>>,
    handle: Option<JoinHandle<()>>,
}

impl SnapshotWriter {
    /// Create a snapshot writer and its NV12 tap closure.
    ///
    /// Returns `(writer, tap)` where `tap` is a closure that can be
    /// passed to `session.set_nv12_tap()`. The tap internally tracks
    /// the frame interval and uses `try_send` so it never blocks.
    ///
    /// The directory is created if it does not exist.
    pub fn new(dir: &Path, interval: u64) -> std::io::Result<(Self, Nv12Tap)> {
        std::fs::create_dir_all(dir)?;

        let snapshot_path = dir.join("snapshot.jpg");
        let tmp_path = dir.join(".snapshot.jpg.tmp");

        // Capacity 1: the tap can always try_send without blocking.
        // If the background thread is still encoding the previous
        // snapshot, the new frame is silently dropped.
        let (tx, rx) = mpsc::sync_channel::<SnapshotJob>(1);

        let handle = thread::Builder::new()
            .name("snapshot".into())
            .spawn(move || encode_loop(rx, &snapshot_path, &tmp_path))
            .expect("spawn snapshot thread");

        let tap_tx = tx.clone();
        let interval = interval.max(1);
        let mut frame_count: u64 = 0;

        let tap = Box::new(move |data: &[u8], w: u32, h: u32| {
            let count = frame_count;
            frame_count += 1;

            if !count.is_multiple_of(interval) {
                return;
            }

            let job = SnapshotJob {
                nv12_data: data.to_vec(),
                width: w,
                height: h,
            };
            // try_send: drop the frame if the channel is full.
            let _ = tap_tx.try_send(job);
        });

        let writer = Self {
            _tx: Some(tx),
            handle: Some(handle),
        };

        Ok((writer, tap))
    }

    /// Shut down the background thread gracefully.
    pub fn shutdown(&mut self) {
        // Drop the sender so the background thread's recv() returns Err.
        self._tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
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
