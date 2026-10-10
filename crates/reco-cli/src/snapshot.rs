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
//! channel without waiting for the JPEG encoder. If the channel is
//! full, the newly submitted snapshot is dropped.

use std::io::Write;
use std::path::Path;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use reco_core::render::color::bt709_yuv_to_rgb;
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
            .spawn(move || encode_loop(rx, &snapshot_path, &tmp_path))?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            interval: interval.max(1),
            frame_count: 0,
        })
    }

    /// Shut down the background thread gracefully.
    fn shutdown(&mut self) -> Result<(), SinkError> {
        // Drop the sender so the background thread's recv() returns Err.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|_| SinkError::Finish {
                reason: "snapshot worker panicked".into(),
            })?;
        }
        Ok(())
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
        let reject = |reason: &str| SinkError::Consume {
            frame_index: None,
            reason: reason.into(),
        };
        let Some(tx) = self.tx.as_ref() else {
            return Err(reject("snapshot writer is finished"));
        };
        let count = self.frame_count;
        self.frame_count = count.wrapping_add(1);

        if !count.is_multiple_of(self.interval) {
            return Ok(());
        }
        let pixels = (frame.width as usize).checked_mul(frame.height as usize);
        let expected = pixels.and_then(|n| n.checked_add(n / 2));
        if frame.format != PixelFormat::Nv12
            || frame.width == 0
            || frame.height == 0
            || !frame.width.is_multiple_of(2)
            || !frame.height.is_multiple_of(2)
            || expected != Some(frame.data.len())
            || pixels.and_then(|n| n.checked_mul(3)).is_none()
        {
            return Err(reject("snapshot requires tightly packed, even-sized NV12"));
        }

        let job = SnapshotJob {
            nv12_data: frame.data.to_vec(),
            width: frame.width,
            height: frame.height,
        };
        // try_send: drop the frame if the channel is full.
        match tx.try_send(job) {
            Ok(()) | Err(mpsc::TrySendError::Full(_)) => Ok(()),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(reject("snapshot worker stopped")),
        }
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        self.shutdown()
    }
}

impl Drop for SnapshotWriter {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            log::warn!("{error}");
        }
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

    // consume validates the tightly packed NV12 layout before queuing.
    let y_plane = &job.nv12_data[..y_size];
    let uv_plane = &job.nv12_data[y_size..];

    let mut rgb = vec![0u8; w * h * 3];

    for row in 0..h {
        for col in 0..w {
            let yi = row * w + col;
            let uv_idx = (row / 2) * w + (col & !1);

            // Both core output converters produce BT.709 limited-range NV12.
            let colour = bt709_yuv_to_rgb(
                y_plane[yi] as f64 / 255.0,
                uv_plane[uv_idx] as f64 / 255.0,
                uv_plane[uv_idx + 1] as f64 / 255.0,
                false,
            );
            let pi = yi * 3;
            rgb[pi..pi + 3].copy_from_slice(&colour.map(|c| (c * 255.0).round() as u8));
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
    out.flush()?;
    drop(out);
    std::fs::rename(tmp_path, snapshot_path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "reco-snapshot-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn solid(rgb: [u8; 3]) -> Vec<u8> {
        let rgba = [rgb[0], rgb[1], rgb[2], 255].repeat(64 * 64);
        let mut nv12 = Vec::new();
        reco_core::render::nv12_cpu::rgba_to_nv12_into(&rgba, 64, 64, 64, &mut nv12).unwrap();
        nv12
    }

    fn frame(data: &[u8]) -> OutputFrame<'_> {
        OutputFrame {
            data,
            width: 64,
            height: 64,
            format: PixelFormat::Nv12,
            pts_us: None,
        }
    }

    #[test]
    fn snapshots_round_trip_rendered_colours_and_replace_previous_file() {
        let dir = Directory::new();
        for rgb in [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [50, 150, 40],
        ] {
            let mut writer = SnapshotWriter::new(&dir.0, 1).unwrap();
            let data = solid(rgb);
            writer.consume(frame(&data)).unwrap();
            writer.finish().unwrap();
            writer.finish().unwrap();
            assert!(writer.consume(frame(&data)).is_err());
            let image = image::open(dir.0.join("snapshot.jpg")).unwrap().to_rgb8();
            assert_eq!(image.dimensions(), (64, 64));
            let pixel = image.get_pixel(32, 32).0;
            for (actual, expected) in pixel.into_iter().zip(rgb) {
                assert!(actual.abs_diff(expected) <= 3, "{rgb:?}: {pixel:?}");
            }
            assert!(!dir.0.join(".snapshot.jpg.tmp").exists());
        }
    }

    #[test]
    fn drop_finishes_the_queued_snapshot() {
        let dir = Directory::new();
        let mut writer = SnapshotWriter::new(&dir.0, 0).unwrap();
        writer.consume(frame(&solid([0, 0, 0]))).unwrap();
        drop(writer);
        assert!(image::open(dir.0.join("snapshot.jpg")).is_ok());
    }

    #[test]
    fn cadence_and_full_queue_are_bounded_without_an_encoder_thread() {
        let (tx, rx) = mpsc::sync_channel(1);
        let mut writer = SnapshotWriter {
            tx: Some(tx),
            handle: None,
            interval: 3,
            frame_count: 0,
        };
        let data = solid([50, 150, 40]);
        writer.consume(frame(&data)).unwrap(); // 0: accepted
        for _ in 0..6 {
            writer.consume(frame(&data)).unwrap(); // 3 and 6: full
        }
        assert_eq!(rx.try_recv().unwrap().nv12_data, data);
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        for _ in 0..2 {
            writer.consume(frame(&data)).unwrap(); // 7 and 8: skipped
        }
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        writer.consume(frame(&data)).unwrap(); // 9: accepted
        rx.try_recv().unwrap();
        drop(rx);
        writer.frame_count = 12;
        assert!(writer.consume(frame(&data)).is_err());
    }

    #[test]
    fn malformed_frames_are_rejected_before_the_worker() {
        let (tx, rx) = mpsc::sync_channel(1);
        let mut writer = SnapshotWriter {
            tx: Some(tx),
            handle: None,
            interval: 1,
            frame_count: 0,
        };
        let data = solid([0, 0, 0]);
        for (width, height, format, len) in [
            (64, 64, PixelFormat::Rgba8, data.len()),
            (0, 64, PixelFormat::Nv12, 0),
            (63, 64, PixelFormat::Nv12, data.len()),
            (64, 63, PixelFormat::Nv12, data.len()),
            (64, 64, PixelFormat::Nv12, data.len() - 1),
            (2, 2, PixelFormat::Nv12, data.len()),
            (u32::MAX - 1, u32::MAX - 1, PixelFormat::Nv12, 0),
        ] {
            assert!(
                writer
                    .consume(OutputFrame {
                        data: &data[..len],
                        width,
                        height,
                        format,
                        pts_us: None,
                    })
                    .is_err()
            );
        }
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    }

    #[test]
    fn worker_panic_is_reported_at_finish() {
        let mut writer = SnapshotWriter {
            tx: None,
            handle: Some(thread::spawn(|| panic!("test worker failure"))),
            interval: 1,
            frame_count: 0,
        };
        assert!(matches!(writer.finish(), Err(SinkError::Finish { .. })));
        writer.finish().unwrap();
    }

    #[test]
    fn failed_write_preserves_the_last_snapshot() {
        let dir = Directory::new();
        let snapshot = dir.0.join("snapshot.jpg");
        let temporary = dir.0.join(".snapshot.jpg.tmp");
        std::fs::write(&snapshot, b"previous snapshot").unwrap();
        std::fs::create_dir(&temporary).unwrap();
        let job = SnapshotJob {
            nv12_data: solid([0, 0, 0]),
            width: 64,
            height: 64,
        };
        assert!(write_snapshot(&job, &snapshot, &temporary).is_err());
        assert_eq!(std::fs::read(&snapshot).unwrap(), b"previous snapshot");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn flush_failure_does_not_publish_a_partial_snapshot() {
        let dir = Directory::new();
        let snapshot = dir.0.join("snapshot.jpg");
        let temporary = dir.0.join(".snapshot.jpg.tmp");
        std::fs::write(&snapshot, b"previous snapshot").unwrap();
        std::os::unix::fs::symlink("/dev/full", &temporary).unwrap();
        let job = SnapshotJob {
            nv12_data: solid([0, 0, 0]),
            width: 64,
            height: 64,
        };
        assert!(write_snapshot(&job, &snapshot, &temporary).is_err());
        assert_eq!(std::fs::read(&snapshot).unwrap(), b"previous snapshot");
        assert!(std::fs::symlink_metadata(&temporary).unwrap().is_symlink());
    }
}
