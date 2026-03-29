//! Shared CLI helpers: calibration loading, progress reporting, platform detection.

use reco_core::calibration::MatchCalibration;
use std::path::Path;

/// Maximum calibration file size (1 MB) to prevent DoS from large files.
const MAX_CALIBRATION_SIZE: u64 = 1_048_576;

/// Load and validate a calibration file.
///
/// Checks file size, parses JSON, and runs validation. Returns a descriptive
/// error on any failure.
pub fn load_calibration(path: &Path) -> anyhow::Result<MatchCalibration> {
    let meta = std::fs::metadata(path)?;
    if meta.len() > MAX_CALIBRATION_SIZE {
        anyhow::bail!(
            "Calibration file too large ({} bytes, max {})",
            meta.len(),
            MAX_CALIBRATION_SIZE
        );
    }
    let json = std::fs::read_to_string(path)?;
    let cal: MatchCalibration = serde_json::from_str(&json)?;
    cal.validate()?;
    Ok(cal)
}

/// Progress reporter that prints frame count and FPS every N frames.
#[derive(Clone, Copy)]
pub struct ProgressReporter {
    start: std::time::Instant,
    interval: u64,
}

impl ProgressReporter {
    /// Create a reporter that prints every `interval` frames.
    pub fn new(interval: u64) -> Self {
        Self {
            start: std::time::Instant::now(),
            interval,
        }
    }

    /// Report progress if frame_count is at the interval boundary.
    pub fn report(&self, frame_count: u64) {
        use std::io::Write;
        if frame_count > 0 && frame_count.is_multiple_of(self.interval) {
            let elapsed = self.start.elapsed().as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            print!("\rProcessed {frame_count} frames ({fps:.1} fps)");
            let _ = std::io::stdout().flush();
        }
    }

    /// Print the final summary line.
    pub fn finish(&self, frame_count: u64, output_path: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let fps = frame_count as f64 / elapsed;
        println!(
            "\n\nDone: {frame_count} frames in {elapsed:.1}s ({fps:.1} fps) \u{2192} {output_path}"
        );
    }
}

/// Detect NVIDIA Jetson (Tegra) platform.
#[cfg(feature = "gstreamer")]
pub fn is_tegra() -> bool {
    Path::new("/etc/nv_tegra_release").exists()
        || std::fs::read_to_string("/proc/device-tree/compatible")
            .map_or(false, |s| s.contains("nvidia,tegra"))
}
