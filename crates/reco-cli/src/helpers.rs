//! Shared CLI helpers: progress reporting, platform detection.

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
        if frame_count > 0 && frame_count % self.interval == 0 {
            let elapsed = self.start.elapsed().as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            print!("\rProcessed {frame_count} frames ({fps:.1} fps)");
            let _ = std::io::stdout().flush();
        }
    }

    /// Print the final summary line.
    #[cfg(feature = "gstreamer")]
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
    std::path::Path::new("/etc/nv_tegra_release").exists()
        || std::fs::read_to_string("/proc/device-tree/compatible")
            .is_ok_and(|s| s.contains("nvidia,tegra"))
}
