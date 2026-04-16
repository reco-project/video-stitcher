//! # reco-heatmap
//!
//! Panorama-space ball position heatmap.
//!
//! Bucket the panorama's yaw/pitch range into a 2-D grid, and increment each
//! bucket every time a `MappedDetection` falls into it. At the end of the run,
//! blur the grid and render it as a PNG.
//!
//! The yaw axis is horizontal: negative yaw = left side of the panorama,
//! positive yaw = right side. The pitch axis is vertical: positive pitch =
//! looking up.
//!
//! ## Coverage range
//!
//! The heatmap spans `[yaw_min, yaw_max]` by `[pitch_min, pitch_max]`, all in
//! radians. Good defaults for a stereo football rig are `±45°` horizontal and
//! `±20°` vertical — see [`HeatmapConfig::default`]. If a detection lands
//! outside the configured range it's clamped into the border buckets (so you
//! still see a single column of "off-panorama" heat instead of silently
//! dropping samples).

use reco_core::director::MappedDetection;

/// Configuration for the heatmap accumulator.
#[derive(Debug, Clone)]
pub struct HeatmapConfig {
    /// Detection class to accumulate (default: `0`, ball).
    pub target_class_id: u16,
    /// Minimum confidence required for a detection to count.
    pub min_confidence: f32,
    /// Grid width in cells (yaw axis).
    pub width: u32,
    /// Grid height in cells (pitch axis).
    pub height: u32,
    /// Lowest yaw (radians) mapped to column 0.
    pub yaw_min: f32,
    /// Highest yaw (radians) mapped to column `width - 1`.
    pub yaw_max: f32,
    /// Lowest pitch (radians) mapped to row `height - 1` (pitch axis is flipped).
    pub pitch_min: f32,
    /// Highest pitch (radians) mapped to row `0`.
    pub pitch_max: f32,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        let forty_five = 45.0_f32.to_radians();
        let twenty = 20.0_f32.to_radians();
        Self {
            target_class_id: 0,
            min_confidence: 0.30,
            width: 640,
            height: 180,
            yaw_min: -forty_five,
            yaw_max: forty_five,
            pitch_min: -twenty,
            pitch_max: twenty,
        }
    }
}

/// Accumulator that turns detection batches into a 2-D histogram.
#[derive(Debug)]
pub struct HeatmapAccumulator {
    cfg: HeatmapConfig,
    cells: Vec<u32>,
    samples: u64,
}

impl HeatmapAccumulator {
    /// Create a new accumulator with the given config.
    pub fn new(cfg: HeatmapConfig) -> Self {
        let cells = vec![0; (cfg.width * cfg.height) as usize];
        Self {
            cfg,
            cells,
            samples: 0,
        }
    }

    /// Feed one frame's detections into the heatmap.
    pub fn push(&mut self, detections: &[MappedDetection], _frame_index: u64, _timestamp_ms: f64) {
        for det in detections {
            if det.class_id != self.cfg.target_class_id {
                continue;
            }
            if det.confidence < self.cfg.min_confidence {
                continue;
            }
            let Some(pos) = det.position else { continue };
            self.add_sample(pos.yaw, pos.pitch);
        }
    }

    /// Number of samples that were added to the grid.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Access the raw cell counts row-major, `[y * width + x]`.
    pub fn cells(&self) -> &[u32] {
        &self.cells
    }

    /// Width of the grid in cells.
    pub fn width(&self) -> u32 {
        self.cfg.width
    }

    /// Height of the grid in cells.
    pub fn height(&self) -> u32 {
        self.cfg.height
    }

    /// Render the heatmap to an RGBA8 buffer using a simple
    /// black -> red -> yellow -> white gradient.
    ///
    /// Returns a tightly packed `width * height * 4` byte buffer.
    pub fn render(&self) -> Vec<u8> {
        let w = self.cfg.width as usize;
        let h = self.cfg.height as usize;

        // One pass to find peak count for normalization.
        let peak = self.cells.iter().copied().max().unwrap_or(0).max(1) as f32;

        let mut rgba = vec![0u8; w * h * 4];
        for (idx, &count) in self.cells.iter().enumerate() {
            // Log scale so a few hot cells don't wash out the rest.
            let normalized = (1.0 + count as f32).ln() / (1.0 + peak).ln();
            let [r, g, b] = colormap(normalized);
            let o = idx * 4;
            rgba[o] = r;
            rgba[o + 1] = g;
            rgba[o + 2] = b;
            rgba[o + 3] = 255;
        }
        rgba
    }

    fn add_sample(&mut self, yaw: f32, pitch: f32) {
        let w = self.cfg.width as i32;
        let h = self.cfg.height as i32;
        let u = (yaw - self.cfg.yaw_min) / (self.cfg.yaw_max - self.cfg.yaw_min);
        let v = (self.cfg.pitch_max - pitch) / (self.cfg.pitch_max - self.cfg.pitch_min);
        let x = (u * w as f32).floor() as i32;
        let y = (v * h as f32).floor() as i32;
        let x = x.clamp(0, w - 1) as usize;
        let y = y.clamp(0, h - 1) as usize;
        let idx = y * self.cfg.width as usize + x;
        self.cells[idx] = self.cells[idx].saturating_add(1);
        self.samples += 1;
    }
}

/// A three-stop black -> red -> yellow -> white colormap.
fn colormap(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    // Anchors at t = 0.0, 0.33, 0.66, 1.0
    let (r, g, b) = if t < 0.33 {
        let u = t / 0.33;
        (u, 0.0, 0.0)
    } else if t < 0.66 {
        let u = (t - 0.33) / 0.33;
        (1.0, u, 0.0)
    } else {
        let u = (t - 0.66) / 0.34;
        (1.0, 1.0, u)
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detector::CameraId;
    use reco_core::director::{MappedDetection, ViewportPosition};

    fn det(yaw: f32, pitch: f32, conf: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Left,
            class_id: 0,
            confidence: conf,
            camera_center: (0.5, 0.5),
            camera_size: (0.05, 0.05),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    #[test]
    fn buckets_samples() {
        let cfg = HeatmapConfig {
            width: 10,
            height: 10,
            yaw_min: -1.0,
            yaw_max: 1.0,
            pitch_min: -1.0,
            pitch_max: 1.0,
            min_confidence: 0.0,
            ..HeatmapConfig::default()
        };
        let mut h = HeatmapAccumulator::new(cfg);
        // yaw=0, pitch=0 should land near cell (5,5). pitch axis is flipped.
        h.push(&[det(0.0, 0.0, 1.0)], 0, 0.0);
        assert_eq!(h.samples(), 1);
        let peak = h.cells().iter().copied().max().unwrap();
        assert_eq!(peak, 1);
    }

    #[test]
    fn clamps_out_of_range() {
        let cfg = HeatmapConfig {
            width: 4,
            height: 4,
            yaw_min: -0.5,
            yaw_max: 0.5,
            pitch_min: -0.5,
            pitch_max: 0.5,
            min_confidence: 0.0,
            ..HeatmapConfig::default()
        };
        let mut h = HeatmapAccumulator::new(cfg);
        // Way outside the range — still counted at the border.
        h.push(&[det(10.0, 10.0, 1.0)], 0, 0.0);
        h.push(&[det(-10.0, -10.0, 1.0)], 1, 0.0);
        assert_eq!(h.samples(), 2);
        // The corner cells should be non-zero.
        assert!(h.cells()[0] > 0 || h.cells()[3] > 0);
    }

    #[test]
    fn ignores_wrong_class() {
        let cfg = HeatmapConfig {
            width: 4,
            height: 4,
            target_class_id: 0,
            min_confidence: 0.0,
            ..HeatmapConfig::default()
        };
        let mut h = HeatmapAccumulator::new(cfg);
        let mut d = det(0.0, 0.0, 1.0);
        d.class_id = 5;
        h.push(&[d], 0, 0.0);
        assert_eq!(h.samples(), 0);
    }
}
