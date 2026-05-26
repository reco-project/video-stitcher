//! Panner that reads precomputed viewport positions from a CSV file.
//!
//! CSV format: `frame,yaw,pitch,fov` (header line, radians).

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use reco_core::detect::director::ViewportPosition;
use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::WorldState;

/// Replays precomputed poses from a CSV file.
pub struct FilePanner {
    poses: HashMap<u64, ViewportPosition>,
    last: ViewportPosition,
}

impl FilePanner {
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut poses = HashMap::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if i == 0 || line.is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 3 {
                continue;
            }
            let frame: u64 = cols[0].trim().parse()?;
            let yaw: f32 = cols[1].trim().parse()?;
            let pitch: f32 = cols[2].trim().parse()?;
            let fov: Option<f32> = cols.get(3).and_then(|s| s.trim().parse().ok());
            poses.insert(
                frame,
                ViewportPosition {
                    yaw,
                    pitch,
                    fov_degrees: fov,
                },
            );
        }

        log::info!(
            "FilePanner: loaded {} poses from {}",
            poses.len(),
            path.display()
        );
        Ok(Self {
            poses,
            last: ViewportPosition::default(),
        })
    }
}

impl Panner for FilePanner {
    fn decide(&mut self, _world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        if let Some(&pose) = self.poses.get(&ctx.frame_index) {
            self.last = pose;
        }
        self.last
    }
}
