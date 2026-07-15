//! Panner that reads precomputed viewport positions from a CSV file.
//!
//! CSV format: `frame,yaw,pitch,fov` (header line, radians).

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::WorldState;
use reco_core::geometry::Pose;

/// Replays precomputed poses from a CSV file.
pub struct FilePanner {
    poses: HashMap<u64, Pose>,
    last: Pose,
}

impl FilePanner {
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut poses = HashMap::new();
        let mut last_fov = Pose::default().fov_degrees;

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
            // Rows without an fov column hold the previous row's zoom,
            // mirroring how FieldPanner holds fov on ball-only frames.
            if let Some(fov) = cols.get(3).and_then(|s| s.trim().parse().ok()) {
                last_fov = fov;
            }
            poses.insert(
                frame,
                Pose {
                    yaw,
                    pitch,
                    fov_degrees: last_fov,
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
            last: Pose::default(),
        })
    }
}

impl Panner for FilePanner {
    fn decide(&mut self, _world: &WorldState, ctx: &PanContext<'_>) -> Pose {
        if let Some(&pose) = self.poses.get(&ctx.frame_index) {
            self.last = pose;
        }
        self.last
    }
}
