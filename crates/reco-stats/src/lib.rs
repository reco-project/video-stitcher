//! # reco-stats
//!
//! Stream detections from a [`reco_core::session::StitchSession`] to a CSV file.
//!
//! One row per detection (not per frame). Empty-detection frames are skipped
//! entirely so consumers don't have to filter them out. Columns:
//!
//! | column          | unit             | source                              |
//! |-----------------|------------------|-------------------------------------|
//! | `frame`         | frame index      | callback arg                        |
//! | `timestamp_ms`  | ms from start    | callback arg                        |
//! | `camera`        | `L` / `R`        | `MappedDetection.camera`            |
//! | `class_id`      | integer          | `MappedDetection.class_id`          |
//! | `confidence`    | `[0.0, 1.0]`     | `MappedDetection.confidence`        |
//! | `cam_x`         | `[0.0, 1.0]`     | `camera_center.0`                   |
//! | `cam_y`         | `[0.0, 1.0]`     | `camera_center.1`                   |
//! | `cam_w`         | `[0.0, 1.0]`     | `camera_size.0`                     |
//! | `cam_h`         | `[0.0, 1.0]`     | `camera_size.1`                     |
//! | `pano_yaw_deg`  | degrees          | `position.yaw` (empty if None)      |
//! | `pano_pitch_deg`| degrees          | `position.pitch` (empty if None)    |
//!
//! Designed to be piped straight into pandas / DuckDB / a spreadsheet.

use std::io::Write;

use reco_core::detector::CameraId;
use reco_core::director::MappedDetection;

/// Sink that writes one CSV row per detection.
pub struct CsvDetectionSink<W: Write> {
    writer: W,
    rows: u64,
}

impl<W: Write> CsvDetectionSink<W> {
    /// Create a sink, writing the header row immediately.
    pub fn new(mut writer: W) -> std::io::Result<Self> {
        writeln!(
            writer,
            "frame,timestamp_ms,camera,class_id,confidence,cam_x,cam_y,cam_w,cam_h,pano_yaw_deg,pano_pitch_deg"
        )?;
        Ok(Self { writer, rows: 0 })
    }

    /// Push one frame's detections into the CSV.
    pub fn push(
        &mut self,
        detections: &[MappedDetection],
        frame_index: u64,
        timestamp_ms: f64,
    ) -> std::io::Result<()> {
        for det in detections {
            let cam = match det.camera {
                CameraId::Left => 'L',
                CameraId::Right => 'R',
            };
            let (yaw_deg, pitch_deg) = match det.position {
                Some(pos) => (
                    format!("{:.4}", pos.yaw.to_degrees()),
                    format!("{:.4}", pos.pitch.to_degrees()),
                ),
                None => (String::new(), String::new()),
            };
            writeln!(
                self.writer,
                "{frame},{ts:.3},{cam},{cls},{conf:.4},{cx:.4},{cy:.4},{cw:.4},{ch:.4},{yaw},{pitch}",
                frame = frame_index,
                ts = timestamp_ms,
                cam = cam,
                cls = det.class_id,
                conf = det.confidence,
                cx = det.camera_center.0,
                cy = det.camera_center.1,
                cw = det.camera_size.0,
                ch = det.camera_size.1,
                yaw = yaw_deg,
                pitch = pitch_deg,
            )?;
            self.rows += 1;
        }
        Ok(())
    }

    /// Number of rows written so far (excluding the header).
    pub fn rows_written(&self) -> u64 {
        self.rows
    }

    /// Flush and return the inner writer.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detector::CameraId;
    use reco_core::director::{MappedDetection, ViewportPosition};

    fn det(cam: CameraId, class_id: u16, conf: f32, yaw: f32) -> MappedDetection {
        MappedDetection {
            camera: cam,
            class_id,
            confidence: conf,
            camera_center: (0.5, 0.5),
            camera_size: (0.05, 0.05),
            position: Some(ViewportPosition {
                yaw,
                pitch: 0.0,
                fov_degrees: None,
            }),
        }
    }

    #[test]
    fn writes_header_and_rows() {
        let mut buf = Vec::new();
        {
            let mut sink = CsvDetectionSink::new(&mut buf).unwrap();
            sink.push(&[det(CameraId::Left, 0, 0.9, 0.25)], 10, 333.3)
                .unwrap();
            sink.push(&[], 11, 366.6).unwrap(); // empty frame — no row
            sink.push(&[det(CameraId::Right, 1, 0.8, -0.1)], 12, 400.0)
                .unwrap();
            assert_eq!(sink.rows_written(), 2);
        }
        let out = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 rows
        assert!(lines[0].starts_with("frame,"));
        assert!(lines[1].starts_with("10,333.300,L,0,"));
        assert!(lines[2].starts_with("12,400.000,R,1,"));
    }

    #[test]
    fn handles_missing_position() {
        let mut buf = Vec::new();
        let mut sink = CsvDetectionSink::new(&mut buf).unwrap();
        let mut d = det(CameraId::Left, 0, 0.9, 0.0);
        d.position = None;
        sink.push(&[d], 0, 0.0).unwrap();
        let out = String::from_utf8(buf).unwrap();
        let last = out.lines().last().unwrap();
        // last two columns should be empty
        assert!(last.ends_with(",,"));
    }
}
