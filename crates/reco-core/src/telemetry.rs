//! Always-on telemetry for the stitch pipeline.
//!
//! Collects per-frame timing and detection statistics with near-zero
//! overhead (~100ns per frame). No heap allocation in the hot path.
//!
//! Consumers query [`TelemetryCollector::snapshot`] for a point-in-time
//! view, or attach a [`TelemetrySink`] for periodic push delivery.

use std::time::{Duration, Instant};

/// Per-frame timing breakdown for all pipeline stages.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTiming {
    pub decode: Option<Duration>,
    pub upload: Option<Duration>,
    pub stitch: Option<Duration>,
    pub readback: Option<Duration>,
    pub encode: Option<Duration>,
    pub detection: Option<Duration>,
    pub tracking: Option<Duration>,
    pub total: Option<Duration>,
}

/// Which pipeline stage is the current bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Decode,
    Upload,
    Stitch,
    Readback,
    Encode,
    Detection,
    Tracking,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode => write!(f, "decode"),
            Self::Upload => write!(f, "upload"),
            Self::Stitch => write!(f, "stitch"),
            Self::Readback => write!(f, "readback"),
            Self::Encode => write!(f, "encode"),
            Self::Detection => write!(f, "detection"),
            Self::Tracking => write!(f, "tracking"),
        }
    }
}

/// Fixed-size ring buffer. No heap allocation.
struct RingBuffer<const N: usize> {
    buf: [FrameTiming; N],
    head: usize,
    len: usize,
}

impl<const N: usize> RingBuffer<N> {
    fn new() -> Self {
        Self {
            buf: [FrameTiming::default(); N],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, item: FrameTiming) {
        self.buf[self.head] = item;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    fn iter(&self) -> impl Iterator<Item = &FrameTiming> {
        let start = if self.len < N { 0 } else { self.head };
        (0..self.len).map(move |i| &self.buf[(start + i) % N])
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Always-on telemetry accumulator.
///
/// Call [`record_frame`](Self::record_frame) once per frame from the
/// session's frame loop. Call [`snapshot`](Self::snapshot) to get a
/// read-only view of the current state.
pub struct TelemetryCollector {
    frame_count: u64,
    ring: RingBuffer<256>,
    total_detections: u64,
    ball_present_frames: u64,
    active_tracks: u32,
    max_frame_time: Duration,
    session_start: Option<Instant>,
    gpu_name: String,
    encoder_name: Option<String>,
    decode_mode: Option<String>,
    sink: Option<Box<dyn TelemetrySink>>,
    sink_interval: u64,
}

impl TelemetryCollector {
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            ring: RingBuffer::new(),
            total_detections: 0,
            ball_present_frames: 0,
            active_tracks: 0,
            max_frame_time: Duration::ZERO,
            session_start: None,
            gpu_name: String::new(),
            encoder_name: None,
            decode_mode: None,
            sink: None,
            sink_interval: 30,
        }
    }

    pub fn set_gpu_name(&mut self, name: String) {
        self.gpu_name = name;
    }

    pub fn set_encoder_name(&mut self, name: String) {
        self.encoder_name = Some(name);
    }

    pub fn set_decode_mode(&mut self, mode: String) {
        self.decode_mode = Some(mode);
    }

    pub fn set_sink(&mut self, sink: Box<dyn TelemetrySink>, interval_frames: u64) {
        self.sink = Some(sink);
        self.sink_interval = interval_frames.max(1);
    }

    pub fn record_frame(&mut self, timing: FrameTiming) {
        if self.session_start.is_none() {
            self.session_start = Some(Instant::now());
        }
        self.frame_count += 1;
        if let Some(total) = timing.total
            && total > self.max_frame_time
        {
            self.max_frame_time = total;
        }
        self.ring.push(timing);

        if self.frame_count.is_multiple_of(self.sink_interval)
            && let Some(mut sink) = self.sink.take()
        {
            let snap = self.build_snapshot();
            sink.on_snapshot(&snap);
            self.sink = Some(sink);
        }
    }

    pub fn record_detections(&mut self, count: u32, active_tracks: u32, ball_present: bool) {
        self.total_detections += count as u64;
        self.active_tracks = active_tracks;
        if ball_present {
            self.ball_present_frames += 1;
        }
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        self.build_snapshot()
    }

    fn build_snapshot(&self) -> TelemetrySnapshot {
        let elapsed = self
            .session_start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);
        let fps_average = if elapsed.as_secs_f32() > 0.0 {
            self.frame_count as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };

        let ring_len = self.ring.len() as f32;
        let avg = |extract: fn(&FrameTiming) -> Option<Duration>| -> f32 {
            if ring_len == 0.0 {
                return 0.0;
            }
            let sum: f32 = self
                .ring
                .iter()
                .filter_map(|t| extract(t).map(|d| d.as_secs_f32() * 1000.0))
                .sum();
            let count = self.ring.iter().filter(|t| extract(t).is_some()).count() as f32;
            if count > 0.0 { sum / count } else { 0.0 }
        };

        let avg_decode = avg(|t| t.decode);
        let avg_upload = avg(|t| t.upload);
        let avg_stitch = avg(|t| t.stitch);
        let avg_readback = avg(|t| t.readback);
        let avg_encode = avg(|t| t.encode);
        let avg_detection = avg(|t| t.detection);
        let avg_total = avg(|t| t.total);

        let p99_total = self.percentile_ms(|t| t.total, 0.99);

        let fps_recent = if ring_len > 1.0 {
            let mut total_secs: f32 = 0.0;
            let mut timed_frames: u32 = 0;
            for t in self.ring.iter() {
                if let Some(d) = t.total {
                    total_secs += d.as_secs_f32();
                    timed_frames += 1;
                }
            }
            if timed_frames > 1 && total_secs > 0.0 {
                timed_frames as f32 / total_secs
            } else {
                fps_average
            }
        } else {
            fps_average
        };

        let ball_pct = if self.frame_count > 0 {
            self.ball_present_frames as f32 / self.frame_count as f32 * 100.0
        } else {
            0.0
        };

        let bottleneck = self.detect_bottleneck(
            avg_decode,
            avg_upload,
            avg_stitch,
            avg_readback,
            avg_encode,
            avg_detection,
        );

        TelemetrySnapshot {
            frames_processed: self.frame_count,
            elapsed,
            fps_average,
            fps_recent,
            avg_decode_ms: avg_decode,
            avg_upload_ms: avg_upload,
            avg_stitch_ms: avg_stitch,
            avg_readback_ms: avg_readback,
            avg_encode_ms: avg_encode,
            avg_detection_ms: avg_detection,
            avg_total_ms: avg_total,
            p99_total_ms: p99_total,
            max_frame_ms: self.max_frame_time.as_secs_f32() * 1000.0,
            total_detections: self.total_detections,
            detections_per_frame: if self.frame_count > 0 {
                self.total_detections as f32 / self.frame_count as f32
            } else {
                0.0
            },
            ball_presence_pct: ball_pct,
            active_tracks: self.active_tracks,
            gpu_name: self.gpu_name.clone(),
            encoder_name: self.encoder_name.clone(),
            decode_mode: self.decode_mode.clone(),
            bottleneck,
        }
    }

    fn percentile_ms(&self, extract: fn(&FrameTiming) -> Option<Duration>, pct: f32) -> f32 {
        let mut values: Vec<f32> = self
            .ring
            .iter()
            .filter_map(|t| extract(t).map(|d| d.as_secs_f32() * 1000.0))
            .collect();
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((values.len() as f32 * pct) as usize).min(values.len() - 1);
        values[idx]
    }

    fn detect_bottleneck(
        &self,
        decode: f32,
        upload: f32,
        stitch: f32,
        readback: f32,
        encode: f32,
        detection: f32,
    ) -> Option<PipelineStage> {
        // Upload (staging copies) is a sub-step of decode for zero-copy paths.
        // Combine them so the bottleneck correctly reports "decode" when
        // the hardware decoder is the limiting factor.
        let stages = [
            (PipelineStage::Decode, decode + upload),
            (PipelineStage::Stitch, stitch),
            (PipelineStage::Readback, readback),
            (PipelineStage::Encode, encode),
            (PipelineStage::Detection, detection),
        ];
        stages
            .iter()
            .filter(|(_, ms)| *ms > 0.1)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(stage, _)| *stage)
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time telemetry snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TelemetrySnapshot {
    pub frames_processed: u64,
    #[serde(with = "duration_secs")]
    pub elapsed: Duration,
    pub fps_average: f32,
    pub fps_recent: f32,
    pub avg_decode_ms: f32,
    pub avg_upload_ms: f32,
    pub avg_stitch_ms: f32,
    pub avg_readback_ms: f32,
    pub avg_encode_ms: f32,
    pub avg_detection_ms: f32,
    pub avg_total_ms: f32,
    pub p99_total_ms: f32,
    pub max_frame_ms: f32,
    pub total_detections: u64,
    pub detections_per_frame: f32,
    pub ball_presence_pct: f32,
    pub active_tracks: u32,
    pub gpu_name: String,
    pub encoder_name: Option<String>,
    pub decode_mode: Option<String>,
    pub bottleneck: Option<PipelineStage>,
}

mod duration_secs {
    use serde::Serializer;
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(d.as_secs_f64())
    }
}

/// End-of-session summary.
pub struct SessionSummary {
    pub snapshot: TelemetrySnapshot,
}

impl std::fmt::Display for SessionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = &self.snapshot;
        writeln!(f, "--- Session Summary ---")?;
        writeln!(f, "Frames:    {} processed", s.frames_processed)?;
        writeln!(f, "Duration:  {:.1}s", s.elapsed.as_secs_f64())?;
        writeln!(
            f,
            "FPS:       {:.1} avg, {:.1} recent",
            s.fps_average, s.fps_recent
        )?;
        writeln!(f, "GPU:       {}", s.gpu_name)?;
        if let Some(enc) = &s.encoder_name {
            writeln!(f, "Encoder:   {enc}")?;
        }
        if let Some(dec) = &s.decode_mode {
            writeln!(f, "Decode:    {dec}")?;
        }
        writeln!(f)?;
        writeln!(f, "Per-frame timing (avg / p99):")?;
        let total_decode = s.avg_decode_ms + s.avg_upload_ms;
        if s.avg_upload_ms > 0.05 {
            writeln!(
                f,
                "  Decode:    {:.1} ms (wait {:.1} + staging {:.1})",
                total_decode, s.avg_decode_ms, s.avg_upload_ms
            )?;
        } else {
            writeln!(f, "  Decode:    {:.1} ms", total_decode)?;
        }
        writeln!(f, "  Stitch:    {:.1} ms", s.avg_stitch_ms)?;
        writeln!(f, "  Readback:  {:.1} ms", s.avg_readback_ms)?;
        writeln!(f, "  Encode:    {:.1} ms", s.avg_encode_ms)?;
        if s.avg_detection_ms > 0.0 {
            writeln!(f, "  Detection: {:.1} ms", s.avg_detection_ms)?;
        }
        writeln!(
            f,
            "  Total:     {:.1} / {:.1} ms",
            s.avg_total_ms, s.p99_total_ms
        )?;
        if let Some(stage) = &s.bottleneck {
            writeln!(f, "  Bottleneck: {stage}")?;
        }
        if s.total_detections > 0 {
            writeln!(f)?;
            writeln!(f, "Detection:")?;
            writeln!(f, "  Ball present:  {:.0}% of frames", s.ball_presence_pct)?;
            writeln!(f, "  Detections:    {:.1}/frame", s.detections_per_frame)?;
        }
        Ok(())
    }
}

/// Receives periodic telemetry snapshots.
pub trait TelemetrySink: Send {
    fn on_snapshot(&mut self, snapshot: &TelemetrySnapshot);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_timing_collection() {
        let mut tc = TelemetryCollector::new();
        for _ in 0..100 {
            tc.record_frame(FrameTiming {
                decode: Some(Duration::from_millis(2)),
                stitch: Some(Duration::from_millis(1)),
                encode: Some(Duration::from_millis(3)),
                total: Some(Duration::from_millis(8)),
                ..Default::default()
            });
        }
        let snap = tc.snapshot();
        assert_eq!(snap.frames_processed, 100);
        assert!((snap.avg_decode_ms - 2.0).abs() < 0.1);
        assert!((snap.avg_encode_ms - 3.0).abs() < 0.1);
        assert!(snap.bottleneck == Some(PipelineStage::Encode));
    }

    #[test]
    fn bottleneck_detects_slowest_stage() {
        let mut tc = TelemetryCollector::new();
        tc.record_frame(FrameTiming {
            decode: Some(Duration::from_millis(10)),
            encode: Some(Duration::from_millis(20)),
            readback: Some(Duration::from_millis(5)),
            total: Some(Duration::from_millis(35)),
            ..Default::default()
        });
        let snap = tc.snapshot();
        assert_eq!(snap.bottleneck, Some(PipelineStage::Encode));
    }

    #[test]
    fn percentile_with_few_frames() {
        let mut tc = TelemetryCollector::new();
        tc.record_frame(FrameTiming {
            total: Some(Duration::from_millis(5)),
            ..Default::default()
        });
        let snap = tc.snapshot();
        assert!((snap.p99_total_ms - 5.0).abs() < 0.1);
    }

    #[test]
    fn fps_recent_uses_timed_frame_count() {
        let mut tc = TelemetryCollector::new();
        // 10 frames at 10ms each = 100 FPS
        for _ in 0..10 {
            tc.record_frame(FrameTiming {
                total: Some(Duration::from_millis(10)),
                ..Default::default()
            });
        }
        let snap = tc.snapshot();
        assert!(
            (snap.fps_recent - 100.0).abs() < 5.0,
            "fps_recent should be ~100, got {}",
            snap.fps_recent
        );
    }

    #[test]
    fn session_summary_display() {
        let mut tc = TelemetryCollector::new();
        tc.set_gpu_name("Test GPU".into());
        for _ in 0..10 {
            tc.record_frame(FrameTiming {
                decode: Some(Duration::from_millis(2)),
                total: Some(Duration::from_millis(5)),
                ..Default::default()
            });
        }
        let summary = SessionSummary {
            snapshot: tc.snapshot(),
        };
        let text = format!("{summary}");
        assert!(text.contains("Session Summary"));
        assert!(text.contains("Test GPU"));
    }
}
