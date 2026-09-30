//! Always-on telemetry for the stitch pipeline.
//!
//! Collects host-side timing and observation statistics in a fixed-size ring.
//! No GPU synchronization or allocation is added when recording a frame.
//!
//! Sessions combine these measurements with output-worker statistics for polling
//! and periodic [`TelemetrySink`] delivery. Host-stage timings do not identify a
//! pipeline bottleneck when GPU and worker execution overlap.

use std::time::{Duration, Instant};

/// Host processing time attributed to one submitted output frame.
///
/// `None` means unmeasured; `Some(Duration::ZERO)` means known skipped work.
/// Buffered frames retain producer work, excluding residence in lookahead queues.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTiming {
    pub decode: Option<Duration>,
    pub upload: Option<Duration>,
    pub stitch: Option<Duration>,
    pub readback: Option<Duration>,
    /// Synchronous submit cost on the pipeline thread: NV12 memcpy into the
    /// sink buffer + channel enqueue (+ backpressure wait if the
    /// sink is the bottleneck). NOT the encode compute, which runs
    /// overlapped on the sink thread (see `SinkThread`).
    pub submit: Option<Duration>,
    /// Detection dispatch, including preprocessing, inference and mapping.
    pub detection: Option<Duration>,
    /// Observation expiry, trackers, panning and coverage resolution.
    pub tracking: Option<Duration>,
    /// Total attributed host processing, including source waits and sink submission.
    pub total: Option<Duration>,
}

/// Fixed-size ring buffer. No heap allocation.
struct RingBuffer<const N: usize> {
    buf: [(FrameTiming, Option<Instant>); N],
    head: usize,
    len: usize,
}

impl<const N: usize> RingBuffer<N> {
    fn new() -> Self {
        Self {
            buf: [(FrameTiming::default(), None); N],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, item: FrameTiming, completed_at: Instant) {
        self.buf[self.head] = (item, Some(completed_at));
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    fn iter(&self) -> impl Iterator<Item = &FrameTiming> {
        let start = if self.len < N { 0 } else { self.head };
        (0..self.len).map(move |i| &self.buf[(start + i) % N].0)
    }

    fn recent_fps(&self) -> Option<f32> {
        if self.len < 2 {
            return None;
        }
        let first = if self.len < N { 0 } else { self.head };
        let last = (self.head + N - 1) % N;
        let elapsed = self.buf[last].1?.duration_since(self.buf[first].1?);
        (elapsed > Duration::ZERO).then(|| (self.len - 1) as f32 / elapsed.as_secs_f32())
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

    /// Anchor session wall time before reading the first source frame.
    /// Subsequent runs retain the original origin and cumulative counters.
    pub(crate) fn start(&mut self) {
        self.session_start.get_or_insert_with(Instant::now);
    }

    pub fn record_frame(&mut self, timing: FrameTiming) {
        self.record_frame_at(timing, Instant::now());
    }

    fn record_frame_at(&mut self, timing: FrameTiming, completed_at: Instant) {
        self.session_start.get_or_insert(completed_at);
        self.frame_count += 1;
        if let Some(total) = timing.total
            && total > self.max_frame_time
        {
            self.max_frame_time = total;
        }
        self.ring.push(timing, completed_at);
    }

    pub fn record_detections(&mut self, count: u32, active_tracks: u32, ball_present: bool) {
        self.total_detections += count as u64;
        self.active_tracks = active_tracks;
        if ball_present {
            self.ball_present_frames += 1;
        }
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        self.build_snapshot(Instant::now())
    }

    fn build_snapshot(&self, now: Instant) -> TelemetrySnapshot {
        let elapsed = self
            .session_start
            .map(|s| now.duration_since(s))
            .unwrap_or(Duration::ZERO);
        let fps_average = if elapsed.as_secs_f32() > 0.0 {
            self.frame_count as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };

        let avg = |extract: fn(&FrameTiming) -> Option<Duration>| -> f32 {
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
        let avg_submit = avg(|t| t.submit);
        let avg_detection = avg(|t| t.detection);
        let avg_tracking = avg(|t| t.tracking);
        let avg_total = avg(|t| t.total);

        let p99_total = self.percentile_ms(|t| t.total, 0.99);

        let fps_recent = self.ring.recent_fps().unwrap_or(fps_average);

        let ball_pct = if self.frame_count > 0 {
            self.ball_present_frames as f32 / self.frame_count as f32 * 100.0
        } else {
            0.0
        };

        TelemetrySnapshot {
            frames_processed: self.frame_count,
            elapsed,
            fps_average,
            fps_recent,
            avg_decode_ms: avg_decode,
            avg_upload_ms: avg_upload,
            avg_stitch_ms: avg_stitch,
            avg_readback_ms: avg_readback,
            avg_submit_ms: avg_submit,
            // Populated by the session from the async encode thread; the
            // collector itself has no view of the overlapped encode cost.
            avg_encode_worker_ms: 0.0,
            backpressure_stalls: 0,
            backpressure_ms: 0.0,
            avg_detection_ms: avg_detection,
            avg_tracking_ms: avg_tracking,
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
    /// Wall time since the first run started, including prefill and gaps between reused runs.
    pub elapsed: Duration,
    /// Submitted frames divided by session wall time, not reciprocal processing time.
    pub fps_average: f32,
    /// Submission cadence between the oldest and newest entries in the timing window.
    pub fps_recent: f32,
    pub avg_decode_ms: f32,
    pub avg_upload_ms: f32,
    pub avg_stitch_ms: f32,
    pub avg_readback_ms: f32,
    pub avg_submit_ms: f32,
    /// Real overlapped encode cost on the async encode thread (per frame).
    /// Zero if no async encoder is attached.
    pub avg_encode_worker_ms: f32,
    /// Times `submit` blocked on a full output channel.
    pub backpressure_stalls: u64,
    /// Total time spent blocked on encode backpressure (ms).
    pub backpressure_ms: f32,
    /// Mean detection-dispatch host time per measured output frame, including known skipped ticks.
    pub avg_detection_ms: f32,
    /// Mean tracker, panner and coverage host time per measured output frame.
    pub avg_tracking_ms: f32,
    pub avg_total_ms: f32,
    pub p99_total_ms: f32,
    pub max_frame_ms: f32,
    /// Sum of cached mapped observations visible to output frames, not newly inferred detections.
    pub total_detections: u64,
    pub detections_per_frame: f32,
    pub ball_presence_pct: f32,
    pub active_tracks: u32,
    pub gpu_name: String,
    pub encoder_name: Option<String>,
    pub decode_mode: Option<String>,
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
        writeln!(f, "Host processing per output frame (avg / p99):")?;
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
        writeln!(f, "  Submit:    {:.1} ms", s.avg_submit_ms)?;
        if s.avg_encode_worker_ms > 0.0 || s.backpressure_stalls > 0 {
            writeln!(
                f,
                "  Encode:    {:.1} ms (overlapped); backpressure {} stalls / {:.1} ms",
                s.avg_encode_worker_ms, s.backpressure_stalls, s.backpressure_ms
            )?;
        }
        if s.avg_tracking_ms > 0.0 {
            writeln!(f, "  Tracking:  {:.1} ms", s.avg_tracking_ms)?;
        }
        if s.avg_detection_ms > 0.0 {
            writeln!(f, "  Detection: {:.1} ms", s.avg_detection_ms)?;
        }
        writeln!(
            f,
            "  Total:     {:.1} / {:.1} ms",
            s.avg_total_ms, s.p99_total_ms
        )?;
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
                submit: Some(Duration::from_millis(3)),
                total: Some(Duration::from_millis(8)),
                ..Default::default()
            });
        }
        let snap = tc.snapshot();
        assert_eq!(snap.frames_processed, 100);
        assert!((snap.avg_decode_ms - 2.0).abs() < 0.1);
        assert!((snap.avg_submit_ms - 3.0).abs() < 0.1);
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
    fn sparse_work_keeps_known_zero_distinct_from_unknown() {
        let mut tc = TelemetryCollector::new();
        for i in 0..256 {
            tc.record_frame(FrameTiming {
                detection: Some(Duration::from_millis(if i % 16 == 0 { 16 } else { 0 })),
                tracking: Some(Duration::from_millis(2)),
                upload: (i % 16 == 0).then_some(Duration::from_millis(8)),
                ..Default::default()
            });
        }
        let snap = tc.snapshot();
        assert_eq!(snap.avg_detection_ms, 1.0);
        assert_eq!(snap.avg_tracking_ms, 2.0);
        assert_eq!(snap.avg_upload_ms, 8.0);
    }

    #[test]
    fn throughput_uses_completion_cadence_including_ring_wrap() {
        let mut tc = TelemetryCollector::new();
        let start = Instant::now();
        tc.session_start = Some(start);
        for i in 1..=600 {
            tc.record_frame_at(
                FrameTiming {
                    total: Some(Duration::from_millis(1)),
                    ..Default::default()
                },
                start + Duration::from_millis(i * 10),
            );
        }
        let snap = tc.build_snapshot(start + Duration::from_secs(6));
        assert_eq!(snap.frames_processed, 600);
        assert_eq!(snap.fps_average, 100.0);
        assert_eq!(snap.fps_recent, 100.0);
        assert_eq!(snap.avg_total_ms, 1.0);
    }

    #[test]
    fn prefill_and_repeated_runs_keep_one_wall_clock_origin() {
        let mut tc = TelemetryCollector::new();
        tc.start();
        let start = tc.session_start.unwrap();
        tc.record_frame_at(FrameTiming::default(), start + Duration::from_secs(2));
        tc.start();
        assert_eq!(tc.session_start, Some(start));
        tc.record_frame_at(FrameTiming::default(), start + Duration::from_secs(3));
        let snap = tc.build_snapshot(start + Duration::from_secs(4));
        assert_eq!(snap.elapsed, Duration::from_secs(4));
        assert_eq!(snap.fps_average, 0.5);
        assert_eq!(snap.fps_recent, 1.0);
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
