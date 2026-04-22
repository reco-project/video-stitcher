//! JSON Lines [`PipelineEventSink`] - the default shipping sink.
//!
//! Opens a file, serializes each [`PipelineEvent`] to JSON, and writes
//! one event per line terminated by `\n`. Compatible with the usual
//! jq / streaming parsers: every line is a complete JSON object.
//!
//! Wrap in [`reco_core::pipeline_event::BackpressuredSink`] so serde
//! serialization and file I/O run on a background thread rather than
//! the render loop.
//!
//! # Example
//!
//! ```rust,no_run
//! use reco_core::pipeline_event::BackpressuredSink;
//! use reco_io::jsonl_sink::JsonlSink;
//!
//! let inner = JsonlSink::create("trace.jsonl").unwrap();
//! let sink = BackpressuredSink::new(Box::new(inner), 256, None);
//! // session.set_event_sink(Box::new(sink));
//! ```

use reco_core::pipeline_event::{PipelineEvent, PipelineEventSink};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// JSON Lines file sink. One [`PipelineEvent`] per line.
///
/// Internally holds a [`BufWriter<File>`]; the buffer flushes on drop
/// so no events are lost even when the process exits mid-stream. A
/// write error logs once per power-of-two failure count; the sink does
/// not panic the render thread.
pub struct JsonlSink {
    writer: BufWriter<File>,
    write_failures: u64,
}

impl JsonlSink {
    /// Create (or truncate) a file at `path` and return a sink that
    /// writes JSON Lines to it. Parent directory must already exist.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)?;
        log::info!("JsonlSink: writing pipeline events to {}", path.display());
        Ok(Self {
            writer: BufWriter::new(file),
            write_failures: 0,
        })
    }

    fn write_event(&mut self, event: &PipelineEvent) -> io::Result<()> {
        serde_json::to_writer(&mut self.writer, event).map_err(io::Error::other)?;
        self.writer.write_all(b"\n")
    }
}

impl PipelineEventSink for JsonlSink {
    fn emit(&mut self, event: PipelineEvent) {
        if let Err(e) = self.write_event(&event) {
            self.write_failures += 1;
            if self.write_failures.is_power_of_two() {
                log::warn!(
                    "JsonlSink: write failed ({} total): {e}",
                    self.write_failures
                );
            }
        }
    }
}

impl Drop for JsonlSink {
    fn drop(&mut self) {
        if let Err(e) = self.writer.flush() {
            log::warn!("JsonlSink: final flush failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::pipeline_event::PipelineEvent;
    use std::io::BufRead;

    fn mk_frame(i: u64) -> PipelineEvent {
        PipelineEvent::FrameStart {
            frame_index: i,
            timestamp_ms: i as f64 * 16.6,
        }
    }

    #[test]
    fn writes_one_json_object_per_line() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut sink = JsonlSink::create(tmp.path()).unwrap();
            for i in 0..5 {
                sink.emit(mk_frame(i));
            }
            // Drop flushes the BufWriter.
        }

        let file = File::open(tmp.path()).unwrap();
        let reader = io::BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<io::Result<_>>().unwrap();
        assert_eq!(lines.len(), 5, "one line per event");

        // Every line must be a complete JSON object with the tagged
        // kind field (lock the schema at the file level).
        for (i, line) in lines.iter().enumerate() {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(parsed["kind"], "frame_start");
            assert_eq!(parsed["frame_index"], i as u64);
        }
    }

    #[test]
    fn events_survive_drop_mid_stream() {
        // Writes inside a scope, then reopens the file and confirms
        // every event is there. Guards against a silent "lost the
        // last buffer" regression in Drop.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let mut sink = JsonlSink::create(tmp.path()).unwrap();
            for i in 0..100 {
                sink.emit(mk_frame(i));
            }
        }
        let contents = std::fs::read_to_string(tmp.path()).unwrap();
        let count = contents.lines().count();
        assert_eq!(count, 100, "all events must survive the buffered drop");
    }
}
