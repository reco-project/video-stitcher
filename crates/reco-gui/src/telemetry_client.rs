//! Opt-in telemetry client for the reco-telemetry Cloud Run service.
//!
//! Sends anonymous usage events (bug reports, export outcomes, context)
//! to a self-hosted endpoint. Fully opt-in: no events are sent unless
//! the user enables telemetry in preferences. No PII, no file paths,
//! no video content. The client_id is a random UUID generated once and
//! stored in settings.

use serde::Serialize;
use std::sync::mpsc;
use std::thread;

const ENDPOINT: &str = "https://telemetry-ingestion-204135919265.us-central1.run.app/telemetry";
const APP_NAME: &str = "video-stitcher";

#[derive(Serialize)]
struct Batch {
    schema_version: u32,
    client_id: String,
    app: App,
    sent_at: String,
    batch_id: String,
    events: Vec<Event>,
}

#[derive(Serialize)]
struct App {
    name: String,
    version: String,
}

#[derive(Serialize, Clone)]
struct Event {
    schema_version: u32,
    ts: String,
    name: String,
    client_id: String,
    props: Option<serde_json::Value>,
}

pub struct TelemetryClient {
    tx: mpsc::Sender<Event>,
    client_id: String,
}

fn now_iso() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    let (y, mo, day) = civil_from_days((secs / 86400) as i64);
    format!("{y}-{mo:02}-{day:02}T{h:02}:{m:02}:{s:02}.000Z")
}

/// Convert days since Unix epoch to (year, month, day).
/// Algorithm from Howard Hinnant (public domain, used in chrono/date.h).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

impl TelemetryClient {
    pub fn new(client_id: String) -> Self {
        let (tx, rx) = mpsc::channel::<Event>();
        let cid = client_id.clone();

        thread::spawn(move || {
            let version = env!("CARGO_PKG_VERSION").to_string();
            let agent = ureq::Agent::config_builder()
                .timeout_global(Some(std::time::Duration::from_secs(5)))
                .build()
                .new_agent();
            while let Ok(event) = rx.recv() {
                let batch = Batch {
                    schema_version: 1,
                    client_id: cid.clone(),
                    app: App {
                        name: APP_NAME.into(),
                        version: version.clone(),
                    },
                    sent_at: now_iso(),
                    batch_id: uuid::Uuid::new_v4().to_string(),
                    events: vec![event],
                };

                let json = match serde_json::to_string(&batch) {
                    Ok(j) => j,
                    Err(e) => {
                        log::warn!("Telemetry serialize error: {e}");
                        continue;
                    }
                };

                match agent
                    .post(ENDPOINT)
                    .header("Content-Type", "application/json")
                    .send(json.as_str())
                {
                    Ok(_) => log::debug!("Telemetry event sent"),
                    Err(e) => log::debug!("Telemetry send failed (non-fatal): {e}"),
                }
            }
        });

        Self { tx, client_id }
    }

    fn send(&self, name: &str, props: Option<serde_json::Value>) {
        let event = Event {
            schema_version: 1,
            ts: now_iso(),
            name: name.into(),
            client_id: self.client_id.clone(),
            props,
        };
        let _ = self.tx.send(event);
    }

    pub fn app_open(&self) {
        self.send("app_open", None);
    }

    pub fn context(&self, gpu: &str, os: &str, ai_status: &str) {
        self.send(
            "context",
            Some(serde_json::json!({
                "os": os,
                "gpu": gpu,
                "ai": ai_status,
            })),
        );
    }

    pub fn source_info(&self, width: u32, height: u32, fps: f64, decoder: &str, sync_offset: i64) {
        self.send(
            "source_info",
            Some(serde_json::json!({
                "width": width,
                "height": height,
                "fps": fps,
                "decoder": decoder,
                "sync_offset": sync_offset,
            })),
        );
    }

    pub fn bug_report(&self, report: &str) {
        self.send(
            "bug_report",
            Some(serde_json::json!({
                "report": fit_bug_report(report),
            })),
        );
    }

    pub fn export_complete(&self, frames: u64, duration_secs: f64, codec: &str) {
        self.send(
            "export_complete",
            Some(serde_json::json!({
                "frames": frames,
                "duration_sec": duration_secs,
                "codec": codec,
            })),
        );
    }

    pub fn export_error(&self, error: &str, codec: &str) {
        self.send(
            "export_error",
            Some(serde_json::json!({
                "error_type": "export_failed",
                "error_message": truncate_at_char_boundary(error, 500),
                "codec": codec,
            })),
        );
    }

    pub fn calibration_complete(&self, confidence: f64, matches: usize) {
        self.send(
            "calibration_complete",
            Some(serde_json::json!({
                "confidence": confidence,
                "matches": matches,
            })),
        );
    }

    pub fn calibration_error(&self, error: &str) {
        self.send(
            "calibration_error",
            Some(serde_json::json!({
                "error_message": truncate_at_char_boundary(error, 500),
            })),
        );
    }
}

/// Raw-byte budget for a bug report. The ingestion server rejects events
/// whose props exceed its size cap (128KB, measured after JSON re-encoding)
/// rather than truncating them; escaping inflates a raw byte at most
/// sixfold, so 16KB raw can never reach the cap. Covers the full 200-line
/// log tail in the common case.
const MAX_REPORT_BYTES: usize = 16 * 1024;

/// Fit a bug report under [`MAX_REPORT_BYTES`], dropping the OLDEST log
/// lines first: the head (user description, contact, environment) and the
/// newest log lines carry the diagnosis. The head is only cut if it alone
/// exceeds the budget.
fn fit_bug_report(report: &str) -> String {
    if report.len() <= MAX_REPORT_BYTES {
        return report.to_string();
    }
    // The exact header `build_bug_report` appends: a header line, an
    // opening fence, the log lines, a closing fence. Matching through
    // "(last " keeps a user-typed "## Log" in the free-text description
    // from being misread as the section start.
    let Some((head, log_rest)) = report.split_once("\n## Log (last ") else {
        return truncate_at_char_boundary(report, MAX_REPORT_BYTES).to_string();
    };
    let lines: Vec<&str> = log_rest.lines().collect();
    // lines[0] is the header remainder and lines[1] the opening fence;
    // both are re-created below, so keep only the log content (whose last
    // line is the closing fence).
    let content: &[&str] = if lines.len() > 2 { &lines[2..] } else { &[] };
    // Budgeting with the largest possible marker text never undercounts.
    let header = |dropped: usize| format!("\n## Log (oldest {dropped} lines dropped)\n```\n");
    let overhead = head.len() + header(content.len()).len();
    if overhead >= MAX_REPORT_BYTES {
        return truncate_at_char_boundary(head, MAX_REPORT_BYTES).to_string();
    }
    let budget = MAX_REPORT_BYTES - overhead;
    let mut keep = 0;
    let mut used = 0;
    for line in content.iter().rev() {
        if used + line.len() + 1 > budget {
            break;
        }
        used += line.len() + 1;
        keep += 1;
    }
    let kept = &content[content.len() - keep..];
    let mut s = String::with_capacity(overhead + used + 8);
    s.push_str(head);
    s.push_str(&header(content.len() - keep));
    for l in kept {
        s.push_str(l);
        s.push('\n');
    }
    // A non-empty suffix always ends with the original closing fence;
    // only the everything-dropped case needs one re-added.
    if kept.is_empty() {
        s.push_str("```\n");
    }
    s
}

/// Truncate to at most `max` bytes at a char boundary; a raw byte slice
/// panics when `max` lands inside a multi-byte char.
fn truncate_at_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_report_is_unchanged() {
        let report = "## User description\nit broke\n\n## Log (last 200 lines)\n```\nline\n```\n";
        assert_eq!(fit_bug_report(report), report);
    }

    fn synthetic_report(n_lines: usize) -> String {
        let mut r = String::from(
            "## User description\ncrash on export\n\n## Contact\nuser@example.com\n\
             \n## Environment\n- Reco v1.0\n- OS: linux x86_64\n",
        );
        r.push_str("\n## Log (last 200 lines)\n```\n");
        for i in 0..n_lines {
            r.push_str(&format!(
                "[INFO] log line number {i} padded with enough detail to take up realistic space\n"
            ));
        }
        r.push_str("```\n");
        r
    }

    #[test]
    fn oversized_report_keeps_head_and_newest_lines() {
        let fitted = fit_bug_report(&synthetic_report(400));
        assert!(fitted.len() <= MAX_REPORT_BYTES);
        assert!(fitted.contains("## User description"));
        assert!(fitted.contains("user@example.com"));
        assert!(fitted.contains("log line number 399"));
        assert!(!fitted.contains("log line number 0 "));
        assert!(fitted.contains("lines dropped"));
    }

    #[test]
    fn oversized_report_without_log_section_is_cut_at_cap() {
        let fitted = fit_bug_report(&"x".repeat(20_000));
        assert!(fitted.len() <= MAX_REPORT_BYTES);
        assert!(fitted.starts_with("xxx"));
    }

    #[test]
    fn oversized_head_is_cut_even_with_log_section() {
        let report = format!(
            "{}\n## Log (last 200 lines)\n```\nline\n```\n",
            "y".repeat(20_000)
        );
        let fitted = fit_bug_report(&report);
        assert!(fitted.len() <= MAX_REPORT_BYTES);
        assert!(fitted.starts_with("yyy"));
    }

    #[test]
    fn char_boundary_truncation_never_panics() {
        // Multi-byte char straddling the cut: é is 2 bytes.
        let s = "é".repeat(300);
        assert_eq!(truncate_at_char_boundary(&s, 499).len(), 498);
        assert_eq!(truncate_at_char_boundary("short", 500), "short");
    }
}
