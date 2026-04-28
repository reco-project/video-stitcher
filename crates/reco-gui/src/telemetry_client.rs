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

const ENDPOINT: &str =
    "https://telemetry-ingestion-204135919265.us-central1.run.app/telemetry";
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
    let days = secs / 86400;
    let y = 1970 + days / 365;
    format!("{y}-01-01T{h:02}:{m:02}:{s:02}.000Z")
}

impl TelemetryClient {
    pub fn new(client_id: String) -> Self {
        let (tx, rx) = mpsc::channel::<Event>();
        let cid = client_id.clone();

        thread::spawn(move || {
            let version = env!("CARGO_PKG_VERSION").to_string();
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

                match ureq::post(ENDPOINT)
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

    pub fn bug_report(&self, report: &str) {
        self.send(
            "bug_report",
            Some(serde_json::json!({
                "report": &report[..report.len().min(1800)],
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

    pub fn export_error(&self, error: &str) {
        self.send(
            "export_error",
            Some(serde_json::json!({
                "error_type": "export_failed",
                "error_message": &error[..error.len().min(500)],
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
}
