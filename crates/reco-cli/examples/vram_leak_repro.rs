//! Repro harness for the cross-export VRAM leak.
//!
//! The GUI runs multiple exports per session; users on 8 GB GPUs report the
//! second export starting VRAM-starved because the first export's GPU
//! resources are not reclaimed. This runs [`StitchJob::run`] N times in ONE
//! process and prints GPU free/used VRAM (via `nvidia-smi`) before and after
//! each run, so a leak shows up as free VRAM that never recovers.
//!
//! Usage: `vram_leak_repro <left> <right> <cal.json> [runs=2] [max_frames=120]`

use std::process::Command;
use std::sync::atomic::AtomicBool;

fn vram() -> String {
    Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| format!("free/used MB = {}", s.trim()))
        .unwrap_or_else(|| "nvidia-smi unavailable".into())
}

fn main() {
    // Surface reco's `log::*` output (decode path, pool type, teardown) so the
    // leak source is visible. Bridges `log` -> tracing like the real CLI does.
    let _ = tracing_log::LogTracer::init();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <left> <right> <cal.json> [runs=2] [max_frames=120]",
            args[0]
        );
        std::process::exit(2);
    }
    let left = args[1].clone();
    let right = args[2].clone();
    let cal = args[3].clone();
    let runs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    let max_frames: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(120);
    // Optional: enable AI tracking + lookahead (allocates the lookahead pool,
    // the resource the bug report blames for the ~3.9 GB cross-export leak).
    let model = args.get(6).cloned();
    let lookahead: f64 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let interrupted = AtomicBool::new(false);
    println!(
        "baseline   {}   (mode: {})",
        vram(),
        if model.is_some() {
            "AI+lookahead"
        } else {
            "plain"
        }
    );

    for run in 1..=runs {
        println!("[run {run}] before: {}", vram());
        let out = format!("vram_repro_run{run}.mp4");
        let mut job = reco_io::StitchJob::new(left.as_str(), right.as_str(), cal.as_str(), &out)
            .resolution(1920, 1080)
            .max_frames(max_frames);

        if let Some(ref model_path) = model {
            let model_path = model_path.clone();
            job = job.lookahead(lookahead);
            job = job.on_session(move |session, source| {
                let cfg = reco_autocam::AutocamConfig::new(&model_path)
                    .with_tracking_mode(reco_autocam::TrackingMode::Field)
                    // High interval: detection accuracy is irrelevant to the
                    // leak test, and fewer detections = much faster runs.
                    .with_detection_interval(30);
                let fps = source.info().fps as f32;
                let gpu = source.is_gpu_resident();
                match reco_autocam::setup_autocam(session, &cfg, fps, gpu) {
                    Ok(active) => println!("    autocam active={active}"),
                    Err(e) => println!("    autocam setup failed: {e}"),
                }
            });
        }

        match job.run(&interrupted) {
            Ok(r) => println!("[run {run}] OK: {} frames written", r.frames_processed),
            Err(e) => println!("[run {run}] FAILED: {e}"),
        }
        println!("[run {run}] after:  {}", vram());
    }
    println!("final          {}", vram());
    // If VRAM recovers after a delay, the leak is late thread/device teardown
    // (decode threads exiting after run() returns); if it stays, it's a true leak.
    std::thread::sleep(std::time::Duration::from_secs(6));
    println!("final (+6s)    {}", vram());
}
