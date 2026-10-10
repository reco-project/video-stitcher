//! Run native calibration on tightly packed luma fixtures without a GPU or decoder.

use reco_calibrate::Yuv420pFrame;
use std::sync::atomic::AtomicBool;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 7 || (args.len() - 1) % 6 != 0 {
        return Err("usage: align_luma PROFILE_CALIBRATION_OR_- LEFT_Y WIDTH HEIGHT RIGHT_Y WIDTH HEIGHT [...]".into());
    }
    let cal = if args[0] == "-" {
        None
    } else {
        Some(reco_core::calibration::Calibration::from_file(
            std::path::Path::new(&args[0]),
        )?)
    };
    let mut frames = Vec::new();
    for (i, pair) in args[1..].chunks_exact(6).enumerate() {
        let read = |args: &[String]| -> Result<_, Box<dyn std::error::Error>> {
            let width: u32 = args[1].parse()?;
            let height: u32 = args[2].parse()?;
            if width > 8192 || height > 8192 {
                return Err("oversized fixture".into());
            }
            Ok(Yuv420pFrame {
                width,
                height,
                y: std::fs::read(&args[0])?,
                u: vec![128; (width.div_ceil(2) * height.div_ceil(2)) as usize],
                v: vec![128; (width.div_ceil(2) * height.div_ceil(2)) as usize],
                timestamp_us: i as i64 * 1_000_000,
            })
        };
        frames.push((read(&pair[..3])?, read(&pair[3..])?));
    }
    let started = std::time::Instant::now();
    let result = reco_calibrate::calibrate(
        &frames,
        [
            cal.as_ref().and_then(|c| c.lenses.first()),
            cal.as_ref().and_then(|c| c.lenses.get(1)),
        ],
        &AtomicBool::new(false),
    )?;
    eprintln!(
        "Complete native calibration: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
