//! CLI arguments and publication policy for the shared video-calibration workflow.

use reco_calibrate::{
    CalibrationConfig,
    video::{CalibrateVideosOptions, calibrate_videos},
};
use std::{path::Path, sync::atomic::AtomicBool};

/// File-calibration inputs and explicit overrides.
#[derive(clap::Args)]
pub(crate) struct Args {
    /// Path to the left camera video file.
    left: String,

    /// Path to the right camera video file.
    right: String,

    /// Path to the left camera lens profile JSON.
    /// Missing profiles are estimated from footage; supplied lenses stay fixed.
    /// With only one supplied, estimates the other camera's optics.
    #[arg(long)]
    left_profile: Option<String>,

    /// Path to the right camera lens profile JSON.
    /// Independent of --left-profile; a missing side is estimated.
    #[arg(long)]
    right_profile: Option<String>,

    /// Number of frame pairs (1..=16). The first fits; the rest verify alignment.
    #[arg(long, default_value_t = 4)]
    frames: usize,

    /// Frame offset for temporal sync between cameras.
    /// Right frame index minus left frame index for the same event.
    /// Omit to measure timing automatically. Use 0 for synchronized videos.
    #[arg(long, allow_hyphen_values = true)]
    sync_offset: Option<i64>,

    /// Seconds to skip from the start of the recording.
    #[arg(long, default_value_t = 0.0)]
    skip_start: f64,

    /// Seconds to skip from the end of the video (e.g. teardown).
    #[arg(long, default_value_t = 0.0)]
    skip_end: f64,

    /// Save a candidate even when independent checks require visual review. Invalid fits still fail.
    #[arg(long)]
    accept_unconfirmed: bool,

    /// Output calibration JSON file path.
    #[arg(short, long, default_value = "match.json")]
    output: String,
}

/// Fit through the same workflow as the desktop consumer.
pub(crate) fn run_calibrate(args: Args, interrupted: &AtomicBool) -> anyhow::Result<()> {
    let load = |path: Option<&str>| -> anyhow::Result<_> {
        path.map(|path| reco_calibrate::lens_database::load_from_file(Path::new(path)))
            .transpose()
            .map_err(Into::into)
    };
    let result = calibrate_videos(
        Path::new(&args.left),
        Path::new(&args.right),
        CalibrateVideosOptions {
            left_params: load(args.left_profile.as_deref())?,
            right_params: load(args.right_profile.as_deref())?,
            config: Some(CalibrationConfig {
                num_frames: args.frames,
                skip_start_secs: args.skip_start,
                skip_end_secs: args.skip_end,
            }),
            sync_offset: args.sync_offset,
        },
        &mut |progress| eprintln!("[{}] {}", progress.step, progress.detail),
        interrupted,
    )
    .map_err(|error| match error {
        reco_calibrate::video::CalibrateVideosError::TimingRequired => anyhow::anyhow!(
            "Could not determine synchronization from IMU or audio. Rerun with --sync-offset <frames> (right frame minus left frame for the same event). Use 0 if the videos are synchronized."
        ),
        other => other.into(),
    })?;
    publish_result(&result, Path::new(&args.output), args.accept_unconfirmed)?;
    print_results(&result, &args.output, args.frames);
    Ok(())
}

fn publish_result(
    result: &reco_calibrate::CalibrationResult,
    output: &Path,
    accept_unconfirmed: bool,
) -> anyhow::Result<()> {
    if let reco_calibrate::CalibrationAssessment::NeedsReview { reason } = &result.assessment
        && !accept_unconfirmed
    {
        // A report retains the candidate without publishing it as an active calibration.
        let mut report = output.as_os_str().to_os_string();
        report.push(".review.json");
        let report = std::path::PathBuf::from(report);
        std::fs::write(&report, serde_json::to_string_pretty(&result)?)?;
        anyhow::bail!(
            "{reason} Candidate and measurements: {}. Requested output was not modified.",
            report.display()
        );
    }

    // Write output
    let json = serde_json::to_string_pretty(&result.calibration)?;
    std::fs::write(output, &json)?;

    Ok(())
}

fn print_results(result: &reco_calibrate::CalibrationResult, output: &str, total_pairs: usize) {
    eprintln!("\nCalibration results:");
    eprintln!("  Output:          {output}");
    eprintln!("  Frames used:     {}/{total_pairs}", result.frames_used);
    eprintln!("  Total matches:   {}", result.total_matches);
    eprintln!("  Assessment:      {:?}", result.assessment);
    eprintln!("\nPlacement parameters:");
    let l = result
        .calibration
        .projection
        .l_shape()
        .expect("stereo calibration always produces the L-shape projection");
    eprintln!(
        "  cameraAxisOffset: {:.4}",
        result.calibration.framing.axis_offset
    );
    eprintln!("  intersect:        {:.4}", l.intersect);
    eprintln!("  xTy:              {:.6}", l.x_ty);
    eprintln!("  xRz:              {:.6}", l.x_rz);
    eprintln!("  zRx:              {:.6}", l.z_rx);
    eprintln!("  zRz:              {:.6}", l.z_rz);

    for (i, check) in result.overlap_checks.iter().enumerate() {
        eprintln!(
            "  Independent pair {}: {} matches, median {:?}px, p95 {:?}px",
            i + 1,
            check.matches,
            check.median_pixels,
            check.p95_pixels
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_calibrate::{CalibrationAssessment, CalibrationResult};
    use reco_core::calibration::{Calibration, Framing, Lens};
    use reco_core::projection::LShape;

    #[test]
    fn explicit_zero_is_distinct_from_automatic_timing() {
        use clap::Parser;
        #[derive(Parser)]
        struct Command {
            #[command(flatten)]
            args: Args,
        }
        let defaults = Command::try_parse_from(["calibrate", "left.mp4", "right.mp4"])
            .unwrap()
            .args;
        assert_eq!(defaults.sync_offset, None);
        let explicit =
            Command::try_parse_from(["calibrate", "left.mp4", "right.mp4", "--sync-offset", "0"])
                .unwrap()
                .args;
        assert_eq!(explicit.sync_offset, Some(0));
        for obsolete in ["--estimate-optics", "--no-auto-imu"] {
            assert!(
                Command::try_parse_from(["calibrate", "left.mp4", "right.mp4", obsolete]).is_err()
            );
        }
    }

    #[test]
    fn review_report_preserves_active_file_until_explicit_acceptance() {
        let directory = std::env::temp_dir().join(format!(
            "reco-calibration-publication-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let output = directory.join("calibration.json");
        std::fs::write(&output, "original document").unwrap();
        let mut result = CalibrationResult {
            calibration: Calibration::new(
                vec![Lens::fisheye(64, 64, 40.0, 40.0, 32.0, 32.0, [0.0; 4]); 2],
                LShape {
                    intersect: 0.6,
                    x_ty: 0.0,
                    x_rx: 0.0,
                    x_rz: 0.0,
                    z_rx: 0.0,
                    z_rz: 0.0,
                    blend_width: 0.05,
                },
                Framing {
                    axis_offset: 0.23,
                    tilt: 0.0,
                    roll: 0.0,
                },
            ),
            total_matches: 24,
            frames_used: 1,
            assessment: CalibrationAssessment::NeedsReview {
                reason: "insufficient evidence".into(),
            },
            overlap_checks: vec![],
            left_lens_profile: None,
            right_lens_profile: None,
        };
        let error = publish_result(&result, &output, false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Requested output was not modified")
        );
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "original document"
        );
        let report = directory.join("calibration.json.review.json");
        let candidate: CalibrationResult =
            serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
        assert_eq!(candidate.calibration, result.calibration);
        assert_eq!(candidate.assessment, result.assessment);
        publish_result(&result, &output, true).unwrap();
        assert_eq!(Calibration::from_file(&output).unwrap(), result.calibration);
        result.assessment = CalibrationAssessment::Confirmed;
        result.calibration.framing.roll = 0.1;
        publish_result(&result, &output, false).unwrap();
        assert_eq!(Calibration::from_file(&output).unwrap(), result.calibration);
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(report).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
