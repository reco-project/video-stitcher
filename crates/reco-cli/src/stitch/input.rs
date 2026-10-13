//! Live input policy at the command boundary; execution stays in reco-io.

use reco_io::rtsp::{RtspConfig, RtspInput};
use std::time::Duration;

/// Parse one selected camera's encoded audio, or an explicit opt-out.
pub fn parse_audio_source(value: &str) -> Result<reco_io::output::AudioMode, String> {
    if value == "none" {
        return Ok(reco_io::output::AudioMode::Disabled);
    }
    value
        .parse()
        .map(reco_io::output::AudioMode::CopyFrom)
        .map_err(|_| "expected a zero-based input index or 'none'".into())
}

/// RTSP controls for the existing stitch command.
#[derive(clap::Args)]
#[group(skip)]
pub struct LiveInputArgs {
    /// RTSP output cadence, e.g. 30 or 30000/1001. Required for RTSP input.
    #[arg(long, value_parser = parse_rate, requires = "rtsp_offsets_us", conflicts_with_all = ["start_time", "end_time", "sync_offset"])]
    rtsp_fps: Option<(u32, u32)>,

    /// One media timestamp offset per input, comma-separated in microseconds. Zero assumes an already shared timestamp origin; arrival order cannot synchronize cameras.
    #[arg(
        long,
        value_delimiter = ',',
        allow_hyphen_values = true,
        requires = "rtsp_fps"
    )]
    rtsp_offsets_us: Vec<i64>,

    /// Maximum wait for an RTSP output slot in milliseconds.
    #[arg(long, default_value_t = 150, requires = "rtsp_fps")]
    rtsp_wait_ms: u32,

    /// Maximum capture-time spread for a new matched RTSP image set in milliseconds.
    #[arg(long, default_value_t = 40, requires = "rtsp_fps")]
    rtsp_skew_ms: u32,

    /// Maximum RTSP image reuse age on the mapped media timeline in milliseconds.
    #[arg(long, default_value_t = 500, requires = "rtsp_fps")]
    rtsp_max_age_ms: u32,

    /// Pending images per camera in each RTSP decode and selection queue.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..), requires = "rtsp_fps")]
    rtsp_queue: u32,

    /// RTSP initial-image deadline in milliseconds.
    #[arg(long, default_value_t = 5000, value_parser = clap::value_parser!(u32).range(1..), requires = "rtsp_fps")]
    rtsp_open_timeout_ms: u32,

    /// RTSP decoded-image read deadline in milliseconds.
    #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u32).range(1..), requires = "rtsp_fps")]
    rtsp_read_timeout_ms: u32,
}

impl LiveInputArgs {
    pub(super) fn job(
        &self,
        inputs: &[String],
        calibration: reco_core::calibration::Calibration,
        output: &str,
    ) -> anyhow::Result<reco_io::StitchJob> {
        let network = inputs.iter().any(|input| input.contains("://"));
        if !network {
            anyhow::ensure!(self.rtsp_fps.is_none(), "RTSP options require RTSP inputs");
            let inputs = inputs
                .iter()
                .map(|input| reco_io::InputPath::parse_segments(input));
            return Ok(reco_io::StitchJob::with_calibration(
                inputs,
                calibration,
                output,
            ));
        }
        anyhow::ensure!(
            inputs.iter().all(|input| input.starts_with("rtsp://")),
            "network input requires only rtsp:// URLs; file/stream mixing is unsupported"
        );
        let fps = self.rtsp_fps.ok_or_else(|| {
            anyhow::anyhow!("RTSP input requires --rtsp-fps and --rtsp-offsets-us")
        })?;
        anyhow::ensure!(
            self.rtsp_offsets_us.len() == inputs.len(),
            "--rtsp-offsets-us needs one offset per RTSP input"
        );
        anyhow::ensure!(
            self.rtsp_max_age_ms >= self.rtsp_skew_ms,
            "--rtsp-max-age-ms must be at least --rtsp-skew-ms"
        );
        let config = RtspConfig {
            fps,
            synchronization: reco_io::frame_sync::Config {
                wait_us: i64::from(self.rtsp_wait_ms) * 1000,
                skew_us: i64::from(self.rtsp_skew_ms) * 1000,
                max_age_us: i64::from(self.rtsp_max_age_ms) * 1000,
                capacity: self.rtsp_queue as usize,
            },
            open_timeout: Duration::from_millis(u64::from(self.rtsp_open_timeout_ms)),
            read_timeout: Duration::from_millis(u64::from(self.rtsp_read_timeout_ms)),
            software_decode: false,
        };
        let inputs = inputs
            .iter()
            .zip(&self.rtsp_offsets_us)
            .map(|(url, &timestamp_offset_us)| RtspInput {
                url: url.clone(),
                timestamp_offset_us,
            })
            .collect();
        log::info!(
            "RTSP video input: {}/{}, wait={} ms, skew={} ms, max age={} ms, queue={}; available selected audio follows the same connection",
            fps.0,
            fps.1,
            self.rtsp_wait_ms,
            self.rtsp_skew_ms,
            self.rtsp_max_age_ms,
            self.rtsp_queue
        );
        Ok(reco_io::StitchJob::from_rtsp(
            inputs,
            config,
            calibration,
            output,
        ))
    }
}

fn parse_rate(value: &str) -> Result<(u32, u32), String> {
    let invalid =
        || "expected positive FPS or numerator/denominator (e.g. 30 or 30000/1001)".to_string();
    let (n, d) = value.split_once('/').unwrap_or((value, "1"));
    let n = n.parse::<u32>().map_err(|_| invalid())?;
    let d = d.parse::<u32>().map_err(|_| invalid())?;
    if n == 0
        || d == 0
        || n > i32::MAX as u32
        || d > i32::MAX as u32
        || u64::from(n) > 1_000_000 * u64::from(d)
    {
        return Err(invalid());
    }
    Ok((n, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn audio_selection_is_one_camera_or_explicitly_disabled() {
        use reco_io::output::AudioMode;
        let base = ["reco", "stitch", "a.mp4", "b.mp4", "-c", "cal.json"];
        for (text, expected) in [("0", Some(0)), ("1", Some(1)), ("none", None)] {
            let parsed =
                crate::Cli::try_parse_from(base.into_iter().chain(["--audio-source", text]))
                    .unwrap();
            let crate::Commands::Stitch { audio_source, .. } = parsed.command else {
                panic!("stitch");
            };
            match (audio_source, expected) {
                (AudioMode::CopyFrom(actual), Some(expected)) => assert_eq!(actual, expected),
                (AudioMode::Disabled, None) => {}
                _ => panic!("audio selection changed"),
            }
        }
        for value in ["-1", "all", "0,1", "1.5", ""] {
            assert!(parse_audio_source(value).is_err());
        }
    }

    #[test]
    fn rtsp_controls_require_explicit_clock_mapping_and_no_file_window() {
        let base = [
            "reco",
            "stitch",
            "rtsp://127.0.0.1/left",
            "rtsp://127.0.0.1/right",
            "-c",
            "cal.json",
        ];
        assert!(crate::Cli::try_parse_from(base.into_iter().chain(["--rtsp-fps", "30"])).is_err());
        assert!(
            crate::Cli::try_parse_from(base.into_iter().chain(["--rtsp-offsets-us", "0,0"]))
                .is_err()
        );
        let complete = ["--rtsp-fps", "30000/1001", "--rtsp-offsets-us=0,-7000000"];
        let parsed = crate::Cli::try_parse_from(base.into_iter().chain(complete)).unwrap();
        let crate::Commands::Stitch { live_input, .. } = parsed.command else {
            panic!("stitch")
        };
        assert_eq!(live_input.rtsp_fps, Some((30000, 1001)));
        assert_eq!(live_input.rtsp_offsets_us, [0, -7_000_000]);
        for option in ["--start-time", "--end-time", "--sync-offset"] {
            assert!(
                crate::Cli::try_parse_from(base.into_iter().chain(complete).chain([option, "0"]))
                    .is_err(),
                "{option}"
            );
        }
        for value in ["0", "0/1", "1/0", "30/1/1", "NaN", "2147483648", "1000001"] {
            assert!(parse_rate(value).is_err(), "{value}");
        }
        assert_eq!(parse_rate("30").unwrap(), (30, 1));
        assert_eq!(parse_rate("24000/1001").unwrap(), (24000, 1001));
    }
}
