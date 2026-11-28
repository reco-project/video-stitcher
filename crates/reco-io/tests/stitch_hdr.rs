//! File exports select delivery precision from the source colour metadata.

#![cfg(feature = "ffmpeg")]

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::projection::LShape;
use reco_io::StitchJob;
use reco_io::stitch_job::StitchError;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

fn run_command(command: &mut Command) -> Vec<u8> {
    let output = command.output().expect("run FFmpeg tool");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn fixture(path: &Path, hdr: bool) {
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg.args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"]);
    let source = format!("nullsrc=s={WIDTH}x{HEIGHT}:r=10");
    if hdr {
        // The adjacent neutral luma values 512 and 513 both round to 128 in 8-bit YUV.
        // Tag the generated frames so FFmpeg passes the colour metadata to the encoder.
        ffmpeg.arg(format!(
            "{source},format=yuv420p10le,geq=lum='512+mod(N,2)':cb=512:cr=512,\
             setparams=range=tv:color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc"
        ));
        ffmpeg.args([
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "lossless=1:pools=none:frame-threads=1",
            "-color_range",
            "tv",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-colorspace",
            "bt2020nc",
        ]);
    } else {
        ffmpeg.arg(format!("{source},format=yuv420p,geq=lum=128:cb=128:cr=128"));
        ffmpeg.args(["-c:v", "libx264", "-preset", "ultrafast"]);
    }
    run_command(ffmpeg.args(["-frames:v", "6"]).arg(path));
    if hdr {
        assert_hlg_metadata(&video_info(path), path);
    }
}

fn job(input: &Path, output: &Path) -> StitchJob {
    let lens = Lens::fisheye(WIDTH, HEIGHT, 128.0, 128.0, 128.0, 128.0, [0.0; 4]);
    let calibration = Calibration::new(
        vec![lens.clone(), lens],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    StitchJob::with_calibration([input, input], calibration, output).resolution(WIDTH, HEIGHT)
}

fn video_info(path: &Path) -> serde_json::Value {
    let json = run_command(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
            ])
            .arg(path),
    );
    let info: serde_json::Value = serde_json::from_slice(&json).unwrap();
    let stream = &info["streams"][0];
    assert_eq!(stream["nb_read_frames"], "6");
    stream.clone()
}

fn assert_hlg_metadata(stream: &serde_json::Value, path: &Path) {
    for (field, expected) in [
        ("color_primaries", "bt2020"),
        ("color_transfer", "arib-std-b67"),
        ("color_space", "bt2020nc"),
    ] {
        assert_eq!(stream[field], expected, "{}: {field}", path.display());
    }
}

#[test]
fn sdr_defaults_to_8_bit_h264() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("sdr.mp4");
    let output = dir.path().join("stitched.mp4");
    fixture(&input, false);
    let job = job(&input, &output);
    #[cfg(feature = "gpu")]
    let job = if gpu_available() { job } else { job.cpu() };
    job.run(&Arc::new(AtomicBool::new(false))).unwrap();
    let video = video_info(&output);
    assert_eq!(video["codec_name"], "h264");
    assert_eq!(video["pix_fmt"], "yuv420p");
}

#[test]
fn hdr_rejects_cpu_rendering_before_opening_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("hlg.mp4");
    let output = dir.path().join("stitched.mp4");
    fixture(&input, true);
    let job = job(&input, &output).on_session(|_, _| panic!("created a session for HDR on CPU"));
    // Without GPU support the ordinary job also takes the CPU path.
    #[cfg(feature = "gpu")]
    let job = job.cpu();
    let error = job.run(&Arc::new(AtomicBool::new(false))).unwrap_err();
    assert!(matches!(error, StitchError::Other(message)
        if message.contains("10-bit HDR export requires GPU rendering")));
    assert!(!output.exists());
}

#[cfg(feature = "gpu")]
fn gpu_available() -> bool {
    let available = reco_core::gpu::GpuContext::new_blocking().is_ok();
    if !available {
        eprintln!("no GPU context available");
    }
    available
}

#[cfg(feature = "gpu")]
#[test]
fn hdr_rejects_h264_before_opening_outputs() {
    if !gpu_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("hlg.mp4");
    let output = dir.path().join("stitched.mp4");
    fixture(&input, true);
    let error = job(&input, &output)
        .codec(reco_io::output::Codec::H264)
        .on_session(|_, _| panic!("created a session for HDR with H.264"))
        .run(&Arc::new(AtomicBool::new(false)))
        .unwrap_err();
    assert!(matches!(error, StitchError::Other(message)
        if message.contains("10-bit HDR export requires HEVC Main 10")));
    assert!(!output.exists());
}

#[cfg(feature = "gpu")]
#[test]
fn hdr_defaults_to_main10_with_hlg_metadata_and_8_bit_replay() {
    if !gpu_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("hlg.mp4");
    let output = dir.path().join("stitched.mp4");
    let replay = dir.path().join("replay.mkv");
    fixture(&input, true);
    let native = Arc::new(AtomicBool::new(false));
    let source_residency = native.clone();
    let result = job(&input, &output)
        .quality(reco_io::output::Quality::Value(100))
        .with_replay_recording(&replay)
        .on_session(move |_, source| {
            source_residency.store(
                source.is_gpu_resident(),
                std::sync::atomic::Ordering::Relaxed,
            );
            Ok(())
        })
        .run(&Arc::new(AtomicBool::new(false)))
        .unwrap();
    assert_eq!(result.frames_processed, 6);
    let video = video_info(&output);
    assert_eq!(video["codec_name"], "hevc");
    assert_eq!(video["profile"], "Main 10");
    assert_eq!(video["pix_fmt"], "yuv420p10le");
    assert_hlg_metadata(&video, &output);

    let replay = video_info(&replay);
    assert_eq!(replay["codec_name"], "h264");
    assert_eq!(replay["pix_fmt"], "yuv420p");

    if native.load(std::sync::atomic::Ordering::Relaxed) {
        let bytes = run_command(
            Command::new("ffmpeg")
                .args(["-v", "error", "-i"])
                .arg(&output)
                .args([
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv420p10le",
                    "-threads",
                    "1",
                    "pipe:1",
                ]),
        );
        let pixels = (WIDTH * HEIGHT) as usize;
        let center = (HEIGHT as usize / 2 * WIDTH as usize + WIDTH as usize / 2) * 2;
        let luma: Vec<_> = bytes
            .chunks_exact(pixels * 3)
            .map(|frame| u16::from_le_bytes(frame[center..center + 2].try_into().unwrap()))
            .collect();
        assert_eq!(luma, [512, 513, 512, 513, 512, 513]);
    } else {
        eprintln!(
            "{} quantizes input before rendering; native P010 precision check skipped",
            result.decode_mode
        );
    }
}
