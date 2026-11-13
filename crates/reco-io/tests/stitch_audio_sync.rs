//! File-job audio must follow the same source offset as synchronized video.

#![cfg(feature = "ffmpeg")]

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::projection::LShape;
use reco_io::StitchJob;
use reco_io::ffmpeg::decoder::VideoDecoder;
use reco_io::output::AudioMode;

fn fixture(path: &Path) {
    // Each second has a different tone, exposing a one-second audio/video slip.
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("color=c=gray:s=64x64:r=30:d=4")
        .args(["-f", "lavfi", "-i"])
        .arg("aevalsrc=0.5*sin(2*PI*(400+200*floor(t))*t):s=48000:d=4")
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-y",
        ])
        .arg(path)
        .output()
        .expect("ffmpeg fixture encoder");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn calibration(offset: i64) -> Calibration {
    let lens = Lens::fisheye(64, 64, 32.0, 32.0, 32.0, 32.0, [0.0; 4]);
    let mut cal = Calibration::new(
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
    cal.sync_offset = offset;
    cal
}

fn tone_hz(path: &Path) -> f64 {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args([
            "-ss", "0.1", "-t", "0.2", "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1",
        ])
        .output()
        .expect("ffmpeg audio decoder");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let samples: Vec<_> = output
        .stdout
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert!(samples.len() >= 9000, "missing output audio");
    let crossings = samples
        .windows(2)
        .filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0)
        .count();
    crossings as f64 * 48_000.0 / samples.len() as f64
}

fn check_offsets(cases: &[(i64, usize, f64, bool)]) {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    fixture(&input);
    for &(offset, camera, start, explicit) in cases {
        let output = dir
            .path()
            .join(format!("out-{offset}-{camera}-{start}-{explicit}.mp4"));
        // A conflicting saved offset proves an explicit job offset takes precedence.
        let mut job = StitchJob::with_calibration(
            [input.clone(), input.clone()],
            calibration(if explicit { -offset } else { offset }),
            &output,
        )
        .cpu()
        .resolution(64, 64)
        .audio(AudioMode::CopyFrom(camera))
        .encoder_name("libx264")
        .start_time(start)
        .max_frames(15)
        .on_session(|session, _| {
            session.core_mut().set_constrained_look(false);
            Ok(())
        });
        if explicit {
            job = job.sync_offset(offset);
        }
        job.run(&Arc::new(AtomicBool::new(false))).unwrap();

        let skipped = match (offset, camera) {
            (value, 0) if value < 0 => -value,
            (value, 1) if value > 0 => value,
            _ => 0,
        };
        let expected = 400.0 + 200.0 * (start + skipped as f64 / 30.0).floor();
        let measured = tone_hz(&output);
        assert!(
            (measured - expected).abs() < 15.0,
            "offset={offset} camera={camera} start={start} explicit={explicit}: tone {measured}, expected {expected}"
        );
        let mut decoder = VideoDecoder::open_software(&output).unwrap();
        let mut count = 0;
        while decoder.next_frame().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 15);
    }
}

#[test]
fn negative_sync_trims_selected_left_audio() {
    check_offsets(&[(-30, 0, 0.0, false), (-30, 0, 1.0, true)]);
}

#[test]
fn positive_sync_trims_selected_right_audio() {
    check_offsets(&[(30, 1, 0.0, true), (30, 1, 1.0, false)]);
}

#[test]
fn unshifted_camera_and_zero_offset_keep_their_audio_origin() {
    check_offsets(&[
        (-30, 1, 1.0, false),
        (30, 0, 1.0, true),
        (0, 0, 0.0, false),
        (0, 1, 1.0, true),
    ]);
}
