//! Integration tests for calibration I/O helpers.
//!
//! These tests require real video files at the standard test footage
//! location. Run with `cargo test -p reco-io -- --ignored` when footage
//! is available.

use std::path::Path;

use reco_io::ffmpeg::calibration_io;

const LEFT_4K: &str = "/media/guelzim/HDD/reco-test-footage/gopro-hero10/left_4k.mp4";

fn have_test_footage() -> bool {
    Path::new(LEFT_4K).exists()
}

#[test]
#[ignore]
fn probe_video_returns_valid_metadata() {
    if !have_test_footage() {
        eprintln!("Skipping: test footage not available");
        return;
    }
    reco_io::init();

    let probe = calibration_io::probe_video(Path::new(LEFT_4K)).unwrap();

    // GoPro Hero 10 "4K" is actually 5312x2988 (5.3K sensor)
    assert!(probe.width > 0);
    assert!(probe.height > 0);
    assert!(probe.fps > 29.0 && probe.fps < 31.0, "fps: {}", probe.fps);
    assert!(probe.total_frames > 100, "total_frames: {}", probe.total_frames);
}

#[test]
#[ignore]
fn extract_frames_returns_correct_dimensions() {
    if !have_test_footage() {
        eprintln!("Skipping: test footage not available");
        return;
    }
    reco_io::init();

    let indices = vec![30, 60, 90];
    let frames = calibration_io::extract_frames(Path::new(LEFT_4K), &indices).unwrap();

    assert_eq!(frames.len(), 3);
    for frame in &frames {
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        // Y plane: width * height bytes
        assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);
        // U plane: (width/2) * (height/2) bytes
        assert_eq!(
            frame.u.len(),
            (frame.width / 2 * frame.height / 2) as usize
        );
    }
}

#[test]
#[ignore]
fn extract_audio_returns_samples() {
    if !have_test_footage() {
        eprintln!("Skipping: test footage not available");
        return;
    }

    let samples = calibration_io::extract_audio_pcm(Path::new(LEFT_4K), 44100).unwrap();

    // 60 seconds at 44100 Hz = 2_646_000 samples max
    assert!(!samples.is_empty());
    assert!(samples.len() > 44100, "too few samples: {}", samples.len());
}

#[test]
fn probe_nonexistent_file_returns_error() {
    reco_io::init();
    let result = calibration_io::probe_video(Path::new("/nonexistent/video.mp4"));
    assert!(result.is_err());
}

#[test]
fn extract_frames_nonexistent_file_returns_error() {
    reco_io::init();
    let result = calibration_io::extract_frames(Path::new("/nonexistent/video.mp4"), &[0]);
    assert!(result.is_err());
}
