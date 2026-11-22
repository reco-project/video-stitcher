//! Integration tests for calibration I/O helpers.
//!
//! Self-contained fixtures exercise exact image selection. Ignored checks additionally require the local test footage collection.

#![cfg(feature = "ffmpeg")]

use std::path::Path;

use reco_io::ffmpeg::calibration_io;

fn fixture() -> std::path::PathBuf {
    std::env::var_os("RECO_TEST_VIDEO")
        .expect("set RECO_TEST_VIDEO to a video with audio")
        .into()
}

#[test]
#[ignore]
fn probe_video_returns_valid_metadata() {
    reco_io::init();

    let probe = calibration_io::probe_video(&fixture()).unwrap();

    assert!(probe.width > 0);
    assert!(probe.height > 0);
    assert!(probe.fps > 0.0, "fps: {}", probe.fps);
    assert!(
        probe.total_frames > 100,
        "total_frames: {}",
        probe.total_frames
    );
}

#[test]
#[ignore]
fn extract_frames_returns_correct_dimensions() {
    reco_io::init();

    let indices = vec![30, 60, 90];
    let frames = calibration_io::extract_frames(&fixture(), &indices).unwrap();

    assert_eq!(frames.len(), 3);
    for frame in &frames {
        assert!(frame.width > 0);
        assert!(frame.height > 0);
        // Y plane: width * height bytes
        assert_eq!(frame.y.len(), (frame.width * frame.height) as usize);
        // U plane: (width/2) * (height/2) bytes
        assert_eq!(frame.u.len(), (frame.width / 2 * frame.height / 2) as usize);
    }
}

#[test]
#[ignore]
fn extract_audio_returns_samples() {
    let samples = calibration_io::extract_audio_pcm(&fixture(), 44100).unwrap();

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

fn numbered_clip(path: &Path, times: &[Option<i64>]) {
    use reco_core::sink::{OutputFrame, OutputSink, PixelFormat};
    use reco_io::adapters::FfmpegFileEncoder;
    use reco_io::ffmpeg::encoder::EncoderConfig;

    let config = EncoderConfig {
        encoder_name: Some("libx264".into()),
        ..Default::default()
    };
    let mut encoder = FfmpegFileEncoder::new(path, 64, 64, (30, 1), &config).unwrap();
    let mut pixels = vec![128; 64 * 64 * 3 / 2];
    for (index, &pts_us) in times.iter().enumerate() {
        pixels[..64 * 64].fill(40 + index as u8 * 40);
        encoder
            .consume(OutputFrame {
                data: &pixels,
                width: 64,
                height: 64,
                format: PixelFormat::Nv12,
                pts_us,
            })
            .unwrap();
    }
    encoder.finish().unwrap();
}

#[test]
fn extraction_preserves_requested_order_duplicates_and_image_time() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("numbered.mp4");
    numbered_clip(&path, &[None; 3]);
    let indices = [2, 0, 1, 2];
    let frames = calibration_io::extract_frames(&path, &indices).unwrap();
    assert_eq!(frames.len(), indices.len());
    for (frame, index) in frames.iter().zip(indices) {
        assert!(
            (frame.timestamp_us - (index as f64 / 30.0 * 1_000_000.0).round() as i64).abs() <= 1
        );
        assert!(frame.y[32 * 64 + 32].abs_diff(40 + index as u8 * 40) <= 2);
    }
}

#[test]
fn extraction_does_not_substitute_or_skip_unavailable_frames() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("short.mp4");
    numbered_clip(&path, &[None; 3]);
    for indices in [&[3][..], &[0, 10, 2][..], &[u64::MAX][..]] {
        assert!(
            calibration_io::extract_frames(&path, indices).is_err(),
            "accepted {indices:?}"
        );
    }
    assert!(matches!(
        calibration_io::extract_frames(&path, &[0, 3, 1]),
        Err(calibration_io::CalibrationIoError::FrameUnavailable(3))
    ));
}

#[test]
fn extraction_does_not_substitute_a_later_image_across_a_timestamp_gap() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gap.mp4");
    numbered_clip(&path, &[Some(0), Some(33_333), Some(200_000)]);
    assert!(matches!(
        calibration_io::extract_frames(&path, &[1]),
        Err(calibration_io::CalibrationIoError::FrameUnavailable(1))
    ));
}
