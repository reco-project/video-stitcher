//! Native D3D11 ownership checks using a short file fixture.

use super::*;
use crate::InputPath;
use crate::ffmpeg::decoder::DecodeBackend;
use reco_core::source::{FrameSet, FrameSource};

fn input() -> InputPath {
    let path = std::path::PathBuf::from(std::env::var("RECO_DECODE_TEST_VIDEO").unwrap());
    assert_eq!(
        VideoDecoder::open(&path).unwrap().backend(),
        DecodeBackend::D3d11va
    );
    InputPath::Single(path)
}

#[test]
#[ignore = "requires Windows D3D11VA and RECO_DECODE_TEST_VIDEO, a short multi-frame file"]
fn d3d11_owned_pair_drains_eof_and_propagates_failure() {
    let input = input();
    let mut reference = VideoDecoder::open_input_software(&input).unwrap();
    let mut total = 0;
    while reference.next_frame().unwrap().is_some() {
        total += 1;
    }
    assert!(total > 3);
    for offset in [-3_i64, 0, 3] {
        let stream = spawn_d3d11_decode_pair(&input, &input, offset).unwrap();
        let mut count = 0;
        while stream.receive().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, total - offset.unsigned_abs());
    }
    let missing = input
        .first_path()
        .with_file_name("missing-d3d11-ownership-test.mp4");
    assert!(!missing.exists());
    let broken = InputPath::Chained(vec![input.first_path().to_owned(), missing]);
    for (left, right) in [(&broken, &input), (&input, &broken)] {
        let stream = spawn_d3d11_decode_pair(left, right, 0).unwrap();
        let mut count = 0;
        loop {
            match stream.receive() {
                Ok(Some(_)) => count += 1,
                Err(_) => break,
                Ok(None) => panic!("missing segment reported as EOF"),
            }
        }
        assert!(count > 0 && count <= total);
    }
}

#[test]
#[ignore = "requires Windows D3D11VA and RECO_DECODE_TEST_VIDEO, a short multi-frame file"]
fn d3d11_source_repeated_close_releases_current_and_queued_frames() {
    let input = input();
    let gpu = reco_core::gpu::GpuContext::new_blocking().unwrap();
    for offset in [-3_i64, 0, 3, i64::MIN, i64::MAX] {
        for _ in 0..3 {
            let mut source = crate::SmartFileSource::open(&input, &input, &gpu, offset).unwrap();
            assert!(source.is_gpu_resident());
            source.start_decoding();
            if offset.unsigned_abs() <= 3 {
                assert!(matches!(
                    source.next_frame().unwrap(),
                    Some(FrameSet::D3d11Resident(_))
                ));
            }
            // Model a stalled consumer so native surfaces remain queued at close.
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(source);
        }
    }
}
