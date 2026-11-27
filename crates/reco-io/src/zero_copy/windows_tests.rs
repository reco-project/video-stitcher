//! Native D3D11 ownership checks using a short file fixture.

use super::*;
use crate::InputPath;
use crate::ffmpeg::decoder::{DecodeBackend, SharedHwDevice, create_shared_hw_device};
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
fn shared_d3d11_decoders_start_concurrently_and_drain() {
    let single = input();
    let chained = InputPath::Chained(vec![
        single.first_path().to_owned(),
        single.first_path().to_owned(),
    ]);
    for input in [single, chained] {
        let mut reference = VideoDecoder::open_input_software(&input).unwrap();
        let mut expected = 0;
        while reference.next_frame().unwrap().is_some() {
            expected += 1;
        }
        assert!(expected > 3);
        for _ in 0..128 {
            let shared = create_shared_hw_device().unwrap();
            assert_eq!(shared.backend(), DecodeBackend::D3d11va);
            let start = std::sync::Barrier::new(2);
            let decode = |device: SharedHwDevice| {
                start.wait();
                let mut decoder =
                    VideoDecoder::open_input_with_shared_device(&input, &device).unwrap();
                let mut count = 0;
                while decoder.next_frame_d3d11().unwrap().is_some() {
                    count += 1;
                }
                assert_eq!(count, expected);
            };
            std::thread::scope(|scope| {
                let other = shared.new_ref();
                let worker = scope.spawn(|| decode(other));
                decode(shared);
                worker.join().unwrap();
            });
        }
    }
}

#[test]
#[ignore = "requires Windows D3D11VA and RECO_DECODE_TEST_VIDEO, a short multi-frame file"]
fn d3d11_owned_frames_drain_eof_and_propagate_failure() {
    let input = input();
    let mut reference = VideoDecoder::open_input_software(&input).unwrap();
    let mut total = 0;
    while reference.next_frame().unwrap().is_some() {
        total += 1;
    }
    assert!(total > 3);
    for offset in [-3_i64, 0, 3] {
        let stream =
            spawn_d3d11_decode_frames(&[input.clone(), input.clone()], offset, None, None).unwrap();
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
        let stream =
            spawn_d3d11_decode_frames(&[left.clone(), right.clone()], 0, None, None).unwrap();
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
            let mut source =
                crate::SmartFileSource::open(&[input.clone(), input.clone()], &gpu, offset)
                    .unwrap();
            assert!(source.is_gpu_resident());
            source.start_decoding();
            if offset.unsigned_abs() <= 3 {
                assert!(matches!(
                    source.next_frame().unwrap().map(|frame| frame.images),
                    Some(FrameSet::D3d11Resident(_))
                ));
            }
            // Model a stalled consumer so native surfaces remain queued at close.
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(source);
        }
    }
}
