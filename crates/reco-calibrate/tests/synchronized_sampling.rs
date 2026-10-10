//! Sampling stays inside the shared portion of synchronized source clips.

use reco_calibrate::CalibrationConfig;
use reco_calibrate::pipeline::{CalibrationPipeline, VideoInfo};

fn indices(left_count: u64, right_count: u64, offset: i64) -> (Vec<u64>, Vec<u64>) {
    let info = |total_frames| VideoInfo {
        path: "unused.mp4".into(),
        width: 320,
        height: 180,
        fps: 30.0,
        total_frames,
    };
    let mut pipeline = CalibrationPipeline::new(
        info(left_count),
        info(right_count),
        CalibrationConfig::default(),
    );
    pipeline.set_sync_offset(offset);
    pipeline.frame_indices()
}

#[test]
fn both_offset_directions_sample_only_overlapping_frames() {
    for (left_count, right_count, offset) in [
        (100, 100, 70),
        (100, 100, -70),
        (60, 100, 70),
        (100, 60, -70),
    ] {
        let (left, right) = indices(left_count, right_count, offset);
        assert!(!left.is_empty());
        assert_eq!(left.len(), right.len());
        for (left, right) in left.into_iter().zip(right) {
            assert!(
                left < left_count,
                "left={left}, count={left_count}, offset={offset}"
            );
            assert!(
                right < right_count,
                "right={right}, count={right_count}, offset={offset}"
            );
            assert_eq!(right as i128 - left as i128, offset as i128);
        }
    }
}

#[test]
fn nonoverlapping_clips_and_signed_extremes_return_no_samples() {
    for offset in [100, -100, 101, -101, i64::MAX, i64::MIN] {
        let (left, right) = indices(100, 100, offset);
        assert!(left.is_empty(), "offset {offset}: {left:?}");
        assert!(right.is_empty(), "offset {offset}: {right:?}");
    }
}

#[test]
fn one_shared_frame_is_a_valid_pair() {
    assert_eq!(indices(100, 100, 99), (vec![0], vec![99]));
    assert_eq!(indices(100, 100, -99), (vec![99], vec![0]));
}

#[test]
fn unsynchronized_sampling_keeps_existing_distribution() {
    let config = CalibrationConfig::default();
    let expected = reco_calibrate::sampling::select_frame_indices(
        100,
        30.0,
        config.num_frames,
        config.skip_start_secs,
        config.skip_end_secs,
    );
    assert_eq!(indices(100, 150, 0), (expected.clone(), expected));
}

#[test]
fn varied_clip_lengths_keep_pairs_in_bounds_and_order() {
    for left_count in [0, 1, 2, 17, 63, 100, 1000, u64::MAX] {
        for right_count in [0, 1, 2, 17, 63, 100, 1000, u64::MAX] {
            for offset in [
                -1001,
                -100,
                -63,
                -1,
                0,
                1,
                63,
                100,
                1001,
                i64::MIN,
                i64::MAX,
            ] {
                let (left, right) = indices(left_count, right_count, offset);
                assert_eq!(left.len(), right.len());
                assert!(left.windows(2).all(|p| p[0] < p[1]));
                assert!(right.windows(2).all(|p| p[0] < p[1]));
                for (left, right) in left.into_iter().zip(right) {
                    assert!(left < left_count);
                    assert!(right < right_count);
                    assert_eq!(right as i128 - left as i128, offset as i128);
                }
            }
        }
    }
}
