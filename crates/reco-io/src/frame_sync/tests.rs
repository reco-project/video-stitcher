use super::*;
use std::sync::Arc;

fn config() -> Config {
    Config {
        wait_us: 20,
        skew_us: 5,
        max_age_us: 65,
        capacity: 4,
    }
}
fn selector(n: usize) -> Selector<Arc<i64>> {
    Selector::new(n, config()).unwrap()
}
fn push(s: &mut Selector<Arc<i64>>, camera: usize, capture: i64, arrived: i64) {
    s.push(
        camera,
        Capture {
            time_us: capture,
            image: Arc::new(capture),
        },
        arrived,
    )
    .unwrap();
}
fn seed(s: &mut Selector<Arc<i64>>, n: usize) {
    for camera in 0..n {
        push(s, camera, 0, 0);
    }
    assert!(s.poll(0, 0).unwrap().is_some());
}
fn captures(set: &Selection<Arc<i64>>) -> Vec<i64> {
    set.captures.iter().map(|capture| capture.time_us).collect()
}

#[test]
fn skipped_slot_preserves_future_captures_and_clock_guards() {
    let mut s = selector(2);
    assert_eq!(s.skip_pending(), Err(Error::Clock));
    push(&mut s, 0, 10, 10);
    push(&mut s, 1, 20, 20);
    push(&mut s, 0, 30, 30);
    push(&mut s, 1, 32, 32);
    assert!(s.poll(10, 29).unwrap().is_none());
    assert_eq!(s.skip_pending(), Err(Error::Clock));
    assert!(matches!(s.poll(10, 32), Err(Error::Missing(1))));
    s.skip_pending().unwrap();
    assert_eq!(s.skip_pending(), Err(Error::Clock));
    assert!(matches!(s.poll(10, 32), Err(Error::Clock)));
    assert!(matches!(s.poll(30, 31), Err(Error::Clock)));
    let next = s.poll(30, 32).unwrap().unwrap();
    assert_eq!(captures(&next), vec![30, 32]);
    assert_eq!(s.skip_pending(), Err(Error::Clock));
    assert!(matches!(s.poll(30, 50), Err(Error::Clock)));
}

#[test]
fn skipping_skew_failure_does_not_change_identity_or_limits() {
    let mut s = selector(2);
    seed(&mut s, 2);
    push(&mut s, 0, 100, 100);
    push(&mut s, 1, 90, 90);
    push(&mut s, 0, 130, 130);
    push(&mut s, 1, 131, 131);
    assert!(matches!(s.poll(100, 120), Err(Error::Skew)));
    s.skip_pending().unwrap();
    assert_eq!(
        captures(&s.poll(130, 150).unwrap().unwrap()),
        vec![130, 131]
    );
    assert!(matches!(s.poll(200, 220), Err(Error::Expired(0))));
    s.skip_pending().unwrap();
    assert_eq!(
        s.push(
            0,
            Capture {
                time_us: 130,
                image: Arc::new(130)
            },
            230
        ),
        Err(Error::Clock)
    );
    // Missing one deadline does not prevent use in a later admissible slot.
    push(&mut s, 0, 230, 251);
    push(&mut s, 1, 231, 251);
    assert!(matches!(s.poll(230, 251), Err(Error::Expired(0))));
    s.skip_pending().unwrap();
    assert_eq!(
        captures(&s.poll(260, 280).unwrap().unwrap()),
        vec![230, 231]
    );
}

#[test]
fn skipping_releases_retained_and_obsolete_owners_without_flushing_future_images() {
    let mut s = selector(2);
    push(&mut s, 0, 0, 5);
    push(&mut s, 1, 5, 5);
    s.poll(0, 5).unwrap().unwrap();
    let held = s.previous[1].as_ref().unwrap().image.clone();
    let old = Arc::new(60);
    let future = Arc::new(100);
    s.push(
        1,
        Capture {
            time_us: 60,
            image: old.clone(),
        },
        60,
    )
    .unwrap();
    s.push(
        0,
        Capture {
            time_us: 100,
            image: future.clone(),
        },
        100,
    )
    .unwrap();
    assert_eq!(Arc::strong_count(&held), 2);
    // Camera 0's previous image expires, but the other retained image is still age-valid.
    assert!(s.poll(70, 89).unwrap().is_none());
    assert_eq!(s.skip_pending(), Err(Error::Clock));
    assert_eq!(Arc::strong_count(&old), 2);
    assert_eq!(Arc::strong_count(&future), 2);
    // The pending slot remains available after the caller observes a clock error.
    assert!(matches!(s.poll(71, 90), Err(Error::Clock)));
    assert_eq!(s.skip_pending(), Err(Error::Clock));
    assert!(matches!(s.poll(70, 90), Err(Error::Expired(0))));
    s.skip_pending().unwrap();
    assert_eq!(Arc::strong_count(&held), 1);
    assert_eq!(Arc::strong_count(&old), 1);
    assert_eq!(Arc::strong_count(&future), 2);
    push(&mut s, 1, 101, 101);
    let next = s.poll(100, 101).unwrap().unwrap();
    assert_eq!(captures(&next), vec![100, 101]);
    assert!(Arc::ptr_eq(&next.captures[0].image, &future));
}

#[test]
fn repeated_missing_slots_resume_from_queued_images_for_each_camera() {
    for inputs in 1..=3 {
        for missing in 0..inputs {
            let mut s = selector(inputs);
            seed(&mut s, inputs);
            for cycle in 1..=1_000 {
                let slot = cycle * 100;
                for camera in 0..inputs {
                    if camera != missing {
                        push(&mut s, camera, slot, slot);
                    }
                    push(&mut s, camera, slot + 30 + camera as i64, slot + 32);
                }
                assert!(matches!(s.poll(slot, slot + 20), Err(Error::Expired(c)) if c == missing));
                s.skip_pending().unwrap();
                assert!(matches!(s.poll(slot, slot + 32), Err(Error::Clock)));
                let selected = s.poll(slot + 30, slot + 32).unwrap().unwrap();
                for (camera, capture) in selected.captures.iter().enumerate() {
                    assert_eq!(capture.time_us, slot + 30 + camera as i64);
                    assert_eq!(*capture.image, capture.time_us);
                }
                assert!(s.queues.iter().all(VecDeque::is_empty));
            }
        }
    }
}

#[test]
fn expired_owners_are_released_before_waiting_for_replacement() {
    let mut s = selector(1);
    push(&mut s, 0, 0, 0);
    let first = s.poll(0, 0).unwrap().unwrap();
    let weak = Arc::downgrade(&first.captures[0].image);
    drop(first);
    assert!(s.poll(100, 90).unwrap().is_none());
    assert!(
        weak.upgrade().is_none(),
        "an expired image must not starve its producer's pool"
    );
    assert!(matches!(s.poll(100, 120), Err(Error::Expired(0))));
    assert!(matches!(
        s.push(
            0,
            Capture {
                time_us: 0,
                image: Arc::new(0)
            },
            120
        ),
        Err(Error::Clock)
    ));
    push(&mut s, 0, 102, 110);
    assert_eq!(captures(&s.poll(100, 120).unwrap().unwrap()), vec![102]);
}

#[test]
fn independent_receipt_queues_can_be_drained_after_a_poll() {
    let mut s = selector(2);
    seed(&mut s, 2);
    push(&mut s, 0, 30, 38);
    assert!(s.poll(30, 40).unwrap().is_none());
    push(&mut s, 1, 30, 32);
    assert_eq!(captures(&s.poll(30, 40).unwrap().unwrap()), vec![30, 30]);
}

#[test]
fn future_receipts_cannot_be_selected_early() {
    let mut s = selector(2);
    push(&mut s, 0, 30, 45);
    push(&mut s, 1, 30, 35);
    assert!(s.poll(30, 40).unwrap().is_none());
    assert_eq!(captures(&s.poll(30, 45).unwrap().unwrap()), vec![30, 30]);
}

#[test]
fn delivery_bursts_preserve_the_due_frame_and_next_slot() {
    for inputs in 1..=3 {
        for origin in [0, 9_000_000_000_000, i64::MAX - 1_000_000] {
            let mut s = Selector::new(
                inputs,
                Config {
                    wait_us: 150_000,
                    skew_us: 40_000,
                    max_age_us: 500_000,
                    capacity: 4,
                },
            )
            .unwrap();
            for camera in 0..inputs {
                push(&mut s, camera, origin, origin);
            }
            s.poll(origin, origin).unwrap().unwrap();
            let due = origin + 33_366;
            let following = origin + 66_733;
            let arrived = origin + 75_000;
            for camera in 0..inputs {
                push(&mut s, camera, due, arrived);
                push(&mut s, camera, following, arrived);
            }
            let first = s.poll(due, arrived).unwrap().unwrap();
            assert_eq!(captures(&first), vec![due; inputs]);
            assert_eq!(first.at_us, due);
            assert!(s.queues.iter().all(|queue| queue.len() == 1));
            let second = s.poll(following, arrived).unwrap().unwrap();
            assert_eq!(captures(&second), vec![following; inputs]);
            assert_eq!(second.at_us, following);
            assert!(s.queues.iter().all(VecDeque::is_empty));
        }
    }
}

#[test]
fn nearest_set_is_neither_oldest_nor_newest_and_keeps_future_captures() {
    let mut s = Selector::new(
        2,
        Config {
            skew_us: 20,
            ..config()
        },
    )
    .unwrap();
    for camera in 0..2 {
        for time in [10, 29, 40] {
            push(&mut s, camera, time, 40);
        }
    }
    assert_eq!(captures(&s.poll(30, 40).unwrap().unwrap()), vec![29; 2]);
    assert_eq!(captures(&s.poll(40, 40).unwrap().unwrap()), vec![40; 2]);
}

#[test]
fn nearest_peer_is_found_when_the_earliest_queue_has_no_successor() {
    let mut s = Selector::new(
        2,
        Config {
            skew_us: 20,
            ..config()
        },
    )
    .unwrap();
    push(&mut s, 0, 1, 19);
    for time in [9, 10, 19] {
        push(&mut s, 1, time, 19);
    }
    assert_eq!(captures(&s.poll(10, 19).unwrap().unwrap()), vec![1, 10]);
}

#[test]
fn equal_timestamp_distance_prefers_the_later_compatible_set() {
    let mut s = selector(2);
    for camera in 0..2 {
        for time in [25, 35] {
            push(&mut s, camera, time, 35);
        }
    }
    assert_eq!(captures(&s.poll(30, 35).unwrap().unwrap()), vec![35; 2]);
}

#[test]
fn one_two_three_inputs_recover_in_every_camera_position() {
    for n in 1..=3 {
        for missing in 0..n {
            let mut s = selector(n);
            seed(&mut s, n);
            for camera in 0..n {
                if camera != missing {
                    push(&mut s, camera, 30, 30);
                }
            }
            assert!(s.poll(30, 49).unwrap().is_none());
            let set = s.poll(30, 50).unwrap().unwrap();
            assert_eq!(set.at_us, 30);
            for (camera, frame) in set.captures.iter().enumerate() {
                assert_eq!(frame.time_us, if camera == missing { 0 } else { 30 });
            }
            for camera in 0..n {
                push(&mut s, camera, 60, 60);
            }
            assert_eq!(captures(&s.poll(60, 60).unwrap().unwrap()), vec![60; n]);
        }
    }
}

#[test]
fn future_of_slot_has_valid_media_assessment_time() {
    let mut s = selector(3);
    for camera in 0..3 {
        push(&mut s, camera, camera as i64 * 2, 4);
    }
    let set = s.poll(0, 4).unwrap().unwrap();
    assert_eq!(set.at_us, 4);
    assert_eq!(captures(&set), vec![0, 2, 4]);
    assert!(set.captures.iter().all(|frame| frame.time_us <= set.at_us));
    assert!(matches!(s.poll(4, 5), Err(Error::Clock)));
}

#[test]
fn a_retained_image_can_pair_better_than_its_newer_replacement() {
    let mut s = Selector::new(
        2,
        Config {
            wait_us: 20,
            skew_us: 20,
            max_age_us: 65,
            capacity: 4,
        },
    )
    .unwrap();
    push(&mut s, 0, 12, 12);
    push(&mut s, 1, 0, 12);
    let first = s.poll(0, 12).unwrap().unwrap();
    push(&mut s, 0, 45, 45);
    push(&mut s, 1, 24, 45);
    assert!(s.poll(30, 45).unwrap().is_none());
    let held = s.poll(30, 50).unwrap().unwrap();
    assert_eq!(captures(&held), vec![12, 24]);
    assert!(Arc::ptr_eq(
        &first.captures[0].image,
        &held.captures[0].image
    ));
    push(&mut s, 1, 55, 55);
    assert_eq!(captures(&s.poll(60, 60).unwrap().unwrap()), vec![45, 55]);
}

#[test]
fn output_phase_does_not_discard_an_age_valid_new_pair() {
    let mut s = selector(2);
    push(&mut s, 0, 22, 24);
    push(&mut s, 1, 24, 24);
    assert_eq!(captures(&s.poll(30, 50).unwrap().unwrap()), vec![22, 24]);
}

#[test]
fn max_individual_error_does_not_allow_double_pair_skew() {
    let mut s = selector(2);
    push(&mut s, 0, 0, 10);
    push(&mut s, 1, 10, 10);
    assert!(matches!(s.poll(5, 25), Err(Error::Skew)));
}

#[test]
fn incompatible_new_images_hold_the_previous_set_without_weakening_skew() {
    let mut s = selector(3);
    seed(&mut s, 3);
    push(&mut s, 0, 25, 35);
    push(&mut s, 1, 35, 35);
    assert_eq!(captures(&s.poll(30, 50).unwrap().unwrap()), vec![0; 3]);
    assert!(matches!(s.poll(66, 86), Err(Error::Expired(_))));
}

#[test]
fn delayed_backlog_preserves_an_age_valid_held_set_then_rejoins() {
    for inputs in 2..=4 {
        for delayed in 0..inputs {
            for recover in [false, true] {
                let mut s = Selector::new(
                    inputs,
                    Config {
                        wait_us: 150_000,
                        skew_us: 40_000,
                        max_age_us: 500_000,
                        capacity: 4,
                    },
                )
                .unwrap();
                seed(&mut s, inputs);
                for camera in 0..inputs {
                    if camera != delayed {
                        push(&mut s, camera, 100_000, 100_000);
                    }
                }
                let held = s.poll(100_000, 250_000).unwrap().unwrap();
                assert_eq!(held.captures[delayed].time_us, 0);
                for camera in 0..inputs {
                    if camera == delayed {
                        push(&mut s, camera, 33_333, 300_000);
                    } else {
                        assert_eq!(held.captures[camera].time_us, 100_000);
                        push(&mut s, camera, 200_000, 200_000);
                    }
                }
                assert!(s.poll(200_000, 349_999).unwrap().is_none());
                let still_held = s.poll(200_000, 350_000).unwrap().unwrap();
                assert_eq!(captures(&still_held), captures(&held));
                assert_eq!(still_held.at_us, 200_000);
                for camera in 0..inputs {
                    assert!(Arc::ptr_eq(
                        &held.captures[camera].image,
                        &still_held.captures[camera].image
                    ));
                    assert_eq!(s.queues[camera].len(), 1);
                }
                if recover {
                    for camera in 0..inputs {
                        push(&mut s, camera, 333_333, 400_000);
                    }
                    let recovered = s.poll(333_333, 400_000).unwrap().unwrap();
                    assert_eq!(captures(&recovered), vec![333_333; inputs]);
                    assert!(s.queues.iter().all(VecDeque::is_empty));
                } else {
                    let last = s.poll(500_000, 650_000).unwrap().unwrap();
                    assert_eq!(captures(&last), captures(&held));
                    assert!(s.poll(500_001, 650_001).is_err());
                }
            }
        }
    }
}

#[test]
fn partial_inputs_use_the_same_joint_search_as_a_complete_set() {
    let mut s = selector(3);
    seed(&mut s, 3);
    push(&mut s, 0, 25, 35);
    push(&mut s, 0, 35, 35);
    push(&mut s, 1, 26, 35);
    assert_eq!(captures(&s.poll(30, 50).unwrap().unwrap()), vec![25, 26, 0]);
}

#[test]
fn joint_search_finds_a_set_that_independent_nearest_can_miss() {
    let mut s = selector(3);
    push(&mut s, 0, 1, 9);
    push(&mut s, 0, 9, 9);
    push(&mut s, 1, 8, 9);
    push(&mut s, 2, 7, 9);
    assert_eq!(captures(&s.poll(5, 9).unwrap().unwrap()), vec![9, 8, 7]);
}

#[test]
fn deadline_is_absolute_even_if_poll_is_late() {
    let mut s = selector(2);
    seed(&mut s, 2);
    push(&mut s, 0, 30, 30);
    push(&mut s, 1, 30, 51);
    let set = s.poll(30, 52).unwrap().unwrap();
    assert_eq!(captures(&set), vec![30, 0]);
    assert_eq!(set.at_us, 30);
}

#[test]
fn exact_deadline_delivery_is_eligible() {
    let mut s = selector(2);
    push(&mut s, 0, 30, 30);
    push(&mut s, 1, 30, 50);
    assert_eq!(captures(&s.poll(30, 51).unwrap().unwrap()), vec![30; 2]);
}

#[test]
fn waiting_does_not_reset_deadline_or_change_age() {
    let mut s = selector(1);
    seed(&mut s, 1);
    for now in 30..50 {
        assert!(s.poll(30, now).unwrap().is_none());
    }
    assert!(matches!(s.poll(31, 49), Err(Error::Clock)));
    assert_eq!(s.poll(30, 50).unwrap().unwrap().at_us, 30);
    assert_eq!(s.poll(60, 80).unwrap().unwrap().at_us, 60);
    assert!(matches!(s.poll(90, 110), Err(Error::Expired(0))));
}

#[test]
fn age_bound_includes_future_peer_shift_and_failure_is_atomic() {
    let mut s = selector(2);
    seed(&mut s, 2);
    push(&mut s, 1, 66, 66);
    assert!(matches!(s.poll(65, 85), Err(Error::Expired(0))));
    assert_eq!(s.emitted, Some(0));
    assert_eq!(s.queues[1][0].capture.time_us, 66);
}

#[test]
fn inclusive_age_limit_then_both_frozen_expire() {
    let mut s = selector(3);
    seed(&mut s, 3);
    assert_eq!(captures(&s.poll(65, 85).unwrap().unwrap()), vec![0; 3]);
    assert!(matches!(s.poll(66, 86), Err(Error::Expired(0))));
}

#[test]
fn missing_startup_does_not_invent_a_capture() {
    let mut s = selector(2);
    push(&mut s, 1, 0, 0);
    assert!(matches!(s.poll(0, 20), Err(Error::Missing(0))));
}

#[test]
fn owned_nonclone_payload_is_retained_and_queues_release_old_handles() {
    struct Native;
    let mut s = Selector::new(1, config()).unwrap();
    let original = Arc::new(Native);
    let weak = Arc::downgrade(&original);
    s.push(
        0,
        Capture {
            time_us: 0,
            image: original.clone(),
        },
        0,
    )
    .unwrap();
    let first = s.poll(0, 0).unwrap().unwrap();
    assert!(Arc::ptr_eq(&original, &first.captures[0].image));
    let held = s.poll(30, 50).unwrap().unwrap();
    assert!(Arc::ptr_eq(&original, &held.captures[0].image));
    drop(original);
    drop(first);
    drop(s);
    assert!(weak.upgrade().is_some());
    drop(held);
    assert!(weak.upgrade().is_none());

    let mut s = selector(3);
    let old = Arc::new(1);
    let weak = Arc::downgrade(&old);
    s.push(
        0,
        Capture {
            time_us: 1,
            image: old,
        },
        1,
    )
    .unwrap();
    for time in 2..100 {
        for camera in 0..3 {
            push(&mut s, camera, time, time);
        }
        assert!(s.queues.iter().all(|queue| queue.len() <= 4));
    }
    assert!(weak.upgrade().is_none());
    assert_eq!(captures(&s.poll(99, 119).unwrap().unwrap()), vec![99; 3]);
}

#[test]
fn invalid_time_input_does_not_poison_next_valid_delivery() {
    let mut s = selector(2);
    let frame = |time| Capture {
        time_us: time,
        image: Arc::new(time),
    };
    assert_eq!(s.push(2, frame(0), 0), Err(Error::Clock));
    assert_eq!(s.push(0, frame(-1), 0), Err(Error::Clock));
    assert_eq!(s.push(0, frame(1), 0), Err(Error::Clock));
    push(&mut s, 0, 0, 0);
    assert_eq!(s.push(0, frame(0), 1), Err(Error::Clock));
    push(&mut s, 1, 0, 0);
    assert!(s.poll(0, 0).unwrap().is_some());
    assert!(matches!(s.poll(i64::MAX, i64::MAX), Err(Error::Clock)));
    assert!(s.poll(1, 1).unwrap().is_none());
    assert_eq!(s.push(0, frame(1), 0), Err(Error::Clock));
}

#[test]
fn configuration_rejects_nonsensical_limits() {
    assert!(Selector::<()>::new(0, config()).is_err());
    for config in [
        Config {
            capacity: 0,
            ..config()
        },
        Config {
            wait_us: -1,
            ..config()
        },
        Config {
            skew_us: -1,
            ..config()
        },
        Config {
            max_age_us: 4,
            ..config()
        },
    ] {
        assert!(Selector::<()>::new(1, config).is_err());
    }
}

// Exhaustive Cartesian oracle, intentionally different from the ordered sweep.
fn nearest_match(queues: &[Vec<i64>], slot: i64, skew: i64) -> Option<Vec<i64>> {
    fn visit(
        queues: &[Vec<i64>],
        selected: &mut Vec<i64>,
        slot: i64,
        skew: i64,
        best: &mut Option<(u128, Vec<i64>)>,
    ) {
        if let Some((queue, rest)) = queues.split_first() {
            for &time in queue {
                selected.push(time);
                visit(rest, selected, slot, skew, best);
                selected.pop();
            }
        } else if selected.iter().max().unwrap() - selected.iter().min().unwrap() <= skew {
            let distance = selected
                .iter()
                .map(|&time| u128::from(time.abs_diff(slot)))
                .sum();
            match best {
                Some((previous, values)) if *previous == distance => {
                    for (value, &time) in values.iter_mut().zip(selected.iter()) {
                        *value = (*value).max(time);
                    }
                }
                _ if best
                    .as_ref()
                    .is_none_or(|(previous, _)| distance < *previous) =>
                {
                    *best = Some((distance, selected.clone()));
                }
                _ => {}
            }
        }
    }
    let mut best = None;
    visit(queues, &mut Vec::new(), slot, skew, &mut best);
    best.map(|(_, values)| values)
}

#[test]
fn three_input_sweep_matches_brute_force_nearest_set() {
    let possibilities: Vec<Vec<i64>> = (1u32..16)
        .map(|mask| {
            [0, 3, 7, 10]
                .into_iter()
                .enumerate()
                .filter_map(|(i, time)| (mask & (1 << i) != 0).then_some(time))
                .collect()
        })
        .collect();
    for a in &possibilities {
        for b in &possibilities {
            for c in &possibilities {
                let queues = [a.clone(), b.clone(), c.clone()];
                let expected = nearest_match(&queues, 5, 5);
                let mut s = selector(3);
                for (camera, times) in queues.iter().enumerate() {
                    for &time in times {
                        push(&mut s, camera, time, 10);
                    }
                }
                match s.poll(5, 25) {
                    Ok(Some(set)) => {
                        assert_eq!(Some(captures(&set)), expected, "{queues:?}");
                    }
                    Err(Error::Skew) => assert!(expected.is_none(), "{queues:?}"),
                    other => panic!("unexpected result: {other:?}"),
                }
            }
        }
    }
}

#[test]
fn retained_candidate_sweep_matches_exhaustive_nearest_compatible_set() {
    let possibilities: Vec<Vec<i64>> = (1u32..16)
        .map(|mask| {
            [6, 8, 12, 15]
                .into_iter()
                .enumerate()
                .filter_map(|(i, time)| (mask & (1 << i) != 0).then_some(time))
                .collect()
        })
        .collect();
    for a in &possibilities {
        for b in &possibilities {
            for c in &possibilities {
                let mut s = selector(3);
                for (camera, time) in [0, 3, 5].into_iter().enumerate() {
                    push(&mut s, camera, time, 5);
                }
                s.poll(5, 5).unwrap().unwrap();
                let queues: Vec<Vec<i64>> = [a, b, c]
                    .into_iter()
                    .enumerate()
                    .map(|(camera, times)| {
                        for &time in times {
                            push(&mut s, camera, time, 15);
                        }
                        std::iter::once([0, 3, 5][camera])
                            .chain(times.iter().copied())
                            .collect()
                    })
                    .collect();
                let actual = s.poll(10, 30).unwrap().unwrap();
                assert_eq!(
                    Some(captures(&actual)),
                    nearest_match(&queues, 10, 5),
                    "{queues:?}"
                );
                assert!(
                    actual
                        .captures
                        .iter()
                        .all(|capture| *capture.image == capture.time_us)
                );
            }
        }
    }
}

#[test]
fn varied_input_counts_and_windows_match_exhaustive_nearest_sets() {
    let mut state = 42u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state >> 32
    };
    for _ in 0..4_000 {
        let inputs = 1 + next() as usize % 5;
        let slot = (next() % 60) as i64;
        let skew = (next() % 40) as i64;
        let mut s = Selector::new(
            inputs,
            Config {
                wait_us: 100,
                skew_us: skew,
                max_age_us: 200,
                capacity: 4,
            },
        )
        .unwrap();
        let mut queues = Vec::new();
        for camera in 0..inputs {
            let mut times = (0..4).map(|_| (next() % 81) as i64).collect::<Vec<_>>();
            times.sort_unstable();
            times.dedup();
            for &time in &times {
                push(&mut s, camera, time, 100);
            }
            times.retain(|&time| time <= slot + skew);
            queues.push(times);
        }
        let expected = nearest_match(&queues, slot, skew);
        match s.poll(slot, 200) {
            Ok(Some(actual)) => assert_eq!(
                Some(captures(&actual)),
                expected,
                "slot={slot}, skew={skew}, queues={queues:?}"
            ),
            Err(Error::Skew | Error::Missing(_)) => assert!(expected.is_none(), "{queues:?}"),
            other => panic!("unexpected result: {other:?}"),
        }
    }
}

#[test]
fn long_loss_sweep_is_translation_invariant_and_bounded() {
    fn run(origin: i64) -> Vec<Vec<i64>> {
        let mut s = selector(3);
        let mut output = Vec::new();
        for slot in 0..10_000 {
            let target = origin + slot * 30;
            for camera in 0..3 {
                if slot == 0 || (slot + camera as i64 * 3) % 17 != 0 {
                    push(&mut s, camera, target, target);
                }
            }
            let set = s.poll(target, target + 20).unwrap().unwrap();
            assert!(
                set.captures
                    .iter()
                    .all(|frame| set.at_us - frame.time_us <= 65)
            );
            assert!(
                set.captures
                    .iter()
                    .all(|frame| *frame.image == frame.time_us)
            );
            output.push(captures(&set).iter().map(|time| time - origin).collect());
        }
        output
    }
    assert_eq!(run(0), run(9_000_000_000_000));
}
