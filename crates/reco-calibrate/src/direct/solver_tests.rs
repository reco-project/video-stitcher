use super::diagnostics::{difference_step, inspect, solve};
use super::*;

#[test]
fn fitting_parameters_never_modify_a_supplied_lens() {
    let (seed, stage) = fixture(1);
    let seed: Calibration = serde_json::from_value(seed).unwrap();
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(10),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for fixed in 0..2 {
        let mut model = Model::new(seed.clone(), &stage, &pool, true, N, &control).unwrap();
        model.optics[fixed] = false;
        let changed = model.decode(&[0.1; N]).unwrap();
        assert_eq!(changed.lenses[fixed], seed.lenses[fixed]);
        assert_ne!(changed.lenses[1 - fixed], seed.lenses[1 - fixed]);
    }
}
use crate::direct::kernel::Pair;
use reco_core::calibration::{Framing, Lens};
use reco_core::projection::LShape;
use serde_json::{Value, json};

fn fixture(patches: usize) -> (Value, Stage) {
    fixture_with_frames(patches, 1)
}

fn fixture_with_frames(patches: usize, frames: usize) -> (Value, Stage) {
    let lens = Lens::fisheye(640, 480, 384.0, 384.0, 320.0, 240.0, [0.0; 4]);
    let cal = Calibration::new(
        vec![lens.clone(), lens],
        LShape {
            intersect: 0.59,
            x_ty: 0.0,
            x_rz: 0.0,
            x_rx: 0.0,
            z_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        Framing {
            axis_offset: 0.27,
            tilt: 0.0,
            roll: 0.0,
        },
    );
    let points = (0..patches * 49)
        .map(|i| {
            [
                0.80 + (i % 7) as f64 * 0.001,
                0.4 + (i % 49 / 7) as f64 * 0.001,
            ]
        })
        .collect();
    let stage = Stage::new(
        64,
        48,
        49,
        points,
        (0..frames)
            .map(|_| Pair {
                target: (0..patches * 49).map(|i| (i as f64).sin()).collect(),
                right: (0..64 * 48)
                    .map(|i| (i as f64 * 0.37).sin() * 0.1 + 0.4)
                    .collect(),
            })
            .collect(),
    )
    .unwrap();
    (serde_json::to_value(cal).unwrap(), stage)
}

fn request(seed: Value, method: &str) -> Value {
    json!({"seed":seed,"method":method,"x":vec![0.0;N],"budget":1000,"difference_step":0.00002})
}

#[test]
fn positive_sensor_margin_does_not_hide_a_boundary_crossing_stencil() {
    let (value, stage) = fixture(1);
    let mut seed: Calibration = serde_json::from_value(value).unwrap();
    // Equidistant capacity is analytic; only the second lens approaches it.
    let capacity = std::f64::consts::FRAC_PI_2 - 2e-6;
    seed.lenses[1].fx = 400.0 / (capacity - 1e-7);
    seed.lenses[1].fy = seed.lenses[1].fx;
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(10),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut model = Model::new(seed.clone(), &stage, &pool, true, N, &control).unwrap();
    model.difference_step = 0.00002;
    for shift in [-1e-8, 0.0, 1e-8] {
        let mut x = vec![0.0; N];
        x[12] = shift;
        assert!(model.margin(&x).unwrap() > 0.0);
        assert!(model.sensor_boundary_active(&x).unwrap());
    }
    let mut invalid = vec![0.0; N];
    invalid[12] = -0.001;
    assert!(model.margin(&invalid).unwrap() < 0.0);
    assert!(model.sensor_boundary_active(&invalid).unwrap());

    let fixed = Model::new(seed.clone(), &stage, &pool, true, 7, &control).unwrap();
    assert!(!fixed.sensor_boundary_active(&[0.0; 7]).unwrap());
    let constrained = Model::new(seed, &stage, &pool, false, N, &control).unwrap();
    assert!(!constrained.sensor_boundary_active(&[0.0; N]).unwrap());
}

#[test]
fn interior_optics_do_not_request_constrained_refinement() {
    let (value, stage) = fixture(1);
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(10),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let model = Model::new(
        serde_json::from_value(value).unwrap(),
        &stage,
        &pool,
        true,
        N,
        &control,
    )
    .unwrap();
    assert!(!model.sensor_boundary_active(&[0.0; N]).unwrap());
    interrupted.store(true, Ordering::Relaxed);
    assert_eq!(
        model.sensor_boundary_active(&[0.0; N]).unwrap_err(),
        "calibration cancelled"
    );
}

#[test]
fn fixed_optics_solve_and_inspection_preserve_lenses() {
    let (seed, _) = fixture(1);
    let cal = serde_json::from_value(seed.clone()).unwrap();
    let points: Vec<_> = (0..49)
        .map(|i| [0.80 + (i % 7) as f64 * 0.002, 0.4 + (i / 7) as f64 * 0.002])
        .collect();
    let right: Vec<_> = (0..64 * 48)
        .map(|i| (i as f64 * 0.37).sin() * 0.1 + 0.4)
        .collect();
    let construct = |target| {
        Stage::new(
            64,
            48,
            49,
            points.clone(),
            vec![Pair {
                target,
                right: right.clone(),
            }],
        )
        .unwrap()
    };
    let mut target = Vec::new();
    assert_eq!(
        construct(vec![0.0; 49])
            .residual(&cal, &mut target)
            .unwrap(),
        0
    );
    let stage = construct(target);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for method in ["lm", "trf", "slsqp"] {
        let mut input = request(seed.clone(), method);
        input["x"] = json!(vec![0.0; 7]);
        let result = solve(&input, &stage, &pool, &AtomicBool::new(false)).unwrap();
        assert_eq!(result["success"], true, "{method}: {result}");
        assert_eq!(result["calibration"]["lenses"], seed["lenses"]);
        assert_eq!(result["x"].as_array().unwrap().len(), 7);
        input["jacobian"] = json!(true);
        let (meta, values) = inspect(&input, &stage, &pool).unwrap();
        assert_eq!(meta["cols"], 7);
        assert_eq!(meta["rows"], stage.output_len() + 7);
        assert_eq!(values.len(), (stage.output_len() + 7) * 7);
    }
}

#[test]
fn fixed_optics_rejects_invalid_second_sensor() {
    let (mut seed, stage) = fixture(1);
    seed["lenses"][1]["fx"] = json!(120.0);
    seed["lenses"][1]["fy"] = json!(120.0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut input = request(seed, "lm");
    input["x"] = json!(vec![0.0; 7]);
    assert_eq!(
        solve(&input, &stage, &pool, &AtomicBool::new(false)).unwrap_err(),
        "fixed optics do not cover both sensors"
    );
}

#[test]
fn fixed_optics_columns_match_existing_placement_columns() {
    let (seed, stage) = fixture_with_frames(2, 2);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(5),
    };
    let fixed = Model::new(
        serde_json::from_value(seed.clone()).unwrap(),
        &stage,
        &pool,
        true,
        7,
        &control,
    )
    .unwrap();
    let free = Model::new(
        serde_json::from_value(seed).unwrap(),
        &stage,
        &pool,
        true,
        N,
        &control,
    )
    .unwrap();
    let x = vec![0.05; 7];
    let mut expanded = vec![0.0; N];
    expanded[..7].copy_from_slice(&x);
    let compare = |a: &[f64], b: &[f64]| {
        let count = stage.output_len();
        assert_eq!(&a[..count], &b[..count]);
        for frame in 0..stage.frame_count() {
            assert_eq!(
                &a[count + frame * 7..count + (frame + 1) * 7],
                &b[count + frame * (N + 1)..count + frame * (N + 1) + 7]
            );
        }
    };
    compare(&fixed.values(&x).unwrap(), &free.values(&expanded).unwrap());
    let columns = fixed.columns(&x).unwrap();
    assert_eq!(columns.len(), 7);
    for (a, b) in columns.iter().zip(free.columns(&expanded).unwrap()) {
        compare(a, &b);
    }
}

#[test]
fn unsupported_parameter_counts_are_rejected_before_optimization() {
    let (seed, stage) = fixture(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for count in [0, 1, 6, 8, 11, 13, 100] {
        let mut input = request(seed.clone(), "lm");
        input["x"] = json!(vec![0.0; count]);
        assert_eq!(
            solve(&input, &stage, &pool, &AtomicBool::new(false)).unwrap_err(),
            "invalid parameter count"
        );
    }
}

#[test]
fn duplicating_observations_preserves_normalized_objective() {
    let (seed, one) = fixture_with_frames(100, 1);
    let (_, two) = fixture_with_frames(100, 2);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(10),
    };
    let first = Model::new(
        serde_json::from_value(seed.clone()).unwrap(),
        &one,
        &pool,
        false,
        N,
        &control,
    )
    .unwrap();
    let duplicate = Model::new(
        serde_json::from_value(seed).unwrap(),
        &two,
        &pool,
        false,
        N,
        &control,
    )
    .unwrap();
    let x = vec![0.5; N];
    let soft_cost = |model: &Model<'_>| {
        model
            .values(&x)
            .unwrap()
            .iter()
            .map(|r| 0.09 * ((1.0 + (r / 0.3).powi(2)).sqrt() - 1.0))
            .sum::<f64>()
            / model.stage.output_len() as f64
    };
    let (a, b) = (soft_cost(&first), soft_cost(&duplicate));
    assert!(
        (a - b).abs() < 1e-12,
        "duplicate image changes normalized cost: {a} versus {b}"
    );
    assert!(
        (first.constrained_cost(&x, None).unwrap() - duplicate.constrained_cost(&x, None).unwrap())
            .abs()
            < 1e-12
    );
}

#[test]
fn duplicated_pairs_preserve_residual_and_jacobian_blocks() {
    let (mut seed, one) = fixture_with_frames(2, 1);
    let (_, repeated) = fixture_with_frames(2, 3);
    // Keep the second lens outside its sensor domain so the penalty is active.
    seed["lenses"][1]["fx"] = json!(120.0);
    seed["lenses"][1]["fy"] = json!(120.0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(10),
    };
    for sensor in [false, true] {
        for active in [false, true] {
            let mut first = Model::new(
                serde_json::from_value(seed.clone()).unwrap(),
                &one,
                &pool,
                sensor,
                N,
                &control,
            )
            .unwrap();
            let mut duplicate = Model::new(
                serde_json::from_value(seed.clone()).unwrap(),
                &repeated,
                &pool,
                sensor,
                N,
                &control,
            )
            .unwrap();
            first.active_penalty_derivative = active;
            duplicate.active_penalty_derivative = active;
            let x = vec![0.05; N];
            assert!(first.margin(&x).unwrap() < 0.0);
            let compare = |a: &[f64], b: &[f64]| {
                let count = one.output_len();
                let regularizers = N + usize::from(sensor);
                assert_eq!(a.len(), count + regularizers);
                assert_eq!(b.len(), (count + regularizers) * 3);
                for frame in 0..3 {
                    assert_eq!(&a[..count], &b[frame * count..(frame + 1) * count]);
                    let begin = repeated.output_len() + frame * regularizers;
                    assert_eq!(&a[count..], &b[begin..begin + regularizers]);
                }
            };
            compare(&first.values(&x).unwrap(), &duplicate.values(&x).unwrap());
            for (a, b) in first
                .columns(&x)
                .unwrap()
                .iter()
                .zip(duplicate.columns(&x).unwrap())
            {
                compare(a, &b);
            }
            if !sensor {
                let mut a = vec![0.0; N];
                let mut b = vec![0.0; N];
                first.constrained_cost(&x, Some(&mut a)).unwrap();
                duplicate.constrained_cost(&x, Some(&mut b)).unwrap();
                assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-10));
            }
        }
    }
}

#[test]
fn cancellation_is_checked_inside_an_image_evaluation() {
    let (seed, stage) = fixture(100);
    let cal = serde_json::from_value(seed).unwrap();
    let calls = AtomicUsize::new(0);
    let mut residual = Vec::new();
    let result = stage.residual_checked(&cal, &mut residual, || {
        if calls.fetch_add(1, Ordering::Relaxed) == 5 {
            Err("cancelled within patch loop".into())
        } else {
            Ok(())
        }
    });
    assert_eq!(result.unwrap_err(), "cancelled within patch loop");
    assert_eq!(calls.load(Ordering::Relaxed), 6);
    assert_eq!(stage.residual(&cal, &mut residual).unwrap(), 0);
}

#[test]
fn cancelled_parallel_derivatives_return_no_partial_jacobian() {
    let (seed, stage) = fixture(1800);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(5),
    };
    let model = Model::new(
        serde_json::from_value(seed).unwrap(),
        &stage,
        &pool,
        true,
        N,
        &control,
    )
    .unwrap();
    std::thread::scope(|scope| {
        let job = scope.spawn(|| model.columns(&[0.0; N]));
        while model.residuals.load(Ordering::Relaxed) == 0 && Instant::now() < control.deadline {
            std::thread::yield_now();
        }
        assert!(model.residuals.load(Ordering::Relaxed) > 0);
        interrupted.store(true, Ordering::Relaxed);
        let started = Instant::now();
        assert_eq!(job.join().unwrap().unwrap_err(), "calibration cancelled");
        eprintln!(
            "parallel derivative cancellation returned in {:?}",
            started.elapsed()
        );
    });
}

#[test]
fn all_methods_reject_precancelled_work_and_expired_image_evaluations() {
    let (seed, stage) = fixture(1800);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for method in ["lm", "trf", "slsqp"] {
        for count in [7, N] {
            let mut request = request(seed.clone(), method);
            request["x"] = json!(vec![0.0; count]);
            assert_eq!(
                solve(&request, &stage, &pool, &AtomicBool::new(true)).unwrap_err(),
                "calibration cancelled"
            );
            request["time_limit_ms"] = json!(1);
            let started = Instant::now();
            assert_eq!(
                solve(&request, &stage, &pool, &AtomicBool::new(false)).unwrap_err(),
                "calibration time limit reached"
            );
            eprintln!("{method} 1ms timeout returned in {:?}", started.elapsed());
        }
    }
}

#[test]
fn invalid_deadlines_do_not_start_a_solve() {
    let (seed, stage) = fixture(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for value in [
        json!(0),
        json!(30_001),
        json!(-1),
        json!(null),
        json!("short"),
    ] {
        let mut request = request(seed.clone(), "lm");
        request["time_limit_ms"] = value;
        assert_eq!(
            solve(&request, &stage, &pool, &AtomicBool::new(false)).unwrap_err(),
            "invalid time limit"
        );
    }
}

#[test]
fn sensor_constraint_cannot_ignore_an_invalid_second_lens() {
    let (mut seed, stage) = fixture(1);
    seed["lenses"][1]["fx"] = json!(120.0);
    seed["lenses"][1]["fy"] = json!(120.0);
    let cal: Calibration = serde_json::from_value(seed.clone()).unwrap();
    assert!(reco_core::lens::sensor_domain_margin(&cal.lenses[0]) > 0.0);
    let second = reco_core::lens::sensor_domain_margin(&cal.lenses[1]);
    assert!(second < 0.0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let interrupted = AtomicBool::new(false);
    let control = Control {
        interrupted: &interrupted,
        deadline: Instant::now() + Duration::from_secs(5),
    };
    let model = Model::new(
        serde_json::from_value(seed).unwrap(),
        &stage,
        &pool,
        true,
        N,
        &control,
    )
    .unwrap();
    assert_eq!(model.margin(&[0.0; N]).unwrap(), second);
    let residual = model.values(&[0.0; N]).unwrap();
    assert_eq!(
        *residual.last().unwrap(),
        -second * 50.0 * (stage.output_len() as f64).sqrt()
    );
    let mut swapped = cal.clone();
    swapped.lenses.swap(0, 1);
    assert_eq!(sensor_margin(&swapped).unwrap(), second);
}

#[test]
fn hinge_derivative_uses_the_current_branch() {
    assert_eq!(penalty_derivative(-1e-9, 2.0), -2.0);
    assert_eq!(penalty_derivative(1e-9, 2.0), 0.0);
    assert_eq!(penalty_derivative(0.0, 2.0), 0.0);
    assert_eq!(penalty_derivative(-1e-9, -2.0), 2.0);
}

#[test]
fn difference_step_is_bounded_and_explicit() {
    assert_eq!(difference_step(&json!({})).unwrap(), 0.002);
    for value in [json!(null), json!("small"), json!(0.0), json!(0.1)] {
        assert!(difference_step(&json!({"difference_step":value})).is_err());
    }
    assert_eq!(
        difference_step(&json!({"difference_step":0.00002})).unwrap(),
        0.00002
    );
}

#[test]
fn typed_parameters_preserve_unfitted_document_fields_and_lens_bits() {
    let (seed, _) = fixture(1);
    let mut seed: Calibration = serde_json::from_value(seed).unwrap();
    seed.sync_offset = -17;
    seed.framing.tilt = 0.23;
    seed.framing.roll = -0.14;
    seed.lenses[1].fx = 394.0123456789012;
    seed.lenses[1].distortion[3] = -0.0009822150067739363;
    seed.projection.l_shape_mut().unwrap().blend_width = 0.17;
    let before = seed.clone();
    for count in [7, N] {
        assert_eq!(decode_parameters(&seed, &vec![0.0; count]).unwrap(), seed);
    }
    let fitted = decode_parameters(&seed, &[0.1; 7]).unwrap();
    assert_eq!(seed, before);
    assert_eq!(fitted.lenses, seed.lenses);
    assert_eq!(fitted.sync_offset, seed.sync_offset);
    assert_eq!(fitted.framing.tilt, seed.framing.tilt);
    assert_eq!(fitted.framing.roll, seed.framing.roll);
    assert_eq!(fitted.projection.l_shape().unwrap().blend_width, 0.17);
    assert_eq!(
        fitted.lenses[1].distortion[3].to_bits(),
        seed.lenses[1].distortion[3].to_bits()
    );
    let mut invalid = vec![0.0; N];
    invalid[3] = f64::NAN;
    assert!(decode_parameters(&seed, &invalid).is_err());
}

#[test]
fn typed_solver_shares_cancellation_and_does_not_renew_deadline() {
    let (seed, stage) = fixture(1);
    let seed: Calibration = serde_json::from_value(seed).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let interrupted = AtomicBool::new(false);
    let mut control = Control {
        interrupted: &interrupted,
        deadline: Instant::now(),
    };
    let options = SolveOptions {
        method: Method::LeastSquares,
        budget: 75,
        tolerance: 1e-5,
        difference_step: 0.00002,
        active_penalty: false,
        optics: [true; 2],
    };
    let run = |control: &Control<'_>| {
        solve_native(seed.clone(), vec![0.0; 7], options, &stage, &pool, control)
    };
    assert_eq!(run(&control).unwrap_err(), "calibration time limit reached");
    control.deadline = Instant::now() + Duration::from_secs(1);
    let deadline = control.deadline;
    let first = control.remaining().unwrap();
    let second = control.remaining().unwrap();
    assert!(second <= first);
    assert_eq!(control.deadline, deadline);
    interrupted.store(true, Ordering::Relaxed);
    assert_eq!(run(&control).unwrap_err(), "calibration cancelled");
    assert_eq!(control.deadline, deadline);
}

#[test]
fn each_camera_has_independent_optical_parameters() {
    let (seed, _) = fixture(1);
    let seed: Calibration = serde_json::from_value(seed).unwrap();
    for camera in 0..2 {
        for parameter in 0..5 {
            let mut x = vec![0.0; N];
            x[7 + camera * 5 + parameter] = 0.2;
            let fitted = decode_parameters(&seed, &x).unwrap();
            assert_eq!(fitted.lenses[1 - camera], seed.lenses[1 - camera]);
            assert_ne!(fitted.lenses[camera], seed.lenses[camera]);
            assert_eq!(fitted.projection, seed.projection);
            assert_eq!(fitted.framing, seed.framing);
        }
    }
}
