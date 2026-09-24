//! External-fixture construction-cost and approximation measurements.

use super::*;

#[test]
#[ignore = "measures coverage-map construction against an external calibration"]
fn measure_roof_construction() {
    let calibration: Calibration = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RECO_COVERAGE_CALIBRATION").unwrap()).unwrap(),
    )
    .unwrap();
    let step: f32 = std::env::var("RECO_PAN_STEP").unwrap().parse().unwrap();
    let tolerance: f32 = std::env::var("RECO_FOV_TOLERANCE")
        .unwrap()
        .parse()
        .unwrap();
    let workers: usize = std::env::var("RECO_BUILD_WORKERS").map_or(1, |v| v.parse().unwrap());
    let domain: [f32; 4] =
        serde_json::from_str(&std::env::var("RECO_PAN_DOMAIN").unwrap()).unwrap();
    let size: [u32; 2] =
        serde_json::from_str(&std::env::var("RECO_VIEWPORT_SIZE").unwrap()).unwrap();
    let size = ViewportSize {
        width: size[0],
        height: size[1],
    };
    assert!(step.is_finite() && step > 0.0);
    let grid = Grid {
        yaw: [domain[0], domain[1]],
        pitch: [domain[2], domain[3]],
        columns: ((domain[1] - domain[0]) / step).ceil() as usize,
        rows: ((domain[3] - domain[2]) / step).ceil() as usize,
    };
    // Permit a finer diagnostic grid without changing the normal builder's limit.
    assert!(
        Grid {
            columns: 1,
            rows: 1,
            ..grid
        }
        .valid()
    );
    assert!(grid.columns > 0 && grid.rows > 0);
    assert!(
        grid.columns
            .checked_mul(grid.rows)
            .is_some_and(|n| n <= 1_000_000)
    );
    assert!(calibration.validate().is_ok() && size.validate().is_ok());
    assert!(tolerance.is_finite() && tolerance > 0.0);
    assert!((1..=16).contains(&workers));
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..3 {
        let started = std::time::Instant::now();
        let roof = Roof::build_cells(&calibration, &size, grid, [20.0, 120.0], tolerance, workers);
        times.push(started.elapsed().as_secs_f64());
        if let Some(previous) = &last {
            let previous: &Roof = previous;
            assert_eq!(previous.heights, roof.heights);
        }
        eprintln!("build {}: {:.6}s", times.len(), times.last().unwrap());
        last = Some(roof);
    }
    let roof = last.unwrap();
    let started = std::time::Instant::now();
    let cap = roof.connected_cap(Some([-0.8, 0.8]), 1e-6);
    let cap_seconds = started.elapsed().as_secs_f64();
    // Identical coordinates across grid settings, independent of their vertices.
    let ceilings: Vec<_> = (0..=80)
        .flat_map(|x| {
            let roof = &roof;
            (0..=40).map(move |y| {
                let pose = Pose {
                    yaw: -1.2 + x as f32 * 0.03,
                    pitch: -0.6 + y as f32 * 0.025,
                    fov_degrees: 20.0,
                };
                roof.ceiling(pose).unwrap()
            })
        })
        .collect();
    let output = serde_json::json!({
        "seconds": times, "cap": cap, "cap_seconds": cap_seconds,
        "step": step, "tolerance": tolerance, "workers": workers,
        "size": [size.width, size.height], "domain": domain,
        "columns": grid.columns, "rows": grid.rows,
        "bytes": roof.heights.len() * 4,
        "heights": roof.heights, "ceilings": ceilings,
    });
    serde_json::to_writer(
        std::fs::File::create_new(std::env::var_os("RECO_ROOF_RESULT").unwrap()).unwrap(),
        &output,
    )
    .unwrap();
    eprintln!("connected cap {cap:?}");
}

#[test]
fn parallel_diagnostic_matches_serial_map_exactly() {
    let cal = super::super::stereo();
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let grid = Grid {
        yaw: [-0.6, 0.6],
        pitch: [-0.4, 0.4],
        columns: 8,
        rows: 8,
    };
    let serial = Roof::build_cells(&cal, &size, grid, [20.0, 120.0], 0.05, 1);
    let automatic = Roof::build(&cal, &size, grid, [20.0, 120.0], 0.05).unwrap();
    assert_eq!(serial.heights, automatic.heights);
    for workers in [2, 4] {
        let parallel = Roof::build_cells(&cal, &size, grid, [20.0, 120.0], 0.05, workers);
        assert_eq!(serial.heights, parallel.heights);
    }
}

#[test]
#[ignore = "checks a saved outline map against dense source samples"]
fn validate_saved_outline_map() {
    let cal: Calibration = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RECO_COVERAGE_CALIBRATION").unwrap()).unwrap(),
    )
    .unwrap();
    let data: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RECO_ROOF_RESULT").unwrap()).unwrap(),
    )
    .unwrap();
    let domain: [f32; 4] = serde_json::from_value(data["domain"].clone()).unwrap();
    let dimensions: [u32; 2] = serde_json::from_value(data["size"].clone()).unwrap();
    let size = ViewportSize {
        width: dimensions[0],
        height: dimensions[1],
    };
    let roof = Roof {
        grid: Grid {
            yaw: [domain[0], domain[1]],
            pitch: [domain[2], domain[3]],
            columns: data["columns"].as_u64().unwrap() as usize,
            rows: data["rows"].as_u64().unwrap() as usize,
        },
        minimum_fov: 20.0,
        heights: serde_json::from_value(data["heights"].clone()).unwrap(),
    };
    let mut checked = 0;
    let mut continuous_unknown = 0;
    for i in 0..4000 {
        let p = Pose {
            yaw: domain[0] + (domain[1] - domain[0]) * ((i * 73 % 4001) as f32 / 4001.0),
            pitch: domain[2] + (domain[3] - domain[2]) * ((i * 167 % 4003) as f32 / 4003.0),
            fov_degrees: 20.0,
        };
        let ceiling = roof.ceiling(p).unwrap() as f32 - 0.001;
        if ceiling < 20.0 {
            continue;
        }
        for fov in [20.0, (20.0 + ceiling) * 0.5, ceiling] {
            let p = Pose {
                fov_degrees: fov,
                ..p
            };
            assert!(roof.contains(p));
            match boundary::pose(&cal, &size, p) {
                Certificate::Uncovered => panic!("unsupported outline at {p:?}"),
                Certificate::Unknown => continuous_unknown += 1,
                Certificate::Covered => {}
            }
            let maps = cal.surface_maps(&size, p);
            // Resolution-independent NDC probes, including the exact perimeter.
            for y in 0..=36 {
                for x in 0..=64 {
                    let q = [2.0 * x as f64 / 64.0 - 1.0, 2.0 * y as f64 / 36.0 - 1.0];
                    assert!(
                        maps.iter().any(|(s, _)| s.sample_uv(q).is_some()),
                        "{p:?}, {q:?}"
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 1000);
    eprintln!("validated {checked} poses, {continuous_unknown} inconclusive continuous perimeters");
}
