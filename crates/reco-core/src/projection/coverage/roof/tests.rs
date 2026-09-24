use super::*;

#[test]
fn periodic_direct_input_has_bounded_travel_for_large_finite_angles() {
    let grid = Grid {
        yaw: [-std::f32::consts::PI, std::f32::consts::PI],
        pitch: [-0.5, 0.5],
        columns: 16,
        rows: 2,
    };
    let roof = Roof::from_cells(grid, 1.0, &[80.0; 32]);
    for yaw in [1e6, -1e6, 1e12, -1e12, 1e30, -1e30, f32::MAX, f32::MIN] {
        for reference in [0.0, 3.0, -3.0, 20.0, -20.0, yaw] {
            let from = RenderPose {
                yaw: reference,
                pitch: 0.0,
                fov_degrees: 10.0,
            };
            let request = RenderPose { yaw, ..from };
            let near = grid.near_yaw(request, reference);
            assert!(
                (f64::from(near.yaw) - f64::from(reference)).abs() <= std::f64::consts::PI + 1e-6
            );
            let accepted = roof.constrain(from, request);
            assert_eq!(accepted, near);
            assert!(roof.contains(accepted));
        }
    }
    for sign in [-1.0, 1.0] {
        let from = RenderPose {
            yaw: sign * 3.13,
            pitch: 0.0,
            fov_degrees: 10.0,
        };
        let accepted = roof.constrain(
            from,
            RenderPose {
                yaw: -from.yaw,
                ..from
            },
        );
        assert!(sign * (accepted.yaw - from.yaw) > 0.0);
        assert!((accepted.yaw - from.yaw).abs() < 0.03);
    }
}

fn grid(n: usize) -> Grid {
    Grid {
        yaw: [-0.6, 0.6],
        pitch: [-0.4, 0.4],
        columns: n,
        rows: n,
    }
}

fn cylinder() -> Calibration {
    Calibration::new(
        vec![Lens::flat(3840, 1000)],
        Cylinder {
            radius: 1000.0,
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.2,
            roll: -0.15,
        },
    )
}

#[test]
fn diagonal_ridges_remain_connected_in_both_orientations() {
    for (heights, endpoints) in [
        (vec![20.0, 60.0, 60.0, 20.0], [(1, 0), (0, 1)]),
        (vec![60.0, 20.0, 20.0, 60.0], [(0, 0), (1, 1)]),
    ] {
        let roof = Roof {
            grid: grid(1),
            minimum_fov: 10.0,
            heights,
        };
        let cap = roof.connected_cap(None, 1e-6).unwrap();
        assert!(
            cap > 59.99,
            "a triangulation must not introduce a bottleneck: {cap}"
        );
        let [a, b] = endpoints.map(|(x, y)| roof.grid.pose(x, y, cap));
        assert!(roof.segment(a, b) && roof.segment(b, a));
        assert_eq!(roof.constrain(a, b), b);
        assert_eq!(roof.constrain(b, a), a);
        for i in 0..roof.heights.len() {
            for (j, _) in roof.neighbors(i) {
                assert!(roof.neighbors(j).any(|(k, _)| k == i));
                let a = roof.grid.pose(i % 2, i / 2, 10.0);
                let b = roof.grid.pose(j % 2, j / 2, 10.0);
                let midpoint = interpolate(a, b, 0.5);
                let expected = f64::from(roof.heights[i] + roof.heights[j]) * 0.5;
                assert!((roof.ceiling(midpoint).unwrap() - expected).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn reflecting_a_map_does_not_change_its_connected_cap() {
    let grid = grid(8);
    for case in 0..100 {
        let cells: Vec<_> = (0..64)
            .map(|i| 20.0 + ((i * 37 + case * 13) % 71) as f32)
            .collect();
        let reflected: Vec<_> = (0..64).map(|i| cells[i / 8 * 8 + 7 - i % 8]).collect();
        let a = Roof::from_cells(grid, 10.0, &cells);
        let b = Roof::from_cells(grid, 10.0, &reflected);
        assert_eq!(
            a.connected_cap(None, 0.0),
            b.connected_cap(None, 0.0),
            "case {case}"
        );
    }
}

#[test]
fn interpolation_is_continuous_and_never_exceeds_a_cell_certificate() {
    let grid = grid(3);
    let cells = [40.0, 70.0, 50.0, 60.0, 0.0, 90.0, 55.0, 65.0, 80.0];
    let roof = Roof::from_cells(grid, 10.0, &cells);
    for y in 0..3 {
        for x in 0..3 {
            for i in 0..=20 {
                for j in 0..=20 {
                    let p = [x as f64 + i as f64 / 20.0, y as f64 + j as f64 / 20.0];
                    assert!(roof.height_at(p) <= f64::from(cells[y * 3 + x]) + 1e-10);
                }
            }
        }
    }
    for i in 0..=300 {
        let q = i as f64 / 100.0;
        for k in [1.0, 2.0] {
            assert!((roof.height_at([k - 1e-8, q]) - roof.height_at([k + 1e-8, q])).abs() < 1e-5);
            assert!((roof.height_at([q, k - 1e-8]) - roof.height_at([q, k + 1e-8])).abs() < 1e-5);
        }
    }
}

#[test]
fn segment_cannot_jump_a_valley_between_covered_endpoints() {
    let grid = Grid {
        columns: 5,
        rows: 1,
        ..grid(1)
    };
    let roof = Roof::from_cells(grid, 10.0, &[60.0, 60.0, 20.0, 60.0, 60.0]);
    let a = grid.pose(0, 0, 40.0);
    let b = grid.pose(5, 1, 40.0);
    assert!(roof.contains(a) && roof.contains(b));
    assert!(!roof.segment(a, b));
    assert!(roof.segment(
        RenderPose {
            fov_degrees: 19.0,
            ..a
        },
        RenderPose {
            fov_degrees: 19.0,
            ..b
        }
    ));
    assert!(!roof.contains(RenderPose {
        fov_degrees: 9.0,
        ..a
    }));
    assert!(!roof.contains(RenderPose { yaw: f32::NAN, ..a }));
    assert!(!roof.contains(RenderPose { yaw: 1.0, ..a }));
    assert!(!roof.contains(RenderPose {
        fov_degrees: f32::NAN,
        ..a
    }));
}

#[test]
fn triangle_crossings_match_dense_segment_checks_in_both_directions() {
    let grid = grid(8);
    let cells: Vec<_> = (0..64).map(|i| 20.0 + (i * 37 % 70) as f32).collect();
    let roof = Roof::from_cells(grid, 10.0, &cells);
    for i in 0..1000 {
        let a = RenderPose {
            yaw: -0.6 + (i * 73 % 1200) as f32 * 0.001,
            pitch: -0.4 + (i * 31 % 800) as f32 * 0.001,
            fov_degrees: 10.0 + (i % 40) as f32,
        };
        let b = RenderPose {
            yaw: -0.6 + (i * 59 % 1200) as f32 * 0.001,
            pitch: -0.4 + (i * 67 % 800) as f32 * 0.001,
            fov_degrees: 10.0 + (i * 7 % 40) as f32,
        };
        assert_eq!(roof.segment(a, b), roof.segment(b, a));
        if roof.segment(a, b) {
            for j in 0..=1000 {
                assert!(roof.contains(interpolate(a, b, j as f32 / 1000.0)));
            }
        }
    }
}

#[test]
fn tilted_cylinder_roof_agrees_with_independent_source_samples() {
    let cal = cylinder();
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    let roof = Roof::build(&cal, &size, grid(16), [10.0, 80.0], 0.05).unwrap();
    let mut checked = 0;
    for i in 0..1000 {
        let p = RenderPose {
            yaw: -0.6 + (i * 73 % 1200) as f32 * 0.001,
            pitch: -0.4 + (i * 31 % 800) as f32 * 0.001,
            fov_degrees: 10.0,
        };
        let ceiling = roof.ceiling(p).unwrap() as f32 - 0.001;
        if ceiling < 10.0 {
            continue;
        }
        for fov in [10.0, (10.0 + ceiling) * 0.5, ceiling] {
            let p = RenderPose {
                fov_degrees: fov,
                ..p
            };
            assert!(roof.contains(p));
            let maps = cal.surface_maps(&size, p);
            assert_eq!(exhaustive(&maps, &size), 0);
            checked += 1;
        }
    }
    assert!(checked > 500, "{checked}");
}

#[test]
fn invalid_builds_are_rejected() {
    let cal = cylinder();
    let size = ViewportSize {
        width: 97,
        height: 55,
    };
    for bad in [
        Grid {
            columns: 0,
            ..grid(1)
        },
        Grid {
            rows: usize::MAX,
            ..grid(2)
        },
        Grid {
            yaw: [f32::NAN, 1.0],
            ..grid(1)
        },
    ] {
        assert!(Roof::build(&cal, &size, bad, [10.0, 80.0], 0.05).is_none());
    }
    assert!(Roof::build(&cal, &size, grid(1), [10.0, 80.0], 0.0).is_none());
}

#[test]
fn connected_cap_includes_side_branches_and_every_intermediate_fov() {
    let grid = Grid {
        columns: 5,
        rows: 1,
        ..grid(1)
    };
    let roof = Roof::from_cells(grid, 10.0, &[110.0, 110.0, 50.0, 60.0, 60.0]);
    // The required pan span lies in the large room. The smaller room can be
    // entered at narrow FOV, so its return bottleneck still limits the cap.
    let cap = roof.connected_cap(Some([-0.6, -0.4]), 1e-6).unwrap();
    assert!(cap < 50.0 && cap > 49.999);
    let a = grid.pose(0, 0, cap);
    let b = grid.pose(5, 1, cap);
    assert!(roof.segment(a, b));
    let middle = RenderPose {
        yaw: 0.0,
        pitch: 0.0,
        fov_degrees: 51.0,
    };
    assert!(roof.contains(RenderPose {
        fov_degrees: 51.0,
        ..a
    }));
    assert!(roof.contains(RenderPose {
        fov_degrees: 51.0,
        ..b
    }));
    assert!(!roof.contains(middle));
    // Checking connectivity only at 70 degrees would miss the split at 51:
    // the smaller room has disappeared from the valid set by then.
    assert!(roof.contains(RenderPose {
        fov_degrees: 70.0,
        ..a
    }));
    assert!(!roof.contains(RenderPose {
        fov_degrees: 70.0,
        ..b
    }));
}

#[test]
fn connected_cap_preserves_the_requested_pan_span() {
    let grid = Grid {
        columns: 3,
        rows: 1,
        ..grid(1)
    };
    let roof = Roof::from_cells(grid, 10.0, &[80.0, 60.0, 40.0]);
    let wide = roof.connected_cap(Some(grid.yaw), 1e-6).unwrap();
    let narrow = roof.connected_cap(Some([-0.6, -0.3]), 1e-6).unwrap();
    assert!(wide < 40.0 && wide > 39.999);
    assert!(narrow < 60.0 && narrow > 59.999);
    assert!(roof.connected_cap(Some([-0.7, 0.6]), 1e-6).is_none());
    assert!(roof.connected_cap(Some([f32::NAN, 0.6]), 1e-6).is_none());
    assert!(roof.connected_cap(Some([0.5, -0.5]), 1e-6).is_none());
    assert!(roof.connected_cap(Some(grid.yaw), f32::NAN).is_none());
    assert!(roof.connected_cap(Some(grid.yaw), -1.0).is_none());
}

#[test]
fn disconnected_or_empty_minimum_fov_domains_have_no_connected_cap() {
    let grid = Grid {
        columns: 5,
        rows: 1,
        ..grid(1)
    };
    for cells in [[60.0, 60.0, 0.0, 60.0, 60.0], [0.0; 5], [10.0; 5]] {
        let roof = Roof::from_cells(grid, 10.0, &cells);
        assert!(roof.connected_cap(Some(grid.yaw), 1e-6).is_none());
    }
}

#[test]
fn all_admitted_vertices_remain_connected_below_the_cap() {
    let grid = Grid {
        yaw: [-1.0, 1.0],
        pitch: [-0.5, 0.5],
        columns: 4,
        rows: 4,
    };
    let stride = grid.columns + 1;
    for case in 0..32 {
        let cells: Vec<_> = (0..16)
            .map(|i| 20.0 + ((i * 53 + case * 17) % 81) as f32)
            .collect();
        let roof = Roof::from_cells(grid, 10.0, &cells);
        let cap = roof.connected_cap(Some([-0.5, 0.5]), 1e-6).unwrap();
        for step in 0..=32 {
            let fov = 10.0 + (cap - 10.0) * step as f32 / 32.0;
            let poses: Vec<_> = (0..roof.heights.len())
                .map(|i| grid.pose(i % stride, i / stride, fov))
                .collect();
            let supported: Vec<_> = poses.iter().map(|p| roof.contains(*p)).collect();
            let root = supported.iter().position(|p| *p).unwrap();
            let mut reached = vec![false; poses.len()];
            let mut pending = vec![root];
            reached[root] = true;
            while let Some(i) = pending.pop() {
                let x = i % stride;
                let y = i / stride;
                for (dx, dy) in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                ] {
                    let Some(nx) = x.checked_add_signed(dx).filter(|x| *x <= grid.columns) else {
                        continue;
                    };
                    let Some(ny) = y.checked_add_signed(dy).filter(|y| *y <= grid.rows) else {
                        continue;
                    };
                    let j = ny * stride + nx;
                    if !reached[j] && supported[j] && roof.segment(poses[i], poses[j]) {
                        reached[j] = true;
                        pending.push(j);
                    }
                }
            }
            assert_eq!(reached, supported, "case {case}, FOV {fov}");
        }
    }
}

#[test]
fn cap_resolves_a_point_contact_at_the_supplied_motion_precision() {
    let roof = Roof {
        grid: Grid {
            yaw: [-1.0, 1.0],
            pitch: [-1.0, 1.0],
            columns: 2,
            rows: 2,
        },
        minimum_fov: 10.0,
        heights: vec![60.0, 20.0, 20.0, 20.0, 50.0, 20.0, 20.0, 20.0, 60.0],
    };
    let precision = 0.001;
    let exact = roof.connected_cap(Some([-0.5, 0.5]), 0.0).unwrap();
    let resolved = roof.connected_cap(Some([-0.5, 0.5]), precision).unwrap();
    assert!(exact > resolved && exact - resolved < 0.051);
    assert!(!roof.contains(RenderPose {
        yaw: precision,
        pitch: 0.0,
        fov_degrees: exact
    }));
    for i in 1..100 {
        let coordinate = -1.0 + i as f32 / 50.0;
        for j in 0..16 {
            let angle = j as f32 * std::f32::consts::TAU / 16.0;
            assert!(roof.contains(RenderPose {
                yaw: coordinate + precision * angle.cos(),
                pitch: coordinate + precision * angle.sin(),
                fov_degrees: resolved,
            }));
        }
    }
}

#[test]
#[ignore = "builds an external calibration map and compares a saved trajectory"]
fn probe_roof() {
    let cal: Calibration = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RECO_COVERAGE_CALIBRATION").unwrap()).unwrap(),
    )
    .unwrap();
    let domain: [f32; 4] =
        serde_json::from_str(&std::env::var("RECO_PAN_DOMAIN").unwrap()).unwrap();
    let step: f32 = std::env::var("RECO_PAN_STEP").unwrap().parse().unwrap();
    let grid = Grid {
        yaw: [domain[0], domain[1]],
        pitch: [domain[2], domain[3]],
        columns: ((domain[1] - domain[0]) / step).ceil() as usize,
        rows: ((domain[3] - domain[2]) / step).ceil() as usize,
    };
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let started = std::time::Instant::now();
    let roof = Roof::build(&cal, &size, grid, [20.0, 120.0], 0.05).unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let records: Vec<serde_json::Value> =
        std::fs::read_to_string(std::env::var_os("RECO_SOLVED_MOTION").unwrap())
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
    let samples: Vec<_> = records.iter().map(|r| {
        let p: RenderPose = serde_json::from_value(r["pose"].clone()).unwrap();
        serde_json::json!({"frame": r["frame"], "pose": p, "ceiling": roof.ceiling(p), "contained": roof.contains(p)})
    }).collect();
    // A graph edge connects neighbouring roof vertices, where height is affine.
    // A supported graph route therefore gives a continuous passage, not jumps.
    let passage = |fov: f32| {
        let mut reached = vec![false; roof.heights.len()];
        let mut queue = std::collections::VecDeque::new();
        let stride = grid.columns + 1;
        for y in 0..=grid.rows {
            let i = y * stride;
            if roof.heights[i] > fov {
                reached[i] = true;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            let x = i % stride;
            if x == grid.columns {
                return true;
            }
            for (j, _) in roof.neighbors(i) {
                if !reached[j] && roof.heights[j] > fov {
                    reached[j] = true;
                    queue.push_back(j);
                }
            }
        }
        false
    };
    let (mut low, mut high) = (20.0, 120.0);
    let reachable = passage(low);
    if reachable {
        for _ in 0..20 {
            let mid = (low + high) * 0.5;
            if passage(mid) {
                low = mid;
            } else {
                high = mid;
            }
        }
    }
    let output = serde_json::json!({"seconds": seconds, "domain": domain, "columns": grid.columns, "rows": grid.rows, "bytes": roof.heights.len()*4, "heights": roof.heights, "passage_fov": reachable.then_some(low), "samples": samples});
    serde_json::to_writer(
        std::fs::File::create_new(std::env::var_os("RECO_ROOF_RESULT").unwrap()).unwrap(),
        &output,
    )
    .unwrap();
    eprintln!(
        "roof {}x{} built in {seconds:.3}s, passage {:?}",
        grid.columns,
        grid.rows,
        reachable.then_some(low)
    );
}
