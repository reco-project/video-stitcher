//! External-fixture map-motion replay.

use super::*;
use std::io::Write;

#[test]
#[ignore = "replays external requests without rendering, recording requested and achieved FOV"]
fn replay_motion_schedule() {
    let cal: Calibration = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RECO_COVERAGE_CALIBRATION").unwrap()).unwrap(),
    )
    .unwrap();
    let requests: Vec<Pose> = if let Some(path) = std::env::var_os("RECO_COVERAGE_TRAJECTORY") {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    } else {
        let camera = cal.topology.projection().virtual_camera(&cal.framing);
        std::fs::read_to_string(std::env::var_os("RECO_COVERAGE_EVENTS").unwrap())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|event| event["kind"] == "pan_decision")
            .map(|event| {
                let pose: Pose = serde_json::from_value(event["pose"].clone()).unwrap();
                aim(
                    &cal,
                    &camera.yaw_pitch_to_direction(pose.yaw, pose.pitch),
                    pose.fov_degrees,
                )
            })
            .collect()
    };
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    let mut cap =
        std::env::var("RECO_FOV_CAP").map_or(179.0, |value| value.parse::<f32>().unwrap());
    assert!((1.0..=179.0).contains(&cap));
    let roof = {
        let step = std::env::var("RECO_ROOF_STEP").unwrap();
        let step: f32 = step.parse().unwrap();
        assert!(step.is_finite() && step > 0.0);
        let domain: [f32; 4] =
            serde_json::from_str(&std::env::var("RECO_PAN_DOMAIN").unwrap()).unwrap();
        let started = std::time::Instant::now();
        let roof = roof::Roof::build(
            &cal,
            &size,
            roof::Grid {
                yaw: [domain[0], domain[1]],
                pitch: [domain[2], domain[3]],
                columns: ((domain[1] - domain[0]) / step).ceil() as usize,
                rows: ((domain[3] - domain[2]) / step).ceil() as usize,
            },
            [20.0, 120.0],
            0.05,
        )
        .unwrap();
        eprintln!("replay roof built in {:?}", started.elapsed());
        if let Ok(span) = std::env::var("RECO_CONNECTED_YAW") {
            let span: [f32; 2] = serde_json::from_str(&span).unwrap();
            let started = std::time::Instant::now();
            let connected = roof
                .connected_cap(Some(span), CONTACT_TOLERANCE)
                .expect("no connected FOV interval for pan span");
            cap = cap.min(connected);
            eprintln!(
                "connected cap {cap:.9} deg computed in {:?}",
                started.elapsed()
            );
        }
        roof
    };
    let prepared = std::time::Instant::now();
    let mut route = Route::new(&cal, &roof);
    eprintln!(
        "route views prepared in {:?}, {} bytes",
        prepared.elapsed(),
        roof.vertices().count() * std::mem::size_of::<(Matrix4<f32>, Vector3<f64>)>()
    );
    let first_view = std::time::Instant::now();
    let mut pose = initial(
        &cal,
        &size,
        &direction(&cal, requests[0]),
        requests[0].fov_degrees.min(cap),
        &roof,
        &route,
    )
    .expect("no supported starting view at the admitted FOV");
    eprintln!("initial view selected in {:?}", first_view.elapsed());
    let mut velocity = [0.0; 2];
    let mut first_frame = 0;
    if let Some(path) = std::env::var_os("RECO_MOTION_SEED") {
        let seed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        pose = serde_json::from_value(seed["pose"].clone()).unwrap();
        velocity = serde_json::from_value(seed["velocity"].clone()).unwrap();
        first_frame = seed["frame"].as_u64().unwrap() as usize + 1;
    }
    assert!(roof.contains(pose), "initial pose outside roof");
    assert!(
        std::env::var_os("RECO_CONNECTED_YAW").is_none() || pose.fov_degrees <= cap,
        "connected cap requires a map and an admissible starting FOV"
    );
    let limit = std::env::var("RECO_MOTION_LIMIT").map_or(requests.len(), |v| v.parse().unwrap());
    let mut output =
        std::fs::File::create_new(std::env::var_os("RECO_BOUNDARY_PROBE").unwrap()).unwrap();
    for (frame, requested) in requests.iter().enumerate().skip(first_frame).take(limit) {
        let started = std::time::Instant::now();
        let next = solve_motion(
            &cal,
            &size,
            MotionWindow {
                previous: pose,
                velocity,
                dt: 1.0 / 30.0,
            },
            &direction(&cal, *requested),
            requested.fov_degrees.min(cap),
            &roof,
            &mut route,
        );
        let Some(next) = next else {
            writeln!(output, "{}", serde_json::json!({ "frame": frame, "rejected": true, "previous": pose, "velocity": velocity, "target": requested })).unwrap();
            panic!("rejected at {frame}: {pose:?} -> {requested:?}, velocity {velocity:?}");
        };
        let acceleration =
            (next.velocity[0] - velocity[0]).hypot(next.velocity[1] - velocity[1]) * 30.0;
        assert!(acceleration <= ACCELERATION + 0.001);
        assert!(distance(pose, next.pose) <= RATE / 30.0 + 1e-6);
        assert!((next.pose.fov_degrees - pose.fov_degrees).abs() <= ZOOM_RATE / 30.0 + 1e-5);
        let xy = screen(&cal, &size, next.pose, &direction(&cal, *requested));
        writeln!(output, "{}", serde_json::json!({ "frame": frame, "requested_fov": requested.fov_degrees, "pose": next.pose, "velocity": next.velocity, "acceleration": acceleration, "queries": next.support_queries, "objective": next.objective, "visible": xy.is_some_and(|p| p.iter().all(|v| v.abs()<=1.0)), "coverage_narrowing": next.pose.fov_degrees < pose.fov_degrees && requested.fov_degrees >= pose.fov_degrees, "elapsed_ms": started.elapsed().as_secs_f64()*1000.0 })).unwrap();
        pose = next.pose;
        velocity = next.velocity;
        if frame % 100 == 0 {
            eprintln!("replayed {frame}/{}", requests.len());
        }
    }
}
