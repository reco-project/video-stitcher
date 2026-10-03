//! Calibrated motion artifacts with explicit renderer selection and coverage checks.

use super::*;
use crate::{
    render::{planes::Nv12Planes, renderer::InputFormat},
    stitch::{
        CpuExecutor, GpuExecutor, GpuExecutorConfig, StitchExecutor, test_support::gpu_or_skip,
    },
};
use std::{
    io::{Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
};

fn synthetic_target(frame: usize, fov: f32) -> RenderPose {
    let keys = [
        (0.0, [0.0, -0.3]),
        (2.0, [-0.8, -0.15]),
        (5.0, [-1.45, 0.1]),
        (7.0, [-1.45, -0.55]),
        (11.0, [0.0, -0.72]),
        (14.0, [1.45, -0.45]),
        (16.0, [1.45, 0.1]),
        (19.0, [0.0, -0.3]),
        (20.0, [0.0, -0.3]),
        (25.0, [0.0, -0.08]),
        (33.0, [-1.4, 0.12]),
        (41.0, [1.4, 0.12]),
        (44.0, [-1.5, -0.6]),
        (50.0, [1.5, -0.6]),
        (52.0, [-1.5, -0.6]),
        (58.0, [-1.5, -0.6]),
        (60.0, [0.0, -0.3]),
    ];
    let t = frame as f32 / 30.0;
    let pair = keys.windows(2).find(|k| t <= k[1].0).unwrap();
    let u = ((t - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0.0, 1.0);
    let u = u * u * (3.0 - 2.0 * u);
    RenderPose {
        yaw: pair[0].1[0] * (1.0 - u) + pair[1].1[0] * u,
        pitch: pair[0].1[1] * (1.0 - u) + pair[1].1[1] * u,
        fov_degrees: fov,
    }
}

fn gradient(width: usize, height: usize, side: usize) -> Vec<u8> {
    let mut frame = vec![128; width * height * 3 / 2];
    for y in 0..height {
        for x in 0..width {
            frame[y * width + x] = if x % 64 < 3 || y % 40 < 3 {
                220
            } else {
                (55 + x * 95 / width + y * 35 / height + side * 15) as u8
            };
        }
    }
    frame
}

fn paint_target(frame: &mut [u8], width: usize, height: usize, uv: crate::stitch::SurfaceUv) {
    let cx = (uv.u * width as f64) as i32;
    let cy = (uv.v * height as f64) as i32;
    for y in (cy - 7).max(0)..=(cy + 7).min(height as i32 - 1) {
        for x in (cx - 7).max(0)..=(cx + 7).min(width as i32 - 1) {
            if (x - cx).pow(2) + (y - cy).pow(2) <= 49 {
                frame[y as usize * width + x as usize] = 150;
                let chroma = width * height + y as usize / 2 * width + x as usize / 2 * 2;
                frame[chroma] = 175;
                frame[chroma + 1] = 220;
            }
        }
    }
}

fn missing(image: &[u8]) -> usize {
    image
        .chunks_exact(4)
        .filter(|p| p[..3].iter().any(|v| *v < 250))
        .count()
}

fn missing_border(image: &[u8], width: usize) -> usize {
    let height = image.len() / 4 / width;
    image
        .chunks_exact(4)
        .enumerate()
        .filter(|(i, p)| {
            let (x, y) = (i % width, i / width);
            (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
                && p[..3].iter().any(|v| *v < 250)
        })
        .count()
}

fn visible(xy: Option<[f64; 2]>) -> bool {
    xy.is_some_and(|xy| xy.iter().all(|v| v.abs() <= 1.0))
}

#[test]
#[ignore = "checks saved accepted motion at full-HD output resolution"]
fn probe_high_resolution_motion() {
    let dir = PathBuf::from(std::env::var_os("RECO_COVERAGE_ARTIFACT_DIR").unwrap());
    let mut cal: Calibration = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RECO_COVERAGE_CALIBRATION").unwrap()).unwrap(),
    )
    .unwrap();
    if let Some(shape) = cal.projection.l_shape_mut() {
        shape.blend_width = 0.0;
    }
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(dir.join("motion.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!rows.is_empty());
    let mut indices = std::collections::BTreeSet::new();
    indices.extend((0..rows.len()).step_by((rows.len() / 32).max(1)));
    indices.extend(rows.len().saturating_sub(10)..rows.len());
    for (index, row) in rows.iter().enumerate() {
        if row["coverage_narrowing"].as_bool() == Some(true) {
            indices.extend(index.saturating_sub(1)..=(index + 1).min(rows.len() - 1));
        }
    }
    let gpu = gpu_or_skip().expect("GPU required for full-HD coverage probe");
    let mut renderer = GpuExecutor::new(
        gpu,
        GpuExecutorConfig {
            viewport_size: ViewportSize {
                width: 1920,
                height: 1080,
            },
            ..{
                let source_sizes = vec![(640, 360); cal.projection.camera_count()];
                GpuExecutorConfig::new(cal, source_sizes, InputFormat::Nv12)
            }
        },
    )
    .unwrap();
    let white = vec![235; 640 * 360];
    let neutral = vec![128; 640 * 360 / 2];
    let sources = [Nv12Planes {
        y: &white,
        uv: &neutral,
    }; 2];
    let mut results = Vec::new();
    let mut holes = 0;
    let mut border_holes = 0;
    for index in indices {
        let pose: RenderPose = serde_json::from_value(rows[index]["pose"].clone()).unwrap();
        let image = renderer.stitch(&sources, pose).unwrap();
        let count = missing(&image);
        let border_count = missing_border(&image, 1920);
        holes += count;
        border_holes += border_count;
        results.push(serde_json::json!({ "frame": index, "missing_pixels": count, "missing_border_pixels": border_count }));
    }
    let summary = serde_json::json!({ "width": 1920, "height": 1080, "poses": results, "missing_pixels": holes, "missing_border_pixels": border_holes });
    let mut output = std::fs::File::create_new(dir.join("high-resolution.json")).unwrap();
    serde_json::to_writer_pretty(&mut output, &summary).unwrap();
    assert_eq!(border_holes, 0);
}

#[test]
#[ignore = "renders fixed-FOV calibrated motion using external calibration and optional source footage"]
fn render_fixed_fov_calibrated_motion() {
    let dir = PathBuf::from(std::env::var_os("RECO_COVERAGE_ARTIFACT_DIR").unwrap());
    let calibration = std::env::var_os("RECO_COVERAGE_CALIBRATION").unwrap();
    let mut cal: Calibration =
        serde_json::from_slice(&std::fs::read(calibration).unwrap()).unwrap();
    if let Some(shape) = cal.projection.l_shape_mut() {
        shape.blend_width = 0.0;
    }
    cal.validate().unwrap();
    let mut reference_cal: Calibration = std::env::var_os("RECO_REFERENCE_CALIBRATION")
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .unwrap_or_else(|| cal.clone());
    if let Some(shape) = reference_cal.projection.l_shape_mut() {
        shape.blend_width = 0.0;
    }
    reference_cal.validate().unwrap();
    let fov = std::env::var("RECO_FIXED_FOV")
        .unwrap_or_else(|_| "45".into())
        .parse::<f32>()
        .unwrap();
    let tracked_fov = std::env::var("RECO_TRACKED_FOV").is_ok_and(|v| v == "1");
    let trajectory: Option<Vec<RenderPose>> = std::env::var_os("RECO_COVERAGE_TRAJECTORY")
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap());
    let events = std::env::var_os("RECO_COVERAGE_EVENTS").map(|path| {
        let mut targets = std::collections::BTreeMap::new();
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let event: serde_json::Value = serde_json::from_str(line).unwrap();
            if event["kind"] == "pan_decision" {
                let p: crate::geometry::WorldPose =
                    serde_json::from_value(event["pose"].clone()).unwrap();
                let camera = cal.projection.geometry().virtual_camera(&cal.framing);
                targets.insert(
                    event["frame_index"].as_u64().unwrap() as usize,
                    (camera.yaw_pitch_to_direction(p.yaw, p.pitch), p.fov_degrees),
                );
            }
        }
        assert!(targets.keys().copied().eq(0..targets.len()));
        targets
    });
    assert!(
        !tracked_fov || events.is_some() || trajectory.is_some(),
        "tracked FOV requires recorded or synthetic requests"
    );
    assert!(events.is_none() || trajectory.is_none());
    let frame_count = std::env::var("RECO_COVERAGE_FRAMES")
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or_else(|| {
            trajectory.as_ref().map_or_else(
                || events.as_ref().map_or(600, |events| events.len()),
                Vec::len,
            )
        });
    assert!(frame_count > 0);
    assert!(
        events
            .as_ref()
            .is_none_or(|events| events.len() >= frame_count)
    );
    let size = ViewportSize {
        width: 640,
        height: 360,
    };
    assert!(trajectory.as_ref().is_none_or(|poses| {
        poses.len() >= frame_count && poses.iter().all(|p| valid_query(&cal, &size, *p))
    }));
    let references = std::env::var_os("RECO_FIXED_REFERENCE").map(|path| {
        let poses: Vec<RenderPose> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| {
                let row: serde_json::Value = serde_json::from_str(line).unwrap();
                serde_json::from_value(row["pose"].clone()).unwrap()
            })
            .collect();
        assert!(poses.len() >= frame_count);
        assert!(poses.iter().all(|p| valid_query(&reference_cal, &size, *p)));
        assert!(tracked_fov || poses.iter().all(|p| p.fov_degrees == fov));
        poses
    });
    // Render a previously measured controller run without running it a second time.
    let solved: Vec<serde_json::Value> =
        std::fs::read_to_string(std::env::var_os("RECO_SOLVED_MOTION").unwrap())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
    assert!(solved.len() >= frame_count);
    let input_dir = std::env::var_os("RECO_FIXED_INPUT_DIR").map(PathBuf::from);
    assert!(
        events.is_none() || input_dir.is_some(),
        "recorded AI replay requires the synchronized source frames"
    );
    let height = std::env::var("RECO_COVERAGE_INPUT_HEIGHT")
        .unwrap_or_else(|_| "360".into())
        .parse::<usize>()
        .unwrap();
    let width = 640;
    let mut inputs = input_dir.map(|path| {
        assert_eq!(
            cal.lenses.len(),
            2,
            "external fixture uses two synchronized files"
        );
        ["left.nv12", "right.nv12"].map(|name| std::fs::File::open(path.join(name)).unwrap())
    });
    let source_template: Vec<_> = (0..cal.lenses.len())
        .map(|i| gradient(width, height, i))
        .collect();
    let mut sources = source_template.clone();
    let cpu = std::env::var("RECO_CPU_RENDER").is_ok_and(|v| v == "1");
    let gpu = (!cpu).then(|| gpu_or_skip().expect("GPU required for calibrated motion artifacts"));
    let make_renderer = |calibration: Calibration| -> Box<dyn StitchExecutor> {
        if let Some(gpu) = &gpu {
            Box::new(
                GpuExecutor::new(
                    gpu.clone(),
                    GpuExecutorConfig {
                        viewport_size: size.clone(),
                        ..{
                            let source_sizes = vec![
                                (width as u32, height as u32);
                                calibration.projection.camera_count()
                            ];
                            GpuExecutorConfig::new(calibration, source_sizes, InputFormat::Nv12)
                        }
                    },
                )
                .unwrap(),
            )
        } else {
            Box::new(
                {
                    let source_sizes =
                        vec![(width as u32, height as u32); calibration.projection.camera_count()];
                    CpuExecutor::new(calibration, size.clone(), source_sizes, false)
                }
                .unwrap(),
            )
        }
    };
    let mut reference_renderer =
        (reference_cal != cal).then(|| make_renderer(reference_cal.clone()));
    let mut renderer = make_renderer(cal.clone());
    eprintln!("Motion renderer: {}", renderer.name());
    let white = vec![235; width * height];
    let neutral = vec![128; width * height / 2];
    let whites = vec![
        Nv12Planes {
            y: &white,
            uv: &neutral,
        };
        cal.lenses.len()
    ];
    let target_at = |frame| {
        if let Some(poses) = &trajectory {
            return direction(&cal, poses[frame]);
        }
        events.as_ref().map_or_else(
            || direction(&cal, synthetic_target(frame, fov)),
            |events| events[&frame].0,
        )
    };
    let fov_at = |frame| {
        if tracked_fov {
            trajectory.as_ref().map_or_else(
                || events.as_ref().unwrap()[&frame].1,
                |poses| poses[frame].fov_degrees,
            )
        } else {
            fov
        }
    };
    let fov_range = (0..frame_count)
        .map(fov_at)
        .fold([f32::INFINITY, f32::NEG_INFINITY], |r, f| {
            [r[0].min(f), r[1].max(f)]
        });
    let mut current: RenderPose = serde_json::from_value(solved[0]["pose"].clone()).unwrap();
    let mut velocity = [0.0; 2];
    let mode = if events.is_some() {
        "recorded-ai"
    } else {
        "synthetic"
    };
    let mut metadata = std::fs::File::create_new(dir.join("motion.jsonl")).unwrap();
    let mut scene = Vec::new();
    for y in 0..121 {
        for x in 0..241 {
            let p = RenderPose {
                yaw: -2.0 + x as f32 / 240.0 * 4.0,
                pitch: 1.0 - y as f32 / 120.0 * 2.0,
                fov_degrees: fov,
            };
            let maps = cal.surface_maps(&size, p);
            scene.push(
                maps.iter()
                    .any(|(map, _)| map.sample_uv([0.0; 2]).is_some()),
            );
        }
    }
    let mut config = std::fs::File::create_new(dir.join("scene.json")).unwrap();
    let info = serde_json::json!({ "mode": mode, "renderer": if cpu { "cpu" } else { "gpu" }, "coverage_scope": "viewport-sides", "reference": if references.is_some() { "previous-solver" } else { "requested-view" }, "fov": fov, "fov_mode": if tracked_fov { "contact-limited" } else { "fixed" }, "fov_range": fov_range, "aspect": size.aspect_ratio(), "yaw_range": [-2.0, 2.0], "pitch_range": [-1.0, 1.0], "columns": 241, "rows": 121, "covered_direction": scene, "rate": RATE, "zoom_rate": ZOOM_RATE, "calibration": cal, "reference_calibration": reference_cal });
    serde_json::to_writer(&mut config, &info).unwrap();
    let mut encoder = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            "1280x360",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(dir.join("rendered-views.mp4"))
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = encoder.stdin.take().unwrap();
    let mut combined = vec![0; 1280 * 360 * 4];
    let (mut holes, mut reference_holes, mut lost, mut reference_lost, mut source_absent) =
        (0, 0, 0, 0, 0);
    let mut max_step = 0.0_f32;
    let mut border_holes = 0;
    let mut rendered_frames = 0;
    for frame in 0..frame_count {
        let target = target_at(frame);
        let target_pose = aim(&cal, &target, fov_at(frame));
        let previous = current;
        let result = Advance {
            pose: serde_json::from_value(solved[frame]["pose"].clone()).unwrap(),
            velocity: serde_json::from_value(solved[frame]["velocity"].clone()).unwrap(),
            support_queries: solved[frame]["queries"].as_u64().unwrap() as usize,
            objective: serde_json::from_value(solved[frame]["objective"].clone()).unwrap(),
        };
        assert!((result.pose.fov_degrees - previous.fov_degrees).abs() <= ZOOM_RATE / 30.0 + 1e-5);
        let acceleration =
            (result.velocity[0] - velocity[0]).hypot(result.velocity[1] - velocity[1]) * 30.0;
        assert!(
            acceleration <= ACCELERATION + 0.001,
            "frame {frame}: {acceleration}"
        );
        velocity = result.velocity;
        current = result.pose;
        let reference = references
            .as_ref()
            .map_or(target_pose, |poses| poses[frame]);
        max_step = max_step.max(distance(previous, current));
        let target_maps = cal.surface_maps(&size, target_pose);
        let mut supported_target = false;
        if let Some(inputs) = &mut inputs {
            for (input, source) in inputs.iter_mut().zip(&mut sources) {
                input.read_exact(source).unwrap();
            }
            supported_target = target_maps
                .iter()
                .any(|(map, _)| map.sample_uv([0.0; 2]).is_some());
        } else {
            sources.clone_from(&source_template);
            for ((map, _), source) in target_maps.iter().zip(&mut sources) {
                if let Some(uv) = map.sample_uv([0.0; 2]) {
                    supported_target = true;
                    paint_target(source, width, height, uv);
                }
            }
        }
        source_absent += usize::from(!supported_target);
        let new_white = renderer.stitch(&whites, current).unwrap();
        let old_white = reference_renderer
            .as_mut()
            .unwrap_or(&mut renderer)
            .stitch(&whites, reference)
            .unwrap();
        let new_holes = missing(&new_white);
        let new_border_holes = missing_border(&new_white, 640);
        border_holes += new_border_holes;
        let old_holes = missing(&old_white);
        holes += new_holes;
        reference_holes += old_holes;
        let frames: Vec<_> = sources
            .iter()
            .map(|source| Nv12Planes {
                y: &source[..width * height],
                uv: &source[width * height..],
            })
            .collect();
        let left = renderer.stitch(&frames, current).unwrap();
        let right = reference_renderer
            .as_mut()
            .unwrap_or(&mut renderer)
            .stitch(&frames, reference)
            .unwrap();
        for y in 0..360 {
            let from = y * 640 * 4;
            let to = y * 1280 * 4;
            combined[to..to + 640 * 4].copy_from_slice(&left[from..from + 640 * 4]);
            combined[to + 640 * 4..to + 1280 * 4].copy_from_slice(&right[from..from + 640 * 4]);
        }
        pipe.write_all(&combined).unwrap();
        let new_xy = screen(&cal, &size, current, &target);
        let old_xy = screen(&reference_cal, &size, reference, &target);
        lost += usize::from(!visible(new_xy));
        reference_lost += usize::from(!visible(old_xy));
        let mut reference_footprint = footprint(&reference_cal, &size, reference);
        if reference_cal != cal {
            for point in &mut reference_footprint {
                let ray = direction(
                    &reference_cal,
                    RenderPose {
                        yaw: point[0],
                        pitch: point[1],
                        ..reference
                    },
                );
                let p = aim(&cal, &ray, reference.fov_degrees);
                *point = [p.yaw, p.pitch];
            }
        }
        let record = serde_json::json!({ "frame": frame, "coverage_narrowing": current.fov_degrees < previous.fov_degrees && fov_at(frame) >= previous.fov_degrees, "target": target_pose, "pose": current, "reference": reference, "target_screen": new_xy, "reference_target_screen": old_xy, "visible": visible(new_xy), "reference_visible": visible(old_xy), "source_target_supported": supported_target, "missing_pixels": new_holes, "missing_border_pixels": new_border_holes, "reference_missing_pixels": old_holes, "step": distance(previous, current), "velocity": velocity, "acceleration": acceleration, "queries": result.support_queries, "footprint": footprint(&cal, &size, current), "reference_footprint": reference_footprint });
        serde_json::to_writer(&mut metadata, &record).unwrap();
        writeln!(metadata).unwrap();
        rendered_frames += 1;
        if frame % 100 == 0 {
            eprintln!(
                "{mode} {frame}/{frame_count}: holes={holes}, lost target={lost}, reference lost={reference_lost}, source absent={source_absent}"
            );
        }
    }
    drop(pipe);
    assert!(encoder.wait().unwrap().success());
    let summary = serde_json::json!({ "frames": rendered_frames, "requested_frames": frame_count, "rejection": null, "mode": mode, "renderer": if cpu { "cpu" } else { "gpu" }, "coverage_scope": "viewport-sides", "fixed_fov": if tracked_fov { None } else { Some(fov) }, "fov_range": fov_range, "missing_pixels": holes, "missing_border_pixels": border_holes, "reference_missing_pixels": reference_holes, "target_outside_frames": lost, "reference_target_outside_frames": reference_lost, "source_target_absent_frames": source_absent, "max_step": max_step, "step_limit": RATE / 30.0 });
    let mut file = std::fs::File::create_new(dir.join("results.json")).unwrap();
    serde_json::to_writer_pretty(&mut file, &summary).unwrap();
    eprintln!("{summary}");
    assert_eq!(border_holes, 0);

    assert!(max_step <= RATE / 30.0 + 1e-6);
}
