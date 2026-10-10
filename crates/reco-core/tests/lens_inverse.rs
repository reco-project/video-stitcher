//! Source mapping regression against the actual forward surface mapping.
use nalgebra::{Perspective3, Vector4};
use reco_core::calibration::{Calibration, Framing, Lens};
use reco_core::geometry::{FAR_PLANE, NEAR_PLANE, RenderPose, view_matrix};
use reco_core::projection::LShape;
use reco_core::render::viewport::ViewportSize;

struct CountAlloc;
thread_local! { static ALLOCS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;
// SAFETY: Memory operations forward the original pointer and layout to System unchanged.
unsafe impl std::alloc::GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCS.with(|count| {
            if let Some(value) = count.get() {
                count.set(Some(value + 1));
            }
        });
        unsafe { std::alloc::GlobalAlloc::alloc(&std::alloc::System, layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::GlobalAlloc::dealloc(&std::alloc::System, ptr, layout) }
    }
}

#[test]
fn source_mapping_recovers_rendered_rays_on_the_central_fisheye_branch() {
    let lens = Lens::fisheye(
        3840,
        2880,
        1500.0,
        1500.0,
        1920.0,
        1440.0,
        [0.5, 0.0, -0.08, 0.0],
    );
    let mut cal = Calibration::new(
        vec![lens.clone(), lens],
        LShape {
            intersect: 0.54,
            x_ty: 0.004,
            x_rz: 0.01,
            z_rx: -0.008,
            x_rx: 0.014,
            z_rz: -0.019,
            blend_width: 0.0,
        },
        Framing {
            axis_offset: 0.23,
            tilt: 0.17,
            roll: -0.11,
        },
    );
    cal.lenses[1].width = 1920;
    cal.lenses[1].height = 1080;
    cal.lenses[1].fx *= 0.5;
    cal.lenses[1].fy *= 0.375;
    cal.lenses[1].cx *= 0.5;
    cal.lenses[1].cy *= 0.375;
    cal.validate().unwrap();
    let viewport = ViewportSize {
        width: 640,
        height: 360,
    };
    let camera = cal.virtual_camera();
    let mut samples = 0;
    let mut failed = 0;
    let mut wrong = 0;
    let mut max_error = 0.0_f64;
    for correction in [0.0, 0.4, 1.0] {
        for lens in &mut cal.lenses {
            lens.correction = correction;
        }
        for yaw in [-0.9, 0.0, 0.9] {
            for pitch in [-0.3, 0.0, 0.3] {
                for fov in [50.0, 100.0] {
                    let pose = RenderPose {
                        yaw,
                        pitch,
                        fov_degrees: fov,
                    };
                    let maps = cal.surface_maps(&viewport, pose);
                    let view = view_matrix(
                        &camera,
                        yaw,
                        pitch,
                        cal.framing.tilt as f32,
                        cal.framing.roll as f32,
                    );
                    let projection = Perspective3::new(
                        640.0 / 360.0,
                        pose.render_fov().to_radians(),
                        NEAR_PLANE,
                        FAR_PLANE,
                    )
                    .to_homogeneous();
                    for (index, (map, _)) in maps.iter().enumerate() {
                        for y in (0..360).step_by(4) {
                            for x in (0..640).step_by(4) {
                                let ndc = [
                                    (x as f64 + 0.5) / 320.0 - 1.0,
                                    1.0 - (y as f64 + 0.5) / 180.0,
                                ];
                                let Some(uv) = map.sample_uv(ndc) else {
                                    continue;
                                };
                                samples += 1;
                                ALLOCS.with(|count| count.set(Some(0)));
                                let mapped =
                                    cal.map_source_point(index, [uv.u as f32, uv.v as f32]);
                                let allocations = ALLOCS.with(|count| count.replace(None).unwrap());
                                assert_eq!(allocations, 0, "source mapping allocated heap storage");
                                let Ok(angles) = mapped else {
                                    failed += 1;
                                    continue;
                                };
                                let d = camera.yaw_pitch_to_direction(angles.yaw, angles.pitch);
                                let clip = projection * view * Vector4::new(d.x, d.y, d.z, 0.0);
                                let error = ((clip.x as f64 / clip.w as f64 - ndc[0]) * 320.0)
                                    .hypot((clip.y as f64 / clip.w as f64 - ndc[1]) * 180.0);
                                max_error = max_error.max(error);
                                if clip.w <= 0.0 || !error.is_finite() || error > 0.01 {
                                    wrong += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("samples={samples} failed={failed} wrong={wrong} max_error_px={max_error}");
    assert!(samples > 100_000);
    assert_eq!(failed, 0);
    assert_eq!(wrong, 0);
}
