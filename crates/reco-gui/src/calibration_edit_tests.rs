//! Headless checks for calibration editor acceptance and rollback.

use super::*;
use reco_core::calibration::{Framing, Lens};
use reco_core::projection::LShape;
use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};

struct HeadlessPlatform;

impl slint::platform::Platform for HeadlessPlatform {
    fn create_window_adapter(
        &self,
    ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
        Ok(MinimalSoftwareWindow::new(RepaintBufferType::ReusedBuffer))
    }
}

fn calibration() -> Calibration {
    Calibration::new(
        vec![Lens::fisheye(64, 36, 32.0, 32.0, 32.0, 18.0, [0.0; 4]); 2],
        LShape {
            intersect: 0.5,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
            blend_width: 0.05,
        },
        Framing {
            axis_offset: 0.25,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

fn exercise_editor(state: &mut AppState, app: &RecoApp) {
    let accepted = serde_json::to_value(state.calibration.as_ref().unwrap()).unwrap();
    app.set_cal_camera_axis_offset(0.0);
    assert!(!state.edit_calibration(app, |cal| cal.framing.axis_offset = 0.0));
    assert_eq!(app.get_cal_camera_axis_offset(), 0.25);
    assert!(!app.get_cal_dirty());
    assert_eq!(
        serde_json::to_value(state.calibration.as_ref().unwrap()).unwrap(),
        accepted
    );
    if let Some(bridge) = &state.bridge {
        assert_eq!(
            serde_json::to_value(bridge.engine().calibration()).unwrap(),
            accepted
        );
    }

    state.lens_preview_active = true;
    assert!(state.edit_calibration(app, |cal| {
        for lens in &mut cal.lenses {
            lens.correction = 0.0;
        }
    }));
    assert_eq!(state.lens_correction_amount, 0.0);
    let untouched_right =
        serde_json::to_value(&state.calibration.as_ref().unwrap().lenses[1]).unwrap();
    let profile = Lens::fisheye(128, 72, 80.0, 82.0, 66.0, 38.0, [-0.01, 0.0, 0.0, 0.0]);
    assert!(state.apply_lens_profile(app, &profile, "left"));
    state.sync_calibration_controls(app);
    let cal = state.calibration.as_ref().unwrap();
    assert_eq!((cal.lenses[0].width, cal.lenses[0].height), (64, 36));
    assert_eq!((cal.lenses[0].fx, cal.lenses[0].fy), (40.0, 41.0));
    assert_eq!(cal.lenses[0].correction, 0.0);
    assert_eq!(
        serde_json::to_value(&cal.lenses[1]).unwrap(),
        untouched_right
    );
    assert_eq!(app.get_lens_left_fx(), 40.0);
    assert_eq!(app.get_lens_right_fx(), 32.0);
    if let Some(bridge) = &state.bridge {
        assert_eq!(
            serde_json::to_value(bridge.engine().calibration()).unwrap(),
            serde_json::to_value(cal).unwrap()
        );
    }

    let accepted = state.calibration.clone().unwrap();
    let mut invalid = profile.clone();
    invalid.height = 0;
    assert!(!state.apply_lens_profile(app, &invalid, "both"));
    assert_eq!(state.calibration.as_ref().unwrap(), &accepted);
    if let Some(bridge) = &state.bridge {
        assert_eq!(bridge.engine().calibration(), &accepted);
    }
    assert!(state.apply_lens_profile(app, &profile, "both"));
    let accepted = state.calibration.as_ref().unwrap();
    assert_eq!(accepted.lenses[0], accepted.lenses[1]);
    assert_eq!(accepted.lenses[1].correction, 0.0);
}

#[test]
fn editor_rolls_back_invalid_controls_and_keeps_preview_and_document_together() {
    slint::platform::set_platform(Box::new(HeadlessPlatform)).unwrap();
    let app = RecoApp::new().unwrap();
    let mut state = AppState::new(crate::settings::GuiSettings::default());
    state.calibration = Some(calibration());
    exercise_editor(&mut state, &app);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: reco_core::gpu::GpuContext::select_backends(),
        ..Default::default()
    });
    let adapter =
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        {
            Ok(adapter) => adapter,
            Err(error) => {
                assert!(
                    std::env::var("RECO_REQUIRE_GPU").is_err(),
                    "GPU required: {error}"
                );
                eprintln!("Skipping GPU editor checks: {error}");
                return;
            }
        };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .unwrap();
    state.calibration = Some(calibration());
    state.bridge = Some(
        PreviewBridge::new(
            device,
            queue,
            adapter.get_info(),
            calibration(),
            64,
            36,
            64,
            36,
        )
        .unwrap(),
    );
    exercise_editor(&mut state, &app);
}
