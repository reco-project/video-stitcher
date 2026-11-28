//! The GUI's real opening, polling and processed-output lifecycle with owned loopback peers.

use super::*;
use crate::{ExportOutcome, GpuContext, PlayState};
use reco_core::calibration::{Framing, Lens};
use reco_core::projection::Cylinder;
use slint::Model;
use slint::platform::software_renderer::{MinimalSoftwareWindow, RepaintBufferType};
use std::path::PathBuf;
use std::time::Instant;

struct Platform;
impl slint::platform::Platform for Platform {
    fn create_window_adapter(
        &self,
    ) -> Result<Rc<dyn slint::platform::WindowAdapter>, slint::PlatformError> {
        Ok(MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer))
    }
}

fn app() -> RecoApp {
    slint::platform::set_platform(Box::new(Platform)).unwrap();
    RecoApp::new().unwrap()
}

fn calibration() -> Calibration {
    Calibration::new(
        vec![Lens::flat(64, 32)],
        Cylinder {
            surface_height: Some(2400.0),
            ..Default::default()
        },
        Framing {
            axis_offset: 0.0,
            tilt: 0.0,
            roll: 0.0,
        },
    )
}

fn config(url: &str) -> Config {
    Config {
        inputs: vec![RtspInput {
            url: url.into(),
            timestamp_offset_us: 0,
        }],
        options: RtspConfig {
            fps: (10, 1),
            synchronization: reco_io::frame_sync::Config {
                wait_us: 150_000,
                skew_us: 30_000,
                max_age_us: 350_000,
                capacity: 4,
            },
            open_timeout: Duration::from_secs(3),
            read_timeout: Duration::from_millis(500),
            software_decode: true,
        },
    }
}

fn wait_open(state: &mut AppState, app: &RecoApp) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while state.network.is_opening() {
        assert!(Instant::now() < deadline);
        poll(state, &app.as_weak());
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn form_requires_explicit_clocks_and_preserves_camera_order_without_echoing_urls() {
    let app = app();
    let rows = Rc::new(slint::VecModel::from(vec![
        NetworkInputRow {
            url: "rtsp://private:password@127.0.0.1/cam0".into(),
            offset: "".into(),
        },
        NetworkInputRow {
            url: "rtsp://127.0.0.1/cam1".into(),
            offset: "-7000000".into(),
        },
    ]));
    app.set_network_inputs(rows.clone().into());
    assert!(parse(&app, 2).err().unwrap().contains("offset"));
    let mut first = rows.row_data(0).unwrap();
    first.offset = "0".into();
    rows.set_row_data(0, first.clone());
    app.set_network_fps("30000/1001".into());
    let request = parse(&app, 2).unwrap();
    assert_eq!(request.options.fps, (30000, 1001));
    assert_eq!(request.inputs[0].timestamp_offset_us, 0);
    assert_eq!(request.inputs[1].timestamp_offset_us, -7_000_000);
    assert!(parse(&app, 1).is_err());
    for invalid in ["0", "nan", "30/0", "30/-1", "30000/1001/2"] {
        app.set_network_fps(invalid.into());
        assert!(parse(&app, 2).is_err());
    }
    first.url = "https://private:password@127.0.0.1/".into();
    rows.set_row_data(0, first);
    assert!(!parse(&app, 2).err().unwrap().contains("password"));
}

#[test]
fn cancellation_and_stale_open_do_not_replace_the_document_or_reconnect() {
    let app = app();
    let mut state = AppState::new(Default::default());
    state.calibration = Some(calibration());
    let peer = peer::Peer::silent();
    state.network.begin(config(&peer.url), calibration());
    state.network.cancel();
    wait_open(&mut state, &app);
    assert!(state.network.active.is_none());
    assert_eq!(state.calibration, Some(calibration()));
    assert!(!app.get_network_message().is_empty());
    let peer = peer::Peer::mapped(30, 0, None);
    state.network.begin(config(&peer.url), calibration());
    state.calibration.as_mut().unwrap().framing.roll = 0.1;
    wait_open(&mut state, &app);
    assert!(state.network.active.is_none());
    assert!(state.bridge.is_none());
    assert_eq!(state.calibration.as_ref().unwrap().framing.roll, 0.1);
    assert!(app.get_network_message().contains("changed"));
    assert!(!state.network.is_opening());
}

#[test]
fn live_playback_polls_pauses_and_preserves_the_displayed_image_on_failure() {
    let peer = peer::Peer::video(40, None);
    let cleanup = reco_io::DecoderCleanup::default();
    let mut source = RtspSource::open(
        config(&peer.url).inputs,
        config(&peer.url).options,
        Arc::default(),
    )
    .unwrap();
    let first = source.next_frame().unwrap().unwrap();
    let mut playback = Playback::from_live(source, first, &cleanup);
    assert!(playback.is_live());
    assert!(playback.seek_frame(0).is_err());
    playback.pause();
    let index = playback.frame_index();
    assert!(!playback.tick().unwrap());
    std::thread::sleep(Duration::from_millis(500));
    playback.toggle();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !playback.tick().unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(playback.frame_index(), index + 1);
    drop(peer);
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        assert!(Instant::now() < deadline);
        if playback.tick().is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(playback.state(), PlayState::Paused);
    assert!(playback.current_frame().is_some());
    drop(playback);
    drop(cleanup);
}

#[test]
#[ignore = "requires a GPU; uses an owned RTSP/AAC peer and writes only to a temporary directory"]
fn actual_live_preview_then_recording_stops_and_finalizes_audio() {
    let app = app();
    let mut state = AppState::new(Default::default());
    state.shared_gpu = Some(GpuContext::new_blocking().unwrap());
    state.use_constrained_look = false;
    state.calibration = Some(calibration());
    let peer = peer::Peer::video(50, None);
    state.network.begin(config(&peer.url), calibration());
    wait_open(&mut state, &app);
    assert!(state.playback.is_live(), "{}", app.get_network_message());
    assert!(app.get_network_active());
    assert!(app.get_files_loaded());
    let deadline = Instant::now() + Duration::from_secs(2);
    while state.playback.frame_index() < 5 {
        assert!(Instant::now() < deadline);
        if state.playback.tick().unwrap() {
            assert!(state.render_current().is_some());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    state.suspend_preview_for_export();
    let preview_cleanup = std::mem::take(&mut state.decoder_cleanup);
    assert!(state.bridge.is_none());
    assert!(state.network.active.is_some());
    assert_eq!(state.playback.state(), PlayState::Empty);
    drop(preview_cleanup);
    drop(peer);

    let peer = peer::Peer::with_audio(80, 0, 125_000);
    let interrupted = Arc::new(AtomicBool::new(false));
    let cancel = interrupted.clone();
    let stop = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(3));
        cancel.store(true, Ordering::Relaxed);
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live.mp4");
    let outcome = crate::export::run_export(
        crate::export::ExportRequest {
            calibration_path: None,
            inputs: crate::export::Inputs::Network(config(&peer.url)),
            calibration: calibration(),
            output: path.clone(),
            stream_url: None,
            audio: reco_io::output::AudioMode::CopyFrom(0),
            replay_enabled: false,
            events_path: None,
            width: 64,
            height: 36,
            codec: "h264".into(),
            quality: "fast".into(),
            start_secs: 0.0,
            end_secs: 0.0,
            autocam: crate::export::AutocamUiConfig::default(),
        },
        Arc::new(std::sync::OnceLock::new()),
        slint::Weak::default(),
        &interrupted,
    );
    stop.join().unwrap();
    let ExportOutcome::Ok(frames, output) = outcome else {
        panic!("{outcome:?}");
    };
    assert!(frames >= 5);
    assert_eq!(output, path);
    let mut decoder = reco_io::ffmpeg::decoder::VideoDecoder::open_software(&path).unwrap();
    for _ in 0..frames {
        assert!(decoder.next_frame().unwrap().is_some());
    }
    assert!(decoder.next_frame().unwrap().is_none());
    let mut media = ffmpeg_next::format::input(&path).unwrap();
    let audio = media
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .unwrap()
        .index();
    assert!(
        media
            .packets()
            .filter(|(stream, _)| stream.index() == audio)
            .count()
            > 20
    );
    if let Some(directory) = std::env::var_os("RECO_GUI_TEST_ARTIFACT_DIR") {
        let destination = PathBuf::from(directory).join("network-recording.mp4");
        assert!(!destination.exists());
        std::fs::copy(&path, destination).unwrap();
    }
    state.restore_preview_after_export().unwrap();
    assert!(
        !state.network.is_opening(),
        "completion must not silently reconnect a live source"
    );
    assert!(state.bridge.is_none());
    app.set_network_active(false);
    app.set_files_loaded(false);
    app.set_left_path("left.mp4".into());
    app.set_right_path("right.mp4".into());
    assert!(
        app.get_has_both_videos(),
        "a previous live connection must not disable two-file calibration"
    );
}
