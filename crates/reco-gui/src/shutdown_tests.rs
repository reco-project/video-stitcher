//! Worker ownership at application shutdown.

use super::*;

#[test]
fn application_drop_cancels_export_and_drains_recording() {
    let mut state = AppState::new(crate::settings::GuiSettings::default());
    let cancelled = state.export_interrupted.clone();
    let exported = Arc::new(AtomicBool::new(false));
    let finished = exported.clone();
    state.export_thread = Some(std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cancelled.load(Ordering::Relaxed) {
            assert!(Instant::now() < deadline, "export was not cancelled");
            std::thread::yield_now();
        }
        finished.store(true, Ordering::Relaxed);
    }));
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    state.recording_tx = Some(tx);
    let recorded = Arc::new(AtomicBool::new(false));
    let finished = recorded.clone();
    state.recording_thread = Some(std::thread::spawn(move || {
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(5)),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
        ));
        finished.store(true, Ordering::Relaxed);
    }));
    drop(state);
    assert!(exported.load(Ordering::Relaxed));
    assert!(recorded.load(Ordering::Relaxed));
}
