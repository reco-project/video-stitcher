//! Ownership and freshness of a background calibration result.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

use super::{AppState, Calibration, CalibrationResult};

/// One worker retained until completion, including after cooperative cancellation.
pub(super) struct PendingCalibration {
    pub receiver: mpsc::Receiver<CalibrationResult>,
    interrupted: Arc<AtomicBool>,
    snapshot: CalibrationSnapshot,
}

pub(super) struct CalibrationSnapshot {
    inputs: [Option<Vec<PathBuf>>; 2],
    calibration: Option<Calibration>,
    path: Option<PathBuf>,
}

impl PendingCalibration {
    pub fn new(
        state: &AppState,
        receiver: mpsc::Receiver<CalibrationResult>,
        interrupted: Arc<AtomicBool>,
    ) -> Self {
        Self {
            receiver,
            interrupted,
            snapshot: CalibrationSnapshot::new(state),
        }
    }

    pub fn matches(&self, state: &AppState) -> bool {
        self.snapshot.matches(state)
    }

    pub fn cancel(&self) {
        self.interrupted.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.interrupted.load(Ordering::Relaxed)
    }
}

impl CalibrationSnapshot {
    pub fn new(state: &AppState) -> Self {
        Self {
            inputs: Self::inputs(state),
            calibration: state.calibration.clone(),
            path: state.calibration_path.clone(),
        }
    }

    fn inputs(state: &AppState) -> [Option<Vec<PathBuf>>; 2] {
        [&state.left_input, &state.right_input]
            .map(|input| input.as_ref().map(reco_io::InputPath::all_paths))
    }

    pub fn matches(&self, state: &AppState) -> bool {
        self.inputs == Self::inputs(state)
            && self.calibration == state.calibration
            && self.path == state.calibration_path
    }
}

impl Drop for PendingCalibration {
    fn drop(&mut self) {
        self.cancel();
    }
}
