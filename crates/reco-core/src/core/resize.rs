//! Nonblocking coverage preparation shared by preview resize and calibration edits.

use super::{StitchCore, StitchCoreError};
use crate::{
    calibration::Calibration,
    projection::coverage::controller::{Controller, MapConfig},
    render::viewport::ViewportSize,
};
use std::sync::mpsc::{Receiver, TryRecvError};

pub(super) struct PendingCoverage {
    calibration: Calibration,
    size: ViewportSize,
    result: Receiver<Result<Controller, String>>,
    error: Option<String>,
}

impl StitchCore {
    /// Prepare the current viewport's coverage without blocking a preview thread.
    ///
    /// Returns `None` while the map builds. The accepted viewport and pose remain usable until a later call returns its FOV cap. Preparation failures are cached for the same calibration and size.
    pub fn prepare_coverage_preview(&mut self) -> Result<Option<f32>, StitchCoreError> {
        let size = self.executor.viewport_size().clone();
        if let Some(controller) = &self.coverage_motion
            && controller.matches_viewport(&size)
        {
            return Ok(Some(controller.cap()));
        }
        let calibration = self.calibration().clone();
        let Some(controller) = self.poll_coverage(&calibration, &size)? else {
            return Ok(None);
        };
        let cap = controller.cap();
        self.coverage_motion = Some(controller);
        Ok(Some(cap))
    }

    /// Request a preview resize without waiting for coverage construction.
    ///
    /// Poll with the latest desired dimensions until `Ok(true)`. `Ok(false)` keeps the previous viewport and coverage available for rendering. At most one map builds at a time; superseded results are discarded. A failed request leaves the engine unchanged. Identical-aspect resizes reuse the map and motion state. GPU allocation still happens on the calling thread when the result is accepted.
    pub fn resize_preview(&mut self, width: u32, height: u32) -> Result<bool, StitchCoreError> {
        let size = ViewportSize { width, height };
        size.validate()
            .map_err(|e| StitchCoreError::Config(e.to_string()))?;
        if (width, height) == self.output_dims() && self.executor.viewport_size() == &size {
            return Ok(true);
        }
        if self
            .coverage_motion
            .as_ref()
            .is_none_or(|c| c.matches_viewport(&size))
        {
            return self
                .resize(width, height)
                .map(|_| true)
                .ok_or_else(|| StitchCoreError::Config("viewport resize rejected".into()));
        }
        let calibration = self.calibration().clone();
        match self.poll_coverage(&calibration, &size)? {
            None => Ok(false),
            Some(controller) => self
                .apply_resize(width, height, Some(controller))
                .map(|_| true)
                .ok_or_else(|| StitchCoreError::Config("viewport resize rejected".into())),
        }
    }

    pub(super) fn poll_coverage(
        &mut self,
        calibration: &Calibration,
        size: &ViewportSize,
    ) -> Result<Option<Controller>, StitchCoreError> {
        if let Some(pending) = &self.pending_coverage {
            let current = pending.size == *size && pending.calibration == *calibration;
            if current && let Some(error) = &pending.error {
                return Err(StitchCoreError::Config(error.clone()));
            }
            let result = if pending.error.is_some() {
                Err(TryRecvError::Disconnected)
            } else {
                pending.result.try_recv()
            };
            match result {
                Err(TryRecvError::Empty) => return Ok(None),
                Ok(Ok(controller)) if current => {
                    self.pending_coverage = None;
                    return Ok(Some(controller));
                }
                result if current => {
                    let error = match result {
                        Ok(Err(error)) => error,
                        _ => "coverage preparation worker stopped".into(),
                    };
                    self.pending_coverage.as_mut().unwrap().error = Some(error.clone());
                    return Err(StitchCoreError::Config(error));
                }
                _ => self.pending_coverage = None,
            }
        }
        let (sender, result) = std::sync::mpsc::sync_channel(1);
        let task_calibration = calibration.clone();
        let task_size = size.clone();
        std::thread::Builder::new()
            .name("reco-coverage-prepare".into())
            .spawn(move || {
                let result = Controller::new(&task_calibration, &task_size, MapConfig::default());
                let _ = sender.send(result);
            })
            .map_err(|error| StitchCoreError::Config(error.to_string()))?;
        self.pending_coverage = Some(PendingCoverage {
            calibration: calibration.clone(),
            size: size.clone(),
            result,
            error: None,
        });
        Ok(None)
    }
}
