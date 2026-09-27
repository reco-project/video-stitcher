//! Nonblocking coverage preparation shared by preview resize and calibration edits.

use super::{StitchCore, StitchCoreError};
use crate::{
    calibration::Calibration,
    projection::coverage::controller::{Controller, MapConfig},
    render::viewport::ViewportSize,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, TryRecvError},
};

pub(super) struct PendingCoverage {
    calibration: Calibration,
    size: ViewportSize,
    result: Receiver<Result<Controller, String>>,
    error: Option<String>,
    cancelled: Arc<AtomicBool>,
}

impl Drop for PendingCoverage {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
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

    /// Apply preview dimensions immediately, invalidating coverage when the aspect changes.
    ///
    /// Coverage is prepared separately by [`Self::prepare_coverage_preview`]. Same-aspect resizes preserve coverage and motion. Delivery allocation still happens on the calling thread. Recording consumers must retain their encoder dimensions.
    pub fn resize_preview(&mut self, width: u32, height: u32) -> Result<(), StitchCoreError> {
        let size = ViewportSize { width, height };
        size.validate()
            .map_err(|e| StitchCoreError::Config(e.to_string()))?;
        if (width, height) == self.output_dims() && self.executor.viewport_size() == &size {
            return Ok(());
        }
        let aspect_changed = self.executor.viewport_size().aspect_ratio() != size.aspect_ratio();
        self.apply_resize(width, height, None)
            .ok_or_else(|| StitchCoreError::Config("viewport resize rejected".into()))?;
        if aspect_changed {
            self.invalidate_preview_coverage();
        }
        Ok(())
    }

    pub(super) fn invalidate_preview_coverage(&mut self) {
        self.coverage_motion = None;
        if let Some(pending) = &self.pending_coverage {
            pending.cancelled.store(true, Ordering::Relaxed);
        }
    }

    pub(super) fn poll_coverage(
        &mut self,
        calibration: &Calibration,
        size: &ViewportSize,
    ) -> Result<Option<Controller>, StitchCoreError> {
        if let Some(pending) = &self.pending_coverage {
            let current = !pending.cancelled.load(Ordering::Relaxed)
                && pending.size.aspect_ratio() == size.aspect_ratio()
                && super::calibration_update::same_geometry(&pending.calibration, calibration);
            if !current {
                pending.cancelled.store(true, Ordering::Relaxed);
            }
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
        let cancelled = Arc::new(AtomicBool::new(false));
        let task_cancelled = Arc::clone(&cancelled);
        std::thread::Builder::new()
            .name("reco-coverage-prepare".into())
            .spawn(move || {
                // Coalesce continuous edits without occupying compute workers.
                for _ in 0..12 {
                    if task_cancelled.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                let result = Controller::new_cancellable(
                    &task_calibration,
                    &task_size,
                    MapConfig::default(),
                    Some(&task_cancelled),
                );
                let _ = sender.send(result);
            })
            .map_err(|error| StitchCoreError::Config(error.to_string()))?;
        self.pending_coverage = Some(PendingCoverage {
            calibration: calibration.clone(),
            size: size.clone(),
            result,
            error: None,
            cancelled,
        });
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stitch::{CpuExecutor, Executor};

    fn core() -> StitchCore {
        let cal = crate::stitch::test_support::calib(64, 32);
        let size = ViewportSize {
            width: 64,
            height: 32,
        };
        StitchCore::new(Executor::Cpu(Box::new(
            CpuExecutor::new(cal, size, 64, 32, false).unwrap(),
        )))
        .unwrap()
    }

    fn controlled_job(
        core: &mut StitchCore,
    ) -> std::sync::mpsc::SyncSender<Result<Controller, String>> {
        let (sender, result) = std::sync::mpsc::sync_channel(1);
        core.pending_coverage = Some(PendingCoverage {
            calibration: core.calibration().clone(),
            size: core.executor.viewport_size().clone(),
            result,
            error: None,
            cancelled: Arc::new(AtomicBool::new(false)),
        });
        sender
    }

    #[test]
    fn pending_result_failure_and_disconnect_are_not_success() {
        for disconnected in [false, true] {
            let mut core = core();
            let sender = controlled_job(&mut core);
            assert!(core.prepare_coverage_preview().unwrap().is_none());
            assert!(core.coverage_motion.is_none());
            if !disconnected {
                sender.send(Err("controlled failure".into())).unwrap();
            }
            drop(sender);
            let error = core.prepare_coverage_preview().unwrap_err().to_string();
            assert!(error.contains(if disconnected {
                "worker stopped"
            } else {
                "controlled failure"
            }));
            assert_eq!(
                core.prepare_coverage_preview().unwrap_err().to_string(),
                error
            );
            assert!(core.coverage_motion.is_none());
        }
    }

    #[test]
    fn completed_superseded_result_is_rejected_before_replacement() {
        let mut core = core();
        core.prepare_coverage().unwrap();
        let old = core.coverage_motion.take().unwrap();
        let sender = controlled_job(&mut core);
        let cancelled = core.pending_coverage.as_ref().unwrap().cancelled.clone();
        sender.send(Ok(old)).unwrap();
        core.resize_preview(64, 64).unwrap();
        assert!(cancelled.load(Ordering::Relaxed));
        assert!(core.prepare_coverage_preview().unwrap().is_none());
        assert!(core.coverage_motion.is_none());
        let current = core.pending_coverage.as_ref().unwrap();
        assert!(!Arc::ptr_eq(&cancelled, &current.cancelled));
        assert_eq!(
            current.size,
            ViewportSize {
                width: 64,
                height: 64
            }
        );
    }

    #[test]
    fn blend_edits_reuse_pending_job_and_cached_failure() {
        let mut core = core();
        core.prepare_coverage_preview().unwrap();
        let cancelled = Arc::clone(&core.pending_coverage.as_ref().unwrap().cancelled);
        let mut cal = core.calibration().clone();
        cal.projection.l_shape_mut().unwrap().blend_width *= 0.5;
        core.update_calibration_preview(cal).unwrap();
        core.prepare_coverage_preview().unwrap();
        assert!(Arc::ptr_eq(
            &cancelled,
            &core.pending_coverage.as_ref().unwrap().cancelled
        ));
        assert!(!cancelled.load(Ordering::Relaxed));

        core.pending_coverage.as_mut().unwrap().error = Some("unavailable".into());
        let mut cal = core.calibration().clone();
        cal.projection.l_shape_mut().unwrap().blend_width *= 0.5;
        core.update_calibration_preview(cal).unwrap();
        assert!(core.prepare_coverage_preview().is_err());
        assert!(Arc::ptr_eq(
            &cancelled,
            &core.pending_coverage.as_ref().unwrap().cancelled
        ));
        assert!(!cancelled.load(Ordering::Relaxed));
        let mut cal = core.calibration().clone();
        cal.framing.roll += 0.1;
        core.update_calibration_preview(cal).unwrap();
        assert!(cancelled.load(Ordering::Relaxed));
        assert!(core.prepare_coverage_preview().unwrap().is_none());
    }

    #[test]
    fn same_aspect_resize_reuses_pending_job_and_drop_cancels_it() {
        let mut core = core();
        assert!(core.prepare_coverage_preview().unwrap().is_none());
        let cancelled = Arc::clone(&core.pending_coverage.as_ref().unwrap().cancelled);
        core.resize_preview(128, 64).unwrap();
        assert!(core.prepare_coverage_preview().unwrap().is_none());
        assert!(Arc::ptr_eq(
            &cancelled,
            &core.pending_coverage.as_ref().unwrap().cancelled
        ));
        assert!(!cancelled.load(Ordering::Relaxed));
        drop(core);
        assert!(cancelled.load(Ordering::Relaxed));
    }

    #[test]
    fn invalidation_cancels_before_replacement_and_never_publishes_old_result() {
        let mut core = core();
        core.prepare_coverage_preview().unwrap();
        let cancelled = Arc::clone(&core.pending_coverage.as_ref().unwrap().cancelled);
        core.resize_preview(64, 64).unwrap();
        assert!(cancelled.load(Ordering::Relaxed));
        assert!(core.coverage_max_fov_degrees().is_none());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            core.prepare_coverage_preview().unwrap();
            let pending = core.pending_coverage.as_ref().unwrap();
            if !Arc::ptr_eq(&cancelled, &pending.cancelled) {
                assert_eq!(
                    pending.size,
                    ViewportSize {
                        width: 64,
                        height: 64
                    }
                );
                assert!(!pending.cancelled.load(Ordering::Relaxed));
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
    }
}
