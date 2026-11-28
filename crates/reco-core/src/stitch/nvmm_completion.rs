//! Retire native camera inputs after their GPU reads, even while capture waits.

use crate::gpu::{GpuContext, wait_for_reads};
use crate::source::NvmmFrame;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::JoinHandle;

type Pending = [NvmmFrame; 2];

pub(crate) struct NvmmCompletion {
    device: wgpu::Device,
    sender: Option<SyncSender<Pending>>,
    worker: Option<JoinHandle<Result<(), String>>>,
}

impl NvmmCompletion {
    pub(crate) fn new(gpu: &GpuContext) -> Result<Self, String> {
        // At most two queued pairs plus the one being retired. Backpressure
        // bounds ownership without depending on further capture/submit calls.
        let (sender, receiver) = sync_channel::<Pending>(2);
        let device = gpu.device.clone();
        let worker = std::thread::Builder::new()
            .name("nvmm_completion".into())
            .spawn(move || {
                let mut result = Ok(());
                while let Ok(frames) = receiver.recv() {
                    if let Err(error) = wait_for_reads(&device) {
                        result = Err(error);
                    }
                    drop(frames);
                }
                result
            })
            .map_err(|e| format!("start NVMM completion worker: {e}"))?;
        Ok(Self {
            device: gpu.device.clone(),
            sender: Some(sender),
            worker: Some(worker),
        })
    }

    pub(crate) fn hold(&self, frames: [NvmmFrame; 2]) -> NvmmRead {
        NvmmRead {
            device: self.device.clone(),
            sender: self
                .sender
                .as_ref()
                .expect("active completion worker")
                .clone(),
            frames: Some(frames),
        }
    }

    pub(crate) fn finish(&mut self) -> Result<(), String> {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| "NVMM completion worker panicked".to_string())??;
        }
        Ok(())
    }
}

impl Drop for NvmmCompletion {
    fn drop(&mut self) {
        if let Err(error) = self.finish() {
            log::error!("{error}");
        }
    }
}

/// Scope covers render, sink errors and replay submission, including unwinding.
pub(crate) struct NvmmRead {
    device: wgpu::Device,
    sender: SyncSender<Pending>,
    frames: Option<[NvmmFrame; 2]>,
}

impl Drop for NvmmRead {
    fn drop(&mut self) {
        let frames = self.frames.take().expect("owned native inputs");
        // Every native read in this scope has been submitted before retirement.
        // The worker polls independently, including when the next capture blocks.
        if let Err(error) = self.sender.send(frames) {
            // A failed worker must not turn a failed send into early recycling.
            let frames = error.0;
            if let Err(error) = wait_for_reads(&self.device) {
                log::error!("{error}");
            }
            drop(frames);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::time::Duration;

    struct Recycle {
        completed: Arc<AtomicBool>,
        free: mpsc::Sender<()>,
    }
    impl Drop for Recycle {
        fn drop(&mut self) {
            assert!(
                self.completed.load(Ordering::Acquire),
                "recycled before GPU completion"
            );
            self.free.send(()).unwrap();
        }
    }

    #[test]
    #[ignore = "requires a GPU; lifetime-only inputs, no NVMM import"]
    fn nvmm_completion_progresses_with_one_capture_buffer() {
        let gpu = GpuContext::new_blocking().unwrap();
        let mut completion = NvmmCompletion::new(&gpu).unwrap();
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("completion regression"),
            size: 1024 * 1024,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (free, recycled) = mpsc::channel();
        for _ in 0..900 {
            let completed = Arc::new(AtomicBool::new(false));
            let frame = NvmmFrame::lifetime_test_frame(Recycle {
                completed: completed.clone(),
                free: free.clone(),
            });
            let reads = completion.hold([frame.clone(), frame]);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&buffer, 0, None);
            gpu.queue.submit([encoder.finish()]);
            gpu.queue
                .on_submitted_work_done(move || completed.store(true, Ordering::Release));
            drop(reads);
            // Simulate capture waiting for its sole buffer. No more submissions
            // or main-thread polling can drive completion during this wait.
            recycled.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        completion.finish().unwrap();
        completion.finish().unwrap();
    }

    #[test]
    #[ignore = "requires a GPU; lifetime-only inputs, no NVMM import"]
    fn nvmm_completion_drains_after_failed_frame_and_drop() {
        let gpu = GpuContext::new_blocking().unwrap();
        for explicit_finish in [false, true] {
            let mut completion = NvmmCompletion::new(&gpu).unwrap();
            let owner = Arc::new(());
            let weak = Arc::downgrade(&owner);
            let frame = NvmmFrame::lifetime_test_frame(owner);
            let submit_then_fail = || -> Result<(), &'static str> {
                let _reads = completion.hold([frame.clone(), frame.clone()]);
                gpu.queue.submit([]);
                Err("sink failed")
            };
            assert_eq!(submit_then_fail(), Err("sink failed"));
            drop(frame);
            if explicit_finish {
                completion.finish().unwrap();
            }
            drop(completion);
            assert!(weak.upgrade().is_none());
        }
    }
}
