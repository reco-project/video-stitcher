//! Composite detector that preprocesses wgpu NV12 views on the GPU.
//!
//! Wraps any `UnifiedDetector` (typically `CpuYoloDetector` with
//! DirectML EP) and handles `DetectorFrame::WgpuNv12` by running
//! the `WgpuPreprocessor` compute shader before delegating to the
//! inner detector with `DetectorFrame::PreprocessedChw`.

use reco_core::detect::detector::{
    CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector,
};
use reco_detect::wgpu_preprocess::WgpuPreprocessor;

/// Detector wrapper that adds wgpu NV12 preprocessing.
///
/// Created by `setup_autocam` on Windows when CUDA detection is
/// unavailable (Pascal, AMD, Intel). The inner detector handles
/// `PreprocessedChw` (skipping its own preprocessing entirely).
pub struct WgpuPreprocessingDetector {
    inner: Box<dyn UnifiedDetector>,
    preprocessor: WgpuPreprocessor,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuPreprocessingDetector {
    /// Wrap a detector with wgpu NV12 preprocessing.
    pub fn new(
        inner: Box<dyn UnifiedDetector>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        input_size: u32,
        frame_width: u32,
        frame_height: u32,
    ) -> Self {
        let preprocessor =
            WgpuPreprocessor::new(&device, &queue, input_size, frame_width, frame_height);
        Self {
            inner,
            preprocessor,
            device,
            queue,
        }
    }
}

impl UnifiedDetector for WgpuPreprocessingDetector {
    fn name(&self) -> &'static str {
        "wgpu-preprocess"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        match frame {
            DetectorFrame::WgpuNv12 {
                y_view,
                uv_view,
                width,
                height,
                rotation,
            } => {
                let tensor = self.preprocessor.preprocess(
                    &self.device,
                    &self.queue,
                    y_view,
                    uv_view,
                    *rotation,
                );
                let inner_frame = DetectorFrame::PreprocessedChw {
                    data: &tensor,
                    input_size: self.preprocessor.input_size(),
                    src_width: *width,
                    src_height: *height,
                };
                self.inner.detect(camera, &inner_frame)
            }
            // Pass through other frame types to the inner detector
            other => self.inner.detect(camera, other),
        }
    }

    fn class_names(&self) -> Option<&[String]> {
        self.inner.class_names()
    }
}
