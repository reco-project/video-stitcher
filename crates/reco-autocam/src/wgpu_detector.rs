//! Composite detector that preprocesses wgpu NV12 views on the GPU.
//!
//! Wraps any `Detector` (typically `CpuYoloDetector` with
//! DirectML EP) and handles `DetectorFrame::WgpuNv12` by running
//! the `WgpuPreprocessor` compute shader before delegating to the
//! inner detector with `DetectorFrame::PreprocessedChw`.

use reco_core::detect::detector::{Detection, Detector, DetectorError, DetectorFrame};
use reco_core::geometry::CameraIndex;
use reco_detect::wgpu_preprocess::WgpuPreprocessor;

#[cfg(test)]
mod tests;

/// Detector wrapper that adds wgpu NV12 preprocessing.
///
/// Created by `setup_autocam` on Windows when CUDA detection is
/// unavailable (Pascal, AMD, Intel). The inner detector handles
/// `PreprocessedChw` (skipping its own preprocessing entirely).
pub struct WgpuPreprocessingDetector {
    inner: Box<dyn Detector>,
    preprocessor: WgpuPreprocessor,
    device: wgpu::Device,
    queue: wgpu::Queue,
    source_sizes: Vec<(u32, u32)>,
}

impl WgpuPreprocessingDetector {
    /// Wrap one detector with reusable wgpu NV12/P010 preprocessing for the declared cameras.
    /// Rejects empty/odd source dimensions and unsupported model tensor sizes.
    pub fn new(
        inner: Box<dyn Detector>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        input_size: u32,
        source_sizes: Vec<(u32, u32)>,
    ) -> Result<Self, DetectorError> {
        if source_sizes.is_empty()
            || source_sizes
                .iter()
                .any(|&(w, h)| w == 0 || h == 0 || !w.is_multiple_of(2) || !h.is_multiple_of(2))
        {
            return Err(DetectorError::InferenceFailed(
                "wgpu detection requires nonzero, even camera dimensions".into(),
            ));
        }
        let preprocessor = WgpuPreprocessor::new(&device, input_size)?;
        Ok(Self {
            inner,
            preprocessor,
            device,
            queue,
            source_sizes,
        })
    }
}

impl Detector for WgpuPreprocessingDetector {
    fn name(&self) -> &'static str {
        "wgpu-preprocess"
    }

    fn detect(
        &mut self,
        camera: CameraIndex,
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
                if self.source_sizes.get(camera) != Some(&(*width, *height)) {
                    return Err(DetectorError::InferenceFailed(format!(
                        "camera {camera} has undeclared source dimensions {width}x{height}"
                    )));
                }
                let input_size = self.preprocessor.input_size();
                let tensor = self.preprocessor.preprocess(
                    &self.device,
                    &self.queue,
                    (y_view, uv_view),
                    (*width, *height),
                    *rotation,
                )?;
                let inner_frame = DetectorFrame::PreprocessedChw {
                    data: tensor,
                    input_size,
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
