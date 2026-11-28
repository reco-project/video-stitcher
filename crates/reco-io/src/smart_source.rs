//! GPU-aware file source with automatic backend selection.
//!
//! [`SmartFileSource`] is the standard way to open video files for stitching.
//! It probes the input, detects the best decode path (GPU zero-copy or CPU),
//! and delivers frames through the [`FrameSource`] trait.
//!
//! On capable hardware (NVIDIA + Vulkan, Apple VideoToolbox + Metal), frames
//! are GPU-resident - no CPU-GPU transfer needed. On other hardware, frames
//! are decoded to CPU transparently.
//!
//! # Example
//!
//! ```rust,ignore
//! use reco_io::SmartFileSource;
//!
//! let gpu = reco_core::gpu::GpuContext::new_blocking()?;
//! let inputs = [reco_io::InputPath::Single("panorama.mp4".into())];
//! let mut source = SmartFileSource::open(&inputs, &gpu, 0)?;
//! // source implements FrameSource - pass to session.run()
//! ```

#[cfg(feature = "gpu")]
use reco_core::render::renderer::GpuPixelFormat;
use reco_core::source::{FrameSource, SourceError, SourceFrame, SourceInfo};

#[cfg(all(feature = "gpu", target_os = "linux"))]
mod linux;
#[cfg(all(feature = "gpu", target_os = "linux"))]
use linux as platform;

#[cfg(all(feature = "gpu", target_os = "macos"))]
mod macos;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use macos as platform;

#[cfg(all(feature = "gpu", target_os = "windows"))]
mod windows;
#[cfg(all(feature = "gpu", target_os = "windows"))]
use windows as platform;

#[cfg(all(
    feature = "gpu",
    not(any(target_os = "linux", target_os = "macos", target_os = "windows"))
))]
mod unsupported;
#[cfg(all(
    feature = "gpu",
    not(any(target_os = "linux", target_os = "macos", target_os = "windows"))
))]
use unsupported as platform;

#[cfg(all(test, feature = "gpu", any(target_os = "windows", target_os = "macos")))]
mod native_tests;

#[cfg(feature = "gpu")]
pub(super) struct SkipOutcome {
    pub(super) skipped: u64,
    pub(super) exhausted: bool,
}

/// GPU-aware file source that auto-selects the optimal decode path.
///
/// Probes the input files at construction and selects the best available
/// backend: CUDA/Vulkan zero-copy on Linux, D3D11VA/DX12 on Windows, VideoToolbox/Metal
/// on macOS, or CPU software decode as a fallback. The selection is
/// transparent to the consumer.
///
/// For GPU zero-copy paths, this source owns the shared textures and
/// decode threads. The session creates bind groups from the source's
/// textures at the start of `run()`.
pub struct SmartFileSource {
    mode: SourceMode,
    info: SourceInfo,
    #[cfg(feature = "gpu")]
    pixel_format: GpuPixelFormat,
    full_range: bool,
    matrix: reco_core::render::color::YuvMatrix,
    color_metadata: crate::ffmpeg::ColorMetadata,
    left_rotation: i32,
    right_rotation: i32,
    /// Human-readable description of the active decode path.
    decode_mode: &'static str,
    /// True once the active backend has signaled end-of-stream.
    ///
    /// For the CPU mode this tracks the inner `FfmpegFileSource`'s flag;
    /// for zero-copy modes this is set by an explicit EOF message.
    exhausted: bool,
    terminal_error: Option<SourceError>,
}

enum SourceMode {
    Cpu(Box<crate::adapters::FfmpegFileSource>),
    #[cfg(feature = "gpu")]
    Native(Box<platform::NativeSource>),
}

impl SmartFileSource {
    /// Open a file source with automatic backend selection.
    ///
    /// Probes input metadata and chooses native residency when the decoder,
    /// GPU and input layout support it. CUDA/Vulkan supports one or two
    /// inputs; D3D11 and Metal imports require two.
    ///
    /// `sync_offset` applies temporal alignment between cameras:
    /// positive skips camera 1, negative skips camera 0. Ignored for mono.
    #[cfg(feature = "gpu")]
    pub fn open(
        inputs: &[crate::stitch_job::InputPath],
        gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
    ) -> Result<Self, SourceError> {
        let Some(left) = inputs.first() else {
            return Err(SourceError::Init {
                path: "(none)".into(),
                reason: "at least one input video is required".into(),
            });
        };
        let left_probe_path = left.first_path();

        let probe = crate::ffmpeg::decoder::VideoDecoder::open(left_probe_path).map_err(|e| {
            SourceError::Init {
                path: left_probe_path.display().to_string(),
                reason: format!("{e}"),
            }
        })?;

        let fps_r = probe.frame_rate();
        let mut info = SourceInfo {
            camera_sizes: vec![(probe.width(), probe.height())],
            fps: probe.fps().unwrap_or(0.0),
            fps_rational: fps_r.map(|r| (r.0, r.1)),
            total_frames: None,
        };
        let pixel_format = probe.pixel_format();
        let full_range = probe.is_full_range();
        let first_color = probe.color_metadata();
        let left_rotation = probe.rotation();
        let decode_backend = probe.backend();
        drop(probe);

        let mut right_rotation = 0;
        let mut matching_layout = true;
        let mut camera_colors = vec![first_color];
        for (camera, input) in inputs.iter().enumerate().skip(1) {
            let probe =
                crate::ffmpeg::decoder::VideoDecoder::open(input.first_path()).map_err(|e| {
                    SourceError::Init {
                        path: input.first_path().display().to_string(),
                        reason: e.to_string(),
                    }
                })?;
            if probe.is_full_range() != full_range {
                return Err(SourceError::Init {
                    path: input.first_path().display().to_string(),
                    reason: "camera color ranges differ; mixed full-range and limited-range inputs are not supported".into(),
                });
            }
            crate::adapters::merge_frame_rate(&mut info, probe.frame_rate(), input.first_path())?;
            matching_layout &= (platform::NativeSource::allows_mixed_camera_sizes()
                || (probe.width() == info.camera_sizes[0].0
                    && probe.height() == info.camera_sizes[0].1))
                && probe.pixel_format() == pixel_format
                && platform::NativeSource::supports_backend(probe.backend());
            info.camera_sizes.push((probe.width(), probe.height()));
            if camera == 1 {
                right_rotation = probe.rotation();
            }
            camera_colors.push(probe.color_metadata());
        }
        let (color_metadata, matrix) = crate::ffmpeg::ColorMetadata::reconcile(&camera_colors)
            .map_err(|reason| SourceError::Init {
                path: left_probe_path.display().to_string(),
                reason,
            })?;
        // Automatic Windows/Metal native selection is restricted to validated layouts.
        let native_input_count = platform::NativeSource::supports_input_count(inputs.len());

        // Detect zero-copy capability
        let hwaccel_disabled = std::env::var("RECO_NO_HWACCEL").is_ok();
        if hwaccel_disabled {
            log::info!("RECO_NO_HWACCEL set, forcing CPU decode path");
        }
        let backend_capable = platform::NativeSource::supports_backend(decode_backend);
        let gpu_capable = gpu.supports_zero_copy();
        let use_zero_copy = !hwaccel_disabled
            && backend_capable
            && gpu_capable
            && matching_layout
            && native_input_count;

        if !use_zero_copy && !hwaccel_disabled {
            log::info!(
                "Zero-copy disabled: decode_backend={decode_backend} (capable={backend_capable}), \
                 gpu_zero_copy={gpu_capable}, matching_layout={matching_layout}, \
                 supported_input_count={native_input_count}. Using CPU upload path."
            );
        }

        if use_zero_copy {
            info.total_frames = crate::adapters::input_frame_count(inputs, sync_offset);
            Self::open_zero_copy(
                inputs,
                gpu,
                sync_offset,
                info,
                pixel_format,
                full_range,
                (color_metadata, matrix),
                left_rotation,
                right_rotation,
            )
        } else {
            Self::open_cpu_only(inputs, sync_offset, false)
        }
    }

    /// Open host-backed YUV420P frames for any number of inputs in projection order. Hardware decoding can still be used unless `software_decode` is true. Format, fps and color range are probed from input 0; rotation is probed per input.
    pub fn open_cpu_only(
        inputs: &[crate::stitch_job::InputPath],
        sync_offset: i64,
        software_decode: bool,
    ) -> Result<Self, SourceError> {
        let source =
            crate::adapters::FfmpegFileSource::open_inputs(inputs, sync_offset, software_decode)?;
        let info = source.info();
        #[cfg(feature = "gpu")]
        let pixel_format = source.pixel_format();
        let full_range = source.is_full_range();
        let color_metadata = source.color_metadata();
        let matrix = source.yuv_matrix();
        log::info!(
            "SmartFileSource: host-backed frames ({} camera(s), {}x{}{})",
            inputs.len(),
            info.camera_sizes[0].0,
            info.camera_sizes[0].1,
            if full_range { ", full-range" } else { "" }
        );
        let left_rotation = source.rotation(0);
        let right_rotation = source.rotation(1);
        Ok(Self {
            mode: SourceMode::Cpu(Box::new(source)),
            info,
            #[cfg(feature = "gpu")]
            pixel_format,
            full_range,
            matrix,
            color_metadata,
            left_rotation,
            right_rotation,
            decode_mode: "Host YUV420P",
            exhausted: false,
            terminal_error: None,
        })
    }

    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn open_zero_copy(
        inputs: &[crate::stitch_job::InputPath],
        gpu: &reco_core::gpu::GpuContext,
        sync_offset: i64,
        info: SourceInfo,
        pixel_format: GpuPixelFormat,
        full_range: bool,
        (color_metadata, matrix): (
            crate::ffmpeg::ColorMetadata,
            reco_core::render::color::YuvMatrix,
        ),
        left_rotation: i32,
        right_rotation: i32,
    ) -> Result<Self, SourceError> {
        let Some(native) = platform::NativeSource::try_open(
            inputs,
            gpu,
            sync_offset,
            &info,
            pixel_format,
            full_range,
        )?
        else {
            return Self::open_cpu_only(inputs, sync_offset, false);
        };
        let decode_mode = native.decode_mode();
        Ok(Self {
            mode: SourceMode::Native(Box::new(native)),
            info,
            pixel_format,
            full_range,
            matrix,
            color_metadata,
            left_rotation,
            right_rotation,
            decode_mode,
            exhausted: false,
            terminal_error: None,
        })
    }

    /// Human-readable description of the active decode path.
    pub fn decode_mode(&self) -> &'static str {
        self.decode_mode
    }

    /// Reconciled source colourimetry metadata across all cameras.
    pub fn color_metadata(&self) -> crate::ffmpeg::ColorMetadata {
        self.color_metadata
    }

    /// Access the shared texture set (GPU zero-copy only).
    ///
    /// The session uses this to create bind groups at the start of `run()`.
    /// Returns `None` for CPU-mode sources.
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    pub fn shared_texture_set(&self) -> Option<&reco_core::interop::SharedTextureSet> {
        match &self.mode {
            SourceMode::Native(native) => Some(native.shared_texture_set()),
            SourceMode::Cpu(_) => None,
        }
    }
}

impl FrameSource for SmartFileSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn total_frames(&self) -> Option<u64> {
        self.info.total_frames
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if self.exhausted {
            return Ok(None);
        }
        match &mut self.mode {
            SourceMode::Cpu(source) => {
                let frame = source.next_frame()?;
                if frame.is_none() {
                    self.exhausted = true;
                }
                Ok(frame)
            }
            #[cfg(feature = "gpu")]
            SourceMode::Native(native) => {
                match native
                    .next_frame()
                    .inspect_err(|error| self.terminal_error = Some(error.clone()))?
                {
                    Some(frame) => Ok(Some(frame)),
                    None => {
                        self.exhausted = true;
                        Ok(None)
                    }
                }
            }
        }
    }

    fn is_gpu_resident(&self) -> bool {
        !matches!(self.mode, SourceMode::Cpu(_))
    }

    fn skip_frames(&mut self, count: u64) -> Result<u64, SourceError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if self.exhausted {
            return Ok(0);
        }
        match &mut self.mode {
            SourceMode::Cpu(source) => {
                source.seek(count)?;
                log::debug!("CPU seek: jumped to frame {count}");
                Ok(count)
            }
            #[cfg(feature = "gpu")]
            SourceMode::Native(native) => {
                let outcome = native
                    .skip_frames(count, &self.info)
                    .inspect_err(|error| self.terminal_error = Some(error.clone()))?;
                if outcome.exhausted {
                    self.exhausted = true;
                }
                Ok(outcome.skipped)
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn gpu_pixel_format(&self) -> GpuPixelFormat {
        self.pixel_format
    }

    fn yuv_matrix(&self) -> reco_core::render::color::YuvMatrix {
        self.matrix
    }

    fn is_full_range(&self) -> bool {
        self.full_range
    }

    fn left_rotation(&self) -> i32 {
        self.left_rotation
    }

    fn right_rotation(&self) -> i32 {
        self.right_rotation
    }

    fn is_exhausted(&self) -> bool {
        // CPU mode also honors the inner source's exhaustion state.
        // SmartFileSource exposes blocking reads, not polling.
        match &self.mode {
            SourceMode::Cpu(source) => self.exhausted || source.is_exhausted(),
            #[cfg(feature = "gpu")]
            SourceMode::Native(_) => self.exhausted,
        }
    }

    fn start_decoding(&mut self) {
        #[cfg(feature = "gpu")]
        if let SourceMode::Native(native) = &mut self.mode {
            native.start_decoding();
        }
    }
}

/// Compute the absolute `(target_frame, target_secs)` for a fast keyframe seek,
/// or `None` when the caller should fall back to decode-and-discard (zero skip,
/// unknown/non-positive/non-finite frame rate, or overflowing timestamp).
#[cfg(any(test, all(feature = "gpu", target_os = "windows")))]
fn skip_seek_target(current_frame: u64, count: u64, info: &SourceInfo) -> Option<(u64, f64)> {
    if count == 0 || info.fps_rational.is_none() || !info.fps.is_finite() || info.fps <= 0.0 {
        return None;
    }
    let target_frame = current_frame.saturating_add(count);
    let secs = target_frame as f64 / info.fps;
    secs.is_finite().then_some((target_frame, secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info_with_fps(fps: f64, fps_rational: Option<(i32, i32)>) -> SourceInfo {
        SourceInfo {
            camera_sizes: vec![(1920, 1080); 2],
            fps,
            fps_rational,
            total_frames: None,
        }
    }

    #[test]
    fn skip_seek_target_computes_absolute_frame_and_seconds() {
        let info = info_with_fps(30.0, Some((30, 1)));
        assert_eq!(skip_seek_target(0, 9000, &info), Some((9000, 300.0)));
        assert_eq!(skip_seek_target(15, 45, &info), Some((60, 2.0)));
    }

    #[test]
    fn skip_seek_target_rejects_zero_and_invalid_frame_rates() {
        let valid = info_with_fps(30.0, Some((30, 1)));
        assert_eq!(skip_seek_target(0, 0, &valid), None);
        assert_eq!(skip_seek_target(10, 0, &valid), None);

        for invalid in [
            info_with_fps(0.0, None),
            info_with_fps(30.0, None),
            info_with_fps(0.0, Some((30, 1))),
            info_with_fps(-30.0, Some((30, 1))),
            info_with_fps(f64::NAN, Some((30, 1))),
            info_with_fps(f64::INFINITY, Some((30, 1))),
            info_with_fps(f64::MIN_POSITIVE, Some((1, 1))),
        ] {
            let count = if invalid.fps == f64::MIN_POSITIVE {
                u64::MAX
            } else {
                100
            };
            assert_eq!(skip_seek_target(0, count, &invalid), None);
        }
    }
}
