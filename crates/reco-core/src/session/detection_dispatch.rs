//! Detection entry points for [`StitchSession`].
//!
//! Each `detect_and_update_world_pose_*` variant wraps its residency's
//! handles into per-camera [`DetectorFrame`]s and hands them to the
//! engine ([`StitchCore::run_detection_frames`]), which owns the
//! detector, the schedule, panorama mapping, and the tracker/panner
//! chain. The session-side job is purely "wrap residency handles into
//! frames"; these wrappers live beside the platform residency state
//! they read and move wherever that state moves.

use super::StitchSession;
use crate::core::StitchCore;
use crate::detect::detector::DetectorFrame;
#[cfg(feature = "gpu")]
use crate::geometry::CameraIndex;
use crate::session::types::SessionError;
use crate::source::FrameSet;

/// Borrow native images without preparing model-specific pixels in the session.
#[cfg(all(feature = "gpu", target_os = "linux"))]
pub(super) fn nvmm_frames(
    cameras: &[crate::source::NvmmFrame; 2],
) -> [(CameraIndex, DetectorFrame<'_>); 2] {
    [
        (0, DetectorFrame::Nvmm(&cameras[0])),
        (1, DetectorFrame::Nvmm(&cameras[1])),
    ]
}

/// Per-camera detection input from D3D11 staging views.
#[cfg(all(feature = "gpu", target_os = "windows"))]
pub(super) fn wgpu_nv12_frames<'a>(
    views: &'a [(wgpu::TextureView, wgpu::TextureView)],
    sizes: &[(u32, u32)],
    rotations: [i32; 2],
) -> Vec<(CameraIndex, DetectorFrame<'a>)> {
    views
        .iter()
        .enumerate()
        .map(|(camera, (y_view, uv_view))| {
            (
                camera,
                DetectorFrame::WgpuNv12 {
                    y_view,
                    uv_view,
                    width: sizes[camera].0,
                    height: sizes[camera].1,
                    rotation: rotations[camera],
                },
            )
        })
        .collect()
}

/// Per-camera Metal CVPixelBuffer frames (backends import natively).
#[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
pub(super) fn metal_frames(
    cameras: &[crate::interop::metal::RetainedCVPixelBuffer],
) -> Vec<(CameraIndex, DetectorFrame<'_>)> {
    cameras
        .iter()
        .enumerate()
        .map(|(camera, frame)| {
            (
                camera,
                DetectorFrame::Metal {
                    cv_pixel_buffer: frame.as_ptr(),
                    width: frame.width(),
                    height: frame.height(),
                },
            )
        })
        .collect()
}

impl StitchSession {
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    pub(super) fn detect_cuda(
        &mut self,
        cameras: &[crate::interop::cuda_lease::CudaLease],
        capture_times_us: &[i64],
    ) -> Result<(), SessionError> {
        let frames: Vec<_> = if self.core.detector_prefers_cuda_nv12() {
            cameras
                .iter()
                .enumerate()
                .map(|(camera, lease)| (camera, DetectorFrame::CudaNv12(lease.cuda_frame())))
                .collect()
        } else {
            cameras
                .iter()
                .enumerate()
                .map(|(camera, lease)| {
                    let (y_view, uv_view) = lease.views();
                    let frame = lease.cuda_frame();
                    (
                        camera,
                        DetectorFrame::WgpuNv12 {
                            y_view,
                            uv_view,
                            width: frame.width,
                            height: frame.height,
                            rotation: frame.rotation,
                        },
                    )
                })
                .collect()
        };
        self.core.run_detection_frames(&frames, capture_times_us)?;
        Ok(())
    }

    /// Shared detection skeleton: gate by the engine's schedule, let
    /// the closure feed the engine's detector, then drive the
    /// tracker/panner chain.
    ///
    /// Every `detect_and_update_world_pose_*` variant is a one-liner
    /// wrapper that passes a closure here. Adding a new detection
    /// backend means writing one closure, not copying 15 lines.
    fn detect_and_update_world_pose_with(
        &mut self,
        elapsed: std::time::Duration,
        detect_fn: impl FnOnce(&mut StitchCore) -> Result<(), crate::core::types::StitchCoreError>,
    ) -> Result<(), SessionError> {
        let due = self.core.detection_due(self.frame_count);
        if due {
            detect_fn(&mut self.core)?;
        }
        self.update_world_pose(elapsed)
    }

    /// Run detection + trackers only (no panner). For the lookahead
    /// produce phase where we want the WorldState but don't want to
    /// advance the panner.
    pub(crate) fn detect_and_track_only(
        &mut self,
        frame: &FrameSet,
        elapsed: std::time::Duration,
        produce_index: u64,
        capture: Option<&crate::source::CaptureTiming>,
        frame_seconds: f32,
    ) -> Result<
        (
            crate::detect::tracker::WorldState,
            crate::telemetry::FrameTiming,
        ),
        SessionError,
    > {
        let expiry_t0 = std::time::Instant::now();
        self.core.expire_observations(capture, frame_seconds)?;
        let expiry_time = expiry_t0.elapsed();
        let capture_times_us = capture.map_or(&[][..], |capture| capture.times_us.as_slice());
        let due = self.core.detection_due(produce_index);
        let detect_t0 = std::time::Instant::now();
        if due {
            match frame {
                #[cfg(all(feature = "gpu", target_os = "linux"))]
                FrameSet::GpuResident(cameras) => self.detect_cuda(cameras, capture_times_us)?,
                #[cfg(all(feature = "gpu", target_os = "windows"))]
                FrameSet::D3d11Resident(..) => {
                    let views = self
                        .gpu_exec_ref()
                        .d3d11_slot(produce_index)
                        .and_then(|slot| self.gpu_exec_ref().d3d11_views(slot));
                    if let Some(views) = views {
                        crate::profile_scope!("detect_wgpu_nv12");
                        let frames = wgpu_nv12_frames(
                            &views,
                            self.core.source_sizes(),
                            [self.left_rotation, self.right_rotation],
                        );
                        self.core.run_detection_frames(&frames, capture_times_us)?;
                    }
                }
                #[cfg(all(feature = "gpu", any(target_os = "macos", target_os = "ios")))]
                FrameSet::MetalResident(cameras) => {
                    let frames = metal_frames(cameras);
                    self.core.run_detection_frames(&frames, capture_times_us)?;
                }
                #[cfg(all(feature = "gpu", target_os = "linux"))]
                FrameSet::NvmmResident(cameras) => {
                    let frames = nvmm_frames(cameras);
                    self.core.run_detection_frames(&frames, capture_times_us)?;
                }
                _ => {
                    self.core.run_cpu_detection(frame, capture_times_us)?;
                }
            }
        }

        let detection = if due {
            detect_t0.elapsed()
        } else {
            std::time::Duration::ZERO
        };
        let tracking_t0 = std::time::Instant::now();
        let world = self
            .core
            .update_trackers(produce_index, elapsed.as_secs_f64() * 1000.0);
        Ok((
            world,
            crate::telemetry::FrameTiming {
                detection: Some(detection),
                tracking: Some(expiry_time + tracking_t0.elapsed()),
                ..Default::default()
            },
        ))
    }

    /// Run detection on a CPU-resident frame set (YUV420P / NV12).
    /// Capture timing follows [`crate::source::SourceFrame::capture`]. `frame_seconds` is the nominal media frame duration, used to bridge scheduled detection gaps, not processing elapsed time.
    pub fn detect_and_update_world_pose(
        &mut self,
        frame: &FrameSet,
        capture: Option<&crate::source::CaptureTiming>,
        frame_seconds: f32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.core.expire_observations(capture, frame_seconds)?;
        let capture_times_us = capture.map_or(&[][..], |capture| capture.times_us.as_slice());
        self.detect_and_update_world_pose_with(elapsed, |core| {
            core.run_cpu_detection(frame, capture_times_us)
        })
    }

    /// Whether detection should run on the current frame.
    pub fn detection_due(&self) -> bool {
        self.core.detection_due(self.frame_count)
    }

    /// Run detection on CPU-resident RGBA frames.
    pub fn detect_and_update_world_pose_rgba(
        &mut self,
        left_rgba: &[u8],
        right_rgba: &[u8],
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_world_pose_with(elapsed, |core| {
            core.run_detection_frames(
                &[
                    (
                        0,
                        DetectorFrame::Rgba {
                            data: left_rgba,
                            width,
                            height,
                        },
                    ),
                    (
                        1,
                        DetectorFrame::Rgba {
                            data: right_rgba,
                            width,
                            height,
                        },
                    ),
                ],
                &[],
            )
        })
    }

    /// Run detection on CUDA-resident RGBA frames (Bayer zero-copy).
    #[cfg(all(feature = "gpu", any(target_os = "linux", target_os = "windows")))]
    #[allow(clippy::too_many_arguments)]
    pub fn detect_and_update_world_pose_cuda_rgba(
        &mut self,
        left_ptr: crate::interop::cuda::CUdeviceptr,
        left_pitch: usize,
        right_ptr: crate::interop::cuda::CUdeviceptr,
        right_pitch: usize,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_world_pose_with(elapsed, |core| {
            core.run_detection_frames(
                &[
                    (
                        0,
                        DetectorFrame::CudaRgba {
                            ptr: left_ptr,
                            pitch: left_pitch,
                            width,
                            height,
                        },
                    ),
                    (
                        1,
                        DetectorFrame::CudaRgba {
                            ptr: right_ptr,
                            pitch: right_pitch,
                            width,
                            height,
                        },
                    ),
                ],
                &[],
            )
        })
    }

    /// Update the requested world pose from cached detections and record tracking telemetry.
    ///
    /// With a panner attached, runs trackers and asks it for a pose. Without one, neither tracking nor the stored request advances. Does not run inference, resolve coverage or render a frame.
    pub fn update_world_pose(&mut self, elapsed: std::time::Duration) -> Result<(), SessionError> {
        let stats = self.core.dispatch_pose(
            self.frame_count,
            elapsed.as_secs_f64() * 1000.0,
            "StitchSession",
        );
        self.telemetry
            .record_detections(stats.detections, stats.active_tracks, stats.ball_present);
        Ok(())
    }
}
