//! Detection entry points for [`StitchSession`].
//!
//! Each `detect_and_update_director_*` variant wraps its residency's
//! handles into per-camera [`DetectorFrame`]s and hands them to the
//! engine ([`StitchCore::run_detection_frames`]), which owns the
//! detector, the schedule, panorama mapping, and the tracker/panner
//! chain. The session-side job is purely "wrap residency handles into
//! frames"; these wrappers live beside the platform residency state
//! they read and move wherever that state moves.

use super::StitchSession;
use crate::core::StitchCore;
use crate::detect::detector::DetectorFrame;
use crate::geometry::CameraIndex;
use crate::session::types::SessionError;
use crate::source::FrameSet;

/// Per-camera CPU frames for a CPU-resident two-camera set; `None` for
/// GPU-resident variants (their arms build residency-specific frames)
/// and for set shapes without a detection mapping (mono).
fn cpu_frames<'a>(
    frame: &'a FrameSet,
    width: u32,
    height: u32,
) -> Option<[(CameraIndex, DetectorFrame<'a>); 2]> {
    use crate::detect::detector::{ChromaFormat, RawFrame};
    match frame {
        FrameSet::Yuv420p(cams) => match cams.as_slice() {
            [left, right] => Some([
                (
                    0,
                    DetectorFrame::Cpu(RawFrame {
                        y: &left.y,
                        chroma: ChromaFormat::Yuv420p {
                            u: &left.u,
                            v: &left.v,
                        },
                        width,
                        height,
                    }),
                ),
                (
                    1,
                    DetectorFrame::Cpu(RawFrame {
                        y: &right.y,
                        chroma: ChromaFormat::Yuv420p {
                            u: &right.u,
                            v: &right.v,
                        },
                        width,
                        height,
                    }),
                ),
            ]),
            _ => None,
        },
        FrameSet::Nv12(cams) => match cams.as_slice() {
            [left, right] => Some([
                (
                    0,
                    DetectorFrame::Cpu(RawFrame {
                        y: &left.y,
                        chroma: ChromaFormat::Nv12 { uv: &left.uv },
                        width,
                        height,
                    }),
                ),
                (
                    1,
                    DetectorFrame::Cpu(RawFrame {
                        y: &right.y,
                        chroma: ChromaFormat::Nv12 { uv: &right.uv },
                        width,
                        height,
                    }),
                ),
            ]),
            _ => None,
        },
        _ => None,
    }
}

/// Per-camera CUDA NV12 frames from the shared-texture slot pointers.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(super) fn cuda_nv12_frames(
    left_buf: &crate::interop::zero_copy::GpuBufInfo,
    right_buf: &crate::interop::zero_copy::GpuBufInfo,
    left_slot: u8,
    right_slot: u8,
    left_rotation: i32,
    right_rotation: i32,
) -> [(CameraIndex, DetectorFrame<'static>); 2] {
    use crate::detect::detector::GpuNv12Frame;
    let ls = left_slot as usize;
    let rs = right_slot as usize;
    let is_10bit = left_buf.pixel_format == crate::render::renderer::GpuPixelFormat::P010;
    [
        (
            0,
            DetectorFrame::Cuda(GpuNv12Frame {
                y_ptr: left_buf.y_ptr[ls],
                uv_ptr: left_buf.uv_ptr[ls],
                y_pitch: left_buf.y_pitch[ls],
                uv_pitch: left_buf.uv_pitch[ls],
                width: left_buf.width,
                height: left_buf.height,
                rotation: left_rotation,
                is_10bit,
            }),
        ),
        (
            1,
            DetectorFrame::Cuda(GpuNv12Frame {
                y_ptr: right_buf.y_ptr[rs],
                uv_ptr: right_buf.uv_ptr[rs],
                y_pitch: right_buf.y_pitch[rs],
                uv_pitch: right_buf.uv_pitch[rs],
                width: right_buf.width,
                height: right_buf.height,
                rotation: right_rotation,
                is_10bit,
            }),
        ),
    ]
}

/// Per-camera wgpu NV12 texture-view frames (shared decode textures /
/// D3D11 staging views).
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn wgpu_nv12_frames<'a>(
    left_y: &'a wgpu::TextureView,
    left_uv: &'a wgpu::TextureView,
    right_y: &'a wgpu::TextureView,
    right_uv: &'a wgpu::TextureView,
    width: u32,
    height: u32,
    left_rotation: i32,
    right_rotation: i32,
) -> [(CameraIndex, DetectorFrame<'a>); 2] {
    [
        (
            0,
            DetectorFrame::WgpuNv12 {
                y_view: left_y,
                uv_view: left_uv,
                width,
                height,
                rotation: left_rotation,
            },
        ),
        (
            1,
            DetectorFrame::WgpuNv12 {
                y_view: right_y,
                uv_view: right_uv,
                width,
                height,
                rotation: right_rotation,
            },
        ),
    ]
}

/// Per-camera Metal CVPixelBuffer frames (backends import natively).
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) fn metal_frames(
    left_cvpb: crate::interop::metal::CVPixelBufferRef,
    right_cvpb: crate::interop::metal::CVPixelBufferRef,
    width: u32,
    height: u32,
) -> [(CameraIndex, DetectorFrame<'static>); 2] {
    [
        (
            0,
            DetectorFrame::Metal {
                cv_pixel_buffer: left_cvpb,
                width,
                height,
            },
        ),
        (
            1,
            DetectorFrame::Metal {
                cv_pixel_buffer: right_cvpb,
                width,
                height,
            },
        ),
    ]
}

impl StitchSession {
    /// Shared detection skeleton: gate by the engine's schedule, let
    /// the closure feed the engine's detector, then drive the
    /// tracker/panner chain.
    ///
    /// Every `detect_and_update_director_*` variant is a one-liner
    /// wrapper that passes a closure here. Adding a new detection
    /// backend means writing one closure, not copying 15 lines.
    fn detect_and_update_director_with(
        &mut self,
        elapsed: std::time::Duration,
        detect_fn: impl FnOnce(&mut StitchCore),
    ) -> Result<(), SessionError> {
        let due = self.core.detection_due(self.frame_count);
        if due {
            detect_fn(&mut self.core);
        }
        self.fire_sink_and_update_director(elapsed, due)
    }

    /// Run detection + trackers only (no panner). For the lookahead
    /// produce phase where we want the WorldState but don't want to
    /// advance the panner.
    pub(crate) fn detect_and_track_only(
        &mut self,
        frame: &FrameSet,
        elapsed: std::time::Duration,
        produce_index: u64,
    ) -> Result<crate::detect::tracker::WorldState, SessionError> {
        if self.core.detection_due(produce_index) {
            match frame {
                #[cfg(target_os = "linux")]
                FrameSet::GpuResident { slots } => {
                    if self.core.detector_needs_cuda_frames() {
                        let bufs = self.core.executor.gpu().and_then(|g| g.cuda_buf_info());
                        if let Some((left_buf, right_buf)) = bufs {
                            crate::profile_scope!("gpu_detect_total");
                            let frames = cuda_nv12_frames(
                                &left_buf,
                                &right_buf,
                                slots[0],
                                slots[1],
                                self.left_rotation,
                                self.right_rotation,
                            );
                            self.core.run_detection_frames(&frames);
                        }
                    } else if let Some(views) = self.shared_views() {
                        crate::profile_scope!("detect_wgpu_nv12");
                        let ls = slots[0] as usize;
                        let rs = slots[1] as usize;
                        let (w, h) = self.core.source_info();
                        let frames = wgpu_nv12_frames(
                            &views[ls * 2],
                            &views[ls * 2 + 1],
                            &views[4 + rs * 2],
                            &views[4 + rs * 2 + 1],
                            w,
                            h,
                            self.left_rotation,
                            self.right_rotation,
                        );
                        self.core.run_detection_frames(&frames);
                    }
                }
                #[cfg(target_os = "windows")]
                FrameSet::D3d11Resident(..) => {
                    let views = self
                        .gpu_exec_ref()
                        .d3d11_slots(produce_index)
                        .and_then(|(ls, rs)| self.gpu_exec_ref().d3d11_views(ls, rs));
                    if let Some(views) = views {
                        crate::profile_scope!("detect_wgpu_nv12");
                        let (w, h) = self.core.source_info();
                        let frames = wgpu_nv12_frames(
                            &views[0],
                            &views[1],
                            &views[2],
                            &views[3],
                            w,
                            h,
                            self.left_rotation,
                            self.right_rotation,
                        );
                        self.core.run_detection_frames(&frames);
                    }
                }
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                FrameSet::MetalResident([left, right]) => {
                    let frames =
                        metal_frames(left.as_ptr(), right.as_ptr(), left.width(), left.height());
                    self.core.run_detection_frames(&frames);
                }
                #[cfg(target_os = "linux")]
                FrameSet::NvmmResident([left, right]) => {
                    crate::profile_scope!("detect_preletterboxed_total");
                    if let Some(frames) = self.gpu_exec().nvmm_detector_frames(left, right) {
                        self.core.run_detection_frames(&frames);
                    }
                }
                _ => {
                    let (w, h) = self.core.source_info();
                    if let Some(frames) = cpu_frames(frame, w, h) {
                        self.core.run_detection_frames(&frames);
                    }
                }
            }
        }

        Ok(self
            .core
            .track_only(produce_index, elapsed.as_secs_f64() * 1000.0))
    }

    /// Allocate the NVMM detection surfaces for the Jetson zero-copy path.
    ///
    /// Call once before [`run`](Self::run) when feeding a
    /// `FrameSet::NvmmResident` source (mirrors
    /// [`setup_gpu_source`](Self::setup_gpu_source) for the desktop GPU
    /// path). `model_size` is the detector's square input dimension (e.g.
    /// 1280); `src_width`/`src_height` are the capture resolution, used to
    /// compute the letterbox geometry. Without this the NVMM detection arm
    /// no-ops (the director still advances, just without detections).
    #[cfg(target_os = "linux")]
    pub fn setup_nvmm_detection(
        &mut self,
        model_size: u32,
        src_width: u32,
        src_height: u32,
    ) -> Result<(), SessionError> {
        self.gpu_exec()
            .setup_nvmm_detection(model_size, src_width, src_height)
            .map_err(SessionError::ZeroCopy)
    }

    /// Run detection on a CPU-resident frame set (YUV420P / NV12).
    pub fn detect_and_update_director(
        &mut self,
        frame: &FrameSet,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let (w, h) = self.core.source_info();
        self.detect_and_update_director_with(elapsed, |core| {
            if let Some(frames) = cpu_frames(frame, w, h) {
                core.run_detection_frames(&frames);
            }
        })
    }

    /// Whether detection should run on the current frame.
    pub fn detection_should_run(&self) -> bool {
        self.core.detection_due(self.frame_count)
    }

    /// Run detection on CPU-resident RGBA frames.
    pub fn detect_and_update_director_rgba(
        &mut self,
        left_rgba: &[u8],
        right_rgba: &[u8],
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |core| {
            core.run_detection_frames(&[
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
            ])
        })
    }

    /// Run detection on CUDA-resident RGBA frames (Bayer zero-copy).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[allow(clippy::too_many_arguments)]
    pub fn detect_and_update_director_cuda_rgba(
        &mut self,
        left_ptr: crate::interop::cuda::CUdeviceptr,
        left_pitch: usize,
        right_ptr: crate::interop::cuda::CUdeviceptr,
        right_pitch: usize,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |core| {
            core.run_detection_frames(&[
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
            ])
        })
    }

    /// Detect on pre-letterboxed CUDA RGBA (NvBufSurfTransform output).
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn detect_and_update_director_preletterboxed(
        &mut self,
        left_ptr: crate::interop::cuda::CUdeviceptr,
        right_ptr: crate::interop::cuda::CUdeviceptr,
        src_width: u32,
        src_height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |core| {
            crate::profile_scope!("detect_preletterboxed_total");
            core.run_detection_frames(&[
                (
                    0,
                    DetectorFrame::CudaRgbaLetterboxed {
                        ptr: left_ptr,
                        src_width,
                        src_height,
                    },
                ),
                (
                    1,
                    DetectorFrame::CudaRgbaLetterboxed {
                        ptr: right_ptr,
                        src_width,
                        src_height,
                    },
                ),
            ])
        })
    }

    /// Update the director without detection.
    ///
    /// Advances the panner/tracker state without running object
    /// detection. Used when the frame residency has no detection
    /// backend (e.g. D3D11VA without CUDA).
    pub fn update_director(&mut self, elapsed: std::time::Duration) -> Result<(), SessionError> {
        self.fire_sink_and_update_director(elapsed, false)
    }

    /// Run GPU-resident detection from CUDA NV12 shared textures.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) fn detect_and_update_director_gpu(
        &mut self,
        left_buf: &crate::interop::zero_copy::GpuBufInfo,
        right_buf: &crate::interop::zero_copy::GpuBufInfo,
        left_slot: u8,
        right_slot: u8,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        let lr = self.left_rotation;
        let rr = self.right_rotation;
        self.detect_and_update_director_with(elapsed, |core| {
            crate::profile_scope!("gpu_detect_total");
            let frames = cuda_nv12_frames(left_buf, right_buf, left_slot, right_slot, lr, rr);
            core.run_detection_frames(&frames);
        })
    }

    /// Run Metal-resident detection from CVPixelBuffers.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn detect_and_update_director_metal(
        &mut self,
        left_cvpb: crate::interop::metal::CVPixelBufferRef,
        right_cvpb: crate::interop::metal::CVPixelBufferRef,
        width: u32,
        height: u32,
        elapsed: std::time::Duration,
    ) -> Result<(), SessionError> {
        self.detect_and_update_director_with(elapsed, |core| {
            let frames = metal_frames(left_cvpb, right_cvpb, width, height);
            core.run_detection_frames(&frames);
        })
    }

    /// Drive the tracker/panner chain after detection.
    ///
    /// Shared tail for all detection paths (CPU, GPU, Metal,
    /// no-detection). Delegates to the engine's dispatch - one AI
    /// stack, which emits the trace events through the engine's sink -
    /// and records the outcome into the session's telemetry.
    pub(crate) fn fire_sink_and_update_director(
        &mut self,
        elapsed: std::time::Duration,
        _fresh_detection: bool,
    ) -> Result<(), SessionError> {
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
