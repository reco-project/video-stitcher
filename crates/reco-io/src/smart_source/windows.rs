//! Windows D3D11VA zero-copy native file source.

use reco_core::gpu::GpuContext;
use reco_core::render::renderer::GpuPixelFormat;
use reco_core::source::{D3d11CameraFrame, FrameSet, SourceError, SourceFrame, SourceInfo};

use super::SkipOutcome;
use crate::decode_stream::OwnedFrames;
use crate::ffmpeg::decoder::{D3d11Frame, DecodeBackend, SharedHwDevice};
use crate::stitch_job::InputPath;

enum WindowsDecodeState {
    Pending,
    Running { frames: OwnedFrames<D3d11Frame> },
    Failed(SourceError),
}

pub(super) struct NativeSource {
    inputs: Vec<InputPath>,
    sync_offset: i64,
    current_frame: u64,
    // Drop decode workers before releasing the shared D3D11 device handle.
    decode: WindowsDecodeState,
    shared_device: Option<SharedHwDevice>,
}

impl NativeSource {
    pub(super) fn supports_backend(backend: DecodeBackend) -> bool {
        matches!(backend, DecodeBackend::D3d11va)
    }

    pub(super) fn allows_mixed_camera_sizes() -> bool {
        false
    }

    pub(super) fn supports_input_count(count: usize) -> bool {
        count == 2
    }

    pub(super) fn try_open(
        inputs: &[InputPath],
        _gpu: &GpuContext,
        sync_offset: i64,
        _info: &SourceInfo,
        _pixel_format: GpuPixelFormat,
        _full_range: bool,
    ) -> Result<Option<Self>, SourceError> {
        log::info!("SmartFileSource: D3D11VA zero-copy decode enabled");

        Ok(Some(Self {
            inputs: inputs.to_vec(),
            sync_offset,
            current_frame: 0,
            decode: WindowsDecodeState::Pending,
            shared_device: None,
        }))
    }

    pub(super) fn decode_mode(&self) -> &'static str {
        "D3D11VA zero-copy"
    }

    fn start_at(&mut self, start_secs: Option<f64>) -> Result<(), SourceError> {
        // Cancel and join any previous decode generation before opening
        // replacement D3D11VA decoders.
        self.decode = WindowsDecodeState::Pending;
        if self.shared_device.is_none() {
            let Some(device) = crate::ffmpeg::decoder::create_shared_hw_device() else {
                let error = SourceError::Init {
                    path: "D3D11VA".into(),
                    reason: "hardware device creation failed".into(),
                };
                self.decode = WindowsDecodeState::Failed(error.clone());
                return Err(error);
            };
            self.shared_device = Some(device);
        }
        let frames = match crate::zero_copy::spawn_d3d11_decode_frames(
            &self.inputs,
            self.sync_offset,
            start_secs,
            self.shared_device.as_ref(),
        ) {
            Ok(frames) => frames,
            Err(error) => {
                self.decode = WindowsDecodeState::Failed(error.clone());
                return Err(error);
            }
        };
        self.decode = WindowsDecodeState::Running { frames };
        Ok(())
    }

    fn ensure_running(&mut self) {
        if matches!(self.decode, WindowsDecodeState::Pending) {
            let _ = self.start_at(None);
        }
    }

    /// Jump the decode threads to a new position, replacing whatever
    /// pipeline is currently pending or running.
    ///
    /// Mirrors [`crate::adapters::FfmpegFileSource`]'s
    /// [`reco_core::source::FrameSource::seek`] strategy: cancel and join the
    /// previous decode generation, then respawn decoders on the same shared
    /// D3D11 device seeked to the nearest keyframe before `secs` and
    /// decode-discard only the remaining GOP gap to the target PTS.
    fn seek_to_secs(&mut self, secs: f64) -> Result<(), SourceError> {
        self.start_at(Some(secs))
    }

    pub(super) fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        self.ensure_running();
        let result = match &self.decode {
            WindowsDecodeState::Running { frames } => frames.receive(),
            WindowsDecodeState::Failed(error) => Err(error.clone()),
            WindowsDecodeState::Pending => unreachable!(),
        };
        match result? {
            Some(decoded) => {
                self.current_frame = self.current_frame.saturating_add(1);
                let frames = FrameSet::D3d11Resident(
                    decoded
                        .into_iter()
                        .map(|frame| {
                            // SAFETY: the decoded AVFrame owns the texture and pins this slice.
                            unsafe {
                                D3d11CameraFrame::from_raw(frame.texture, frame.array_slice, frame)
                            }
                        })
                        .collect(),
                );
                Ok(Some(SourceFrame::untimed(frames)))
            }
            None => Ok(None),
        }
    }

    pub(super) fn skip_frames(
        &mut self,
        count: u64,
        info: &SourceInfo,
    ) -> Result<SkipOutcome, SourceError> {
        if let WindowsDecodeState::Failed(error) = &self.decode {
            return Err(error.clone());
        }
        if count == 0 {
            return Ok(SkipOutcome {
                skipped: 0,
                exhausted: false,
            });
        }
        if let Some((target_frame, secs)) = super::skip_seek_target(self.current_frame, count, info)
        {
            self.seek_to_secs(secs)?;
            self.current_frame = target_frame;
            log::debug!(
                "D3D11VA seek: jumped to {secs:.1}s (frame {target_frame}, skipped {count})"
            );
            Ok(SkipOutcome {
                skipped: count,
                exhausted: false,
            })
        } else {
            for i in 0..count {
                if self.next_frame()?.is_none() {
                    return Ok(SkipOutcome {
                        skipped: i,
                        exhausted: true,
                    });
                }
            }
            Ok(SkipOutcome {
                skipped: count,
                exhausted: false,
            })
        }
    }

    pub(super) fn start_decoding(&mut self) {
        self.ensure_running();
    }
}
