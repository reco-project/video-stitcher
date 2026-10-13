//! Input-specific opening and native ownership, separate from shared job execution.

use super::*;
use crate::rtsp::{RtspConfig, RtspInput, RtspSource};

pub(super) enum Inputs {
    Files(Vec<InputPath>),
    Rtsp {
        inputs: Vec<RtspInput>,
        config: RtspConfig,
    },
    Supplied(Box<dyn FrameSource>),
}

impl Inputs {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Files(inputs) => inputs.len(),
            Self::Rtsp { inputs, .. } => inputs.len(),
            Self::Supplied(source) => source.info().camera_sizes.len(),
        }
    }

    pub(super) fn validate_options(
        &self,
        start: Option<f64>,
        end: Option<f64>,
        offset: Option<i64>,
        audio: &AudioMode,
    ) -> Result<(), StitchError> {
        if let AudioMode::CopyFrom(camera) = audio
            && *camera >= self.len()
        {
            return Err(StitchError::Other(
                "audio camera index exceeds the input count".into(),
            ));
        }
        if matches!(self, Self::Rtsp { .. })
            && (start.is_some() || end.is_some() || offset.is_some())
        {
            return Err(StitchError::Other(
                    "RTSP jobs do not support file time windows or frame sync offsets; configure RTSP media timestamp offsets instead".into(),
            ));
        }
        if let Self::Supplied(source) = self {
            if source.is_gpu_resident() {
                return Err(StitchError::Other(
                    "supplied native GPU sources require device-aware StitchSession setup".into(),
                ));
            }
            if start.is_some() || offset.is_some() {
                return Err(StitchError::Other(
                    "supplied sources own seeking and synchronization; job start time and frame sync offsets are unavailable".into(),
                ));
            }
            if !matches!(audio, AudioMode::Disabled) {
                return Err(StitchError::Other(
                    "supplied FrameSource delivers images only; source audio copying is unavailable".into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn open(
        self,
        sync_offset: i64,
        software_only: bool,
        force_cpu_decode: bool,
        #[cfg(feature = "gpu")] gpu: Option<&reco_core::gpu::GpuContext>,
        interrupted: &Arc<AtomicBool>,
        audio: &AudioMode,
    ) -> Result<OpenedSource, StitchError> {
        #[cfg(not(feature = "gpu"))]
        let _ = force_cpu_decode;
        match self {
            Self::Files(inputs) => {
                #[cfg(feature = "gpu")]
                if let Some(gpu) = gpu.filter(|_| !force_cpu_decode) {
                    return Ok(OpenedSource::Files(crate::SmartFileSource::open(
                        &inputs,
                        gpu,
                        sync_offset,
                    )?));
                }
                Ok(OpenedSource::Files(crate::SmartFileSource::open_cpu_only(
                    &inputs,
                    sync_offset,
                    software_only,
                )?))
            }
            Self::Rtsp { inputs, config } => {
                let mut config = config;
                config.software_decode |= software_only;
                Ok(OpenedSource::Rtsp(Box::new(RtspSource::open_with_audio(
                    inputs,
                    config,
                    interrupted.clone(),
                    match audio {
                        AudioMode::CopyFrom(camera) => Some(*camera),
                        AudioMode::Disabled => None,
                    },
                    #[cfg(feature = "gpu")]
                    gpu.filter(|_| !force_cpu_decode),
                )?)))
            }
            Self::Supplied(source) => Ok(OpenedSource::Supplied(source)),
        }
    }

    pub(super) fn audio_paths(&self, mode: &AudioMode) -> Option<Vec<PathBuf>> {
        let (Self::Files(inputs), AudioMode::CopyFrom(camera)) = (self, mode) else {
            return None;
        };
        inputs.get(*camera).map(InputPath::all_paths)
    }
}

pub(super) enum OpenedSource {
    Files(crate::SmartFileSource),
    Rtsp(Box<RtspSource>),
    Supplied(Box<dyn FrameSource>),
}

impl OpenedSource {
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    pub(super) fn shared_texture_set(&self) -> Option<&reco_core::interop::SharedTextureSet> {
        match self {
            Self::Files(source) => source.shared_texture_set(),
            Self::Rtsp(source) => source.shared_texture_set(),
            Self::Supplied(_) => None,
        }
    }
    pub(super) fn audio(&self) -> Option<crate::ffmpeg::live_audio::LiveAudio> {
        match self {
            Self::Files(_) | Self::Supplied(_) => None,
            Self::Rtsp(source) => source.audio(),
        }
    }

    pub(super) fn frames(&self) -> &dyn FrameSource {
        match self {
            Self::Files(source) => source,
            Self::Rtsp(source) => source.as_ref(),
            Self::Supplied(source) => source.as_ref(),
        }
    }

    pub(super) fn into_frames(self) -> Box<dyn FrameSource> {
        match self {
            Self::Files(source) => Box::new(source),
            Self::Rtsp(source) => source,
            Self::Supplied(source) => source,
        }
    }

    pub(super) fn decode_mode(&self) -> &'static str {
        match self {
            Self::Files(source) => source.decode_mode(),
            Self::Rtsp(source) => source.decode_mode(),
            Self::Supplied(_) => "supplied host source",
        }
    }
}
