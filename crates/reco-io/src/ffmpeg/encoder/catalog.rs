//! Declarative encoder catalog, platform filtering, and selection validation.
//!
//! Each supported FFmpeg video encoder is described once in [`ALL_ENCODERS`]
//! with its codec, hardware family, CPU/GPU staging path, and quality model.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::Pixel;
use ffmpeg::{codec, encoder};

use super::{EncodeError, EncoderConfig, EncoderInfo};
use crate::output::{BitDepth, Codec, Quality};

impl Codec {
    pub(super) const fn id(self) -> codec::Id {
        match self {
            Self::H264 => codec::Id::H264,
            Self::HEVC => codec::Id::HEVC,
            Self::AV1 => codec::Id::AV1,
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::HEVC => "HEVC",
            Self::AV1 => "AV1",
        }
    }
}

/// How CPU frames are staged and delivered to an FFmpeg video encoder.
///
/// Replaces the implicit `Pixel::VAAPI` sentinel with an explicit model of the
/// encoder's input format, CPU staging format, and whether explicit VAAPI
/// surface upload is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StagingMode {
    /// CPU stages NV12 and hands it directly to the encoder (implicit driver upload).
    Nv12,
    /// CPU stages planar YUV420P for software encoders.
    Yuv420p,
    /// CPU stages P010LE for Main 10 hardware encoders with implicit upload.
    P010,
    /// CPU stages planar YUV420P10LE for libx265.
    Yuv420p10,
    /// CPU stages NV12 or P010LE and uploads it to a VAAPI surface.
    VaapiSurface(BitDepth),
}

impl StagingMode {
    /// Pixel format configured on the FFmpeg `AVCodecContext`.
    pub(super) const fn encoder_pixel_format(self) -> Pixel {
        match self {
            Self::VaapiSurface(_) => Pixel::VAAPI,
            _ => self.staging_pixel_format(),
        }
    }

    /// CPU software pixel format for the reusable staging frame.
    pub(super) const fn staging_pixel_format(self) -> Pixel {
        match self {
            Self::Nv12 | Self::VaapiSurface(BitDepth::Eight) => Pixel::NV12,
            Self::Yuv420p => Pixel::YUV420P,
            Self::P010 | Self::VaapiSurface(BitDepth::Ten) => Pixel::P010LE,
            Self::Yuv420p10 => Pixel::YUV420P10LE,
        }
    }

    /// Whether frames must be transferred via [`super::super::hw_upload::HardwareUpload`].
    pub(super) const fn uses_vaapi_upload(self) -> bool {
        matches!(self, Self::VaapiSurface(_))
    }
}

/// Hardware or software backend family for an FFmpeg encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EncoderFamily {
    Nvenc,
    Qsv,
    VideoToolbox,
    Amf,
    Vaapi,
    V4l2M2m,
    MediaCodec,
    MediaFoundation,
    Software,
}

impl EncoderFamily {
    /// Whether this backend uses hardware acceleration.
    pub(super) const fn is_hardware(self) -> bool {
        !matches!(self, Self::Software)
    }

    /// Infer backend family from a custom user-supplied encoder name suffix.
    pub(super) fn from_suffix(name: &str) -> Option<Self> {
        if name.ends_with("_nvenc") {
            Some(Self::Nvenc)
        } else if name.ends_with("_qsv") {
            Some(Self::Qsv)
        } else if name.ends_with("_videotoolbox") {
            Some(Self::VideoToolbox)
        } else if name.ends_with("_amf") {
            Some(Self::Amf)
        } else if name.ends_with("_vaapi") {
            Some(Self::Vaapi)
        } else if name.ends_with("_v4l2m2m") {
            Some(Self::V4l2M2m)
        } else if name.ends_with("_mediacodec") {
            Some(Self::MediaCodec)
        } else if name.ends_with("_mf") {
            Some(Self::MediaFoundation)
        } else {
            None
        }
    }

    /// Whether automatic encoder selection may consider this backend on the current platform.
    pub(super) fn auto_candidate_allowed(self, name: &str, bit_depth: BitDepth) -> bool {
        let _ = (name, bit_depth);
        match self {
            Self::Nvenc => cfg!(any(target_os = "linux", target_os = "windows")),
            Self::Qsv => {
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                {
                    false
                }
                #[cfg(target_os = "linux")]
                {
                    if bit_depth == BitDepth::Eight
                        && let Some(vaapi_name) = vaapi_peer_encoder(name)
                        && encoder::find_by_name(vaapi_name).is_some()
                    {
                        return false;
                    }
                    true
                }
                #[cfg(target_os = "windows")]
                {
                    true
                }
            }
            Self::VideoToolbox => cfg!(target_os = "macos"),
            Self::Amf | Self::MediaFoundation => cfg!(target_os = "windows"),
            Self::Vaapi | Self::V4l2M2m => cfg!(target_os = "linux"),
            Self::MediaCodec => cfg!(target_os = "android"),
            Self::Software => true,
        }
    }

    /// Validate an explicit `config.device` value for this backend family.
    pub(super) fn validate_device(self, name: &str, device: &str) -> Result<(), EncodeError> {
        if device.is_empty() || device.contains('\0') {
            return Err(EncodeError::InvalidConfig(
                "device must be nonempty and contain no NUL bytes".into(),
            ));
        }
        match self {
            Self::Nvenc => {
                if device.parse::<u32>().is_err() {
                    return Err(EncodeError::InvalidConfig(
                        "NVENC device must be a nonnegative GPU index".into(),
                    ));
                }
                Ok(())
            }
            Self::Vaapi => Ok(()),
            _ => Err(EncodeError::InvalidConfig(format!(
                "explicit device selection is not supported for {name}"
            ))),
        }
    }

    /// Whether a custom (non-catalog) encoder with this family suffix supports numeric quality.
    pub(super) const fn supports_numeric_quality_fallback(self) -> bool {
        matches!(
            self,
            Self::Nvenc | Self::Qsv | Self::Vaapi | Self::Amf | Self::VideoToolbox
        )
    }
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn vaapi_peer_encoder(name: &str) -> Option<&'static str> {
    match name {
        "h264_qsv" => Some("h264_vaapi"),
        "hevc_qsv" => Some("hevc_vaapi"),
        "av1_qsv" => Some("av1_vaapi"),
        _ => None,
    }
}

/// Per-tier NVENC VBR parameters (bitrates are 1080p baselines in Mbps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NvencTier {
    pub(super) preset: &'static str,
    pub(super) cq: &'static str,
    pub(super) bitrate_mbps: u32,
    pub(super) maxrate_mbps: u32,
}

/// Per-tier preset + CRF parameters for software CRF encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SoftwareCrfTier {
    pub(super) preset: &'static str,
    pub(super) crf: &'static str,
}

/// Per-tier CRF + `cpu-used` parameters for `libaom-av1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AomTier {
    pub(super) crf: &'static str,
    pub(super) cpu_used: &'static str,
}

/// Declarative mapping from [`Quality`] to FFmpeg encoder options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QualityModel {
    /// NVENC VBR with resolution-scaled target bitrate and maxrate ceilings.
    NvencVbr {
        fast: NvencTier,
        balanced: NvencTier,
        high: NvencTier,
        profile: Option<&'static str>,
        adaptive_quantization: bool,
    },
    /// Intel QuickSync `global_quality` with `preset=medium`.
    Qsv { profile: Option<&'static str> },
    /// Apple VideoToolbox `global_quality` paired with resolution-scaled `maxrate`.
    VideoToolbox {
        fast_maxrate_mbps: u32,
        balanced_maxrate_mbps: u32,
        high_maxrate_mbps: u32,
        profile: &'static str,
        is_hevc: bool,
    },
    /// VAAPI constant-QP (`qp`).
    VaapiQp { profile: Option<&'static str> },
    /// AMD AMF constant-QP (`rc=cqp`, `qp_i`, `qp_p`).
    AmfCqp,
    /// Software CRF + `preset` encoder (`libx264`, `libx265`, `libsvtav1`).
    SoftwareCrf {
        fast: SoftwareCrfTier,
        balanced: SoftwareCrfTier,
        high: SoftwareCrfTier,
        profile: Option<&'static str>,
        integer_crf: bool,
        tegra_thread_cap: bool,
    },
    /// `libaom-av1` CRF + `cpu-used` + `row-mt=1`.
    SoftwareAomCrf {
        fast: AomTier,
        balanced: AomTier,
        high: AomTier,
    },
    /// Driver-managed quality defaults with no portable constant-quality control.
    DriverDefault,
}

impl QualityModel {
    /// Whether this encoder supports normalized numeric quality (`Quality::Value(0..=100)`).
    pub(super) const fn supports_numeric_quality(self) -> bool {
        !matches!(self, Self::DriverDefault)
    }
}

/// Declarative specification of a known FFmpeg video encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EncoderSpec {
    pub(super) name: &'static str,
    pub(super) codec: Codec,
    pub(super) family: EncoderFamily,
    pub(super) staging: StagingMode,
    pub(super) quality_model: QualityModel,
}

impl EncoderSpec {
    pub(super) const fn is_hardware(&self) -> bool {
        self.family.is_hardware()
    }

    pub(super) fn auto_candidate_allowed(&self, bit_depth: BitDepth) -> bool {
        self.family.auto_candidate_allowed(self.name, bit_depth)
    }

    pub(super) fn staging_for(&self, bit_depth: BitDepth) -> Option<StagingMode> {
        if bit_depth == BitDepth::Eight {
            return Some(self.staging);
        }
        if self.codec != Codec::HEVC {
            return None;
        }
        match self.family {
            EncoderFamily::Nvenc
            | EncoderFamily::Qsv
            | EncoderFamily::VideoToolbox
            | EncoderFamily::Amf => Some(StagingMode::P010),
            EncoderFamily::Vaapi => Some(StagingMode::VaapiSurface(BitDepth::Ten)),
            EncoderFamily::Software => Some(StagingMode::Yuv420p10),
            EncoderFamily::V4l2M2m | EncoderFamily::MediaCodec | EncoderFamily::MediaFoundation => {
                None
            }
        }
    }
}

/// Complete encoder preference catalog across all supported codecs.
///
/// Within each codec, entries are ordered in strict selection preference:
/// NVIDIA (NVENC), Intel (QSV), Apple (VideoToolbox), AMD (AMF/VAAPI),
/// embedded Linux (V4L2 M2M), Android (MediaCodec), Windows fallback (MF),
/// and software encoders last.
pub(super) const ALL_ENCODERS: &[EncoderSpec] = &[
    // ── H.264 ────────────────────────────────────────────────────────────
    EncoderSpec {
        name: "h264_nvenc",
        codec: Codec::H264,
        family: EncoderFamily::Nvenc,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::NvencVbr {
            fast: NvencTier {
                preset: "p3",
                cq: "28",
                bitrate_mbps: 8,
                maxrate_mbps: 12,
            },
            balanced: NvencTier {
                preset: "p4",
                cq: "23",
                bitrate_mbps: 12,
                maxrate_mbps: 18,
            },
            high: NvencTier {
                preset: "p5",
                cq: "19",
                bitrate_mbps: 20,
                maxrate_mbps: 30,
            },
            profile: Some("high"),
            adaptive_quantization: true,
        },
    },
    EncoderSpec {
        name: "h264_qsv",
        codec: Codec::H264,
        family: EncoderFamily::Qsv,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::Qsv {
            profile: Some("high"),
        },
    },
    EncoderSpec {
        name: "h264_videotoolbox",
        codec: Codec::H264,
        family: EncoderFamily::VideoToolbox,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::VideoToolbox {
            fast_maxrate_mbps: 12,
            balanced_maxrate_mbps: 18,
            high_maxrate_mbps: 30,
            profile: "high",
            is_hevc: false,
        },
    },
    EncoderSpec {
        name: "h264_amf",
        codec: Codec::H264,
        family: EncoderFamily::Amf,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::AmfCqp,
    },
    EncoderSpec {
        name: "h264_vaapi",
        codec: Codec::H264,
        family: EncoderFamily::Vaapi,
        staging: StagingMode::VaapiSurface(BitDepth::Eight),
        quality_model: QualityModel::VaapiQp {
            profile: Some("high"),
        },
    },
    EncoderSpec {
        name: "h264_v4l2m2m",
        codec: Codec::H264,
        family: EncoderFamily::V4l2M2m,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::DriverDefault,
    },
    EncoderSpec {
        name: "h264_mediacodec",
        codec: Codec::H264,
        family: EncoderFamily::MediaCodec,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::DriverDefault,
    },
    EncoderSpec {
        name: "h264_mf",
        codec: Codec::H264,
        family: EncoderFamily::MediaFoundation,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::DriverDefault,
    },
    EncoderSpec {
        name: "libx264",
        codec: Codec::H264,
        family: EncoderFamily::Software,
        staging: StagingMode::Yuv420p,
        quality_model: QualityModel::SoftwareCrf {
            fast: SoftwareCrfTier {
                preset: "ultrafast",
                crf: "32",
            },
            balanced: SoftwareCrfTier {
                preset: "veryfast",
                crf: "28",
            },
            high: SoftwareCrfTier {
                preset: "fast",
                crf: "23",
            },
            profile: Some("high"),
            integer_crf: false,
            tegra_thread_cap: true,
        },
    },
    // ── HEVC ─────────────────────────────────────────────────────────────
    EncoderSpec {
        name: "hevc_nvenc",
        codec: Codec::HEVC,
        family: EncoderFamily::Nvenc,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::NvencVbr {
            fast: NvencTier {
                preset: "p3",
                cq: "28",
                bitrate_mbps: 6,
                maxrate_mbps: 10,
            },
            balanced: NvencTier {
                preset: "p4",
                cq: "23",
                bitrate_mbps: 9,
                maxrate_mbps: 14,
            },
            high: NvencTier {
                preset: "p5",
                cq: "19",
                bitrate_mbps: 15,
                maxrate_mbps: 22,
            },
            profile: Some("main"),
            adaptive_quantization: true,
        },
    },
    EncoderSpec {
        name: "hevc_qsv",
        codec: Codec::HEVC,
        family: EncoderFamily::Qsv,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::Qsv { profile: None },
    },
    EncoderSpec {
        name: "hevc_videotoolbox",
        codec: Codec::HEVC,
        family: EncoderFamily::VideoToolbox,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::VideoToolbox {
            fast_maxrate_mbps: 10,
            balanced_maxrate_mbps: 14,
            high_maxrate_mbps: 22,
            profile: "main",
            is_hevc: true,
        },
    },
    EncoderSpec {
        name: "hevc_amf",
        codec: Codec::HEVC,
        family: EncoderFamily::Amf,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::AmfCqp,
    },
    EncoderSpec {
        name: "hevc_vaapi",
        codec: Codec::HEVC,
        family: EncoderFamily::Vaapi,
        staging: StagingMode::VaapiSurface(BitDepth::Eight),
        quality_model: QualityModel::VaapiQp { profile: None },
    },
    EncoderSpec {
        name: "hevc_v4l2m2m",
        codec: Codec::HEVC,
        family: EncoderFamily::V4l2M2m,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::DriverDefault,
    },
    EncoderSpec {
        name: "hevc_mediacodec",
        codec: Codec::HEVC,
        family: EncoderFamily::MediaCodec,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::DriverDefault,
    },
    EncoderSpec {
        name: "hevc_mf",
        codec: Codec::HEVC,
        family: EncoderFamily::MediaFoundation,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::DriverDefault,
    },
    EncoderSpec {
        name: "libx265",
        codec: Codec::HEVC,
        family: EncoderFamily::Software,
        staging: StagingMode::Yuv420p,
        quality_model: QualityModel::SoftwareCrf {
            fast: SoftwareCrfTier {
                preset: "veryfast",
                crf: "28",
            },
            balanced: SoftwareCrfTier {
                preset: "fast",
                crf: "25",
            },
            high: SoftwareCrfTier {
                preset: "medium",
                crf: "21",
            },
            profile: Some("main"),
            integer_crf: false,
            tegra_thread_cap: false,
        },
    },
    // ── AV1 ──────────────────────────────────────────────────────────────
    EncoderSpec {
        name: "av1_nvenc",
        codec: Codec::AV1,
        family: EncoderFamily::Nvenc,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::NvencVbr {
            fast: NvencTier {
                preset: "p3",
                cq: "32",
                bitrate_mbps: 5,
                maxrate_mbps: 8,
            },
            balanced: NvencTier {
                preset: "p4",
                cq: "27",
                bitrate_mbps: 7,
                maxrate_mbps: 11,
            },
            high: NvencTier {
                preset: "p5",
                cq: "22",
                bitrate_mbps: 12,
                maxrate_mbps: 18,
            },
            profile: None,
            adaptive_quantization: false,
        },
    },
    EncoderSpec {
        name: "av1_qsv",
        codec: Codec::AV1,
        family: EncoderFamily::Qsv,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::Qsv { profile: None },
    },
    EncoderSpec {
        name: "av1_amf",
        codec: Codec::AV1,
        family: EncoderFamily::Amf,
        staging: StagingMode::Nv12,
        quality_model: QualityModel::AmfCqp,
    },
    EncoderSpec {
        name: "av1_vaapi",
        codec: Codec::AV1,
        family: EncoderFamily::Vaapi,
        staging: StagingMode::VaapiSurface(BitDepth::Eight),
        quality_model: QualityModel::VaapiQp { profile: None },
    },
    EncoderSpec {
        name: "libsvtav1",
        codec: Codec::AV1,
        family: EncoderFamily::Software,
        staging: StagingMode::Yuv420p,
        quality_model: QualityModel::SoftwareCrf {
            fast: SoftwareCrfTier {
                preset: "10",
                crf: "35",
            },
            balanced: SoftwareCrfTier {
                preset: "7",
                crf: "30",
            },
            high: SoftwareCrfTier {
                preset: "4",
                crf: "25",
            },
            profile: None,
            integer_crf: true,
            tegra_thread_cap: false,
        },
    },
    EncoderSpec {
        name: "libaom-av1",
        codec: Codec::AV1,
        family: EncoderFamily::Software,
        staging: StagingMode::Yuv420p,
        quality_model: QualityModel::SoftwareAomCrf {
            fast: AomTier {
                crf: "35",
                cpu_used: "8",
            },
            balanced: AomTier {
                crf: "30",
                cpu_used: "6",
            },
            high: AomTier {
                crf: "25",
                cpu_used: "4",
            },
        },
    },
];

/// Look up a catalog specification by FFmpeg encoder name.
pub(super) fn find_encoder_spec(name: &str) -> Option<&'static EncoderSpec> {
    ALL_ENCODERS.iter().find(|spec| spec.name == name)
}

/// Iterate over known encoder specs for `codec` in preference order.
pub(super) fn specs_for_codec(codec: Codec) -> impl Iterator<Item = &'static EncoderSpec> {
    ALL_ENCODERS.iter().filter(move |spec| spec.codec == codec)
}

/// List encoders registered in FFmpeg for a given codec.
///
/// Returns encoders in preference order (hardware first, then software). Registration does not guarantee that the required hardware or driver is usable.
pub fn available_encoders(codec: Codec) -> Vec<EncoderInfo> {
    crate::init();
    specs_for_codec(codec)
        .filter_map(|spec| {
            encoder::find_by_name(spec.name).map(|found| EncoderInfo {
                name: spec.name.to_string(),
                description: found.description().to_string(),
                is_hardware: spec.is_hardware(),
            })
        })
        .collect()
}

/// Detect which H.264 encoders are available (convenience wrapper).
pub fn available_h264_encoders() -> Vec<EncoderInfo> {
    available_encoders(Codec::H264)
}

/// Validate device and numeric-quality selections against a candidate encoder.
///
/// Uses [`EncoderSpec`] for catalog entries and falls back to backend suffix
/// matching for custom user-supplied `encoder_name`s.
pub(super) fn validate_selection(name: &str, config: &EncoderConfig) -> Result<(), EncodeError> {
    let spec = find_encoder_spec(name);
    let family = spec
        .map(|s| s.family)
        .or_else(|| EncoderFamily::from_suffix(name))
        .unwrap_or(EncoderFamily::Software);

    if let Some(device) = &config.device {
        family.validate_device(name, device)?;
    }

    if matches!(config.quality, Quality::Value(_)) {
        let supports_numeric = match spec {
            Some(spec) => spec.quality_model.supports_numeric_quality(),
            None => family.supports_numeric_quality_fallback(),
        };
        if !supports_numeric {
            return Err(EncodeError::InvalidConfig(format!(
                "quality selection is not supported by {name}"
            )));
        }
    }

    Ok(())
}
