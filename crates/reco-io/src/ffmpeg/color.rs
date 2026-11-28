//! Stream colourimetry metadata (VUI / container colour description).

use ffmpeg_next as ffmpeg;
use reco_core::render::color::YuvMatrix;
use std::fmt;

pub(crate) fn swscale_matrix(matrix: YuvMatrix) -> i32 {
    use ffmpeg::ffi::*;
    match matrix {
        YuvMatrix::Bt709 => SWS_CS_ITU709,
        YuvMatrix::Bt2020Ncl => SWS_CS_BT2020,
        YuvMatrix::Bt601 => SWS_CS_SMPTE170M,
    }
}

/// ITU-T H.273 colourimetry metadata carried by video streams and containers.
///
/// Each field is `None` when unspecified, reserved, or otherwise absent.
/// Range (`full` vs `limited`) is tracked separately because the renderer
/// emits 8-bit limited-range YUV in the resolved source matrix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColorMetadata {
    /// Colour primaries (e.g. `BT2020`, `BT709`).
    pub primaries: Option<ffmpeg::color::Primaries>,
    /// Transfer characteristics (e.g. `ARIB_STD_B67` for HLG, `SMPTE2084` for PQ).
    pub transfer: Option<ffmpeg::color::TransferCharacteristic>,
    /// Matrix coefficients / YUV colour space (e.g. `BT2020NCL`, `BT709`).
    pub space: Option<ffmpeg::color::Space>,
}

impl ColorMetadata {
    /// Construct normalized metadata from raw FFmpeg codec parameters.
    pub(crate) fn from_raw(
        primaries: ffmpeg::ffi::AVColorPrimaries,
        transfer: ffmpeg::ffi::AVColorTransferCharacteristic,
        space: ffmpeg::ffi::AVColorSpace,
    ) -> Self {
        Self {
            primaries: Some(ffmpeg::color::Primaries::from(primaries)),
            transfer: Some(ffmpeg::color::TransferCharacteristic::from(transfer)),
            space: Some(ffmpeg::color::Space::from(space)),
        }
        .normalized()
    }

    /// Construct normalized metadata from a decoded FFmpeg video frame.
    pub(crate) fn from_frame(frame: &ffmpeg::frame::Video) -> Self {
        Self {
            primaries: Some(frame.color_primaries()),
            transfer: Some(frame.color_transfer_characteristic()),
            space: Some(frame.color_space()),
        }
        .normalized()
    }

    /// Drop unspecified, reserved, or non-YUV values so only valid VUI tags remain.
    pub fn normalized(self) -> Self {
        Self {
            primaries: self.primaries.filter(|p| {
                !matches!(
                    p,
                    ffmpeg::color::Primaries::Unspecified
                        | ffmpeg::color::Primaries::Reserved
                        | ffmpeg::color::Primaries::Reserved0
                )
            }),
            transfer: self.transfer.filter(|t| {
                !matches!(
                    t,
                    ffmpeg::color::TransferCharacteristic::Unspecified
                        | ffmpeg::color::TransferCharacteristic::Reserved
                        | ffmpeg::color::TransferCharacteristic::Reserved0
                )
            }),
            space: self.space.filter(|s| {
                !matches!(
                    s,
                    ffmpeg::color::Space::Unspecified
                        | ffmpeg::color::Space::Reserved
                        | ffmpeg::color::Space::RGB
                )
            }),
        }
    }

    /// Returns `true` when no colourimetry fields are specified.
    pub fn is_empty(self) -> bool {
        let normalized = self.normalized();
        normalized.primaries.is_none()
            && normalized.transfer.is_none()
            && normalized.space.is_none()
    }

    /// Returns `true` when the metadata signals an HDR transfer function or BT.2020 colourimetry.
    pub fn is_hdr(self) -> bool {
        let normalized = self.normalized();
        matches!(
            normalized.transfer,
            Some(
                ffmpeg::color::TransferCharacteristic::ARIB_STD_B67
                    | ffmpeg::color::TransferCharacteristic::SMPTE2084
            )
        ) || normalized.primaries == Some(ffmpeg::color::Primaries::BT2020)
            || matches!(
                normalized.space,
                Some(ffmpeg::color::Space::BT2020NCL | ffmpeg::color::Space::BT2020CL)
            )
    }

    /// Fill any unspecified fields in `self` from `fallback`, preserving explicit fields.
    pub(crate) fn or(self, fallback: Self) -> Self {
        let primary = self.normalized();
        let fallback = fallback.normalized();
        Self {
            primaries: primary.primaries.or(fallback.primaries),
            transfer: primary.transfer.or(fallback.transfer),
            space: primary.space.or(fallback.space),
        }
    }

    /// Resolve the YCbCr matrix of decoded input. Missing matrix tags
    /// default to BT.2020 NCL for BT.2020 primaries and to BT.709 for SDR.
    pub(crate) fn matrix(self) -> Result<YuvMatrix, String> {
        use ffmpeg::color::{Primaries, Space};

        let source = self.normalized();
        match source.space {
            Some(Space::BT709) => Ok(YuvMatrix::Bt709),
            Some(Space::BT2020NCL) => Ok(YuvMatrix::Bt2020Ncl),
            Some(Space::SMPTE170M | Space::BT470BG) => Ok(YuvMatrix::Bt601),
            Some(space) => Err(format!("unsupported YCbCr matrix {space:?}")),
            None if source.primaries == Some(Primaries::BT2020) => Ok(YuvMatrix::Bt2020Ncl),
            None if source.is_hdr() => {
                Err("HDR input requires a YCbCr matrix tag or BT.2020 primaries".into())
            }
            None => Ok(YuvMatrix::Bt709),
        }
    }

    /// Tag the matrix actually used for encoded YCbCr, retaining the BT.601 alias
    /// from the source when applicable.
    pub(crate) fn with_matrix(self, matrix: YuvMatrix) -> Self {
        use ffmpeg::color::Space;

        let mut metadata = self.normalized();
        metadata.space = Some(match matrix {
            YuvMatrix::Bt709 => Space::BT709,
            YuvMatrix::Bt2020Ncl => Space::BT2020NCL,
            YuvMatrix::Bt601 if metadata.space == Some(Space::BT470BG) => Space::BT470BG,
            YuvMatrix::Bt601 => Space::SMPTE170M,
        });
        metadata
    }

    /// Compare effective tags, including the defaults for untagged SDR.
    pub(crate) fn matches_description(self, other: Self) -> Result<bool, String> {
        use ffmpeg::color::{Primaries, TransferCharacteristic};

        let left = self.normalized();
        let right = other.normalized();
        Ok(left.matrix()? == right.matrix()?
            && left.primaries.unwrap_or(Primaries::BT709)
                == right.primaries.unwrap_or(Primaries::BT709)
            && left.transfer.unwrap_or(TransferCharacteristic::BT709)
                == right.transfer.unwrap_or(TransferCharacteristic::BT709))
    }

    /// Resolve a shared colour description for compositing.
    pub(crate) fn reconcile(cameras: &[Self]) -> Result<(Self, YuvMatrix), String> {
        let Some((&first, rest)) = cameras.split_first() else {
            return Ok((Self::default(), YuvMatrix::Bt709));
        };
        let first = first.normalized();
        let matrix = first.matrix()?;
        let mut shared = first;
        for (index, &other) in rest.iter().enumerate() {
            let other = other.normalized();
            if !first.matches_description(other)? {
                return Err(format!(
                    "camera color metadata differ (camera 0: {first}; camera {}: {other})",
                    index + 1,
                ));
            }
            shared = shared.or(other);
        }
        Ok((shared, matrix))
    }
}

impl fmt::Display for ColorMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let normalized = self.normalized();
        let primaries = normalized
            .primaries
            .and_then(|p| p.name())
            .unwrap_or("unspecified");
        let transfer = normalized
            .transfer
            .and_then(|t| t.name())
            .unwrap_or("unspecified");
        let space = normalized
            .space
            .and_then(|s| s.name())
            .unwrap_or("unspecified");
        write!(
            f,
            "primaries={primaries}, transfer={transfer}, space={space}"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg::color::{Primaries, Space, TransferCharacteristic};

    fn hlg_bt2020() -> ColorMetadata {
        ColorMetadata {
            primaries: Some(Primaries::BT2020),
            transfer: Some(TransferCharacteristic::ARIB_STD_B67),
            space: Some(Space::BT2020NCL),
        }
    }

    fn pq_bt2020() -> ColorMetadata {
        ColorMetadata {
            primaries: Some(Primaries::BT2020),
            transfer: Some(TransferCharacteristic::SMPTE2084),
            space: Some(Space::BT2020NCL),
        }
    }

    fn sdr_bt709() -> ColorMetadata {
        ColorMetadata {
            primaries: Some(Primaries::BT709),
            transfer: Some(TransferCharacteristic::BT709),
            space: Some(Space::BT709),
        }
    }

    #[test]
    fn normalized_strips_unspecified_reserved_and_rgb() {
        let dirty = ColorMetadata {
            primaries: Some(Primaries::Unspecified),
            transfer: Some(TransferCharacteristic::Reserved),
            space: Some(Space::RGB),
        };
        assert_eq!(dirty.normalized(), ColorMetadata::default());
        assert!(dirty.is_empty());

        let reserved0 = ColorMetadata {
            primaries: Some(Primaries::Reserved0),
            transfer: Some(TransferCharacteristic::Reserved0),
            space: Some(Space::Reserved),
        };
        assert_eq!(reserved0.normalized(), ColorMetadata::default());
        assert!(reserved0.is_empty());
    }

    #[test]
    fn hdr_classification_identifies_hlg_pq_and_bt2020() {
        assert!(hlg_bt2020().is_hdr());
        assert!(pq_bt2020().is_hdr());
        assert!(!sdr_bt709().is_hdr());
        assert!(!ColorMetadata::default().is_hdr());
    }

    #[test]
    fn reconcile_rejects_incompatible_camera_color() {
        assert_eq!(
            ColorMetadata::reconcile(&[]).unwrap().0,
            ColorMetadata::default()
        );
        assert_eq!(
            ColorMetadata::reconcile(&[hlg_bt2020(), hlg_bt2020()])
                .unwrap()
                .0,
            hlg_bt2020()
        );
        assert!(ColorMetadata::reconcile(&[hlg_bt2020(), sdr_bt709()]).is_err());
        assert!(ColorMetadata::reconcile(&[hlg_bt2020(), ColorMetadata::default()]).is_err());
        assert!(ColorMetadata::reconcile(&[hlg_bt2020(), pq_bt2020()]).is_err());
        assert_eq!(
            ColorMetadata::reconcile(&[sdr_bt709(), ColorMetadata::default()])
                .unwrap()
                .0,
            sdr_bt709()
        );
    }

    #[test]
    fn display_formats_known_and_unspecified_fields() {
        assert_eq!(
            hlg_bt2020().to_string(),
            "primaries=bt2020, transfer=arib-std-b67, space=bt2020nc"
        );
        assert_eq!(
            ColorMetadata::default().to_string(),
            "primaries=unspecified, transfer=unspecified, space=unspecified"
        );
    }

    #[test]
    fn or_backfills_only_missing_fields() {
        let partial_codecpar = ColorMetadata {
            primaries: Some(Primaries::Unspecified),
            transfer: None,
            space: Some(Space::BT2020NCL),
        };
        let frame_vui = ColorMetadata {
            primaries: Some(Primaries::BT2020),
            transfer: Some(TransferCharacteristic::ARIB_STD_B67),
            space: Some(Space::BT709),
        };
        assert_eq!(
            partial_codecpar.or(frame_vui),
            ColorMetadata {
                primaries: Some(Primaries::BT2020),
                transfer: Some(TransferCharacteristic::ARIB_STD_B67),
                space: Some(Space::BT2020NCL),
            }
        );
        assert_eq!(ColorMetadata::default().or(hlg_bt2020()), hlg_bt2020());
    }

    #[test]
    fn resolves_supported_matrices_and_rejects_constant_luminance() {
        assert_eq!(ColorMetadata::default().matrix().unwrap(), YuvMatrix::Bt709);
        let partial_hdr = ColorMetadata {
            space: None,
            ..hlg_bt2020()
        };
        assert_eq!(partial_hdr.matrix().unwrap(), YuvMatrix::Bt2020Ncl);
        assert_eq!(partial_hdr.with_matrix(YuvMatrix::Bt2020Ncl), hlg_bt2020());
        for space in [Space::SMPTE170M, Space::BT470BG] {
            let metadata = ColorMetadata {
                space: Some(space),
                ..Default::default()
            };
            assert_eq!(metadata.matrix().unwrap(), YuvMatrix::Bt601);
            assert_eq!(metadata.with_matrix(YuvMatrix::Bt601), metadata);
        }
        assert!(
            ColorMetadata {
                space: Some(Space::BT2020CL),
                ..hlg_bt2020()
            }
            .matrix()
            .is_err()
        );
    }
}
