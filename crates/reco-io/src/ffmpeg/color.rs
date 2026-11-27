//! Stream colourimetry metadata (VUI / container colour description).

use ffmpeg_next as ffmpeg;
use std::fmt;

/// ITU-T H.273 colourimetry metadata carried by video streams and containers.
///
/// Each field is `None` when unspecified, reserved, or otherwise absent.
/// Range (`full` vs `limited`) is tracked separately because the renderer
/// converts inputs to limited-range YUV while preserving the source's colour
/// description.
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

    /// Resolve metadata for rendered exports.
    ///
    /// Preserves explicit source colourimetry, and defaults an unspecified matrix to BT.709
    /// only when the metadata is empty or BT.709-compatible SDR. Partial HDR or non-BT.709
    /// metadata keeps `space: None` rather than fabricating a contradictory BT.709 matrix.
    pub(crate) fn for_rendered_output(self) -> Self {
        let source = self.normalized();
        let default_bt709_space = source.space.is_none()
            && !source.is_hdr()
            && matches!(
                source.primaries,
                None | Some(ffmpeg::color::Primaries::BT709)
            )
            && matches!(
                source.transfer,
                None | Some(ffmpeg::color::TransferCharacteristic::BT709)
            );
        Self {
            primaries: source.primaries,
            transfer: source.transfer,
            space: if default_bt709_space {
                Some(ffmpeg::color::Space::BT709)
            } else {
                source.space
            },
        }
    }

    /// Reconcile colour metadata across one or more camera streams.
    ///
    /// Returns the shared metadata when all cameras agree. When cameras disagree,
    /// logs a warning and returns `ColorMetadata::default()` so exports fall back
    /// conservatively to default rendered metadata.
    pub(crate) fn reconcile(cameras: &[Self]) -> Self {
        let Some((&first, rest)) = cameras.split_first() else {
            return Self::default();
        };
        let first = first.normalized();
        for (index, &other) in rest.iter().enumerate() {
            let other = other.normalized();
            if other != first {
                log::warn!(
                    "Camera 0 and camera {} colour metadata differ ({first} vs {other}); \
                     falling back to default export colour metadata",
                    index + 1,
                );
                return Self::default();
            }
        }
        first
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
    fn reconcile_propagates_agreement_and_falls_back_on_mismatch() {
        assert_eq!(ColorMetadata::reconcile(&[]), ColorMetadata::default());
        assert_eq!(ColorMetadata::reconcile(&[hlg_bt2020()]), hlg_bt2020());
        assert_eq!(
            ColorMetadata::reconcile(&[hlg_bt2020(), hlg_bt2020()]),
            hlg_bt2020()
        );
        // Mismatched cameras conservatively fall back to empty default.
        assert_eq!(
            ColorMetadata::reconcile(&[hlg_bt2020(), sdr_bt709()]),
            ColorMetadata::default()
        );
        assert_eq!(
            ColorMetadata::reconcile(&[hlg_bt2020(), ColorMetadata::default()]),
            ColorMetadata::default()
        );
        assert_eq!(
            ColorMetadata::reconcile(&[hlg_bt2020(), pq_bt2020()]),
            ColorMetadata::default()
        );
        // Unspecified fields normalize identically to None.
        let untagged_explicit = ColorMetadata {
            primaries: Some(Primaries::Unspecified),
            transfer: Some(TransferCharacteristic::Unspecified),
            space: Some(Space::Unspecified),
        };
        assert_eq!(
            ColorMetadata::reconcile(&[untagged_explicit, ColorMetadata::default()]),
            ColorMetadata::default()
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
    fn for_rendered_output_defaults_bt709_only_for_empty_or_bt709_sdr() {
        assert_eq!(
            ColorMetadata::default().for_rendered_output(),
            ColorMetadata {
                primaries: None,
                transfer: None,
                space: Some(Space::BT709),
            }
        );

        let partial_bt709 = ColorMetadata {
            primaries: Some(Primaries::BT709),
            transfer: Some(TransferCharacteristic::BT709),
            space: None,
        };
        assert_eq!(
            partial_bt709.for_rendered_output(),
            ColorMetadata {
                primaries: Some(Primaries::BT709),
                transfer: Some(TransferCharacteristic::BT709),
                space: Some(Space::BT709),
            }
        );

        let partial_hdr = ColorMetadata {
            primaries: Some(Primaries::BT2020),
            transfer: Some(TransferCharacteristic::ARIB_STD_B67),
            space: None,
        };
        assert_eq!(partial_hdr.for_rendered_output(), partial_hdr);

        let partial_bt601 = ColorMetadata {
            primaries: Some(Primaries::SMPTE170M),
            transfer: None,
            space: None,
        };
        assert_eq!(partial_bt601.for_rendered_output(), partial_bt601);

        assert_eq!(hlg_bt2020().for_rendered_output(), hlg_bt2020());
    }
}
