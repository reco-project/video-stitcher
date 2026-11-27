//! Container muxer policies, colour metadata state, codec tags, and HEVC header verification.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::{codec, format};

use super::EncodeError;
use crate::output::Container;

impl Container {
    /// FFmpeg muxer short name for this container format.
    pub(super) const fn muxer_name(self) -> &'static str {
        match self {
            Self::Mp4 | Self::Mp4Fragmented => "mp4",
            Self::Mov => "mov",
            Self::Mkv => "matroska",
            Self::Flv => "flv",
        }
    }
}

/// FourCC codec tag for Apple-compatible HEVC sample entries (`MKTAG('h', 'v', 'c', '1')`).
pub(super) const HVC1_CODEC_TAG: u32 = u32::from_le_bytes(*b"hvc1");

/// Select an explicit container `codec_tag` for a video stream when the muxer default is
/// incompatible with standard players.
///
/// FFmpeg's MP4/MOV muxers default to `hev1` for HEVC when `codec_tag` is `0`, which Apple's
/// playback stack (QuickTime, Safari, Photos, Finder) rejects. Setting `hvc1` writes parameter
/// sets in the `hvcC` sample entry when global headers are active. Other codecs and non-ISOBMFF
/// containers (Matroska, FLV, MPEG-TS, etc.) keep the muxer default (`0`).
pub(super) fn video_codec_tag(
    codec_id: codec::Id,
    muxer_name: &str,
    has_global_header: bool,
) -> Option<u32> {
    if codec_id == codec::Id::HEVC && has_global_header && matches!(muxer_name, "mp4" | "mov") {
        Some(HVC1_CODEC_TAG)
    } else {
        None
    }
}

/// Resolved VUI and container colour configuration for an opened encoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct OutputColor {
    /// Colour primaries, transfer characteristics, and matrix coefficients to tag.
    pub(super) metadata: super::super::ColorMetadata,
    /// Full/limited YUV range to tag, if known (`Some(true)` = full/JPEG, `Some(false)` = limited/MPEG).
    pub(super) full_range: Option<bool>,
    /// Configure CPU `swscale` RGBA->YUV conversion for the renderer's BT.709 limited-range path.
    ///
    /// Kept on BT.709 even when propagating non-BT.709 source VUI metadata because the render
    /// shaders decode source YUV with the BT.709 matrix; matching matrices cancel on round-trip
    /// and keep RGBA and NV12 output paths identical.
    pub(super) rendered_rgba_scaler: bool,
}

impl OutputColor {
    pub(super) fn has_tags(self) -> bool {
        !self.metadata.is_empty() || self.full_range.is_some()
    }
}

/// Pure, unit-testable container muxer policy for a [`Container`].
///
/// Isolates ISOBMFF quirks (`movflags` for fragmentation and `write_colr`,
/// `hvc1` codec tag selection, and muxer name resolution) without requiring
/// an opened FFmpeg context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ContainerPolicy {
    container: Container,
}

impl ContainerPolicy {
    pub(super) const fn for_container(container: Container) -> Self {
        Self { container }
    }

    pub(super) const fn muxer_name(self) -> &'static str {
        self.container.muxer_name()
    }

    pub(super) const fn is_isobmff(self) -> bool {
        matches!(
            self.container,
            Container::Mp4 | Container::Mp4Fragmented | Container::Mov
        )
    }

    pub(super) const fn is_fragmented(self) -> bool {
        matches!(self.container, Container::Mp4Fragmented)
    }

    /// Compute the FFmpeg `movflags` option string for this container and colour state.
    ///
    /// Fragmented MP4 sets `empty_moov+frag_keyframe` so a concurrent reader can
    /// parse self-contained fragments mid-write. ISOBMFF containers (`mp4`, `mov`,
    /// `fmp4`) append `write_colr` whenever colour metadata or range tags are
    /// present so hardware encoders that omit VUI labels still produce tagged files.
    /// Plain MP4 intentionally omits `faststart` to avoid rewriting large recordings
    /// at trailer time.
    pub(super) fn movflags(self, color: OutputColor) -> Option<String> {
        let mut flags = Vec::new();
        if self.is_fragmented() {
            flags.extend(["empty_moov", "frag_keyframe"]);
        }
        if color.has_tags() && self.is_isobmff() {
            flags.push("write_colr");
        }
        (!flags.is_empty()).then(|| flags.join("+"))
    }

    /// Build the FFmpeg muxer options dictionary passed to `write_header`.
    pub(super) fn muxer_options(self, color: OutputColor) -> ffmpeg::Dictionary<'static> {
        let mut opts = ffmpeg::Dictionary::new();
        if let Some(movflags) = self.movflags(color) {
            opts.set("movflags", &movflags);
        }
        opts
    }

    /// Compute the explicit video `codec_tag` for `codec_id` under this container's muxer.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn codec_tag(self, codec_id: codec::Id, has_global_header: bool) -> Option<u32> {
        video_codec_tag(codec_id, self.muxer_name(), has_global_header)
    }
}

/// Whether the opened output muxer requires `AV_CODEC_FLAG_GLOBAL_HEADER`.
pub(super) fn muxer_requires_global_header(octx: &format::context::Output) -> bool {
    octx.format().flags().contains(format::Flags::GLOBAL_HEADER)
}

/// Apply container `codec_tag` to `stream_index` after all `set_parameters` calls
/// (including `HevcColor`), which copy parameters with `codec_tag = 0`.
pub(super) fn apply_video_codec_tag(
    octx: &mut format::context::Output,
    stream_index: usize,
    codec_id: codec::Id,
) {
    if let Some(tag) = video_codec_tag(
        codec_id,
        octx.format().name(),
        muxer_requires_global_header(octx),
    ) {
        let stream = octx
            .stream_mut(stream_index)
            .expect("video stream we just added");
        // SAFETY: the newly added output stream owns writable codec parameters
        // until `write_header` consumes them.
        unsafe {
            (*stream.parameters().as_mut_ptr()).codec_tag = tag;
        }
    }
}

/// Verifies that the encoded HEVC bitstream advertises the requested visible dimensions.
pub(super) struct HevcHeaderCheck {
    parser: std::ptr::NonNull<ffmpeg::sys::AVCodecParserContext>,
    pub(super) context: codec::context::Context,
}

impl HevcHeaderCheck {
    pub(super) fn new(encoder: &ffmpeg::encoder::video::Encoder) -> Result<Self, EncodeError> {
        let context = codec::context::Context::from_parameters(codec::Parameters::from(encoder))?;
        // SAFETY: allocate an independent parser, owned until Drop.
        let parser =
            unsafe { ffmpeg::sys::av_parser_init(ffmpeg::sys::AVCodecID::AV_CODEC_ID_HEVC as _) };
        let parser = std::ptr::NonNull::new(parser).ok_or_else(|| {
            EncodeError::InvalidConfig("HEVC header parser is unavailable".into())
        })?;
        // SAFETY: parser is live and uniquely owned; encoder packets are complete access units.
        unsafe {
            (*parser.as_ptr()).flags |= ffmpeg::sys::PARSER_FLAG_COMPLETE_FRAMES;
        }
        Ok(Self { parser, context })
    }

    pub(super) fn verify(
        &mut self,
        packet: &ffmpeg::Packet,
        expected: (u32, u32),
    ) -> Result<bool, EncodeError> {
        use ffmpeg::packet::Ref;
        // SAFETY: FFmpeg packets include parser padding; all pointers remain live for this call.
        let (result, actual) = unsafe {
            let packet = &*packet.as_ptr();
            let mut parsed = std::ptr::null_mut();
            let mut size = 0;
            let result = ffmpeg::sys::av_parser_parse2(
                self.parser.as_ptr(),
                self.context.as_mut_ptr(),
                &mut parsed,
                &mut size,
                packet.data,
                packet.size,
                packet.pts,
                packet.dts,
                packet.pos,
            );
            let parser = self.parser.as_ref();
            let actual = (parser.width, parser.height);
            (result, actual)
        };
        if result < 0 {
            return Err(ffmpeg::Error::from(result).into());
        }
        // Some backends publish parameter sets separately, before a picture supplies dimensions.
        if actual == (0, 0) {
            return Ok(false);
        }
        if actual != (expected.0 as i32, expected.1 as i32) {
            return Err(EncodeError::EncodedDimensions { expected, actual });
        }
        Ok(true)
    }
}

impl Drop for HevcHeaderCheck {
    fn drop(&mut self) {
        // SAFETY: this object exclusively owns this parser and closes it exactly once.
        unsafe {
            ffmpeg::sys::av_parser_close(self.parser.as_ptr());
        }
    }
}
