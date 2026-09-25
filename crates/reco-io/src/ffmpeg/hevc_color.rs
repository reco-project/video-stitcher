//! Known HEVC colour metadata in both global and repeated in-band headers.

use ffmpeg::codec::packet::Mut;
use ffmpeg::ffi;
use ffmpeg_next as ffmpeg;
use std::ptr::{self, NonNull};

/// Owns FFmpeg's header-only filter. Compressed picture data is not decoded.
pub(super) struct HevcColor(NonNull<ffi::AVBSFContext>);

// SAFETY: this owns an independent FFmpeg context. All access is exclusive through &mut self;
// moving ownership between encoder threads does not share or concurrently access the context.
unsafe impl Send for HevcColor {}

impl HevcColor {
    pub(super) fn available() -> bool {
        // SAFETY: lookup only reads FFmpeg's static filter registry.
        !unsafe { ffi::av_bsf_get_by_name(c"hevc_metadata".as_ptr()) }.is_null()
    }

    pub(super) fn new(
        encoder: &ffmpeg::encoder::video::Encoder,
        matrix: Option<ffmpeg::color::Space>,
        full_range: bool,
    ) -> Result<Self, ffmpeg::Error> {
        // SAFETY: the filter name is static; allocation returns an exclusively owned context.
        let mut context = ptr::null_mut();
        unsafe {
            let filter = ffi::av_bsf_get_by_name(c"hevc_metadata".as_ptr());
            if filter.is_null() {
                return Err(ffmpeg::Error::BsfNotFound);
            }
            check(ffi::av_bsf_alloc(filter, &mut context))?;
        }
        let mut this = Self(NonNull::new(context).ok_or(ffmpeg::Error::Bug)?);
        // SAFETY: this owns the allocated context, whose parameter and private storage are live.
        // FFmpeg copies the encoder parameters; no borrowed encoder storage is retained.
        unsafe {
            let context = this.0.as_mut();
            check(ffi::avcodec_parameters_from_context(
                context.par_in,
                encoder.as_ptr(),
            ))?;
            context.time_base_in = encoder.time_base().into();
            let matrix = matrix
                .map(|space| ffi::AVColorSpace::from(space) as i64)
                .unwrap_or(-1);
            for (name, value) in [
                (c"matrix_coefficients", matrix),
                (c"video_full_range_flag", i64::from(full_range)),
            ] {
                check(ffi::av_opt_set_int(
                    context.priv_data,
                    name.as_ptr(),
                    value,
                    0,
                ))?;
            }
            check(ffi::av_bsf_init(this.0.as_ptr()))?;
        }
        Ok(this)
    }

    pub(super) fn parameters(&self) -> Result<ffmpeg::codec::Parameters, ffmpeg::Error> {
        let mut parameters = ffmpeg::codec::Parameters::new();
        // SAFETY: the initialized context owns par_out; copy into independently owned parameters.
        unsafe {
            check(ffi::avcodec_parameters_copy(
                parameters.as_mut_ptr(),
                self.0.as_ref().par_out,
            ))?;
        }
        Ok(parameters)
    }

    pub(super) fn send(
        &mut self,
        packet: Option<&mut ffmpeg::Packet>,
    ) -> Result<(), ffmpeg::Error> {
        // SAFETY: the context is exclusively borrowed. On success FFmpeg moves packet ownership
        // and clears the input packet; None signals EOF. The caller drains before sending again.
        unsafe {
            check(ffi::av_bsf_send_packet(
                self.0.as_ptr(),
                packet.map_or(ptr::null_mut(), |p| p.as_mut_ptr()),
            ))
        }
    }

    pub(super) fn receive(&mut self, packet: &mut ffmpeg::Packet) -> Result<bool, ffmpeg::Error> {
        // SAFETY: both context and destination packet are live and exclusively borrowed.
        let result = unsafe { ffi::av_bsf_receive_packet(self.0.as_ptr(), packet.as_mut_ptr()) };
        match ffmpeg::Error::from(result) {
            _ if result >= 0 => Ok(true),
            ffmpeg::Error::Eof
            | ffmpeg::Error::Other {
                errno: ffmpeg::error::EAGAIN,
            } => Ok(false),
            error => Err(error),
        }
    }
}

impl Drop for HevcColor {
    fn drop(&mut self) {
        // SAFETY: this exclusively owns the context and releases it exactly once.
        unsafe { ffi::av_bsf_free(&mut self.0.as_ptr()) }
    }
}

fn check(result: i32) -> Result<(), ffmpeg::Error> {
    if result < 0 {
        Err(ffmpeg::Error::from(result))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_and_repeated_headers_change_labels_without_changing_pixels_or_timing() {
        crate::init();
        for global in [false, true] {
            let codec = ffmpeg::encoder::find_by_name("libx265").unwrap();
            let mut video = ffmpeg::codec::context::Context::new_with_codec(codec)
                .encoder()
                .video()
                .unwrap();
            video.set_width(64);
            video.set_height(64);
            video.set_format(ffmpeg::format::Pixel::YUV420P);
            video.set_time_base((1, 30));
            video.set_gop(1);
            if global {
                video.set_flags(ffmpeg::codec::Flags::GLOBAL_HEADER);
            }
            let mut options = ffmpeg::Dictionary::new();
            options.set("preset", "ultrafast");
            options.set("tune", "zerolatency");
            options.set("x265-params", "log-level=error:pools=none");
            let mut encoder = video.open_with(options).unwrap();
            let original_parameters = ffmpeg::codec::Parameters::from(&encoder);
            let mut filter =
                HevcColor::new(&encoder, Some(ffmpeg::color::Space::BT709), false).unwrap();
            let mut original =
                ffmpeg::codec::context::Context::from_parameters(original_parameters)
                    .unwrap()
                    .decoder()
                    .video()
                    .unwrap();
            let mut corrected =
                ffmpeg::codec::context::Context::from_parameters(filter.parameters().unwrap())
                    .unwrap()
                    .decoder()
                    .video()
                    .unwrap();
            let mut before = Vec::new();
            let mut after = Vec::new();
            for index in 0..4 {
                let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 64, 64);
                for (plane, value) in [70 + index as u8, 80, 180].into_iter().enumerate() {
                    frame.data_mut(plane).fill(value);
                }
                frame.set_pts(Some(index));
                encoder.send_frame(&frame).unwrap();
                let mut packet = ffmpeg::Packet::empty();
                encoder.receive_packet(&mut packet).unwrap();
                let timing = (packet.pts(), packet.dts(), packet.duration());
                original.send_packet(&packet).unwrap();
                drain(&mut original, &mut before);
                filter.send(Some(&mut packet)).unwrap();
                assert!(filter.receive(&mut packet).unwrap());
                assert_eq!((packet.pts(), packet.dts(), packet.duration()), timing);
                corrected.send_packet(&packet).unwrap();
                drain(&mut corrected, &mut after);
                assert!(!filter.receive(&mut ffmpeg::Packet::empty()).unwrap());
            }
            encoder.send_eof().unwrap();
            assert_eq!(
                encoder.receive_packet(&mut ffmpeg::Packet::empty()),
                Err(ffmpeg::Error::Eof)
            );
            filter.send(None).unwrap();
            assert!(!filter.receive(&mut ffmpeg::Packet::empty()).unwrap());
            original.send_eof().unwrap();
            corrected.send_eof().unwrap();
            drain(&mut original, &mut before);
            drain(&mut corrected, &mut after);
            assert_eq!(before.len(), 4);
            assert_eq!(after.len(), 4);
            for (before, after) in before.iter().zip(&after) {
                assert_eq!(before.color_space(), ffmpeg::color::Space::Unspecified);
                assert_eq!(after.color_space(), ffmpeg::color::Space::BT709);
                assert_eq!(after.color_range(), ffmpeg::color::Range::MPEG);
                assert_eq!(
                    after.color_primaries(),
                    ffmpeg::color::Primaries::Unspecified
                );
                assert_eq!(
                    after.color_transfer_characteristic(),
                    ffmpeg::color::TransferCharacteristic::Unspecified
                );
                assert_eq!(before.pts(), after.pts());
                for plane in 0..3 {
                    for row in 0..before.plane_height(plane) as usize {
                        let width = before.plane_width(plane) as usize;
                        assert_eq!(
                            &before.data(plane)
                                [row * before.stride(plane)..row * before.stride(plane) + width],
                            &after.data(plane)
                                [row * after.stride(plane)..row * after.stride(plane) + width]
                        );
                    }
                }
            }
        }
    }

    fn drain(decoder: &mut ffmpeg::decoder::Video, frames: &mut Vec<ffmpeg::frame::Video>) {
        loop {
            let mut frame = ffmpeg::frame::Video::empty();
            match decoder.receive_frame(&mut frame) {
                Ok(()) => frames.push(frame),
                Err(
                    ffmpeg::Error::Eof
                    | ffmpeg::Error::Other {
                        errno: ffmpeg::error::EAGAIN,
                    },
                ) => break,
                Err(error) => panic!("decode failed: {error}"),
            }
        }
    }
}
