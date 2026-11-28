//! VAAPI hardware-frame upload for the FFmpeg encoder.
//!
//! `h264_vaapi` / `hevc_vaapi` / `av1_vaapi` only accept frames that live
//! in VAAPI surfaces, not CPU pixel formats. This module owns the VAAPI
//! hwdevice + hwframe context and uploads CPU NV12 or P010LE staging frames to GPU
//! surfaces via `av_hwframe_transfer_data`. All of the unsafe FFmpeg FFI
//! for that path is isolated here so the encoder core stays readable.

extern crate ffmpeg_next as ffmpeg;

use ffmpeg::format::Pixel;
use ffmpeg::util::frame::video::Video as VideoFrame;
use std::ptr;

use super::encoder::EncodeError;

/// Owns a VAAPI hwframe context and uploads CPU NV12 or P010LE frames to GPU surfaces.
pub(super) struct HardwareUpload {
    frame: VideoFrame,
    frames_ref: *mut ffmpeg::sys::AVBufferRef,
}

impl HardwareUpload {
    /// Create a VAAPI device and hwframe pool for `width`x`height`.
    ///
    /// Returns an error if VAAPI is unavailable, which lets the encoder
    /// fallback chain move on to the next candidate.
    pub(super) fn new_vaapi(
        width: u32,
        height: u32,
        device: Option<&str>,
        sw_format: Pixel,
    ) -> Result<Self, EncodeError> {
        let device = device
            .map(std::ffi::CString::new)
            .transpose()
            .map_err(|_| EncodeError::InvalidConfig("device contains a NUL byte".into()))?;
        unsafe {
            let mut device_ref = ptr::null_mut();
            let ret = ffmpeg::sys::av_hwdevice_ctx_create(
                &mut device_ref,
                ffmpeg::sys::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                device
                    .as_ref()
                    .map_or(ptr::null(), |device| device.as_ptr()),
                ptr::null_mut(),
                0,
            );
            if ret < 0 {
                return Err(ffmpeg::Error::from(ret).into());
            }

            let mut frames_ref = ffmpeg::sys::av_hwframe_ctx_alloc(device_ref);
            if frames_ref.is_null() {
                ffmpeg::sys::av_buffer_unref(&mut device_ref);
                return Err(ffmpeg::Error::Unknown.into());
            }
            ffmpeg::sys::av_buffer_unref(&mut device_ref);

            let frames_ctx = (*frames_ref).data as *mut ffmpeg::sys::AVHWFramesContext;
            if frames_ctx.is_null() {
                ffmpeg::sys::av_buffer_unref(&mut frames_ref);
                return Err(ffmpeg::Error::Unknown.into());
            }

            (*frames_ctx).format = Pixel::VAAPI.into();
            (*frames_ctx).sw_format = sw_format.into();
            (*frames_ctx).width = width as i32;
            (*frames_ctx).height = height as i32;
            (*frames_ctx).initial_pool_size = 8;

            let ret = ffmpeg::sys::av_hwframe_ctx_init(frames_ref);
            if ret < 0 {
                ffmpeg::sys::av_buffer_unref(&mut frames_ref);
                return Err(ffmpeg::Error::from(ret).into());
            }

            Ok(Self {
                frame: VideoFrame::empty(),
                frames_ref,
            })
        }
    }

    /// Attach this hwframe context to the encoder before it is opened.
    ///
    /// VAAPI encoders need `hw_frames_ctx` set before `open` so they know
    /// their input lives in the surface pool. Takes a fresh ref on the
    /// buffer: the encoder releases it on free, `Drop` releases ours.
    ///
    /// `codec_ctx` must be a live, not-yet-opened `AVCodecContext`.
    pub(super) fn attach_to_encoder(
        &self,
        codec_ctx: *mut ffmpeg::sys::AVCodecContext,
    ) -> Result<(), EncodeError> {
        unsafe {
            let frames_ref = ffmpeg::sys::av_buffer_ref(self.frames_ref);
            if frames_ref.is_null() {
                return Err(ffmpeg::Error::Unknown.into());
            }
            (*codec_ctx).hw_frames_ctx = frames_ref;
        }
        Ok(())
    }

    /// Upload a CPU NV12 or P010LE `source` frame to a VAAPI surface and return it.
    ///
    /// The returned frame is owned by `self` and reused on the next call.
    pub(super) fn upload(&mut self, source: &VideoFrame) -> Result<&VideoFrame, EncodeError> {
        let frame = &mut self.frame;
        unsafe {
            ffmpeg::sys::av_frame_unref(frame.as_mut_ptr());
            let ret = ffmpeg::sys::av_hwframe_get_buffer(self.frames_ref, frame.as_mut_ptr(), 0);
            if ret < 0 {
                return Err(ffmpeg::Error::from(ret).into());
            }
            let ret = ffmpeg::sys::av_hwframe_transfer_data(frame.as_mut_ptr(), source.as_ptr(), 0);
            if ret < 0 {
                return Err(ffmpeg::Error::from(ret).into());
            }
        }
        frame.set_pts(source.pts());
        Ok(frame)
    }
}

impl Drop for HardwareUpload {
    fn drop(&mut self) {
        unsafe {
            ffmpeg::sys::av_frame_unref(self.frame.as_mut_ptr());
            ffmpeg::sys::av_buffer_unref(&mut self.frames_ref);
        }
    }
}
