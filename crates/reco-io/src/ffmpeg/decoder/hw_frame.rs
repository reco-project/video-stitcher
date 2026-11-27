//! Hardware-accelerated frame wrappers, shared hardware device management,
//! and hwaccel probing/setup.

use super::{DecodeBackend, ffmpeg};
use ffmpeg::ffi;
use ffmpeg::util::frame::video::Video as VideoFrame;
use std::ptr;
#[cfg(target_os = "windows")]
use std::sync::{Arc, Mutex};

/// A decoded NV12 or P010 frame still on the GPU (CUDA device memory).
///
/// Owns a reference to the decoded allocation. Its CUDA pointers remain valid
/// across subsequent decode calls and decoder destruction until this frame is dropped.
/// Consumers must finish GPU reads before releasing their last frame reference.
#[cfg(feature = "gpu")]
pub struct GpuFrame {
    /// CUDA device pointer to the Y (luma) plane.
    pub y_ptr: u64,
    /// CUDA device pointer to the UV (chroma) plane (NV12 interleaved).
    pub uv_ptr: u64,
    /// Row stride (pitch) of the Y plane in bytes.
    pub y_pitch: usize,
    /// Row stride (pitch) of the UV plane in bytes.
    pub uv_pitch: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
    /// GPU pixel format (NV12 or P010). Determines shared texture formats
    /// and CUDA copy byte widths.
    pub pixel_format: reco_core::render::renderer::GpuPixelFormat,
    pub(super) _owner: VideoFrame,
}

/// A decoded frame from VideoToolbox, holding a `CVPixelBufferRef`.
///
/// The pixel buffer is backed by an IOSurface and can be imported into
/// Metal as a texture via `CVMetalTextureCache` without a CPU copy.
///
/// **Lifetime:** The `CVPixelBufferRef` is only valid until the next
/// `next_frame_vt()` call. The caller must import it into Metal textures
/// (or retain it) before decoding the next frame.
#[cfg(all(target_os = "macos", feature = "gpu"))]
pub struct VtFrame {
    /// Opaque `CVPixelBufferRef` pointer from the decoded AVFrame.
    /// Cast from `frame->data[3]` per FFmpeg's hwaccel convention.
    pub cv_pixel_buffer: *mut std::ffi::c_void,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
}

/// A decoded D3D11VA frame referencing the D3D11 array texture pool.
///
/// `texture` is the pool's `ID3D11Texture2D` (same pointer for all frames)
/// and `array_slice` selects which slice within the pool holds this frame.
/// Holds a D3D11VA decoded frame with a retained AVFrame reference.
///
/// The `_frame_ref` keeps the underlying texture slice pinned in FFmpeg's
/// decode pool until this struct is dropped. Without it, FFmpeg recycles
/// the slice and the decoder writes new frame data into it before the
/// staging copy reads the original.
#[cfg(all(target_os = "windows", feature = "gpu"))]
pub struct D3d11Frame {
    /// Pointer to the D3D11VA decode pool texture (`ID3D11Texture2D*`).
    pub texture: *mut std::ffi::c_void,
    /// Array slice index within the pool texture.
    pub array_slice: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: i64,
    /// Cloned AVFrame that pins the texture slice in the decode pool.
    pub(super) _frame_ref: *mut ffi::AVFrame,
}

#[cfg(all(target_os = "windows", feature = "gpu"))]
impl Drop for D3d11Frame {
    fn drop(&mut self) {
        if !self._frame_ref.is_null() {
            // SAFETY: `_frame_ref` was created by `av_frame_clone` and is
            // exclusively owned by this `D3d11Frame` until drop.
            unsafe {
                ffi::av_frame_free(&mut self._frame_ref);
            }
        }
    }
}

// SAFETY: The AVFrame and its texture slice are refcounted. The D3d11Frame
// owns an independent reference that keeps the slice alive across threads.
#[cfg(all(target_os = "windows", feature = "gpu"))]
unsafe impl Send for D3d11Frame {}

/// Check if a frame is hardware-decoded (lives on GPU memory).
pub(super) fn is_hw_frame(frame: &VideoFrame) -> bool {
    // SAFETY: `frame` wraps a live `AVFrame` managed by ffmpeg-next.
    unsafe { !(*frame.as_ptr()).hw_frames_ctx.is_null() }
}

/// Pre-created hw device that can be shared across multiple decoders.
///
/// D3D11VA textures are device-bound: `CopySubresourceRegion` requires source
/// and destination on the same device. When two decoders each create their own
/// device, the staging pool can only copy from one. Sharing a single device
/// solves this.
pub struct SharedHwDevice {
    device_ref: *mut ffi::AVBufferRef,
    backend: DecodeBackend,
    #[cfg(target_os = "windows")]
    pub(super) decoder_init: Arc<Mutex<()>>,
}

// SAFETY: The underlying AVBufferRef is refcounted. The D3D11VA/CUDA device
// it wraps is either thread-safe (CUDA) or protected via ID3D11Multithread.
unsafe impl Send for SharedHwDevice {}

impl SharedHwDevice {
    /// Create a new reference suitable for passing to another decoder.
    ///
    /// Each reference is independently owned and freed on drop.
    pub fn new_ref(&self) -> Self {
        Self {
            // SAFETY: `self.device_ref` is a valid refcounted `AVBufferRef`.
            device_ref: unsafe { ffi::av_buffer_ref(self.device_ref) },
            backend: self.backend,
            #[cfg(target_os = "windows")]
            decoder_init: Arc::clone(&self.decoder_init),
        }
    }

    /// The decode backend this device supports.
    pub fn backend(&self) -> DecodeBackend {
        self.backend
    }
}

impl Drop for SharedHwDevice {
    fn drop(&mut self) {
        if !self.device_ref.is_null() {
            // SAFETY: `self.device_ref` is an owned reference released once on drop.
            unsafe {
                ffi::av_buffer_unref(&mut self.device_ref);
            }
        }
    }
}

/// Create a shared hardware device for multi-decoder use.
///
/// Returns `None` if no hardware device is available (falls back to software).
pub fn create_shared_hw_device() -> Option<SharedHwDevice> {
    let candidates = hwaccel_candidates();
    for (hw_type, backend) in candidates {
        let mut device_ref: *mut ffi::AVBufferRef = ptr::null_mut();
        // SAFETY: `device_ref` is a valid out-pointer; null device/options select defaults.
        let ret = unsafe {
            ffi::av_hwdevice_ctx_create(&mut device_ref, hw_type, ptr::null(), ptr::null_mut(), 0)
        };
        if ret >= 0 {
            log::info!("Shared hw device created: {backend}");
            return Some(SharedHwDevice {
                device_ref,
                backend,
                #[cfg(target_os = "windows")]
                decoder_init: Arc::new(Mutex::new(())),
            });
        }
    }
    None
}

/// Attach a pre-created hw device to a codec context.
pub(super) fn attach_shared_hw_device(
    context: &mut ffmpeg::codec::context::Context,
    shared: &SharedHwDevice,
) -> (DecodeBackend, *mut ffi::AVBufferRef) {
    // SAFETY: `shared.device_ref` is a valid refcounted `AVBufferRef`.
    let our_ref = unsafe { ffi::av_buffer_ref(shared.device_ref) };
    // SAFETY: `context` owns a live `AVCodecContext` and takes its own reference.
    unsafe {
        (*context.as_mut_ptr()).hw_device_ctx = ffi::av_buffer_ref(shared.device_ref);
    }
    log::debug!(
        "Hardware decode attached: {} (shared device)",
        shared.backend
    );
    (shared.backend, our_ref)
}

/// Platform-preferred hwaccel candidates.
fn hwaccel_candidates() -> Vec<(ffi::AVHWDeviceType, DecodeBackend)> {
    vec![
        #[cfg(target_os = "windows")]
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
            DecodeBackend::D3d11va,
        ),
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            DecodeBackend::Cuda,
        ),
        #[cfg(target_os = "linux")]
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            DecodeBackend::Vaapi,
        ),
        #[cfg(target_os = "macos")]
        (
            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            DecodeBackend::VideoToolbox,
        ),
    ]
}

/// Try to enable hardware-accelerated decoding on the codec context.
///
/// Attempts platform-preferred backends first. Returns the active backend
/// and the hw device reference (must be kept alive and freed on drop).
pub(super) fn try_hwaccel(
    context: &mut ffmpeg::codec::context::Context,
) -> (DecodeBackend, *mut ffi::AVBufferRef) {
    let candidates = hwaccel_candidates();

    for (hw_type, backend) in candidates {
        // SAFETY: `av_hwdevice_get_type_name` returns either null or a static
        // NUL-terminated C string owned by FFmpeg.
        let hw_type_name = unsafe {
            let name = ffi::av_hwdevice_get_type_name(hw_type);
            if name.is_null() {
                "unknown"
            } else {
                std::ffi::CStr::from_ptr(name).to_str().unwrap_or("unknown")
            }
        };

        // Try to create a hardware device context
        let mut device_ref: *mut ffi::AVBufferRef = ptr::null_mut();
        // SAFETY: `device_ref` is a valid out-pointer; null device/options select defaults.
        let ret = unsafe {
            ffi::av_hwdevice_ctx_create(
                &mut device_ref,
                hw_type,
                ptr::null(),     // default device
                ptr::null_mut(), // no options
                0,               // no flags
            )
        };

        if ret < 0 {
            log::debug!("Hardware decode {hw_type_name}: not available (error {ret})");
            continue;
        }

        // Assign the hardware device context to the codec context.
        // The codec context takes a reference (av_buffer_ref), so we keep our own.
        // SAFETY: `context` owns a live `AVCodecContext` and `device_ref` was
        // just created successfully by `av_hwdevice_ctx_create`.
        unsafe {
            (*context.as_mut_ptr()).hw_device_ctx = ffi::av_buffer_ref(device_ref);
        }

        log::debug!("Hardware decode enabled: {backend} ({hw_type_name})");
        return (backend, device_ref);
    }

    log::info!("No hardware decoder available - using software decode");
    (DecodeBackend::Software, ptr::null_mut())
}

/// Convert a raw `i32` codec format to `ffi::AVPixelFormat`.
///
/// `AVPixelFormat` is `#[repr(i32)]`, so this is a layout-safe reinterpret
/// via `union` rather than `std::mem::transmute`. FFmpeg accepts any i32
/// and returns NULL/error for unknown values, so no Rust-side validation
/// of the enum discriminant is needed.
pub(super) fn raw_i32_to_pix_fmt(v: i32) -> ffi::AVPixelFormat {
    #[repr(C)]
    union Reinterpret {
        i: i32,
        fmt: ffi::AVPixelFormat,
    }
    // SAFETY: AVPixelFormat is #[repr(i32)], so i32 and AVPixelFormat have
    // identical size and alignment. Reading fmt after writing i is defined
    // because both are integer types with the same representation.
    unsafe { Reinterpret { i: v }.fmt }
}
