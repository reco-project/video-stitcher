//! Types for zero-copy GPU decode interop.
//!
//! These types are shared between `reco-core` (which runs the frame loop)
//! and `reco-io` (which spawns decode threads). The split keeps `reco-core`
//! free of FFmpeg/GStreamer dependencies while letting it orchestrate the
//! zero-copy pipeline.

/// CUDA buffer info passed to decode threads for `cuMemcpy2D` destination.
///
/// Each camera has two slots (double-buffered) with Y and UV textures.
/// The decode thread writes NVDEC output to these via CUDA, and the
/// render thread reads them as wgpu textures via Vulkan shared memory.
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Clone)]
pub struct GpuBufInfo {
    /// CUDA device pointers for double-buffered Y textures.
    pub y_ptr: [u64; 2],
    /// CUDA device pointers for double-buffered UV textures.
    pub uv_ptr: [u64; 2],
    /// Row pitch of shared Y textures (may differ from width due to alignment).
    pub y_pitch: [usize; 2],
    /// Row pitch of shared UV textures.
    pub uv_pitch: [usize; 2],
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format (NV12 8-bit or P010 10-bit). Determines CUDA copy
    /// width via [`GpuPixelFormat::bytes_per_sample`].
    pub pixel_format: crate::renderer::GpuPixelFormat,
}

/// A pair of double-buffer slot indices from the decode threads.
///
/// Indicates which slots contain the latest decoded frames for left
/// and right cameras. The render thread uses these to select the
/// correct bind group.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub struct GpuFrameSignal {
    /// Left camera double-buffer slot index (0 or 1).
    pub left_slot: u8,
    /// Right camera double-buffer slot index (0 or 1).
    pub right_slot: u8,
}

/// Handles for GPU decode threads, used for lifecycle management.
///
/// The session drives the frame loop by receiving signals from
/// `frame_rx`, rendering, and releasing slots. On shutdown, the
/// session drops senders, then joins threads before dropping
/// shared textures (ordering prevents CUDA error 700).
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub struct GpuDecodeHandles {
    /// Receives paired frame signals (slot indices).
    pub frame_rx: std::sync::mpsc::Receiver<GpuFrameSignal>,
    /// Join handles for the 2 decode threads + 1 pairing thread.
    /// Must be joined before dropping shared textures to ensure
    /// FFmpeg's CUDA context cleanup completes while shared memory
    /// is still valid.
    pub join_handles: Vec<std::thread::JoinHandle<()>>,
}

/// A retained CVPixelBuffer pair from two VideoToolbox decode threads.
///
/// Each buffer has been `CFRetain`-ed so it remains valid until dropped.
/// The session imports these as Metal textures for zero-copy rendering.
#[cfg(target_os = "macos")]
pub struct VtFramePair {
    /// Left camera retained pixel buffer.
    pub left: crate::metal_interop::RetainedCVPixelBuffer,
    /// Right camera retained pixel buffer.
    pub right: crate::metal_interop::RetainedCVPixelBuffer,
}
