//! Linux CUDA/Vulkan zero-copy types.
//!
//! Contains [`SharedTextureSet`], the bundle of double-buffered shared
//! textures + CUDA pointers + slot channels used by the GPU zero-copy
//! decode path. The decode side (reco-io) constructs it; the render
//! side consumes it via `StitchSession::setup_gpu_source`.

use crate::interop::vulkan::SharedTexture;
use crate::interop::zero_copy::GpuBufInfo;

/// One camera's double-buffered CUDA/Vulkan textures and decode backpressure.
/// Must outlive the decode workers that write these textures.
pub struct SharedCameraTextures {
    /// Y/UV planes for slots 0 and 1, in that order.
    pub textures: [SharedTexture; 4],
    /// CUDA destinations for this camera's decoder.
    pub buf: GpuBufInfo,
    /// Returns a consumed slot to the decoder.
    pub slot_free_tx: std::sync::mpsc::SyncSender<u8>,
}

/// Shared decode surfaces in projection input order.
/// The source owns these until its decode workers have joined.
pub struct SharedTextureSet {
    /// One double-buffered surface per input.
    pub cameras: Vec<SharedCameraTextures>,
}
