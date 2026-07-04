//! Resident-frame state the GPU executor owns: shared zero-copy decode
//! textures, the VRAM lookahead pool, and decode backpressure channels.
//!
//! Zero-copy sources hand the executor platform handles; everything
//! needed to render from them without a CPU round-trip lives here,
//! next to the pipeline that consumes them. A session that never
//! feeds resident frames allocates nothing here.

use crate::gpu::vram_pool::VramPool;

/// Lazily-populated residency state for one [`GpuExecutor`](super::GpuExecutor).
#[derive(Default)]
pub(crate) struct Residency {
    /// Bind groups for the shared zero-copy decode textures.
    #[cfg(target_os = "linux")]
    pub(crate) bind_groups: Option<crate::render::pipeline::GpuSourceBindGroups>,
    /// Slot-free senders for decode backpressure.
    #[cfg(target_os = "linux")]
    pub(crate) slot_free_tx: Option<(
        std::sync::mpsc::SyncSender<u8>,
        std::sync::mpsc::SyncSender<u8>,
    )>,
    /// CUDA buffer info for GPU detection on the shared textures.
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(crate) cuda_buf_info: Option<(
        crate::interop::zero_copy::GpuBufInfo,
        crate::interop::zero_copy::GpuBufInfo,
    )>,
    /// Views over the 8 shared textures, layout `[left_y_0, left_uv_0,
    /// left_y_1, left_uv_1, right_y_0, right_uv_0, right_y_1,
    /// right_uv_1]`. Views hold an Arc on the underlying texture, so
    /// the shared-memory lifetime stays bound to the source's
    /// [`SharedTextureSet`](crate::interop::SharedTextureSet).
    #[cfg(target_os = "linux")]
    pub(crate) shared_views: Option<[wgpu::TextureView; 8]>,
    /// The 8 shared textures (2 slots x 2 cameras x Y/UV), cloned for
    /// `copy_texture_to_texture` into the VRAM pool (cheap, Arc inside).
    #[cfg(target_os = "linux")]
    pub(crate) shared_textures: Option<[wgpu::Texture; 8]>,
    /// VRAM pool for GPU-resident lookahead buffering.
    pub(crate) pool: Option<VramPool>,
}
