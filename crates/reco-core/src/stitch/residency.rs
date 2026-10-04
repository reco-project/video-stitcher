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
    pub(crate) slot_free_tx: Option<Vec<std::sync::mpsc::SyncSender<u8>>>,
    /// CUDA buffer info for GPU detection on the shared textures.
    #[cfg(target_os = "linux")]
    pub(crate) cuda_buf_info: Option<Vec<crate::interop::zero_copy::GpuBufInfo>>,
    /// Per-camera views: Y/UV for slots 0 and 1.
    #[cfg(target_os = "linux")]
    pub(crate) shared_views: Option<Vec<[wgpu::TextureView; 4]>>,
    /// Texture handles for device-local copies into the lookahead pool.
    #[cfg(target_os = "linux")]
    pub(crate) shared_textures: Option<Vec<[wgpu::Texture; 4]>>,
    /// DMA-buf -> Vulkan texture cache for the NVMM zero-copy path
    /// (Jetson). Reuses matching allocation identities and plane layouts.
    #[cfg(target_os = "linux")]
    pub(crate) nvmm_cache: Option<crate::interop::dmabuf::DmaBufTextureCache>,
    /// Keeps native inputs alive until their immediate GPU reads finish.
    #[cfg(target_os = "linux")]
    pub(crate) nvmm_completion: Option<super::nvmm_completion::NvmmCompletion>,
    /// CVPixelBuffer -> Metal texture import cache for the
    /// VideoToolbox zero-copy path.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) metal_cache: Option<crate::interop::metal::MetalTextureCache>,
    /// D3D11VA staging pool: SHARED_NTHANDLE textures that bridge
    /// FFmpeg's decode device to wgpu's DX12 device. Doubles as the
    /// lookahead buffer on Windows - slots are sized for peak
    /// occupancy instead of allocating a separate `VramPool`.
    #[cfg(target_os = "windows")]
    pub(crate) d3d11_staging: Option<crate::interop::d3d11::D3d11StagingPool>,
    /// VRAM pool for GPU-resident lookahead buffering.
    pub(crate) pool: Option<VramPool>,
}
