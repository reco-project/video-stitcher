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
    /// DMA-buf -> Vulkan texture cache for the NVMM zero-copy path
    /// (Jetson). Keyed by DMA-buf fd; the ISP rotates a small fd pool
    /// so each buffer is imported once.
    #[cfg(target_os = "linux")]
    pub(crate) nvmm_cache: Option<crate::interop::dmabuf::DmaBufTextureCache>,
    /// NvBufSurfTransform letterbox surfaces for NVMM detection
    /// (left, right). Render and detection consume different handles
    /// of the same NVMM frame: the DMA-buf fd renders, the raw
    /// NvBufSurface pointer letterboxes here for the detector.
    #[cfg(target_os = "linux")]
    pub(crate) nvmm_det: Option<(
        crate::nvbuf_transform::NvBufDetectionSurface,
        crate::nvbuf_transform::NvBufDetectionSurface,
    )>,
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
