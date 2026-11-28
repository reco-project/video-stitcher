//! GPU device initialization and resource management.
//!
//! Handles [`wgpu`] adapter selection, device creation, and provides the
//! [`GpuContext`] that all pipeline stages share.
//!
//! ## Platform Support
//!
//! `wgpu` selects the best available backend per platform:
//! - **Linux**: Vulkan
//! - **macOS/iOS**: Metal
//! - **Windows**: Vulkan or DirectX 12
//! - **Headless (Jetson, CI)**: Vulkan with no surface
//!
//! Override backend selection with `WGPU_BACKEND=vulkan|dx12|metal|gl`.

use thiserror::Error;

/// Output pixel format for the render target.
///
/// Wraps the subset of [`wgpu::TextureFormat`] variants actually used by
/// the stitching pipeline. Headless consumers use this instead of depending
/// on `wgpu` directly. Windowed consumers that need the surface's native
/// format can pass a raw `wgpu::TextureFormat` via the re-export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// 8-bit RGBA, linear (typical for encoding).
    Rgba8Unorm,
    /// 8-bit RGBA, sRGB (typical for on-screen display).
    Rgba8UnormSrgb,
    /// 8-bit BGRA, sRGB (some surface formats on macOS/Windows).
    Bgra8UnormSrgb,
    /// Packed 10-bit RGB with 2-bit alpha. A session delivers P010 with this target.
    Rgb10a2Unorm,
}

impl From<OutputFormat> for wgpu::TextureFormat {
    fn from(fmt: OutputFormat) -> Self {
        match fmt {
            OutputFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
            OutputFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            OutputFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            OutputFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        }
    }
}

/// Errors that can occur during GPU initialization.
///
/// All variants are `Clone + Send + Sync` so callers that post
/// results to worker-thread channels (reco-gui export thread,
/// live-calibration worker) can carry the error as a typed enum
/// rather than the legacy `Result<_, String>` pattern. The wgpu
/// request errors are flattened to `String` at the `From` boundary
/// because `wgpu::Request*Error` is not `Clone`.
#[derive(Debug, Clone, Error)]
pub enum GpuError {
    /// No compatible GPU adapter found.
    #[error("no compatible GPU adapter found")]
    NoAdapter,

    /// Failed to request a GPU adapter.
    #[error("failed to request GPU adapter: {0}")]
    AdapterRequest(String),

    /// Failed to request a GPU device.
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(String),

    /// Buffer mapping failed (GPU readback).
    #[error("GPU buffer map failed: {0}")]
    BufferMapFailed(String),
}

impl From<wgpu::RequestAdapterError> for GpuError {
    fn from(e: wgpu::RequestAdapterError) -> Self {
        Self::AdapterRequest(e.to_string())
    }
}

impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::DeviceRequest(e.to_string())
    }
}

/// Information about a surface's capabilities, returned by
/// [`GpuContext::for_surface`].
pub struct SurfaceInfo {
    /// The preferred texture format for this surface.
    pub format: wgpu::TextureFormat,
    /// Supported alpha compositing modes.
    pub alpha_modes: Vec<wgpu::CompositeAlphaMode>,
}

/// Snapshot of the lookahead image-memory allowance for a [`GpuContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookaheadMemoryBudget {
    /// Usable byte allowance for retained lookahead frames.
    pub usable_bytes: usize,
    /// Driver-reported free GPU memory (or host-clamped headroom on unified
    /// memory) in bytes.
    pub reported_free_bytes: u64,
    /// Total GPU memory (or physical unified RAM on Apple Silicon) in bytes.
    pub total_bytes: u64,
    /// Whether this adapter shares unified CPU+GPU memory.
    pub unified_memory: bool,
}

/// Shared GPU context used by all pipeline stages.
///
/// Created once at startup and passed to the pipeline, scene renderer,
/// and viewport modules. Wrapping in `Arc` is left to the caller.
///
/// Headless consumers create this with [`GpuContext::new`]. Windowed
/// consumers that need surface compatibility use [`GpuContext::for_surface`].
/// The context retains the instance and adapter that created its device so
/// windowed renderers can use the same handles. Clones share those handles
/// and the command queue without allocating GPU resources.
#[derive(Clone)]
pub struct GpuContext {
    /// The wgpu device handle.
    pub(crate) device: wgpu::Device,
    /// The command submission queue.
    pub(crate) queue: wgpu::Queue,
    /// Information about the selected adapter.
    pub(crate) adapter_info: wgpu::AdapterInfo,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
}

impl GpuContext {
    /// Initialize a GPU context using the platform backend policy.
    ///
    /// Enables supported optional video texture formats and clamps device
    /// limits to the adapter's capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if adapter selection or device creation fails.
    pub async fn new() -> Result<Self, GpuError> {
        Self::request(Self::create_instance(), None).await
    }

    /// Blocking convenience wrapper for [`Self::new`].
    ///
    /// Uses `pollster` internally so callers in synchronous contexts
    /// (OBS plugin callbacks, CLI entry points, test harnesses) can
    /// construct a context without pulling pollster in themselves or
    /// spinning up a runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if adapter selection or device creation fails.
    pub fn new_blocking() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    fn create_instance() -> wgpu::Instance {
        let backends = Self::select_backends();
        log::debug!("wgpu backends: {backends:?}");
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        })
    }

    async fn request_adapter(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<wgpu::Adapter, GpuError> {
        Ok(instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
            })
            .await?)
    }

    async fn request(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, GpuError> {
        let adapter = Self::request_adapter(&instance, surface).await?;
        let limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        Self::from_adapter(instance, adapter, limits).await
    }

    /// Request a device with the supported video texture features and limits
    /// capped at what the adapter exposes.
    async fn from_adapter(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        limits: wgpu::Limits,
    ) -> Result<Self, GpuError> {
        let adapter_info = adapter.get_info();
        log::info!(
            "Selected GPU: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let features = video_texture_features(adapter.features());
        if !features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
            log::warn!(
                "GPU does not support 16-bit texture formats (TEXTURE_FORMAT_16BIT_NORM). \
                 10-bit video input (P010/HEVC) will not be available."
            );
        }
        log::debug!("Requesting device with features: {features:?}");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("reco"),
                required_features: features,
                required_limits: clamp_limits(&limits, &adapter.limits()),
                ..Default::default()
            })
            .await?;
        log::debug!("Device created successfully");

        Ok(Self {
            instance,
            adapter,
            adapter_info,
            device,
            queue,
        })
    }

    /// Create a context with reduced limits to exercise consumer validation.
    #[cfg(test)]
    pub(crate) async fn with_test_limits(limits: wgpu::Limits) -> Result<Self, GpuError> {
        let instance = Self::create_instance();
        let adapter = Self::request_adapter(&instance, None).await?;
        Self::from_adapter(instance, adapter, limits).await
    }

    /// Select wgpu backends based on environment and platform.
    ///
    /// Checks `WGPU_BACKEND` first (user override). Otherwise uses
    /// platform defaults: DX12 on Windows, Vulkan on Linux, Metal
    /// on macOS.
    ///
    /// Use this policy when creating an instance before a window surface.
    pub fn select_backends() -> wgpu::Backends {
        if let Ok(val) = std::env::var("WGPU_BACKEND") {
            match val.to_lowercase().as_str() {
                "vulkan" | "vk" => return wgpu::Backends::VULKAN,
                "dx12" | "d3d12" => return wgpu::Backends::DX12,
                "metal" | "mtl" => return wgpu::Backends::METAL,
                "gl" | "opengl" => return wgpu::Backends::GL,
                _ => log::warn!("Unknown WGPU_BACKEND={val:?}, using platform default"),
            }
        }

        if cfg!(target_os = "windows") {
            // DX12 only — some AMD Vulkan drivers crash during instance
            // creation (STATUS_HEAP_CORRUPTION). Users can opt into
            // Vulkan via WGPU_BACKEND=vulkan if their driver supports it.
            wgpu::Backends::DX12
        } else if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
            wgpu::Backends::METAL
        } else if Self::is_v3d_gpu() {
            // Broadcom V3D (Raspberry Pi 5): Vulkan driver renders black
            // frames due to a V3DV driver bug. Use GL instead.
            log::info!("Detected V3D GPU (RPi), using GL backend to avoid Vulkan driver bug");
            wgpu::Backends::GL
        } else {
            // Linux, Android, etc.
            wgpu::Backends::VULKAN
        }
    }

    /// Detect Broadcom V3D GPU (Raspberry Pi 5) via sysfs.
    ///
    /// V3D's Vulkan driver has a rendering bug that produces black frames.
    /// When detected, we auto-select the GL backend instead.
    fn is_v3d_gpu() -> bool {
        // V3D creates /sys/devices/platform/*.v3d on RPi5
        std::path::Path::new("/sys/bus/platform/drivers/v3d").exists()
    }

    /// Initialize a GPU context for a windowed surface.
    ///
    /// Uses the same video texture features and device limits as [`Self::new`],
    /// while selecting an adapter that can present to the given surface.
    /// The surface must belong to the supplied instance.
    ///
    /// ```rust,ignore
    /// let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
    ///     backends: GpuContext::select_backends(),
    ///     ..Default::default()
    /// });
    /// let surface = instance.create_surface(window)?;
    /// let (gpu, surface_info) = GpuContext::for_surface(&instance, &surface).await?;
    /// ```
    pub async fn for_surface(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Result<(Self, SurfaceInfo), GpuError> {
        let ctx = Self::request(instance.clone(), Some(surface)).await?;
        let caps = surface.get_capabilities(ctx.adapter());
        let surface_info = SurfaceInfo {
            format: caps.formats.first().copied().ok_or(GpuError::NoAdapter)?,
            alpha_modes: caps.alpha_modes,
        };
        Ok((ctx, surface_info))
    }

    /// The name of the selected GPU adapter (e.g. "NVIDIA GeForce RTX 5070").
    pub fn gpu_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Best-effort query of (free, total) VRAM in bytes for the selected
    /// adapter, or `None` if this backend exposes no budget API.
    ///
    /// On Linux NVIDIA this uses the CUDA driver (`cuMemGetInfo`), whose
    /// `free` figure is device-wide and therefore counts the decode pool
    /// and TensorRT engine alongside the wgpu allocations. On Windows it
    /// uses the DXGI budget (`IDXGIAdapter3::QueryVideoMemoryInfo`) via the
    /// DX12 HAL adapter. On macOS/iOS Metal it queries
    /// `recommendedMaxWorkingSetSize` and `currentAllocatedSize`; on
    /// unified-memory Apple Silicon devices, `free` is additionally clamped by
    /// actual host free+reclaimable RAM from `host_statistics64` so callers do
    /// not mistake unused GPU working-set headroom for spare physical RAM.
    /// Other backends return `None`; callers must fall back to allocation-time
    /// OOM handling when this returns `None`. The figure is a snapshot: it can
    /// shift as other processes allocate, so it is a sizing aid, not a
    /// guarantee.
    pub fn available_vram(&self) -> Option<(u64, u64)> {
        #[cfg(target_os = "linux")]
        {
            if let Ok((free, total)) = crate::interop::cuda::cuda_mem_get_info() {
                return Some((free as u64, total as u64));
            }
        }
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::Graphics::Dxgi::{
                DXGI_MEMORY_SEGMENT_GROUP, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
            };
            // The DXGI budget API is reachable only through the DX12 backend.
            if self.adapter_info.backend != wgpu::Backend::Dx12 {
                return None;
            }
            // SAFETY: downcast to the DX12 HAL adapter; the returned guard
            // borrows the adapter and lives until the end of this scope, so the
            // `&DxgiAdapter` from raw_adapter() stays valid for the query.
            let hal_adapter = unsafe { self.adapter().as_hal::<wgpu::hal::api::Dx12>()? };
            let dxgi = hal_adapter.raw_adapter();
            let query = |group: DXGI_MEMORY_SEGMENT_GROUP| -> Option<(u64, u64)> {
                let info = dxgi.query_video_memory_info(group).ok()?;
                Some((info.Budget.saturating_sub(info.CurrentUsage), info.Budget))
            };
            // wgpu allocates GpuOnly textures against the LOCAL segment. On
            // discrete GPUs that is dedicated VRAM. On integrated/UMA parts
            // (e.g. AMD iGPUs) the real budget can live in NON_LOCAL (shared
            // system RAM), so consult both and keep the larger budget - the
            // allocation-time catch handles any over-estimate.
            let local = query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
            if self.adapter_info.device_type == wgpu::DeviceType::IntegratedGpu {
                let non_local = query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
                return match (local, non_local) {
                    (Some(l), Some(nl)) => Some(if nl.1 > l.1 { nl } else { l }),
                    (l, None) => l,
                    (None, nl) => nl,
                };
            }
            return local;
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let (metal_free, metal_total, unified) = self.metal_memory_info()?;
            if unified {
                let (host_available, _host_total) = macos_host_memory()?;
                return Some((metal_free.min(host_available), metal_total));
            }
            return Some((metal_free, metal_total));
        }
        #[allow(unreachable_code)]
        None
    }

    /// Whether the selected GPU shares unified memory with the CPU
    /// (e.g. Apple Silicon).
    pub fn has_unified_memory(&self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some((_, _, unified)) = self.metal_memory_info() {
            return unified;
        }
        self.adapter_info.device_type == wgpu::DeviceType::IntegratedGpu
    }

    /// Compute the lookahead image-memory budget for native GPU-resident
    /// frames, accounting for unified CPU+GPU memory on Apple Silicon.
    ///
    /// On discrete GPUs (Linux CUDA, Windows DX12, Intel Mac dGPUs), this
    /// applies the dedicated-VRAM budget policy to [`Self::available_vram`].
    /// On unified-memory Metal devices, Metal's `recommendedMaxWorkingSetSize`
    /// does not reflect actual free system memory, so the budget is clamped
    /// against live host free+reclaimable RAM (`host_statistics64`) and a
    /// conservative share of physical RAM (`hw.memsize`), logging whenever
    /// unified-memory limits tighten the budget below Metal's working set.
    /// Returns `None` when no trustworthy budget query is available.
    pub fn lookahead_memory_budget(&self) -> Option<LookaheadMemoryBudget> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let (metal_free, metal_total, unified) = self.metal_memory_info()?;
            if unified {
                let Some((host_available, host_total)) = macos_host_memory() else {
                    log::warn!(
                        "Unified memory host-RAM query failed; refusing to trust Metal's \
                         {:.2} GB working-set limit as free system memory",
                        metal_total as f64 / 1e9,
                    );
                    return None;
                };
                let metal_only_budget =
                    crate::lookahead::discrete_vram_budget_bytes(metal_free, metal_total);
                let usable_bytes = crate::lookahead::unified_memory_budget_bytes(
                    metal_free,
                    metal_total,
                    host_available,
                    host_total,
                );
                if usable_bytes == 0 {
                    log::warn!(
                        "Unified memory lookahead budget clamped to 0 GB under memory pressure \
                         (host {:.2} GB available of {:.2} GB physical RAM; \
                         Metal working set {:.2} GB free of {:.2} GB)",
                        host_available as f64 / 1e9,
                        host_total as f64 / 1e9,
                        metal_free as f64 / 1e9,
                        metal_total as f64 / 1e9,
                    );
                } else if usable_bytes < metal_only_budget {
                    log::info!(
                        "Unified memory: clamping lookahead memory budget from {:.2} GB \
                         (Metal working set) to {:.2} GB (host {:.2} GB available of {:.2} GB \
                         physical RAM; Metal {:.2} GB free of {:.2} GB)",
                        metal_only_budget as f64 / 1e9,
                        usable_bytes as f64 / 1e9,
                        host_available as f64 / 1e9,
                        host_total as f64 / 1e9,
                        metal_free as f64 / 1e9,
                        metal_total as f64 / 1e9,
                    );
                } else {
                    log::info!(
                        "Unified memory lookahead budget: {:.2} GB \
                         (host {:.2} GB available of {:.2} GB physical RAM; \
                         Metal {:.2} GB free of {:.2} GB)",
                        usable_bytes as f64 / 1e9,
                        host_available as f64 / 1e9,
                        host_total as f64 / 1e9,
                        metal_free as f64 / 1e9,
                        metal_total as f64 / 1e9,
                    );
                }
                return Some(LookaheadMemoryBudget {
                    usable_bytes,
                    reported_free_bytes: metal_free.min(host_available),
                    total_bytes: host_total,
                    unified_memory: true,
                });
            }
            let usable_bytes =
                crate::lookahead::discrete_vram_budget_bytes(metal_free, metal_total);
            Some(LookaheadMemoryBudget {
                usable_bytes,
                reported_free_bytes: metal_free,
                total_bytes: metal_total,
                unified_memory: false,
            })
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            let (free, total) = self.available_vram()?;
            let usable_bytes = crate::lookahead::discrete_vram_budget_bytes(free, total);
            Some(LookaheadMemoryBudget {
                usable_bytes,
                reported_free_bytes: free,
                total_bytes: total,
                unified_memory: self.has_unified_memory(),
            })
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn metal_memory_info(&self) -> Option<(u64, u64, bool)> {
        if self.adapter_info.backend != wgpu::Backend::Metal {
            return None;
        }
        // SAFETY: downcast to the Metal HAL device; the guard borrows
        // self.device and lives until end of scope, so raw_device() is
        // valid for the property reads below.
        let hal_device = unsafe { self.device.as_hal::<wgpu::hal::api::Metal>()? };
        let raw = hal_device.raw_device();
        let total = raw.recommended_max_working_set_size();
        let used = raw.current_allocated_size();
        let unified = raw.has_unified_memory();
        Some((total.saturating_sub(used), total, unified))
    }

    /// The GPU backend name (e.g. "Vulkan", "Dx12", "Metal").
    pub fn backend_name(&self) -> &str {
        match self.adapter_info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Dx12 => "Dx12",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Gl => "OpenGL",
            _ => "Unknown",
        }
    }

    /// The GPU driver version string.
    pub fn driver_info(&self) -> &str {
        &self.adapter_info.driver_info
    }

    /// Whether the GPU backend is Vulkan (needed for CUDA/Vulkan interop).
    pub fn is_vulkan(&self) -> bool {
        self.adapter_info.backend == wgpu::Backend::Vulkan
    }

    /// Whether the GPU backend is Metal (needed for VideoToolbox interop).
    pub fn is_metal(&self) -> bool {
        self.adapter_info.backend == wgpu::Backend::Metal
    }

    /// Whether the GPU backend is DX12 (needed for D3D11VA interop on Windows).
    pub fn is_dx12(&self) -> bool {
        self.adapter_info.backend == wgpu::Backend::Dx12
    }

    /// Whether this GPU context supports zero-copy decode.
    ///
    /// On Linux, requires Vulkan to match both the current CUDA device and
    /// the default decode device. Other platforms require their native backend.
    /// Input formats and import capabilities are validated when opening a source.
    pub fn supports_zero_copy(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            match crate::interop::vulkan::require_matching_cuda_device(self) {
                Ok(()) => true,
                Err(error) => {
                    log::info!("CUDA zero-copy unavailable: {error}");
                    false
                }
            }
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            self.is_metal()
        }
        #[cfg(target_os = "windows")]
        {
            self.is_dx12()
        }
        #[cfg(not(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "windows"
        )))]
        {
            false
        }
    }

    /// Access the wgpu instance that selected this context's adapter.
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// Access the wgpu adapter that created this context's device.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Access the wgpu device handle.
    ///
    /// Windowed consumers need this for surface configuration.
    /// Headless consumers should not need direct device access.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the wgpu command queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Enable only video formats supported by the adapter. Native P010 is
/// needed for Windows D3D11VA imports. Other backends use 16-bit planes.
fn video_texture_features(supported: wgpu::Features) -> wgpu::Features {
    let formats = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM | wgpu::Features::TEXTURE_FORMAT_NV12;
    #[cfg(target_os = "windows")]
    let formats = formats | wgpu::Features::TEXTURE_FORMAT_P010;
    supported & formats
}

/// Clamp each limit field to what the adapter actually supports.
///
/// For "more is better" limits (max_*), takes the minimum of desired and
/// supported. For "less is better" limits (min_*), takes the maximum.
fn clamp_limits(desired: &wgpu::Limits, supported: &wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_1d: desired
            .max_texture_dimension_1d
            .min(supported.max_texture_dimension_1d),
        max_texture_dimension_2d: desired
            .max_texture_dimension_2d
            .min(supported.max_texture_dimension_2d),
        max_texture_dimension_3d: desired
            .max_texture_dimension_3d
            .min(supported.max_texture_dimension_3d),
        max_texture_array_layers: desired
            .max_texture_array_layers
            .min(supported.max_texture_array_layers),
        max_bind_groups: desired.max_bind_groups.min(supported.max_bind_groups),
        max_bindings_per_bind_group: desired
            .max_bindings_per_bind_group
            .min(supported.max_bindings_per_bind_group),
        max_dynamic_uniform_buffers_per_pipeline_layout: desired
            .max_dynamic_uniform_buffers_per_pipeline_layout
            .min(supported.max_dynamic_uniform_buffers_per_pipeline_layout),
        max_dynamic_storage_buffers_per_pipeline_layout: desired
            .max_dynamic_storage_buffers_per_pipeline_layout
            .min(supported.max_dynamic_storage_buffers_per_pipeline_layout),
        max_sampled_textures_per_shader_stage: desired
            .max_sampled_textures_per_shader_stage
            .min(supported.max_sampled_textures_per_shader_stage),
        max_samplers_per_shader_stage: desired
            .max_samplers_per_shader_stage
            .min(supported.max_samplers_per_shader_stage),
        max_storage_buffers_per_shader_stage: desired
            .max_storage_buffers_per_shader_stage
            .min(supported.max_storage_buffers_per_shader_stage),
        max_storage_textures_per_shader_stage: desired
            .max_storage_textures_per_shader_stage
            .min(supported.max_storage_textures_per_shader_stage),
        max_uniform_buffers_per_shader_stage: desired
            .max_uniform_buffers_per_shader_stage
            .min(supported.max_uniform_buffers_per_shader_stage),
        max_uniform_buffer_binding_size: desired
            .max_uniform_buffer_binding_size
            .min(supported.max_uniform_buffer_binding_size),
        max_storage_buffer_binding_size: desired
            .max_storage_buffer_binding_size
            .min(supported.max_storage_buffer_binding_size),
        max_vertex_buffers: desired.max_vertex_buffers.min(supported.max_vertex_buffers),
        max_vertex_attributes: desired
            .max_vertex_attributes
            .min(supported.max_vertex_attributes),
        max_vertex_buffer_array_stride: desired
            .max_vertex_buffer_array_stride
            .min(supported.max_vertex_buffer_array_stride),
        max_inter_stage_shader_components: desired
            .max_inter_stage_shader_components
            .min(supported.max_inter_stage_shader_components),
        max_color_attachments: desired
            .max_color_attachments
            .min(supported.max_color_attachments),
        max_color_attachment_bytes_per_sample: desired
            .max_color_attachment_bytes_per_sample
            .min(supported.max_color_attachment_bytes_per_sample),
        max_compute_workgroup_storage_size: desired
            .max_compute_workgroup_storage_size
            .min(supported.max_compute_workgroup_storage_size),
        max_compute_invocations_per_workgroup: desired
            .max_compute_invocations_per_workgroup
            .min(supported.max_compute_invocations_per_workgroup),
        max_compute_workgroup_size_x: desired
            .max_compute_workgroup_size_x
            .min(supported.max_compute_workgroup_size_x),
        max_compute_workgroup_size_y: desired
            .max_compute_workgroup_size_y
            .min(supported.max_compute_workgroup_size_y),
        max_compute_workgroup_size_z: desired
            .max_compute_workgroup_size_z
            .min(supported.max_compute_workgroup_size_z),
        max_compute_workgroups_per_dimension: desired
            .max_compute_workgroups_per_dimension
            .min(supported.max_compute_workgroups_per_dimension),
        max_buffer_size: desired.max_buffer_size.min(supported.max_buffer_size),
        // "less is better" limits: take the max (stricter alignment)
        min_uniform_buffer_offset_alignment: desired
            .min_uniform_buffer_offset_alignment
            .max(supported.min_uniform_buffer_offset_alignment),
        min_storage_buffer_offset_alignment: desired
            .min_storage_buffer_offset_alignment
            .max(supported.min_storage_buffer_offset_alignment),
        // Copy remaining fields from desired (zero-default features like mesh shaders, RT)
        ..*desired
    }
}

/// Query `(available, total)` host physical memory in bytes on macOS/iOS via
/// `sysctl(CTL_HW, HW_MEMSIZE)` and Mach `host_statistics64(HOST_VM_INFO64)`.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn macos_host_memory() -> Option<(u64, u64)> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }

    let mut host_total: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let mut mib = [libc::CTL_HW, libc::HW_MEMSIZE];
    // SAFETY: `mib` has 2 elements and `host_total` is a valid `u64` out-buffer of length `len`.
    let sysctl_ok = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            2,
            (&raw mut host_total).cast::<libc::c_void>(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        ) == 0
    };
    if !sysctl_ok || host_total == 0 {
        return None;
    }

    // SAFETY: `_SC_PAGESIZE` is a standard POSIX sysconf key.
    let page_size = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    if page_size == 0 {
        return None;
    }

    let mut stat = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count: libc::mach_msg_type_number_t = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: `mach_host_self` returns a send right to the host name port.
    #[allow(deprecated)]
    let host = unsafe { libc::mach_host_self() };
    // SAFETY: `stat` points to zeroed `vm_statistics64` storage of `HOST_VM_INFO64_COUNT` integers.
    let kr = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stat.as_mut_ptr().cast::<libc::integer_t>(),
            &raw mut count,
        )
    };
    // SAFETY: release the send right returned by `mach_host_self`.
    unsafe {
        #[allow(deprecated)]
        let task = libc::mach_task_self();
        let _ = mach_port_deallocate(task, host);
    }
    if kr != libc::KERN_SUCCESS {
        return None;
    }
    // SAFETY: `host_statistics64` returned `KERN_SUCCESS`, populating `stat`.
    let stat = unsafe { stat.assume_init() };
    let available = crate::lookahead::macos_vm_available_bytes(crate::lookahead::MacosVmSample {
        free_count: stat.free_count,
        speculative_count: stat.speculative_count,
        external_page_count: stat.external_page_count,
        purgeable_count: stat.purgeable_count,
        internal_page_count: stat.internal_page_count,
        wire_count: stat.wire_count,
        compressor_page_count: stat.compressor_page_count,
        page_size,
        host_total,
    });
    Some((available, host_total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_retains_the_selected_adapter_and_device_features() {
        let Some(ctx) = crate::stitch::test_support::gpu_or_skip() else {
            return;
        };
        assert_eq!(ctx.adapter_info, ctx.adapter().get_info());
        assert_eq!(
            ctx.device().features(),
            video_texture_features(ctx.adapter().features())
        );

        let shared = ctx.clone();
        assert_eq!(shared.instance(), ctx.instance());
        assert_eq!(shared.adapter(), ctx.adapter());
        assert_eq!(shared.device(), ctx.device());
        assert_eq!(shared.queue(), ctx.queue());
    }

    #[test]
    fn video_texture_features_are_optional_and_platform_specific() {
        use wgpu::Features;

        assert_eq!(video_texture_features(Features::empty()), Features::empty());
        for format in [
            Features::TEXTURE_FORMAT_16BIT_NORM,
            Features::TEXTURE_FORMAT_NV12,
        ] {
            assert_eq!(video_texture_features(format), format);
        }
        let p010 = Features::TEXTURE_FORMAT_P010;
        let expected = if cfg!(target_os = "windows") {
            p010
        } else {
            Features::empty()
        };
        assert_eq!(
            video_texture_features(p010 | Features::TIMESTAMP_QUERY),
            expected
        );
    }
}

pub mod color_grade;
pub mod rgba_readback;
pub mod vram_pool;
pub mod yuv_converter;
pub mod yuv_stack_packer;

#[cfg(target_os = "linux")]
pub(crate) fn wait_for_reads(device: &wgpu::Device) -> Result<(), String> {
    // No index means the most recent submission at the start of this poll,
    // which includes all reads submitted before the worker received the frame.
    match device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(_) => Ok(()),
        Err(error) => {
            // There is no index or timeout to reject. Do not recycle camera
            // memory on an unexpected backend failure before proving completion.
            while let Err(retry) = device.poll(wgpu::PollType::wait_indefinitely()) {
                log::error!("GPU completion failed: {retry}");
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(format!("GPU submission completion: {error}"))
        }
    }
}
