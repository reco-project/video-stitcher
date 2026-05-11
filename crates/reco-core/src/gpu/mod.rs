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
}

impl From<OutputFormat> for wgpu::TextureFormat {
    fn from(fmt: OutputFormat) -> Self {
        match fmt {
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

/// Shared GPU context used by all pipeline stages.
///
/// Created once at startup and passed to the pipeline, scene renderer,
/// and viewport modules. Wrapping in `Arc` is left to the caller.
///
/// Headless consumers create this with [`GpuContext::new`]. Windowed
/// consumers that need surface compatibility use [`GpuContext::for_surface`].
/// Cloning a `GpuContext` shares the same GPU device and queue (wgpu types
/// are internally reference-counted). No GPU resources are duplicated.
/// Multiple pipelines on a cloned context share the same command queue.
#[derive(Clone)]
pub struct GpuContext {
    /// The wgpu device handle.
    pub(crate) device: wgpu::Device,
    /// The command submission queue.
    pub(crate) queue: wgpu::Queue,
    /// Information about the selected adapter.
    pub(crate) adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize a GPU context, selecting the best available adapter.
    ///
    /// Requests a device with default limits and no required features beyond
    /// what `wgpu` provides by default. This works on all target platforms
    /// including headless environments (Jetson, CI).
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] if no compatible GPU is found.
    pub async fn new() -> Result<Self, GpuError> {
        Self::with_surface(None).await
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
    /// Returns [`GpuError::NoAdapter`] if no compatible GPU is found.
    pub fn new_blocking() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    /// Initialize a GPU context with an optional compatible surface.
    ///
    /// When a surface is provided, the adapter selection will prefer GPUs
    /// that can present to that surface (needed for windowed rendering).
    ///
    /// Respects `WGPU_BACKEND` environment variable for backend selection.
    /// On Windows, tries DX12 first if no backend is specified (Vulkan
    /// drivers on some AMD iGPUs crash during instance creation).
    pub async fn with_surface(surface: Option<&wgpu::Surface<'_>>) -> Result<Self, GpuError> {
        let backends = Self::select_backends();
        log::info!("wgpu backends: {:?}", backends);

        let desc = wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        };
        let instance = wgpu::Instance::new(&desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
            })
            .await?;

        let adapter_info = adapter.get_info();
        log::info!(
            "Selected GPU: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        // Request 16-bit texture formats for 10-bit video (P010) if the
        // adapter supports it. Not all backends do (e.g. GL on RPi5).
        let mut features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            features |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        } else {
            log::warn!(
                "GPU does not support 16-bit texture formats (TEXTURE_FORMAT_16BIT_NORM). \
                 10-bit video input (P010/HEVC) will not be available."
            );
        }
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
        {
            features |= wgpu::Features::TEXTURE_FORMAT_NV12;
        }

        let (device, queue) = Self::request_device_with_fallback(&adapter, features).await?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    /// Request a device, clamping requested limits to what the adapter
    /// actually supports.
    ///
    /// `downlevel_defaults()` can exceed what embedded GPUs report (e.g.
    /// VeriSilicon GC8000 supports `max_compute_workgroup_size_y = 128`
    /// but `downlevel_defaults` asks for 256). We take the minimum of
    /// our desired defaults and the adapter's reported limits for each
    /// field, ensuring we never ask for more than the hardware offers.
    async fn request_device_with_fallback(
        adapter: &wgpu::Adapter,
        features: wgpu::Features,
    ) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
        let supported = adapter.limits();
        let desired = wgpu::Limits::downlevel_defaults().using_resolution(supported.clone());
        let clamped = clamp_limits(&desired, &supported);

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("reco"),
                required_features: features,
                required_limits: clamped,
                ..Default::default()
            })
            .await
    }

    /// Select wgpu backends based on environment and platform.
    ///
    /// Checks `WGPU_BACKEND` first (user override). Otherwise uses
    /// platform defaults: DX12 on Windows, Vulkan on Linux, Metal
    /// on macOS.
    fn select_backends() -> wgpu::Backends {
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
    /// Creates a device compatible with the given surface and returns
    /// the surface's preferred format and alpha modes. The caller is
    /// responsible for creating the `wgpu::Instance` and `wgpu::Surface`
    /// (since surface creation requires a platform window handle).
    ///
    /// ```rust,ignore
    /// let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    /// let surface = instance.create_surface(window)?;
    /// let (gpu, surface_info) = GpuContext::for_surface(&instance, &surface).await?;
    /// ```
    pub async fn for_surface(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Result<(Self, SurfaceInfo), GpuError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(surface),
            })
            .await?;

        let adapter_info = adapter.get_info();
        log::info!(
            "Selected GPU: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let caps = surface.get_capabilities(&adapter);
        let surface_info = SurfaceInfo {
            format: caps.formats[0],
            alpha_modes: caps.alpha_modes,
        };

        let mut features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            features |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        } else {
            log::warn!(
                "GPU does not support 16-bit texture formats. \
                 10-bit video input (P010/HEVC) will not be available."
            );
        }
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
        {
            features |= wgpu::Features::TEXTURE_FORMAT_NV12;
        }

        let (device, queue) = Self::request_device_with_fallback(&adapter, features).await?;

        let ctx = Self {
            device,
            queue,
            adapter_info,
        };
        Ok((ctx, surface_info))
    }

    /// Create a GPU context from an existing wgpu device, queue, and adapter info.
    ///
    /// Use this when another framework (egui, bevy, etc.) already owns the
    /// GPU device and you want to share it with reco's stitching pipeline
    /// instead of creating a second device.
    ///
    /// The `adapter_info` must come from the same adapter that created the
    /// device, since pipeline features like zero-copy decode depend on the
    /// reported backend.
    ///
    /// ```rust,ignore
    /// // egui integration:
    /// let render_state = cc.egui_ctx.render_state().unwrap();
    /// let gpu = GpuContext::from_device_queue(
    ///     render_state.device.clone(),
    ///     render_state.queue.clone(),
    ///     render_state.adapter.get_info(),
    /// );
    /// ```
    pub fn from_device_queue(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
    ) -> Self {
        log::info!(
            "GpuContext from external device: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );
        Self {
            device,
            queue,
            adapter_info,
        }
    }

    /// The name of the selected GPU adapter (e.g. "NVIDIA GeForce RTX 5070").
    pub fn gpu_name(&self) -> &str {
        &self.adapter_info.name
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
    /// Checks that the GPU backend matches a supported interop path
    /// (Vulkan + CUDA on Linux, Metal on macOS) and that the necessary
    /// runtime libraries are available.
    pub fn supports_zero_copy(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.is_vulkan() && crate::interop::cuda::is_cuda_available()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_initializes() {
        // This test requires a GPU. In CI (headless), it is expected to fail
        // with NoAdapter or AdapterRequest — skip gracefully.
        let result = pollster::block_on(GpuContext::new());
        match result {
            Ok(ctx) => {
                assert!(!ctx.adapter_info.name.is_empty());
                log::info!("GPU test passed: {}", ctx.adapter_info.name);
            }
            Err(GpuError::NoAdapter | GpuError::AdapterRequest(_)) => {
                eprintln!("Skipping GPU test: no adapter available");
            }
            Err(e) => panic!("Unexpected GPU error: {e}"),
        }
    }

    #[test]
    fn gpu_context_from_device_queue() {
        // Create a normal context, then reconstruct from its parts.
        let result = pollster::block_on(GpuContext::new());
        let original = match result {
            Ok(ctx) => ctx,
            Err(GpuError::NoAdapter | GpuError::AdapterRequest(_)) => {
                eprintln!("Skipping GPU test: no adapter available");
                return;
            }
            Err(e) => panic!("Unexpected GPU error: {e}"),
        };

        let name = original.gpu_name().to_owned();
        let backend = original.backend_name().to_owned();
        let info = original.adapter_info;

        let reconstructed = GpuContext::from_device_queue(original.device, original.queue, info);

        assert_eq!(reconstructed.gpu_name(), name);
        assert_eq!(reconstructed.backend_name(), backend);
        // Device and queue are valid (moved, not cloned)
        let _ = reconstructed.device();
        let _ = reconstructed.queue();
    }
}

pub mod color_grade;
pub mod nv12_converter;
pub mod rgba_readback;
pub mod yuv_stack_packer;
