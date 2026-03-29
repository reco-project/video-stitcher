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

/// Errors that can occur during GPU initialization.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No compatible GPU adapter found.
    #[error("no compatible GPU adapter found")]
    NoAdapter,

    /// Failed to request a GPU adapter.
    #[error("failed to request GPU adapter: {0}")]
    AdapterRequest(#[from] wgpu::RequestAdapterError),

    /// Failed to request a GPU device.
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
}

/// Shared GPU context used by all pipeline stages.
///
/// Created once at startup and passed to the pipeline, scene renderer,
/// and viewport modules. Wrapping in `Arc` is left to the caller.
pub struct GpuContext {
    /// The wgpu device handle.
    pub device: wgpu::Device,
    /// The command submission queue.
    pub queue: wgpu::Queue,
    /// Information about the selected adapter.
    pub adapter_info: wgpu::AdapterInfo,
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

        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        let instance = wgpu::Instance::new(desc);

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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("reco"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
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
        } else {
            // Linux, Android, etc.
            wgpu::Backends::VULKAN
        }
    }

    /// The name of the selected GPU adapter (e.g. "NVIDIA GeForce RTX 5070").
    pub fn gpu_name(&self) -> &str {
        &self.adapter_info.name
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
}
