//! Vulkan side of CUDA/Vulkan interop.
//!
//! Imports CUDA-exported shared memory into Vulkan, then wraps the
//! resulting `VkImage` into a [`wgpu::Texture`] via the HAL escape hatch.
//!
//! The flow:
//! 1. CUDA allocates shareable memory and exports a POSIX fd
//! 2. This module creates a `VkImage` with `VK_KHR_external_memory_fd`
//! 3. Imports the fd as the backing memory for the image
//! 4. Wraps the `VkImage` into `wgpu::Texture` via `create_texture_from_hal`
//!
//! ## References
//! - Gyroflow `wgpu_interop_vulkan.rs` and `wgpu_interop_cuda.rs`
//! - `VK_KHR_external_memory_fd` specification
//! - wgpu HAL interop API (`texture_from_raw`, `create_texture_from_hal`)

use crate::cuda_interop::{CudaInteropError, CudaSharedMemory};
use crate::gpu::GpuContext;

/// A wgpu texture backed by CUDA shared memory.
///
/// Owns both the wgpu texture and the underlying CUDA allocation.
/// When dropped, the wgpu texture is released first, then the CUDA memory.
pub struct SharedTexture {
    /// The wgpu texture, usable in bind groups and render passes.
    pub texture: wgpu::Texture,
    /// The CUDA device pointer to the shared memory.
    /// Used for `cuMemcpy2D` from NVDEC output to this texture.
    pub cuda_ptr: crate::cuda_interop::CUdeviceptr,
    /// Pitch (row stride in bytes) of the Vulkan image.
    /// May differ from `width * bpp` due to alignment requirements.
    pub pitch: usize,
    /// Keep the shared memory alive (dropped after texture).
    _shared_mem: CudaSharedMemory,
}

/// Create a wgpu texture backed by CUDA shared memory.
///
/// This is the main entry point for zero-copy interop. The returned texture
/// can be used in wgpu bind groups just like any other texture, but its memory
/// is shared with CUDA — writes via `cuMemcpy2D` are visible to wgpu.
///
/// # Arguments
/// - `gpu`: the wgpu GPU context (must be Vulkan backend)
/// - `width`, `height`: texture dimensions in pixels
/// - `format`: wgpu texture format (e.g. `R8Unorm` for Y/U/V planes)
///
/// # Errors
/// - `NotVulkan` if the wgpu backend is not Vulkan
/// - `CudaError` if shared memory allocation fails
/// - `VulkanError` if Vulkan image creation or memory import fails
#[cfg(target_os = "linux")]
pub fn create_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<SharedTexture, CudaInteropError> {
    use ash::vk;
    use wgpu::hal::api::Vulkan;

    let bpp = format_bytes_per_pixel(format);

    // Allocate row-aligned: Vulkan may require specific row pitch alignment.
    // Start with a generous pitch (aligned to 256 bytes, common GPU requirement).
    let row_bytes = width as usize * bpp;
    let pitch = (row_bytes + 255) & !255; // align to 256
    let alloc_size = pitch * height as usize;

    let shared_mem = crate::cuda_interop::allocate_shared_memory(alloc_size)?;
    let cuda_ptr = shared_mem.device_ptr;
    let fd = shared_mem.shared_handle;

    // Access the raw Vulkan device through wgpu's HAL
    let (vk_image, device_memory, actual_pitch) = unsafe {
        gpu.device.as_hal::<Vulkan, _, _>(|hal_device| {
            let hal_device = hal_device.ok_or(CudaInteropError::NotVulkan)?;
            let raw_device = hal_device.raw_device();
            let physical_device = hal_device.raw_physical_device();
            let vk_format = wgpu_format_to_vk(format);

            // Create VkImage with external memory support
            let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk_format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::PREINITIALIZED)
                .push_next(&mut external_info);

            let vk_image = raw_device
                .create_image(&image_info, None)
                .map_err(|e| CudaInteropError::VulkanError(format!("vkCreateImage: {e:?}")))?;

            // Get memory requirements
            let mem_reqs = raw_device.get_image_memory_requirements(vk_image);

            // Get actual row pitch from the image layout
            let subresource = vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                array_layer: 0,
            };
            let layout = raw_device.get_image_subresource_layout(vk_image, subresource);
            let actual_pitch = layout.row_pitch as usize;

            // Find a DEVICE_LOCAL memory type
            let mem_props = {
                let instance = hal_device.shared_instance().raw_instance();
                instance.get_physical_device_memory_properties(physical_device)
            };

            let memory_type_index = (0..mem_props.memory_type_count)
                .find(|&i| {
                    let type_bits = 1 << i;
                    let is_supported = (mem_reqs.memory_type_bits & type_bits) != 0;
                    let props = mem_props.memory_types[i as usize].property_flags;
                    is_supported && props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .ok_or_else(|| {
                    CudaInteropError::VulkanError(
                        "no DEVICE_LOCAL memory type for imported image".into(),
                    )
                })?;

            // Import the CUDA fd as Vulkan memory
            let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
                .fd(fd);

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(shared_mem.alloc_size as u64)
                .memory_type_index(memory_type_index)
                .push_next(&mut import_info);

            let device_memory = raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
                CudaInteropError::VulkanError(format!("vkAllocateMemory (import fd): {e:?}"))
            })?;

            // Bind the imported memory to the image
            raw_device
                .bind_image_memory(vk_image, device_memory, 0)
                .map_err(|e| CudaInteropError::VulkanError(format!("vkBindImageMemory: {e:?}")))?;

            log::info!(
                "Vulkan image created: {}x{} {:?}, pitch={}, imported fd={}",
                width,
                height,
                format,
                actual_pitch,
                fd
            );

            Ok((vk_image, device_memory, actual_pitch))
        })
    }?;

    // Wrap the VkImage into a wgpu texture via HAL.
    //
    // We provide a drop_callback that destroys the VkImage and frees the
    // imported VkDeviceMemory. This way wgpu won't try to manage the memory
    // through its own allocator, and cleanup happens when the texture is dropped.
    // Build the drop callback outside the unsafe block to avoid nested-unsafe warning.
    // This closure is called by wgpu when the texture is dropped.
    let device_for_drop = gpu.device.clone();
    let drop_image = vk_image;
    let drop_memory = device_memory;
    let drop_callback: Box<dyn FnOnce() + Send + Sync> = Box::new(move || {
        // SAFETY: these Vulkan resources are no longer referenced after
        // the wgpu texture is dropped.
        unsafe {
            device_for_drop.as_hal::<Vulkan, _, _>(|hal_device| {
                if let Some(hal_device) = hal_device {
                    let raw = hal_device.raw_device();
                    raw.destroy_image(drop_image, None);
                    raw.free_memory(drop_memory, None);
                }
            });
        }
    });

    // SAFETY: we've created a valid VkImage backed by imported CUDA memory,
    // and the drop_callback will clean up both when the texture is released.
    let wgpu_texture = unsafe {
        let hal_texture = <Vulkan as wgpu::hal::Api>::Device::texture_from_raw(
            vk_image,
            &wgpu::hal::TextureDescriptor {
                label: Some("cuda_shared"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_DST,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            Some(drop_callback),
        );

        gpu.device.create_texture_from_hal::<Vulkan>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("cuda_shared"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )
    };

    Ok(SharedTexture {
        texture: wgpu_texture,
        cuda_ptr,
        pitch: actual_pitch,
        _shared_mem: shared_mem,
    })
}

/// Bytes per pixel for the texture formats we use.
fn format_bytes_per_pixel(format: wgpu::TextureFormat) -> usize {
    match format {
        wgpu::TextureFormat::R8Unorm => 1,
        wgpu::TextureFormat::Rg8Unorm => 2,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => 4,
        _ => panic!("unsupported format for CUDA interop: {format:?}"),
    }
}

/// Map wgpu texture format to Vulkan format.
fn wgpu_format_to_vk(format: wgpu::TextureFormat) -> ash::vk::Format {
    match format {
        wgpu::TextureFormat::R8Unorm => ash::vk::Format::R8_UNORM,
        wgpu::TextureFormat::Rg8Unorm => ash::vk::Format::R8G8_UNORM,
        wgpu::TextureFormat::Rgba8Unorm => ash::vk::Format::R8G8B8A8_UNORM,
        wgpu::TextureFormat::Rgba8UnormSrgb => ash::vk::Format::R8G8B8A8_SRGB,
        _ => panic!("unsupported format for Vulkan interop: {format:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_shared_texture() {
        if !crate::cuda_interop::is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let gpu = match pollster::block_on(GpuContext::new()) {
            Ok(g) => g,
            Err(e) => {
                println!("Skipping: no GPU: {e}");
                return;
            }
        };

        // Only works on Vulkan backend
        if gpu.adapter_info.backend != wgpu::Backend::Vulkan {
            println!(
                "Skipping: not Vulkan backend ({:?})",
                gpu.adapter_info.backend
            );
            return;
        }

        let tex = create_shared_texture(&gpu, 1920, 1080, wgpu::TextureFormat::R8Unorm)
            .expect("should create shared texture");

        println!(
            "Shared texture created: {}x{}, cuda_ptr=0x{:x}, pitch={}",
            1920, 1080, tex.cuda_ptr, tex.pitch
        );

        // Verify the texture is usable by creating a view
        let _view = tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        println!("Texture view created successfully");
    }
}
