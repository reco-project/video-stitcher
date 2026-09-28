//! Vulkan side of CUDA/Vulkan interop.
//!
//! Imports CUDA-exported shared memory into Vulkan, then wraps the
//! resulting `VkImage` into a [`wgpu::Texture`] via the HAL escape hatch.
//!
//! The flow:
//! 1. Create a linear `VkImage` and query its allocation requirements
//! 2. CUDA allocates shareable memory and exports a POSIX fd
//! 3. Imports the fd as the backing memory for the image
//! 4. Wraps the `VkImage` into `wgpu::Texture` via `create_texture_from_hal`
//!
//! ## References
//! - [Gyroflow](https://github.com/gyroflow/gyroflow) for the general CUDA/Vulkan interop approach
//! - `VK_KHR_external_memory_fd` specification
//! - wgpu HAL interop API (`texture_from_raw`, `create_texture_from_hal`)

use crate::gpu::GpuContext;
use crate::interop::cuda::{CudaInteropError, CudaSharedMemory};

/// A wgpu texture backed by CUDA shared memory.
///
/// Owns both the wgpu texture and the underlying CUDA allocation.
/// When dropped, the wgpu texture is released first, then the CUDA memory.
pub struct SharedTexture {
    /// The wgpu texture, usable in bind groups and render passes.
    pub texture: wgpu::Texture,
    /// The CUDA device pointer to the shared memory.
    /// Used for `cuMemcpy2D` from NVDEC output to this texture.
    pub cuda_ptr: crate::interop::cuda::CUdeviceptr,
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
    use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
    use wgpu::hal::api::Vulkan;

    // Access the raw Vulkan device through wgpu's HAL
    let (vk_image, device_memory, layout, shared_mem) = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let hal_device = &*hal_device_guard;
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
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
            )
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

        // Find a DEVICE_LOCAL memory type
        let mem_props = {
            let instance = hal_device.shared_instance().raw_instance();
            instance.get_physical_device_memory_properties(physical_device)
        };

        // Prefer DEVICE_LOCAL, but fall back to any supported memory type.
        // On unified-memory GPUs (Jetson/Tegra) the driver may not flag
        // imported-fd-compatible types as DEVICE_LOCAL.
        let memory_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                let type_bits = 1 << i;
                let is_supported = (mem_reqs.memory_type_bits & type_bits) != 0;
                let props = mem_props.memory_types[i as usize].property_flags;
                is_supported && props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .or_else(|| {
                log::warn!(
                    "No DEVICE_LOCAL memory type for imported image, \
                         falling back to any supported type (unified memory GPU?)"
                );
                (0..mem_props.memory_type_count)
                    .find(|&i| (mem_reqs.memory_type_bits & (1 << i)) != 0)
            })
            .ok_or_else(|| {
                raw_device.destroy_image(vk_image, None);
                CudaInteropError::VulkanError("no compatible memory type for imported image".into())
            })?;

        // Use the image's requirements, not a guessed row alignment.
        let shared_mem = crate::interop::cuda::allocate_shared_memory(mem_reqs.size as usize)
            .inspect_err(|_| raw_device.destroy_image(vk_image, None))?;
        let fd = shared_mem.shared_handle;
        let fd_owner = OwnedFd::from_raw_fd(fd);

        // Each allocation backs exactly one image. Explicit dedication also
        // prevents zero-valued reads of large imported linear images; exercised
        // by shared_texture_preserves_cuda_pixels.
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(fd);

        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(vk_image);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(shared_mem.alloc_size as u64)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated);

        let device_memory = raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
            raw_device.destroy_image(vk_image, None);
            CudaInteropError::VulkanError(format!("vkAllocateMemory (import fd): {e:?}"))
        })?;
        // A successful import transfers descriptor ownership to Vulkan.
        let _ = fd_owner.into_raw_fd();

        // Bind the imported memory to the image
        raw_device
            .bind_image_memory(vk_image, device_memory, 0)
            .map_err(|e| {
                raw_device.destroy_image(vk_image, None);
                raw_device.free_memory(device_memory, None);
                CudaInteropError::VulkanError(format!("vkBindImageMemory: {e:?}"))
            })?;

        log::info!(
            "Vulkan image created: {}x{} {:?}, pitch={}, imported fd={}",
            width,
            height,
            format,
            layout.row_pitch,
            fd
        );

        (vk_image, device_memory, layout, shared_mem)
    };

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
            if let Some(hal_device) = device_for_drop.as_hal::<Vulkan>() {
                let raw = hal_device.raw_device();
                raw.destroy_image(drop_image, None);
                raw.free_memory(drop_memory, None);
            }
        }
    });

    // SAFETY: we've created a valid VkImage backed by imported CUDA memory,
    // and the drop_callback will clean up both when the texture is released.
    let wgpu_texture = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;

        let hal_texture = hal_device_guard.texture_from_raw(
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
                usage: wgpu::TextureUses::RESOURCE
                    | wgpu::TextureUses::COPY_DST
                    | wgpu::TextureUses::COPY_SRC,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            Some(drop_callback),
            wgpu::hal::vulkan::TextureMemory::External,
        );

        // Drop the HAL device guard before calling create_texture_from_hal
        drop(hal_device_guard);

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
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    };

    Ok(SharedTexture {
        texture: wgpu_texture,
        cuda_ptr: shared_mem.device_ptr + layout.offset,
        pitch: layout.row_pitch as usize,
        _shared_mem: shared_mem,
    })
}

/// Map wgpu texture format to Vulkan format.
fn wgpu_format_to_vk(format: wgpu::TextureFormat) -> ash::vk::Format {
    match format {
        wgpu::TextureFormat::R8Unorm => ash::vk::Format::R8_UNORM,
        wgpu::TextureFormat::Rg8Unorm => ash::vk::Format::R8G8_UNORM,
        wgpu::TextureFormat::R16Unorm => ash::vk::Format::R16_UNORM,
        wgpu::TextureFormat::Rg16Unorm => ash::vk::Format::R16G16_UNORM,
        wgpu::TextureFormat::Rgba8Unorm => ash::vk::Format::R8G8B8A8_UNORM,
        wgpu::TextureFormat::Rgba8UnormSrgb => ash::vk::Format::R8G8B8A8_SRGB,
        _ => panic!("unsupported format for Vulkan interop: {format:?}"),
    }
}

/// NV12 plane identifier for [`create_nv12_shared_texture`].
#[derive(Debug, Clone, Copy)]
pub enum Nv12Plane {
    /// Luminance plane (full resolution, `R8Unorm` or `R16Unorm` for 10-bit).
    Y,
    /// Chrominance plane (half resolution in each dimension, `Rg8Unorm` or `Rg16Unorm` for 10-bit).
    Uv,
}

/// Create a shared texture sized and formatted for an NV12 plane.
///
/// This is a convenience wrapper around [`create_shared_texture`] that
/// infers the wgpu format and dimensions from the plane type and pixel
/// format. The texture formats are determined by
/// `GpuPixelFormat::y_format()` and `GpuPixelFormat::uv_format()`.
///
/// Unorm normalization maps both 8-bit and 16-bit values to `[0.0, 1.0]`
/// in the shader, so the fragment shader works unchanged across formats.
#[cfg(target_os = "linux")]
pub fn create_nv12_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    plane: Nv12Plane,
    pixel_format: crate::render::renderer::GpuPixelFormat,
) -> Result<SharedTexture, CudaInteropError> {
    match plane {
        Nv12Plane::Y => create_shared_texture(gpu, width, height, pixel_format.y_format()),
        Nv12Plane::Uv => {
            create_shared_texture(gpu, width / 2, height / 2, pixel_format.uv_format())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires NVIDIA CUDA and Vulkan external-memory interop"]
    fn shared_texture_preserves_cuda_pixels() {
        use crate::interop::cuda::{cuda_ensure_context, cuda_memcpy_htod_2d, cuda_synchronize};
        let gpu = GpuContext::new_blocking().unwrap();
        assert!(
            gpu.is_vulkan() && gpu.adapter_info.vendor == 0x10de,
            "select an NVIDIA Vulkan adapter for CUDA external-memory interop"
        );
        cuda_ensure_context().unwrap();
        for (width, height, format, bpp) in [
            (128_u32, 96_u32, wgpu::TextureFormat::R8Unorm, 1),
            (960, 540, wgpu::TextureFormat::Rg8Unorm, 2),
            (7500, 2500, wgpu::TextureFormat::R8Unorm, 1),
            (3750, 1250, wgpu::TextureFormat::Rg8Unorm, 2),
            (7500, 2500, wgpu::TextureFormat::R16Unorm, 2),
            (3750, 1250, wgpu::TextureFormat::Rg16Unorm, 4),
        ] {
            let texture = create_shared_texture(&gpu, width, height, format).unwrap();
            let _view = texture.texture.create_view(&Default::default());
            let row_bytes = width * bpp;
            let stride = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("interop_pixel_check"),
                size: u64::from(stride) * u64::from(height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut expected = vec![0_u8; (row_bytes * height) as usize];
            for generation in 0..3 {
                for (index, byte) in expected.iter_mut().enumerate() {
                    *byte = ((index + generation * 79) % 251) as u8;
                }
                cuda_memcpy_htod_2d(
                    texture.cuda_ptr,
                    texture.pitch,
                    expected.as_ptr(),
                    row_bytes as usize,
                    row_bytes as usize,
                    height as usize,
                )
                .unwrap();
                cuda_synchronize().unwrap();
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                encoder.copy_texture_to_buffer(
                    texture.texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(stride),
                            rows_per_image: None,
                        },
                    },
                    texture.texture.size(),
                );
                gpu.queue.submit([encoder.finish()]);
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |result| result.unwrap());
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                {
                    let mapped = buffer.slice(..).get_mapped_range();
                    for (row, (actual, expected)) in mapped
                        .chunks_exact(stride as usize)
                        .zip(expected.chunks_exact(row_bytes as usize))
                        .enumerate()
                    {
                        assert!(
                            &actual[..row_bytes as usize] == expected,
                            "{width}x{height} {format:?}, generation {generation}, row {row}"
                        );
                    }
                }
                buffer.unmap();
            }
        }
    }
}
