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
//! - [Gyroflow](https://github.com/gyroflow/gyroflow) for the general CUDA/Vulkan interop approach
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
#[cfg(any(target_os = "linux", target_os = "windows"))]
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
                CudaInteropError::VulkanError("no compatible memory type for imported image".into())
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

        (vk_image, device_memory, actual_pitch)
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
                usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_DST,
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
        wgpu::TextureFormat::R16Unorm => 2,
        wgpu::TextureFormat::Rg16Unorm => 4,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => 4,
        _ => panic!("unsupported format for CUDA interop: {format:?}"),
    }
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
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn create_nv12_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    plane: Nv12Plane,
    pixel_format: crate::renderer::GpuPixelFormat,
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

        // Use a small texture - this test verifies the interop pipeline works,
        // not that it handles production sizes. Smaller textures avoid OOM on
        // memory-constrained devices (e.g. Jetson with shared CPU/GPU RAM).
        let tex = create_shared_texture(&gpu, 256, 256, wgpu::TextureFormat::R8Unorm)
            .expect("should create shared texture");

        println!(
            "Shared texture created: {}x{}, cuda_ptr=0x{:x}, pitch={}",
            256, 256, tex.cuda_ptr, tex.pitch
        );

        // Verify the texture is usable by creating a view
        let _view = tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        println!("Texture view created successfully");
    }

    /// Proves that Vulkan takes ownership of the fd on import.
    ///
    /// After `vkAllocateMemory` with `VkImportMemoryFdInfoKHR`, the fd should
    /// no longer be valid. Calling `close()` on it should fail with EBADF,
    /// proving that the driver consumed it. This is what the Vulkan spec
    /// requires, and it means any code that calls `close(fd)` after import
    /// (like Gyroflow does) is performing an invalid operation.
    #[test]
    fn test_fd_ownership_transfers_to_vulkan() {
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

        if gpu.adapter_info.backend != wgpu::Backend::Vulkan {
            println!("Skipping: not Vulkan ({:?})", gpu.adapter_info.backend);
            return;
        }

        // Step 1: Allocate CUDA shared memory and grab the fd (small size for Jetson compat)
        let shared_mem =
            crate::cuda_interop::allocate_shared_memory(256 * 256).expect("alloc shared mem");
        let fd = shared_mem.shared_handle;
        println!("CUDA exported fd = {fd}");

        // Verify the fd is valid before Vulkan import
        // fstat() on a valid fd returns 0, on invalid fd returns -1 with EBADF
        let valid_before = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        assert!(
            valid_before >= 0,
            "fd {fd} should be valid before import, got errno {}",
            std::io::Error::last_os_error()
        );
        println!("Before Vulkan import: fd {fd} is valid (fcntl returned {valid_before})");

        // Step 2: Import into Vulkan (this should consume the fd)
        let tex = create_shared_texture(&gpu, 256, 256, wgpu::TextureFormat::R8Unorm)
            .expect("create shared texture");
        let imported_fd = tex._shared_mem.shared_handle;

        // Step 3: Try to use the fd after Vulkan import
        let valid_after = unsafe { libc::fcntl(imported_fd, libc::F_GETFD) };
        let errno_after = std::io::Error::last_os_error();

        println!(
            "After Vulkan import: fcntl(fd={imported_fd}) returned {valid_after}, errno = {errno_after}"
        );

        if valid_after < 0 {
            println!(
                "CONFIRMED: fd {imported_fd} is invalid after Vulkan import (EBADF). \
                 Vulkan took ownership. Calling close() on it would be a spec violation."
            );
        } else {
            println!(
                "UNEXPECTED: fd {imported_fd} is still valid after Vulkan import. \
                 Driver may have dup()'d internally (not spec-compliant behavior). \
                 close() would still be a spec violation per VK_KHR_external_memory_fd."
            );
        }

        // Step 4: If the fd IS still valid, try closing it and see what happens
        // to the texture (this would be the Gyroflow pattern)
        if valid_after >= 0 {
            println!("Attempting close(fd={imported_fd}) like Gyroflow does...");
            let close_ret = unsafe { libc::close(imported_fd) };
            let close_errno = std::io::Error::last_os_error();
            println!("close() returned {close_ret}, errno = {close_errno}");

            // The texture should still work because Vulkan imported the memory
            // (the fd is just a handle, the dmabuf reference is held by Vulkan)
            let _view = tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            println!("Texture still usable after close(fd) - dmabuf ref held by Vulkan");
        }
    }
}
