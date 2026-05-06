//! DMA-buf import for NVMM zero-copy camera capture on Jetson.
//!
//! Imports NV12 frames from NVIDIA's NVMM (NvBufSurface) directly into
//! wgpu textures via Vulkan's `VK_EXT_external_memory_dma_buf` extension.
//! The ISP writes NV12 to NVMM buffers backed by DMA-buf fds; this module
//! imports those fds as Vulkan images for the stitch shader to sample.

use std::collections::HashMap;

use crate::gpu::GpuContext;
use ash::vk;
use thiserror::Error;
use wgpu::hal::api::Vulkan;

#[derive(Debug, Error)]
pub enum DmaBufImportError {
    #[error("wgpu backend is not Vulkan")]
    NotVulkan,
    #[error("Vulkan error: {0}")]
    Vulkan(String),
    #[error("dup() failed: {0}")]
    DupFd(std::io::Error),
}

/// Y + UV textures imported from a single NV12 DMA-buf.
pub struct DmaBufNv12Textures {
    pub y_texture: wgpu::Texture,
    pub uv_texture: wgpu::Texture,
}

/// Caches Vulkan textures by DMA-buf fd to avoid per-frame Vulkan
/// object creation. NVMM ISP uses a rotating pool of ~4 buffers per
/// camera; each buffer has a stable DMA-buf fd. On first encounter we
/// import (dup + vkAllocateMemory + vkCreateImage), on subsequent
/// frames we reuse the cached wgpu::Texture. The ISP writes new pixel
/// content to the same physical memory and the GPU sees it
/// automatically (shared LPDDR5, no separate VRAM).
#[derive(Default)]
pub struct DmaBufTextureCache {
    cache: HashMap<i32, DmaBufNv12Textures>,
}

impl DmaBufTextureCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::with_capacity(8),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Import the DMA-buf if not already cached. Call this for each fd
    /// before calling [`get`] to borrow the textures.
    pub fn ensure_imported(
        &mut self,
        gpu: &GpuContext,
        fd: i32,
        width: u32,
        height: u32,
        y_offset: u32,
        uv_offset: u32,
        total_size: u32,
    ) -> Result<(), DmaBufImportError> {
        if !self.cache.contains_key(&fd) {
            crate::profile_scope!("dmabuf_cache_miss");
            let textures =
                import_dmabuf_nv12(gpu, fd, width, height, y_offset, uv_offset, total_size)?;
            log::info!(
                "DMA-buf cache: imported fd={} ({}x{} NV12, pool size: {})",
                fd,
                width,
                height,
                self.cache.len() + 1,
            );
            self.cache.insert(fd, textures);
        }
        Ok(())
    }

    /// Borrow cached textures for an already-imported fd.
    ///
    /// Panics if the fd was not previously imported via [`ensure_imported`].
    pub fn get(&self, fd: i32) -> &DmaBufNv12Textures {
        self.cache
            .get(&fd)
            .expect("DmaBufTextureCache::get called before ensure_imported")
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }
}

/// Import an NV12 DMA-buf as two wgpu textures (Y: R8, UV: RG8).
///
/// The `fd` is dup'd internally so the caller can close its copy after
/// this returns. The Vulkan driver holds the DMA-buf reference.
pub fn import_dmabuf_nv12(
    gpu: &GpuContext,
    fd: i32,
    width: u32,
    height: u32,
    y_offset: u32,
    uv_offset: u32,
    total_size: u32,
) -> Result<DmaBufNv12Textures, DmaBufImportError> {
    crate::profile_scope!("dmabuf_import_nv12");
    let y_fd = unsafe { libc::dup(fd) };
    if y_fd < 0 {
        return Err(DmaBufImportError::DupFd(std::io::Error::last_os_error()));
    }
    let uv_fd = unsafe { libc::dup(fd) };
    if uv_fd < 0 {
        unsafe { libc::close(y_fd) };
        return Err(DmaBufImportError::DupFd(std::io::Error::last_os_error()));
    }

    let y_texture = import_single_plane(
        gpu,
        y_fd,
        width,
        height,
        wgpu::TextureFormat::R8Unorm,
        y_offset,
        total_size,
        "nvmm_y",
    )?;

    let uv_texture = import_single_plane(
        gpu,
        uv_fd,
        width / 2,
        height / 2,
        wgpu::TextureFormat::Rg8Unorm,
        uv_offset,
        total_size,
        "nvmm_uv",
    )?;

    Ok(DmaBufNv12Textures {
        y_texture,
        uv_texture,
    })
}

fn import_single_plane(
    gpu: &GpuContext,
    fd: i32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    offset: u32,
    total_size: u32,
    label: &'static str,
) -> Result<wgpu::Texture, DmaBufImportError> {
    let (vk_image, device_memory) = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(DmaBufImportError::NotVulkan)?;
        let hal_device = &*hal_device_guard;
        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();

        let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(match format {
                wgpu::TextureFormat::R8Unorm => vk::Format::R8_UNORM,
                wgpu::TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
                _ => {
                    return Err(DmaBufImportError::Vulkan(format!(
                        "unsupported format: {format:?}"
                    )));
                }
            })
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_info);

        let vk_image = raw_device
            .create_image(&image_info, None)
            .map_err(|e| DmaBufImportError::Vulkan(format!("vkCreateImage: {e:?}")))?;

        let mem_reqs = raw_device.get_image_memory_requirements(vk_image);

        let mem_props = {
            let instance = hal_device.shared_instance().raw_instance();
            instance.get_physical_device_memory_properties(physical_device)
        };

        let memory_type_index = (0..mem_props.memory_type_count)
            .find(|&i| (mem_reqs.memory_type_bits & (1 << i)) != 0)
            .ok_or_else(|| {
                DmaBufImportError::Vulkan("no compatible memory type for DMA-buf import".into())
            })?;

        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(total_size as u64)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info);

        let device_memory = raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
            DmaBufImportError::Vulkan(format!("vkAllocateMemory (DMA-buf fd={fd}): {e:?}"))
        })?;

        raw_device
            .bind_image_memory(vk_image, device_memory, offset as u64)
            .map_err(|e| DmaBufImportError::Vulkan(format!("vkBindImageMemory: {e:?}")))?;

        (vk_image, device_memory)
    };

    let device_for_drop = gpu.device.clone();
    let drop_image = vk_image;
    let drop_memory = device_memory;
    let drop_callback: Box<dyn FnOnce() + Send + Sync> = Box::new(move || unsafe {
        if let Some(hal_device) = device_for_drop.as_hal::<Vulkan>() {
            let raw = hal_device.raw_device();
            raw.destroy_image(drop_image, None);
            raw.free_memory(drop_memory, None);
        }
    });

    let wgpu_texture = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(DmaBufImportError::NotVulkan)?;

        let hal_texture = hal_device_guard.texture_from_raw(
            vk_image,
            &wgpu::hal::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUses::RESOURCE,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            Some(drop_callback),
            wgpu::hal::vulkan::TextureMemory::External,
        );

        drop(hal_device_guard);

        gpu.device.create_texture_from_hal::<Vulkan>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )
    };

    Ok(wgpu_texture)
}
