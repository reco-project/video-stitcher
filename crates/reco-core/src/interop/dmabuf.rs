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
    /// before calling [`Self::get`] to borrow the textures.
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
        self.ensure_imported_tiling(
            gpu, fd, width, height, y_offset, uv_offset, total_size, false,
        )
    }

    /// Import with explicit tiling control.
    pub fn ensure_imported_tiling(
        &mut self,
        gpu: &GpuContext,
        fd: i32,
        width: u32,
        height: u32,
        y_offset: u32,
        uv_offset: u32,
        total_size: u32,
        linear: bool,
    ) -> Result<(), DmaBufImportError> {
        if !self.cache.contains_key(&fd) {
            crate::profile_scope!("dmabuf_cache_miss");
            let textures = import_dmabuf_nv12_tiling(
                gpu, fd, width, height, y_offset, uv_offset, total_size, linear,
            )?;
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
    /// Panics if the fd was not previously imported via [`Self::ensure_imported`].
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
    import_dmabuf_nv12_tiling(
        gpu, fd, width, height, y_offset, uv_offset, total_size, false,
    )
}

/// Import an NV12 DMA-buf with explicit tiling control.
///
/// `linear`: if true, uses `VK_IMAGE_TILING_LINEAR` (for ION/CMA buffers
/// where pixel data is stored row-by-row). If false, uses `OPTIMAL`
/// (for NVMM/tiled buffers where the driver handles layout).
pub fn import_dmabuf_nv12_tiling(
    gpu: &GpuContext,
    fd: i32,
    width: u32,
    height: u32,
    y_offset: u32,
    uv_offset: u32,
    total_size: u32,
    linear: bool,
) -> Result<DmaBufNv12Textures, DmaBufImportError> {
    crate::profile_scope!("dmabuf_import_nv12");

    let tiling = if linear {
        vk::ImageTiling::LINEAR
    } else {
        vk::ImageTiling::OPTIMAL
    };

    {
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
            tiling,
            "dmabuf_y",
        )?;
        let uv_texture = import_single_plane(
            gpu,
            uv_fd,
            width / 2,
            height / 2,
            wgpu::TextureFormat::Rg8Unorm,
            uv_offset,
            total_size,
            tiling,
            "dmabuf_uv",
        )?;
        Ok(DmaBufNv12Textures {
            y_texture,
            uv_texture,
        })
    }
}

/// Import a single DMA-BUF plane as a wgpu texture.
/// Use for platforms where Y and UV have separate fds (non-contiguous NV12).
pub fn import_dmabuf_plane(
    gpu: &GpuContext,
    fd: i32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    linear: bool,
) -> Result<wgpu::Texture, DmaBufImportError> {
    let dup_fd = unsafe { libc::dup(fd) };
    if dup_fd < 0 {
        return Err(DmaBufImportError::DupFd(std::io::Error::last_os_error()));
    }
    let tiling = if linear {
        vk::ImageTiling::LINEAR
    } else {
        vk::ImageTiling::OPTIMAL
    };
    import_single_plane(
        gpu,
        dup_fd,
        width,
        height,
        format,
        0,
        0,
        tiling,
        "dmabuf_plane",
    )
}

/// Cache for individually-imported DMA-BUF plane textures.
/// Used on platforms where NV12 Y and UV have separate fds.
#[derive(Default)]
pub struct DmaBufPlaneCache {
    cache: HashMap<i32, wgpu::Texture>,
}

impl DmaBufPlaneCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::with_capacity(8),
        }
    }

    pub fn ensure_imported(
        &mut self,
        gpu: &GpuContext,
        fd: i32,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        linear: bool,
    ) -> Result<(), DmaBufImportError> {
        if !self.cache.contains_key(&fd) {
            let tex = import_dmabuf_plane(gpu, fd, width, height, format, linear)?;
            log::info!(
                "DMA-BUF plane cache: imported fd={} {}x{} {:?} (pool: {})",
                fd,
                width,
                height,
                format,
                self.cache.len() + 1
            );
            self.cache.insert(fd, tex);
        }
        Ok(())
    }

    pub fn get(&self, fd: i32) -> &wgpu::Texture {
        self.cache
            .get(&fd)
            .expect("DmaBufPlaneCache::get before ensure_imported")
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }
}

fn import_single_plane(
    gpu: &GpuContext,
    fd: i32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    offset: u32,
    total_size: u32,
    tiling: vk::ImageTiling,
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
            .tiling(tiling)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::PREINITIALIZED)
            .push_next(&mut external_info);

        let vk_image = raw_device
            .create_image(&image_info, None)
            .map_err(|e| DmaBufImportError::Vulkan(format!("vkCreateImage: {e:?}")))?;

        let mem_reqs = raw_device.get_image_memory_requirements(vk_image);

        // Query the driver's expected row pitch for LINEAR tiling
        let subresource = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        let layout = raw_device.get_image_subresource_layout(vk_image, subresource);
        log::info!(
            "DMA-BUF import {label}: {}x{} {format:?} tiling={tiling:?} \
             mreq.size={} mreq.align={} mreq.type_bits=0x{:x} \
             layout: offset={} size={} rowPitch={} arrayPitch={} depthPitch={}",
            width,
            height,
            mem_reqs.size,
            mem_reqs.alignment,
            mem_reqs.memory_type_bits,
            layout.offset,
            layout.size,
            layout.row_pitch,
            layout.array_pitch,
            layout.depth_pitch,
        );

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

        // VK_KHR_dedicated_allocation: some drivers require this for imported memory
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(vk_image);

        // Vulkan spec requires allocationSize = lseek(fd, 0, SEEK_END) for DMA-BUF.
        let dmabuf_size = unsafe {
            let size = libc::lseek(fd, 0, libc::SEEK_END);
            libc::lseek(fd, 0, libc::SEEK_SET);
            if size > 0 { size as u64 } else { 0 }
        };
        let alloc_size = if total_size > 0 {
            total_size as u64
        } else if dmabuf_size > 0 {
            dmabuf_size
        } else {
            mem_reqs.size
        };
        log::info!(
            "DMA-BUF import {label}: allocationSize={alloc_size} (dmabuf_lseek={dmabuf_size} mreq.size={}) memTypeIdx={memory_type_index}",
            mem_reqs.size
        );
        // Some embedded Vulkan drivers look for VkImportMemoryFdInfoKHR in
        // dedicatedInfo->pNext instead of the main pNext chain. Pushing
        // import first then dedicated produces: alloc -> dedicated -> import,
        // which puts import_info as dedicated_info.pNext.
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(alloc_size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);

        let device_memory = raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
            DmaBufImportError::Vulkan(format!("vkAllocateMemory (DMA-buf fd={fd}): {e:?}"))
        })?;

        raw_device
            .bind_image_memory(vk_image, device_memory, offset as u64)
            .map_err(|e| DmaBufImportError::Vulkan(format!("vkBindImageMemory: {e:?}")))?;

        // Queue family ownership transfer from external producer (ISP/camera)
        // to our Vulkan queue. Without this, the GPU cache may not see the
        // DMA-BUF data written by the external producer.
        // VK_QUEUE_FAMILY_FOREIGN_EXT = 0xFFFFFFFE
        const VK_QUEUE_FAMILY_FOREIGN_EXT: u32 = !1u32; // 0xFFFFFFFE
        let queue_family = hal_device.queue_family_index();

        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::PREINITIALIZED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(VK_QUEUE_FAMILY_FOREIGN_EXT)
            .dst_queue_family_index(queue_family)
            .image(vk_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = raw_device
            .create_command_pool(&pool_info, None)
            .map_err(|e| DmaBufImportError::Vulkan(format!("cmd pool: {e:?}")))?;

        let alloc_cmd = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = raw_device
            .allocate_command_buffers(&alloc_cmd)
            .map_err(|e| DmaBufImportError::Vulkan(format!("cmd alloc: {e:?}")))?;
        let cmd = cmd_bufs[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        raw_device
            .begin_command_buffer(cmd, &begin)
            .map_err(|e| DmaBufImportError::Vulkan(format!("begin cmd: {e:?}")))?;
        raw_device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
        raw_device
            .end_command_buffer(cmd)
            .map_err(|e| DmaBufImportError::Vulkan(format!("end cmd: {e:?}")))?;

        let raw_queue = hal_device.raw_queue();
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        raw_device
            .queue_submit(raw_queue, &[submit], vk::Fence::null())
            .map_err(|e| DmaBufImportError::Vulkan(format!("queue submit: {e:?}")))?;
        raw_device
            .queue_wait_idle(raw_queue)
            .map_err(|e| DmaBufImportError::Vulkan(format!("queue wait: {e:?}")))?;

        raw_device.destroy_command_pool(cmd_pool, None);
        log::info!(
            "DMA-BUF import {label}: foreign queue family transfer done (foreign -> qf={queue_family})"
        );

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
                usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
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
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    };

    Ok(wgpu_texture)
}
