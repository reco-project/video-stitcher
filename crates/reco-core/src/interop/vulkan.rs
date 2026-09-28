//! Vulkan side of CUDA/Vulkan interop.
//!
//! Allocates Vulkan image memory, exports it for CUDA writes, then wraps
//! the `VkImage` into a [`wgpu::Texture`] via the HAL escape hatch.
//!
//! The flow:
//! 1. Create a linear `VkImage` and query its allocation requirements
//! 2. Allocate dedicated image memory and export a POSIX fd
//! 3. CUDA imports and maps the allocation
//! 4. Wraps the `VkImage` into `wgpu::Texture` via `create_texture_from_hal`
//!
//! ## References
//! - [Gyroflow](https://github.com/gyroflow/gyroflow) for the general CUDA/Vulkan interop approach
//! - `VK_KHR_external_memory_fd` specification
//! - wgpu HAL interop API (`texture_from_raw`, `create_texture_from_hal`)

use crate::gpu::GpuContext;
use crate::interop::cuda::{CudaImageMemory, CudaInteropError};

pub(crate) fn require_matching_cuda_device(gpu: &GpuContext) -> Result<(), CudaInteropError> {
    let render_uuid = unsafe {
        let device = gpu
            .device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let mut id = ash::vk::PhysicalDeviceIDProperties::default();
        let mut properties = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        device
            .shared_instance()
            .raw_instance()
            .get_physical_device_properties2(device.raw_physical_device(), &mut properties);
        id.device_uuid
    };
    let decode_uuid = crate::interop::cuda::default_cuda_device_uuid()?;
    // Reject other render devices without creating a CUDA context on this thread.
    if render_uuid != decode_uuid {
        return Err(CudaInteropError::DeviceMismatch);
    }
    matching_devices(
        render_uuid,
        decode_uuid,
        crate::interop::cuda::get_cuda_device_uuid()?,
    )
}

fn matching_devices(
    render: [u8; 16],
    decode: [u8; 16],
    allocation: [u8; 16],
) -> Result<(), CudaInteropError> {
    if render == decode && render == allocation {
        Ok(())
    } else {
        Err(CudaInteropError::DeviceMismatch)
    }
}

/// A wgpu texture backed by CUDA shared memory.
///
/// The texture retains both the CUDA mapping and Vulkan allocation until
/// its last GPU use, releasing the mapping before the allocation.
pub struct SharedTexture {
    /// The wgpu texture, usable in bind groups and render passes.
    pub texture: wgpu::Texture,
    /// The CUDA device pointer to the shared memory.
    /// Used for `cuMemcpy2D` from NVDEC output to this texture.
    pub cuda_ptr: crate::interop::cuda::CUdeviceptr,
    /// Pitch (row stride in bytes) of the Vulkan image.
    /// May differ from `width * bpp` due to alignment requirements.
    pub pitch: usize,
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
/// - `DeviceMismatch` if allocation, decode and rendering devices differ
/// - `CudaError` if CUDA import or mapping fails
/// - `VulkanError` if image allocation, export or initialization fails
#[cfg(target_os = "linux")]
pub fn create_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<SharedTexture, CudaInteropError> {
    use ash::vk;
    use std::os::fd::{FromRawFd, OwnedFd};
    use wgpu::hal::api::Vulkan;

    require_matching_cuda_device(gpu)?;

    // Access the raw Vulkan device through wgpu's HAL
    let (vk_image, device_memory, layout, shared_mem) = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let hal_device = &*hal_device_guard;
        if !hal_device
            .enabled_device_extensions()
            .contains(&ash::khr::external_memory_fd::NAME)
        {
            return Err(CudaInteropError::VulkanError(
                "external memory fd export is unavailable".into(),
            ));
        }
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
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_info);

        let mut external_query = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let query = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(vk_format)
            .ty(image_info.image_type)
            .tiling(image_info.tiling)
            .usage(image_info.usage)
            .push_next(&mut external_query);
        let mut external_properties = vk::ExternalImageFormatProperties::default();
        let mut properties =
            vk::ImageFormatProperties2::default().push_next(&mut external_properties);
        hal_device
            .shared_instance()
            .raw_instance()
            .get_physical_device_image_format_properties2(physical_device, &query, &mut properties)
            .map_err(|error| {
                CudaInteropError::VulkanError(format!("external image support: {error:?}"))
            })?;
        let features = external_properties
            .external_memory_properties
            .external_memory_features;
        if !features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE) {
            return Err(CudaInteropError::VulkanError(
                "linear image does not support CUDA memory export".into(),
            ));
        }

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

        // Prefer device-local memory; unified-memory devices may expose
        // another compatible type for external images.
        let memory_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                let type_bits = 1 << i;
                let is_supported = (mem_reqs.memory_type_bits & type_bits) != 0;
                let props = mem_props.memory_types[i as usize].property_flags;
                is_supported && props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .or_else(|| {
                log::warn!(
                    "No DEVICE_LOCAL memory type for shared image, \
                         falling back to any supported type (unified memory GPU?)"
                );
                (0..mem_props.memory_type_count)
                    .find(|&i| (mem_reqs.memory_type_bits & (1 << i)) != 0)
            })
            .ok_or_else(|| {
                raw_device.destroy_image(vk_image, None);
                CudaInteropError::VulkanError("no compatible memory type for shared image".into())
            })?;

        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(vk_image);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_info)
            .push_next(&mut dedicated);

        let device_memory = raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
            raw_device.destroy_image(vk_image, None);
            CudaInteropError::VulkanError(format!("vkAllocateMemory (export): {e:?}"))
        })?;

        // Bind before exporting so both APIs refer to the same dedicated image.
        raw_device
            .bind_image_memory(vk_image, device_memory, 0)
            .map_err(|e| {
                raw_device.destroy_image(vk_image, None);
                raw_device.free_memory(device_memory, None);
                CudaInteropError::VulkanError(format!("vkBindImageMemory: {e:?}"))
            })?;

        let export = ash::khr::external_memory_fd::Device::new(
            hal_device.shared_instance().raw_instance(),
            raw_device,
        );
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(device_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let shared_mem = export
            .get_memory_fd(&fd_info)
            .map_err(|error| CudaInteropError::VulkanError(format!("vkGetMemoryFdKHR: {error:?}")))
            .and_then(|fd| CudaImageMemory::import(OwnedFd::from_raw_fd(fd), mem_reqs.size))
            .inspect_err(|_| {
                raw_device.destroy_image(vk_image, None);
                raw_device.free_memory(device_memory, None);
            })?;

        log::info!(
            "Vulkan image shared with CUDA: {}x{} {:?}, pitch={}",
            width,
            height,
            format,
            layout.row_pitch,
        );

        (vk_image, device_memory, layout, shared_mem)
    };

    // Wrap the VkImage into a wgpu texture via HAL.
    //
    // External memory bypasses wgpu's allocator. Its retirement callback owns
    // the CUDA mapping, image and allocation so cloned texture handles stay valid.
    let device_for_drop = gpu.device.clone();
    let cuda_ptr = shared_mem.device_ptr + layout.offset;
    let drop_image = vk_image;
    let drop_memory = device_memory;
    let drop_callback: Box<dyn FnOnce() + Send + Sync> = Box::new(move || {
        // Release the CUDA view before its Vulkan allocation, including when
        // wgpu keeps the texture alive past the source's last handle.
        drop(shared_mem);
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

    // SAFETY: we've created a valid VkImage backed by shared memory,
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

    let texture = SharedTexture {
        texture: wgpu_texture,
        cuda_ptr,
        pitch: layout.row_pitch as usize,
    };
    // Complete the initial UNDEFINED transition before CUDA writes. Otherwise
    // the first wgpu use may discard those writes, even though CUDA finished.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.transition_resources(
        std::iter::empty(),
        std::iter::once(wgpu::TextureTransition {
            texture: &texture.texture,
            selector: None,
            state: wgpu::TextureUses::RESOURCE,
        }),
    );
    let submission = gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|error| {
            CudaInteropError::VulkanError(format!("initialize shared texture: {error}"))
        })?;
    Ok(texture)
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
    use std::sync::{
        Once,
        atomic::{AtomicUsize, Ordering},
    };

    struct ValidationLog(AtomicUsize);
    static VALIDATION: ValidationLog = ValidationLog(AtomicUsize::new(0));

    impl log::Log for ValidationLog {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Warn
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                eprintln!("{}: {}", record.target(), record.args());
                if (record.level() == log::Level::Error && record.target().starts_with("wgpu"))
                    || record.target().contains("interop::cuda")
                {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        fn flush(&self) {}
    }

    fn capture_validation_errors() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            log::set_logger(&VALIDATION).unwrap();
            log::set_max_level(log::LevelFilter::Warn);
        });
    }

    #[test]
    fn device_identity_requires_all_three_devices() {
        assert!(matching_devices([1; 16], [1; 16], [1; 16]).is_ok());
        for (render, decode, allocation) in [(1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 2, 3)] {
            assert!(matches!(
                matching_devices([render; 16], [decode; 16], [allocation; 16]),
                Err(CudaInteropError::DeviceMismatch)
            ));
        }
    }

    fn assert_sampled_pixels(gpu: &GpuContext, texture: &SharedTexture, expected: &[u8], bpp: u32) {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("interop_sample_check"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
                @group(0) @binding(0) var image: texture_2d<f32>;
                @group(0) @binding(1) var<storage, read_write> samples: array<vec4<f32>>;
                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let pos = id.xy * (textureDimensions(image) - vec2<u32>(1)) / 15u;
                    samples[id.y * 16u + id.x] = textureLoad(image, pos, 0);
                }
            "#
                    .into(),
                ),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("interop_sample_check"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16 * 16 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let view = texture.texture.create_view(&Default::default());
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(2, 2, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        gpu.queue.submit([encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let bytes = readback.slice(..).get_mapped_range();
        let (sample_bytes, channels) = match texture.texture.format() {
            wgpu::TextureFormat::R8Unorm => (1, 1),
            wgpu::TextureFormat::Rg8Unorm => (1, 2),
            wgpu::TextureFormat::R16Unorm => (2, 1),
            wgpu::TextureFormat::Rg16Unorm => (2, 2),
            _ => unreachable!(),
        };
        for y in 0..16 {
            for x in 0..16 {
                let px = x * (texture.texture.width() - 1) as usize / 15;
                let py = y * (texture.texture.height() - 1) as usize / 15;
                let offset = (py * texture.texture.width() as usize + px) * bpp as usize;
                for channel in 0..channels {
                    let sample = offset + channel * sample_bytes;
                    let expected = if sample_bytes == 1 {
                        f32::from(expected[sample]) / 255.0
                    } else {
                        f32::from(u16::from_le_bytes(
                            expected[sample..sample + 2].try_into().unwrap(),
                        )) / 65535.0
                    };
                    let actual = (y * 16 + x) * 16 + channel * 4;
                    let actual = f32::from_le_bytes(bytes[actual..actual + 4].try_into().unwrap());
                    assert!(
                        (actual - expected).abs() < 0.00001,
                        "sample ({px},{py}) channel {channel}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA and a non-matching Vulkan adapter"]
    fn mismatched_device_is_rejected_before_import() {
        capture_validation_errors();
        let gpu = GpuContext::new_blocking().unwrap();
        assert!(!gpu.supports_zero_copy());
        assert!(matches!(
            create_shared_texture(&gpu, 128, 96, wgpu::TextureFormat::R8Unorm),
            Err(CudaInteropError::DeviceMismatch)
        ));
        drop(gpu);
        assert_eq!(VALIDATION.0.load(Ordering::Relaxed), 0);
    }

    #[test]
    #[ignore = "requires NVIDIA CUDA and Vulkan external-memory interop"]
    fn shared_texture_preserves_cuda_pixels() {
        use crate::interop::cuda::{cuda_ensure_context, cuda_memcpy_htod_2d, cuda_synchronize};
        capture_validation_errors();
        let gpu = GpuContext::new_blocking().unwrap();
        assert!(
            gpu.supports_zero_copy(),
            "select the default CUDA device's Vulkan adapter"
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
            let retained = SharedTexture {
                texture: texture.texture.clone(),
                cuda_ptr: texture.cuda_ptr,
                pitch: texture.pitch,
            };
            drop(texture);
            let texture = retained;
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
                assert_sampled_pixels(&gpu, &texture, &expected, bpp);
            }
            std::thread::spawn(move || drop(texture)).join().unwrap();
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }
        drop(gpu);
        assert_eq!(VALIDATION.0.load(Ordering::Relaxed), 0);
    }
}
