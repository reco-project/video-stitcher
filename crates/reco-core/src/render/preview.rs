//! Prepared camera images shared by panorama, lens and recording draws.

use std::sync::Arc;

use super::color::{YuvColor, YuvSampleFormat};
use super::pipeline::PipelineError;
use super::renderer::{
    GpuPixelFormat, InputFormat, VideoSampling, video_bind_group, video_plane, video_sampler,
    video_texture_layout, write_video_plane,
};
use crate::gpu::GpuContext;
use crate::source::FrameSet;

#[cfg(target_os = "windows")]
use crate::interop::d3d11::D3d11StagingPool;

/// One complete set of prepared camera images, in calibration projection order.
///
/// A frame owns a slot on the preparing GPU device, with each camera's plane
/// layout, sample depth, YUV matrix/range and sampling orientation. It can be
/// drawn by preview renderers on that device with matching camera dimensions.
/// Its storage is independent of viewport dimensions, pose and lens parameters.
///
/// Host planes are uploaded once into preview storage. CUDA and Metal planes
/// are copied once on the preview queue, before any draw that samples them.
/// CUDA leases and the retained CoreVideo buffers and texture wrappers survive
/// until the copy completes. Completion is driven even while playback is paused.
/// On D3D11 the frame holds the shared staging slot itself. The native stage
/// must complete before publication, and the slot cannot be overwritten while
/// a frame owns it or its GPU draws are outstanding.
///
/// Keep the displayed frame until preparation succeeds, then replace it with
/// the returned candidate. Invalid input or a failed import cannot change the
/// displayed frame. Paused redraws sample the same planes without uploads or
/// copies, even after the source stops or is dropped.
pub struct PreviewFrame {
    device: wgpu::Device,
    slot: Arc<Slot>,
    sampling: [VideoSampling; 2],
}

impl PreviewFrame {
    pub(crate) fn validate(
        &self,
        device: &wgpu::Device,
        sizes: &[(u32, u32)],
    ) -> Result<(), PipelineError> {
        if self.device != *device || self.slot.sizes != sizes {
            return Err(invalid(
                "preview frame device or camera dimensions do not match",
            ));
        }
        Ok(())
    }

    pub(crate) fn lens_camera(
        &self,
        device: &wgpu::Device,
        camera: usize,
        size: (u32, u32),
    ) -> Result<(&wgpu::BindGroup, VideoSampling), PipelineError> {
        if self.device != *device || self.slot.sizes.get(camera) != Some(&size) {
            return Err(invalid(
                "preview frame device, camera index or lens preview dimensions do not match",
            ));
        }
        Ok(self.camera(camera))
    }

    pub(crate) fn camera(&self, index: usize) -> (&wgpu::BindGroup, VideoSampling) {
        (&self.slot.bindings[index], self.sampling[index])
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Owned(InputFormat, GpuPixelFormat),
    #[cfg(target_os = "windows")]
    D3d11(u64),
}

struct Slot {
    layout: Layout,
    sizes: Vec<(u32, u32)>,
    textures: Vec<Vec<wgpu::Texture>>,
    bindings: Vec<wgpu::BindGroup>,
}

impl Slot {
    fn owned(
        gpu: &GpuContext,
        sizes: &[(u32, u32)],
        layout: Layout,
        format: InputFormat,
        depth: GpuPixelFormat,
    ) -> Self {
        let texture = |width, height, format| video_plane(gpu.device(), width, height, format);
        let textures: Vec<_> = sizes
            .iter()
            .map(|&(w, h)| {
                let planar = format == InputFormat::Yuv420p;
                let mut planes = vec![
                    texture(w, h, depth.y_format()),
                    texture(
                        w / 2,
                        h / 2,
                        if planar {
                            depth.y_format()
                        } else {
                            depth.uv_format()
                        },
                    ),
                ];
                if planar {
                    planes.push(texture(w / 2, h / 2, depth.y_format()));
                }
                planes
            })
            .collect();
        let bindings = bind_planes(
            gpu,
            textures.iter().map(|planes| {
                std::array::from_fn(|i| {
                    planes[i.min(planes.len() - 1)].create_view(&Default::default())
                })
            }),
        );
        Self {
            layout,
            sizes: sizes.to_vec(),
            textures,
            bindings,
        }
    }
}

fn bind_planes(
    gpu: &GpuContext,
    views: impl Iterator<Item = [wgpu::TextureView; 3]>,
) -> Vec<wgpu::BindGroup> {
    let layout = video_texture_layout(gpu.device());
    let sampler = video_sampler(gpu.device());
    views
        .map(|[y, u, v]| {
            video_bind_group(
                gpu.device(),
                &layout,
                &sampler,
                [&y, &u, &v],
                "preview_camera",
            )
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct PreviewStorage {
    slots: [Option<Arc<Slot>>; 2],
    #[cfg(test)]
    pub(super) plane_transfers: usize,
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
    completion: Option<std::sync::mpsc::SyncSender<()>>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    metal: Option<crate::interop::metal::MetalTextureCache>,
    #[cfg(target_os = "windows")]
    d3d11: Option<(u64, D3d11StagingPool)>,
}

impl PreviewStorage {
    pub(crate) fn prepare(
        &mut self,
        gpu: &GpuContext,
        sizes: &[(u32, u32)],
        frames: FrameSet,
        color: YuvColor,
        _rotations: [i32; 2],
    ) -> Result<PreviewFrame, PipelineError> {
        if sizes.iter().any(|&(w, h)| w % 2 != 0 || h % 2 != 0) {
            return Err(invalid("preview requires even camera dimensions"));
        }
        let index = self
            .slots
            .iter()
            .position(|slot| {
                slot.as_ref()
                    .is_none_or(|slot| Arc::strong_count(slot) == 1)
            })
            .ok_or_else(|| invalid("both preview frame slots are in use"))?;

        let mut sampling = [video_sampling(InputFormat::Nv12, GpuPixelFormat::Nv12, color, 0)?; 2];
        frames.validate_host_images(sizes).map_err(invalid)?;
        let slot = match frames {
            FrameSet::Yuv420p(images) => {
                sampling.fill(video_sampling(
                    InputFormat::Yuv420p,
                    GpuPixelFormat::Nv12,
                    color,
                    0,
                )?);
                let slot = self.owned(
                    index,
                    gpu,
                    sizes,
                    InputFormat::Yuv420p,
                    GpuPixelFormat::Nv12,
                );
                for (planes, image) in slot.textures.iter().zip(images) {
                    for (plane, bytes) in planes.iter().zip([&image.y, &image.u, &image.v]) {
                        write_video_plane(gpu, plane, bytes, plane.width());
                        #[cfg(test)]
                        {
                            self.plane_transfers += 1;
                        }
                    }
                }
                slot
            }
            FrameSet::Nv12(images) => {
                let slot = self.owned(index, gpu, sizes, InputFormat::Nv12, GpuPixelFormat::Nv12);
                for (planes, image) in slot.textures.iter().zip(images) {
                    write_video_plane(gpu, &planes[0], &image.y, planes[0].width());
                    write_video_plane(gpu, &planes[1], &image.uv, planes[1].width() * 2);
                    #[cfg(test)]
                    {
                        self.plane_transfers += 2;
                    }
                }
                slot
            }
            #[cfg(target_os = "linux")]
            FrameSet::GpuResident(images) => {
                check_count(images.len(), sizes)?;
                let format = images[0].pixel_format();
                for (i, image) in images.iter().enumerate() {
                    if image.device() != gpu.device() || image.pixel_format() != format {
                        return Err(invalid(format!(
                            "camera {i} CUDA image device or format does not match"
                        )));
                    }
                    sampling[i] = video_sampling(
                        InputFormat::Nv12,
                        format,
                        image.color(),
                        image.cuda_frame().rotation,
                    )?;
                }
                let slot = self.copy_native(
                    index,
                    gpu,
                    sizes,
                    format,
                    images.iter().map(|image| image.textures()),
                )?;
                gpu.queue().on_submitted_work_done(move || drop(images));
                self.drive_completion(gpu);
                slot
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            FrameSet::MetalResident(images) => {
                use crate::interop::metal::MetalTextureCache;
                check_count(images.len(), sizes)?;
                let (format, _) = images[0].video_layout().map_err(interop_error)?;
                validate_copy_format(gpu, format)?;
                for (i, image) in images.iter().enumerate() {
                    let (pixel_format, full_range) = image.video_layout().map_err(interop_error)?;
                    if pixel_format != format || (image.width(), image.height()) != sizes[i] {
                        return Err(invalid(format!(
                            "camera {i} Metal image format or size does not match"
                        )));
                    }
                    sampling[i] = video_sampling(
                        InputFormat::Nv12,
                        format,
                        YuvColor {
                            full_range,
                            ..color
                        },
                        _rotations[i],
                    )?;
                }
                let cache = match &mut self.metal {
                    Some(cache) => cache,
                    cache @ None => {
                        cache.insert(MetalTextureCache::new(gpu).map_err(interop_error)?)
                    }
                };
                let imported = cache.import_frame(&images, gpu).map_err(interop_error)?;
                let copied = self.copy_native(index, gpu, sizes, format, imported.textures());
                let slot = match copied {
                    Ok(slot) => slot,
                    Err(error) => {
                        imported.discard_unsubmitted();
                        return Err(error);
                    }
                };
                // Drop registers both native owners with the queue after the copy submission.
                drop(imported);
                self.drive_completion(gpu);
                slot
            }
            #[cfg(target_os = "windows")]
            FrameSet::D3d11Resident(images) => {
                check_count(images.len(), sizes)?;
                let format = if color.samples == YuvSampleFormat::P010 {
                    GpuPixelFormat::P010
                } else {
                    GpuPixelFormat::Nv12
                };
                for i in 0..images.len() {
                    sampling[i] = video_sampling(InputFormat::Nv12, format, color, _rotations[i])?;
                }
                self.stage_d3d11(index, gpu, sizes, &images, format)?
            }
            #[cfg(target_os = "linux")]
            FrameSet::NvmmResident(_) => return Err(invalid("NVMM is not a preview input")),
        };
        self.slots[index] = Some(Arc::clone(&slot));
        Ok(PreviewFrame {
            device: gpu.device().clone(),
            slot,
            sampling,
        })
    }

    fn owned(
        &self,
        index: usize,
        gpu: &GpuContext,
        sizes: &[(u32, u32)],
        format: InputFormat,
        depth: GpuPixelFormat,
    ) -> Arc<Slot> {
        let layout = Layout::Owned(format, depth);
        match &self.slots[index] {
            Some(slot) if slot.layout == layout => Arc::clone(slot),
            _ => Arc::new(Slot::owned(gpu, sizes, layout, format, depth)),
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
    fn copy_native<'a>(
        &mut self,
        index: usize,
        gpu: &GpuContext,
        sizes: &[(u32, u32)],
        format: GpuPixelFormat,
        images: impl ExactSizeIterator<Item = (&'a wgpu::Texture, &'a wgpu::Texture)> + Clone,
    ) -> Result<Arc<Slot>, PipelineError> {
        check_count(images.len(), sizes)?;
        for ((y, uv), &(w, h)) in images.clone().zip(sizes) {
            if y.format() != format.y_format()
                || uv.format() != format.uv_format()
                || (y.width(), y.height()) != (w, h)
                || (uv.width(), uv.height()) != (w / 2, h / 2)
                || [y, uv]
                    .iter()
                    .any(|plane| !plane.usage().contains(wgpu::TextureUsages::COPY_SRC))
            {
                return Err(invalid(
                    "native preview plane format, dimensions or copy usage do not match",
                ));
            }
        }
        validate_copy_format(gpu, format)?;
        let slot = self.owned(index, gpu, sizes, InputFormat::Nv12, format);
        if self.completion.is_none() {
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            let device = gpu.device().clone();
            std::thread::Builder::new()
                .name("preview-copy-completion".into())
                .spawn(move || {
                    while rx.recv().is_ok() {
                        poll_completion(&device);
                    }
                })
                .map_err(interop_error)?;
            self.completion = Some(tx);
        }
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview_native_copy"),
            });
        for ((y, uv), planes) in images.zip(&slot.textures) {
            for (source, target) in [y, uv].into_iter().zip(planes) {
                encoder.copy_texture_to_texture(
                    source.as_image_copy(),
                    target.as_image_copy(),
                    target.size(),
                );
                #[cfg(test)]
                {
                    self.plane_transfers += 1;
                }
            }
        }
        gpu.queue().submit(Some(encoder.finish()));
        Ok(slot)
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
    fn drive_completion(&self, gpu: &GpuContext) {
        use std::sync::mpsc::TrySendError;
        // A queued wake also covers callbacks registered before this send.
        let wake = self
            .completion
            .as_ref()
            .expect("native copy starts a poller");
        if let Err(TrySendError::Disconnected(())) = wake.try_send(()) {
            poll_completion(gpu.device());
        }
    }

    #[cfg(target_os = "windows")]
    fn stage_d3d11(
        &mut self,
        index: usize,
        gpu: &GpuContext,
        sizes: &[(u32, u32)],
        images: &[crate::source::D3d11CameraFrame],
        format: GpuPixelFormat,
    ) -> Result<Arc<Slot>, PipelineError> {
        let mut next = None;
        let (generation, pool, reusing) = match &mut self.d3d11 {
            Some((generation, pool))
                if pool.matches(sizes, self.slots.len(), format)
                    && pool
                        .matches_source_device(&images[0])
                        .map_err(interop_error)? =>
            {
                (*generation, pool, true)
            }
            previous => {
                let generation = previous
                    .as_ref()
                    .map_or(0, |(generation, _)| generation + 1);
                let pool = D3d11StagingPool::new(gpu, sizes, self.slots.len(), format)
                    .map_err(interop_error)?;
                let (_, pool) = next.insert((generation, pool));
                (generation, pool, false)
            }
        };
        let layout = Layout::D3d11(generation);
        let cached = self.slots[index]
            .as_ref()
            .filter(|slot| slot.layout == layout);
        if reusing {
            // External writes are outside wgpu queue ordering. The frame owner
            // is gone, but its submitted draws may still read this slot.
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(interop_error)?;
        }
        // Each image owns the producer pin through native query completion.
        pool.stage_frames(images, index).map_err(interop_error)?;
        let slot = cached.cloned().unwrap_or_else(|| {
            let views = pool.views(index).expect("completed staging slot");
            let bindings = bind_planes(gpu, views.into_iter().map(|(y, uv)| [y, uv.clone(), uv]));
            Arc::new(Slot {
                layout,
                sizes: sizes.to_vec(),
                textures: Vec::new(),
                bindings,
            })
        });
        if next.is_some() {
            self.d3d11 = next;
        }
        Ok(slot)
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
fn poll_completion(device: &wgpu::Device) {
    if let Err(error) = device.poll(wgpu::PollType::wait_indefinitely()) {
        log::error!("Preview copy completion: {error}");
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
fn validate_copy_format(gpu: &GpuContext, format: GpuPixelFormat) -> Result<(), PipelineError> {
    if format == GpuPixelFormat::P010
        && !gpu
            .device()
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
    {
        return Err(invalid("P010 preview requires 16-bit normalized textures"));
    }
    Ok(())
}

fn video_sampling(
    format: InputFormat,
    depth: GpuPixelFormat,
    mut color: YuvColor,
    rotation: i32,
) -> Result<VideoSampling, PipelineError> {
    color.samples = match depth {
        GpuPixelFormat::Nv12 => YuvSampleFormat::Unorm8,
        GpuPixelFormat::P010 => YuvSampleFormat::P010,
    };
    let orientation = match rotation.rem_euclid(360) {
        0 => 0,
        180 => 1,
        90 => 2,
        270 => 3,
        _ => return Err(invalid(format!("unsupported preview rotation {rotation}°"))),
    };
    Ok(VideoSampling {
        format,
        orientation,
        color,
    })
}

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "windows"
))]
fn check_count(count: usize, sizes: &[(u32, u32)]) -> Result<(), PipelineError> {
    if count != sizes.len() {
        return Err(invalid(format!(
            "expected {} preview cameras, got {count}",
            sizes.len()
        )));
    }
    Ok(())
}

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "windows"
))]
fn interop_error(error: impl std::fmt::Display) -> PipelineError {
    invalid(error.to_string())
}

fn invalid(reason: impl Into<String>) -> PipelineError {
    PipelineError::InvalidConfig {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
