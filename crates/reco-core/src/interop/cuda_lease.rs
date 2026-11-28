//! Lease ownership for Linux CUDA/Vulkan decode surfaces.
//!
//! A pool allocates two Y/UV images per camera. Acquiring a lease reserves one
//! image exclusively and keeps its CUDA mappings and Vulkan allocation alive,
//! even after the producer has stopped. A producer finishes all CUDA writes
//! before moving the lease into the camera-ordered frame set.
//!
//! Dropping a lease returns its image exactly once, without waiting for a
//! receiver. Partial sets, failed sends and queued frames follow the same rule.
//! Cancel a waiting acquire with the worker's stop flag, then join that worker.
//! Restarted workers can use the same pool with a fresh stop flag. They can
//! only acquire images whose leases have already dropped.
//!
//! Consumers must keep leases until all CUDA and Vulkan reads have completed,
//! including reads scheduled by detection, stitching, replay or lookahead.
//! Cloning a texture handle or copying a CUDA pointer does not retain the lease.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::cuda::CudaInteropError;
use super::vulkan::{Nv12Plane, SharedTexture, create_nv12_shared_texture};
use crate::detect::detector::CudaNv12Frame;
use crate::gpu::GpuContext;
use crate::render::{
    color::{YuvColor, YuvSampleFormat},
    renderer::GpuPixelFormat,
};

/// A double-buffered pool for one camera, shareable across decoder generations.
#[derive(Clone)]
pub struct CudaSlotPool(Arc<SlotPool<CudaImage>>);

impl CudaSlotPool {
    /// Allocate two shared NV12 or P010 images with the camera's source metadata.
    pub fn new(
        gpu: &GpuContext,
        size: (u32, u32),
        format: GpuPixelFormat,
        color: YuvColor,
        rotation: i32,
    ) -> Result<Self, CudaInteropError> {
        let color = YuvColor {
            samples: if format == GpuPixelFormat::P010 {
                YuvSampleFormat::P010
            } else {
                YuvSampleFormat::Unorm8
            },
            ..color
        };
        let allocate = || {
            let (width, height) = size;
            let y = create_nv12_shared_texture(gpu, width, height, Nv12Plane::Y, format)?;
            let uv = create_nv12_shared_texture(gpu, width, height, Nv12Plane::Uv, format)?;
            Ok(CudaImage {
                y_view: y.texture.create_view(&Default::default()),
                uv_view: uv.texture.create_view(&Default::default()),
                y,
                uv,
                format,
                device: gpu.device().clone(),
                color,
                rotation,
            })
        };
        Ok(Self(SlotPool::new([allocate()?, allocate()?])))
    }

    /// Wait for an image, checking the worker's cancellation flag even when
    /// all images remain leased. Cancellation does not change pool capacity.
    pub fn acquire(&self, stop: &AtomicBool) -> Option<CudaLease> {
        self.0.acquire(stop).map(CudaLease)
    }

    /// Reserve an image immediately, returning None when both images are leased.
    pub fn try_acquire(&self) -> Option<CudaLease> {
        self.0.try_acquire().map(CudaLease)
    }
}

/// Exclusive ownership of one camera's shared image until the lease drops.
///
/// Borrow the planes for rendering or derive a CUDA descriptor for detection.
/// All reads must complete before dropping the lease or returning its image to
/// a new producer. See the [module contract](self).
pub struct CudaLease(SlotLease<CudaImage>);

impl CudaLease {
    /// Borrow the Y and interleaved UV textures for GPU copies.
    pub fn textures(&self) -> (&wgpu::Texture, &wgpu::Texture) {
        let image = self.0.image();
        (&image.y.texture, &image.uv.texture)
    }

    /// Borrow the Y and interleaved UV views for sampling.
    pub fn views(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        let image = self.0.image();
        (&image.y_view, &image.uv_view)
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.0.image().device
    }

    /// Layout of the shared Y and UV planes.
    pub fn pixel_format(&self) -> GpuPixelFormat {
        self.0.image().format
    }

    /// Colour matrix and range associated with the source image.
    pub fn color(&self) -> YuvColor {
        self.0.image().color
    }

    /// Describe the mapped CUDA planes. Use the pointers only while the lease
    /// remains live, with exclusive access for staging writes.
    pub fn cuda_frame(&self) -> CudaNv12Frame {
        let image = self.0.image();
        CudaNv12Frame {
            y_ptr: image.y.cuda_ptr,
            uv_ptr: image.uv.cuda_ptr,
            y_pitch: image.y.pitch,
            uv_pitch: image.uv.pitch,
            width: image.y.texture.width(),
            height: image.y.texture.height(),
            rotation: image.rotation,
            is_10bit: image.format == GpuPixelFormat::P010,
            is_full_range: image.color.full_range,
        }
    }
}

struct CudaImage {
    y_view: wgpu::TextureView,
    uv_view: wgpu::TextureView,
    y: SharedTexture,
    uv: SharedTexture,
    format: GpuPixelFormat,
    device: wgpu::Device,
    color: YuvColor,
    rotation: i32,
}

struct SlotPool<T> {
    free: Mutex<Receiver<T>>,
    recycle: SyncSender<T>,
}

impl<T> SlotPool<T> {
    fn new(images: [T; 2]) -> Arc<Self> {
        let (recycle, free) = sync_channel(images.len());
        for image in images {
            assert!(
                recycle.try_send(image).is_ok(),
                "new pool has room for every image"
            );
        }
        Arc::new(Self {
            free: Mutex::new(free),
            recycle,
        })
    }

    fn acquire(self: &Arc<Self>, stop: &AtomicBool) -> Option<SlotLease<T>> {
        while !stop.load(Ordering::Acquire) {
            let image = self
                .free
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_millis(10));
            if let Ok(image) = image {
                let lease = self.lease(image);
                return (!stop.load(Ordering::Acquire)).then_some(lease);
            }
        }
        None
    }

    fn try_acquire(self: &Arc<Self>) -> Option<SlotLease<T>> {
        self.free
            .try_lock()
            .ok()?
            .try_recv()
            .ok()
            .map(|image| self.lease(image))
    }

    fn lease(self: &Arc<Self>, image: T) -> SlotLease<T> {
        SlotLease {
            image: Some(image),
            pool: self.clone(),
        }
    }
}

struct SlotLease<T> {
    image: Option<T>,
    pool: Arc<SlotPool<T>>,
}

impl<T> SlotLease<T> {
    fn image(&self) -> &T {
        self.image.as_ref().expect("live lease")
    }
}

impl<T> Drop for SlotLease<T> {
    fn drop(&mut self) {
        let image = self.image.take().expect("lease returns its image once");
        // The pool is still alive and has room for every outstanding lease.
        assert!(
            self.pool.recycle.try_send(image).is_ok(),
            "leased image has a vacant slot"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{TrySendError, channel};
    use std::thread;
    use std::time::Instant;

    fn assert_available(pool: &Arc<SlotPool<u8>>, expected: [u8; 2]) {
        let first = pool.try_acquire().unwrap();
        let second = pool.try_acquire().unwrap();
        let mut available = [*first.image(), *second.image()];
        available.sort_unstable();
        assert_eq!(available, expected);
        assert!(pool.try_acquire().is_none());
    }

    #[test]
    fn leases_return_once_on_drop_partial_set_and_failed_delivery() {
        let left = SlotPool::new([0, 1]);
        let right = SlotPool::new([2, 3]);

        let frame = vec![left.try_acquire().unwrap(), right.try_acquire().unwrap()];
        drop(frame);
        assert_available(&left, [0, 1]);
        assert_available(&right, [2, 3]);

        let partial = [Ok(left.try_acquire().unwrap()), Err(())];
        assert!(partial.into_iter().collect::<Result<Vec<_>, _>>().is_err());
        assert_available(&left, [0, 1]);

        let (tx, rx) = sync_channel(1);
        tx.send(left.try_acquire().unwrap()).ok().unwrap();
        drop(rx);
        assert!(matches!(
            tx.try_send(left.try_acquire().unwrap()),
            Err(TrySendError::Disconnected(_))
        ));
        assert_available(&left, [0, 1]);
    }

    #[test]
    fn holding_a_lease_prevents_reuse_until_drop() {
        let pool = SlotPool::new([0, 1]);
        let held = pool.try_acquire().unwrap();
        for _ in 0..4 {
            let next = pool.try_acquire().unwrap();
            assert_eq!(*next.image(), 1);
            assert!(pool.try_acquire().is_none());
        }
        drop(held);
        assert_available(&pool, [0, 1]);
    }

    #[test]
    fn cancelled_acquire_joins_while_leases_survive_into_the_next_generation() {
        let pool = SlotPool::new([0, 1]);
        let old_leases = [pool.try_acquire().unwrap(), pool.try_acquire().unwrap()];
        let stop = Arc::new(AtomicBool::new(false));
        let (done, completed) = channel();
        let worker = {
            let pool = pool.clone();
            let stop = stop.clone();
            thread::spawn(move || done.send(pool.acquire(&stop).is_none()).unwrap())
        };

        let deadline = Instant::now() + Duration::from_secs(2);
        while pool.free.try_lock().is_ok() {
            assert!(
                Instant::now() < deadline,
                "worker did not wait for an image"
            );
            thread::yield_now();
        }
        stop.store(true, Ordering::Release);
        assert!(completed.recv_timeout(Duration::from_secs(2)).unwrap());
        worker.join().unwrap();

        let [first, second] = old_leases;
        drop(second);
        let next = pool.acquire(&AtomicBool::new(false)).unwrap();
        assert_eq!(*next.image(), 1);
        assert!(pool.try_acquire().is_none());
        drop(first);
        drop(next);
        assert_available(&pool, [0, 1]);
    }

    #[test]
    #[ignore = "requires NVIDIA CUDA and Vulkan external-memory interop"]
    fn cuda_leases_describe_nv12_and_p010_planes() {
        use crate::render::color::YuvMatrix;

        let gpu = GpuContext::new_blocking().unwrap();
        assert!(gpu.supports_zero_copy());
        for (format, samples, full_range, rotation) in [
            (GpuPixelFormat::Nv12, YuvSampleFormat::Unorm8, false, 0),
            (GpuPixelFormat::P010, YuvSampleFormat::P010, true, 180),
        ] {
            let color = YuvColor {
                matrix: YuvMatrix::Bt2020Ncl,
                full_range,
                samples,
            };
            let pool = CudaSlotPool::new(&gpu, (128, 96), format, color, rotation).unwrap();
            let lease = pool.try_acquire().unwrap();
            drop(pool);

            let frame = lease.cuda_frame();
            assert_eq!(
                (frame.width, frame.height, frame.rotation),
                (128, 96, rotation)
            );
            assert_eq!(frame.is_full_range, full_range);
            assert_eq!(frame.is_10bit, format == GpuPixelFormat::P010);
            assert_eq!(lease.pixel_format(), format);
            assert_eq!(lease.color(), color);
            let (y, uv) = lease.views();
            assert_eq!(y.texture().format(), format.y_format());
            assert_eq!(uv.texture().format(), format.uv_format());
            assert_eq!((uv.texture().width(), uv.texture().height()), (64, 48));
        }
    }

    #[test]
    fn lease_keeps_image_and_pool_alive_after_producer_teardown() {
        let pool = SlotPool::new([Arc::new(7), Arc::new(8)]);
        let weak_pool = Arc::downgrade(&pool);
        let held = pool.try_acquire().unwrap();
        let weak_image = Arc::downgrade(held.image());

        drop(pool);
        assert!(weak_pool.upgrade().is_some());
        assert_eq!(**held.image(), 7);
        drop(held);
        assert!(weak_image.upgrade().is_none());
        assert!(weak_pool.upgrade().is_none());
    }
}
