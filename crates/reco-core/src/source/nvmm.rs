//! Owned NVMM input, independent of the native capture backend.

use std::ffi::c_void;
use std::sync::Arc;

/// NV12 allocation metadata for a native NVMM frame.
/// Descriptors alone do not retain the allocation or its mapped metadata.
#[derive(Debug, Clone, Copy)]
pub struct NvmmPlaneInfo {
    /// DMA-buf descriptor for Vulkan import.
    pub dmabuf_fd: i32,
    /// Luma width in pixels.
    pub width: u32,
    /// Luma height in pixels.
    pub height: u32,
    /// Y plane byte offset.
    pub y_offset: u32,
    /// UV plane byte offset.
    pub uv_offset: u32,
    /// Allocation size in bytes.
    pub total_size: u32,
    /// Mapped `NvBufSurface*` for detection preprocessing.
    pub surface_ptr: *mut c_void,
}

/// An NVMM image with ownership of its buffer and metadata mapping.
/// Cloning retains ownership without copying pixels. GPU consumers must keep
/// a clone until all submitted reads of the native image have completed.
#[derive(Clone)]
pub struct NvmmFrame {
    info: NvmmPlaneInfo,
    _owner: Arc<dyn Send + Sync>,
}

impl NvmmFrame {
    /// Retain a native NV12 image with its capture-backend owner.
    ///
    /// # Safety
    ///
    /// `info` must describe a valid NV12 DMA-buf and mapped NvBufSurface.
    /// `owner` must keep both valid and prevent producer writes/recycling until
    /// it is dropped. Metadata must be immutable and safe to read on another
    /// thread. Dropping the owner on any thread must be safe.
    pub unsafe fn new(info: NvmmPlaneInfo, owner: impl Send + Sync + 'static) -> Self {
        Self {
            info,
            _owner: Arc::new(owner),
        }
    }

    /// Borrow descriptors whose lifetime is bounded by this owned frame.
    pub fn info(&self) -> &NvmmPlaneInfo {
        &self.info
    }

    /// Lifetime-only fixture. Never import or dereference its descriptors.
    #[cfg(test)]
    pub(crate) fn lifetime_test_frame(owner: impl Send + Sync + 'static) -> Self {
        Self {
            info: NvmmPlaneInfo {
                dmabuf_fd: -1,
                width: 4,
                height: 2,
                y_offset: 0,
                uv_offset: 8,
                total_size: 12,
                surface_ptr: std::ptr::null_mut(),
            },
            _owner: Arc::new(owner),
        }
    }
}

// SAFETY: the constructor requires immutable, cross-thread metadata and an
// owner that prevents recycling. Raw pointers are never dereferenced here.
unsafe impl Send for NvmmFrame {}
unsafe impl Sync for NvmmFrame {}

impl std::fmt::Debug for NvmmFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvmmFrame")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clones_keep_owner_until_last_frame_drops() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let frame = NvmmFrame::lifetime_test_frame(owner);
        let other = frame.clone();
        drop(frame);
        assert!(weak.upgrade().is_some());
        std::thread::spawn(move || drop(other)).join().unwrap();
        assert!(weak.upgrade().is_none());
    }
}
