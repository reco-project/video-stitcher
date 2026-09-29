//! NvBufSurface metadata extraction for NVMM camera buffers on Jetson.
//!
//! On Jetson, `nvarguscamerasrc` outputs NV12 frames in NVMM memory.
//! `gst_buffer_map` returns a pointer to an `NvBufSurface` metadata struct
//! (not pixel data). The `bufferDesc` field holds the DMA-buf file descriptor
//! for zero-copy Vulkan import, and the surface pointer is passed to
//! NvBufSurfTransform for detection preprocessing.
//!
//! Struct layouts verified against nvbufsurface.h on JetPack 6 (L4T 36.x)
//! via offsetof() on Jetson Orin Nano aarch64.

use std::ffi::c_void;

const NVBUF_MAX_PLANES: usize = 4;
const STRUCTURE_PADDING: usize = 4;

/// NVBUF_MEM_SURFACE_ARRAY = 4 (hardware-allocated NVMM buffers from ISP/NVDEC).
const NVBUF_MEM_SURFACE_ARRAY: u32 = 4;

/// Per-plane layout information (sizeof verified on aarch64).
#[repr(C)]
pub(crate) struct NvBufSurfacePlaneParams {
    pub num_planes: u32,
    pub width: [u32; NVBUF_MAX_PLANES],
    pub height: [u32; NVBUF_MAX_PLANES],
    pub pitch: [u32; NVBUF_MAX_PLANES],
    pub offset: [u32; NVBUF_MAX_PLANES],
    pub psize: [u32; NVBUF_MAX_PLANES],
    pub bytes_per_pix: [u32; NVBUF_MAX_PLANES],
    _reserved: [*mut c_void; STRUCTURE_PADDING * NVBUF_MAX_PLANES],
}

/// Mapped address pointers (used only for CPU mapping, not for zero-copy).
#[repr(C)]
pub(crate) struct NvBufSurfaceMappedAddr {
    pub addr: [*mut c_void; NVBUF_MAX_PLANES],
    pub egl_image: *mut c_void,
    _reserved: [*mut c_void; STRUCTURE_PADDING],
}

/// Per-buffer parameters (sizeof=384, verified on aarch64).
#[repr(C)]
pub(crate) struct NvBufSurfaceParams {
    pub width: u32,                            // offset 0
    pub height: u32,                           // offset 4
    pub pitch: u32,                            // offset 8
    pub color_format: u32,                     // offset 12
    pub layout: u32,                           // offset 16
    _pad0: u32,                                // offset 20
    pub buffer_desc: i64,                      // offset 24 (DMA-buf fd as signed)
    pub data_size: u32,                        // offset 32
    _pad1: u32,                                // offset 36
    pub data_ptr: *mut c_void,                 // offset 40
    pub plane_params: NvBufSurfacePlaneParams, // offset 48
    pub mapped_addr: NvBufSurfaceMappedAddr,
    pub paramex: *mut c_void,
    _reserved: [*mut c_void; STRUCTURE_PADDING - 1],
}

/// Top-level batched buffer container (sizeof=64, verified on aarch64).
#[repr(C)]
pub(crate) struct NvBufSurface {
    pub gpu_id: u32,                           // offset 0
    pub batch_size: u32,                       // offset 4
    pub num_filled: u32,                       // offset 8
    pub is_contiguous: u8,                     // offset 12
    _pad0: [u8; 3],                            // offset 13
    pub mem_type: u32,                         // offset 16
    _pad1: u32,                                // offset 20
    pub surface_list: *mut NvBufSurfaceParams, // offset 24
    pub is_imported_buf: u8,                   // offset 32
    _pad2: [u8; 7],
    _reserved: [*mut c_void; STRUCTURE_PADDING - 1],
}

/// Extract an owned mapping from an NVMM-producing pipeline.
pub(super) fn extract_sample(
    sample: &gstreamer::Sample,
) -> Result<reco_core::source::NvmmFrame, reco_core::source::SourceError> {
    use reco_core::source::{NvmmFrame, SourceError};
    let read_error = |reason: &str| SourceError::Read {
        reason: reason.into(),
    };
    let caps = sample
        .caps()
        .ok_or_else(|| read_error("NVMM sample has no caps"))?;
    if !caps
        .features(0)
        .is_some_and(|features| features.contains("memory:NVMM"))
        || !caps
            .structure(0)
            .is_some_and(|s| s.name() == "video/x-raw" && s.get::<&str>("format") == Ok("NV12"))
    {
        return Err(read_error("expected NVMM NV12 sample"));
    }
    let buffer = sample
        .buffer()
        .ok_or_else(|| read_error("NVMM sample has no buffer"))?;
    let map = buffer
        .to_owned()
        .into_mapped_buffer_readable()
        .map_err(|_| read_error("NVMM buffer map failed"))?;
    // SAFETY: this extractor is private to NVMM ISP/decoder pipelines. The
    // mapped payload uses JetPack's NvBufSurface ABI, not raw image bytes.
    let info =
        unsafe { extract_info(map.as_slice()) }.map_err(|reason| SourceError::Read { reason })?;
    // SAFETY: the owned mapping pins the GstBuffer and metadata together.
    // GStreamer will not recycle the pool buffer until its final owner drops.
    Ok(unsafe { NvmmFrame::new(info, map) })
}

/// The slice must come from an NVMM buffer with valid native pointer fields.
unsafe fn extract_info(mapped: &[u8]) -> Result<reco_core::source::NvmmPlaneInfo, String> {
    let ptr = mapped.as_ptr();
    if mapped.len() < std::mem::size_of::<NvBufSurface>()
        || !ptr.cast::<NvBufSurface>().is_aligned()
    {
        return Err("invalid NvBufSurface mapping size/alignment".into());
    }
    let surface = unsafe { &*ptr.cast::<NvBufSurface>() };
    if surface.mem_type != NVBUF_MEM_SURFACE_ARRAY {
        return Err(format!(
            "expected NVBUF_MEM_SURFACE_ARRAY (4), got {}",
            surface.mem_type
        ));
    }
    if surface.num_filled != 1
        || surface.batch_size < surface.num_filled
        || surface.surface_list.is_null()
        || !surface.surface_list.is_aligned()
    {
        return Err("expected one valid NvBufSurface buffer".into());
    }
    let params = unsafe { &*surface.surface_list };
    if params.plane_params.num_planes != 2
        || params.width == 0
        || params.height == 0
        || !params.width.is_multiple_of(2)
        || !params.height.is_multiple_of(2)
    {
        return Err("expected nonempty, even-sized NV12 planes".into());
    }
    let dmabuf_fd = i32::try_from(params.buffer_desc)
        .ok()
        .filter(|fd| *fd >= 0)
        .ok_or("invalid NVMM DMA-buf descriptor")?;
    let end = params.plane_params.pitch[1]
        .checked_mul(params.plane_params.height[1])
        .and_then(|size| params.plane_params.offset[1].checked_add(size))
        .filter(|size| *size > params.plane_params.offset[1])
        .ok_or("invalid NVMM plane extent")?;
    // Some capture backends leave dataSize unset; keep the plane extent fallback.
    let total_size = if params.data_size == 0 {
        end
    } else {
        if params.data_size < end {
            return Err("NVMM allocation smaller than plane extent".into());
        }
        params.data_size
    };
    Ok(reco_core::source::NvmmPlaneInfo {
        dmabuf_fd,
        width: params.width,
        height: params.height,
        y_offset: params.plane_params.offset[0],
        uv_offset: params.plane_params.offset[1],
        total_size,
        surface_ptr: ptr as *mut c_void,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gstreamer as gst;
    use std::sync::Arc;

    // Aligned initialized metadata plus the storage addressed by surface_list.
    // Descriptors are never submitted to a native/GPU API in these tests.
    struct Payload {
        header: [u64; 8],
        _params: Box<NvBufSurfaceParams>,
        _lifetime: Arc<()>,
    }
    // The pointer in header addresses our immovable Box. The fixture is frozen
    // once passed to GstBuffer; releasing it from a capture worker is safe.
    unsafe impl Send for Payload {}
    impl AsRef<[u8]> for Payload {
        fn as_ref(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.header.as_ptr().cast(), 64) }
        }
    }
    fn sample(lifetime: Arc<()>) -> gst::Sample {
        gst::init().unwrap();
        assert_eq!(std::mem::size_of::<NvBufSurface>(), 64);
        assert_eq!(std::mem::size_of::<NvBufSurfaceParams>(), 384);
        let mut params: Box<NvBufSurfaceParams> = Box::new(unsafe { std::mem::zeroed() });
        params.width = 64;
        params.height = 32;
        params.buffer_desc = 7;
        params.data_size = 3072;
        params.plane_params.num_planes = 2;
        params.plane_params.offset[1] = 2048;
        params.plane_params.pitch[1] = 64;
        params.plane_params.height[1] = 16;
        let mut payload = Payload {
            header: [0; 8],
            _params: params,
            _lifetime: lifetime,
        };
        let header = unsafe { &mut *payload.header.as_mut_ptr().cast::<NvBufSurface>() };
        header.batch_size = 1;
        header.num_filled = 1;
        header.mem_type = NVBUF_MEM_SURFACE_ARRAY;
        header.surface_list = &mut *payload._params;
        let buffer = gst::Buffer::from_slice(payload);
        let caps: gst::Caps = "video/x-raw(memory:NVMM),format=NV12,width=64,height=32"
            .parse()
            .unwrap();
        gst::Sample::builder().buffer(&buffer).caps(&caps).build()
    }

    #[test]
    fn nvmm_mapping_outlives_sample_and_all_frame_clones() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let sample = sample(owner);
        let frame = extract_sample(&sample).unwrap();
        let clone = frame.clone();
        drop(sample);
        drop(frame);
        assert!(weak.upgrade().is_some());
        assert_eq!(clone.info().width, 64);
        let metadata = unsafe { &*clone.info().surface_ptr.cast::<NvBufSurface>() };
        assert_eq!(metadata.num_filled, 1);
        assert_eq!(unsafe { &*metadata.surface_list }.width, 64);
        std::thread::spawn(move || drop(clone)).join().unwrap();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn nvmm_rejects_wrong_caps_short_mapping_and_invalid_header() {
        gst::init().unwrap();
        let buffer = gst::Buffer::from_slice(vec![0u8; 8]);
        for caps in [
            "video/x-raw,format=NV12",
            "video/x-raw(memory:NVMM),format=I420",
            "video/x-raw(memory:NVMM),format=NV12",
        ] {
            let caps: gst::Caps = caps.parse().unwrap();
            let sample = gst::Sample::builder().buffer(&buffer).caps(&caps).build();
            assert!(extract_sample(&sample).is_err());
        }
        let data = [0u64; 8];
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), 64) };
        assert!(unsafe { extract_info(bytes) }.is_err());
    }
}
