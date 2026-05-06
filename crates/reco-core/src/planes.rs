//! Frame plane types for pipeline input.
//!
//! Pure data types with no GPU dependency. These represent borrowed views
//! into decoded video frame data in various pixel formats (YUV420P, NV12,
//! RGBA/BGRA). Used at the boundary between decode backends and the
//! stitch pipeline.

/// Borrowed YUV420P plane references for pipeline input.
///
/// Tightly packed (no stride padding):
/// - `y`: `width * height` bytes
/// - `u`: `(width/2) * (height/2)` bytes
/// - `v`: `(width/2) * (height/2)` bytes
pub struct YuvPlanes<'a> {
    /// Y (luma) plane, full resolution.
    pub y: &'a [u8],
    /// U (Cb) plane, half resolution.
    pub u: &'a [u8],
    /// V (Cr) plane, half resolution.
    pub v: &'a [u8],
}

/// A single stride-aware plane view.
///
/// External frameworks (OBS, V4L2, GStreamer) typically expose planes as
/// a pointer plus a row stride that may exceed the plane width (padding).
/// [`StridedYuvPlanes::copy_into`] repacks padded data into tight
/// [`YuvPlanes`] without per-frame allocation.
pub struct FramePlaneView<'a> {
    /// Plane pixel bytes. Must have length at least `stride * height`.
    pub data: &'a [u8],
    /// Bytes per row, including any trailing padding.
    pub stride: u32,
    /// Plane width in pixels (bytes per row of usable data).
    pub width: u32,
    /// Plane height in rows.
    pub height: u32,
}

/// Stride-aware YUV420P planes for hardware-decoded or framework-callback frames.
///
/// Use [`Self::copy_into`] to produce a tightly packed [`YuvPlanes`]
/// view ready for the stitch pipeline.
pub struct StridedYuvPlanes<'a> {
    /// Y (luma) plane, full resolution.
    pub y: FramePlaneView<'a>,
    /// U (Cb) plane, half resolution per dimension.
    pub u: FramePlaneView<'a>,
    /// V (Cr) plane, half resolution per dimension.
    pub v: FramePlaneView<'a>,
}

impl StridedYuvPlanes<'_> {
    /// Repack the strided planes into a caller-owned tight buffer and
    /// return a [`YuvPlanes`] view into it.
    ///
    /// The `buffer` is resized to `y_len + u_len + v_len` bytes. Cache
    /// the buffer across frames to avoid per-frame allocation.
    pub fn copy_into<'b>(&self, buffer: &'b mut Vec<u8>) -> YuvPlanes<'b> {
        let y_len = (self.y.width as usize) * (self.y.height as usize);
        let u_len = (self.u.width as usize) * (self.u.height as usize);
        let v_len = (self.v.width as usize) * (self.v.height as usize);
        buffer.resize(y_len + u_len + v_len, 0);
        {
            let (y_dst, rest) = buffer.split_at_mut(y_len);
            let (u_dst, v_dst) = rest.split_at_mut(u_len);
            copy_plane_tight(&self.y, y_dst);
            copy_plane_tight(&self.u, u_dst);
            copy_plane_tight(&self.v, v_dst);
        }
        let (y, rest) = buffer.split_at(y_len);
        let (u, v) = rest.split_at(u_len);
        YuvPlanes { y, u, v }
    }
}

/// Copy `src` rows into `dst`, skipping stride padding.
///
/// Malformed planes (stride < width, short buffer) are zero-filled
/// with a warning so the pipeline continues with a black plane.
pub(crate) fn copy_plane_tight(src: &FramePlaneView<'_>, dst: &mut [u8]) {
    let width = src.width as usize;
    let height = src.height as usize;
    let stride = src.stride as usize;

    if dst.len() != width.saturating_mul(height) {
        log::warn!(
            "copy_plane_tight: dst {} bytes != width*height {} bytes; zero-filling",
            dst.len(),
            width.saturating_mul(height),
        );
        dst.fill(0);
        return;
    }
    if stride < width {
        log::warn!(
            "copy_plane_tight: stride {stride} < width {width}; zero-filling plane",
        );
        dst.fill(0);
        return;
    }
    if src.data.len() < stride.saturating_mul(height) {
        log::warn!(
            "copy_plane_tight: source buffer {} bytes < stride*height {} bytes; zero-filling plane",
            src.data.len(),
            stride.saturating_mul(height),
        );
        dst.fill(0);
        return;
    }

    if stride == width {
        dst.copy_from_slice(&src.data[..width * height]);
        return;
    }

    for row in 0..height {
        let src_start = row * stride;
        let dst_start = row * width;
        dst[dst_start..dst_start + width].copy_from_slice(&src.data[src_start..src_start + width]);
    }
}

/// Borrowed NV12 plane references for pipeline input.
///
/// Tightly packed (no stride padding):
/// - `y`: `width * height` bytes
/// - `uv`: `width * (height/2)` bytes (interleaved U,V)
pub struct Nv12Planes<'a> {
    /// Y (luma) plane, full resolution.
    pub y: &'a [u8],
    /// Interleaved UV (CbCr) plane, half resolution in each dimension.
    pub uv: &'a [u8],
}

/// Borrowed packed RGBA plane for pipeline input.
///
/// Tightly packed, 4 bytes per pixel in (R, G, B, A) byte order.
/// Callers with BGRA data use [`BgraPlanes::from_bgra_swizzle_into`]
/// to reorder before upload.
pub struct BgraPlanes<'a> {
    /// Packed RGBA bytes, length `width * height * 4`.
    pub rgba: &'a [u8],
}

impl<'a> BgraPlanes<'a> {
    /// Wrap an already-RGBA byte slice without copying.
    pub fn from_rgba(rgba: &'a [u8]) -> Self {
        Self { rgba }
    }

    /// Swizzle a BGRA-ordered source into a caller-owned `Vec<u8>` and
    /// return a [`BgraPlanes`] borrowing from it.
    ///
    /// Cache the buffer across frames to avoid per-frame allocation.
    pub fn from_bgra_swizzle_into(bgra: &[u8], buffer: &'a mut Vec<u8>) -> Self {
        buffer.resize(bgra.len(), 0);
        for (src, dst) in bgra.chunks_exact(4).zip(buffer.chunks_exact_mut(4)) {
            dst[0] = src[2]; // R <- B
            dst[1] = src[1]; // G
            dst[2] = src[0]; // B <- R
            dst[3] = src[3]; // A
        }
        Self { rgba: buffer }
    }
}
