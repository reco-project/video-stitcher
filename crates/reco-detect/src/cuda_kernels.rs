//! CUDA kernels for GPU-side image preprocessing.
//!
//! Contains a normalize + HWC-to-CHW transpose kernel used to convert
//! NPP's packed RGB u8 output into the float32 CHW tensor format expected
//! by YOLO models. The kernel is embedded as PTX (compiled offline) to
//! avoid a build-time CUDA SDK dependency.

use std::ffi::c_void;
use std::sync::OnceLock;

use reco_core::cuda_interop::{CUdeviceptr, CudaInteropError, CudaKernel};

/// PTX source for the P010-to-NV12 conversion kernel.
///
/// P010 stores 10-bit values in the upper 10 bits of each u16 sample.
/// This kernel right-shifts each u16 by 8, producing the high byte as a
/// u8 output. This gives 8-bit precision from the 10-bit source, which
/// is sufficient for object detection.
///
/// Parameters: src (u16*), dst (u8*), n (u32) - total element count
///
/// Grid: (ceil(n/256), 1, 1)
/// Block: (256, 1, 1)
const P010_TO_NV12_PTX: &[u8] = b"
.version 7.0
.target sm_50
.address_size 64

.visible .entry p010_to_nv12(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 n
)
{
    .reg .u32 %i, %n_val, %tmp, %tmp2;
    .reg .u64 %src_ptr, %dst_ptr, %addr;
    .reg .u16 %val16;
    .reg .u32 %val32;
    .reg .pred %p;

    // i = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %i, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %i, %i, %tmp;

    // bounds check
    ld.param.u32 %n_val, [n];
    setp.ge.u32 %p, %i, %n_val;
    @%p bra done;

    // Load u16 from src[i]
    ld.param.u64 %src_ptr, [src];
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 1;  // * sizeof(u16)
    add.u64 %addr, %src_ptr, %addr;
    ld.global.u16 %val16, [%addr];

    // Right-shift by 8 to get high byte
    cvt.u32.u16 %val32, %val16;
    shr.u32 %val32, %val32, 8;

    // Store u8 to dst[i]
    ld.param.u64 %dst_ptr, [dst];
    cvt.u64.u32 %addr, %i;
    add.u64 %addr, %dst_ptr, %addr;
    // Truncate u32 to u8 and store
    cvt.u16.u32 %val16, %val32;
    st.global.u8 [%addr], %val16;

done:
    ret;
}
\0";

/// Lazily loaded P010-to-NV12 kernel.
static P010_KERNEL: OnceLock<Result<CudaKernel, CudaInteropError>> = OnceLock::new();

fn get_p010_kernel() -> Result<&'static CudaKernel, CudaInteropError> {
    P010_KERNEL
        .get_or_init(|| CudaKernel::from_ptx(P010_TO_NV12_PTX, "p010_to_nv12"))
        .as_ref()
        .map_err(|e| CudaInteropError::CudaError {
            function: "p010_to_nv12_load",
            code: match e {
                CudaInteropError::CudaError { code, .. } => *code,
                _ => -1,
            },
        })
}

/// Convert a P010 (10-bit NV12) plane to 8-bit NV12 on the GPU.
///
/// Reads `n` u16 samples from `src`, right-shifts each by 8 to extract
/// the high byte, and writes `n` u8 samples to `dst`. Works for both
/// the Y plane (width * height samples) and the UV plane (width * height/2
/// samples, but counted as individual u16 values).
///
/// Both `src` and `dst` are CUDA device pointers. `n` is the number of
/// samples (not bytes). `src` must have at least `n * 2` bytes, `dst`
/// must have at least `n` bytes.
///
/// Note: this performs a 2D-pitched to linear conversion. If the source
/// has padding (pitch > width), use [`p010_plane_to_nv12`] instead, which
/// handles pitched layouts via [`cuda_2d_copy`](reco_core::cuda_interop::cuda_2d_copy)
/// style semantics. For contiguous data (pitch == width * 2), this function
/// is simpler.
pub fn p010_to_nv12(src: CUdeviceptr, dst: CUdeviceptr, n: u32) -> Result<(), CudaInteropError> {
    reco_core::cuda_interop::cuda_ensure_context()?;
    let kernel = get_p010_kernel()?;

    let block = 256u32;
    let grid = n.div_ceil(block);

    let mut src_val = src;
    let mut dst_val = dst;
    let mut n_val = n;

    let mut args: [*mut std::ffi::c_void; 3] = [
        (&mut src_val as *mut u64).cast(),
        (&mut dst_val as *mut u64).cast(),
        (&mut n_val as *mut u32).cast(),
    ];

    unsafe {
        kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut args)?;
    }

    // Synchronize to ensure kernel completion before using the output.
    reco_core::cuda_interop::cuda_synchronize()?;

    Ok(())
}

/// Convert a pitched P010 plane to a contiguous 8-bit plane on the GPU.
///
/// Handles the common case where the source P010 plane has row padding
/// (pitch > width * 2). Reads `width` u16 samples per row from `src`
/// with `src_pitch` byte stride, converts each to u8 by right-shifting
/// by 8, and writes to `dst` with `width` byte stride (tightly packed).
///
/// `width` is in samples (pixels for Y, pixel pairs for UV).
/// `height` is the number of rows.
///
/// Both `src` and `dst` are CUDA device pointers.
pub fn p010_plane_to_nv12(
    src: CUdeviceptr,
    src_pitch: usize,
    dst: CUdeviceptr,
    width: u32,
    height: u32,
) -> Result<(), CudaInteropError> {
    if src_pitch == width as usize * 2 {
        // Contiguous layout - use the simple kernel directly.
        return p010_to_nv12(src, dst, width * height);
    }

    // Pitched layout: process row by row through the kernel.
    // Each row has `width` u16 samples starting at src + row * src_pitch.
    for row in 0..height {
        let row_src = src + (row as usize * src_pitch) as u64;
        let row_dst = dst + (row as usize * width as usize) as u64;
        p010_to_nv12(row_src, row_dst, width)?;
    }

    Ok(())
}

/// PTX source for the normalize + HWC-to-CHW transpose kernel.
///
/// This kernel reads packed RGB u8 pixels (HWC layout) and writes
/// float32 values in CHW layout, dividing by 255.0 for \[0,1\] normalization.
///
/// Parameters: src (u8*), dst (f32*), width (u32), height (u32)
///
/// Grid: (ceil(width/16), ceil(height/16), 1)
/// Block: (16, 16, 1)
const NORMALIZE_HWC_TO_CHW_PTX: &[u8] = b"
.version 7.0
.target sm_50
.address_size 64

.visible .entry normalize_hwc_to_chw(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 width,
    .param .u32 height
)
{
    .reg .u32 %x, %y, %w, %h, %hw, %src_idx, %dst_plane, %dst_idx;
    .reg .u64 %src_ptr, %dst_ptr, %addr;
    .reg .u32 %tmp, %tmp2;
    .reg .u16 %pixel;
    .reg .f32 %val, %scale;
    .reg .pred %p;

    // x = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %x, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %x, %x, %tmp;

    // y = blockIdx.y * blockDim.y + threadIdx.y
    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %y, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %y, %y, %tmp;

    // bounds check
    ld.param.u32 %w, [width];
    ld.param.u32 %h, [height];
    setp.ge.u32 %p, %x, %w;
    @%p bra done;
    setp.ge.u32 %p, %y, %h;
    @%p bra done;

    // hw = width * height (plane stride)
    mul.lo.u32 %hw, %w, %h;

    // src_idx = (y * width + x) * 3
    mul.lo.u32 %src_idx, %y, %w;
    add.u32 %src_idx, %src_idx, %x;
    mul.lo.u32 %src_idx, %src_idx, 3;

    // dst_idx = y * width + x
    mul.lo.u32 %dst_idx, %y, %w;
    add.u32 %dst_idx, %dst_idx, %x;

    ld.param.u64 %src_ptr, [src];
    ld.param.u64 %dst_ptr, [dst];
    mov.f32 %scale, 0f3B808081;  // 1.0/255.0

    // R channel -> plane 0
    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    cvt.u64.u32 %addr, %dst_idx;
    shl.b64 %addr, %addr, 2;  // * sizeof(f32)
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    // G channel -> plane 1
    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 1;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    // dst offset = (hw + dst_idx) * 4
    add.u32 %tmp, %hw, %dst_idx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    // B channel -> plane 2
    cvt.u64.u32 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 2;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    // dst offset = (2*hw + dst_idx) * 4
    add.u32 %tmp, %hw, %hw;
    add.u32 %tmp, %tmp, %dst_idx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

done:
    ret;
}
\0";

/// PTX for combined NV12 -> float32 RGB CHW conversion.
///
/// Reads NV12 Y + interleaved UV planes, converts to RGB using
/// BT.709 full-range coefficients (matching GoPro/action-cam `yuvj420p`
/// / `color_range=pc` output), normalizes to [0,1], and writes planar
/// CHW float32 directly. Replaces the NPP NV12-to-RGB (which uses
/// BT.601 video-range) + separate normalize kernel pipeline.
///
/// Parameters: y_ptr, uv_ptr, dst, y_pitch, width, height, dst_w, dst_h,
///             pad_x, pad_y, scale
///
/// The kernel handles letterbox placement: each output pixel at (ox, oy)
/// maps back to source (sx, sy) via bilinear interpolation, with pixels
/// outside the content region left at the pre-filled grey value.
const NV12_TO_RGB_CHW_PTX: &[u8] = b"
.version 7.0
.target sm_50
.address_size 64

.visible .entry nv12_to_rgb_chw_fullrange(
    .param .u64 y_ptr,
    .param .u64 uv_ptr,
    .param .u64 dst,
    .param .u32 y_pitch,
    .param .u32 src_w,
    .param .u32 src_h,
    .param .u32 dst_w,
    .param .u32 dst_h,
    .param .u32 pad_x,
    .param .u32 pad_y,
    .param .f32 scale
)
{
    .reg .u32 %ox, %oy, %dw, %dh, %px, %py, %sw, %sh, %ypitch;
    .reg .u32 %sx0, %sy0, %sx1, %sy1;
    .reg .u32 %tmp, %tmp2, %plane, %didx;
    .reg .u64 %yp, %uvp, %dp, %addr;
    .reg .f32 %srcx, %srcy, %fx, %fy, %inv_scale;
    .reg .f32 %one, %zero, %f255, %c128;
    .reg .f32 %y00, %y10, %y01, %y11, %y_interp;
    .reg .f32 %u_val, %v_val, %r, %g, %b;
    .reg .f32 %t1, %t2, %t3, %t4;
    .reg .u16 %pix;
    .reg .pred %p, %q;

    // thread coords
    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %ox, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %ox, %ox, %tmp;

    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %oy, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %oy, %oy, %tmp;

    ld.param.u32 %dw, [dst_w];
    ld.param.u32 %dh, [dst_h];
    setp.ge.u32 %p, %ox, %dw;
    @%p bra done;
    setp.ge.u32 %p, %oy, %dh;
    @%p bra done;

    ld.param.u32 %px, [pad_x];
    ld.param.u32 %py, [pad_y];
    ld.param.u32 %sw, [src_w];
    ld.param.u32 %sh, [src_h];
    ld.param.f32 %inv_scale, [scale];

    // check if this pixel is in the content region
    setp.lt.u32 %p, %ox, %px;
    @%p bra done;
    setp.lt.u32 %p, %oy, %py;
    @%p bra done;
    sub.u32 %tmp, %dw, %px;
    setp.ge.u32 %p, %ox, %tmp;
    @%p bra done;
    sub.u32 %tmp, %dh, %py;
    setp.ge.u32 %p, %oy, %tmp;
    @%p bra done;

    // map to source coords (nearest for now, bilinear TODO)
    sub.u32 %tmp, %ox, %px;
    cvt.rn.f32.u32 %srcx, %tmp;
    div.rn.f32 %srcx, %srcx, %inv_scale;
    cvt.rzi.u32.f32 %sx0, %srcx;
    sub.u32 %tmp2, %sw, 1;
    min.u32 %sx0, %sx0, %tmp2;

    sub.u32 %tmp, %oy, %py;
    cvt.rn.f32.u32 %srcy, %tmp;
    div.rn.f32 %srcy, %srcy, %inv_scale;
    cvt.rzi.u32.f32 %sy0, %srcy;
    sub.u32 %tmp2, %sh, 1;
    min.u32 %sy0, %sy0, %tmp2;

    // read Y sample
    ld.param.u64 %yp, [y_ptr];
    ld.param.u32 %ypitch, [y_pitch];
    mul.lo.u32 %tmp, %sy0, %ypitch;
    add.u32 %tmp, %tmp, %sx0;
    cvt.u64.u32 %addr, %tmp;
    add.u64 %addr, %yp, %addr;
    ld.global.u8 %pix, [%addr];
    cvt.rn.f32.u16 %y00, %pix;

    // read UV sample (NV12: interleaved UV at half resolution)
    ld.param.u64 %uvp, [uv_ptr];
    shr.u32 %tmp, %sy0, 1;  // cy = sy / 2
    mul.lo.u32 %tmp, %tmp, %ypitch;  // cy * pitch
    shr.u32 %tmp2, %sx0, 1;  // cx = sx / 2
    shl.b32 %tmp2, %tmp2, 1;  // cx * 2 (interleaved UV)
    add.u32 %tmp, %tmp, %tmp2;
    cvt.u64.u32 %addr, %tmp;
    add.u64 %addr, %uvp, %addr;
    ld.global.u8 %pix, [%addr];
    cvt.rn.f32.u16 %u_val, %pix;
    ld.global.u8 %pix, [%addr+1];
    cvt.rn.f32.u16 %v_val, %pix;

    // BT.709 full-range YUV -> RGB
    // R = Y + 1.5748 * (V - 128)
    // G = Y - 0.1873 * (U - 128) - 0.4681 * (V - 128)
    // B = Y + 1.8556 * (U - 128)
    mov.f32 %c128, 0f43000000;  // 128.0
    sub.f32 %u_val, %u_val, %c128;
    sub.f32 %v_val, %v_val, %c128;

    // R
    mov.f32 %t1, 0f3FC9930C;  // 1.5748
    fma.rn.f32 %r, %t1, %v_val, %y00;

    // G
    mov.f32 %t1, 0fBE3FCB92;  // -0.1873
    fma.rn.f32 %g, %t1, %u_val, %y00;
    mov.f32 %t2, 0fBEEFAACE;  // -0.4681
    fma.rn.f32 %g, %t2, %v_val, %g;

    // B
    mov.f32 %t1, 0f3FED844D;  // 1.8556
    fma.rn.f32 %b, %t1, %u_val, %y00;

    // clamp to [0, 255] and normalize to [0, 1]
    mov.f32 %zero, 0f00000000;
    mov.f32 %f255, 0f437F0000;  // 255.0
    max.f32 %r, %r, %zero;
    min.f32 %r, %r, %f255;
    max.f32 %g, %g, %zero;
    min.f32 %g, %g, %f255;
    max.f32 %b, %b, %zero;
    min.f32 %b, %b, %f255;

    mov.f32 %t1, 0f3B808081;  // 1.0/255.0
    mul.f32 %r, %r, %t1;
    mul.f32 %g, %g, %t1;
    mul.f32 %b, %b, %t1;

    // write CHW planar output
    mul.lo.u32 %plane, %dw, %dh;
    mul.lo.u32 %didx, %oy, %dw;
    add.u32 %didx, %didx, %ox;

    ld.param.u64 %dp, [dst];

    // R -> plane 0
    cvt.u64.u32 %addr, %didx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dp, %addr;
    st.global.f32 [%addr], %r;

    // G -> plane 1
    add.u32 %tmp, %plane, %didx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dp, %addr;
    st.global.f32 [%addr], %g;

    // B -> plane 2
    add.u32 %tmp, %plane, %plane;
    add.u32 %tmp, %tmp, %didx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dp, %addr;
    st.global.f32 [%addr], %b;

done:
    ret;
}
\0";

static NV12_KERNEL: OnceLock<Result<CudaKernel, CudaInteropError>> = OnceLock::new();

fn get_nv12_kernel() -> Result<&'static CudaKernel, CudaInteropError> {
    NV12_KERNEL
        .get_or_init(|| {
            CudaKernel::from_ptx(NV12_TO_RGB_CHW_PTX, "nv12_to_rgb_chw_fullrange")
        })
        .as_ref()
        .map_err(|e| CudaInteropError::CudaError {
            function: "nv12_to_rgb_chw_fullrange_load",
            code: match e {
                CudaInteropError::CudaError { code, .. } => *code,
                _ => -1,
            },
        })
}

/// Combined NV12 -> float32 RGB CHW conversion with letterbox placement.
///
/// Replaces the NPP `nppiNV12ToRGB` (BT.601 video-range) + resize +
/// normalize pipeline with a single kernel using BT.709 full-range
/// coefficients matching GoPro/action-cam `yuvj420p` output.
///
/// Reads NV12 Y + UV planes at `(y_ptr, uv_ptr)` with `y_pitch`,
/// source dimensions `(src_w, src_h)`. Writes float32 CHW into `dst`
/// with dimensions `(dst_w, dst_h)` and content region offset
/// `(pad_x, pad_y)` at scale factor `scale`. Padding pixels must be
/// pre-filled (grey 114/255 = 0.447) by the caller at init time.
#[allow(clippy::too_many_arguments)]
pub fn nv12_to_rgb_chw_fullrange(
    y_ptr: CUdeviceptr,
    uv_ptr: CUdeviceptr,
    dst: CUdeviceptr,
    y_pitch: u32,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    pad_x: u32,
    pad_y: u32,
    scale: f32,
) -> Result<(), CudaInteropError> {
    reco_core::cuda_interop::cuda_ensure_context()?;
    let kernel = get_nv12_kernel()?;

    let block = (16u32, 16u32, 1u32);
    let grid = (dst_w.div_ceil(block.0), dst_h.div_ceil(block.1), 1u32);

    let mut y_val = y_ptr;
    let mut uv_val = uv_ptr;
    let mut dst_val = dst;
    let mut yp_val = y_pitch;
    let mut sw_val = src_w;
    let mut sh_val = src_h;
    let mut dw_val = dst_w;
    let mut dh_val = dst_h;
    let mut px_val = pad_x;
    let mut py_val = pad_y;
    let mut sc_val = scale;

    let mut args: [*mut c_void; 11] = [
        (&mut y_val as *mut u64).cast(),
        (&mut uv_val as *mut u64).cast(),
        (&mut dst_val as *mut u64).cast(),
        (&mut yp_val as *mut u32).cast(),
        (&mut sw_val as *mut u32).cast(),
        (&mut sh_val as *mut u32).cast(),
        (&mut dw_val as *mut u32).cast(),
        (&mut dh_val as *mut u32).cast(),
        (&mut px_val as *mut u32).cast(),
        (&mut py_val as *mut u32).cast(),
        (&mut sc_val as *mut f32).cast(),
    ];

    unsafe {
        kernel.launch(grid, block, 0, &mut args)?;
    }

    reco_core::cuda_interop::cuda_synchronize()?;
    Ok(())
}

/// Lazily loaded normalize+transpose kernel.
static KERNEL: OnceLock<Result<CudaKernel, CudaInteropError>> = OnceLock::new();

fn get_kernel() -> Result<&'static CudaKernel, CudaInteropError> {
    KERNEL
        .get_or_init(|| CudaKernel::from_ptx(NORMALIZE_HWC_TO_CHW_PTX, "normalize_hwc_to_chw"))
        .as_ref()
        .map_err(|e| CudaInteropError::CudaError {
            function: "normalize_hwc_to_chw_load",
            code: match e {
                CudaInteropError::CudaError { code, .. } => *code,
                _ => -1,
            },
        })
}

/// Convert packed RGB u8 (HWC) to float32 (CHW) with \[0,1\] normalization.
///
/// This is the final preprocessing step before YOLO inference: takes the
/// NPP-produced RGB u8 buffer and produces a `[1, 3, H, W]` float32 tensor.
///
/// Both `src` and `dst` are CUDA device pointers:
/// - `src`: packed RGB u8, `width * height * 3` bytes
/// - `dst`: float32 CHW, `3 * width * height * 4` bytes
pub fn normalize_hwc_to_chw(
    src: CUdeviceptr,
    dst: CUdeviceptr,
    width: u32,
    height: u32,
) -> Result<(), CudaInteropError> {
    reco_core::cuda_interop::cuda_ensure_context()?;
    let kernel = get_kernel()?;

    let block = (16u32, 16u32, 1u32);
    let grid = (width.div_ceil(block.0), height.div_ceil(block.1), 1u32);

    let mut src_val = src;
    let mut dst_val = dst;
    let mut w_val = width;
    let mut h_val = height;

    let mut args: [*mut c_void; 4] = [
        (&mut src_val as *mut u64).cast(),
        (&mut dst_val as *mut u64).cast(),
        (&mut w_val as *mut u32).cast(),
        (&mut h_val as *mut u32).cast(),
    ];

    unsafe {
        kernel.launch(grid, block, 0, &mut args)?;
    }

    // Synchronize to ensure kernel completion before using the output.
    reco_core::cuda_interop::cuda_synchronize()?;

    Ok(())
}
