//! CUDA kernels for GPU-side image preprocessing.
//!
//! P010 conversion, NV12 letterboxing and HWC-to-CHW normalization for detector input. Embedded PTX avoids a build-time CUDA SDK dependency.

use std::ffi::c_void;
use std::sync::OnceLock;

use reco_core::interop::cuda::{CUdeviceptr, CudaInteropError, CudaKernel};

/// Convert a strided P010 plane into tightly packed bytes by retaining each sample's high byte.
const P010_TO_NV12_PTX: &[u8] = b"
.version 7.0
.target sm_50
.address_size 64

.visible .entry p010_plane_to_nv12(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 pitch,
    .param .u32 width,
    .param .u32 height
)
.reqntid 128, 2, 1
{
    .reg .u32 %x, %y, %w, %h, %pitch, %tmp, %tmp2;
    .reg .u64 %src, %dst, %offset, %column;
    .reg .u16 %sample;
    .reg .pred %p;

    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %x, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %x, %x, %tmp;
    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %y, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %y, %y, %tmp;

    ld.param.u32 %w, [width];
    ld.param.u32 %h, [height];
    setp.ge.u32 %p, %x, %w;
    @%p bra done;
    setp.ge.u32 %p, %y, %h;
    @%p bra done;

    ld.param.u64 %src, [src];
    ld.param.u32 %pitch, [pitch];
    mul.wide.u32 %offset, %y, %pitch;
    mul.wide.u32 %column, %x, 2;
    add.u64 %offset, %offset, %column;
    add.u64 %src, %src, %offset;
    // PTX memory is little-endian: the second byte is the retained high byte.
    ld.global.u8 %sample, [%src+1];

    ld.param.u64 %dst, [dst];
    mul.wide.u32 %offset, %y, %w;
    cvt.u64.u32 %column, %x;
    add.u64 %offset, %offset, %column;
    add.u64 %dst, %dst, %offset;
    st.global.u8 [%dst], %sample;

done:
    ret;
}
\0";

static P010_KERNEL: OnceLock<Result<CudaKernel, CudaInteropError>> = OnceLock::new();

fn get_p010_kernel() -> Result<&'static CudaKernel, CudaInteropError> {
    P010_KERNEL
        .get_or_init(|| CudaKernel::from_ptx(P010_TO_NV12_PTX, "p010_plane_to_nv12"))
        .as_ref()
        .map_err(|e| CudaInteropError::CudaError {
            function: "p010_plane_to_nv12_load",
            code: match e {
                CudaInteropError::CudaError { code, .. } => *code,
                _ => -1,
            },
        })
}

/// Convert a CUDA P010 plane to tightly packed 8-bit samples, discarding the low byte of each 16-bit sample.
///
/// `width` counts individual samples, including both components of interleaved UV. `src_pitch` is in bytes and must be even and at least `width * 2`. Dimensions must be positive and within [`reco_core::calibration::MAX_DIM`].
///
/// `src` must reference a 16-bit-aligned device allocation covering all strided rows; `dst` must reference a separate device allocation of at least `width * height` bytes. Conversion finishes before returning, so the result can be used on another CUDA stream.
pub fn p010_plane_to_nv12(
    src: CUdeviceptr,
    src_pitch: usize,
    dst: CUdeviceptr,
    width: u32,
    height: u32,
) -> Result<(), CudaInteropError> {
    const CUDA_ERROR_INVALID_VALUE: i32 = 1;
    if width == 0
        || height == 0
        || width > reco_core::calibration::MAX_DIM
        || height > reco_core::calibration::MAX_DIM
        || src == 0
        || dst == 0
        || !src.is_multiple_of(2)
        || src_pitch > u32::MAX as usize
        || src_pitch < width as usize * 2
        || !src_pitch.is_multiple_of(2)
        || src
            .checked_add(src_pitch as u64 * (height - 1) as u64 + width as u64 * 2)
            .is_none()
        || dst.checked_add(width as u64 * height as u64).is_none()
    {
        return Err(CudaInteropError::CudaError {
            function: "p010_plane_to_nv12",
            code: CUDA_ERROR_INVALID_VALUE,
        });
    }
    reco_core::interop::cuda::cuda_ensure_context()?;
    let kernel = get_p010_kernel()?;
    let block = (128, 2, 1); // Matches the PTX entry's required block dimensions.
    let grid = (width.div_ceil(block.0), height.div_ceil(block.1), 1);
    let mut src_val = src;
    let mut dst_val = dst;
    let mut pitch_val = src_pitch as u32;
    let mut width_val = width;
    let mut height_val = height;
    let mut args: [*mut c_void; 5] = [
        (&mut src_val as *mut u64).cast(),
        (&mut dst_val as *mut u64).cast(),
        (&mut pitch_val as *mut u32).cast(),
        (&mut width_val as *mut u32).cast(),
        (&mut height_val as *mut u32).cast(),
    ];
    // SAFETY: Argument types/order match the PTX entry point. Bounds checks guard partial blocks.
    unsafe {
        kernel.launch(grid, block, 0, &mut args)?;
    }
    reco_core::interop::cuda::cuda_synchronize()?;
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
/// BT.709 coefficients with explicit source range (including `yuvj420p`
/// / `color_range=pc` output), normalizes to [0,1], and writes planar
/// CHW float32 directly. Replaces the NPP NV12-to-RGB (which uses
/// BT.601 video-range) + separate normalize kernel pipeline.
///
/// Each output pixel maps back to a nearest source sample; pixels outside
/// the rounded content extent are written grey on every invocation.
const NV12_TO_RGB_CHW_PTX: &[u8] = b"
.version 7.0
.target sm_50
.address_size 64

.visible .entry nv12_to_rgb_chw(
    .param .u64 y_ptr,
    .param .u64 uv_ptr,
    .param .u64 dst,
    .param .u32 y_pitch,
    .param .u32 uv_pitch,
    .param .u32 src_w,
    .param .u32 src_h,
    .param .u32 dst_w,
    .param .u32 dst_h,
    .param .u32 pad_x,
    .param .u32 pad_y,
    .param .u32 content_w,
    .param .u32 content_h,
    .param .f32 scale,
    .param .u32 flip180,
    .param .u32 full_range
)
{
    .reg .u32 %ox, %oy, %dw, %dh, %px, %py, %sw, %sh, %ypitch, %uvpitch;
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

    mov.f32 %r, 0f3EE4E4E5;  // 114.0/255.0
    mov.f32 %g, %r;
    mov.f32 %b, %r;

    // check if this pixel is in the content region
    setp.lt.u32 %p, %ox, %px;
    @%p bra write_pixel;
    setp.lt.u32 %p, %oy, %py;
    @%p bra write_pixel;
    ld.param.u32 %tmp, [content_w];
    add.u32 %tmp, %tmp, %px;
    setp.ge.u32 %p, %ox, %tmp;
    @%p bra write_pixel;
    ld.param.u32 %tmp, [content_h];
    add.u32 %tmp, %tmp, %py;
    setp.ge.u32 %p, %oy, %tmp;
    @%p bra write_pixel;

    // map to source coordinates using nearest sampling
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

    // 180-degree rotation: invert both source coordinates
    ld.param.u32 %tmp, [flip180];
    setp.ne.u32 %q, %tmp, 0;
    sub.u32 %tmp, %sw, 1;
    @%q sub.u32 %sx0, %tmp, %sx0;
    sub.u32 %tmp, %sh, 1;
    @%q sub.u32 %sy0, %tmp, %sy0;

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
    ld.param.u32 %uvpitch, [uv_pitch];
    shr.u32 %tmp, %sy0, 1;  // cy = sy / 2
    mul.lo.u32 %tmp, %tmp, %uvpitch;
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

    // Expand video-range luma/chroma before the shared BT.709 matrix.
    ld.param.u32 %tmp, [full_range];
    setp.ne.u32 %p, %tmp, 0;
    @%p bra color_ready;
    sub.f32 %y00, %y00, 0f41800000;
    mul.f32 %y00, %y00, 0f3F950A85;
    mul.f32 %u_val, %u_val, 0f3F91B6DB;
    mul.f32 %v_val, %v_val, 0f3F91B6DB;
color_ready:
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

write_pixel:
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
        .get_or_init(|| CudaKernel::from_ptx(NV12_TO_RGB_CHW_PTX, "nv12_to_rgb_chw"))
        .as_ref()
        .map_err(|e| CudaInteropError::CudaError {
            function: "nv12_to_rgb_chw_load",
            code: match e {
                CudaInteropError::CudaError { code, .. } => *code,
                _ => -1,
            },
        })
}

/// Combined NV12 -> float32 RGB CHW conversion with letterbox placement.
///
/// Uses the source range and BT.709 conversion in one fused kernel.
/// Full-range sources retain their scale; video-range sources expand
/// luma and chroma before conversion.
///
/// Reads NV12 Y + UV planes at `(y_ptr, uv_ptr)` with independent byte strides,
/// source dimensions `(src_w, src_h)`. Writes float32 CHW into `dst`
/// with dimensions `(dst_w, dst_h)` and content region offset
/// `(pad_x, pad_y)` at scale factor `scale`. The rounded content extent uses
/// nearest sampling. Every padding pixel is written grey (114/255).
#[allow(clippy::too_many_arguments)]
pub fn nv12_to_rgb_chw(
    y_ptr: CUdeviceptr,
    uv_ptr: CUdeviceptr,
    dst: CUdeviceptr,
    y_pitch: u32,
    uv_pitch: u32,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    pad_x: u32,
    pad_y: u32,
    scale: f32,
    rotation: i32,
    full_range: bool,
) -> Result<(), CudaInteropError> {
    reco_core::interop::cuda::cuda_ensure_context()?;
    let kernel = get_nv12_kernel()?;

    let block = (16u32, 16u32, 1u32);
    let grid = (dst_w.div_ceil(block.0), dst_h.div_ceil(block.1), 1u32);

    let mut y_val = y_ptr;
    let mut uv_val = uv_ptr;
    let mut dst_val = dst;
    let mut yp_val = y_pitch;
    let mut uvp_val = uv_pitch;
    let mut sw_val = src_w;
    let mut sh_val = src_h;
    let mut dw_val = dst_w;
    let mut dh_val = dst_h;
    let mut px_val = pad_x;
    let mut py_val = pad_y;
    let mut cw_val = (src_w as f32 * scale).round() as u32;
    let mut ch_val = (src_h as f32 * scale).round() as u32;
    let mut sc_val = scale;
    let mut flip_val: u32 = if rotation == 180 { 1 } else { 0 };

    let mut full_range_val = u32::from(full_range);
    let mut args: [*mut c_void; 16] = [
        (&mut y_val as *mut u64).cast(),
        (&mut uv_val as *mut u64).cast(),
        (&mut dst_val as *mut u64).cast(),
        (&mut yp_val as *mut u32).cast(),
        (&mut uvp_val as *mut u32).cast(),
        (&mut sw_val as *mut u32).cast(),
        (&mut sh_val as *mut u32).cast(),
        (&mut dw_val as *mut u32).cast(),
        (&mut dh_val as *mut u32).cast(),
        (&mut px_val as *mut u32).cast(),
        (&mut py_val as *mut u32).cast(),
        (&mut cw_val as *mut u32).cast(),
        (&mut ch_val as *mut u32).cast(),
        (&mut sc_val as *mut f32).cast(),
        (&mut flip_val as *mut u32).cast(),
        (&mut full_range_val as *mut u32).cast(),
    ];

    unsafe {
        kernel.launch(grid, block, 0, &mut args)?;
    }

    reco_core::interop::cuda::cuda_synchronize()?;
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
    reco_core::interop::cuda::cuda_ensure_context()?;
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

    reco_core::interop::cuda::cuda_synchronize()?;
    Ok(())
}

// ── RGBA normalize kernel ─────────────────────────────────────────

/// PTX for RGBA u8 (4-channel) -> float32 CHW (3-channel) normalize.
///
/// Identical to `normalize_hwc_to_chw` except the source pixel stride
/// is 4 bytes (RGBA) instead of 3 (RGB). Alpha at offset +3 is ignored.
/// Eliminates the need for a separate RGBA->RGB conversion step.
const NORMALIZE_RGBA_TO_CHW_PTX: &[u8] = b"
.version 7.0
.target sm_50
.address_size 64

.visible .entry normalize_rgba_to_chw(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 pitch,
    .param .u32 width,
    .param .u32 height
)
{
    .reg .u32 %x, %y, %w, %h, %hw, %pitch, %dst_idx;
    .reg .u64 %src_ptr, %dst_ptr, %addr, %src_idx, %column;
    .reg .u32 %tmp, %tmp2;
    .reg .u16 %pixel;
    .reg .f32 %val, %scale;
    .reg .pred %p;

    mov.u32 %tmp, %ctaid.x;
    mov.u32 %tmp2, %ntid.x;
    mul.lo.u32 %x, %tmp, %tmp2;
    mov.u32 %tmp, %tid.x;
    add.u32 %x, %x, %tmp;
    mov.u32 %tmp, %ctaid.y;
    mov.u32 %tmp2, %ntid.y;
    mul.lo.u32 %y, %tmp, %tmp2;
    mov.u32 %tmp, %tid.y;
    add.u32 %y, %y, %tmp;

    ld.param.u32 %w, [width];
    ld.param.u32 %h, [height];
    setp.ge.u32 %p, %x, %w;
    @%p bra done;
    setp.ge.u32 %p, %y, %h;
    @%p bra done;

    mul.lo.u32 %hw, %w, %h;

    ld.param.u32 %pitch, [pitch];
    mul.wide.u32 %src_idx, %y, %pitch;
    mul.wide.u32 %column, %x, 4;
    add.u64 %src_idx, %src_idx, %column;

    mul.lo.u32 %dst_idx, %y, %w;
    add.u32 %dst_idx, %dst_idx, %x;

    ld.param.u64 %src_ptr, [src];
    ld.param.u64 %dst_ptr, [dst];
    mov.f32 %scale, 0f3B808081;

    // R -> plane 0
    mov.u64 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    cvt.u64.u32 %addr, %dst_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    // G -> plane 1
    mov.u64 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 1;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
    add.u32 %tmp, %hw, %dst_idx;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %dst_ptr, %addr;
    st.global.f32 [%addr], %val;

    // B -> plane 2 (alpha at +3 ignored)
    mov.u64 %addr, %src_idx;
    add.u64 %addr, %src_ptr, %addr;
    add.u64 %addr, %addr, 2;
    ld.global.u8 %pixel, [%addr];
    cvt.rn.f32.u16 %val, %pixel;
    mul.f32 %val, %val, %scale;
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

static RGBA_KERNEL: OnceLock<Result<CudaKernel, CudaInteropError>> = OnceLock::new();

fn get_rgba_kernel() -> Result<&'static CudaKernel, CudaInteropError> {
    RGBA_KERNEL
        .get_or_init(|| CudaKernel::from_ptx(NORMALIZE_RGBA_TO_CHW_PTX, "normalize_rgba_to_chw"))
        .as_ref()
        .map_err(|e| CudaInteropError::CudaError {
            function: "normalize_rgba_to_chw_load",
            code: match e {
                CudaInteropError::CudaError { code, .. } => *code,
                _ => -1,
            },
        })
}

/// Convert pitched RGBA u8 (4-channel HWC) to float32 RGB (3-channel CHW).
///
/// Alpha channel is ignored. No intermediate RGB buffer needed -
/// single kernel replaces the NPP RGBA->RGB copy + normalize pipeline.
/// - `src`: RGBA u8, at least `src_pitch * (height - 1) + width * 4` bytes
/// - `src_pitch`: row stride in bytes, at least `width * 4` and at most `u32::MAX`
/// - `dst`: float32 CHW, `3 * width * height * 4` bytes
///
/// Dimensions must be positive and within [`reco_core::calibration::MAX_DIM`]. Source and destination must refer to separate valid device allocations. Conversion completes before returning.
pub fn normalize_rgba_to_chw(
    src: CUdeviceptr,
    src_pitch: usize,
    dst: CUdeviceptr,
    width: u32,
    height: u32,
) -> Result<(), CudaInteropError> {
    if width == 0
        || height == 0
        || width > reco_core::calibration::MAX_DIM
        || height > reco_core::calibration::MAX_DIM
        || src == 0
        || dst == 0
        || !dst.is_multiple_of(4)
        || src_pitch > u32::MAX as usize
        || src_pitch < width as usize * 4
        || src
            .checked_add(src_pitch as u64 * (height - 1) as u64 + width as u64 * 4)
            .is_none()
        || dst.checked_add(width as u64 * height as u64 * 12).is_none()
    {
        return Err(CudaInteropError::CudaError {
            function: "normalize_rgba_to_chw",
            code: 1,
        });
    }
    reco_core::interop::cuda::cuda_ensure_context()?;
    let kernel = get_rgba_kernel()?;

    let block = (16u32, 16u32, 1u32);
    let grid = (width.div_ceil(block.0), height.div_ceil(block.1), 1u32);

    let mut src_val = src;
    let mut dst_val = dst;
    let mut pitch_val = src_pitch as u32;
    let mut w_val = width;
    let mut h_val = height;

    let mut args: [*mut c_void; 5] = [
        (&mut src_val as *mut u64).cast(),
        (&mut dst_val as *mut u64).cast(),
        (&mut pitch_val as *mut u32).cast(),
        (&mut w_val as *mut u32).cast(),
        (&mut h_val as *mut u32).cast(),
    ];

    unsafe {
        kernel.launch(grid, block, 0, &mut args)?;
    }

    reco_core::interop::cuda::cuda_synchronize()?;
    Ok(())
}

#[cfg(test)]
mod tests;
