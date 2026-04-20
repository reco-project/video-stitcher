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
