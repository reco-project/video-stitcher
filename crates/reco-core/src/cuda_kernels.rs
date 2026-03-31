//! CUDA kernels for GPU-side image preprocessing.
//!
//! Contains a normalize + HWC-to-CHW transpose kernel used to convert
//! NPP's packed RGB u8 output into the float32 CHW tensor format expected
//! by YOLO models. The kernel is embedded as PTX (compiled offline) to
//! avoid a build-time CUDA SDK dependency.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::cuda_interop::{CUdeviceptr, CudaInteropError, CudaKernel};

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
    crate::cuda_interop::cuda_ensure_context()?;
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
    crate::cuda_interop::cuda_synchronize()?;

    Ok(())
}
