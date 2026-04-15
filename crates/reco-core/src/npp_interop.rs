//! NPP (NVIDIA Performance Primitives) interop for GPU image processing.
//!
//! Provides GPU-accelerated NV12-to-RGB color conversion and image resize,
//! used by the GPU detection pipeline to preprocess NVDEC-decoded frames
//! without leaving the GPU.
//!
//! NPP libraries are loaded dynamically at runtime - no compile-time
//! CUDA SDK dependency. Returns [`NppError::NotAvailable`] if NPP is
//! not installed.
//!
//! ## Libraries
//!
//! - `libnppicc` - Color conversion (`nppiNV12ToRGB_8u_P2C3R`)
//! - `libnppig` - Geometry transforms (`nppiResize_8u_C3R`)

use std::sync::OnceLock;

use crate::cuda_interop::CUdeviceptr;

/// Errors from NPP operations.
#[derive(Debug, thiserror::Error)]
pub enum NppError {
    /// NPP libraries not available (not installed).
    #[error("NPP not available: {0}")]
    NotAvailable(String),

    /// NPP function returned an error status.
    #[error("NPP error {code} in {function}")]
    NppCall { function: &'static str, code: i32 },
}

// ── NPP types ──────────────────────────────────────────────────────

/// 2D size for NPP operations.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NppiSize {
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
}

/// Rectangle ROI for NPP operations.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NppiRect {
    /// X offset.
    pub x: i32,
    /// Y offset.
    pub y: i32,
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
}

/// NPP interpolation mode: bilinear.
const NPPI_INTER_LINEAR: i32 = 1;

/// NPP mirror axis: flip both horizontally and vertically (180-degree rotation).
const NPPI_AXIS_BOTH: i32 = 2;

/// NPP stream context for `_Ctx` API variants.
///
/// NPP 13 removed the non-Ctx functions; all calls require this struct.
/// A zeroed context uses the default CUDA stream and works for synchronous
/// operations. For production use, populate device properties via the
/// CUDA runtime API.
#[repr(C)]
#[derive(Clone, Copy)]
struct NppStreamContext {
    /// CUDA stream handle (0 = default stream).
    h_stream: *mut std::ffi::c_void,
    /// CUDA device ID.
    n_cuda_device_id: i32,
    /// Number of streaming multiprocessors.
    n_multi_processor_count: i32,
    /// Max threads per SM.
    n_max_threads_per_multi_processor: i32,
    /// Max threads per block.
    n_max_threads_per_block: i32,
    /// Shared memory per block in bytes.
    n_shared_mem_per_block: usize,
    /// Compute capability major.
    n_cuda_dev_attr_compute_capability_major: i32,
    /// Compute capability minor.
    n_cuda_dev_attr_compute_capability_minor: i32,
    /// Stream flags from `cudaStreamGetFlags`.
    n_stream_flags: u32,
    /// Reserved, must be zero.
    n_reserved0: i32,
}

impl Default for NppStreamContext {
    fn default() -> Self {
        // Zero-initialized: default stream, zeroed device props.
        // Works for synchronous NPP operations.
        unsafe { std::mem::zeroed() }
    }
}

// ── Dynamic loader ─────────────────────────────────────────────────

/// Dynamically loaded NPP functions (Ctx variants for NPP 12+/13+).
struct NppFunctions {
    _lib_nppicc: libloading::Library,
    _lib_nppig: libloading::Library,

    /// `nppiNV12ToRGB_8u_P2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx)`
    ///
    /// `pSrc` is an array of two device pointers: `[Y_ptr, UV_ptr]`.
    /// `nSrcStep` is the row pitch for both planes.
    nppi_nv12_to_rgb: unsafe extern "C" fn(
        *const *const u8, // pSrc[2]: [Y, UV]
        i32,              // nSrcStep
        *mut u8,          // pDst (packed RGB)
        i32,              // nDstStep
        NppiSize,         // oSizeROI
        NppStreamContext, // nppStreamCtx
    ) -> i32,

    /// `nppiResize_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, oDstSize, oDstROI, eInterpolation, nppStreamCtx)`
    nppi_resize_c3: unsafe extern "C" fn(
        *const u8,        // pSrc
        i32,              // nSrcStep
        NppiSize,         // oSrcSize
        NppiRect,         // oSrcROI
        *mut u8,          // pDst
        i32,              // nDstStep
        NppiSize,         // oDstSize
        NppiRect,         // oDstROI
        i32,              // eInterpolation
        NppStreamContext, // nppStreamCtx
    ) -> i32,

    /// `nppiMirror_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, flip, nppStreamCtx)`
    ///
    /// In-place operation is supported (pSrc == pDst with matching step).
    nppi_mirror_c3: unsafe extern "C" fn(
        *const u8,        // pSrc
        i32,              // nSrcStep
        *mut u8,          // pDst
        i32,              // nDstStep
        NppiSize,         // oROI
        i32,              // flip (NppiAxis)
        NppStreamContext, // nppStreamCtx
    ) -> i32,
}

// SAFETY: NppFunctions contains function pointers and library handles.
// The libraries remain loaded for the process lifetime (via OnceLock).
// NPP calls operate on GPU memory and are synchronized via cuCtxSynchronize.
unsafe impl Send for NppFunctions {}
unsafe impl Sync for NppFunctions {}

/// Global NPP function table, loaded once.
static NPP: OnceLock<Result<NppFunctions, String>> = OnceLock::new();

impl NppFunctions {
    fn load() -> Result<Self, String> {
        unsafe {
            // Load NPP color conversion library.
            #[cfg(target_os = "linux")]
            let lib_nppicc = libloading::Library::new("libnppicc.so.13")
                .or_else(|_| libloading::Library::new("libnppicc.so.12"))
                .or_else(|_| libloading::Library::new("libnppicc.so.11"))
                .or_else(|_| libloading::Library::new("libnppicc.so"))
                .map_err(|e| format!("libnppicc: {e}"))?;

            #[cfg(target_os = "windows")]
            let lib_nppicc = libloading::Library::new("nppicc64_13.dll")
                .or_else(|_| libloading::Library::new("nppicc64_12.dll"))
                .or_else(|_| libloading::Library::new("nppicc64_11.dll"))
                .map_err(|e| format!("nppicc: {e}"))?;

            // Load NPP image geometry library.
            #[cfg(target_os = "linux")]
            let lib_nppig = libloading::Library::new("libnppig.so.13")
                .or_else(|_| libloading::Library::new("libnppig.so.12"))
                .or_else(|_| libloading::Library::new("libnppig.so.11"))
                .or_else(|_| libloading::Library::new("libnppig.so"))
                .map_err(|e| format!("libnppig: {e}"))?;

            #[cfg(target_os = "windows")]
            let lib_nppig = libloading::Library::new("nppig64_13.dll")
                .or_else(|_| libloading::Library::new("nppig64_12.dll"))
                .or_else(|_| libloading::Library::new("nppig64_11.dll"))
                .map_err(|e| format!("nppig: {e}"))?;

            macro_rules! load_sym {
                ($lib:expr, $name:literal) => {
                    *$lib
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| format!(concat!($name, ": {}"), e))?
                };
            }

            Ok(NppFunctions {
                nppi_nv12_to_rgb: load_sym!(lib_nppicc, "nppiNV12ToRGB_8u_P2C3R_Ctx"),
                nppi_resize_c3: load_sym!(lib_nppig, "nppiResize_8u_C3R_Ctx"),
                nppi_mirror_c3: load_sym!(lib_nppig, "nppiMirror_8u_C3R_Ctx"),
                _lib_nppicc: lib_nppicc,
                _lib_nppig: lib_nppig,
            })
        }
    }
}

fn npp() -> Result<&'static NppFunctions, NppError> {
    NPP.get_or_init(NppFunctions::load)
        .as_ref()
        .map_err(|e| NppError::NotAvailable(e.clone()))
}

fn check_npp(function: &'static str, status: i32) -> Result<(), NppError> {
    if status >= 0 {
        // NPP: 0 = success, positive = warning (acceptable).
        Ok(())
    } else {
        Err(NppError::NppCall {
            function,
            code: status,
        })
    }
}

// ── Public API ─────────────────────────────────────────────────────

/// Check whether NPP libraries are available on this system.
pub fn is_npp_available() -> bool {
    npp().is_ok()
}

/// Convert an NV12 frame to packed RGB on the GPU.
///
/// The source is a GPU-resident NV12 frame (separate Y and interleaved UV
/// planes). The destination is a packed RGB buffer (`width * 3` bytes per
/// row) in device memory.
///
/// Both `src_y` and `src_uv` are CUDA device pointers (from NVDEC or
/// shared textures). `dst` must be pre-allocated with at least
/// `width * height * 3` bytes.
///
/// Note: the `_Ctx` variant expects both planes to share the same pitch.
/// If `y_pitch != uv_pitch`, this will use `y_pitch` for both (correct for
/// NVDEC-style shared textures where both planes have the same alignment).
pub fn npp_nv12_to_rgb(
    src_y: CUdeviceptr,
    y_pitch: usize,
    src_uv: CUdeviceptr,
    uv_pitch: usize,
    dst: CUdeviceptr,
    width: u32,
    height: u32,
) -> Result<(), NppError> {
    // NPP's NV12ToRGB _Ctx variant takes a single nSrcStep for both planes.
    // NVDEC and shared textures always produce matching pitches, but if they
    // ever diverge the UV plane would be read with the wrong stride, producing
    // color corruption that's hard to diagnose. Warn loudly if this happens.
    if y_pitch != uv_pitch {
        log::warn!(
            "npp_nv12_to_rgb: Y pitch ({y_pitch}) != UV pitch ({uv_pitch}). \
             NPP uses a single stride for both planes; UV data may be read incorrectly."
        );
    }

    let npp = npp()?;
    let roi = NppiSize {
        width: width as i32,
        height: height as i32,
    };
    let dst_step = width as i32 * 3;

    // pSrc[2] = [Y_ptr, UV_ptr] as expected by the _Ctx variant.
    let src_ptrs: [*const u8; 2] = [src_y as *const u8, src_uv as *const u8];

    unsafe {
        check_npp(
            "nppiNV12ToRGB_8u_P2C3R_Ctx",
            (npp.nppi_nv12_to_rgb)(
                src_ptrs.as_ptr(),
                y_pitch as i32,
                dst as *mut u8,
                dst_step,
                roi,
                NppStreamContext::default(),
            ),
        )?;
    }

    Ok(())
}

/// Resize a 3-channel (RGB) u8 image on the GPU with bilinear interpolation.
///
/// Both `src` and `dst` are CUDA device pointers. The source is resized
/// into the destination ROI region, allowing letterbox placement.
pub fn npp_resize_c3(
    src: CUdeviceptr,
    src_w: u32,
    src_h: u32,
    dst: CUdeviceptr,
    dst_w: u32,
    dst_h: u32,
    dst_roi: NppiRect,
) -> Result<(), NppError> {
    let npp = npp()?;
    let src_size = NppiSize {
        width: src_w as i32,
        height: src_h as i32,
    };
    let src_roi = NppiRect {
        x: 0,
        y: 0,
        width: src_w as i32,
        height: src_h as i32,
    };
    let dst_size = NppiSize {
        width: dst_w as i32,
        height: dst_h as i32,
    };
    let src_step = src_w as i32 * 3;
    let dst_step = dst_w as i32 * 3;

    unsafe {
        check_npp(
            "nppiResize_8u_C3R_Ctx",
            (npp.nppi_resize_c3)(
                src as *const u8,
                src_step,
                src_size,
                src_roi,
                dst as *mut u8,
                dst_step,
                dst_size,
                dst_roi,
                NPPI_INTER_LINEAR,
                NppStreamContext::default(),
            ),
        )?;
    }

    Ok(())
}

/// Mirror (flip) a 3-channel RGB u8 image on the GPU along both axes.
///
/// This performs a 180-degree rotation by flipping both horizontally and
/// vertically. Used to correct upside-down frames from cameras with
/// rotation=180 metadata (e.g. DJI) in the GPU zero-copy detection path.
///
/// Operates in-place: `src` and `dst` may be the same CUDA device pointer.
pub fn npp_mirror_c3(
    src: CUdeviceptr,
    dst: CUdeviceptr,
    width: u32,
    height: u32,
) -> Result<(), NppError> {
    let npp = npp()?;
    let roi = NppiSize {
        width: width as i32,
        height: height as i32,
    };
    let step = width as i32 * 3;

    unsafe {
        check_npp(
            "nppiMirror_8u_C3R_Ctx",
            (npp.nppi_mirror_c3)(
                src as *const u8,
                step,
                dst as *mut u8,
                step,
                roi,
                NPPI_AXIS_BOTH,
                NppStreamContext::default(),
            ),
        )?;
    }

    Ok(())
}
