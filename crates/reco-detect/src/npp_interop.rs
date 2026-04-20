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

use reco_core::cuda_interop::CUdeviceptr;

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
/// Prior to 2026-04-19 we passed a zeroed context, which uses the
/// **default (NULL) CUDA stream**. That causes every NPP call to
/// implicitly serialize with all other CUDA work on every other
/// stream — specifically the NVDEC decode stream — producing the
/// "waving" 4K decode throughput pattern (100→60→100→70→100→50%)
/// reported in issue #186.
///
/// Now populated with a dedicated non-default CUDA stream created
/// at NPP init. Device property fields stay zeroed: NPP uses
/// internally-queried defaults when these are zero, so populating
/// them is unnecessary. The `h_stream` field is the only one that
/// matters for the serialization fix.
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

// SAFETY: NppStreamContext contains a raw pointer (h_stream) to a
// CUDA stream. CUDA streams are thread-safe for queuing operations
// (cuStreamQuery / cuLaunch* are documented as reentrant). The
// pointer is owned by the NppFunctions singleton and lives for
// the process lifetime, so sharing across threads is sound.
unsafe impl Send for NppStreamContext {}
unsafe impl Sync for NppStreamContext {}

/// Default ctor is kept for tests / diagnostic callsites that
/// legitimately want the default stream (e.g. one-shot synchronous
/// probes). Production call sites use the streamed context stored
/// on [`NppFunctions`].
impl Default for NppStreamContext {
    fn default() -> Self {
        // Zero-initialized: default stream, zeroed device props.
        unsafe { std::mem::zeroed() }
    }
}

// ── CUDA stream bindings (minimal, loaded alongside NPP) ───────────
//
// We use the CUDA **runtime** API (libcudart) rather than the
// driver API (libcuda). The runtime's cudaSetDevice+cudaStreamCreate
// auto-initialize a primary context on the calling thread if none
// exists, which is critical: NPP's OnceLock init runs before any
// other CUDA code in the process has had a chance to set up a
// context. Using driver-API cuStreamCreate returned
// CUDA_ERROR_INVALID_CONTEXT (201) in testing because nothing had
// bound a context yet.
//
// cudaStream_t is a `void*` typedef matching CUstream, so the
// resulting stream handle is interoperable with NPP's h_stream.

type CudaStream = *mut std::ffi::c_void;
type CudaError = i32;

/// `cudaStreamCreateWithFlags(&mut stream, flags)` — creates a
/// stream with explicit flags. `cudaStreamNonBlocking = 1` is
/// what we want: NPP dispatches run in parallel with NVDEC's
/// own stream, not serialized through the default null stream.
const CUDA_STREAM_NON_BLOCKING: u32 = 1;

type CudaSetDevice = unsafe extern "C" fn(i32) -> CudaError;
type CudaStreamCreateWithFlags = unsafe extern "C" fn(*mut CudaStream, u32) -> CudaError;
type CudaStreamDestroy = unsafe extern "C" fn(CudaStream) -> CudaError;

// ── Dynamic loader ─────────────────────────────────────────────────

/// Dynamically loaded NPP functions (Ctx variants for NPP 12+/13+).
struct NppFunctions {
    _lib_nppicc: libloading::Library,
    _lib_nppig: libloading::Library,
    /// libcudart handle (kept alive for the process lifetime so
    /// the stream created below stays valid). `None` when libcudart
    /// isn't available — in that case `stream_ctx` is zeroed and
    /// we're back to the pre-#186 default-stream behavior rather
    /// than failing to load NPP entirely.
    _lib_cudart: Option<libloading::Library>,

    /// Preallocated stream context shared across all NPP calls.
    /// Carries the non-default CUDA stream in `h_stream`. Built
    /// once at load time; cheap to copy (a few words) so each
    /// NPP call takes it by value with no synchronization.
    stream_ctx: NppStreamContext,

    /// `cudaStreamDestroy` handle. Held only so the OnceLock's
    /// drop path (which never runs in practice) could release the
    /// stream cleanly. Present also as documentation that the
    /// stream is intentionally leaked for the process lifetime.
    _stream_destroy: Option<CudaStreamDestroy>,

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

            // Load libcudart for stream management (#186 fix).
            // Best-effort: on failure we keep the zero-stream
            // context and log loudly. NPP still works, just with
            // the pre-#186 serialization behavior.
            #[cfg(target_os = "linux")]
            let lib_cudart = libloading::Library::new("libcudart.so.13")
                .or_else(|_| libloading::Library::new("libcudart.so.12"))
                .or_else(|_| libloading::Library::new("libcudart.so.11"))
                .or_else(|_| libloading::Library::new("libcudart.so"))
                .ok();
            #[cfg(target_os = "windows")]
            let lib_cudart = (11..=13)
                .rev()
                .find_map(|v| libloading::Library::new(format!("cudart64_{v}0.dll")).ok());

            let (stream_ctx, stream_destroy) = create_npp_stream_ctx(lib_cudart.as_ref())
                .unwrap_or_else(|reason| {
                    log::warn!(
                        "NPP: dedicated stream unavailable ({reason}); falling back to \
                         default CUDA stream. NPP calls will serialize with NVDEC work \
                         (pre-#186 behavior, issue #186 mitigation disabled for this run)."
                    );
                    (NppStreamContext::default(), None)
                });

            Ok(NppFunctions {
                nppi_nv12_to_rgb: load_sym!(lib_nppicc, "nppiNV12ToRGB_8u_P2C3R_Ctx"),
                nppi_resize_c3: load_sym!(lib_nppig, "nppiResize_8u_C3R_Ctx"),
                nppi_mirror_c3: load_sym!(lib_nppig, "nppiMirror_8u_C3R_Ctx"),
                _lib_nppicc: lib_nppicc,
                _lib_nppig: lib_nppig,
                _lib_cudart: lib_cudart,
                stream_ctx,
                _stream_destroy: stream_destroy,
            })
        }
    }
}

/// Build the NPP stream context + keep a destroy-fn handle. Returns
/// `Err(reason)` on every failure path so the caller can log one
/// unified warn line. Never panics.
///
/// Uses the CUDA runtime API so the context is auto-initialized on
/// the calling thread (the driver API requires an already-bound
/// context, which isn't guaranteed at NPP load time — NPP's
/// OnceLock init typically runs before any other CUDA code).
///
/// Borrows `lib_cudart` so the returned `stream_destroy` fn pointer
/// stays valid as long as NppFunctions holds `_lib_cudart`. The fn
/// pointer is Copy so we extract and return it by value.
unsafe fn create_npp_stream_ctx(
    lib_cudart: Option<&libloading::Library>,
) -> Result<(NppStreamContext, Option<CudaStreamDestroy>), String> {
    let cudart = lib_cudart.ok_or_else(|| "libcudart not found".to_string())?;

    // SAFETY: libloading::Library::get is unsafe because dereferencing
    // the returned Symbol calls through an untrusted library. We only
    // call these through the CUDA runtime API with well-defined
    // signatures stable since CUDA 4.0; libcudart is the system-
    // provided NVIDIA runtime. If libcudart is hostile, the entire
    // process is already compromised.
    let set_device: libloading::Symbol<CudaSetDevice> = unsafe {
        cudart
            .get(b"cudaSetDevice\0")
            .map_err(|e| format!("cudaSetDevice: {e}"))?
    };
    let stream_create: libloading::Symbol<CudaStreamCreateWithFlags> = unsafe {
        cudart
            .get(b"cudaStreamCreateWithFlags\0")
            .map_err(|e| format!("cudaStreamCreateWithFlags: {e}"))?
    };
    let stream_destroy: libloading::Symbol<CudaStreamDestroy> = unsafe {
        cudart
            .get(b"cudaStreamDestroy\0")
            .map_err(|e| format!("cudaStreamDestroy: {e}"))?
    };

    // Bind device 0 (same one reco-core's cuda_interop + zero-copy
    // path uses). This implicitly creates a primary CUDA context on
    // this thread if one doesn't exist yet — which is the key
    // difference vs the driver-API cuStreamCreate path that fails
    // with CUDA_ERROR_INVALID_CONTEXT in a fresh thread.
    //
    // SAFETY: cudaSetDevice takes an i32 device ordinal and returns
    // a cudaError; no pointer dereference, no UB even on bad input.
    let rc_set = unsafe { (*set_device)(0) };
    if rc_set != 0 {
        return Err(format!("cudaSetDevice(0) returned {rc_set}"));
    }

    let mut stream: CudaStream = std::ptr::null_mut();
    // SAFETY: out-ptr is valid; flag is the documented constant.
    let rc = unsafe { (*stream_create)(&mut stream, CUDA_STREAM_NON_BLOCKING) };
    if rc != 0 || stream.is_null() {
        return Err(format!("cudaStreamCreateWithFlags returned {rc}"));
    }

    log::info!(
        "NPP: created dedicated non-blocking CUDA stream (h_stream={:?}); NPP calls now run \
         in parallel with NVDEC on its own stream (#186 fix active)",
        stream,
    );
    // Construct the ctx directly with h_stream set; device props
    // stay zeroed. NPP uses internal defaults when they are; the
    // three functions we dispatch (NV12→RGB, resize, mirror) do
    // not consult those fields per NPP 13 docs.
    let ctx = NppStreamContext {
        h_stream: stream,
        n_cuda_device_id: 0,
        n_multi_processor_count: 0,
        n_max_threads_per_multi_processor: 0,
        n_max_threads_per_block: 0,
        n_shared_mem_per_block: 0,
        n_cuda_dev_attr_compute_capability_major: 0,
        n_cuda_dev_attr_compute_capability_minor: 0,
        n_stream_flags: 0,
        n_reserved0: 0,
    };
    Ok((ctx, Some(*stream_destroy)))
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
                npp.stream_ctx,
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
                npp.stream_ctx,
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
                npp.stream_ctx,
            ),
        )?;
    }

    Ok(())
}
