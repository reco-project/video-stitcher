//! Owned CUDA allocations and ordered transfers for native TensorRT.

use super::{Result, sys};
use reco_core::interop::cuda::{CUdeviceptr, cuda_ensure_context};
use std::ffi::c_void;

fn check(operation: &str, status: i32) -> Result<()> {
    if status != 0 {
        return Err(format!("{operation} failed: CUDA error {status}").into());
    }
    Ok(())
}

pub(super) fn context_for_drop() -> bool {
    if let Err(error) = cuda_ensure_context() {
        log::warn!("TensorRT cleanup: {error}");
        return false;
    }
    true
}

pub(super) fn synchronize(stream: *mut c_void) -> Result<()> {
    check("cudaStreamSynchronize", unsafe {
        sys::trt_cuda_stream_synchronize(stream)
    })
}

pub(super) struct CudaStream(*mut c_void);

// SAFETY: Used exclusively by an owning detector with the CUDA context current.
unsafe impl Send for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self> {
        cuda_ensure_context()?;
        let mut ptr = std::ptr::null_mut();
        check("cudaStreamCreate", unsafe {
            sys::trt_cuda_stream_create(&mut ptr)
        })?;
        Ok(Self(ptr))
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        synchronize(self.0)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if context_for_drop() {
            if let Err(error) = self.synchronize() {
                log::warn!("TensorRT cleanup: {error}");
            }
            unsafe { sys::trt_cuda_stream_destroy(self.0) };
        }
    }
}

pub(super) struct CudaBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: Exclusive allocation ownership; all queued work finishes before it is freed.
unsafe impl Send for CudaBuffer {}

impl CudaBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        cuda_ensure_context()?;
        let mut ptr = std::ptr::null_mut();
        check("cudaMalloc", unsafe {
            sys::trt_cuda_malloc(&mut ptr, bytes)
        })?;
        Ok(Self { ptr, bytes })
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr as CUdeviceptr
    }

    pub fn fill(&self, value: u8, stream: *mut c_void) -> Result<()> {
        check("cudaMemsetAsync", unsafe {
            sys::trt_cuda_memset(self.ptr, value as i32, self.bytes, stream)
        })
    }

    pub fn copy_to_host(&self, data: &mut [f32], stream: &CudaStream) -> Result<()> {
        if size_of_val(data) != self.bytes {
            return Err("TensorRT output buffer size mismatch".into());
        }
        let result = check("cudaMemcpyAsync D2H", unsafe {
            sys::trt_cuda_memcpy_d2h(
                data.as_mut_ptr().cast(),
                self.ptr,
                self.bytes,
                stream.as_ptr(),
            )
        });
        // The borrowed host storage cannot be released with a transfer in flight.
        let completion = stream.synchronize();
        result.and(completion)
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if context_for_drop() {
            unsafe { sys::trt_cuda_free(self.ptr) };
        }
    }
}
