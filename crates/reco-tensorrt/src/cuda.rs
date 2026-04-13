//! CUDA stream and buffer management for TensorRT inference.
//!
//! These use the CUDA runtime API (cudaMalloc/cudaFree) which is
//! compatible with the driver API pointers (cuMemAlloc) used elsewhere
//! in reco-core. Both share the same device address space.

use std::ffi::c_void;

use crate::engine::TrtError;
use crate::sys;

/// A CUDA stream for asynchronous GPU operations.
///
/// TensorRT inference is enqueued on a stream and runs asynchronously.
/// Call [`synchronize`](Self::synchronize) to wait for completion.
pub struct CudaStream {
    ptr: *mut c_void,
}

/// SAFETY: CUDA streams can be used from any thread as long as
/// operations are properly ordered. The stream handle itself is
/// just a pointer that can be sent between threads.
unsafe impl Send for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self, TrtError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { sys::trt_cuda_stream_create(&mut ptr) };
        if err != 0 {
            return Err(TrtError::Runtime(format!(
                "cudaStreamCreate failed: error {err}"
            )));
        }
        Ok(Self { ptr })
    }

    /// Wait for all operations on this stream to complete.
    pub fn synchronize(&self) -> Result<(), TrtError> {
        let err = unsafe { sys::trt_cuda_stream_synchronize(self.ptr) };
        if err != 0 {
            return Err(TrtError::Runtime(format!(
                "cudaStreamSynchronize failed: error {err}"
            )));
        }
        Ok(())
    }

    /// Raw stream pointer for FFI.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            sys::trt_cuda_stream_destroy(self.ptr);
        }
    }
}

/// A GPU buffer allocated via the CUDA runtime API.
///
/// Used for TensorRT output buffers. For input buffers, you can pass
/// existing driver API pointers (`CUdeviceptr`) directly to
/// [`TrtContext::enqueue`](super::engine::TrtContext::enqueue) by
/// casting: `cudeviceptr as *mut c_void`.
pub struct CudaBuffer {
    ptr: *mut c_void,
    size: usize,
}

/// SAFETY: CUDA device pointers can be used from any thread as long
/// as proper synchronization is done (via streams or context sync).
unsafe impl Send for CudaBuffer {}

impl CudaBuffer {
    /// Allocate a GPU buffer of `size` bytes.
    pub fn new(size: usize) -> Result<Self, TrtError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { sys::trt_cuda_malloc(&mut ptr, size) };
        if err != 0 {
            return Err(TrtError::Runtime(format!(
                "cudaMalloc({size} bytes) failed: error {err}"
            )));
        }
        Ok(Self { ptr, size })
    }

    /// Copy data from this GPU buffer to a host slice.
    ///
    /// The copy is asynchronous on `stream`. Call
    /// [`CudaStream::synchronize`] before reading the data.
    pub fn copy_to_host(&self, data: &mut [u8], stream: &CudaStream) -> Result<(), TrtError> {
        let copy_size = data.len().min(self.size);
        let err = unsafe {
            sys::trt_cuda_memcpy_d2h(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                copy_size,
                stream.as_ptr(),
            )
        };
        if err != 0 {
            return Err(TrtError::Runtime(format!(
                "cudaMemcpy D2H failed: error {err}"
            )));
        }
        Ok(())
    }

    /// Raw device pointer for FFI.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::trt_cuda_free(self.ptr);
        }
    }
}
