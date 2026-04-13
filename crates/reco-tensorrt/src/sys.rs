//! Raw FFI bindings to the TensorRT C wrapper.
//!
//! These match the declarations in `csrc/tensorrt_wrapper.h`.
//! All functions are unsafe C FFI - use the safe wrappers in
//! [`engine`](super::engine) and [`cuda`](super::cuda) instead.

use std::ffi::c_void;
use std::os::raw::c_char;

unsafe extern "C" {
    // Lifecycle
    pub fn trt_create_logger(severity: i32) -> *mut c_void;
    pub fn trt_create_runtime(logger: *mut c_void) -> *mut c_void;
    pub fn trt_deserialize_engine(
        runtime: *mut c_void,
        data: *const c_void,
        size: usize,
    ) -> *mut c_void;
    pub fn trt_create_execution_context(engine: *mut c_void) -> *mut c_void;

    // Query
    pub fn trt_get_nb_bindings(engine: *mut c_void) -> i32;
    pub fn trt_get_binding_name(engine: *mut c_void, index: i32) -> *const c_char;
    pub fn trt_binding_is_input(engine: *mut c_void, index: i32) -> i32;
    pub fn trt_get_binding_dims(engine: *mut c_void, index: i32, nb_dims: *mut i32, dims: *mut i32);
    pub fn trt_get_binding_data_type(engine: *mut c_void, index: i32) -> i32;

    // Inference
    pub fn trt_enqueue_v2(
        context: *mut c_void,
        bindings: *mut *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // CUDA helpers
    pub fn trt_cuda_malloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn trt_cuda_free(ptr: *mut c_void);
    pub fn trt_cuda_memcpy_h2d(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn trt_cuda_memcpy_d2h(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn trt_cuda_stream_create(stream: *mut *mut c_void) -> i32;
    pub fn trt_cuda_stream_destroy(stream: *mut c_void);
    pub fn trt_cuda_stream_synchronize(stream: *mut c_void) -> i32;

    // Cleanup
    pub fn trt_destroy_context(ctx: *mut c_void);
    pub fn trt_destroy_engine(engine: *mut c_void);
    pub fn trt_destroy_runtime(runtime: *mut c_void);
    pub fn trt_destroy_logger(logger: *mut c_void);
}
