#ifndef TENSORRT_WRAPPER_H
#define TENSORRT_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Lifecycle
void* trt_create_logger(int severity);
void* trt_create_runtime(void* logger);
void* trt_deserialize_engine(void* runtime, const void* data, size_t size);
void* trt_create_execution_context(void* engine);

// Query
int trt_get_nb_bindings(void* engine);
const char* trt_get_binding_name(void* engine, int index);
int trt_binding_is_input(void* engine, int index);
void trt_get_binding_dims(void* engine, int index, int* nb_dims, int* dims);
// Returns TensorRT DataType enum value (0=FLOAT, 1=HALF, 2=INT8, 3=INT32)
int trt_get_binding_data_type(void* engine, int index);

// Inference
int trt_enqueue_v2(void* context, void** bindings, void* stream);

// CUDA helpers
int trt_cuda_malloc(void** ptr, size_t size);
void trt_cuda_free(void* ptr);
int trt_cuda_memcpy_h2d(void* dst, const void* src, size_t size, void* stream);
int trt_cuda_memcpy_d2h(void* dst, const void* src, size_t size, void* stream);
int trt_cuda_stream_create(void** stream);
void trt_cuda_stream_destroy(void* stream);
int trt_cuda_stream_synchronize(void* stream);

// Cleanup
void trt_destroy_context(void* ctx);
void trt_destroy_engine(void* engine);
void trt_destroy_runtime(void* runtime);
void trt_destroy_logger(void* logger);

#ifdef __cplusplus
}
#endif

#endif // TENSORRT_WRAPPER_H
