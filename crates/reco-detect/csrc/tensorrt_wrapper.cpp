#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "tensorrt_wrapper.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <cstdio>

class SimpleLogger : public nvinfer1::ILogger {
public:
    Severity mSeverity;
    SimpleLogger(Severity severity) : mSeverity(severity) {}
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= mSeverity) {
            const char* level = "";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
                case Severity::kERROR:          level = "ERROR"; break;
                case Severity::kWARNING:        level = "WARNING"; break;
                case Severity::kINFO:           level = "INFO"; break;
                case Severity::kVERBOSE:        level = "VERBOSE"; break;
            }
            fprintf(stderr, "[TRT %s] %s\n", level, msg);
        }
    }
};

extern "C" {

void* trt_create_logger(int severity) {
    auto sev = static_cast<nvinfer1::ILogger::Severity>(severity);
    return new SimpleLogger(sev);
}

void* trt_create_runtime(void* logger) {
    auto* l = static_cast<nvinfer1::ILogger*>(logger);
    return nvinfer1::createInferRuntime(*l);
}

void* trt_deserialize_engine(void* runtime, const void* data, size_t size) {
    auto* r = static_cast<nvinfer1::IRuntime*>(runtime);
#if NV_TENSORRT_MAJOR >= 10
    return r->deserializeCudaEngine(data, size);
#else
    return r->deserializeCudaEngine(data, size, nullptr);
#endif
}

void* trt_create_execution_context(void* engine) {
    auto* e = static_cast<nvinfer1::ICudaEngine*>(engine);
    return e->createExecutionContext();
}

int trt_get_nb_bindings(void* engine) {
    auto* e = static_cast<nvinfer1::ICudaEngine*>(engine);
#if NV_TENSORRT_MAJOR >= 10
    return e->getNbIOTensors();
#else
    return e->getNbBindings();
#endif
}

const char* trt_get_binding_name(void* engine, int index) {
    auto* e = static_cast<nvinfer1::ICudaEngine*>(engine);
#if NV_TENSORRT_MAJOR >= 10
    return e->getIOTensorName(index);
#else
    return e->getBindingName(index);
#endif
}

int trt_binding_is_input(void* engine, int index) {
    auto* e = static_cast<nvinfer1::ICudaEngine*>(engine);
#if NV_TENSORRT_MAJOR >= 10
    const char* name = e->getIOTensorName(index);
    return (e->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) ? 1 : 0;
#else
    return e->bindingIsInput(index) ? 1 : 0;
#endif
}

void trt_get_binding_dims(void* engine, int index, int* nb_dims, int* dims) {
    auto* e = static_cast<nvinfer1::ICudaEngine*>(engine);
#if NV_TENSORRT_MAJOR >= 10
    const char* name = e->getIOTensorName(index);
    nvinfer1::Dims d = e->getTensorShape(name);
#else
    nvinfer1::Dims d = e->getBindingDimensions(index);
#endif
    *nb_dims = d.nbDims;
    for (int i = 0; i < d.nbDims; i++) {
        dims[i] = d.d[i];
    }
}

int trt_get_binding_data_type(void* engine, int index) {
    auto* e = static_cast<nvinfer1::ICudaEngine*>(engine);
#if NV_TENSORRT_MAJOR >= 10
    const char* name = e->getIOTensorName(index);
    return static_cast<int>(e->getTensorDataType(name));
#else
    return static_cast<int>(e->getBindingDataType(index));
#endif
}

int trt_enqueue_v2(void* context, void** bindings, void* stream) {
    auto* c = static_cast<nvinfer1::IExecutionContext*>(context);
#if NV_TENSORRT_MAJOR >= 10
    // TRT 10+: set tensor addresses by name, then enqueueV3
    const auto& engine = c->getEngine();
    int nb = engine.getNbIOTensors();
    for (int i = 0; i < nb; i++) {
        const char* name = engine.getIOTensorName(i);
        if (!c->setTensorAddress(name, bindings[i])) {
            return -1;
        }
    }
    bool ok = c->enqueueV3(static_cast<cudaStream_t>(stream));
#else
    bool ok = c->enqueueV2(bindings, static_cast<cudaStream_t>(stream), nullptr);
#endif
    return ok ? 0 : -1;
}

int trt_cuda_malloc(void** ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

void trt_cuda_free(void* ptr) {
    cudaFree(ptr);
}

int trt_cuda_memcpy_h2d(void* dst, const void* src, size_t size, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                       static_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int trt_cuda_memcpy_d2h(void* dst, const void* src, size_t size, void* stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                       static_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int trt_cuda_stream_create(void** stream) {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    *stream = static_cast<void*>(s);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

void trt_cuda_stream_destroy(void* stream) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

int trt_cuda_stream_synchronize(void* stream) {
    cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

void trt_destroy_context(void* ctx) {
    auto* c = static_cast<nvinfer1::IExecutionContext*>(ctx);
#if NV_TENSORRT_MAJOR >= 10
    delete c;
#else
    c->destroy();
#endif
}

void trt_destroy_engine(void* engine) {
    auto* e = static_cast<nvinfer1::ICudaEngine*>(engine);
#if NV_TENSORRT_MAJOR >= 10
    delete e;
#else
    e->destroy();
#endif
}

void trt_destroy_runtime(void* runtime) {
    auto* r = static_cast<nvinfer1::IRuntime*>(runtime);
#if NV_TENSORRT_MAJOR >= 10
    delete r;
#else
    r->destroy();
#endif
}

void trt_destroy_logger(void* logger) {
    auto* l = static_cast<SimpleLogger*>(logger);
    delete l;
}

} // extern "C"
