//! Safe wrappers for TensorRT engine, context, and binding metadata.
//!
//! Load a pre-built `.engine` file, query its input/output bindings,
//! and run inference with GPU device pointers.

use std::ffi::{CStr, c_void};
use std::path::Path;

use super::sys;

/// TensorRT data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrtDataType {
    /// 32-bit floating point.
    Float,
    /// 16-bit floating point (FP16).
    Half,
    /// 8-bit integer (INT8).
    Int8,
    /// 32-bit integer.
    Int32,
}

impl TrtDataType {
    /// Parse from TensorRT's raw enum value.
    pub fn from_raw(value: i32) -> Result<Self, TrtError> {
        match value {
            0 => Ok(Self::Float),
            1 => Ok(Self::Half),
            2 => Ok(Self::Int8),
            3 => Ok(Self::Int32),
            _ => Err(TrtError::Runtime(format!(
                "unknown TensorRT data type: {value}"
            ))),
        }
    }

    /// Byte size of a single element.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float | Self::Int32 => 4,
            Self::Half => 2,
            Self::Int8 => 1,
        }
    }
}

/// Metadata for a single engine binding (input or output tensor).
#[derive(Debug, Clone)]
pub struct BindingInfo {
    /// Tensor name.
    pub name: String,
    /// Whether this is an input binding.
    pub is_input: bool,
    /// Tensor dimensions (e.g. `[1, 3, 640, 640]` for input).
    pub dims: Vec<i32>,
    /// Element data type.
    pub data_type: TrtDataType,
    /// Total byte size of the tensor (product of dims * element size).
    pub byte_size: usize,
}

/// Errors from TensorRT operations.
#[derive(Debug)]
pub enum TrtError {
    /// Runtime error from TensorRT or CUDA.
    Runtime(String),
    /// I/O error (e.g. engine file not found).
    Io(std::io::Error),
}

impl std::fmt::Display for TrtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Runtime(msg) => write!(f, "TensorRT: {msg}"),
            Self::Io(err) => write!(f, "TensorRT I/O: {err}"),
        }
    }
}

impl std::error::Error for TrtError {}

impl From<std::io::Error> for TrtError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// A deserialized TensorRT engine.
///
/// Created from a pre-built `.engine` file via [`from_file`](Self::from_file).
/// The engine owns the TRT runtime and logger - they are destroyed on drop.
///
/// # Drop order
///
/// When stored in a struct alongside [`TrtContext`], declare `TrtContext`
/// BEFORE `TrtEngine` so the context is dropped first (Rust drops fields
/// in declaration order).
pub struct TrtEngine {
    engine: *mut c_void,
    runtime: *mut c_void,
    logger: *mut c_void,
}

/// SAFETY: TrtEngine is a GPU resource handle. It can be sent to another
/// thread as long as CUDA context management is done properly (which we
/// handle via cuda_ensure_context). It is NOT Sync (concurrent access
/// from multiple threads is unsafe).
unsafe impl Send for TrtEngine {}

impl TrtEngine {
    /// Load a serialized TensorRT engine from a file.
    ///
    /// The `.engine` file must be built for the exact GPU architecture
    /// and TensorRT version of the target device. Use `trtexec` or
    /// Ultralytics `model.export(format="engine")` to create one.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TrtError> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;

        // Severity 2 = WARNING (suppress INFO/VERBOSE during load)
        let logger = unsafe { sys::trt_create_logger(2) };
        if logger.is_null() {
            return Err(TrtError::Runtime("failed to create TRT logger".into()));
        }

        let runtime = unsafe { sys::trt_create_runtime(logger) };
        if runtime.is_null() {
            unsafe { sys::trt_destroy_logger(logger) };
            return Err(TrtError::Runtime("failed to create TRT runtime".into()));
        }

        let engine = unsafe {
            sys::trt_deserialize_engine(runtime, data.as_ptr() as *const c_void, data.len())
        };
        if engine.is_null() {
            unsafe {
                sys::trt_destroy_runtime(runtime);
                sys::trt_destroy_logger(logger);
            }
            return Err(TrtError::Runtime(format!(
                "failed to deserialize engine from {}",
                path.display()
            )));
        }

        Ok(Self {
            engine,
            runtime,
            logger,
        })
    }

    /// Query all input/output bindings of the engine.
    pub fn bindings(&self) -> Result<Vec<BindingInfo>, TrtError> {
        let nb = unsafe { sys::trt_get_nb_bindings(self.engine) };
        let mut bindings = Vec::with_capacity(nb as usize);

        for i in 0..nb {
            let name_ptr = unsafe { sys::trt_get_binding_name(self.engine, i) };
            let name = if name_ptr.is_null() {
                format!("binding_{i}")
            } else {
                unsafe { CStr::from_ptr(name_ptr) }
                    .to_string_lossy()
                    .into_owned()
            };

            let is_input = unsafe { sys::trt_binding_is_input(self.engine, i) } != 0;

            let mut nb_dims: i32 = 0;
            let mut dims = [0i32; 8]; // TRT max dims is 8
            unsafe {
                sys::trt_get_binding_dims(self.engine, i, &mut nb_dims, dims.as_mut_ptr());
            }
            let dims = dims[..nb_dims as usize].to_vec();

            let raw_dtype = unsafe { sys::trt_get_binding_data_type(self.engine, i) };
            let data_type = TrtDataType::from_raw(raw_dtype)?;

            let num_elements: usize = dims.iter().map(|&d| d.max(1) as usize).product();
            let byte_size = num_elements * data_type.byte_size();

            bindings.push(BindingInfo {
                name,
                is_input,
                dims,
                data_type,
                byte_size,
            });
        }

        Ok(bindings)
    }

    /// Create an execution context for running inference.
    ///
    /// The context borrows from the engine internally. The engine MUST
    /// outlive the context. When stored together, declare the context
    /// field before the engine field.
    pub fn create_context(&self) -> Result<TrtContext, TrtError> {
        let ctx = unsafe { sys::trt_create_execution_context(self.engine) };
        if ctx.is_null() {
            return Err(TrtError::Runtime(
                "failed to create execution context".into(),
            ));
        }
        Ok(TrtContext { ptr: ctx })
    }
}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        unsafe {
            sys::trt_destroy_engine(self.engine);
            sys::trt_destroy_runtime(self.runtime);
            sys::trt_destroy_logger(self.logger);
        }
    }
}

/// A TensorRT execution context for running inference.
///
/// Created from [`TrtEngine::create_context`]. The engine must outlive
/// the context.
pub struct TrtContext {
    ptr: *mut c_void,
}

/// SAFETY: Same rationale as TrtEngine.
unsafe impl Send for TrtContext {}

impl TrtContext {
    /// Run async inference on the given bindings.
    ///
    /// `bindings` must contain one device pointer per engine binding,
    /// in the order returned by [`TrtEngine::bindings`]. Each pointer
    /// must point to GPU memory of at least `BindingInfo::byte_size` bytes.
    ///
    /// Inference is enqueued on `stream` and may not be complete when
    /// this returns. Call [`CudaStream::synchronize`](super::cuda::CudaStream::synchronize)
    /// before reading output data.
    pub fn enqueue(
        &self,
        bindings: &mut [*mut c_void],
        stream: &super::cuda::CudaStream,
    ) -> Result<(), TrtError> {
        let result =
            unsafe { sys::trt_enqueue_v2(self.ptr, bindings.as_mut_ptr(), stream.as_ptr()) };
        if result != 0 {
            return Err(TrtError::Runtime("enqueue inference failed".into()));
        }
        Ok(())
    }
}

impl Drop for TrtContext {
    fn drop(&mut self) {
        unsafe {
            sys::trt_destroy_context(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_from_raw() {
        assert_eq!(TrtDataType::from_raw(0).unwrap(), TrtDataType::Float);
        assert_eq!(TrtDataType::from_raw(1).unwrap(), TrtDataType::Half);
        assert_eq!(TrtDataType::from_raw(2).unwrap(), TrtDataType::Int8);
        assert_eq!(TrtDataType::from_raw(3).unwrap(), TrtDataType::Int32);
        assert!(TrtDataType::from_raw(99).is_err());
    }

    #[test]
    fn data_type_byte_size() {
        assert_eq!(TrtDataType::Float.byte_size(), 4);
        assert_eq!(TrtDataType::Half.byte_size(), 2);
        assert_eq!(TrtDataType::Int8.byte_size(), 1);
        assert_eq!(TrtDataType::Int32.byte_size(), 4);
    }
}
