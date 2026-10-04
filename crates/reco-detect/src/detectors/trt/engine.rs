//! TensorRT ownership and the static end-to-end detector binding contract.

use std::ffi::{CStr, c_void};
use std::path::Path;

use super::{Result, cuda::CudaStream, sys};

pub(super) struct BindingInfo {
    pub name: String,
    pub is_input: bool,
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub linear_device: bool,
}

pub(super) struct ModelLayout {
    pub input_index: usize,
    pub size: u32,
    pub input_bytes: usize,
    pub output_elements: usize,
}

impl ModelLayout {
    pub fn new(bindings: &[BindingInfo]) -> Result<Self> {
        if bindings.len() != 2 || bindings.iter().filter(|b| b.is_input).count() != 1 {
            return Err("TensorRT detector requires exactly one input and one output".into());
        }
        for binding in bindings {
            if binding.data_type != 0 || !binding.linear_device {
                return Err(format!(
                    "TensorRT binding '{}' must be linear device FP32",
                    binding.name
                )
                .into());
            }
        }
        let input_index = usize::from(!bindings[0].is_input);
        let size = match bindings[input_index].dims.as_slice() {
            [1, 3, h, w] if h == w && *h > 0 && *h <= reco_core::calibration::MAX_DIM as i64 => {
                *h as u32
            }
            dims => {
                return Err(
                    format!("TensorRT input must be static [1, 3, S, S], got {dims:?}").into(),
                );
            }
        };
        let output_elements = match bindings[1 - input_index].dims.as_slice() {
            [1, count, 6] if *count > 0 => {
                usize::try_from(*count).ok().and_then(|n| n.checked_mul(6))
            }
            _ => None,
        }
        .filter(|n| *n <= i32::MAX as usize)
        .ok_or("TensorRT output must be static [1, N, 6] with a supported element count")?;
        output_elements
            .checked_mul(size_of::<f32>())
            .ok_or("TensorRT output size overflow")?;
        let input_bytes = (size as usize)
            .checked_mul(size as usize)
            .and_then(|n| n.checked_mul(3 * size_of::<f32>()))
            .ok_or("TensorRT input size overflow")?;
        Ok(Self {
            input_index,
            size,
            input_bytes,
            output_elements,
        })
    }
}

pub(super) struct TrtEngine {
    engine: *mut c_void,
    runtime: *mut c_void,
}

// SAFETY: Exclusive ownership; callers make the CUDA context current before use.
unsafe impl Send for TrtEngine {}

impl TrtEngine {
    pub fn from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let runtime = unsafe { sys::trt_create_runtime() };
        if runtime.is_null() {
            return Err("failed to create TensorRT runtime".into());
        }
        let engine =
            unsafe { sys::trt_deserialize_engine(runtime, data.as_ptr().cast(), data.len()) };
        if engine.is_null() {
            unsafe { sys::trt_destroy_runtime(runtime) };
            return Err(format!(
                "failed to deserialize TensorRT engine from {}",
                path.display()
            )
            .into());
        }
        Ok(Self { engine, runtime })
    }

    pub fn bindings(&self) -> Result<Vec<BindingInfo>> {
        let count = unsafe { sys::trt_get_nb_bindings(self.engine) };
        if count != 2 {
            return Err("TensorRT detector requires exactly two bindings".into());
        }
        (0..count)
            .map(|index| {
                let name = unsafe { sys::trt_get_binding_name(self.engine, index) };
                if name.is_null() {
                    return Err("TensorRT binding has no name".into());
                }
                let name = unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned();
                let mut dims = [0i64; 8];
                let count = unsafe {
                    sys::trt_get_binding_dims(
                        self.engine,
                        index,
                        dims.as_mut_ptr(),
                        dims.len() as i32,
                    )
                };
                if !(0..=dims.len() as i32).contains(&count) {
                    return Err(format!("invalid TensorRT dimensions for '{name}'").into());
                }
                Ok(BindingInfo {
                    name,
                    is_input: unsafe { sys::trt_binding_is_input(self.engine, index) } != 0,
                    dims: dims[..count as usize].to_vec(),
                    data_type: unsafe { sys::trt_get_binding_data_type(self.engine, index) },
                    linear_device: unsafe { sys::trt_binding_is_linear_device(self.engine, index) }
                        != 0,
                })
            })
            .collect()
    }

    pub fn into_context(self) -> Result<TrtContext> {
        let ptr = unsafe { sys::trt_create_execution_context(self.engine) };
        if ptr.is_null() {
            return Err("failed to create TensorRT execution context".into());
        }
        Ok(TrtContext { ptr, _engine: self })
    }
}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        if super::cuda::context_for_drop() {
            unsafe {
                sys::trt_destroy_engine(self.engine);
                sys::trt_destroy_runtime(self.runtime);
            }
        }
    }
}

pub(super) struct TrtContext {
    ptr: *mut c_void,
    _engine: TrtEngine,
}

// SAFETY: Context and its engine have exclusive ownership and are never used concurrently.
unsafe impl Send for TrtContext {}

impl TrtContext {
    // Both buffers must match the validated bindings and outlive the queued inference.
    pub unsafe fn enqueue(
        &mut self,
        bindings: &mut [*mut c_void; 2],
        stream: &CudaStream,
    ) -> Result<()> {
        if unsafe { sys::trt_enqueue_v2(self.ptr, bindings.as_mut_ptr(), stream.as_ptr()) } != 0 {
            return Err("TensorRT enqueue failed".into());
        }
        Ok(())
    }
}

impl Drop for TrtContext {
    fn drop(&mut self) {
        if super::cuda::context_for_drop() {
            unsafe { sys::trt_destroy_context(self.ptr) };
        }
    }
}
