//! Native CoreML inference for macOS.
//!
//! Wraps Apple's CoreML framework via `objc2-core-ml` to run ML models
//! on the Neural Engine (ANE), GPU, or CPU. Unlike the ORT CoreML EP
//! which converts ONNX at runtime (often falling back to CPU), this
//! loads pre-compiled `.mlmodelc` bundles that CoreML can dispatch
//! directly to the ANE.
//!
//! The key optimization is zero-copy input: the Metal preprocessing
//! pipeline produces a float32 CHW tensor in a shared-memory MTLBuffer.
//! We wrap that buffer as an `MLMultiArray` without copying, so the
//! ANE reads directly from unified memory.

use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;

use block2::RcBlock;

use objc2::AnyThread;
use objc2::msg_send_id;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

/// Error type for CoreML inference operations.
#[derive(Debug)]
pub enum CoreMlError {
    /// Failed to load the MLModel from the compiled bundle.
    ModelLoad(String),
    /// Failed to create an MLMultiArray for input/output.
    ArrayCreation(String),
    /// Failed to create a feature provider.
    FeatureProvider(String),
    /// Prediction failed.
    Prediction(String),
    /// Failed to extract output.
    OutputExtraction(String),
}

impl std::fmt::Display for CoreMlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreMlError::ModelLoad(e) => write!(f, "CoreML model load: {e}"),
            CoreMlError::ArrayCreation(e) => write!(f, "CoreML array creation: {e}"),
            CoreMlError::FeatureProvider(e) => write!(f, "CoreML feature provider: {e}"),
            CoreMlError::Prediction(e) => write!(f, "CoreML prediction: {e}"),
            CoreMlError::OutputExtraction(e) => write!(f, "CoreML output extraction: {e}"),
        }
    }
}

impl std::error::Error for CoreMlError {}

fn nserror_to_string(e: &Retained<NSError>) -> String {
    e.localizedDescription().to_string()
}

/// Native CoreML model wrapper for inference on ANE/GPU/CPU.
///
/// Loads a pre-compiled `.mlmodelc` bundle and runs inference using
/// Apple's CoreML framework, which can dispatch to the Neural Engine
/// for maximum performance on Apple Silicon.
pub struct CoreMlModel {
    model: Retained<MLModel>,
    input_name: Retained<NSString>,
    output_name: Retained<NSString>,
    /// Pre-allocated shape and strides arrays for the input tensor.
    input_shape: Retained<NSArray<NSNumber>>,
    input_strides: Retained<NSArray<NSNumber>>,
}

// SAFETY: MLModel.prediction(from:) is documented as thread-safe for
// a single model instance. We only call it from the session's frame loop
// thread, never concurrently.
unsafe impl Send for CoreMlModel {}

impl CoreMlModel {
    /// Load a CoreML model from a compiled `.mlmodelc` directory.
    ///
    /// `input_name` and `output_name` are the tensor names in the model
    /// (typically "images" and "output0" for YOLO models).
    ///
    /// `input_shape` is the NCHW shape, e.g. `[1, 3, 960, 960]`.
    pub fn load(
        model_path: impl AsRef<Path>,
        input_name: &str,
        output_name: &str,
        input_shape: [i64; 4],
    ) -> Result<Self, CoreMlError> {
        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| CoreMlError::ModelLoad("invalid path".into()))?;

        unsafe {
            let config = MLModelConfiguration::new();
            config.setComputeUnits(MLComputeUnits::All);

            let ns_path = NSString::from_str(path_str);
            let url = NSURL::fileURLWithPath(&ns_path);

            let model = MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
                .map_err(|e| CoreMlError::ModelLoad(nserror_to_string(&e)))?;

            log::info!("CoreML model loaded: {} (compute_units=All)", path_str);

            // Pre-allocate shape and strides NSArrays (reused every frame).
            let [n, c, h, w] = input_shape;
            let shape = NSArray::from_retained_slice(&[
                NSNumber::new_i64(n),
                NSNumber::new_i64(c),
                NSNumber::new_i64(h),
                NSNumber::new_i64(w),
            ]);

            // Row-major NCHW strides (in element counts, not bytes).
            let strides = NSArray::from_retained_slice(&[
                NSNumber::new_i64(c * h * w),
                NSNumber::new_i64(h * w),
                NSNumber::new_i64(w),
                NSNumber::new_i64(1),
            ]);

            Ok(Self {
                model,
                input_name: NSString::from_str(input_name),
                output_name: NSString::from_str(output_name),
                input_shape: shape,
                input_strides: strides,
            })
        }
    }

    /// Run inference on a pre-computed float32 CHW tensor.
    ///
    /// `tensor_data` must be a valid pointer to a float32 buffer with
    /// exactly `product(input_shape)` elements. The buffer is NOT copied -
    /// CoreML reads directly from this pointer (unified memory).
    ///
    /// Returns the output tensor data as a Vec<f32> and the number of
    /// detections (second dimension of the output shape).
    pub fn predict(
        &self,
        tensor_ptr: *mut f32,
        _tensor_len: usize,
    ) -> Result<(usize, Vec<f32>), CoreMlError> {
        unsafe {
            // Wrap the Metal shared buffer as an MLMultiArray (zero-copy).
            let data_ptr = NonNull::new(tensor_ptr as *mut c_void)
                .ok_or_else(|| CoreMlError::ArrayCreation("null tensor pointer".into()))?;

            let input_array =
                MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                    MLMultiArray::alloc(),
                    data_ptr,
                    &self.input_shape,
                    MLMultiArrayDataType::Float32,
                    &self.input_strides,
                    None, // caller owns memory
                )
                .map_err(|e| CoreMlError::ArrayCreation(nserror_to_string(&e)))?;

            // Create feature provider with the input tensor.
            let feature_value = MLFeatureValue::featureValueWithMultiArray(&input_array);

            let feature_obj: Retained<AnyObject> =
                Retained::into_super(Retained::into_super(feature_value));
            let dict: Retained<NSDictionary<NSString, AnyObject>> =
                NSDictionary::from_retained_objects(&[&*self.input_name], &[feature_obj]);

            let provider = MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
            .map_err(|e| CoreMlError::FeatureProvider(nserror_to_string(&e)))?;

            // Run prediction.
            let input_ref = ProtocolObject::from_ref(&*provider);
            let output = self
                .model
                .predictionFromFeatures_error(input_ref)
                .map_err(|e| CoreMlError::Prediction(nserror_to_string(&e)))?;

            // Extract output tensor.
            let output_value: Option<Retained<MLFeatureValue>> =
                msg_send_id![&*output, featureValueForName: &*self.output_name];

            let output_value = output_value
                .ok_or_else(|| CoreMlError::OutputExtraction("output feature not found".into()))?;

            let output_array = output_value.multiArrayValue().ok_or_else(|| {
                CoreMlError::OutputExtraction("output is not MLMultiArray".into())
            })?;

            // Read output data. Shape is [1, N, 6] for YOLO end2end.
            let out_shape = output_array.shape();
            let n_detections = if out_shape.count() >= 2 {
                let num: &NSNumber = &out_shape.objectAtIndex(1);
                num.integerValue() as usize
            } else {
                0
            };

            let count = output_array.count() as usize;
            let mut data = Vec::with_capacity(count);
            let data_ptr_out = data.as_mut_ptr();
            let block = RcBlock::new(
                |bytes: NonNull<c_void>, _size: objc2_foundation::NSInteger| {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr() as *const f32,
                        data_ptr_out,
                        count,
                    );
                },
            );
            output_array.getBytesWithHandler(&block);
            drop(block);
            data.set_len(count);

            Ok((n_detections, data))
        }
    }
}
