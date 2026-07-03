//! The wgpu-free geometry leaf (L0).
//!
//! Everything a consumer needs to reason about the virtual camera and
//! its pose without touching the GPU: the view-matrix construction, the
//! clip constants, and (as the leaf grows) the camera basis and rig
//! correction. Both executors and the detection mapping derive from this
//! one source, which is what keeps the CPU/GPU agreement a
//! by-construction property.

mod matrices;

pub(crate) use matrices::{
    FAR_PLANE, NEAR_PLANE, matrix4_to_columns, opengl_to_wgpu_matrix, view_matrix,
};
