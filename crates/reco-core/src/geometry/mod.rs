//! The wgpu-free geometry leaf (L0).
//!
//! Everything a consumer needs to reason about the virtual camera and
//! its pose without touching the GPU: the view-matrix construction, the
//! clip constants, and (as the leaf grows) the camera basis and rig
//! correction. Both executors and the detection mapping derive from this
//! one source, which is what keeps the CPU/GPU agreement a
//! by-construction property.

mod matrices;
mod rig_correction;
mod types;
mod virtual_camera;

// The deliberate public surface: the pose currency, the camera basis,
// and the orient chain a thin consumer needs to reason about (or
// reproduce) the virtual camera without the GPU. The rasterization
// internals (clip-space correction, column packing) stay crate-private
// until a consumer demonstrates the need.
pub use matrices::{FAR_PLANE, NEAR_PLANE, view_matrix};
pub use rig_correction::{resolve_render_pose, world_to_render_pose};
pub use types::{CameraId, Pose};
pub use virtual_camera::VirtualCamera;

#[cfg(feature = "gpu")]
pub(crate) use matrices::matrix4_to_columns;
pub(crate) use matrices::opengl_to_wgpu_matrix;
pub(crate) use rig_correction::render_viewport_roll;
