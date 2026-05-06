//! Rendering subsystem: GPU pipeline, scene geometry, viewport, and frame planes.
//!
//! Groups the modules responsible for turning decoded video frames into a
//! stitched panoramic output on the GPU.

pub mod pipeline;
pub mod planes;
pub mod renderer;
pub mod scene;
pub mod stitch_renderer;
pub mod viewport;
