//! Rendering subsystem: GPU pipeline, scene geometry, viewport, and frame planes.
//!
//! Groups the modules responsible for turning decoded video frames into a
//! stitched panoramic output on the GPU.

pub mod pipeline;
pub mod planes;
pub mod renderer;
pub mod scene;
pub mod viewport;

/// Strip sRGB encoding from a texture format.
///
/// The stitch shader outputs sRGB-encoded values directly (BT.709 YCbCr
/// to R'G'B'). Rendering to an sRGB-format surface would apply sRGB
/// encoding again, causing double-gamma (faded colors). Returns the
/// equivalent linear format for surface creation.
pub fn strip_srgb(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
        other => other,
    }
}
