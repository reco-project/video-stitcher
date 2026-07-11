//! Rendering subsystem: GPU pipeline, scene geometry, viewport, and frame planes.
//!
//! Groups the modules responsible for turning decoded video frames into a
//! stitched panoramic output on the GPU. `planes`, `viewport`, and
//! `scene` are pure value/math modules the CPU stitch path shares, so
//! they stay available without the `gpu` feature.

pub mod nv12_cpu;
#[cfg(feature = "gpu")]
pub mod pipeline;
pub mod planes;
#[cfg(feature = "gpu")]
pub mod renderer;
pub mod viewport;

/// The GPU half of a projection: what the stitch pipeline compiles and
/// binds. Uniforms stay the crate-wide `GpuUniforms` interface (which
/// carries the per-camera colour terms); a projection supplies its
/// shader, entry points, seam blend state, and vertex layout. A second
/// uniform layout joins the descriptor when the first projection that
/// needs one (the mono cylinder) is wired.
#[cfg(feature = "gpu")]
pub struct GpuProgram {
    /// WGSL source, compiled at pipeline creation.
    pub wgsl: &'static str,
    /// Vertex entry point in `wgsl`.
    pub vs_entry: &'static str,
    /// Fragment entry point in `wgsl`.
    pub fs_entry: &'static str,
    /// Blend state for the composite draws (the GPU dual of the CPU
    /// side's `BlendRule` ordering).
    pub blend: wgpu::BlendState,
    /// Vertex buffer layout the vertex stage consumes.
    pub vertex_layout: wgpu::VertexBufferLayout<'static>,
}

/// Strip sRGB encoding from a texture format.
///
/// The stitch shader outputs sRGB-encoded values directly (BT.709 YCbCr
/// to R'G'B'). Rendering to an sRGB-format surface would apply sRGB
/// encoding again, causing double-gamma (faded colors). Returns the
/// equivalent linear format for surface creation.
#[cfg(feature = "gpu")]
pub fn strip_srgb(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
        other => other,
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::strip_srgb;

    #[test]
    fn strip_srgb_converts_known_formats() {
        assert_eq!(
            strip_srgb(wgpu::TextureFormat::Rgba8UnormSrgb),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        assert_eq!(
            strip_srgb(wgpu::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Bgra8Unorm,
        );
    }

    #[test]
    fn strip_srgb_passes_through_non_srgb() {
        assert_eq!(
            strip_srgb(wgpu::TextureFormat::Rgba8Unorm),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        assert_eq!(
            strip_srgb(wgpu::TextureFormat::Rgba16Float),
            wgpu::TextureFormat::Rgba16Float,
        );
    }
}
