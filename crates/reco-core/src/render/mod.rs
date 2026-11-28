//! Rendering subsystem: GPU pipeline, scene geometry, viewport, and frame planes.
//!
//! Groups the modules responsible for turning decoded video frames into a
//! stitched panoramic output on the GPU. The `planes` and `viewport` value types also serve the CPU stitch path and remain available without the `gpu` feature.

pub mod color;
pub mod nv12_cpu;
#[cfg(feature = "gpu")]
pub mod pipeline;
pub mod planes;
#[cfg(feature = "gpu")]
pub(crate) mod preview;
#[cfg(feature = "gpu")]
pub mod renderer;
pub mod viewport;

/// Projection-specific geometry and uniforms for a GPU composite.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuPass {
    /// Two projected fisheye-corrected planes.
    Planes,
    /// One fullscreen ray-to-cylinder inverse map.
    Cylinder,
    /// Two fullscreen ray-to-plane maps on a fixed cylindrical output chart.
    Panorama,
}

/// Shader and draw configuration supplied by a projection.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
pub(crate) struct GpuProgram {
    /// WGSL source, compiled at pipeline creation.
    pub wgsl: &'static str,
    /// Vertex entry point in `wgsl`.
    pub vs_entry: &'static str,
    /// Fragment entry point in `wgsl`.
    pub fs_entry: &'static str,
    /// Blend state for the composite draws (the GPU dual of the CPU
    /// side's `BlendRule` ordering).
    pub blend: wgpu::BlendState,
    /// Geometry, camera resources and uniform layout consumed by this program.
    pub pass: GpuPass,
}

/// Strip sRGB encoding from a texture format.
///
/// The stitch shader writes the source's nonlinear R'G'B' values directly.
/// An sRGB-format render attachment would apply another transfer function.
/// Returns the equivalent UNORM format for surface creation.
#[cfg(feature = "gpu")]
pub fn strip_srgb(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
        other => other,
    }
}

/// Whether a render-target texture format has a dedicated alpha channel.
///
/// Planar (L-shape) GPU compositing tracks left-plane coverage in destination
/// alpha (`DstAlpha` / `OneMinusDstAlpha`). Alpha-free formats make graphics
/// APIs substitute `1.0` for missing destination alpha, breaking uncovered
/// right-plane preservation whenever `blend_width > 0.0`.
#[cfg(feature = "gpu")]
pub(crate) fn has_alpha_channel(format: wgpu::TextureFormat) -> bool {
    format.has_color_aspect()
        && !format.is_compressed()
        && !format.is_multi_planar_format()
        && format.components() == 4
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::strip_srgb;

    #[test]
    fn shared_sampling_shader_runs_in_gpu_undistort() {
        let Some(gpu) = crate::stitch::test_support::gpu_or_skip() else {
            return;
        };
        let lens = crate::calibration::Lens::flat(32, 16);
        let y = vec![160; 32 * 16];
        let chroma = vec![128; 16 * 8];
        let undistort = crate::lens::undistort::GpuUndistort::new(&gpu, 32, 16);
        let output = undistort.undistort(&gpu, &y, &chroma, &chroma, &lens);
        assert_eq!(output.len(), 32 * 16 * 4);
        assert!(output.chunks_exact(4).any(|p| p[0] > 100));
    }

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
