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
}

/// Shader and draw configuration supplied by a projection.
#[cfg(feature = "gpu")]
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
    fn shared_sampling_shader_runs_in_both_lens_consumers() {
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
        let preview = crate::lens::preview::LensPreviewRenderer::new(
            &gpu,
            32,
            16,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let texture = preview.render_yuv(
            &gpu,
            &super::planes::Yuv420pPlanes {
                y: &y,
                u: &chroma,
                v: &chroma,
            },
            &lens,
            false,
        );
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        assert_eq!((texture.width(), texture.height()), (32, 16));
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
