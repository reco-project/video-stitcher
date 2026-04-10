//! Object detection trait for raw camera frames.
//!
//! Detectors run on raw (pre-stitch) camera frames to find objects of interest
//! (e.g. a ball). Detections are mapped to panorama-space coordinates and fed
//! to a [`crate::director::Director`] for panning decisions.
//!
//! ## Why Raw Frames?
//!
//! The stitched panorama is an L-shaped 3D projection, not a flat image.
//! Object detection models (YOLO, etc.) work on standard 2D images, so
//! they must run on the original camera frames before stitching.
//! The slight wide-angle distortion is negligible for detection accuracy.

/// Which camera produced this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraId {
    /// Left camera (plane in X-Z space).
    Left,
    /// Right camera (plane in X-Y space).
    Right,
}

/// Raw camera frame data for detection.
///
/// Provides access to all YUV planes so detectors can use luma-only
/// (fast, sufficient for ball tracking) or full color (needed for
/// jersey classification, field segmentation, etc.).
pub struct RawFrame<'a> {
    /// Y (luma) plane, full resolution (`width x height` bytes).
    pub y: &'a [u8],
    /// Chroma plane data (format depends on the decode pipeline).
    pub chroma: ChromaFormat<'a>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// Chroma plane layout.
///
/// The format matches whatever the decode pipeline produces:
/// - Software decode (FFmpeg): YUV420P with separate U and V planes
/// - Hardware decode (NVDEC, V4L2): NV12 with interleaved UV
pub enum ChromaFormat<'a> {
    /// YUV420P: separate half-resolution U and V planes.
    Yuv420p {
        /// U (Cb) plane, `(width/2) x (height/2)` bytes.
        u: &'a [u8],
        /// V (Cr) plane, `(width/2) x (height/2)` bytes.
        v: &'a [u8],
    },
    /// NV12: interleaved UV plane, `width x (height/2)` bytes.
    Nv12 {
        /// Interleaved U,V data.
        uv: &'a [u8],
    },
}

/// A detected object in a raw camera frame.
///
/// Coordinates are in normalized image space `[0.0, 1.0]` relative to
/// the frame dimensions. Use [`crate::projection::camera_to_panorama`]
/// to map these to panoramic yaw/pitch.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Which camera this detection came from.
    pub camera: CameraId,

    /// Detection class label (e.g. "ball", "player").
    pub label: String,

    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,

    /// Bounding box center X in normalized image coordinates `[0.0, 1.0]`.
    pub center_x: f32,

    /// Bounding box center Y in normalized image coordinates `[0.0, 1.0]`.
    pub center_y: f32,

    /// Bounding box width in normalized image coordinates.
    pub width: f32,

    /// Bounding box height in normalized image coordinates.
    pub height: f32,
}

/// Trait for object detection on raw camera frames.
///
/// Implementations should be async-friendly - detection may run on a
/// separate thread or even a different device (e.g. a Jetson's DLA).
///
/// # Frame Data
///
/// The [`RawFrame`] contains the full YUV data of a single camera frame
/// (before any stitching or undistortion). Most detection models only
/// need the Y (luma) plane for grayscale inference, but full color is
/// available for tasks like jersey classification.
pub trait Detector: Send {
    /// Run detection on a raw camera frame.
    ///
    /// Returns a list of detections found in the frame. May return an
    /// empty vector if no objects are detected.
    fn detect(&mut self, camera: CameraId, frame: &RawFrame<'_>) -> Vec<Detection>;
}

/// Trait for object detection on GPU-resident NV12 frames.
///
/// Unlike [`Detector`] which operates on CPU-accessible [`RawFrame`] data,
/// this trait takes CUDA device pointers directly. Used in the zero-copy
/// pipeline where decoded frames never leave the GPU.
///
/// Implementations must handle GPU-side preprocessing (NV12-to-RGB
/// conversion, resize, normalization) and inference entirely on the GPU,
/// reading back only the small detection output.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub trait GpuDetector: Send {
    /// Run detection on a GPU-resident NV12 frame.
    ///
    /// `y_ptr` and `uv_ptr` are CUDA device pointers to the Y and
    /// interleaved UV planes of an NV12 frame. `y_pitch` and `uv_pitch`
    /// are the row strides (may differ from width due to alignment).
    fn detect_gpu(
        &mut self,
        camera: CameraId,
        y_ptr: u64,
        y_pitch: usize,
        uv_ptr: u64,
        uv_pitch: usize,
        width: u32,
        height: u32,
    ) -> Vec<Detection>;
}

/// Test whether a point lies inside a polygon using the ray-casting algorithm.
///
/// Casts a horizontal ray from the point to the right and counts how many
/// polygon edges it crosses. An odd count means the point is inside.
///
/// Both `point` and `polygon` use `[x, y]` coordinates in any consistent
/// space (typically normalized `[0,1]` camera coordinates).
///
/// Returns `false` for degenerate polygons with fewer than 3 vertices.
pub fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let (px, py) = (point[0], point[1]);
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);

        // Check if the edge from j to i crosses the horizontal ray at py.
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }

        j = i;
    }

    inside
}

/// Trait for object detection on Metal-resident NV12 frames (macOS).
///
/// Unlike [`GpuDetector`] which operates on CUDA device pointers,
/// this trait takes `CVPixelBufferRef` pointers from VideoToolbox.
/// Used in the macOS zero-copy pipeline where decoded frames are
/// IOSurface-backed and can be imported as Metal textures.
///
/// Implementations must handle Metal-side preprocessing (NV12-to-RGB
/// conversion, resize, normalization via compute shaders) and inference,
/// reading back only the small detection output.
#[cfg(target_os = "macos")]
pub trait MetalDetector: Send {
    /// Run detection on a Metal-resident NV12 frame.
    ///
    /// `cv_pixel_buffer` is a `CVPixelBufferRef` from VideoToolbox decode,
    /// backed by an IOSurface. The implementor imports it as Metal textures
    /// via `CVMetalTextureCache` for GPU-side preprocessing.
    ///
    /// `gpu` is needed for importing the CVPixelBuffer planes as Metal
    /// textures via `CVMetalTextureCache`.
    fn detect_metal(
        &mut self,
        camera: CameraId,
        cv_pixel_buffer: crate::metal_interop::CVPixelBufferRef,
        width: u32,
        height: u32,
        gpu: &crate::gpu::GpuContext,
    ) -> Vec<Detection>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- point_in_polygon tests ---

    /// Unit square: [0,0] -> [1,0] -> [1,1] -> [0,1].
    fn unit_square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    #[test]
    fn pip_center_of_square() {
        assert!(point_in_polygon([0.5, 0.5], &unit_square()));
    }

    #[test]
    fn pip_outside_square() {
        assert!(!point_in_polygon([1.5, 0.5], &unit_square()));
        assert!(!point_in_polygon([-0.1, 0.5], &unit_square()));
        assert!(!point_in_polygon([0.5, -0.1], &unit_square()));
        assert!(!point_in_polygon([0.5, 1.1], &unit_square()));
    }

    #[test]
    fn pip_triangle() {
        let triangle = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        // Inside
        assert!(point_in_polygon([0.5, 0.3], &triangle));
        // Outside (right of the triangle)
        assert!(!point_in_polygon([0.9, 0.8], &triangle));
    }

    #[test]
    fn pip_concave_l_shape() {
        // L-shaped polygon (concave):
        //   (0,0) -> (1,0) -> (1,0.5) -> (0.5,0.5) -> (0.5,1) -> (0,1)
        let l_shape = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
            [0.0, 1.0],
        ];
        // Inside the bottom-right arm
        assert!(point_in_polygon([0.75, 0.25], &l_shape));
        // Inside the top-left arm
        assert!(point_in_polygon([0.25, 0.75], &l_shape));
        // In the concave cutout (top-right) - should be outside
        assert!(!point_in_polygon([0.75, 0.75], &l_shape));
    }

    #[test]
    fn pip_degenerate_polygon() {
        // Fewer than 3 vertices: always false.
        assert!(!point_in_polygon([0.5, 0.5], &[]));
        assert!(!point_in_polygon([0.5, 0.5], &[[0.0, 0.0]]));
        assert!(!point_in_polygon([0.5, 0.5], &[[0.0, 0.0], [1.0, 1.0]]));
    }

    #[test]
    fn pip_near_edge_of_square() {
        // Just inside the edge
        assert!(point_in_polygon([0.001, 0.5], &unit_square()));
        assert!(point_in_polygon([0.999, 0.5], &unit_square()));
    }
}
