//! Metal's constant-buffer layout and integer letterbox placement.

use crate::letterbox::Letterbox;
use reco_core::detect::detector::DetectorError;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct PreprocessParams {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_size: u32,
    pub _pad: u32,
    pub scale: f32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub new_w: u32,
    pub new_h: u32,
    pub is_full_range: u32,
    pub _pad2: [u32; 2],
}

impl PreprocessParams {
    pub fn new(size: (u32, u32), input: u32, full_range: bool) -> Result<Self, DetectorError> {
        let layout = Letterbox::new(size, input)?;
        Ok(Self {
            src_w: size.0,
            src_h: size.1,
            dst_size: input,
            _pad: 0,
            scale: layout.scale,
            pad_x: layout.pad_x,
            pad_y: layout.pad_y,
            new_w: layout.width,
            new_h: layout.height,
            is_full_range: u32::from(full_range),
            _pad2: [0; 2],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternating_camera_shapes_use_the_same_placement_as_coordinate_recovery() {
        assert_eq!(std::mem::size_of::<PreprocessParams>(), 48);
        assert_eq!(std::mem::offset_of!(PreprocessParams, scale), 16);
        assert_eq!(std::mem::offset_of!(PreprocessParams, is_full_range), 36);
        for size in [(6, 4), (4, 6), (8, 8), (1920, 1080), (1280, 720), (6, 4)] {
            let params = PreprocessParams::new(size, 8, true).unwrap();
            let layout = Letterbox::new(size, 8).unwrap();
            assert_eq!((params.src_w, params.src_h), size);
            assert_eq!((params.new_w, params.new_h), (layout.width, layout.height));
            assert_eq!((params.pad_x, params.pad_y), (layout.pad_x, layout.pad_y));
            assert_eq!(params.scale, layout.scale);
            let point = (size.0 as f32 * 0.3, size.1 as f32 * 0.7);
            let model = (
                point.0 * params.scale + params.pad_x as f32,
                point.1 * params.scale + params.pad_y as f32,
            );
            assert!(((model.0 - layout.pad_x as f32) / layout.scale - point.0).abs() < 0.001);
            assert!(((model.1 - layout.pad_y as f32) / layout.scale - point.1).abs() < 0.001);
        }
        let landscape = PreprocessParams::new((6, 4), 8, false).unwrap();
        assert_eq!((landscape.pad_y, landscape.new_h), (1, 5));
        for size in [(0, 4), (4, 0), (2, 100_000)] {
            assert!(PreprocessParams::new(size, 8, false).is_err());
        }
        assert!(PreprocessParams::new((4, 4), 0, false).is_err());
    }
}
