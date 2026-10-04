//! Integer raster placement shared by preprocessing and coordinate recovery.

use reco_core::detect::detector::DetectorError;

pub(crate) struct Letterbox {
    pub scale: f32,
    pub width: u32,
    pub height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
}

impl Letterbox {
    pub fn new((width, height): (u32, u32), input: u32) -> Result<Self, DetectorError> {
        if width == 0 || height == 0 || input == 0 {
            return Err(DetectorError::InferenceFailed(
                "letterbox dimensions must be nonzero".into(),
            ));
        }
        let scale = (input as f32 / width as f32).min(input as f32 / height as f32);
        let width = (width as f32 * scale).round() as u32;
        let height = (height as f32 * scale).round() as u32;
        if width == 0 || height == 0 || width > input || height > input {
            return Err(DetectorError::InferenceFailed(
                "source aspect cannot fit the model raster".into(),
            ));
        }
        Ok(Self {
            scale,
            width,
            height,
            pad_x: (input - width) / 2,
            pad_y: (input - height) / 2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_padding_places_the_extra_pixel_after_the_content() {
        for (source, content, padding) in [
            ((6, 4), (8, 5), (0, 1)),
            ((4, 6), (5, 8), (1, 0)),
            ((8, 8), (8, 8), (0, 0)),
        ] {
            let layout = Letterbox::new(source, 8).unwrap();
            assert_eq!((layout.width, layout.height), content);
            assert_eq!((layout.pad_x, layout.pad_y), padding);
        }
    }

    #[test]
    fn empty_or_unrepresentable_content_is_rejected() {
        for (source, input) in [((0, 4), 8), ((4, 0), 8), ((4, 4), 0), ((2, 100_000), 8)] {
            assert!(Letterbox::new(source, input).is_err());
        }
    }
}
