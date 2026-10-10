//! Feature detection and descriptor matching.
//!
//! Uses a vendored AKAZE (Accelerated-KAZE) with bug fixes for
//! detection and M-LDB binary descriptors, plus brute-force Hamming
//! distance matching with Lowe's ratio test and cross-check.
//!
//! The detector backend is modular: the vendored AKAZE is the default
//! (zero system dependencies). Alternative backends (e.g. OpenCV) can
//! be added as separate implementations behind the same interface.

use crate::akaze;

/// Descriptor size in bytes (512 bits = 64 bytes, AKAZE M-LDB).
const DESC_BYTES: usize = akaze::DESC_BYTES;

/// A detected feature keypoint with pixel position and response strength.
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    /// X position in pixels, with the first pixel center at zero.
    pub x: f32,
    /// Y position in pixels, with the first pixel center at zero.
    pub y: f32,
    /// Corner response strength.
    pub response: f32,
}

impl KeyPoint {
    /// Texture coordinates relative to image edges, not pixel indices.
    pub(crate) fn normalized(&self, width: u32, height: u32) -> [f64; 2] {
        [
            (f64::from(self.x) + 0.5) / f64::from(width),
            (f64::from(self.y) + 0.5) / f64::from(height),
        ]
    }
}

/// A binary descriptor (512 bits, AKAZE M-LDB).
pub type Descriptor = [u8; DESC_BYTES];

/// A raw descriptor match between two keypoint indices.
#[derive(Debug, Clone, Copy)]
pub struct RawMatch {
    /// Index into the left keypoint/descriptor list.
    pub left_idx: usize,
    /// Index into the right keypoint/descriptor list.
    pub right_idx: usize,
    /// Hamming distance between descriptors.
    pub distance: u32,
}

/// Number of u64 words in a descriptor (64 bytes / 8 = 8 words).
const DESC_WORDS: usize = DESC_BYTES / 8;

/// Compute Hamming distance between two binary descriptors.
///
/// Processes 8 bytes at a time as u64 words, reducing iteration count
/// 8x versus byte-by-byte. This is the hot inner loop of brute-force
/// matching.
fn hamming_distance(a: &Descriptor, b: &Descriptor) -> u32 {
    let mut dist = 0u32;
    for i in 0..DESC_WORDS {
        let off = i * 8;
        let wa = u64::from_ne_bytes(a[off..off + 8].try_into().unwrap());
        let wb = u64::from_ne_bytes(b[off..off + 8].try_into().unwrap());
        dist += (wa ^ wb).count_ones();
    }
    dist
}

/// Region of interest for feature detection (fractions of image dimensions, 0.0 - 1.0).
#[derive(Debug, Clone, Copy)]
pub struct DetectRegion {
    /// Left edge of the ROI (fraction of width).
    pub x_min: f32,
    /// Right edge of the ROI (fraction of width).
    pub x_max: f32,
    /// Top edge of the ROI (fraction of height).
    pub y_min: f32,
    /// Bottom edge of the ROI (fraction of height).
    pub y_max: f32,
}

/// Maximum width for AKAZE detection. Images wider than this are
/// downscaled before feature detection (keypoints are mapped back to
/// original coordinates). 1920px provides full-quality features while
/// still being faster than raw 4K/5K input.
const DETECT_MAX_WIDTH: u32 = 1920;

/// Detect features and compute descriptors using AKAZE.
///
/// Convenience wrapper that calls [`detect_with_border`] with the default
/// 30px border margin.
pub fn detect(
    rgba: &[u8],
    width: u32,
    height: u32,
    region: Option<DetectRegion>,
    max_keypoints: usize,
    threshold: f64,
) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    detect_with_border(rgba, width, height, region, max_keypoints, threshold, 30)
}

/// Extra pixel margin around the detection region crop.
///
/// Ensures AKAZE has enough context around the border for descriptor
/// computation. Without this margin, keypoints near the crop edge would
/// have incomplete descriptors. 32px is enough for AKAZE's largest
/// descriptor sampling radius.
const CROP_MARGIN_PX: u32 = 32;

/// Detect features and compute descriptors using AKAZE with configurable border filter.
///
/// Accepts RGBA pixel data (from GPU undistortion). When a detection
/// region is specified, the image is cropped to that region (plus a
/// margin for descriptor context) *before* running AKAZE. This avoids
/// building the full scale space on pixels that will be discarded.
/// Keypoint coordinates are mapped back
/// to the original image space.
///
/// Images wider than 1920px are downscaled before detection for
/// performance; keypoint coordinates are mapped back to the original
/// resolution.
///
/// `border_margin`: pixel distance from black edges to reject. Set to 0 to disable.
///
/// # Panics
///
/// Panics if `rgba.len() < (width * height * 4)`.
pub fn detect_with_border(
    rgba: &[u8],
    width: u32,
    height: u32,
    region: Option<DetectRegion>,
    max_keypoints: usize,
    threshold: f64,
    border_margin: i32,
) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .expect("RGBA dimensions overflow");
    assert!(
        rgba.len() >= expected,
        "RGBA buffer too small: {} < {} ({}x{}x4)",
        rgba.len(),
        expected,
        width,
        height,
    );

    if width == 0 || height == 0 || max_keypoints == 0 {
        return (Vec::new(), Vec::new());
    }
    let (crop_x0, crop_y0, detect_w, detect_h) = if let Some(r) = region {
        if ![r.x_min, r.x_max, r.y_min, r.y_max]
            .iter()
            .all(|v| (0.0..=1.0).contains(v))
            || r.x_min >= r.x_max
            || r.y_min >= r.y_max
        {
            return (Vec::new(), Vec::new());
        }
        let x_lo = (r.x_min * width as f32) as u32;
        let x_hi = ((r.x_max * width as f32).ceil() as u32).min(width);
        let y_lo = (r.y_min * height as f32) as u32;
        let y_hi = ((r.y_max * height as f32).ceil() as u32).min(height);

        // Expand by margin, clamping to image bounds
        let cx0 = x_lo.saturating_sub(CROP_MARGIN_PX);
        let cy0 = y_lo.saturating_sub(CROP_MARGIN_PX);
        let cx1 = x_hi.saturating_add(CROP_MARGIN_PX).min(width);
        let cy1 = y_hi.saturating_add(CROP_MARGIN_PX).min(height);

        let cw = cx1 - cx0;
        let ch = cy1 - cy0;

        // Only crop if it actually reduces the image size meaningfully
        if (cw as u64 * ch as u64) < (width as u64 * height as u64 * 3 / 4) {
            log::debug!(
                "cropped {}x{} -> {}x{} for AKAZE (region + {}px margin)",
                width,
                height,
                cw,
                ch,
                CROP_MARGIN_PX,
            );
            (cx0, cy0, cw, ch)
        } else {
            (0, 0, width, height)
        }
    } else {
        (0, 0, width, height)
    };

    // Copy only the selected RGB pixels, without an intermediate RGBA image.
    let rgb_data: Vec<u8> = rgba[..expected]
        .chunks_exact(width as usize * 4)
        .skip(crop_y0 as usize)
        .take(detect_h as usize)
        .flat_map(|row| {
            let start = crop_x0 as usize * 4;
            row[start..start + detect_w as usize * 4].chunks_exact(4)
        })
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let img = image::RgbImage::from_raw(detect_w, detect_h, rgb_data)
        .expect("selected RGB rows match the crop dimensions");
    let dynamic = image::DynamicImage::ImageRgb8(img);

    // Downscale if needed for performance
    let detect_img = if detect_w > DETECT_MAX_WIDTH {
        let s = DETECT_MAX_WIDTH as f32 / detect_w as f32;
        let new_w = DETECT_MAX_WIDTH;
        let new_h = ((detect_h as f32 * s) as u32).max(1);
        let resized = dynamic.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        log::debug!(
            "downscaled {}x{} -> {}x{} for AKAZE",
            detect_w,
            detect_h,
            new_w,
            new_h
        );
        resized
    } else {
        dynamic
    };
    let scale = [
        detect_w as f32 / detect_img.width() as f32,
        detect_h as f32 / detect_img.height() as f32,
    ];

    let detector = akaze::Akaze::new(threshold);
    let (akaze_kps, akaze_descs) = detector.extract(&detect_img);

    let total_detected = akaze_kps.len();

    // Pair up keypoints with descriptors, convert types.
    // Map keypoint coords back to original resolution (undo downscale + crop offset).
    let mut pairs: Vec<(KeyPoint, Descriptor)> = akaze_kps
        .iter()
        .zip(akaze_descs.iter())
        .map(|(kp, d)| {
            let [x, y] = restore_keypoint([kp.point.0, kp.point.1], scale, [crop_x0, crop_y0]);
            (
                KeyPoint {
                    x,
                    y,
                    response: kp.response,
                },
                *d,
            )
        })
        .collect();

    // Filter to ROI if specified (some keypoints in the margin may fall outside)
    if let Some(r) = region {
        pairs.retain(|(kp, _)| {
            let [x, y] = kp.normalized(width, height);
            x >= f64::from(r.x_min)
                && x <= f64::from(r.x_max)
                && y >= f64::from(r.y_min)
                && y <= f64::from(r.y_max)
        });
    }

    // Apply border filter if margin > 0 and image is large enough
    border_filter_pairs(&mut pairs, rgba, width, height, border_margin);

    // Sort by response (strongest first) and cap
    pairs.sort_by(|a, b| {
        b.0.response
            .partial_cmp(&a.0.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(max_keypoints);

    let keypoints: Vec<KeyPoint> = pairs.iter().map(|(kp, _)| *kp).collect();
    let descriptors: Vec<Descriptor> = pairs.iter().map(|(_, d)| *d).collect();

    log::trace!(
        "AKAZE: {} detected, {} in ROI (cap {}), {}x{} (cropped to {}x{})",
        total_detected,
        keypoints.len(),
        max_keypoints,
        width,
        height,
        detect_w,
        detect_h,
    );

    (keypoints, descriptors)
}

fn restore_keypoint(point: [f32; 2], scale: [f32; 2], origin: [u32; 2]) -> [f32; 2] {
    // image::resize_exact aligns pixel centers. Rounded output height can make
    // the vertical scale differ from the horizontal scale.
    std::array::from_fn(|i| (point[i] + 0.5) * scale[i] - 0.5 + origin[i] as f32)
}

/// Reject keypoints near the undistortion border (black pincushion edges).
///
/// Samples 8 pixels at `margin` distance around each keypoint. If any
/// sampled pixel is black (RGB < 10), the keypoint is too close to the
/// undistortion boundary and is removed.
///
/// Only applies to images wider than 1000px to avoid rejecting features
/// in small synthetic test images.
///
/// This is the key filter for wide-FOV cameras (DJI Action 4, XTU Max3)
/// where the pincushion boundary produces unreliable features that
/// corrupt calibration.
pub fn border_filter_pairs(
    pairs: &mut Vec<(KeyPoint, Descriptor)>,
    rgba: &[u8],
    width: u32,
    height: u32,
    margin: i32,
) {
    if margin <= 0 || width < 1000 {
        return;
    }

    let buf_len = rgba.len();
    let black_thresh: u8 = 10;
    let pre = pairs.len();
    pairs.retain(|(kp, _)| {
        let cx = kp.x as i32;
        let cy = kp.y as i32;
        let w = width as i32;
        let h = height as i32;
        for &(dx, dy) in &[
            (-margin, 0),
            (margin, 0),
            (0, -margin),
            (0, margin),
            (-margin, -margin),
            (margin, -margin),
            (-margin, margin),
            (margin, margin),
        ] {
            let sx = cx + dx;
            let sy = cy + dy;
            if sx < 0 || sx >= w || sy < 0 || sy >= h {
                return false;
            }
            let idx = (sy as usize * width as usize + sx as usize) * 4;
            if idx + 2 >= buf_len {
                return false;
            }
            if rgba[idx] < black_thresh
                && rgba[idx + 1] < black_thresh
                && rgba[idx + 2] < black_thresh
            {
                return false;
            }
        }
        true
    });
    if pre != pairs.len() {
        log::debug!(
            "border filter: {} -> {} keypoints ({} near undistortion edge)",
            pre,
            pairs.len(),
            pre - pairs.len()
        );
    }
}

/// Find the best match for each descriptor in `query` against `train`,
/// optionally applying Lowe's ratio test.
///
/// When `ratio >= 1.0`, the ratio test is skipped and only the best
/// match is returned (cross-check in the caller provides filtering).
/// When `ratio < 1.0`, the best match must be significantly better
/// than the second-best (lower = stricter).
fn find_matches_one_way(
    query: &[Descriptor],
    train: &[Descriptor],
    ratio: f64,
) -> Vec<(usize, usize, u32)> {
    let use_ratio_test = ratio < 1.0;
    let mut matches = Vec::new();

    for (q_idx, desc_q) in query.iter().enumerate() {
        let mut best_dist = u32::MAX;
        let mut second_dist = u32::MAX;
        let mut best_idx = 0;

        for (t_idx, desc_t) in train.iter().enumerate() {
            let dist = hamming_distance(desc_q, desc_t);
            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = t_idx;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }

        if use_ratio_test {
            // Lowe's ratio test: best must be significantly better than second
            if second_dist > 0 && (best_dist as f64) < ratio * (second_dist as f64) {
                matches.push((q_idx, best_idx, best_dist));
            }
        } else {
            // Cross-check only mode: accept all best matches
            if best_dist < u32::MAX {
                matches.push((q_idx, best_idx, best_dist));
            }
        }
    }

    matches
}

/// Match two descriptor sets using brute-force Hamming distance with
/// Lowe's ratio test and cross-check verification.
///
/// Cross-check requires that a match is the best in both directions:
/// if L\[i\]'s best match is R\[j\], then R\[j\]'s best match must also be L\[i\].
/// This eliminates many false positives where repetitive textures (field
/// markings, clouds) produce plausible one-way matches.
///
/// Returns matches sorted by distance (best first).
pub fn match_descriptors(left: &[Descriptor], right: &[Descriptor], ratio: f64) -> Vec<RawMatch> {
    let forward = find_matches_one_way(left, right, ratio);
    let backward = find_matches_one_way(right, left, ratio);

    // Build reverse lookup: right_idx -> best left_idx from backward pass
    let mut right_to_left: Vec<Option<usize>> = vec![None; right.len()];
    for &(r_idx, l_idx, _) in &backward {
        right_to_left[r_idx] = Some(l_idx);
    }

    // Keep only matches where forward and backward agree
    let mut matches: Vec<RawMatch> = forward
        .into_iter()
        .filter(|&(l_idx, r_idx, _)| right_to_left[r_idx] == Some(l_idx))
        .map(|(l_idx, r_idx, dist)| RawMatch {
            left_idx: l_idx,
            right_idx: r_idx,
            distance: dist,
        })
        .collect();

    matches.sort_by_key(|m| m.distance);
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restoring_cropped_resized_keypoints_preserves_pixel_centers() {
        let original = [3001.0, 901.0];
        let resized = [1920.0, 576.0];
        let origin = [47, 83];
        for point in [[0.0, 0.0], [713.25, 222.75], [1919.0, 575.0]] {
            let restored = restore_keypoint(
                point,
                [original[0] / resized[0], original[1] / resized[1]],
                origin,
            );
            for i in 0..2 {
                let expected = (point[i] + 0.5) / resized[i];
                let actual = (restored[i] - origin[i] as f32 + 0.5) / original[i];
                assert!((expected - actual).abs() < 1e-6);
            }
        }
        assert_eq!(
            restore_keypoint([3.25, 4.75], [1.0; 2], [0; 2]),
            [3.25, 4.75]
        );
    }

    #[test]
    fn invalid_regions_return_no_features() {
        let image = vec![128; 64 * 48 * 4];
        for (min, max) in [(f32::NAN, 1.0), (0.8, 0.2), (1.1, 2.0), (0.3, 0.3)] {
            let (points, descriptors) = detect(
                &image,
                64,
                48,
                Some(DetectRegion {
                    x_min: min,
                    x_max: max,
                    y_min: 0.0,
                    y_max: 1.0,
                }),
                100,
                0.001,
            );
            assert!(points.is_empty() && descriptors.is_empty());
        }
    }

    #[test]
    fn hamming_distance_identical_is_zero() {
        let a = [0xABu8; DESC_BYTES];
        let b = [0xABu8; DESC_BYTES];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn hamming_distance_one_bit() {
        let a = [0u8; DESC_BYTES];
        let mut b = [0u8; DESC_BYTES];
        b[0] = 1; // one bit different
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    /// Create an RGBA image from a grayscale pattern.
    fn gray_to_rgba(gray: &[u8], _w: u32, _h: u32) -> Vec<u8> {
        gray.iter().flat_map(|&g| [g, g, g, 255]).collect()
    }

    #[test]
    fn detect_on_blank_image_returns_few_keypoints() {
        let rgba = gray_to_rgba(&vec![128; 200 * 200], 200, 200);
        let (kps, descs) = detect(&rgba, 200, 200, None, 2000, 0.001);
        assert_eq!(kps.len(), descs.len());
    }

    #[test]
    fn detect_on_gradient_image() {
        let w = 200u32;
        let h = 200u32;
        let mut gray = vec![0u8; (w * h) as usize];
        for y in 50..150 {
            for x in 50..150 {
                gray[(y * w + x) as usize] = 255;
            }
        }
        let rgba = gray_to_rgba(&gray, w, h);
        let (kps, descs) = detect(&rgba, w, h, None, 2000, 0.001);
        assert_eq!(kps.len(), descs.len());
        assert!(!kps.is_empty(), "should detect features in rectangle image");
    }

    #[test]
    fn match_descriptors_ratio_test() {
        let d0 = [0u8; DESC_BYTES]; // all zeros
        let mut d_close = [0u8; DESC_BYTES];
        d_close[0] = 0b00000011; // 2 bits different
        let d_far = [0xFF; DESC_BYTES]; // very different

        let left = vec![d0];
        let right = vec![d_close, d_far];

        let matches = match_descriptors(&left, &right, 0.7);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].right_idx, 0); // matched to d_close
    }
}
