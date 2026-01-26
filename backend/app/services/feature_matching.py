"""
Feature matching service for calibration.

Detects and matches features between left and right camera images
to generate corresponding point pairs for camera position optimization.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


def match_features(
    img_left: np.ndarray, img_right: np.ndarray, method: str = "auto", min_matches: int = 8
) -> Dict[str, Any]:
    """
    Match features between left and right images.

    Args:
        img_left: Left camera image (BGR format)
        img_right: Right camera image (BGR format)
        method: Feature detection method ("auto", "sift", "orb")
        min_matches: Minimum number of matches required

    Returns:
        Dictionary containing:
        - left_points: List of [x, y] coordinates in left image
        - right_points: List of [x, y] coordinates in right image
        - num_matches: Number of matched points
        - confidence: Confidence score (0-1)

    Raises:
        ValueError: If images are invalid or insufficient matches found
    """
    if img_left is None or img_right is None:
        raise ValueError("Invalid input images")

    if img_left.shape[:2] != img_right.shape[:2]:
        raise ValueError("Images must have the same dimensions")

    # Resize for processing (target width 1920)
    h, w = img_left.shape[:2]
    target_w = 1920
    scale = target_w / w

    img_left_resized = _resize_keep_ratio(img_left, target_w)
    img_right_resized = _resize_keep_ratio(img_right, target_w)

    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left_resized, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right_resized, cv2.COLOR_BGR2GRAY)

    # Check if images are valid (not blank/black) - relaxed threshold
    left_std = gray_left.std()
    right_std = gray_right.std()

    if left_std < 0.1 or right_std < 0.1:
        raise ValueError(f"Images appear to be blank (std: left={left_std:.2f}, right={right_std:.2f})")

    # Detect and match features
    kp1, des1, kp2, des2, norm_type = _detect_features(gray_left, gray_right, method)

    if des1 is None or des2 is None:
        raise ValueError(
            f"Failed to detect features: des1={'None' if des1 is None else 'OK'}, des2={'None' if des2 is None else 'OK'}"
        )

    if len(kp1) == 0 or len(kp2) == 0:
        raise ValueError(f"No keypoints found: kp1={len(kp1)}, kp2={len(kp2)}")

    # Match features using BFMatcher
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_matches:
        raise ValueError(f"Insufficient matches found: {len(good_matches)} < {min_matches}")

    # Sort by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Filter to overlapping regions
    h_left, w_left = gray_left.shape[:2]
    h_right, w_right = gray_right.shape[:2]
    left_threshold = 0.4 * w_left
    right_threshold = 0.4 * w_right

    filtered_pts1 = []
    filtered_pts2 = []
    filtered_matches = []

    for m in good_matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt

        # Filter to overlapping regions (ignore top/bottom 20%)
        if (
            left_threshold <= pt1[0]
            and pt2[0] <= right_threshold
            and 0.2 * h_left <= pt1[1] <= 0.8 * h_left
            and 0.2 * h_right <= pt2[1] <= 0.8 * h_right
        ):
            filtered_pts1.append(pt1)
            filtered_pts2.append(pt2)
            filtered_matches.append(m)

    # Convert to numpy arrays
    pts1 = np.float32(filtered_pts1).reshape(-1, 2)
    pts2 = np.float32(filtered_pts2).reshape(-1, 2)

    # Fallback if not enough filtered matches
    if len(filtered_matches) < min_matches:
        filtered_matches = good_matches[:30]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 2)

    # Use RANSAC to filter outliers
    if len(pts1) >= 8:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.995)
        if mask is not None:
            mask = mask.ravel().astype(bool)
            pts1 = pts1[mask]
            pts2 = pts2[mask]
            filtered_matches = [filtered_matches[i] for i, m in enumerate(mask) if m]

    if len(pts1) < min_matches:
        raise ValueError(f"Insufficient matches after RANSAC: {len(pts1)} < {min_matches}")

    # Normalize points to plane coordinates (width = 1.0)
    img_h, img_w = img_left_resized.shape[:2]
    plane_w = 1.0
    plane_h = plane_w * (img_h / img_w)

    pts1_norm = _normalize_to_plane_coords(pts1, img_w, img_h, plane_w, plane_h)
    pts2_norm = _normalize_to_plane_coords(pts2, img_w, img_h, plane_w, plane_h)

    # Calculate confidence based on number of matches and match quality
    confidence = min(1.0, len(filtered_matches) / 50.0)

    return {
        "left_points": pts1_norm.tolist(),
        "right_points": pts2_norm.tolist(),
        "num_matches": len(filtered_matches),
        "confidence": round(confidence, 3),
    }


def _resize_keep_ratio(img: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image to target width while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def _detect_features(gray_left: np.ndarray, gray_right: np.ndarray, method: str) -> Tuple[Any, Any, Any, Any, int]:
    """
    Detect features using SIFT or ORB.

    Returns:
        kp1, des1, kp2, des2, norm_type
    """
    if method == "sift" or (method == "auto" and hasattr(cv2, 'SIFT_create')):
        sift = cv2.SIFT_create(nfeatures=2000)
        kp1, des1 = sift.detectAndCompute(gray_left, None)
        kp2, des2 = sift.detectAndCompute(gray_right, None)
        norm_type = cv2.NORM_L2
    else:
        # Fallback to ORB
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        kp1, des1 = orb.detectAndCompute(gray_left, None)
        kp2, des2 = orb.detectAndCompute(gray_right, None)
        norm_type = cv2.NORM_HAMMING

    return kp1, des1, kp2, des2, norm_type


def _normalize_to_plane_coords(pts: np.ndarray, img_w: int, img_h: int, plane_w: float, plane_h: float) -> np.ndarray:
    """
    Normalize image coordinates to plane coordinates.

    Converts from image space [0, img_w] x [0, img_h]
    to plane space [-plane_w/2, plane_w/2] x [-plane_h/2, plane_h/2]
    """
    # Normalize to [0, 1]
    x_norm = pts[:, 0] / img_w
    y_norm = pts[:, 1] / img_h

    # Map to plane coordinates centered at origin
    x_plane = (x_norm - 0.5) * plane_w
    y_plane = (y_norm - 0.5) * plane_h

    return np.stack([x_plane, y_plane], axis=1)


def _grass_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Create a mask for grass/green areas in the image.
    Uses HSV color space to identify green-ish pixels, avoiding very dark/bright areas.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # Green-ish pixels (broad range) + avoid very dark/very bright
    mask = (H >= 30) & (H <= 95) & (S >= 35) & (V >= 25) & (V <= 245)
    mask = mask.astype(np.uint8) * 255
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _sky_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Create a mask for sky/blue areas in the image.
    Uses HSV color space to identify blue-ish pixels (sky), avoiding very dark areas.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # Blue sky pixels: hue around 100-130 (blue range), moderate saturation, bright
    mask = (H >= 95) & (H <= 135) & (S >= 20) & (S <= 180) & (V >= 100) & (V <= 255)
    mask = mask.astype(np.uint8) * 255
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _combined_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Create a combined mask for color-rich areas (grass + sky).
    Falls back to excluding very dark/bright pixels if not enough grass/sky found.
    """
    grass = _grass_mask(bgr)
    sky = _sky_mask(bgr)
    combined = cv2.bitwise_or(grass, sky)

    # If combined mask is too small, use all pixels except very dark/bright
    if combined.sum() < 3000 * 255:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        # Exclude very dark and very bright pixels
        combined = ((V >= 20) & (V <= 240)).astype(np.uint8) * 255

    return combined


def _lab_mean_std(bgr: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Compute mean and std of LAB color space for masked region.
    Falls back to full image if mask is too small.

    Uses OpenCV LAB range: L: 0-255, a: 0-255, b: 0-255 (128 is neutral for a/b)
    This matches the user's working Python script exactly.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask.astype(bool)
    if m.sum() < 3000:  # Fallback if mask is too small
        m = np.ones(mask.shape, dtype=bool)
    pixels = lab[m]
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0) + 1e-6  # Avoid division by zero
    return mean, std


def compute_color_correction(img_left: np.ndarray, img_right: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Compute color correction parameters using Reinhard color transfer in LAB space.

    This method analyzes the overlapping regions between cameras and computes
    LAB color space statistics to enable Reinhard-style color transfer.
    The right image will be transformed to match the left image's colors.

    Args:
        img_left: Left camera image (BGR format)
        img_right: Right camera image (BGR format)

    Returns:
        Dictionary with 'left' and 'right' color correction params including
        LAB mean/std for Reinhard transfer
    """
    if img_left is None or img_right is None:
        return _default_color_correction()

    try:
        h, w = img_left.shape[:2]

        # Extract overlapping regions (right 30% of left, left 30% of right)
        overlap_pct = 0.30
        overlap_left = img_left[:, int(w * (1 - overlap_pct)) :, :]
        overlap_right = img_right[:, : int(w * overlap_pct), :]

        # Ensure same size
        min_w = min(overlap_left.shape[1], overlap_right.shape[1])
        overlap_left = overlap_left[:, :min_w, :]
        overlap_right = overlap_right[:, :min_w, :]

        # Create combined masks (grass + sky) for balanced color sampling
        mask_left = _combined_mask(overlap_left)
        mask_right = _combined_mask(overlap_right)

        # Compute LAB statistics from overlap regions
        # Target: left image colors (right will match left)
        tgt_mean, tgt_std = _lab_mean_std(overlap_left, mask_left)
        src_mean, src_std = _lab_mean_std(overlap_right, mask_right)

        # Reinhard transfer parameters for right image:
        # transformed = (pixel - src_mean) / src_std * tgt_std + tgt_mean
        # This can be rewritten as: transformed = pixel * scale + offset
        # where scale = tgt_std / src_std and offset = tgt_mean - src_mean * scale
        scale = tgt_std / src_std
        offset = tgt_mean - src_mean * scale

        return {
            'left': {
                # Left stays unchanged (neutral)
                'brightness': 0.0,
                'contrast': 1.0,
                'saturation': 1.0,
                'colorBalance': [1.0, 1.0, 1.0],
                'temperature': 0.0,
                # LAB Reinhard params (identity transform)
                'labScale': [1.0, 1.0, 1.0],
                'labOffset': [0.0, 0.0, 0.0],
            },
            'right': {
                # Keep legacy params for backward compatibility
                'brightness': 0.0,
                'contrast': 1.0,
                'saturation': 1.0,
                'colorBalance': [1.0, 1.0, 1.0],
                'temperature': 0.0,
                # LAB Reinhard params for color transfer
                'labScale': [round(float(s), 6) for s in scale],
                'labOffset': [round(float(o), 6) for o in offset],
            },
        }

    except Exception as e:
        print(f"Color correction computation failed: {e}")
        return _default_color_correction()


def _default_color_correction() -> Dict[str, Dict[str, Any]]:
    """Return default (neutral) color correction params."""
    default = {
        'brightness': 0,
        'contrast': 1,
        'saturation': 1,
        'colorBalance': [1, 1, 1],
        'temperature': 0,
        'labScale': [1.0, 1.0, 1.0],
        'labOffset': [0.0, 0.0, 0.0],
    }
    return {'left': default.copy(), 'right': default.copy()}
