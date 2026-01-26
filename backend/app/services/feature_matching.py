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


def compute_color_correction(img_left: np.ndarray, img_right: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Compute color correction parameters to match colors between two cameras.

    Uses histogram analysis in HSV space to better match colors like grass.
    Focuses on the overlapping regions and computes per-channel corrections.

    Args:
        img_left: Left camera image (BGR format)
        img_right: Right camera image (BGR format)

    Returns:
        Dictionary with 'left' and 'right' color correction params
    """
    if img_left is None or img_right is None:
        return _default_color_correction()

    try:
        h, w = img_left.shape[:2]

        # Extract overlapping regions (right 30% of left, left 30% of right)
        # Smaller overlap for more accurate matching
        overlap_left = img_left[:, int(w * 0.7) :, :]
        overlap_right = img_right[:, : int(w * 0.3), :]

        # Ensure same size
        min_w = min(overlap_left.shape[1], overlap_right.shape[1])
        overlap_left = overlap_left[:, :min_w, :]
        overlap_right = overlap_right[:, :min_w, :]

        # Convert to HSV for better color analysis
        hsv_left = cv2.cvtColor(overlap_left, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_right = cv2.cvtColor(overlap_right, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Also work in RGB for color balance
        rgb_left = cv2.cvtColor(overlap_left, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_right = cv2.cvtColor(overlap_right, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Compute per-channel statistics using percentiles (more robust than mean)
        def channel_stats(img):
            """Get median and IQR for each channel."""
            medians = np.median(img, axis=(0, 1))
            p25 = np.percentile(img, 25, axis=(0, 1))
            p75 = np.percentile(img, 75, axis=(0, 1))
            return medians, p75 - p25  # median, interquartile range

        # HSV statistics
        hsv_med_left, hsv_iqr_left = channel_stats(hsv_left)
        hsv_med_right, hsv_iqr_right = channel_stats(hsv_right)

        # RGB statistics for color balance
        rgb_med_left, _ = channel_stats(rgb_left)
        rgb_med_right, _ = channel_stats(rgb_right)

        # Target: midpoint between both cameras
        hsv_med_target = (hsv_med_left + hsv_med_right) / 2
        rgb_med_target = (rgb_med_left + rgb_med_right) / 2

        # Brightness correction from V channel (0-255 range)
        # Normalize to our -0.5 to 0.5 range
        v_diff_left = (hsv_med_target[2] - hsv_med_left[2]) / 255.0
        v_diff_right = (hsv_med_target[2] - hsv_med_right[2]) / 255.0
        brightness_left = float(np.clip(v_diff_left, -0.3, 0.3))
        brightness_right = float(np.clip(v_diff_right, -0.3, 0.3))

        # Saturation from S channel (0-255 range)
        # Convert to multiplier (1.0 = no change)
        s_left = max(hsv_med_left[1], 1.0)
        s_right = max(hsv_med_right[1], 1.0)
        s_target = max(hsv_med_target[1], 1.0)
        saturation_left = float(np.clip(s_target / s_left, 0.7, 1.4))
        saturation_right = float(np.clip(s_target / s_right, 0.7, 1.4))

        # Contrast from V channel spread (IQR)
        v_iqr_left = max(hsv_iqr_left[2], 1.0)
        v_iqr_right = max(hsv_iqr_right[2], 1.0)
        v_iqr_target = (v_iqr_left + v_iqr_right) / 2
        contrast_left = float(np.clip(v_iqr_target / v_iqr_left, 0.8, 1.2))
        contrast_right = float(np.clip(v_iqr_target / v_iqr_right, 0.8, 1.2))

        # Color balance from RGB channels
        # Compute ratio to reach target, then normalize so green stays at 1.0
        rgb_med_left = np.maximum(rgb_med_left, 1.0)
        rgb_med_right = np.maximum(rgb_med_right, 1.0)

        cb_left = rgb_med_target / rgb_med_left
        cb_right = rgb_med_target / rgb_med_right

        # Normalize around green channel (index 1) to preserve overall brightness
        cb_left = cb_left / cb_left[1]
        cb_right = cb_right / cb_right[1]

        # Clamp to reasonable range
        cb_left = np.clip(cb_left, 0.85, 1.15).tolist()
        cb_right = np.clip(cb_right, 0.85, 1.15).tolist()

        # Temperature: derived from blue/red balance
        # Positive = warmer (more red), negative = cooler (more blue)
        temp_left = float(np.clip((cb_left[0] - cb_left[2]) * 0.5, -0.2, 0.2))
        temp_right = float(np.clip((cb_right[0] - cb_right[2]) * 0.5, -0.2, 0.2))

        return {
            'left': {
                'brightness': round(brightness_left, 4),
                'contrast': round(contrast_left, 4),
                'saturation': round(saturation_left, 4),
                'colorBalance': [round(c, 4) for c in cb_left],
                'temperature': round(temp_left, 4),
            },
            'right': {
                'brightness': round(brightness_right, 4),
                'contrast': round(contrast_right, 4),
                'saturation': round(saturation_right, 4),
                'colorBalance': [round(c, 4) for c in cb_right],
                'temperature': round(temp_right, 4),
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
    }
    return {'left': default.copy(), 'right': default.copy()}
