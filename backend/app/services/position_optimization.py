"""
Camera position optimization service.

Optimizes camera positions and rotations based on matched feature points
from left and right camera views.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List


def optimize_position(left_points: List[List[float]], right_points: List[List[float]]) -> Dict[str, float]:
    """
    Optimize camera positions based on matched points.

    Args:
        left_points: List of [x, y] coordinates from left camera (normalized)
        right_points: List of [x, y] coordinates from right camera (normalized)

    Returns:
        Dictionary with optimized parameters:
        - cameraAxisOffset (cam_d): Camera distance from origin
        - intersect: Plane intersection ratio
        - xTy (x_ty): Y-axis translation for left plane
        - xRz (x_rz): Z-axis rotation for left plane
        - zRx (z_rx): X-axis rotation for right plane

    Raises:
        ValueError: If optimization fails or inputs are invalid
    """
    # Validate inputs
    if not left_points or not right_points:
        raise ValueError("Point lists cannot be empty")

    if len(left_points) != len(right_points):
        raise ValueError("Point lists must have equal length")

    # Convert to numpy arrays
    x_plane_pts_2d = np.array(left_points)
    z_plane_pts_2d = np.array(right_points)

    # Convert 2D points to 3D
    x_plane = _convert_to_3d(x_plane_pts_2d, 'x')
    z_plane = _convert_to_3d(z_plane_pts_2d, 'z')

    # Minimize sum of angles
    result = _minimize_sum_of_angles(x_plane, z_plane)

    if result is None:
        raise ValueError("Position optimization failed")

    return result


def _convert_to_3d(points: np.ndarray, plane_direction: str) -> np.ndarray:
    """
    Convert 2D points to 3D based on plane direction.

    Args:
        points: Array of [x, y] coordinates
        plane_direction: 'x' for left plane, 'z' for right plane

    Returns:
        Array of [x, y, z] 3D coordinates
    """
    if plane_direction == 'x':
        # Left plane: x-z plane (x, -y, 0)
        return np.array([[x, -y, 0] for x, y in points])
    elif plane_direction == 'z':
        # Right plane: y-z plane (0, -y, -z)
        return np.array([[0, -y, -z] for z, y in points])
    else:
        raise ValueError("Invalid plane direction. Use 'x' or 'z'.")


def _sum_of_angles(x_plane: np.ndarray, z_plane: np.ndarray, camera: np.ndarray) -> float:
    """
    Compute sum of angles between camera-to-point vectors.

    For each pair of corresponding points, compute the angle between
    the vectors from the camera to each point.
    """
    total_angle = 0.0

    for x_point, z_point in zip(x_plane, z_plane):
        # Vectors from camera to points
        v_x = x_point - camera
        v_z = z_point - camera

        # Normalize vectors
        v_x_norm = v_x / np.linalg.norm(v_x)
        v_z_norm = v_z / np.linalg.norm(v_z)

        # Compute angle via dot product
        dot = np.clip(np.dot(v_x_norm, v_z_norm), -1.0, 1.0)
        angle = np.arccos(dot)

        total_angle += angle

    return total_angle


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Create 3D rotation matrix from Euler angles."""
    # Rotation around X-axis
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

    # Rotation around Y-axis
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])

    # Rotation around Z-axis
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def _apply_transformations(
    x_plane: np.ndarray, z_plane: np.ndarray, x_tx: float, x_ty: float, x_rz: float, z_tz: float, z_rx: float
) -> tuple:
    """Apply translations and rotations to both planes."""
    # Transform X plane (left camera)
    x_center = np.array([x_tx, x_ty, 0])
    x_rot = _rotation_matrix(0, 0, x_rz)
    x_plane_transformed = (x_plane @ x_rot.T) + x_center

    # Transform Z plane (right camera)
    z_center = np.array([0, 0, z_tz])
    z_rot = _rotation_matrix(z_rx, 0, 0)
    z_plane_transformed = (z_plane @ z_rot.T) + z_center

    return x_plane_transformed, z_plane_transformed


def _minimize_sum_of_angles(x_plane_points: np.ndarray, z_plane_points: np.ndarray) -> Dict[str, float]:
    """
    Minimize the sum of angles between camera-to-point vectors.

    Optimizes camera distance, plane positions, and rotations.
    """
    plane_width = 1.0

    # Parameter bounds
    bounds = [
        (-1.0, 1.0),  # x_ty: Y translation for left plane
        (0.0, 1.0),  # intersect: Plane intersection ratio
        (0.1, 0.35),  # cam_d: Camera distance
        (-np.pi, np.pi),  # x_rz: Left plane Z rotation
        (-np.pi, np.pi),  # z_rx: Right plane X rotation
    ]

    # Initial guess (midpoint of bounds)
    x0 = [(b[0] + b[1]) / 2 for b in bounds]

    def objective(params):
        """Objective function to minimize."""
        x_ty, intersect, cam_d, x_rz, z_rx = params

        # Compute translations from intersection ratio
        x_tx = plane_width / 2 * (1 - intersect)
        z_tz = plane_width / 2 * (1 - intersect)

        # Camera position
        camera = np.array([cam_d, 0, cam_d])

        # Apply transformations
        x_plane_trans, z_plane_trans = _apply_transformations(
            x_plane_points, z_plane_points, x_tx, x_ty, x_rz, z_tz, z_rx
        )

        # Compute sum of angles
        return _sum_of_angles(x_plane_trans, z_plane_trans, camera)

    # Run optimization
    result = minimize(objective, x0, bounds=bounds, method='Powell', options={'disp': False, 'maxiter': 1000})

    if not result.success:
        return None

    # Extract optimized parameters
    x_ty, intersect, cam_d, x_rz, z_rx = result.x

    return {
        "cameraAxisOffset": float(cam_d),
        "intersect": float(intersect),
        "xTy": float(x_ty),
        "xRz": float(x_rz),
        "zRx": float(z_rx),
    }
