//! Geometric utility functions for projection math.

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
