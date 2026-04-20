//! Density-based clustering (DBSCAN) for player grouping.
//!
//! Groups nearby detections into clusters, ignoring isolated outliers
//! (goalkeeper, substitutes). O(n^2) which is fine for <50 points.

/// Run DBSCAN clustering on 2D points.
///
/// Returns a label per point: non-negative = cluster ID, -1 = noise.
/// Points within `eps` of each other are neighbors. A point with at
/// least `min_neighbors` neighbors is a "core" point. Connected core
/// points form a cluster; non-core neighbors are included but don't
/// expand the cluster.
pub fn dbscan(points: &[(f32, f32)], eps: f32, min_neighbors: usize) -> Vec<i32> {
    let n = points.len();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dy = points[i].0 - points[j].0;
            let dp = points[i].1 - points[j].1;
            if (dy * dy + dp * dp).sqrt() < eps {
                neighbors[i].push(j);
                neighbors[j].push(i);
            }
        }
    }

    let mut labels: Vec<i32> = vec![-1; n];
    let mut current_cluster = 0_i32;

    for i in 0..n {
        if labels[i] != -1 || neighbors[i].len() < min_neighbors {
            continue;
        }
        let mut queue = vec![i];
        labels[i] = current_cluster;
        while let Some(pt) = queue.pop() {
            for &nb in &neighbors[pt] {
                if labels[nb] == -1 {
                    labels[nb] = current_cluster;
                    if neighbors[nb].len() >= min_neighbors {
                        queue.push(nb);
                    }
                }
            }
        }
        current_cluster += 1;
    }

    labels
}

/// Find the indices of points in the largest cluster.
///
/// Returns an empty vec if all points are noise.
pub fn largest_cluster_indices(labels: &[i32]) -> Vec<usize> {
    let max_label = labels.iter().copied().max().unwrap_or(-1);
    if max_label < 0 {
        return Vec::new();
    }

    let mut sizes = vec![0_usize; (max_label + 1) as usize];
    for &l in labels {
        if l >= 0 {
            sizes[l as usize] += 1;
        }
    }

    let best = sizes
        .iter()
        .enumerate()
        .max_by_key(|&(_, &s)| s)
        .map(|(id, _)| id as i32)
        .unwrap_or(-1);

    labels
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l == best)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let labels = dbscan(&[], 0.1, 2);
        assert!(labels.is_empty());
        assert!(largest_cluster_indices(&labels).is_empty());
    }

    #[test]
    fn all_noise_when_too_far() {
        let points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let labels = dbscan(&points, 0.1, 2);
        assert!(labels.iter().all(|&l| l == -1));
    }

    #[test]
    fn single_cluster() {
        let points = [(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.15, 0.0)];
        let labels = dbscan(&points, 0.08, 1);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn two_clusters() {
        let points = [
            (0.0, 0.0),
            (0.05, 0.0), // cluster A
            (1.0, 0.0),
            (1.05, 0.0), // cluster B
        ];
        let labels = dbscan(&points, 0.08, 1);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn largest_cluster_selected() {
        let points = [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.1, 0.0), // cluster A (3 pts)
            (1.0, 0.0), // noise
        ];
        let labels = dbscan(&points, 0.08, 1);
        let largest = largest_cluster_indices(&labels);
        assert_eq!(largest.len(), 3);
        assert!(largest.contains(&0));
        assert!(largest.contains(&1));
        assert!(largest.contains(&2));
    }

    #[test]
    fn outlier_excluded() {
        let points = [
            (0.3, 0.0),
            (0.35, 0.0),
            (0.4, 0.0), // tight group
            (2.0, 0.0), // outlier (goalkeeper)
        ];
        let labels = dbscan(&points, 0.08, 2);
        let largest = largest_cluster_indices(&labels);
        assert_eq!(largest.len(), 3);
        assert!(!largest.contains(&3));
    }
}
