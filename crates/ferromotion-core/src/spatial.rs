//! **Spatial indices** — a 3-D **k-d tree** (median-split) and a **voxel hash** for fast nearest-neighbour,
//! k-nearest, and radius queries. This is the shared acceleration substrate the sweep flagged as a
//! prerequisite for scaling: LiDAR odometry (KISS-ICP/NDT), incremental ESDF mapping, GP motion planning,
//! and vectorized sampling planners all reduce to "which points are near this query?", and today the ICP
//! path does it brute-force. The k-d tree gives `O(log n)` expected queries with exact pruning; the voxel
//! hash gives `O(1)` cell lookup for grid-scale point sets. Pure `nalgebra` + `std` → WASM-clean.

use nalgebra::Vector3;
use std::collections::HashMap;

/// A static 3-D k-d tree over a point set (built once, queried many times).
#[derive(Clone, Debug)]
pub struct KdTree {
    pts: Vec<Vector3<f64>>,
    nodes: Vec<Node>,
    root: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    idx: usize,
    axis: usize,
    left: Option<usize>,
    right: Option<usize>,
}

fn build_rec(pts: &[Vector3<f64>], nodes: &mut Vec<Node>, idxs: &mut [usize], depth: usize) -> Option<usize> {
    if idxs.is_empty() {
        return None;
    }
    let axis = depth % 3;
    idxs.sort_by(|&a, &b| pts[a][axis].partial_cmp(&pts[b][axis]).unwrap());
    let mid = idxs.len() / 2;
    let me = nodes.len();
    nodes.push(Node { idx: idxs[mid], axis, left: None, right: None });
    let (lo, hi) = idxs.split_at_mut(mid);
    let left = build_rec(pts, nodes, lo, depth + 1);
    let right = build_rec(pts, nodes, &mut hi[1..], depth + 1);
    nodes[me].left = left;
    nodes[me].right = right;
    Some(me)
}

impl KdTree {
    /// Build a k-d tree over `pts`.
    pub fn build(pts: Vec<Vector3<f64>>) -> KdTree {
        let mut nodes = Vec::with_capacity(pts.len());
        let mut idxs: Vec<usize> = (0..pts.len()).collect();
        let root = build_rec(&pts, &mut nodes, &mut idxs, 0);
        KdTree { pts, nodes, root }
    }

    /// The stored point at index `i` (indices are those returned by the query methods).
    pub fn point(&self, i: usize) -> Vector3<f64> {
        self.pts[i]
    }

    /// The nearest point to `q`: `(index, distance)`, or `None` if empty.
    pub fn nearest(&self, q: &Vector3<f64>) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None; // (idx, dist²)
        self.nearest_rec(self.root, q, &mut best);
        best.map(|(i, d2)| (i, d2.sqrt()))
    }

    fn nearest_rec(&self, node: Option<usize>, q: &Vector3<f64>, best: &mut Option<(usize, f64)>) {
        let Some(ni) = node else { return };
        let n = self.nodes[ni];
        let d2 = (self.pts[n.idx] - q).norm_squared();
        if best.is_none() || d2 < best.unwrap().1 {
            *best = Some((n.idx, d2));
        }
        let diff = q[n.axis] - self.pts[n.idx][n.axis];
        let (near, far) = if diff < 0.0 { (n.left, n.right) } else { (n.right, n.left) };
        self.nearest_rec(near, q, best);
        if diff * diff < best.unwrap().1 {
            self.nearest_rec(far, q, best);
        }
    }

    /// The `k` nearest points to `q`, sorted by increasing distance: `(index, distance)`.
    pub fn k_nearest(&self, q: &Vector3<f64>, k: usize) -> Vec<(usize, f64)> {
        let mut heap: Vec<(usize, f64)> = Vec::with_capacity(k + 1); // kept sorted ascending by dist²
        self.knn_rec(self.root, q, k, &mut heap);
        heap.into_iter().map(|(i, d2)| (i, d2.sqrt())).collect()
    }

    fn knn_rec(&self, node: Option<usize>, q: &Vector3<f64>, k: usize, heap: &mut Vec<(usize, f64)>) {
        let Some(ni) = node else { return };
        let n = self.nodes[ni];
        let d2 = (self.pts[n.idx] - q).norm_squared();
        if heap.len() < k || d2 < heap.last().unwrap().1 {
            let pos = heap.partition_point(|&(_, e)| e < d2);
            heap.insert(pos, (n.idx, d2));
            if heap.len() > k {
                heap.pop();
            }
        }
        let diff = q[n.axis] - self.pts[n.idx][n.axis];
        let (near, far) = if diff < 0.0 { (n.left, n.right) } else { (n.right, n.left) };
        self.knn_rec(near, q, k, heap);
        if heap.len() < k || diff * diff < heap.last().unwrap().1 {
            self.knn_rec(far, q, k, heap);
        }
    }

    /// All point indices within Euclidean `radius` of `q`.
    pub fn within_radius(&self, q: &Vector3<f64>, radius: f64) -> Vec<usize> {
        let mut out = Vec::new();
        self.radius_rec(self.root, q, radius * radius, &mut out);
        out
    }

    fn radius_rec(&self, node: Option<usize>, q: &Vector3<f64>, r2: f64, out: &mut Vec<usize>) {
        let Some(ni) = node else { return };
        let n = self.nodes[ni];
        if (self.pts[n.idx] - q).norm_squared() <= r2 {
            out.push(n.idx);
        }
        let diff = q[n.axis] - self.pts[n.idx][n.axis];
        let (near, far) = if diff < 0.0 { (n.left, n.right) } else { (n.right, n.left) };
        self.radius_rec(near, q, r2, out);
        if diff * diff <= r2 {
            self.radius_rec(far, q, r2, out);
        }
    }
}

/// A uniform-grid **voxel hash**: buckets points into cells of side `cell` for `O(1)` neighbourhood
/// lookups — the grid-scale complement to the k-d tree (streaming maps, occupancy, broadphase).
#[derive(Clone, Debug)]
pub struct VoxelHash {
    cell: f64,
    map: HashMap<[i64; 3], Vec<usize>>,
    pts: Vec<Vector3<f64>>,
}

impl VoxelHash {
    /// The cell coordinate a point falls in.
    pub fn cell_of(p: &Vector3<f64>, cell: f64) -> [i64; 3] {
        [(p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64]
    }

    /// Build a voxel hash over `pts` with cell side `cell`.
    pub fn build(pts: Vec<Vector3<f64>>, cell: f64) -> VoxelHash {
        let mut map: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for (i, p) in pts.iter().enumerate() {
            map.entry(Self::cell_of(p, cell)).or_default().push(i);
        }
        VoxelHash { cell, map, pts }
    }

    /// All point indices within `radius` of `q` — scans the cell block that `radius` can reach, then
    /// filters exactly.
    pub fn within_radius(&self, q: &Vector3<f64>, radius: f64) -> Vec<usize> {
        let r2 = radius * radius;
        let reach = (radius / self.cell).ceil() as i64;
        let c = Self::cell_of(q, self.cell);
        let mut out = Vec::new();
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    if let Some(bucket) = self.map.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        for &i in bucket {
                            if (self.pts[i] - q).norm_squared() <= r2 {
                                out.push(i);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Number of occupied cells.
    pub fn occupied_cells(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(n: usize) -> Vec<Vector3<f64>> {
        let mut seed = 0xDEADBEEFu64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) * 10.0 - 5.0
        };
        (0..n).map(|_| Vector3::new(rng(), rng(), rng())).collect()
    }

    fn brute_nearest(pts: &[Vector3<f64>], q: &Vector3<f64>) -> (usize, f64) {
        pts.iter().enumerate().map(|(i, p)| (i, (p - q).norm())).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap()
    }

    #[test]
    fn kd_nearest_matches_brute_force() {
        // THE ORACLE. The tree's nearest neighbour equals the exhaustive nearest for many queries.
        let pts = cloud(500);
        let tree = KdTree::build(pts.clone());
        let qs = cloud(200);
        for q in &qs {
            let (ti, td) = tree.nearest(q).unwrap();
            let (bi, bd) = brute_nearest(&pts, q);
            assert!((td - bd).abs() < 1e-12, "distance mismatch: {td} vs {bd}");
            assert_eq!(ti, bi, "index mismatch (no ties expected in a random cloud)");
        }
    }

    #[test]
    fn kd_k_nearest_matches_the_brute_force_top_k() {
        let pts = cloud(400);
        let tree = KdTree::build(pts.clone());
        let q = Vector3::new(0.3, -0.7, 1.1);
        let k = 12;
        let got: Vec<usize> = tree.k_nearest(&q, k).into_iter().map(|(i, _)| i).collect();
        let mut all: Vec<(usize, f64)> = pts.iter().enumerate().map(|(i, p)| (i, (p - q).norm())).collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let want: Vec<usize> = all[..k].iter().map(|&(i, _)| i).collect();
        assert_eq!(got, want, "k-NN set/order mismatch");
    }

    #[test]
    fn kd_radius_query_matches_brute_force() {
        let pts = cloud(400);
        let tree = KdTree::build(pts.clone());
        let q = Vector3::new(-1.0, 0.5, 0.0);
        let r = 2.0;
        let mut got = tree.within_radius(&q, r);
        got.sort();
        let mut want: Vec<usize> = pts.iter().enumerate().filter(|(_, p)| (*p - q).norm() <= r).map(|(i, _)| i).collect();
        want.sort();
        assert_eq!(got, want, "radius set mismatch");
    }

    #[test]
    fn voxel_hash_radius_query_matches_brute_force() {
        let pts = cloud(600);
        let vh = VoxelHash::build(pts.clone(), 0.5);
        let q = Vector3::new(1.0, -1.0, 0.5);
        let r = 1.3;
        let mut got = vh.within_radius(&q, r);
        got.sort();
        let mut want: Vec<usize> = pts.iter().enumerate().filter(|(_, p)| (*p - q).norm() <= r).map(|(i, _)| i).collect();
        want.sort();
        assert_eq!(got, want, "voxel radius set mismatch");
        assert!(vh.occupied_cells() > 0);
    }
}
