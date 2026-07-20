//! **Rotation averaging** — recover the absolute orientations of a set of frames (cameras, robots, sensors)
//! from noisy **relative** rotation measurements over a graph. Given edges `R_ij ≈ Rᵢ Rⱼᵀ`, find the `Rᵢ`
//! (up to one global gauge). This is the rotation-synchronization step that **initializes global
//! structure-from-motion** (before translation averaging and bundle adjustment) and that **merges
//! multi-robot maps** or fuses a network of relative-pose estimates — the *graph* generalization of the
//! single-rotation [`crate::average_quaternions`] (Markley) mean.
//!
//! The method: a **spanning-tree initialization** propagates absolute rotations from a root (exact when the
//! measurements are consistent), then **Gauss–Seidel refinement** replaces each rotation with the
//! Markley average of the predictions from all its incident edges — so loop closures pull the estimate
//! toward the noise-averaged optimum. Verified: it recovers ground-truth orientations (up to gauge) from
//! consistent measurements, and refinement lowers the error below the tree-only initialization under noise.
//! Pure `nalgebra` → WASM-clean.

use crate::quat_mean::average_quaternions;
use nalgebra::UnitQuaternion;

/// A relative-rotation edge: `rij ≈ R_i · R_jᵀ` between frames `i` and `j`.
#[derive(Clone, Copy, Debug)]
pub struct RotEdge {
    pub i: usize,
    pub j: usize,
    pub rij: UnitQuaternion<f64>,
}

/// Spanning-tree initialization: propagate absolute rotations from node 0 (`= identity`) along a BFS tree.
pub fn spanning_tree_init(n: usize, edges: &[RotEdge]) -> Vec<UnitQuaternion<f64>> {
    let mut r = vec![UnitQuaternion::identity(); n];
    let mut known = vec![false; n];
    known[0] = true;
    // adjacency: node → list of (edge index, is_this_node_i)
    let mut adj: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    for (e, edge) in edges.iter().enumerate() {
        adj[edge.i].push((e, true));
        adj[edge.j].push((e, false));
    }
    let mut queue = std::collections::VecDeque::from([0usize]);
    while let Some(u) = queue.pop_front() {
        for &(e, u_is_i) in &adj[u] {
            let edge = &edges[e];
            let v = if u_is_i { edge.j } else { edge.i };
            if known[v] {
                continue;
            }
            // rij = Ri Rjᵀ ⇒ if u=i (know Ri): Rj = rijᵀ Ri ; if u=j (know Rj): Ri = rij Rj
            r[v] = if u_is_i { edge.rij.inverse() * r[u] } else { edge.rij * r[u] };
            known[v] = true;
            queue.push_back(v);
        }
    }
    r
}

/// Full rotation averaging: spanning-tree initialization followed by `iters` Gauss–Seidel refinement passes
/// (node 0 held fixed as the gauge). Returns the absolute rotations.
pub fn rotation_averaging(n: usize, edges: &[RotEdge], iters: usize) -> Vec<UnitQuaternion<f64>> {
    let mut r = spanning_tree_init(n, edges);
    let mut adj: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    for (e, edge) in edges.iter().enumerate() {
        adj[edge.i].push((e, true));
        adj[edge.j].push((e, false));
    }
    for _ in 0..iters {
        for node in 1..n {
            // predictions of R_node from each incident edge, given the current neighbour estimates
            let preds: Vec<UnitQuaternion<f64>> = adj[node]
                .iter()
                .map(|&(e, node_is_i)| {
                    let edge = &edges[e];
                    if node_is_i {
                        edge.rij * r[edge.j] // Ri = rij Rj
                    } else {
                        edge.rij.inverse() * r[edge.i] // Rj = rijᵀ Ri
                    }
                })
                .collect();
            if !preds.is_empty() {
                r[node] = average_quaternions(&preds, None);
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn q(seed: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(Vector3::new(0.3 + seed, -0.7, 0.5 - 0.2 * seed)), 0.6 + 0.4 * seed)
    }

    // a connected graph: a ring 0-1-2-3-4-0 plus two chords (loop closures)
    fn graph_edges(truth: &[UnitQuaternion<f64>], noise: impl Fn(usize) -> UnitQuaternion<f64>) -> Vec<RotEdge> {
        let pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)];
        pairs
            .iter()
            .enumerate()
            .map(|(k, &(i, j))| RotEdge { i, j, rij: noise(k) * truth[i] * truth[j].inverse() })
            .collect()
    }

    #[test]
    fn it_recovers_ground_truth_from_consistent_measurements() {
        // THE ORACLE. Noise-free relative measurements ⇒ the absolute rotations are recovered up to the
        // global gauge (node 0 fixed to identity, so align by composing with truth[0]).
        let truth: Vec<_> = (0..5).map(|i| q(i as f64 * 0.4)).collect();
        let edges = graph_edges(&truth, |_| UnitQuaternion::identity());
        let est = rotation_averaging(5, &edges, 20);
        for (i, t) in truth.iter().enumerate() {
            let aligned = est[i] * truth[0]; // undo the gauge (est[0] = identity = truth[0]ᵀ·truth[0])
            assert!(aligned.angle_to(t) < 1e-6, "node {i}: off by {}", aligned.angle_to(t));
        }
    }

    #[test]
    fn refinement_beats_the_spanning_tree_under_noise() {
        // THE HEADLINE. With noisy edges, averaging over loop closures yields a lower orientation error than
        // the spanning-tree initialization alone.
        let truth: Vec<_> = (0..5).map(|i| q(i as f64 * 0.4)).collect();
        // small deterministic per-edge rotational noise
        let noise = |k: usize| {
            let a = 0.03 * ((k as f64 * 1.7).sin());
            let b = 0.03 * ((k as f64 * 2.3).cos());
            UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.5, -0.7)), a + b)
        };
        let edges = graph_edges(&truth, noise);
        let err = |est: &[UnitQuaternion<f64>]| -> f64 {
            est.iter().enumerate().map(|(i, e)| (*e * truth[0]).angle_to(&truth[i])).sum::<f64>()
        };
        let tree = spanning_tree_init(5, &edges);
        let avg = rotation_averaging(5, &edges, 30);
        assert!(err(&avg) < err(&tree), "averaging should beat the tree: {} vs {}", err(&avg), err(&tree));
    }

    #[test]
    fn a_single_relative_measurement_is_consistent() {
        // Two frames, one edge: the recovered relative rotation matches the measurement.
        let truth = [q(0.0), q(1.1)];
        let rij = truth[0] * truth[1].inverse();
        let est = rotation_averaging(2, &[RotEdge { i: 0, j: 1, rij }], 5);
        let recovered = est[0] * est[1].inverse();
        assert!(recovered.angle_to(&rij) < 1e-9, "relative rotation should match the measurement");
    }
}
