//! **Translation averaging** — recover the absolute positions of a set of frames from relative translation
//! measurements `t_ij ≈ pᵢ − pⱼ` over a graph. Paired with [`crate::rotation_averaging`], this is the second
//! half of the **global structure-from-motion** initializer: rotation synchronization fixes the
//! orientations, then translation averaging places the cameras/robots (up to one global position gauge),
//! after which [`crate::bundle`] refines everything. It is also the position back-end for multi-robot map
//! merging and relative-pose fusion.
//!
//! With full (metric-scale) relative translations the problem is a **linear least squares** whose normal
//! equations are the graph Laplacian — the three coordinates decouple and share one factorization, so it is
//! fast and exact. (Direction-only SfM translations need the nonlinear LUD/1DSfM machinery; out of scope
//! here.) Verified: it recovers ground-truth positions (up to the global offset) from consistent
//! measurements, a single edge reproduces its relative translation, and it stays least-squares-optimal under
//! noise. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, Vector3};

/// A relative-translation edge: `t_ij ≈ pᵢ − pⱼ` between frames `i` and `j`.
#[derive(Clone, Copy, Debug)]
pub struct TransEdge {
    pub i: usize,
    pub j: usize,
    pub t_ij: Vector3<f64>,
}

/// Recover absolute positions `pᵢ` (with `p₀ = 0` fixing the gauge) minimizing `Σ ‖(pᵢ − pⱼ) − t_ij‖²`.
/// The three coordinates share the graph-Laplacian normal matrix, factored once.
pub fn translation_averaging(n: usize, edges: &[TransEdge]) -> Vec<Vector3<f64>> {
    // A: one row per edge (+1 at i, −1 at j) plus a gauge row pinning node 0 to the origin.
    let m = edges.len() + 1;
    let mut a = DMatrix::zeros(m, n);
    let mut bx = DVector::zeros(m);
    let mut by = DVector::zeros(m);
    let mut bz = DVector::zeros(m);
    for (e, edge) in edges.iter().enumerate() {
        a[(e, edge.i)] = 1.0;
        a[(e, edge.j)] = -1.0;
        bx[e] = edge.t_ij.x;
        by[e] = edge.t_ij.y;
        bz[e] = edge.t_ij.z;
    }
    a[(edges.len(), 0)] = 1.0; // gauge: p0 = 0 (b = 0 already)

    let ata = a.transpose() * &a;
    let at = a.transpose();
    let chol = ata.cholesky().expect("normal matrix must be positive definite (graph connected)");
    let px = chol.solve(&(&at * bx));
    let py = chol.solve(&(&at * by));
    let pz = chol.solve(&(&at * bz));
    (0..n).map(|i| Vector3::new(px[i], py[i], pz[i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges_from(truth: &[Vector3<f64>], noise: impl Fn(usize) -> Vector3<f64>) -> Vec<TransEdge> {
        let pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)];
        pairs.iter().enumerate().map(|(k, &(i, j))| TransEdge { i, j, t_ij: (truth[i] - truth[j]) + noise(k) }).collect()
    }

    fn truth5() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.5, -1.0),
            Vector3::new(3.5, 2.0, 0.5),
            Vector3::new(1.0, 3.0, 1.5),
            Vector3::new(-1.0, 1.5, 0.5),
        ]
    }

    #[test]
    fn it_recovers_positions_from_consistent_measurements() {
        // THE ORACLE. Noise-free relative translations ⇒ absolute positions recovered up to the global
        // offset (node 0 pinned to origin, so shift truth by −truth[0], which is 0 here).
        let truth = truth5();
        let edges = edges_from(&truth, |_| Vector3::zeros());
        let est = translation_averaging(5, &edges);
        for (i, t) in truth.iter().enumerate() {
            assert!((est[i] - (t - truth[0])).norm() < 1e-9, "node {i}: {} vs {}", est[i], t - truth[0]);
        }
    }

    #[test]
    fn a_single_edge_reproduces_its_relative_translation() {
        let truth = [Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, -2.0, 0.5)];
        let edges = [TransEdge { i: 0, j: 1, t_ij: truth[0] - truth[1] }];
        let est = translation_averaging(2, &edges);
        assert!(((est[0] - est[1]) - (truth[0] - truth[1])).norm() < 1e-9, "relative translation should match");
    }

    #[test]
    fn it_is_least_squares_optimal_under_noise() {
        // THE HEADLINE. With noisy edges the solution minimizes Σ‖(pi−pj)−t_ij‖²: no coordinate-wise nudge of
        // the estimate lowers the residual (a first-order optimality check).
        let truth = truth5();
        let noise = |k: usize| Vector3::new(0.05 * (k as f64 * 1.3).sin(), -0.04 * (k as f64 * 2.1).cos(), 0.03 * (k as f64).sin());
        let edges = edges_from(&truth, noise);
        let est = translation_averaging(5, &edges);
        let cost = |p: &[Vector3<f64>]| -> f64 { edges.iter().map(|e| ((p[e.i] - p[e.j]) - e.t_ij).norm_squared()).sum() };
        let base = cost(&est);
        for node in 1..5 {
            for axis in 0..3 {
                for &d in &[1e-3, -1e-3] {
                    let mut perturbed = est.clone();
                    perturbed[node][axis] += d;
                    assert!(cost(&perturbed) >= base - 1e-12, "estimate should be optimal: perturbed {} < {base}", cost(&perturbed));
                }
            }
        }
        // and it beats a deliberately-wrong guess
        let zeros = vec![Vector3::zeros(); 5];
        assert!(base < cost(&zeros), "the LS solution should beat the trivial guess");
    }
}
