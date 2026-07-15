//! **Multi-robot consensus and formation control** (Olfati-Saber; Ren & Beard) — the first
//! *distributed* control in the stack. Every other controller here commands one robot from full
//! state; these agents see only their neighbours, yet the group agrees.
//!
//! The whole theory is the **graph Laplacian** `L = D − A`. The consensus protocol
//! `ẋᵢ = Σ_{j∈Nᵢ} aᵢⱼ(xⱼ − xᵢ)` is exactly `ẋ = −L x`, and three facts follow:
//!
//! * `L·1 = 0` ⇒ the **average is exactly conserved** — the group can only agree on where it already was;
//! * consensus is reached **iff the graph is connected** ⇔ the **Fiedler value** `λ₂ > 0`;
//! * disagreement decays as `e^{−λ₂ t}` — **`λ₂` isn't a bound on the rate, it *is* the rate**.
//!
//! Formation control is the same protocol on the *error* `e = x − d` for desired offsets `d`: driving
//! `e` to consensus makes `xᵢ − xⱼ → dᵢ − dⱼ`, i.e. the shape, centred on the conserved average.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A weighted, undirected interaction graph (symmetric adjacency, no self-loops).
#[derive(Clone, Debug)]
pub struct Graph {
    pub adj: DMatrix<f64>,
}

impl Graph {
    pub fn n(&self) -> usize {
        self.adj.nrows()
    }

    /// Graph Laplacian `L = D − A`.
    pub fn laplacian(&self) -> DMatrix<f64> {
        let n = self.n();
        let mut l = -self.adj.clone();
        for i in 0..n {
            let deg: f64 = (0..n).map(|j| self.adj[(i, j)]).sum();
            l[(i, i)] = deg - self.adj[(i, i)];
        }
        l
    }

    /// **Algebraic connectivity** (Fiedler value) `λ₂` — the second-smallest Laplacian eigenvalue.
    /// Zero iff the graph is disconnected; otherwise it is the consensus rate.
    pub fn fiedler_value(&self) -> f64 {
        let mut ev: Vec<f64> = self.laplacian().symmetric_eigen().eigenvalues.iter().cloned().collect();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ev[1]
    }

    pub fn is_connected(&self) -> bool {
        self.fiedler_value() > 1e-9
    }
}

/// One consensus step `ẋ = −L x` (per scalar state).
pub fn consensus_step(graph: &Graph, x: &mut DVector<f64>, dt: f64) {
    let dx = -(graph.laplacian() * &*x);
    *x += dx * dt;
}

/// One formation step: consensus on the error `e = x − d`, so the group converges to the *shape* `d`.
/// `x` and `offsets` are one coordinate axis at a time (call per axis).
pub fn formation_step(graph: &Graph, x: &mut DVector<f64>, offsets: &DVector<f64>, dt: f64) {
    let mut e = &*x - offsets;
    consensus_step(graph, &mut e, dt);
    *x = e + offsets;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path graph `0–1–2–…–(n−1)` with unit weights.
    fn path(n: usize) -> Graph {
        let mut adj = DMatrix::zeros(n, n);
        for i in 0..n - 1 {
            adj[(i, i + 1)] = 1.0;
            adj[(i + 1, i)] = 1.0;
        }
        Graph { adj }
    }

    /// Two disjoint edges: 0–1 and 2–3 (disconnected).
    fn two_components() -> Graph {
        let mut adj = DMatrix::zeros(4, 4);
        for (i, j) in [(0, 1), (2, 3)] {
            adj[(i, j)] = 1.0;
            adj[(j, i)] = 1.0;
        }
        Graph { adj }
    }

    #[test]
    fn laplacian_has_the_defining_properties() {
        let g = path(5);
        let l = g.laplacian();
        assert!((l.clone() - l.transpose()).norm() < 1e-12, "L must be symmetric");
        // L·1 = 0 — the all-ones vector spans the null space of a connected graph's Laplacian.
        let ones = DVector::from_element(5, 1.0);
        assert!((l * ones).norm() < 1e-12, "L·1 must vanish");
        // PSD, with exactly one zero eigenvalue when connected.
        let mut ev: Vec<f64> = g.laplacian().symmetric_eigen().eigenvalues.iter().cloned().collect();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(ev[0].abs() < 1e-9, "smallest eigenvalue must be 0");
        assert!(ev.iter().all(|&e| e > -1e-9), "L must be positive semidefinite");
        assert!(ev[1] > 1e-9, "a connected graph has λ₂ > 0");
    }

    #[test]
    fn the_average_is_exactly_conserved() {
        // 1ᵀL = 0 ⇒ d/dt Σxᵢ = 0. The group can only agree on where it already was.
        let g = path(5);
        let mut x = DVector::from_row_slice(&[3.0, -1.0, 4.0, 0.5, -2.5]);
        let mean0 = x.sum() / 5.0;
        // λ₂ ≈ 0.382 for this graph, so agreement needs ~40 s to reach 1e-6 (see the rate test).
        for _ in 0..40_000 {
            consensus_step(&g, &mut x, 1e-3);
        }
        assert!((x.sum() / 5.0 - mean0).abs() < 1e-12, "average drifted: {} vs {mean0}", x.sum() / 5.0);
        // …and everyone agrees on exactly that average.
        for i in 0..5 {
            assert!((x[i] - mean0).abs() < 1e-6, "agent {i} did not reach the average");
        }
    }

    #[test]
    fn consensus_requires_connectivity() {
        // Disconnected ⇒ λ₂ = 0 ⇒ no global consensus; each component agrees only within itself.
        let g = two_components();
        assert!(g.fiedler_value().abs() < 1e-9, "disconnected graph must have λ₂ = 0");
        assert!(!g.is_connected());

        let mut x = DVector::from_row_slice(&[0.0, 2.0, 8.0, 10.0]);
        for _ in 0..20_000 {
            consensus_step(&g, &mut x, 1e-3);
        }
        // Each component reaches its own average (1.0 and 9.0) — not the global average (5.0).
        assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
        assert!((x[2] - 9.0).abs() < 1e-6 && (x[3] - 9.0).abs() < 1e-6);
        assert!((x[0] - x[2]).abs() > 1.0, "components must NOT agree with each other");
    }

    #[test]
    fn disagreement_decays_at_exactly_the_fiedler_rate() {
        // λ₂ is not a bound on the convergence rate — it *is* the rate.
        let n = 5;
        let g = path(n);
        let lambda2 = g.fiedler_value();
        // Path graph: λ₂ = 2(1 − cos(π/n)) analytically.
        let analytic = 2.0 * (1.0 - (std::f64::consts::PI / n as f64).cos());
        assert!((lambda2 - analytic).abs() < 1e-9, "Fiedler value {lambda2} vs analytic {analytic}");

        let mut x = DVector::from_row_slice(&[3.0, -1.0, 4.0, 0.5, -2.5]);
        let mean = x.sum() / n as f64;
        let disagreement = |v: &DVector<f64>| (v - DVector::from_element(n, mean)).norm();
        let (dt, warmup, window) = (1e-4, 40_000, 40_000);
        for _ in 0..warmup {
            consensus_step(&g, &mut x, dt); // let the faster modes die out
        }
        let d1 = disagreement(&x);
        for _ in 0..window {
            consensus_step(&g, &mut x, dt);
        }
        let d2 = disagreement(&x);
        let rate = -(d2 / d1).ln() / (window as f64 * dt);
        assert!((rate - lambda2).abs() / lambda2 < 0.02, "decay rate {rate:.4} vs λ₂ {lambda2:.4}");
    }

    #[test]
    fn formation_converges_to_the_shape_about_the_conserved_centroid() {
        // Four agents, desired shape = a unit square, on a connected (path) graph.
        let g = path(4);
        let dx = DVector::from_row_slice(&[0.0, 1.0, 1.0, 0.0]);
        let dy = DVector::from_row_slice(&[0.0, 0.0, 1.0, 1.0]);
        let mut x = DVector::from_row_slice(&[2.0, -1.0, 0.3, 4.0]);
        let mut y = DVector::from_row_slice(&[0.0, 3.0, -2.0, 1.0]);
        let (cx0, cy0) = (x.sum() / 4.0, y.sum() / 4.0);

        for _ in 0..40_000 {
            formation_step(&g, &mut x, &dx, 1e-3);
            formation_step(&g, &mut y, &dy, 1e-3);
        }
        // The *shape* is achieved: every relative offset matches the target.
        for i in 0..4 {
            for j in 0..4 {
                let (rx, ry) = (x[i] - x[j], y[i] - y[j]);
                let (tx, ty) = (dx[i] - dx[j], dy[i] - dy[j]);
                assert!((rx - tx).abs() < 1e-5 && (ry - ty).abs() < 1e-5, "offset {i}-{j} wrong");
            }
        }
        // …centred on the conserved average — formation control never translates the group.
        assert!((x.sum() / 4.0 - cx0).abs() < 1e-9 && (y.sum() / 4.0 - cy0).abs() < 1e-9, "centroid moved");
    }
}

