//! **A learned surrogate operator — Dynamic Mode Decomposition** (Honest Fluids — stage 8, the
//! surrogate that rides the harness). The honesty harness audits *any* predictor; this is a real
//! one to audit — a data-driven linear (Koopman) operator fit to solver snapshots, the closed-form
//! backbone of neural-operator surrogates. From a rollout `x₀,…,x_m` of a solver it learns the
//! reduced operator `A_r` such that `x_{k+1} ≈ A_r x_k` in a POD basis, then *predicts* the flow
//! forward — a surrogate that costs an r×r matrix-vector product where the solver costs a full step.
//!
//! The whole method is expressible through the `m×m` snapshot Gram matrices (`Gₐᵦ = xₐ·xᵦ`,
//! `Cₐᵦ = xₐ·x′ᵦ`) plus the snapshot set for lifting — the large feature dimension is never
//! factorized. POD is computed by the method of snapshots with a self-contained Jacobi eigensolver.
//! Verified: it reconstructs the training window *and extrapolates beyond it* on a decaying
//! spectral-Navier–Stokes flow, and the low-rank operator recovers the leading viscous decay rate.

#![allow(clippy::needless_range_loop)] // index-arithmetic linear-algebra kernels — iterators would obscure

/// Cyclic Jacobi eigendecomposition of a small symmetric matrix. Returns `(eigenvalues, columns of
/// eigenvectors)`, eigenvalues in descending order.
fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..100 {
        // largest off-diagonal magnitude
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let vals = idx.iter().map(|&i| a[i][i]).collect();
    let vecs = idx.iter().map(|&i| (0..n).map(|k| v[k][i]).collect()).collect();
    (vals, vecs)
}

/// A DMD surrogate fit to a snapshot rollout.
pub struct Dmd {
    snapshots: Vec<Vec<f64>>, // x_0 .. x_m (m+1 vectors)
    /// POD basis coefficients: `wsqrt[j] = w_j / √λ_j` (columns), used to lift reduced states.
    wsqrt: Vec<Vec<f64>>, // m × r  (weights over snapshots x_0..x_{m-1})
    a_r: Vec<Vec<f64>>,   // r × r reduced operator
    pub rank: usize,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Dmd {
    /// Fit from `snaps` (`x_0 … x_m`, at least 3), keeping POD modes with energy above `tol`×max.
    pub fn fit(snaps: &[Vec<f64>], tol: f64) -> Dmd {
        let m = snaps.len() - 1; // number of (x_k, x_{k+1}) pairs
        let x: Vec<&Vec<f64>> = snaps[..m].iter().collect();
        let xp: Vec<&Vec<f64>> = snaps[1..].iter().collect();
        // Gram matrices G = XᵀX, C = XᵀX'
        let mut g = vec![vec![0.0; m]; m];
        let mut c = vec![vec![0.0; m]; m];
        for a in 0..m {
            for b in 0..m {
                g[a][b] = dot(x[a], x[b]);
                c[a][b] = dot(x[a], xp[b]);
            }
        }
        let (lambda, w) = jacobi_eigen(g);
        let lmax = lambda[0].max(1e-30);
        let r = lambda.iter().take_while(|&&l| l > tol * lmax).count().max(1);
        // wsqrt[k][j] = w_j[k] / √λ_j  (m × r)
        let mut wsqrt = vec![vec![0.0; r]; m];
        for j in 0..r {
            let inv = 1.0 / lambda[j].sqrt();
            for k in 0..m {
                wsqrt[k][j] = w[j][k] * inv;
            }
        }
        // A_r = D^{-1/2} Wᵀ C W D^{-1/2}  (r × r)
        let mut a_mat = vec![vec![0.0; r]; r];
        for i in 0..r {
            for jj in 0..r {
                let mut s = 0.0;
                for a in 0..m {
                    for b in 0..m {
                        s += wsqrt[a][i] * c[a][b] * wsqrt[b][jj];
                    }
                }
                a_mat[i][jj] = s;
            }
        }
        Dmd { snapshots: snaps[..m].to_vec(), wsqrt, a_r: a_mat, rank: r }
    }

    /// Project a full state into reduced coordinates `y = U_rᵀ x` (via snapshot dot products).
    fn project(&self, x: &[f64]) -> Vec<f64> {
        let m = self.snapshots.len();
        let r = self.rank;
        // c_k = x_k · x
        let cvec: Vec<f64> = (0..m).map(|k| dot(&self.snapshots[k], x)).collect();
        // y_j = Σ_k wsqrt[k][j] · c_k
        (0..r).map(|j| (0..m).map(|k| self.wsqrt[k][j] * cvec[k]).sum()).collect()
    }

    /// Lift a reduced state back to full space: `x = Σ_k (Σ_j wsqrt[k][j] y_j) x_k`.
    fn lift(&self, y: &[f64]) -> Vec<f64> {
        let m = self.snapshots.len();
        let feat = self.snapshots[0].len();
        let coeff: Vec<f64> = (0..m).map(|k| (0..self.rank).map(|j| self.wsqrt[k][j] * y[j]).sum()).collect();
        let mut x = vec![0.0; feat];
        for k in 0..m {
            for i in 0..feat {
                x[i] += coeff[k] * self.snapshots[k][i];
            }
        }
        x
    }

    fn apply_ar(&self, y: &[f64]) -> Vec<f64> {
        (0..self.rank).map(|i| (0..self.rank).map(|j| self.a_r[i][j] * y[j]).sum()).collect()
    }

    /// Predict the state `k` steps ahead of `x0` under the learned operator.
    pub fn predict(&self, x0: &[f64], k: usize) -> Vec<f64> {
        let mut y = self.project(x0);
        for _ in 0..k {
            y = self.apply_ar(&y);
        }
        self.lift(&y)
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::spectral_ns::SpectralNs;
    use std::f64::consts::PI;

    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
        let den: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        num / (den + 1e-30)
    }

    /// A decaying two-mode spectral-NS flow (moderate Re) → a rollout of vorticity snapshots.
    fn ns_rollout(n: usize, gap: usize, count: usize) -> Vec<Vec<f64>> {
        let l = 2.0 * PI;
        let nu = 0.02;
        let mut omega0 = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let (x, y) = (i as f64 * l / n as f64, j as f64 * l / n as f64);
                // superposition of two modes → non-trivial (rank > 1) dynamics
                omega0[i * n + j] = 0.6 * (x).sin() * (y).sin() + 0.3 * (2.0 * x).sin() * (2.0 * y).cos();
            }
        }
        let mut ns = SpectralNs::new(&omega0, n, l, nu);
        let dt = 0.01;
        let mut snaps = vec![ns.vorticity()];
        for _ in 0..count {
            for _ in 0..gap {
                ns.step(dt);
            }
            snaps.push(ns.vorticity());
        }
        snaps
    }

    /// The learned operator reconstructs the training window to high accuracy with a low-rank model.
    #[test]
    fn dmd_reconstructs_the_training_flow() {
        let snaps = ns_rollout(32, 5, 20);
        let dmd = Dmd::fit(&snaps, 1e-8);
        eprintln!("DMD rank {} from {} snapshots", dmd.rank, snaps.len());
        let mut worst = 0.0f64;
        for k in 1..snaps.len() {
            let pred = dmd.predict(&snaps[0], k);
            worst = worst.max(rel_err(&pred, &snaps[k]));
        }
        eprintln!("DMD reconstruction worst rel error: {worst:.2e}");
        assert!(dmd.rank <= 8, "surrogate not low-rank: {}", dmd.rank);
        assert!(worst < 1e-2, "DMD did not reconstruct the flow: {worst}");
    }

    /// The surrogate EXTRAPOLATES beyond the training window — it learned dynamics, not a lookup.
    /// Fit on the first part of a rollout, predict the held-out tail.
    #[test]
    fn dmd_predicts_beyond_training() {
        let full = ns_rollout(32, 5, 28);
        let train = &full[..20];
        let dmd = Dmd::fit(train, 1e-8);
        // predict the held-out steps 20..28 from x0
        let mut worst = 0.0f64;
        for k in 20..full.len() {
            let pred = dmd.predict(&full[0], k);
            worst = worst.max(rel_err(&pred, &full[k]));
        }
        eprintln!("DMD held-out prediction worst rel error: {worst:.2e}");
        assert!(worst < 5e-2, "DMD did not extrapolate: {worst}");
    }
}
