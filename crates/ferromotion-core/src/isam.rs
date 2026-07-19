//! **iSAM — Incremental Smoothing and Mapping** (Kaess, Ranganathan & Dellaert, T-RO 2008): keep a
//! factor-graph least-squares estimate up to date as measurements stream in, *without* refactoring from
//! scratch each step. Where [`crate::solve_factor_graph`] is a **batch** Gauss–Newton re-solve, iSAM keeps
//! the **square-root information factor** `R` (upper-triangular, `RᵀR = AᵀA`) and, when a new measurement
//! row arrives, restores the triangular structure with a few **Givens rotations** — touching only the
//! affected columns instead of re-eliminating the whole graph. This is the difference between a demo and a
//! real-time SLAM/VIO backend that stays fast as the graph grows.
//!
//! This implements that incremental-QR core: add rows online, read off the current least-squares estimate
//! by back-substitution, and track the residual — all bit-for-bit consistent with a batch normal-equations
//! solve (the oracle). (iSAM2's Bayes tree adds fluid relinearization + incremental variable reordering on
//! top; scoped out here, as IMU propagation is for [`crate::Msckf`].) Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// An incremental linear least-squares solver in square-root information form: maintains an upper-
/// triangular `R` and right-hand side `d` such that the estimate solves `R x = d` and
/// `min_x ‖A x − b‖²` over all rows added so far.
#[derive(Clone, Debug)]
pub struct IncrementalLeastSquares {
    n: usize,
    r: DMatrix<f64>,
    d: DVector<f64>,
    residual_sq: f64,
}

impl IncrementalLeastSquares {
    /// An empty problem over `n` variables.
    pub fn new(n: usize) -> Self {
        IncrementalLeastSquares { n, r: DMatrix::zeros(n, n), d: DVector::zeros(n), residual_sq: 0.0 }
    }

    /// Add one weighted measurement row `a·x ≈ b` (pre-whiten by `√Ω` for a covariance-weighted factor),
    /// restoring the triangular factor with Givens rotations. `O(n²)` worst case, but only the columns where
    /// `a` is nonzero from its first nonzero on are touched.
    pub fn add_row(&mut self, a_in: &DVector<f64>, b_in: f64) {
        let mut a = a_in.clone();
        let mut b = b_in;
        for j in 0..self.n {
            if a[j].abs() < 1e-300 {
                continue;
            }
            if self.r[(j, j)].abs() < 1e-300 {
                // empty pivot: this row becomes R's j-th row
                for k in j..self.n {
                    self.r[(j, k)] = a[k];
                }
                self.d[j] = b;
                return; // row fully absorbed as a new independent direction (no residual)
            }
            // Givens rotation zeroing a[j] against R[j][j]
            let (rjj, x) = (self.r[(j, j)], a[j]);
            let h = rjj.hypot(x);
            let (c, s) = (rjj / h, x / h);
            for k in j..self.n {
                let (rjk, ak) = (self.r[(j, k)], a[k]);
                self.r[(j, k)] = c * rjk + s * ak;
                a[k] = -s * rjk + c * ak;
            }
            let (dj, bb) = (self.d[j], b);
            self.d[j] = c * dj + s * bb;
            b = -s * dj + c * bb;
        }
        // whatever residual remains after the row is fully zeroed adds to the least-squares cost
        self.residual_sq += b * b;
    }

    /// The current estimate `x = R⁻¹ d` by back-substitution (0 for any as-yet-unconstrained variable).
    pub fn solve(&self) -> DVector<f64> {
        let mut x = DVector::zeros(self.n);
        for i in (0..self.n).rev() {
            if self.r[(i, i)].abs() < 1e-300 {
                continue;
            }
            let mut acc = self.d[i];
            for k in (i + 1)..self.n {
                acc -= self.r[(i, k)] * x[k];
            }
            x[i] = acc / self.r[(i, i)];
        }
        x
    }

    /// The least-squares residual `‖A x − b‖²` accumulated so far (the part orthogonal to the row space).
    pub fn residual_sq(&self) -> f64 {
        self.residual_sq
    }

    /// The square-root information factor `R` (`RᵀR = AᵀA`).
    pub fn r(&self) -> &DMatrix<f64> {
        &self.r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deterministic pseudo-random overdetermined system A x = b (m rows, n cols).
    fn system(m: usize, n: usize) -> (DMatrix<f64>, DVector<f64>) {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let a = DMatrix::from_fn(m, n, |_, _| rng());
        let b = DVector::from_fn(m, |_, _| rng());
        (a, b)
    }

    fn batch_solve(a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
        let ata = a.transpose() * a;
        let atb = a.transpose() * b;
        ata.lu().solve(&atb).unwrap()
    }

    #[test]
    fn the_incremental_estimate_matches_the_batch_solve() {
        // THE ORACLE. Streaming rows one at a time through Givens updates must reproduce the batch
        // normal-equations solution over all rows.
        let (n, m) = (6, 40);
        let (a, b) = system(m, n);
        let mut isam = IncrementalLeastSquares::new(n);
        for i in 0..m {
            isam.add_row(&a.row(i).transpose(), b[i]);
        }
        let x_inc = isam.solve();
        let x_batch = batch_solve(&a, &b);
        assert!((&x_inc - &x_batch).norm() < 1e-9, "incremental vs batch: {}", (&x_inc - &x_batch).norm());
    }

    #[test]
    fn r_is_the_cholesky_factor_of_the_information_matrix() {
        // RᵀR = AᵀA — the incremental factor is exactly the square-root information matrix.
        let (n, m) = (5, 30);
        let (a, b) = system(m, n);
        let mut isam = IncrementalLeastSquares::new(n);
        for i in 0..m {
            isam.add_row(&a.row(i).transpose(), b[i]);
        }
        let rtr = isam.r().transpose() * isam.r();
        let ata = a.transpose() * &a;
        assert!((rtr - ata).abs().max() < 1e-8, "RᵀR ≠ AᵀA: {}", (isam.r().transpose() * isam.r() - a.transpose() * &a).abs().max());
    }

    #[test]
    fn the_estimate_stays_correct_after_every_new_measurement() {
        // Online consistency: after each added row the current estimate equals the batch solve over the
        // rows seen so far (once the system is determined).
        let (n, m) = (4, 25);
        let (a, b) = system(m, n);
        let mut isam = IncrementalLeastSquares::new(n);
        for i in 0..m {
            isam.add_row(&a.row(i).transpose(), b[i]);
            if i + 1 >= n {
                let sub_a = a.rows(0, i + 1).into_owned();
                let sub_b = b.rows(0, i + 1).into_owned();
                let diff = (isam.solve() - batch_solve(&sub_a, &sub_b)).norm();
                assert!(diff < 1e-8, "online estimate diverged at row {i}: {diff}");
            }
        }
    }

    #[test]
    fn the_residual_matches_the_least_squares_optimum() {
        // The accumulated residual equals ‖A x* − b‖² at the optimum.
        let (n, m) = (5, 35);
        let (a, b) = system(m, n);
        let mut isam = IncrementalLeastSquares::new(n);
        for i in 0..m {
            isam.add_row(&a.row(i).transpose(), b[i]);
        }
        let x = isam.solve();
        let true_res = (&a * &x - &b).norm_squared();
        assert!((isam.residual_sq() - true_res).abs() < 1e-7, "residual {} vs ‖Ax−b‖² {true_res}", isam.residual_sq());
    }
}
