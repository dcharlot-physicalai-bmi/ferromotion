//! ReLU-QP — a dense convex QP solver as unrolled OSQP-style ADMM (Bishop, Tracy, Manchester 2023).
//!
//! Solves `min ½ xᵀP x + qᵀx  s.t.  l ≤ A x ≤ u`. The "ReLU network" view: with the KKT system
//! factored **once**, every ADMM iteration is fixed matmuls plus a clamp — a ReLU-style projection
//! onto the box `[l, u]`. This is the CPU reference; the GPU/Ferric port comes later. Pure Rust +
//! `nalgebra` (LU on the cached, symmetric-indefinite KKT) → WASM-clean and fully deterministic.

use nalgebra::{DMatrix, DVector};

/// Dense OSQP-style ADMM QP solver with a cached KKT factorization.
#[derive(Clone, Debug)]
pub struct ReluQp {
    /// Constraint (primal) penalty ρ > 0.
    pub rho: f64,
    /// Proximal regularization σ > 0 (keeps the (1,1) KKT block `P + σI` positive definite).
    pub sigma: f64,
    /// Maximum ADMM iterations (the solver stops earlier once primal/dual residuals are tiny).
    pub iters: usize,
}

impl Default for ReluQp {
    fn default() -> Self {
        Self { rho: 1.0, sigma: 1e-6, iters: 4000 }
    }
}

impl ReluQp {
    pub fn new(rho: f64, sigma: f64, iters: usize) -> Self {
        Self { rho, sigma, iters }
    }

    /// Solve `min ½ xᵀP x + qᵀx  s.t.  l ≤ A x ≤ u`; returns the primal minimizer `x` (length `n`).
    ///
    /// `P` is symmetric PSD (`n×n`), `A` is `m×n`, and `l`/`u` are the elementwise bounds on `A x`
    /// (each length `m`; use a large-magnitude bound for a one-sided constraint).
    pub fn solve(
        &self,
        p: &DMatrix<f64>,
        q: &[f64],
        a: &DMatrix<f64>,
        l: &[f64],
        u: &[f64],
    ) -> Vec<f64> {
        let n = p.ncols();
        let m = a.nrows();
        let qv = DVector::from_row_slice(q);
        let at = a.transpose(); // n×m, reused for the KKT block and the dual residual.

        // Cache the KKT matrix [[P+σI, Aᵀ], [A, −1/ρ I]] once and factor it. It is symmetric but
        // indefinite (the −1/ρ I block), so LU rather than Cholesky.
        let mut kkt = DMatrix::zeros(n + m, n + m);
        let mut psig = p.clone();
        for i in 0..n {
            psig[(i, i)] += self.sigma;
        }
        kkt.view_mut((0, 0), (n, n)).copy_from(&psig);
        kkt.view_mut((0, n), (n, m)).copy_from(&at);
        kkt.view_mut((n, 0), (m, n)).copy_from(a);
        for i in 0..m {
            kkt[(n + i, n + i)] = -1.0 / self.rho;
        }
        let lu = kkt.lu();

        // ADMM state: primal x, constraint slack z ≈ A x, scaled dual y.
        let mut x = DVector::<f64>::zeros(n);
        let mut z = DVector::<f64>::zeros(m);
        let mut y = DVector::<f64>::zeros(m);
        let mut rhs = DVector::<f64>::zeros(n + m);

        for _ in 0..self.iters {
            // (x, ν) ← KKT⁻¹ · [σx − q; z − y/ρ] via the cached factor.
            for i in 0..n {
                rhs[i] = self.sigma * x[i] - qv[i];
            }
            for i in 0..m {
                rhs[n + i] = z[i] - y[i] / self.rho;
            }
            let sol = lu.solve(&rhs).expect("KKT factorization is nonsingular");
            x = sol.rows(0, n).into_owned();

            // z ← clamp(A x + y/ρ, l, u)  (the ReLU/box projection);  y ← y + ρ(A x − z).
            let ax = a * &x;
            let mut r_prim = 0.0f64;
            for i in 0..m {
                let znew = (ax[i] + y[i] / self.rho).clamp(l[i], u[i]);
                y[i] += self.rho * (ax[i] - znew);
                z[i] = znew;
                r_prim = r_prim.max((ax[i] - znew).abs());
            }

            // Dual residual ‖P x + q + Aᵀ y‖_∞ (stationarity of the KKT system).
            let dres = p * &x + &qv + &at * &y;
            if r_prim < 1e-9 && dres.amax() < 1e-9 {
                break;
            }
        }

        x.as_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::solve_qp;

    /// A small, fixed QP whose unconstrained optimum lies outside the feasible set, so the box
    /// bounds and the extra inequality all bind: `min ½xᵀPx + qᵀx  s.t.  l ≤ A x ≤ u`.
    ///
    /// Rows of `A`: the three box rows (`I`) plus one summed inequality `[1,1,1]`.
    fn problem() -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>, Vec<f64>, Vec<f64>) {
        let p = DMatrix::from_row_slice(
            3,
            3,
            &[2.0, 0.2, 0.0, 0.2, 2.0, 0.1, 0.0, 0.1, 2.0],
        );
        let q = vec![-1.0, -3.0, 1.0];
        let a = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 0.0, 0.0, // x0
                0.0, 1.0, 0.0, // x1
                0.0, 0.0, 1.0, // x2
                1.0, 1.0, 1.0, // x0 + x1 + x2
            ],
        );
        let l = vec![-0.5, -0.5, -0.5, -10.0]; // sum lower bound effectively −∞
        let u = vec![0.5, 0.5, 0.5, 0.3];
        (p, q, a, l, u)
    }

    /// clarabel reference: encode `l ≤ A x ≤ u` as `[A; −A] x ≤ [u; −l]`.
    fn reference(p: &DMatrix<f64>, q: &[f64], a: &DMatrix<f64>, l: &[f64], u: &[f64]) -> Vec<f64> {
        let m = a.nrows();
        let n = a.ncols();
        let mut a2 = DMatrix::zeros(2 * m, n);
        a2.view_mut((0, 0), (m, n)).copy_from(a);
        a2.view_mut((m, 0), (m, n)).copy_from(&(-a));
        let mut b = u.to_vec();
        b.extend(l.iter().map(|v| -v));
        solve_qp(p, q, &a2, &b)
    }

    #[test]
    fn matches_clarabel_and_is_feasible() {
        let (p, q, a, l, u) = problem();
        let x = ReluQp::new(1.0, 1e-6, 20_000).solve(&p, &q, &a, &l, &u);
        let x_ref = reference(&p, &q, &a, &l, &u);

        // Match the clarabel optimum to ~1e-3.
        for i in 0..3 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-3,
                "x[{i}] = {} vs clarabel {}",
                x[i],
                x_ref[i]
            );
        }

        // Constraints satisfied: l ≤ A x ≤ u within tolerance.
        let ax = &a * DVector::from_row_slice(&x);
        for i in 0..a.nrows() {
            assert!(
                ax[i] >= l[i] - 1e-6 && ax[i] <= u[i] + 1e-6,
                "row {i}: {} not in [{}, {}]",
                ax[i],
                l[i],
                u[i]
            );
        }

        // The interesting bounds should actually be active (this is a constrained optimum).
        assert!((ax[1] - u[1]).abs() < 1e-3, "x1 box bound expected active, got {}", ax[1]);
        assert!((ax[3] - u[3]).abs() < 1e-3, "sum inequality expected active, got {}", ax[3]);
    }

    #[test]
    fn is_deterministic() {
        let (p, q, a, l, u) = problem();
        let solver = ReluQp::new(1.0, 1e-6, 20_000);
        let x1 = solver.solve(&p, &q, &a, &l, &u);
        let x2 = solver.solve(&p, &q, &a, &l, &u);
        assert_eq!(x1, x2, "same problem produced different output");
    }

    #[test]
    fn recovers_unconstrained_optimum_when_bounds_are_slack() {
        // With very loose bounds the solution is the unconstrained minimizer x* = −P⁻¹q.
        let p = DMatrix::from_row_slice(2, 2, &[3.0, 0.5, 0.5, 2.0]);
        let q = vec![-2.0, 1.0];
        let a = DMatrix::identity(2, 2);
        let l = vec![-100.0, -100.0];
        let u = vec![100.0, 100.0];
        let x = ReluQp::new(1.0, 1e-6, 20_000).solve(&p, &q, &a, &l, &u);
        let x_star = -(p.clone().try_inverse().unwrap()) * DVector::from_row_slice(&q);
        assert!((x[0] - x_star[0]).abs() < 1e-4 && (x[1] - x_star[1]).abs() < 1e-4, "x = {x:?}");
    }
}
