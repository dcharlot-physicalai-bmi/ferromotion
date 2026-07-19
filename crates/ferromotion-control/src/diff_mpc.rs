//! **Differentiable MPC** (Amos, Rodriguez, Sacks, Boots & Kolter, NeurIPS 2018) — a finite-horizon
//! linear-quadratic MPC whose optimal action is *differentiable* with respect to its cost and reference,
//! turning the controller into a trainable layer. A finite-horizon LQ-MPC is exactly an equality-
//! constrained QP (quadratic stage costs, linear dynamics), so this is a thin, exact wrapper over
//! [`crate::diff_qp`]: solve the QP for the control sequence, and get `∂(loss)/∂(reference, cost)` from the
//! same KKT adjoint — one linear solve, no unrolling.
//!
//! Here the learnable parameter demonstrated is the tracking **reference** `x_ref` (which enters the QP's
//! linear cost). Given a downstream loss's cotangent on the first action `u₀`, the reference gradient is
//! recovered exactly and checked against finite differences. Pure `nalgebra` → WASM-clean.

use crate::diff_qp::{diff_eq_qp, solve_eq_qp, QpSolution};
use nalgebra::{DMatrix, DVector};

/// A finite-horizon LQ-MPC for `x⁺ = A x + B u`, stage cost `½(x−x_ref)ᵀQ(x−x_ref) + ½uᵀRu`, terminal
/// cost `Qf`.
#[derive(Clone, Debug)]
pub struct DiffMpc {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub qf: DMatrix<f64>,
    pub horizon: usize,
}

impl DiffMpc {
    fn n(&self) -> usize {
        self.a.nrows()
    }
    fn m(&self) -> usize {
        self.b.ncols()
    }

    /// Assemble the MPC as an equality-constrained QP over `z = [x₁…x_N, u₀…u_{N−1}]` (x₀ fixed):
    /// returns `(Q_qp, q_qp, A_eq, b_eq)`.
    fn build_qp(&self, x0: &DVector<f64>, x_ref: &DVector<f64>) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>) {
        let (n, m, big_n) = (self.n(), self.m(), self.horizon);
        let xsz = big_n * n;
        let usz = big_n * m;
        let nz = xsz + usz;
        let ix = |t: usize| (t - 1) * n; // x_t, t = 1..=N
        let iu = |t: usize| xsz + t * m; // u_t, t = 0..N-1

        let mut q_qp = DMatrix::zeros(nz, nz);
        let mut q_lin = DVector::zeros(nz);
        for t in 1..=big_n {
            let stage = if t == big_n { &self.qf } else { &self.q };
            q_qp.view_mut((ix(t), ix(t)), (n, n)).copy_from(stage);
            q_lin.rows_mut(ix(t), n).copy_from(&(-(stage * x_ref)));
        }
        for t in 0..big_n {
            q_qp.view_mut((iu(t), iu(t)), (m, m)).copy_from(&self.r);
        }

        // dynamics equalities: x_{t+1} − A x_t − B u_t = 0 ; for t=0, x₀ is fixed ⇒ b = A x₀
        let mut a_eq = DMatrix::zeros(big_n * n, nz);
        let mut b_eq = DVector::zeros(big_n * n);
        for t in 0..big_n {
            let row = t * n;
            a_eq.view_mut((row, ix(t + 1)), (n, n)).copy_from(&DMatrix::identity(n, n)); // +I x_{t+1}
            if t >= 1 {
                a_eq.view_mut((row, ix(t)), (n, n)).copy_from(&(-&self.a)); // −A x_t
            } else {
                b_eq.rows_mut(row, n).copy_from(&(&self.a * x0)); // A x₀
            }
            a_eq.view_mut((row, iu(t)), (n, m)).copy_from(&(-&self.b)); // −B u_t
        }
        (q_qp, q_lin, a_eq, b_eq)
    }

    /// Solve the MPC QP; returns the full QP solution and the first action `u₀`.
    pub fn solve(&self, x0: &DVector<f64>, x_ref: &DVector<f64>) -> (QpSolution, DVector<f64>) {
        let (q_qp, q_lin, a_eq, b_eq) = self.build_qp(x0, x_ref);
        let sol = solve_eq_qp(&q_qp, &q_lin, &a_eq, &b_eq);
        let iu0 = self.horizon * self.n();
        let u0 = sol.z.rows(iu0, self.m()).into_owned();
        (sol, u0)
    }

    /// The gradient of a downstream scalar loss w.r.t. the **reference** `x_ref`, given the loss's cotangent
    /// on the first action, `dl_du0 = ∂L/∂u₀`. Uses the diff-QP KKT adjoint, then maps `∂L/∂q_qp → ∂L/∂x_ref`
    /// (the reference enters only the linear cost, as `−Q·x_ref` on each state block).
    pub fn reference_gradient(&self, x0: &DVector<f64>, x_ref: &DVector<f64>, dl_du0: &DVector<f64>) -> DVector<f64> {
        let (n, m, big_n) = (self.n(), self.m(), self.horizon);
        let (q_qp, q_lin, a_eq, b_eq) = self.build_qp(x0, x_ref);
        let sol = solve_eq_qp(&q_qp, &q_lin, &a_eq, &b_eq);
        // cotangent on the full decision: nonzero only on the u₀ block
        let nz = sol.z.len();
        let mut dl_dz = DVector::zeros(nz);
        let iu0 = big_n * n;
        dl_dz.rows_mut(iu0, m).copy_from(dl_du0);
        let grads = diff_eq_qp(&q_qp, &a_eq, &sol, &dl_dz);
        // q_lin block for x_t = −(stage·x_ref) ⇒ ∂q_lin_block/∂x_ref = −stage ⇒ ∂L/∂x_ref += −stageᵀ·(∂L/∂q_lin_block)
        let ix = |t: usize| (t - 1) * n;
        let mut d_ref = DVector::zeros(n);
        for t in 1..=big_n {
            let stage = if t == big_n { &self.qf } else { &self.q };
            let block = grads.d_q_vec.rows(ix(t), n).into_owned();
            d_ref -= stage.transpose() * block;
        }
        d_ref
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dm(r: usize, c: usize, v: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(r, c, v)
    }
    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    fn mpc() -> DiffMpc {
        let dt = 0.2;
        DiffMpc {
            a: dm(2, 2, &[1.0, dt, 0.0, 1.0]),
            b: dm(2, 1, &[0.5 * dt * dt, dt]),
            q: DMatrix::from_diagonal(&dv(&[2.0, 0.2])),
            r: DMatrix::from_diagonal(&dv(&[0.05])),
            qf: DMatrix::from_diagonal(&dv(&[10.0, 10.0])),
            horizon: 8,
        }
    }

    #[test]
    fn the_mpc_drives_the_state_toward_the_reference() {
        let m = mpc();
        let x0 = dv(&[0.0, 0.0]);
        let x_ref = dv(&[1.0, 0.0]);
        let (sol, u0) = m.solve(&x0, &x_ref);
        // the terminal predicted state should be near the reference
        let xn = sol.z.rows((m.horizon - 1) * m.n(), m.n()).into_owned();
        assert!((&xn - &x_ref).norm() < 0.15, "terminal state {xn} should approach reference {x_ref}");
        // first action pushes in +x (toward the reference ahead)
        assert!(u0[0] > 0.0, "first action should accelerate toward the reference: {u0}");
    }

    #[test]
    fn the_reference_gradient_matches_finite_differences() {
        // THE ORACLE. For a loss L = ½‖u₀ − u_target‖², ∂L/∂x_ref from the KKT adjoint must match central
        // finite differences of a full MPC re-solve — the differentiable-layer guarantee.
        let m = mpc();
        let x0 = dv(&[0.1, -0.2]);
        let x_ref = dv(&[1.0, 0.3]);
        let u_target = dv(&[0.4]);

        let loss = |xr: &DVector<f64>| {
            let (_s, u0) = m.solve(&x0, xr);
            0.5 * (&u0 - &u_target).norm_squared()
        };
        // ∂L/∂u₀ = (u₀ − u_target)
        let (_s, u0) = m.solve(&x0, &x_ref);
        let dl_du0 = &u0 - &u_target;
        let grad = m.reference_gradient(&x0, &x_ref, &dl_du0);

        let eps = 1e-6;
        for i in 0..2 {
            let mut xp = x_ref.clone();
            let mut xm = x_ref.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let fd = (loss(&xp) - loss(&xm)) / (2.0 * eps);
            assert!((grad[i] - fd).abs() < 1e-6, "∂L/∂x_ref[{i}] {} vs fd {fd}", grad[i]);
        }
    }
}
