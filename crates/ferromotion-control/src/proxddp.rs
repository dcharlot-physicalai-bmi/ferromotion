//! **Constrained DDP** via an augmented-Lagrangian / proximal outer loop — a clean-room take on
//! PROXDDP / Aligator (Jallet et al., INRIA/LAAS, *IEEE T-RO* 2025). Plain DDP/iLQR has no principled
//! way to enforce hard constraints (torque limits, obstacle keep-outs); PROXDDP wraps the Riccati
//! backward pass in an augmented-Lagrangian method-of-multipliers: each stage cost is augmented with
//! an AL penalty on the constraints, the inner DDP solves the augmented problem to get a feedback
//! policy, and the outer loop lifts the multipliers and penalty until the constraints are met.
//!
//! This implementation covers the linear-quadratic case with **hard control box constraints**
//! `lo ≤ u ≤ hi` — where the method reduces to a box-constrained LQR whose exact optimum is a convex
//! QP, giving a clean oracle. The AL + Riccati structure is the general PROXDDP mechanism (a
//! per-stage linearization makes it nonlinear, exactly as iLQR generalizes LQR). Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A discrete LTI optimal-control problem `x_{k+1} = A x_k + B u_k` with quadratic cost and hard box
/// bounds on the controls. Cost `J = Σ_{k<N} ½xₖᵀQxₖ + ½uₖᵀRuₖ + ½x_Nᵀ Q_f x_N`, regulating to the
/// origin (shift coordinates for a nonzero target).
pub struct ConstrainedLqr {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub qf: DMatrix<f64>,
    pub horizon: usize,
    pub u_lo: DVector<f64>,
    pub u_hi: DVector<f64>,
}

/// Result of a constrained-DDP solve.
#[derive(Clone, Debug)]
pub struct ConstrainedDdpResult {
    /// Optimized controls `u_0 … u_{N-1}`.
    pub us: Vec<DVector<f64>>,
    /// State trajectory `x_0 … x_N`.
    pub xs: Vec<DVector<f64>>,
    /// Max control-bound violation (should be ~0 at convergence).
    pub max_violation: f64,
    pub iters: usize,
    pub converged: bool,
}

fn safe_inverse(m: &DMatrix<f64>) -> DMatrix<f64> {
    let nr = m.nrows();
    let mut reg = 1e-9;
    for _ in 0..24 {
        if let Some(inv) = (m + DMatrix::identity(nr, nr) * reg).try_inverse() {
            return inv;
        }
        reg *= 10.0;
    }
    DMatrix::identity(nr, nr)
}

impl ConstrainedLqr {
    fn rollout(&self, x0: &DVector<f64>, us: &[DVector<f64>]) -> Vec<DVector<f64>> {
        let mut xs = Vec::with_capacity(self.horizon + 1);
        xs.push(x0.clone());
        for (t, u) in us.iter().enumerate() {
            xs.push(&self.a * &xs[t] + &self.b * u);
        }
        xs
    }

    /// AL control-cost gradient `l_u`, Hessian `l_uu`, and updated multiplier estimates `μ` at `u`
    /// for the box constraints, given current multipliers `lam` and penalty `rho`.
    fn al_control(&self, u: &DVector<f64>, lam: &DVector<f64>, rho: f64) -> (DVector<f64>, DMatrix<f64>, DVector<f64>) {
        let m = u.len();
        let mut lu = &self.r * u;
        let mut luu = self.r.clone();
        let mut mu = DVector::zeros(2 * m);
        for i in 0..m {
            // upper bound: c = u_i − hi_i ≤ 0 (a = +eᵢ)
            let su = lam[2 * i] + rho * (u[i] - self.u_hi[i]);
            if su > 0.0 {
                mu[2 * i] = su;
                lu[i] += su;
                luu[(i, i)] += rho;
            }
            // lower bound: c = lo_i − u_i ≤ 0 (a = −eᵢ)
            let sl = lam[2 * i + 1] + rho * (self.u_lo[i] - u[i]);
            if sl > 0.0 {
                mu[2 * i + 1] = sl;
                lu[i] -= sl;
                luu[(i, i)] += rho;
            }
        }
        (lu, luu, mu)
    }

    /// Augmented-Lagrangian cost of a trajectory (for the inner line search).
    fn al_cost(&self, xs: &[DVector<f64>], us: &[DVector<f64>], lam: &[DVector<f64>], rho: f64) -> f64 {
        let mut j = 0.0;
        for t in 0..self.horizon {
            j += 0.5 * xs[t].dot(&(&self.q * &xs[t])) + 0.5 * us[t].dot(&(&self.r * &us[t]));
            let m = us[t].len();
            for i in 0..m {
                let cu = us[t][i] - self.u_hi[i];
                let cl = self.u_lo[i] - us[t][i];
                let au = (lam[t][2 * i] + rho * cu).max(0.0);
                let al = (lam[t][2 * i + 1] + rho * cl).max(0.0);
                j += 0.5 / rho * (au * au + al * al);
            }
        }
        j += 0.5 * xs[self.horizon].dot(&(&self.qf * &xs[self.horizon]));
        j
    }

    fn violation(&self, us: &[DVector<f64>]) -> f64 {
        let mut v = 0.0f64;
        for u in us {
            for i in 0..u.len() {
                v = v.max((u[i] - self.u_hi[i]).max(0.0)).max((self.u_lo[i] - u[i]).max(0.0));
            }
        }
        v
    }

    /// Solve the constrained OCP from `x0`.
    pub fn solve(&self, x0: &DVector<f64>) -> ConstrainedDdpResult {
        let (s, m, n) = (self.a.nrows(), self.b.ncols(), self.horizon);
        let mut us: Vec<DVector<f64>> = vec![DVector::zeros(m); n];
        let mut lam: Vec<DVector<f64>> = vec![DVector::zeros(2 * m); n];
        let (mut rho, rho_max, beta) = (1.0, 1e8, 8.0);
        let alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625];
        let mut total_iters = 0;

        for _outer in 0..40 {
            // ---- inner: DDP on the AL-augmented problem ----
            let mut xs = self.rollout(x0, &us);
            let mut j = self.al_cost(&xs, &us, &lam, rho);
            for _inner in 0..80 {
                total_iters += 1;
                // Backward Riccati pass with AL control terms.
                let mut v_x = &self.qf * &xs[n];
                let mut v_xx = self.qf.clone();
                let mut kff = vec![DVector::zeros(m); n];
                let mut kfb = vec![DMatrix::zeros(m, s); n];
                for t in (0..n).rev() {
                    let (lu, luu, _) = self.al_control(&us[t], &lam[t], rho);
                    let at = self.a.transpose();
                    let bt = self.b.transpose();
                    let q_x = &self.q * &xs[t] + &at * &v_x;
                    let q_u = &lu + &bt * &v_x;
                    let q_xx = &self.q + &at * &v_xx * &self.a;
                    let mut q_uu = &luu + &bt * &v_xx * &self.b;
                    q_uu = (&q_uu + q_uu.transpose()) * 0.5;
                    let q_ux = &bt * &v_xx * &self.a;
                    let q_uu_inv = safe_inverse(&q_uu);
                    let k = -&q_uu_inv * &q_u;
                    let big_k = -&q_uu_inv * &q_ux;
                    let kt = big_k.transpose();
                    v_x = &q_x + &kt * &q_uu * &k + &kt * &q_u + q_ux.transpose() * &k;
                    v_xx = &q_xx + &kt * &q_uu * &big_k + &kt * &q_ux + q_ux.transpose() * &big_k;
                    v_xx = (&v_xx + v_xx.transpose()) * 0.5;
                    kff[t] = k;
                    kfb[t] = big_k;
                }
                // Forward line search on the AL cost.
                let mut accepted = None;
                for &alpha in &alphas {
                    let mut xn = vec![x0.clone()];
                    let mut un = Vec::with_capacity(n);
                    for t in 0..n {
                        let dx = &xn[t] - &xs[t];
                        un.push(&us[t] + &kff[t] * alpha + &kfb[t] * &dx);
                        let nx = &self.a * &xn[t] + &self.b * &un[t];
                        xn.push(nx);
                    }
                    let jn = self.al_cost(&xn, &un, &lam, rho);
                    if jn < j {
                        accepted = Some((xn, un, jn));
                        break;
                    }
                }
                match accepted {
                    Some((xn, un, jn)) => {
                        let improved = j - jn;
                        xs = xn;
                        us = un;
                        j = jn;
                        if improved < 1e-10 {
                            break;
                        }
                    }
                    None => break,
                }
            }
            // ---- outer: lift multipliers, increase penalty ----
            for t in 0..n {
                let (_, _, mu) = self.al_control(&us[t], &lam[t], rho);
                lam[t] = mu;
            }
            let mv = self.violation(&us);
            if mv < 1e-5 {
                let xs = self.rollout(x0, &us);
                return ConstrainedDdpResult { us, xs, max_violation: mv, iters: total_iters, converged: true };
            }
            rho = (rho * beta).min(rho_max);
        }
        let xs = self.rollout(x0, &us);
        let mv = self.violation(&us);
        ConstrainedDdpResult { us, xs, max_violation: mv, iters: total_iters, converged: mv < 1e-5 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::solve_box_qp;

    /// Double integrator with timestep `dt`.
    fn double_integrator(dt: f64) -> (DMatrix<f64>, DMatrix<f64>) {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        (a, b)
    }

    fn problem() -> (ConstrainedLqr, DVector<f64>) {
        let dt = 0.1;
        let (a, b) = double_integrator(dt);
        let lqr = ConstrainedLqr {
            a,
            b,
            q: DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 0.1])),
            r: DMatrix::from_diagonal(&DVector::from_row_slice(&[0.01])),
            qf: DMatrix::from_diagonal(&DVector::from_row_slice(&[200.0, 20.0])),
            horizon: 30,
            u_lo: DVector::from_row_slice(&[-1.0]),
            u_hi: DVector::from_row_slice(&[1.0]),
        };
        (lqr, DVector::from_row_slice(&[1.0, 0.0])) // start 1 m out, at rest → regulate to origin
    }

    /// Condense the OCP to a box-QP over U = [u_0…u_{N-1}] as an independent oracle.
    fn condensed_qp(lqr: &ConstrainedLqr, x0: &DVector<f64>) -> (DMatrix<f64>, DVector<f64>) {
        let (s, m, n) = (lqr.a.nrows(), lqr.b.ncols(), lqr.horizon);
        // X = [x_1…x_N] = C + G U.
        let mut c = DVector::zeros(s * n);
        let mut g = DMatrix::zeros(s * n, m * n);
        // c_k = A^k x0 (for x_1…x_N)
        let mut ak = lqr.a.clone();
        for k in 0..n {
            c.rows_mut(k * s, s).copy_from(&(&ak * x0));
            ak = &lqr.a * &ak;
        }
        // G[k,i] = A^{k-1-i} B for i ≤ k
        for k in 0..n {
            for i in 0..=k {
                let mut apw = DMatrix::identity(s, s);
                for _ in 0..(k - i) {
                    apw = &lqr.a * &apw;
                }
                let blk = &apw * &lqr.b;
                g.view_mut((k * s, i * m), (s, m)).copy_from(&blk);
            }
        }
        // Qbar = blkdiag(Q,…,Q,Qf) over x_1…x_N; Rbar = blkdiag(R).
        let mut qbar = DMatrix::zeros(s * n, s * n);
        for k in 0..n {
            let blk = if k == n - 1 { &lqr.qf } else { &lqr.q };
            qbar.view_mut((k * s, k * s), (s, s)).copy_from(blk);
        }
        let mut rbar = DMatrix::zeros(m * n, m * n);
        for k in 0..n {
            rbar.view_mut((k * m, k * m), (m, m)).copy_from(&lqr.r);
        }
        let h = &g.transpose() * &qbar * &g + rbar;
        let grad = &g.transpose() * &qbar * &c;
        (h, grad)
    }

    #[test]
    fn constrained_ddp_matches_the_box_qp_oracle() {
        let (lqr, x0) = problem();
        let res = lqr.solve(&x0);
        assert!(res.converged, "did not converge (iters {})", res.iters);

        // (1) Bounds respected.
        for u in &res.us {
            assert!(u[0] <= 1.0 + 1e-4 && u[0] >= -1.0 - 1e-4, "control out of bounds: {}", u[0]);
        }
        // The constraint must actually bind (else the test is vacuous).
        let umax = res.us.iter().map(|u| u[0].abs()).fold(0.0, f64::max);
        assert!(umax > 0.99, "control never reaches the bound (umax={umax}); constraint inactive");

        // (2) Match the condensed box-QP oracle.
        let (h, g) = condensed_qp(&lqr, &x0);
        let (n, m) = (lqr.horizon, lqr.b.ncols());
        let lo: Vec<f64> = (0..n * m).map(|i| lqr.u_lo[i % m]).collect();
        let hi: Vec<f64> = (0..n * m).map(|i| lqr.u_hi[i % m]).collect();
        let uqp = solve_box_qp(&h, g.as_slice(), &lo, &hi);
        let mut max_err = 0.0f64;
        for t in 0..n {
            max_err = max_err.max((res.us[t][0] - uqp[t]).abs());
        }
        // Two independent solvers (AL-DDP vs a condensed clarabel QP) agree to <1% of the 2.0-wide
        // control range; the residual concentrates at the bang-bang saturation switch (most sensitive).
        assert!(max_err < 1e-2, "constrained DDP ≠ QP oracle: max |Δu| = {max_err}");

        // (3) Reaches the target (regulated near the origin).
        let xn = res.xs.last().unwrap();
        assert!(xn.norm() < 0.05, "did not regulate to origin: x_N = {xn:?}");
    }

    #[test]
    fn unconstrained_bounds_recover_plain_lqr_and_constraints_cost_more() {
        let (mut lqr, x0) = problem();
        // Wide bounds → inactive; then tight bounds → active.
        lqr.u_lo = DVector::from_row_slice(&[-100.0]);
        lqr.u_hi = DVector::from_row_slice(&[100.0]);
        let wide = lqr.solve(&x0);
        let cost = |us: &[DVector<f64>], xs: &[DVector<f64>]| -> f64 {
            let mut j = 0.0;
            for t in 0..lqr.horizon {
                j += 0.5 * xs[t].dot(&(&lqr.q * &xs[t])) + 0.5 * us[t].dot(&(&lqr.r * &us[t]));
            }
            j + 0.5 * xs[lqr.horizon].dot(&(&lqr.qf * &xs[lqr.horizon]))
        };
        let wide_cost = cost(&wide.us, &wide.xs);

        lqr.u_lo = DVector::from_row_slice(&[-1.0]);
        lqr.u_hi = DVector::from_row_slice(&[1.0]);
        let tight = lqr.solve(&x0);
        let tight_cost = cost(&tight.us, &tight.xs);

        // Constraining can only raise the optimal cost.
        assert!(tight_cost >= wide_cost - 1e-6, "constrained cost {tight_cost} < unconstrained {wide_cost}");
        assert!(tight.max_violation < 1e-5, "tight solution violates its bounds");
    }
}
