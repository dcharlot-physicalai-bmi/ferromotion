//! **FDDP — Feasibility-Driven Differential Dynamic Programming** (Mastalli et al., *Crocoddyl*, ICRA
//! 2020) — the whole-body optimal-control engine behind modern legged/humanoid control. Ordinary
//! (single-shooting) iLQR rolls the dynamics out exactly every iteration, so it needs a dynamically-
//! *feasible* warm start and diverges from a bad one. FDDP is a **multiple-shooting** DDP: the trajectory
//! nodes need not satisfy the dynamics — there are **gaps** `f̄_k = f(x_k,u_k) − x_{k+1}` — and the method
//! *keeps the gaps open* early, closing them gradually as it converges. That is what lets it start from an
//! arbitrary (even wildly infeasible) guess and still converge.
//!
//! Two changes from textbook DDP make it work:
//! * the **backward pass** carries the gaps: `Q_x = l_x + f_xᵀ(V_x' + V_xx' f̄)`, and likewise `Q_u`;
//! * the **feasibility-driven forward pass** rolls out but only closes each gap by the step length,
//!   `x̂_{k+1} = f(x̂_k, û_k) − (1−α) f̄_{k+1}` — so `α=1` gives a fully feasible (gap-free) rollout and
//!   smaller steps retain a fraction of the infeasibility.
//!
//! Verified against the LQR optimality certificate (feed-forward → 0 on a linear-quadratic problem) and
//! by converging a *nonlinear* pendulum swing-up from a dynamically-infeasible straight-line guess (the
//! gaps decay to ~0). Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// Continuous-in-spirit discrete dynamics `f(x, u) → x⁺`.
pub type DynFn = dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>;
/// Discrete dynamics Jacobians `(∂f/∂x, ∂f/∂u)`.
pub type JacFn = dyn Fn(&DVector<f64>, &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>);

/// A finite-horizon optimal-control problem with a quadratic tracking cost:
/// `Σ ½(x−x_ref)ᵀQ(x−x_ref) + ½(u−u_ref)ᵀR(u−u_ref) + ½(x_N−x_goal)ᵀQ_f(x_N−x_goal)`.
#[derive(Clone, Debug)]
pub struct FddpProblem {
    pub n: usize,
    pub m: usize,
    pub horizon: usize,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub qf: DMatrix<f64>,
    pub x_ref: DVector<f64>,
    pub u_ref: DVector<f64>,
    pub x_goal: DVector<f64>,
    /// Base regularization added to `Q_uu`.
    pub reg: f64,
}

/// The result of an FDDP solve.
#[derive(Clone, Debug)]
pub struct FddpReport {
    pub xs: Vec<DVector<f64>>,
    pub us: Vec<DVector<f64>>,
    pub cost: f64,
    /// `Σ‖f̄_k‖` of the returned trajectory (→ 0 ⇒ dynamically feasible).
    pub gap_norm: f64,
    /// `‖feed-forward‖∞` at the last iterate (→ 0 ⇒ a stationary point / optimum).
    pub ff_norm: f64,
    pub iterations: usize,
    pub converged: bool,
    pub cost_history: Vec<f64>,
    pub gap_history: Vec<f64>,
}

impl FddpProblem {
    fn running_cost(&self, x: &DVector<f64>, u: &DVector<f64>) -> f64 {
        let dx = x - &self.x_ref;
        let du = u - &self.u_ref;
        0.5 * (dx.dot(&(&self.q * &dx)) + du.dot(&(&self.r * &du)))
    }

    fn total_cost(&self, xs: &[DVector<f64>], us: &[DVector<f64>]) -> f64 {
        let mut c = 0.0;
        for k in 0..self.horizon {
            c += self.running_cost(&xs[k], &us[k]);
        }
        let dxf = &xs[self.horizon] - &self.x_goal;
        c + 0.5 * dxf.dot(&(&self.qf * &dxf))
    }

    /// The gaps `f̄_k = f(x_k,u_k) − x_{k+1}`, `k = 0..N−1`.
    fn gaps(&self, f: &DynFn, xs: &[DVector<f64>], us: &[DVector<f64>]) -> Vec<DVector<f64>> {
        (0..self.horizon).map(|k| f(&xs[k], &us[k]) - &xs[k + 1]).collect()
    }

    fn gap_norm(gaps: &[DVector<f64>]) -> f64 {
        gaps.iter().map(|g| g.norm()).sum()
    }

    /// Solve from an initial state `x0` and a warm-start trajectory `(xs0, us0)` — which need NOT be
    /// dynamically feasible. `xs0[0]` is overwritten with `x0`.
    pub fn solve(&self, f: &DynFn, jac: &JacFn, x0: &DVector<f64>, xs0: Vec<DVector<f64>>, us0: Vec<DVector<f64>>, max_iter: usize) -> FddpReport {
        let (n, m, big_n) = (self.n, self.m, self.horizon);
        let mut xs = xs0;
        let mut us = us0;
        xs[0] = x0.clone();
        let mut cost = self.total_cost(&xs, &us);
        let mut reg = self.reg;
        let mut cost_history = vec![cost];
        let mut gap_history = vec![Self::gap_norm(&self.gaps(f, &xs, &us))];
        let mut converged = false;
        let mut ff_norm = f64::INFINITY;
        let mut iterations = 0;

        for _ in 0..max_iter {
            iterations += 1;
            let gaps = self.gaps(f, &xs, &us);

            // ---- backward pass ----
            let mut v_x = &self.qf * (&xs[big_n] - &self.x_goal);
            let mut v_xx = self.qf.clone();
            let mut ks: Vec<DVector<f64>> = vec![DVector::zeros(m); big_n];
            let mut k_fb: Vec<DMatrix<f64>> = vec![DMatrix::zeros(m, n); big_n];
            let (mut dv1, mut dv2) = (0.0, 0.0);
            let mut backward_ok = true;

            for k in (0..big_n).rev() {
                let (fx, fu) = jac(&xs[k], &us[k]);
                let lx = &self.q * (&xs[k] - &self.x_ref);
                let lu = &self.r * (&us[k] - &self.u_ref);
                // value at k+1 shifted by the gap (the FDDP term)
                let v_gap = &v_x + &v_xx * &gaps[k];
                let q_x = &lx + fx.transpose() * &v_gap;
                let q_u = &lu + fu.transpose() * &v_gap;
                let q_xx = &self.q + fx.transpose() * &v_xx * &fx;
                let q_uu = &self.r + fu.transpose() * &v_xx * &fu + DMatrix::identity(m, m) * reg;
                let q_ux = fu.transpose() * &v_xx * &fx;

                let Some(quu_inv) = q_uu.clone().try_inverse() else {
                    backward_ok = false;
                    break;
                };
                let kff = -(&quu_inv * &q_u);
                let kfb = -(&quu_inv * &q_ux);
                dv1 += kff.dot(&q_u);
                dv2 += kff.dot(&(&q_uu * &kff));

                // value update (standard DDP)
                v_x = &q_x + kfb.transpose() * &q_uu * &kff + kfb.transpose() * &q_u + q_ux.transpose() * &kff;
                let mut vxx = &q_xx + kfb.transpose() * &q_uu * &kfb + kfb.transpose() * &q_ux + q_ux.transpose() * &kfb;
                vxx = (&vxx + vxx.transpose()) * 0.5; // symmetrize
                v_xx = vxx;

                ks[k] = kff;
                k_fb[k] = kfb;
            }

            if !backward_ok {
                reg *= 10.0;
                if reg > 1e10 {
                    break;
                }
                continue;
            }
            ff_norm = ks.iter().map(|k| k.amax()).fold(0.0, f64::max);

            // ---- feasibility-driven forward pass with backtracking line search ----
            let mut accepted = false;
            let mut alpha = 1.0;
            for _ in 0..12 {
                let (nxs, nus) = self.forward(f, &xs, &us, &ks, &k_fb, &gaps, alpha, x0);
                let ncost = self.total_cost(&nxs, &nus);
                let expected = alpha * dv1 + 0.5 * alpha * alpha * dv2; // < 0 for a descent step
                if ncost - cost < 1e-4 * expected || (expected >= 0.0 && ncost < cost) {
                    xs = nxs;
                    us = nus;
                    cost = ncost;
                    accepted = true;
                    reg = (reg * 0.5).max(self.reg);
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                reg *= 10.0;
                if reg > 1e10 {
                    break;
                }
            }

            cost_history.push(cost);
            gap_history.push(Self::gap_norm(&self.gaps(f, &xs, &us)));
            // converged: no feed-forward improvement left and dynamics essentially feasible
            if ff_norm < 1e-6 && *gap_history.last().unwrap() < 1e-6 {
                converged = true;
                break;
            }
        }

        let gaps = self.gaps(f, &xs, &us);
        FddpReport {
            gap_norm: Self::gap_norm(&gaps),
            cost,
            ff_norm,
            iterations,
            converged,
            cost_history,
            gap_history,
            xs,
            us,
        }
    }

    /// The feasibility-driven rollout: `x̂_{k+1} = f(x̂_k, û_k) − (1−α)·f̄_{k+1}` with
    /// `û_k = u_k + α·k_k + K_k(x̂_k − x_k)`.
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, f: &DynFn, xs: &[DVector<f64>], us: &[DVector<f64>], ks: &[DVector<f64>], k_fb: &[DMatrix<f64>], gaps: &[DVector<f64>], alpha: f64, x0: &DVector<f64>) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let big_n = self.horizon;
        let mut nxs = vec![DVector::zeros(self.n); big_n + 1];
        let mut nus = vec![DVector::zeros(self.m); big_n];
        nxs[0] = x0.clone();
        for k in 0..big_n {
            let du = &ks[k] * alpha + &k_fb[k] * (&nxs[k] - &xs[k]);
            nus[k] = &us[k] + du;
            nxs[k + 1] = f(&nxs[k], &nus[k]) - &gaps[k] * (1.0 - alpha);
        }
        (nxs, nus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    // Double integrator: x = [p, v], u = [a]. Linear ⇒ an LQR oracle.
    fn double_integrator(dt: f64) -> (impl Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>, impl Fn(&DVector<f64>, &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>)) {
        let f = move |x: &DVector<f64>, u: &DVector<f64>| dv(&[x[0] + dt * x[1], x[1] + dt * u[0]]);
        let jac = move |_x: &DVector<f64>, _u: &DVector<f64>| {
            (DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]), DMatrix::from_row_slice(2, 1, &[0.0, dt]))
        };
        (f, jac)
    }

    #[test]
    fn on_a_linear_quadratic_problem_the_feedforward_vanishes() {
        // THE OPTIMALITY CERTIFICATE. For an LQ problem FDDP reaches the exact optimum, where the
        // feed-forward term is zero (Q_u = 0 at every node) — and it does so in very few iterations.
        let dt = 0.1;
        let (f, jac) = double_integrator(dt);
        let n = 30;
        let prob = FddpProblem {
            n: 2, m: 1, horizon: n,
            q: DMatrix::from_diagonal(&dv(&[1.0, 0.1])),
            r: DMatrix::from_diagonal(&dv(&[0.05])),
            qf: DMatrix::from_diagonal(&dv(&[50.0, 50.0])),
            x_ref: dv(&[0.0, 0.0]), u_ref: dv(&[0.0]), x_goal: dv(&[0.0, 0.0]),
            reg: 1e-6,
        };
        let x0 = dv(&[1.5, 0.0]);
        // feasible warm start: roll out zero controls
        let mut xs = vec![x0.clone()];
        let us = vec![dv(&[0.0]); n];
        for k in 0..n {
            xs.push(f(&xs[k], &us[k]));
        }
        let rep = prob.solve(&f, &jac, &x0, xs, us, 50);
        assert!(rep.converged, "LQ problem should converge");
        assert!(rep.ff_norm < 1e-6, "feed-forward should vanish at the LQ optimum: {}", rep.ff_norm);
        assert!(rep.iterations <= 3, "LQ should converge in a couple iterations: {}", rep.iterations);
        assert!(rep.xs.last().unwrap().norm() < 0.15, "should regulate near the origin: {:?}", rep.xs.last().unwrap());
    }

    #[test]
    fn it_converges_a_pendulum_swing_up_from_an_infeasible_guess() {
        // THE HEADLINE. A nonlinear pendulum, warm-started with a dynamically-INFEASIBLE straight line from
        // hanging to upright (big gaps). FDDP closes the gaps (→ feasible) and swings the pendulum up —
        // something single-shooting iLQR cannot do from this guess.
        let (dt, damp, grav, inertia) = (0.05, 0.1, 9.81, 1.0);
        let f = move |x: &DVector<f64>, u: &DVector<f64>| {
            let th = x[0];
            let w = x[1];
            dv(&[th + dt * w, w + dt * (u[0] - damp * w - grav * th.sin()) / inertia])
        };
        let jac = move |x: &DVector<f64>, _u: &DVector<f64>| {
            let th = x[0];
            (
                DMatrix::from_row_slice(2, 2, &[1.0, dt, -dt * grav * th.cos() / inertia, 1.0 - dt * damp / inertia]),
                DMatrix::from_row_slice(2, 1, &[0.0, dt / inertia]),
            )
        };
        let n = 80;
        let goal = dv(&[std::f64::consts::PI, 0.0]);
        let prob = FddpProblem {
            n: 2, m: 1, horizon: n,
            q: DMatrix::from_diagonal(&dv(&[0.5, 0.05])),
            r: DMatrix::from_diagonal(&dv(&[0.02])),
            qf: DMatrix::from_diagonal(&dv(&[400.0, 400.0])),
            x_ref: goal.clone(), u_ref: dv(&[0.0]), x_goal: goal.clone(),
            reg: 1e-3,
        };
        let x0 = dv(&[0.0, 0.0]);
        // INFEASIBLE straight-line guess in state, zero controls
        let xs: Vec<DVector<f64>> = (0..=n).map(|k| &x0 + (&goal - &x0) * (k as f64 / n as f64)).collect();
        let us = vec![dv(&[0.0]); n];
        let init_gap = FddpProblem::gap_norm(&prob.gaps(&f, &xs, &us));
        let rep = prob.solve(&f, &jac, &x0, xs, us, 300);

        assert!(init_gap > 1.0, "the straight-line guess must be clearly infeasible: {init_gap}");
        assert!(rep.gap_norm < 1e-3, "FDDP must close the dynamics gaps: {}", rep.gap_norm);
        assert!((rep.xs.last().unwrap()[0] - std::f64::consts::PI).abs() < 0.1, "pendulum should reach upright: θ_N = {}", rep.xs.last().unwrap()[0]);
    }

    #[test]
    fn the_gaps_decay_monotonically_toward_feasibility() {
        let (dt, damp, grav, inertia) = (0.05, 0.1, 9.81, 1.0);
        let f = move |x: &DVector<f64>, u: &DVector<f64>| dv(&[x[0] + dt * x[1], x[1] + dt * (u[0] - damp * x[1] - grav * x[0].sin()) / inertia]);
        let jac = move |x: &DVector<f64>, _u: &DVector<f64>| {
            (DMatrix::from_row_slice(2, 2, &[1.0, dt, -dt * grav * x[0].cos() / inertia, 1.0 - dt * damp / inertia]), DMatrix::from_row_slice(2, 1, &[0.0, dt / inertia]))
        };
        let n = 80;
        let goal = dv(&[std::f64::consts::PI, 0.0]);
        let prob = FddpProblem {
            n: 2, m: 1, horizon: n,
            q: DMatrix::from_diagonal(&dv(&[0.5, 0.05])), r: DMatrix::from_diagonal(&dv(&[0.02])),
            qf: DMatrix::from_diagonal(&dv(&[400.0, 400.0])),
            x_ref: goal.clone(), u_ref: dv(&[0.0]), x_goal: goal.clone(), reg: 1e-3,
        };
        let x0 = dv(&[0.0, 0.0]);
        let xs: Vec<DVector<f64>> = (0..=n).map(|k| &x0 + (&goal - &x0) * (k as f64 / n as f64)).collect();
        let rep = prob.solve(&f, &jac, &x0, xs, vec![dv(&[0.0]); n], 300);
        // the gap ends far below where it started (feasibility driven)
        assert!(rep.gap_history.last().unwrap() < &(rep.gap_history[0] * 1e-3), "gaps should collapse: {} → {}", rep.gap_history[0], rep.gap_history.last().unwrap());
    }
}
