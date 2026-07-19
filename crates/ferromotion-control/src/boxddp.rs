//! **Box-DDP — control-limited Differential Dynamic Programming** (Tassa, Mansard & Todorov, ICRA 2014).
//! Actuator saturation is universal in embodied AI, and clamping/penalty heuristics converge poorly.
//! Box-DDP fixes it at the source: the DDP backward pass solves a **box-constrained QP** for the control
//! update instead of the unconstrained `δu = −Q_uu⁻¹ Q_u`,
//!
//! ```text
//!   δu_k = argmin ½ δuᵀ Q_uu δu + Q_uᵀ δu   s.t.   u_min − u_k ≤ δu ≤ u_max − u_k
//! ```
//!
//! so the plan exactly respects torque/current limits, and the **feedback gain rows of clamped controls
//! are zeroed** (a saturated actuator carries no feedback). It reduces to ordinary DDP when the limits are
//! slack. Verified against that reduction, box satisfaction, and a min-effort problem where the tight
//! limit forces saturation. Reuses the crate's `solve_box_qp` (clarabel). Pure Rust → WASM-clean.

use crate::qp::solve_box_qp;
use nalgebra::{DMatrix, DVector};

/// Discrete dynamics `f(x, u) → x⁺`.
pub type DynFn = dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>;
/// Dynamics Jacobians `(∂f/∂x, ∂f/∂u)`.
pub type JacFn = dyn Fn(&DVector<f64>, &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>);

/// A control-limited finite-horizon problem with a quadratic tracking cost and box control limits.
#[derive(Clone, Debug)]
pub struct BoxDdpProblem {
    pub n: usize,
    pub m: usize,
    pub horizon: usize,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub qf: DMatrix<f64>,
    pub x_ref: DVector<f64>,
    pub u_ref: DVector<f64>,
    pub x_goal: DVector<f64>,
    pub u_min: DVector<f64>,
    pub u_max: DVector<f64>,
    pub reg: f64,
}

/// The result of a Box-DDP solve.
#[derive(Clone, Debug)]
pub struct BoxDdpReport {
    pub xs: Vec<DVector<f64>>,
    pub us: Vec<DVector<f64>>,
    pub cost: f64,
    pub ff_norm: f64,
    pub iterations: usize,
    pub converged: bool,
    pub cost_history: Vec<f64>,
}

impl BoxDdpProblem {
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

    fn clamp_u(&self, u: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(self.m, (0..self.m).map(|i| u[i].clamp(self.u_min[i], self.u_max[i])))
    }

    /// Single-shooting rollout of the control sequence from `x0` (always dynamically feasible; controls
    /// clamped into the box).
    fn rollout(&self, f: &DynFn, x0: &DVector<f64>, us: &[DVector<f64>]) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let mut xs = vec![x0.clone()];
        let mut cus = Vec::with_capacity(self.horizon);
        for k in 0..self.horizon {
            let u = self.clamp_u(&us[k]);
            xs.push(f(&xs[k], &u));
            cus.push(u);
        }
        (xs, cus)
    }

    /// Solve from an initial state and a control warm start.
    pub fn solve(&self, f: &DynFn, jac: &JacFn, x0: &DVector<f64>, us0: Vec<DVector<f64>>, max_iter: usize) -> BoxDdpReport {
        let (n, m, big_n) = (self.n, self.m, self.horizon);
        let (mut xs, mut us) = self.rollout(f, x0, &us0);
        let mut cost = self.total_cost(&xs, &us);
        let mut reg = self.reg;
        let mut cost_history = vec![cost];
        let mut converged = false;
        let mut ff_norm = f64::INFINITY;
        let mut iterations = 0;

        for _ in 0..max_iter {
            iterations += 1;
            // ---- backward pass with a box-QP feed-forward ----
            let mut v_x = &self.qf * (&xs[big_n] - &self.x_goal);
            let mut v_xx = self.qf.clone();
            let mut ks: Vec<DVector<f64>> = vec![DVector::zeros(m); big_n];
            let mut k_fb: Vec<DMatrix<f64>> = vec![DMatrix::zeros(m, n); big_n];
            let mut ok = true;

            for k in (0..big_n).rev() {
                let (fx, fu) = jac(&xs[k], &us[k]);
                let lx = &self.q * (&xs[k] - &self.x_ref);
                let lu = &self.r * (&us[k] - &self.u_ref);
                let q_x = &lx + fx.transpose() * &v_x;
                let q_u = &lu + fu.transpose() * &v_x;
                let q_xx = &self.q + fx.transpose() * &v_xx * &fx;
                let q_uu = &self.r + fu.transpose() * &v_xx * &fu + DMatrix::identity(m, m) * reg;
                let q_ux = fu.transpose() * &v_xx * &fx;

                // box-QP for the feed-forward: bounds are the box shifted by the current control
                let lo: Vec<f64> = (0..m).map(|i| self.u_min[i] - us[k][i]).collect();
                let hi: Vec<f64> = (0..m).map(|i| self.u_max[i] - us[k][i]).collect();
                let kff = DVector::from_vec(solve_box_qp(&q_uu, q_u.as_slice(), &lo, &hi));

                // feedback only on the FREE (unclamped) controls; clamped rows are zero
                let free: Vec<usize> = (0..m).filter(|&i| kff[i] > lo[i] + 1e-6 && kff[i] < hi[i] - 1e-6).collect();
                let mut kfb = DMatrix::zeros(m, n);
                if !free.is_empty() {
                    let nf = free.len();
                    let mut quu_ff = DMatrix::zeros(nf, nf);
                    let mut qux_f = DMatrix::zeros(nf, n);
                    for (a, &i) in free.iter().enumerate() {
                        for (b, &j) in free.iter().enumerate() {
                            quu_ff[(a, b)] = q_uu[(i, j)];
                        }
                        qux_f.row_mut(a).copy_from(&q_ux.row(i));
                    }
                    if let Some(inv) = quu_ff.try_inverse() {
                        let kf = -(inv * qux_f); // nf × n
                        for (a, &i) in free.iter().enumerate() {
                            kfb.row_mut(i).copy_from(&kf.row(a));
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }

                // value update (DDP, no gaps)
                v_x = &q_x + kfb.transpose() * &q_uu * &kff + kfb.transpose() * &q_u + q_ux.transpose() * &kff;
                let vxx = &q_xx + kfb.transpose() * &q_uu * &kfb + kfb.transpose() * &q_ux + q_ux.transpose() * &kfb;
                v_xx = (&vxx + vxx.transpose()) * 0.5;
                ks[k] = kff;
                k_fb[k] = kfb;
            }
            if !ok {
                reg *= 10.0;
                if reg > 1e10 {
                    break;
                }
                continue;
            }
            ff_norm = ks.iter().map(|k| k.amax()).fold(0.0, f64::max);

            // ---- forward pass with backtracking ----
            let mut accepted = false;
            let mut alpha = 1.0;
            for _ in 0..12 {
                let mut nxs = vec![x0.clone()];
                let mut nus = Vec::with_capacity(big_n);
                for k in 0..big_n {
                    let du = &ks[k] * alpha + &k_fb[k] * (&nxs[k] - &xs[k]);
                    let u = self.clamp_u(&(&us[k] + du));
                    nxs.push(f(&nxs[k], &u));
                    nus.push(u);
                }
                let ncost = self.total_cost(&nxs, &nus);
                if ncost < cost - 1e-10 {
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
            if ff_norm < 1e-7 {
                converged = true;
                break;
            }
        }

        BoxDdpReport { xs, us, cost, ff_norm, iterations, converged, cost_history }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    // Double integrator: x = [p, v], u = [a].
    fn double_integrator(dt: f64) -> (impl Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>, impl Fn(&DVector<f64>, &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>)) {
        let f = move |x: &DVector<f64>, u: &DVector<f64>| dv(&[x[0] + dt * x[1], x[1] + dt * u[0]]);
        let jac = move |_x: &DVector<f64>, _u: &DVector<f64>| (DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]), DMatrix::from_row_slice(2, 1, &[0.0, dt]));
        (f, jac)
    }

    fn problem(u_lim: f64) -> BoxDdpProblem {
        BoxDdpProblem {
            n: 2, m: 1, horizon: 40,
            q: DMatrix::from_diagonal(&dv(&[0.1, 0.01])),
            r: DMatrix::from_diagonal(&dv(&[0.01])),
            qf: DMatrix::from_diagonal(&dv(&[100.0, 100.0])),
            x_ref: dv(&[0.0, 0.0]), u_ref: dv(&[0.0]), x_goal: dv(&[0.0, 0.0]),
            u_min: dv(&[-u_lim]), u_max: dv(&[u_lim]), reg: 1e-6,
        }
    }

    #[test]
    fn it_respects_the_control_limits() {
        let (f, jac) = double_integrator(0.1);
        let prob = problem(1.0);
        let rep = prob.solve(&f, &jac, &dv(&[2.0, 0.0]), vec![dv(&[0.0]); 40], 100);
        for u in &rep.us {
            assert!(u[0] >= -1.0 - 1e-9 && u[0] <= 1.0 + 1e-9, "control {u:?} outside the box");
        }
    }

    #[test]
    fn a_tight_limit_forces_saturation_and_still_converges() {
        // THE HEADLINE. From far away with a tight torque limit, the optimal effort saturates the actuator
        // (bang-bang-like) — the whole point of a control-limited solver — and it still converges.
        let (f, jac) = double_integrator(0.1);
        let prob = problem(0.6);
        let rep = prob.solve(&f, &jac, &dv(&[3.0, 0.0]), vec![dv(&[0.0]); 40], 200);
        assert!(rep.converged, "should converge under the tight limit");
        let saturated = rep.us.iter().filter(|u| u[0].abs() > 0.6 - 1e-3).count();
        assert!(saturated >= 3, "a tight limit should saturate several controls: {saturated}");
        assert!(rep.xs.last().unwrap().norm() < 0.3, "should still reach near the target: {:?}", rep.xs.last().unwrap());
    }

    #[test]
    fn it_reduces_to_unconstrained_ddp_when_the_box_is_slack() {
        // With very wide limits no control clamps, so the solution equals the unconstrained optimum: the
        // feed-forward vanishes and every control sits strictly inside the box.
        let (f, jac) = double_integrator(0.1);
        let prob = problem(1000.0);
        let rep = prob.solve(&f, &jac, &dv(&[2.0, 0.0]), vec![dv(&[0.0]); 40], 100);
        assert!(rep.converged && rep.ff_norm < 1e-7, "unconstrained optimum: feed-forward should vanish");
        for u in &rep.us {
            assert!(u[0].abs() < 1000.0 - 1e-3, "no control should be at the (slack) bound");
        }
        // wide box can only do at least as well as a tight one
        let tight = problem(0.6);
        let rep_t = tight.solve(&f, &jac, &dv(&[2.0, 0.0]), vec![dv(&[0.0]); 40], 200);
        assert!(rep.cost <= rep_t.cost + 1e-6, "the slack box must not cost more than the tight one: {} vs {}", rep.cost, rep_t.cost);
    }
}
