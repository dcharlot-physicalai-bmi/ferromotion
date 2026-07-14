//! **Direct collocation** trajectory optimization (Hermite-Simpson), the transcription behind DIRCON
//! (Posa, Kuindersma & Tedrake). Unlike our shooting methods ([`crate::solve_ilqr`],
//! [`crate::ConstrainedLqr`], [`crate::Mppi`]), which roll controls forward, direct collocation makes
//! the **states and controls at every knot decision variables** and enforces the dynamics as
//! *collocation constraints* — far better conditioned for stiff or long-horizon problems.
//!
//! Between knots `k` and `k+1` the Hermite-Simpson defect is
//! `x_{k+1} − x_k − (h/6)(f_k + 4 f_c + f_{k+1}) = 0`, with the interpolated midpoint
//! `x_c = ½(x_k + x_{k+1}) + (h/8)(f_k − f_{k+1})`, `u_c = ½(u_k + u_{k+1})`. Together with the
//! boundary conditions and a control-effort cost, this is a nonlinear least-squares problem solved
//! here by Levenberg-Marquardt. (DIRCON's contribution — adding manifold/contact constraints — layers
//! on the same defect structure.) Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A direct-collocation problem: minimize control effort subject to Hermite-Simpson dynamics and
/// fixed boundary states. `f(x, u)` returns the continuous-time derivative `ẋ`.
pub struct DirectCollocation<F: Fn(&[f64], &[f64]) -> Vec<f64>> {
    pub nx: usize,
    pub nu: usize,
    /// Number of knot points (`knots − 1` intervals).
    pub knots: usize,
    pub dt: f64,
    pub x_start: Vec<f64>,
    pub x_goal: Vec<f64>,
    /// Control-effort weight (soft cost).
    pub r_effort: f64,
    pub f: F,
}

/// A collocated trajectory.
#[derive(Clone, Debug)]
pub struct CollocationResult {
    pub xs: Vec<Vec<f64>>,
    pub us: Vec<Vec<f64>>,
    /// Max dynamics-defect magnitude (dynamic feasibility of the transcription).
    pub max_defect: f64,
    pub converged: bool,
}

impl<F: Fn(&[f64], &[f64]) -> Vec<f64>> DirectCollocation<F> {
    fn zdim(&self) -> usize {
        self.knots * (self.nx + self.nu)
    }

    fn unpack(&self, z: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let s = self.nx + self.nu;
        let xs = (0..self.knots).map(|k| z[k * s..k * s + self.nx].to_vec()).collect();
        let us = (0..self.knots).map(|k| z[k * s + self.nx..k * s + s].to_vec()).collect();
        (xs, us)
    }

    /// Equality constraints `c(z)` — the Hermite-Simpson defects followed by the boundary conditions.
    fn constraints(&self, z: &[f64]) -> Vec<f64> {
        let (xs, us) = self.unpack(z);
        let (nx, dt) = (self.nx, self.dt);
        let mut c = Vec::new();
        for k in 0..self.knots - 1 {
            let fk = (self.f)(&xs[k], &us[k]);
            let fk1 = (self.f)(&xs[k + 1], &us[k + 1]);
            let xc: Vec<f64> = (0..nx).map(|i| 0.5 * (xs[k][i] + xs[k + 1][i]) + dt / 8.0 * (fk[i] - fk1[i])).collect();
            let uc: Vec<f64> = (0..self.nu).map(|i| 0.5 * (us[k][i] + us[k + 1][i])).collect();
            let fc = (self.f)(&xc, &uc);
            for i in 0..nx {
                c.push(xs[k + 1][i] - xs[k][i] - dt / 6.0 * (fk[i] + 4.0 * fc[i] + fk1[i]));
            }
        }
        for i in 0..nx {
            c.push(xs[0][i] - self.x_start[i]);
            c.push(xs[self.knots - 1][i] - self.x_goal[i]);
        }
        c
    }

    fn max_defect(&self, z: &[f64]) -> f64 {
        self.constraints(z).iter().fold(0.0f64, |m, &c| m.max(c.abs()))
    }

    /// Simpson-quadrature control effort `∫uᵀu dt ≈ (dt/3)·Σ (u_k² + u_{k+1}² + u_k·u_{k+1})`, scaled
    /// by `r_effort` — the discretization of `∫uᵀR u dt` consistent with Hermite-Simpson.
    fn effort(&self, z: &[f64]) -> f64 {
        let (_, us) = self.unpack(z);
        let mut e = 0.0;
        for k in 0..self.knots - 1 {
            for d in 0..self.nu {
                e += us[k][d] * us[k][d] + us[k + 1][d] * us[k + 1][d] + us[k][d] * us[k + 1][d];
            }
        }
        self.r_effort * self.dt / 3.0 * e
    }

    /// Gradient and (constant) Hessian of [`effort`] w.r.t. `z` — the QP objective for SQP.
    fn effort_gh(&self, z: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
        let (nx, s, n) = (self.nx, self.nx + self.nu, z.len());
        let c = self.r_effort * self.dt / 3.0;
        let mut g = DVector::zeros(n);
        let mut h = DMatrix::zeros(n, n);
        for k in 0..self.knots - 1 {
            for d in 0..self.nu {
                let (a, b) = (k * s + nx + d, (k + 1) * s + nx + d);
                g[a] += c * (2.0 * z[a] + z[b]);
                g[b] += c * (2.0 * z[b] + z[a]);
                h[(a, a)] += 2.0 * c;
                h[(b, b)] += 2.0 * c;
                h[(a, b)] += c;
                h[(b, a)] += c;
            }
        }
        (g, h)
    }

    /// Finite-difference Jacobian of the constraints, `∂c/∂z` (`nc × n`).
    fn constraint_jacobian(&self, z: &[f64], nc: usize) -> DMatrix<f64> {
        let n = z.len();
        let eps = 1e-7;
        let mut j = DMatrix::zeros(nc, n);
        for col in 0..n {
            let (mut zp, mut zm) = (z.to_vec(), z.to_vec());
            zp[col] += eps;
            zm[col] -= eps;
            let (cp, cm) = (self.constraints(&zp), self.constraints(&zm));
            for row in 0..nc {
                j[(row, col)] = (cp[row] - cm[row]) / (2.0 * eps);
            }
        }
        j
    }

    /// Solve by **Sequential Quadratic Programming**: each iteration minimizes the (quadratic) control
    /// effort subject to the *linearized* dynamics + boundary constraints, via the KKT system
    /// `[H Jᵀ; J 0][Δz; ν] = [−∇effort; −c]`, with a merit line search. This yields the exact
    /// min-effort dynamically-feasible trajectory (one step for linear dynamics; iterated for nonlinear).
    pub fn solve(&self) -> CollocationResult {
        let (nx, nu, s) = (self.nx, self.nu, self.nx + self.nu);
        let n = self.zdim();
        let nc = self.constraints(&vec![0.0; n]).len();
        let r = self.r_effort;

        // Initial guess: linear interpolation of states, zero controls.
        let mut z = vec![0.0; n];
        for k in 0..self.knots {
            let a = k as f64 / (self.knots - 1) as f64;
            for i in 0..nx {
                z[k * s + i] = (1.0 - a) * self.x_start[i] + a * self.x_goal[i];
            }
        }

        let _ = (r, nu);
        for _ in 0..60 {
            let c = self.constraints(&z);
            let max_c = c.iter().fold(0.0f64, |m, &ci| m.max(ci.abs()));
            let jc = self.constraint_jacobian(&z, nc);
            let (g_obj, mut h) = self.effort_gh(&z);
            for i in 0..n {
                h[(i, i)] += 1e-9; // regularize the (effort-free) state block so the KKT is nonsingular
            }

            // KKT system [[H, Jᵀ],[J, 0]] [Δz; ν] = [−g; −c].
            let dim = n + nc;
            let mut kkt = DMatrix::zeros(dim, dim);
            let mut rhs = DVector::zeros(dim);
            kkt.view_mut((0, 0), (n, n)).copy_from(&h);
            for i in 0..n {
                rhs[i] = -g_obj[i];
            }
            kkt.view_mut((0, n), (n, nc)).copy_from(&jc.transpose());
            kkt.view_mut((n, 0), (nc, n)).copy_from(&jc);
            for i in 0..nc {
                rhs[n + i] = -c[i];
            }
            let sol = match kkt.lu().solve(&rhs) {
                Some(s) => s,
                None => break,
            };
            let dz: Vec<f64> = (0..n).map(|i| sol[i]).collect();

            // Merit line search: φ = effort + M·‖c‖₁.
            let m_pen = 1e3;
            let merit = |zz: &[f64]| self.effort(zz) + m_pen * self.constraints(zz).iter().map(|v| v.abs()).sum::<f64>();
            let phi0 = merit(&z);
            let mut alpha = 1.0;
            let mut z_new = z.clone();
            for _ in 0..20 {
                z_new = (0..n).map(|i| z[i] + alpha * dz[i]).collect();
                if merit(&z_new) <= phi0 + 1e-12 {
                    break;
                }
                alpha *= 0.5;
            }
            let step = dz.iter().map(|d| (alpha * d).abs()).fold(0.0f64, f64::max);
            z = z_new;
            if max_c < 1e-9 && step < 1e-9 {
                break;
            }
        }

        let (xs, us) = self.unpack(&z);
        let md = self.max_defect(&z);
        CollocationResult { xs, us, max_defect: md, converged: md < 1e-3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_integrator_matches_the_analytic_min_effort_solution() {
        // ẍ = u, from (0,0) to (1,0) in T=2. Analytic min-∫u²: x(t)=3t²/T²−2t³/T³, u(t)=6/T²−12t/T³.
        let t_total = 2.0;
        let knots = 21;
        let dt = t_total / (knots - 1) as f64;
        let prob = DirectCollocation {
            nx: 2,
            nu: 1,
            knots,
            dt,
            x_start: vec![0.0, 0.0],
            x_goal: vec![1.0, 0.0],
            r_effort: 1.0,
            f: |x: &[f64], u: &[f64]| vec![x[1], u[0]],
        };
        let res = prob.solve();
        assert!(res.converged, "collocation did not converge (max defect {})", res.max_defect);
        // Boundary + dynamic feasibility.
        assert!(res.max_defect < 1e-3, "defects too large: {}", res.max_defect);
        assert!((res.xs[knots - 1][0] - 1.0).abs() < 1e-3 && res.xs[knots - 1][1].abs() < 1e-3, "goal not met");
        // Controls match the analytic linear profile u(t) = 6/T² − 12 t/T³.
        for k in 0..knots {
            let t = k as f64 * dt;
            let u_true = 6.0 / (t_total * t_total) - 12.0 * t / t_total.powi(3);
            assert!((res.us[k][0] - u_true).abs() < 0.03, "u[{k}]={} vs analytic {u_true}", res.us[k][0]);
        }
    }

    #[test]
    fn pendulum_swings_up() {
        // θ̈ = u − g·sinθ (m=l=1). Swing from hanging (θ=0) to upright (θ=π) at rest.
        let knots = 31;
        let dt = 3.0 / (knots - 1) as f64;
        let g = 9.81;
        let prob = DirectCollocation {
            nx: 2,
            nu: 1,
            knots,
            dt,
            x_start: vec![0.0, 0.0],
            x_goal: vec![std::f64::consts::PI, 0.0],
            r_effort: 0.01,
            f: move |x: &[f64], u: &[f64]| vec![x[1], u[0] - g * x[0].sin()],
        };
        let res = prob.solve();
        assert!(res.converged, "swing-up did not converge (max defect {})", res.max_defect);
        assert!(res.max_defect < 1e-3, "dynamics not satisfied: {}", res.max_defect);
        let end = &res.xs[knots - 1];
        assert!((end[0] - std::f64::consts::PI).abs() < 1e-2 && end[1].abs() < 1e-2, "did not reach upright: {end:?}");
        // The maneuver is non-trivial (real torque applied).
        let peak_u = res.us.iter().map(|u| u[0].abs()).fold(0.0, f64::max);
        assert!(peak_u > 1.0, "no meaningful control effort: peak |u| = {peak_u}");
    }
}
