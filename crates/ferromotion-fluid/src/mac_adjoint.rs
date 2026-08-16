//! **Reverse-mode adjoint of the MAC solver** (Honest Fluids stage 3b) — the harder gradient,
//! because a step is not a local stencil: it is `predict` (explicit advection + diffusion) then a
//! *global* pressure-Poisson projection that couples every cell. The adjoint exploits the
//! structure the forward solver already has:
//!
//! - The projection `(u*, v*) ↦ (u, v)` is **linear, parameter-free, and self-adjoint** in the
//!   staggered inner product — its VJP is the *same* pre-factored Poisson solve run backward, so
//!   `MacFluid`'s projection VJP costs one Cholesky solve, not a re-derivation.
//! - Only `predict` is nonlinear; its VJP is the transpose of the
//!   advection/diffusion stencil, linearized at the checkpointed state, with the wall ghosts
//!   (no-slip / free-slip / moving lid) routed to the right stored face and to the lid gradient.
//!
//! Backprop-through-time chains these across a rollout: one backward pass yields
//! `∂J/∂(initial field)`, `∂J/∂ν`, and `∂J/∂(lid speed)` — O(1) in the parameter count where
//! finite differences are O(params) forward sims. Every gradient is checked against central FD,
//! and the payoff test recovers viscosity from an observed final state through the projection.

use crate::MacFluid;
use faer::linalg::solvers::Solve;
use faer::Mat;

/// The pre-step states of a rollout, retained so the nonlinear `predict` can be linearized on the
/// backward pass. `pre[t]` is `(u, v)` *before* step `t`; the field after the last step lives in
/// the [`MacFluid`] itself.
pub struct MacTape {
    pub pre: Vec<(Vec<f64>, Vec<f64>)>,
}

/// Gradients accumulated by a backward pass.
pub struct MacGrads {
    /// `∂J/∂u₀`, `∂J/∂v₀` — the adjoint of the initial velocity field (same layout as `u`, `v`).
    pub u0: Vec<f64>,
    pub v0: Vec<f64>,
    /// `∂J/∂ν` — the adjoint of the kinematic viscosity, summed over the rollout.
    pub nu: f64,
    /// `∂J/∂(lid speed)` — the adjoint of the moving-lid boundary condition.
    pub lid: f64,
}

impl MacFluid {
    /// Run `steps` and record the pre-step states for a subsequent [`Self::backward`].
    pub fn rollout_tape(&mut self, steps: usize) -> MacTape {
        let mut pre = Vec::with_capacity(steps);
        for _ in 0..steps {
            pre.push((self.u.clone(), self.v.clone()));
            self.step();
        }
        MacTape { pre }
    }

    /// Backprop-through-time. `(u_bar, v_bar)` seed the adjoint at the final field
    /// (e.g. `∂J/∂u_final`). Returns `∂J/∂` initial-field, viscosity, and lid speed.
    pub fn backward(&mut self, tape: &MacTape, mut u_bar: Vec<f64>, mut v_bar: Vec<f64>) -> MacGrads {
        let mut g_nu = 0.0;
        let mut g_lid = 0.0;
        for (pre_u, pre_v) in tape.pre.iter().rev() {
            // Reverse the projection first (state-free, self-adjoint), then the predictor,
            // linearized at this step's pre-state.
            let (us_bar, vs_bar) = self.project_vjp(&u_bar, &v_bar);
            let (u_in, v_in, dnu, dlid) = self.predict_vjp(pre_u, pre_v, &us_bar, &vs_bar);
            u_bar = u_in;
            v_bar = v_in;
            g_nu += dnu;
            g_lid += dlid;
        }
        MacGrads { u0: u_bar, v0: v_bar, nu: g_nu, lid: g_lid }
    }

    /// Solve the reduced (cell-0-pinned) Poisson system `L x = b_reduced` with the pre-factored
    /// Cholesky. `b` is indexed over all `nx·ny` cells; cell 0 is dropped. Returns the reduced
    /// solution over cells `1..nx·ny`.
    fn poisson_solve_reduced(&self, b: &[f64]) -> Vec<f64> {
        let n = self.nx * self.ny - 1;
        let mut rhs = Mat::<f64>::zeros(n, 1);
        for k in 1..self.nx * self.ny {
            rhs[(k - 1, 0)] = b[k];
        }
        self.poisson.solve_in_place(&mut rhs);
        (0..n).map(|r| rhs[(r, 0)]).collect()
    }

    /// VJP of the pressure projection `(u*, v*) ↦ (u, v)`. Self-adjoint: the map is
    /// `I + G·L⁻¹·D` with `L` the SPD Poisson operator, so its transpose reuses the same factor.
    /// Parameter-free — the projection depends on neither ν nor the lid.
    fn project_vjp(&self, u_bar: &[f64], v_bar: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, h, dt) = (self.nx, self.ny, self.h, self.dt);
        // Identity term: u = u* − (dt/h)·∂p, so u* inherits the output adjoint directly.
        let mut us_bar = u_bar.to_vec();
        let mut vs_bar = v_bar.to_vec();

        // Accumulate the pressure adjoint from the correction `u = u* − (dt/h)(p_i − p_{i-1})`.
        let mut p_bar = vec![0.0f64; nx * ny];
        let c = dt / h;
        for i in 1..nx {
            for j in 0..ny {
                let g = u_bar[i * ny + j] * c;
                p_bar[i * ny + j] -= g;
                p_bar[(i - 1) * ny + j] += g;
            }
        }
        for i in 0..nx {
            for j in 1..ny {
                let g = v_bar[i * (ny + 1) + j] * c;
                p_bar[i * ny + j] -= g;
                p_bar[i * ny + j - 1] += g;
            }
        }

        // Adjoint of p = L⁻¹ rhs is rhs_bar = L⁻ᵀ p_bar = L⁻¹ p_bar (cell 0 pinned out).
        let rhs_bar = self.poisson_solve_reduced(&p_bar);

        // Adjoint of rhs = −(h²/dt)·div, then scatter div_bar back through the divergence stencil.
        let s = -(h * h) / dt / h; // (−h²/dt) from rhs, (1/h) from each divergence difference
        for i in 0..nx {
            for j in 0..ny {
                let k = i * ny + j;
                if k == 0 {
                    continue;
                }
                let d = rhs_bar[k - 1] * s;
                us_bar[(i + 1) * ny + j] += d;
                us_bar[i * ny + j] -= d;
                vs_bar[i * (ny + 1) + j + 1] += d;
                vs_bar[i * (ny + 1) + j] -= d;
            }
        }
        (us_bar, vs_bar)
    }

    /// VJP of `predict`, linearized at `(pre_u, pre_v)`. Returns the input-field adjoint plus the
    /// viscosity and lid-speed gradient contributions for this step.
    fn predict_vjp(
        &mut self,
        pre_u: &[f64],
        pre_v: &[f64],
        us_bar: &[f64],
        vs_bar: &[f64],
    ) -> (Vec<f64>, Vec<f64>, f64, f64) {
        // The ghost accessors read self.{u,v}; point them at the checkpoint for this step.
        self.u.copy_from_slice(pre_u);
        self.v.copy_from_slice(pre_v);
        let (nx, ny, h, dt, nu) = (self.nx, self.ny, self.h, self.dt, self.nu);
        let (inv2h, invh2) = (1.0 / (2.0 * h), 1.0 / (h * h));

        let mut u_bar = vec![0.0f64; (nx + 1) * ny];
        let mut v_bar = vec![0.0f64; nx * (ny + 1)];
        let mut g_nu = 0.0;
        let mut g_lid = 0.0;

        // Wall faces of u* / v* are the un-overwritten clone (columns 0/nx, rows 0/ny) — identity.
        for j in 0..ny {
            u_bar[j] += us_bar[j]; // column 0
            u_bar[nx * ny + j] += us_bar[nx * ny + j]; // column nx
        }
        for i in 0..nx {
            v_bar[i * (ny + 1)] += vs_bar[i * (ny + 1)]; // row 0
            v_bar[i * (ny + 1) + ny] += vs_bar[i * (ny + 1) + ny]; // row ny
        }

        // Interior u-faces (the `for i in 1..nx` predictor loop).
        for i in 1..nx {
            for j in 0..ny {
                let s = us_bar[i * ny + j];
                if s == 0.0 {
                    continue;
                }
                let (ii, jj) = (i as i32, j as i32);
                let uc = self.uu(ii, jj);
                let dudx = (self.uu(ii + 1, jj) - self.uu(ii - 1, jj)) * inv2h;
                let dudy = (self.uu(ii, jj + 1) - self.uu(ii, jj - 1)) * inv2h;
                let vbar = 0.25 * (self.vv(ii - 1, jj) + self.vv(ii, jj) + self.vv(ii - 1, jj + 1) + self.vv(ii, jj + 1));
                let lap = (self.uu(ii + 1, jj) + self.uu(ii - 1, jj) + self.uu(ii, jj + 1) + self.uu(ii, jj - 1) - 4.0 * uc) * invh2;

                let a_c = s * (1.0 - dt * dudx - 4.0 * dt * nu * invh2);
                let a_ip1 = s * (-dt * uc * inv2h + dt * nu * invh2);
                let a_im1 = s * (dt * uc * inv2h + dt * nu * invh2);
                let a_jp1 = s * (-dt * vbar * inv2h + dt * nu * invh2);
                let a_jm1 = s * (dt * vbar * inv2h + dt * nu * invh2);
                self.route_u(ii, jj, a_c, &mut u_bar, &mut g_lid);
                self.route_u(ii + 1, jj, a_ip1, &mut u_bar, &mut g_lid);
                self.route_u(ii - 1, jj, a_im1, &mut u_bar, &mut g_lid);
                self.route_u(ii, jj + 1, a_jp1, &mut u_bar, &mut g_lid);
                self.route_u(ii, jj - 1, a_jm1, &mut u_bar, &mut g_lid);

                let a_v = s * (-dt * dudy) * 0.25;
                self.route_v(ii - 1, jj, a_v, &mut v_bar);
                self.route_v(ii, jj, a_v, &mut v_bar);
                self.route_v(ii - 1, jj + 1, a_v, &mut v_bar);
                self.route_v(ii, jj + 1, a_v, &mut v_bar);

                g_nu += s * dt * lap;
            }
        }

        // Interior v-faces (the `for j in 1..ny` predictor loop).
        for i in 0..nx {
            for j in 1..ny {
                let s = vs_bar[i * (ny + 1) + j];
                if s == 0.0 {
                    continue;
                }
                let (ii, jj) = (i as i32, j as i32);
                let vc = self.vv(ii, jj);
                let dvdx = (self.vv(ii + 1, jj) - self.vv(ii - 1, jj)) * inv2h;
                let dvdy = (self.vv(ii, jj + 1) - self.vv(ii, jj - 1)) * inv2h;
                let ubar = 0.25 * (self.uu(ii, jj - 1) + self.uu(ii + 1, jj - 1) + self.uu(ii, jj) + self.uu(ii + 1, jj));
                let lap = (self.vv(ii + 1, jj) + self.vv(ii - 1, jj) + self.vv(ii, jj + 1) + self.vv(ii, jj - 1) - 4.0 * vc) * invh2;

                let a_c = s * (1.0 - dt * dvdy - 4.0 * dt * nu * invh2);
                let a_ip1 = s * (-dt * ubar * inv2h + dt * nu * invh2);
                let a_im1 = s * (dt * ubar * inv2h + dt * nu * invh2);
                let a_jp1 = s * (-dt * vc * inv2h + dt * nu * invh2);
                let a_jm1 = s * (dt * vc * inv2h + dt * nu * invh2);
                self.route_v(ii, jj, a_c, &mut v_bar);
                self.route_v(ii + 1, jj, a_ip1, &mut v_bar);
                self.route_v(ii - 1, jj, a_im1, &mut v_bar);
                self.route_v(ii, jj + 1, a_jp1, &mut v_bar);
                self.route_v(ii, jj - 1, a_jm1, &mut v_bar);

                let a_u = s * (-dt * dvdx) * 0.25;
                self.route_u(ii, jj - 1, a_u, &mut u_bar, &mut g_lid);
                self.route_u(ii + 1, jj - 1, a_u, &mut u_bar, &mut g_lid);
                self.route_u(ii, jj, a_u, &mut u_bar, &mut g_lid);
                self.route_u(ii + 1, jj, a_u, &mut u_bar, &mut g_lid);

                g_nu += s * dt * lap;
            }
        }
        (u_bar, v_bar, g_nu, g_lid)
    }

    /// Route a `u`-value adjoint at ghost-aware index `(i, j)` back to the stored face adjoint,
    /// mirroring the `uu` ghost rule; the moving-lid top ghost `2·lid − w` contributes to `g_lid`.
    fn route_u(&self, i: i32, j: i32, a: f64, u_bar: &mut [f64], g_lid: &mut f64) {
        if a == 0.0 {
            return;
        }
        let ny = self.ny as i32;
        let free = self.free_slip;
        let iu = i as usize;
        if j < 0 {
            u_bar[iu * self.ny] += if free { a } else { -a };
        } else if j >= ny {
            u_bar[iu * self.ny + (ny as usize - 1)] += if free { a } else { -a };
            if !free {
                *g_lid += 2.0 * a; // ∂(2·lid − w)/∂lid = 2
            }
        } else {
            u_bar[iu * self.ny + j as usize] += a;
        }
    }

    /// Route a `v`-value adjoint at ghost-aware index `(i, j)` back to the stored face adjoint,
    /// mirroring the `vv` ghost rule (no-slip / free-slip left+right walls).
    fn route_v(&self, i: i32, j: i32, a: f64, v_bar: &mut [f64]) {
        if a == 0.0 {
            return;
        }
        let nx = self.nx as i32;
        let free = self.free_slip;
        let ju = j as usize;
        let stride = self.ny + 1;
        if i < 0 {
            v_bar[ju] += if free { a } else { -a };
        } else if i >= nx {
            v_bar[(nx as usize - 1) * stride + ju] += if free { a } else { -a };
        } else {
            v_bar[i as usize * stride + ju] += a;
        }
    }
}

#[cfg(test)]
mod verification {
    use crate::MacFluid;
    use std::f64::consts::PI;

    /// A diffusion-number-safe `dt`, computed once from the largest viscosity in play and held
    /// FIXED across ν perturbations — otherwise a ν-dependent `dt` folds `∂J/∂dt` into the finite
    /// difference while the adjoint (correctly) holds `dt` constant, and the two disagree.
    fn dt_for(n: usize, nu_max: f64) -> f64 {
        0.2 * (1.0 / n as f64).powi(2) / nu_max.max(1e-3)
    }

    /// Seed a decaying Taylor–Green-ish field on a no-slip cavity (so the lid matters), advance,
    /// and evaluate a quadratic terminal objective J = ½‖u_T‖² + ½‖v_T‖².
    fn setup_dt(n: usize, nu: f64, lid: f64, dt: f64) -> MacFluid {
        let mut f = MacFluid::new(n, n, nu, dt, lid);
        let k = 2.0 * PI;
        f.set_velocity(
            |x, y| 0.05 * (k * x).sin() * (k * y).cos(),
            |x, y| -0.05 * (k * x).cos() * (k * y).sin(),
        );
        f
    }

    fn objective(f: &MacFluid) -> f64 {
        0.5 * (f.u.iter().map(|x| x * x).sum::<f64>() + f.v.iter().map(|x| x * x).sum::<f64>())
    }

    /// The adjoint viscosity gradient vs central finite differences.
    #[test]
    fn nu_gradient_matches_fd() {
        let (n, nu, lid, steps) = (16, 0.01, 0.1, 10);
        let dt = dt_for(n, nu + 1e-6);
        let mut f = setup_dt(n, nu, lid, dt);
        let tape = f.rollout_tape(steps);
        let (ub, vb) = (f.u.to_vec(), f.v.to_vec()); // ∂J/∂u_T = u_T
        let g = f.backward(&tape, ub, vb);

        let eps = 1e-6;
        let jp = { let mut f = setup_dt(n, nu + eps, lid, dt); for _ in 0..steps { f.step(); } objective(&f) };
        let jm = { let mut f = setup_dt(n, nu - eps, lid, dt); for _ in 0..steps { f.step(); } objective(&f) };
        let fd = (jp - jm) / (2.0 * eps);
        let rel = (g.nu - fd).abs() / fd.abs().max(1e-12);
        eprintln!("dJ/dnu: adjoint {:.6e}  fd {:.6e}  rel {:.2e}", g.nu, fd, rel);
        assert!(rel < 1e-4, "nu gradient rel err {rel}");
    }

    /// The adjoint lid-speed gradient vs central finite differences (exercises the moving-lid ghost).
    #[test]
    fn lid_gradient_matches_fd() {
        let (n, nu, lid, steps) = (16, 0.02, 0.15, 10);
        let dt = dt_for(n, nu);
        let mut f = setup_dt(n, nu, lid, dt);
        let tape = f.rollout_tape(steps);
        let (ub, vb) = (f.u.to_vec(), f.v.to_vec());
        let g = f.backward(&tape, ub, vb);

        let eps = 1e-6;
        let jp = { let mut f = setup_dt(n, nu, lid + eps, dt); for _ in 0..steps { f.step(); } objective(&f) };
        let jm = { let mut f = setup_dt(n, nu, lid - eps, dt); for _ in 0..steps { f.step(); } objective(&f) };
        let fd = (jp - jm) / (2.0 * eps);
        let rel = (g.lid - fd).abs() / fd.abs().max(1e-12);
        eprintln!("dJ/dlid: adjoint {:.6e}  fd {:.6e}  rel {:.2e}", g.lid, fd, rel);
        assert!(rel < 1e-4, "lid gradient rel err {rel}");
    }

    /// The adjoint initial-field gradient vs FD on a representative interior u-face.
    #[test]
    fn init_field_gradient_matches_fd() {
        let (n, nu, lid, steps) = (16, 0.02, 0.1, 8);
        let dt = dt_for(n, nu);
        let mut f = setup_dt(n, nu, lid, dt);
        let tape = f.rollout_tape(steps);
        let (ub, vb) = (f.u.to_vec(), f.v.to_vec());
        let g = f.backward(&tape, ub, vb);

        let probe = (n / 2) * n + n / 2; // an interior u-face index
        let eps = 1e-7;
        let run = |bump: f64| {
            let mut f = setup_dt(n, nu, lid, dt);
            f.u[probe] += bump;
            for _ in 0..steps {
                f.step();
            }
            objective(&f)
        };
        let fd = (run(eps) - run(-eps)) / (2.0 * eps);
        let rel = (g.u0[probe] - fd).abs() / fd.abs().max(1e-9);
        eprintln!("dJ/du0[probe]: adjoint {:.6e}  fd {:.6e}  rel {:.2e}", g.u0[probe], fd, rel);
        assert!(rel < 1e-3, "init-field gradient rel err {rel}");
    }

    /// **The payoff: recover viscosity from an observed final field, through the projection.**
    /// Gradient descent on J(ν) = ½‖u_T(ν) − u_T(ν*)‖² using the adjoint gradient — O(1) sims per
    /// step regardless of parameter count, converging to the true ν.
    #[test]
    fn viscosity_identifies_from_final_field_by_adjoint() {
        let (n, steps) = (24, 40);
        let nu_true = 0.03;
        let lid = 0.1;
        let dt = dt_for(n, 0.06); // fixed across the search — ν is the only free parameter

        // Observed target trajectory.
        let (u_obs, v_obs) = {
            let mut f = setup_dt(n, nu_true, lid, dt);
            for _ in 0..steps {
                f.step();
            }
            (f.u.to_vec(), f.v.to_vec())
        };

        // Exact adjoint gradient in an adaptive 1-D descent: grow the step on improvement, halve
        // and undo on a worsening — converges fast because the gradient is exact, not sampled.
        let loss_at = |nu: f64| -> (f64, f64) {
            let mut f = setup_dt(n, nu, lid, dt);
            let tape = f.rollout_tape(steps);
            let ub: Vec<f64> = f.u.iter().zip(&u_obs).map(|(a, b)| a - b).collect();
            let vb: Vec<f64> = f.v.iter().zip(&v_obs).map(|(a, b)| a - b).collect();
            let loss = 0.5 * (ub.iter().map(|x| x * x).sum::<f64>() + vb.iter().map(|x| x * x).sum::<f64>());
            let g = f.backward(&tape, ub, vb);
            (loss, g.nu)
        };
        let mut nu = 0.06; // 2× wrong
        let mut lr = 1.0;
        let (mut loss, mut grad) = loss_at(nu);
        for it in 0..120 {
            let cand = (nu - lr * grad).clamp(1e-4, 0.2);
            let (l2, g2) = loss_at(cand);
            if l2 < loss {
                nu = cand;
                loss = l2;
                grad = g2;
                lr *= 1.3;
            } else {
                lr *= 0.5;
            }
            if it % 20 == 0 {
                eprintln!("it {it}: nu {nu:.6} loss {loss:.3e} grad {grad:.3e} lr {lr:.2e}");
            }
            if lr < 1e-6 {
                break;
            }
        }
        let rel = (nu - nu_true).abs() / nu_true;
        eprintln!("recovered nu {nu:.6} (true {nu_true}) rel {rel:.2e}");
        assert!(rel < 0.02, "viscosity not recovered: {nu} vs {nu_true} (rel {rel})");
    }
}
