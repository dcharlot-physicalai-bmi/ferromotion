//! **SCvx — Successive Convexification** (Mao, Szmuk & Açıkmeşe, 2016/2018): the workhorse for non-convex
//! trajectory optimization, and the algorithm behind autonomous rocket powered-descent guidance. A hard
//! problem — nonlinear dynamics, a boundary-value landing constraint — is solved as a *sequence* of convex
//! subproblems, each one linearizing the dynamics about the previous iterate and correcting toward
//! feasibility. Three safeguards, all from the paper, make the sequence actually converge:
//!
//! * **virtual controls** `ν` — a slack added to the linearized dynamics (Eq. 3,
//!   `x_{i+1} = f(x̄_i,ū_i) + A_i d_i + B_i w_i + ν_i`) and driven to zero by an exact `ℓ₁` penalty `λ‖ν‖₁`.
//!   They guarantee every subproblem is feasible, so the method never stalls on *artificial* infeasibility
//!   (a linearization that momentarily has no solution);
//! * a **trust region** `‖d,w‖∞ ≤ r` bounding the step, so the linearization is never trusted past where
//!   it holds — this is what prevents *artificial* unboundedness;
//! * a **trust-region ratio test** (Eq. 9): `ρ = ΔJ / ΔL`, actual nonlinear cost reduction over the
//!   reduction the convex model predicted. `ρ < ρ₀` ⇒ reject the step and shrink `r`; a good `ρ` ⇒ accept
//!   and grow `r`. This is what earns the superlinear convergence.
//!
//! The demo is a 2-D rocket soft-landing in scaled units (so one trust radius covers all states): the
//! dynamics are genuinely non-convex — thrust divided by a *depleting* mass (`v̇ = T/m`, `ṁ = −α‖T‖`) —
//! and SCvx turns a dynamically-infeasible straight-line guess into a feasible, fuel-efficient landing.
//! The convex subproblem is a QP solved with `clarabel`. Pure Rust → WASM-clean.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::{DMatrix, DVector};

/// Continuous dynamics `f(x, u) → ẋ`.
pub type DynFn = dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>;
/// The dynamics Jacobians `(∂f/∂x, ∂f/∂u)` evaluated at a point.
pub type JacFn = dyn Fn(&DVector<f64>, &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>);

/// A continuous-time optimal-control problem for SCvx: `min Σ‖u‖²` (fuel proxy) subject to `ẋ = f(x,u)`,
/// a fixed initial state, a partial terminal state (the trailing `terminal_free_tail` components — e.g. a
/// rocket's mass — are left free), and box control limits. Discretized by forward Euler over `n` steps.
#[derive(Clone, Debug)]
pub struct ScvxProblem {
    pub nx: usize,
    pub nu: usize,
    pub n: usize,
    pub dt: f64,
    pub x_init: DVector<f64>,
    pub x_goal: DVector<f64>,
    /// Number of trailing state components left unconstrained at the terminal time (0 = pin the whole
    /// state; 1 for the rocket = leave final mass free).
    pub terminal_free_tail: usize,
    pub u_min: DVector<f64>,
    pub u_max: DVector<f64>,
    pub fuel_weight: f64,
    /// Exact-penalty weight `λ` on the virtual controls.
    pub lambda: f64,
}

/// SCvx tuning (trust-region ratio thresholds and update factors). Defaults follow the paper's guidance:
/// `ρ₀≈0`, `ρ₁` slightly above 0, `ρ₂` marginally below 1; shrink by `α`, grow by `β`.
#[derive(Clone, Copy, Debug)]
pub struct ScvxOpts {
    pub rho0: f64,
    pub rho1: f64,
    pub rho2: f64,
    pub alpha: f64,
    pub beta: f64,
    pub r0: f64,
    pub r_min: f64,
    pub r_max: f64,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for ScvxOpts {
    fn default() -> Self {
        ScvxOpts { rho0: 0.0, rho1: 0.1, rho2: 0.9, alpha: 2.0, beta: 1.5, r0: 0.5, r_min: 1e-4, r_max: 10.0, max_iter: 60, tol: 1e-5 }
    }
}

/// The result of an SCvx solve: the converged trajectory and the convergence trace.
#[derive(Clone, Debug)]
pub struct ScvxReport {
    pub xs: Vec<DVector<f64>>,
    pub us: Vec<DVector<f64>>,
    pub iterations: usize,
    pub converged: bool,
    /// `Σ‖x_{i+1} − f_d(x_i,u_i)‖₁` of the *true* nonlinear discrete dynamics at the returned trajectory
    /// (→ 0 ⇒ dynamically feasible).
    pub final_defect: f64,
    pub final_fuel: f64,
    /// `Σ‖ν_i‖₁` from the last accepted subproblem (→ 0 ⇒ the linear model needed no slack).
    pub final_virtual: f64,
    pub radius_history: Vec<f64>,
    pub defect_history: Vec<f64>,
}

impl ScvxProblem {
    /// Forward-Euler discrete step `f_d(x,u) = x + dt·f(x,u)` of the true (nonlinear) dynamics.
    fn step_d(&self, f: &DynFn, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        x + self.dt * f(x, u)
    }

    /// `Σ‖x_{i+1} − f_d(x_i,u_i)‖₁` — the true nonlinear dynamic infeasibility of a trajectory.
    fn defect(&self, f: &DynFn, xs: &[DVector<f64>], us: &[DVector<f64>]) -> f64 {
        (0..self.n).map(|i| (&xs[i + 1] - self.step_d(f, &xs[i], &us[i])).abs().sum()).sum()
    }

    fn fuel(&self, us: &[DVector<f64>]) -> f64 {
        self.fuel_weight * us.iter().map(|u| u.dot(u)).sum::<f64>()
    }

    /// The exact-penalty nonlinear cost `J = fuel + λ·defect` used in the ratio test.
    fn penalized_cost(&self, f: &DynFn, xs: &[DVector<f64>], us: &[DVector<f64>]) -> f64 {
        self.fuel(us) + self.lambda * self.defect(f, xs, us)
    }

    /// Run SCvx from the reference trajectory `(xs0, us0)`. `f` is the continuous dynamics, `jac` returns
    /// the continuous Jacobians `(∂f/∂x, ∂f/∂u)` at a point.
    pub fn solve(
        &self,
        f: &DynFn,
        jac: &JacFn,
        xs0: Vec<DVector<f64>>,
        us0: Vec<DVector<f64>>,
        opts: ScvxOpts,
    ) -> ScvxReport {
        let mut xs = xs0;
        let mut us = us0;
        let mut r = opts.r0;
        let mut j_cur = self.penalized_cost(f, &xs, &us);
        let mut radius_history = vec![r];
        let mut defect_history = vec![self.defect(f, &xs, &us)];
        let mut converged = false;
        let mut last_virtual = f64::INFINITY;
        let mut iterations = 0;

        for _ in 0..opts.max_iter {
            iterations += 1;
            // solve the convex (linearized, trust-region, virtual-control) subproblem about (xs, us)
            let (xc, uc, l_new, v_l1) = self.solve_subproblem(f, jac, &xs, &us, r);
            let j_new = self.penalized_cost(f, &xc, &uc);
            let d_j = j_cur - j_new; // actual reduction
            let d_l = j_cur - l_new; // predicted reduction

            // predicted no improvement ⇒ we are at (or numerically near) a stationary point
            if d_l <= opts.tol.max(1e-12) {
                if d_j.abs() < opts.tol {
                    converged = true;
                }
                // still take the step if it helped, then stop
                if d_j > 0.0 {
                    xs = xc;
                    us = uc;
                    j_cur = j_new;
                    last_virtual = v_l1;
                }
                radius_history.push(r);
                defect_history.push(self.defect(f, &xs, &us));
                if converged {
                    break;
                }
                r = (r / opts.alpha).max(opts.r_min);
                continue;
            }

            let rho = d_j / d_l;
            if rho < opts.rho0 {
                // reject: the nonlinear cost did not fall — shrink and retry from the same reference
                r = (r / opts.alpha).max(opts.r_min);
            } else {
                // accept the step
                xs = xc;
                us = uc;
                j_cur = j_new;
                last_virtual = v_l1;
                r = if rho < opts.rho1 {
                    (r / opts.alpha).max(opts.r_min)
                } else if rho < opts.rho2 {
                    r
                } else {
                    (r * opts.beta).min(opts.r_max)
                };
                if d_j >= 0.0 && d_j < opts.tol {
                    converged = true;
                }
            }
            radius_history.push(r);
            defect_history.push(self.defect(f, &xs, &us));
            if converged {
                break;
            }
        }

        ScvxReport {
            final_defect: self.defect(f, &xs, &us),
            final_fuel: self.fuel(&us),
            final_virtual: if last_virtual.is_finite() { last_virtual } else { self.defect(f, &xs, &us) },
            iterations,
            converged,
            radius_history,
            defect_history,
            xs,
            us,
        }
    }

    /// Assemble and solve the convex QP subproblem about the reference `(xbar, ubar)` with trust radius `r`.
    /// Returns `(x_new, u_new, L_new, ‖ν‖₁)` where `L_new = fuel(u_new) + λ‖ν‖₁` is the model's penalized
    /// cost at its optimum.
    fn solve_subproblem(
        &self,
        f: &DynFn,
        jac: &JacFn,
        xbar: &[DVector<f64>],
        ubar: &[DVector<f64>],
        r: f64,
    ) -> (Vec<DVector<f64>>, Vec<DVector<f64>>, f64, f64) {
        let (nx, nu, n) = (self.nx, self.nu, self.n);
        // variable layout: x_0..x_n (nx each), u_0..u_{n-1} (nu each), v_0..v_{n-1} (nx each, virtual),
        // t_0..t_{n-1} (nx each, ℓ₁ upper bounds of v)
        let xsz = (n + 1) * nx;
        let usz = n * nu;
        let vsz = n * nx;
        let tsz = n * nx;
        let nz = xsz + usz + vsz + tsz;
        let ix = |i: usize| i * nx;
        let iu = |i: usize| xsz + i * nu;
        let iv = |i: usize| xsz + usz + i * nx;
        let it = |i: usize| xsz + usz + vsz + i * nx;

        // ---- objective: min ½ zᵀP z + qᵀz = fuel·Σ‖u‖² + λ·Σ t ----
        let mut p = DMatrix::<f64>::zeros(nz, nz);
        let mut q = vec![0.0; nz];
        for i in 0..nz {
            p[(i, i)] = 1e-8; // tiny ridge for numerical PSD
        }
        for i in 0..n {
            for k in 0..nu {
                p[(iu(i) + k, iu(i) + k)] += 2.0 * self.fuel_weight; // ½·2·fuel = fuel
            }
            for k in 0..nx {
                q[it(i) + k] += self.lambda;
            }
        }

        // ---- equality constraints (ZeroCone): A_eq z = b_eq ----
        // E1: x_0 = x_init ; E2 (per i): x_{i+1} − A_d x_i − B_d u_i − v_i = c_i ; E3: terminal.
        let term_dims = nx - self.terminal_free_tail;
        let n_eq = nx + n * nx + term_dims;
        let mut a_eq = DMatrix::<f64>::zeros(n_eq, nz);
        let mut b_eq = vec![0.0; n_eq];
        // E1
        for k in 0..nx {
            a_eq[(k, ix(0) + k)] = 1.0;
            b_eq[k] = self.x_init[k];
        }
        let mut row = nx;
        for i in 0..n {
            let (ac, bc) = jac(&xbar[i], &ubar[i]);
            let ad = DMatrix::<f64>::identity(nx, nx) + self.dt * &ac;
            let bd = self.dt * &bc;
            let fd = self.step_d(f, &xbar[i], &ubar[i]);
            // c_i = f_d(x̄,ū) − A_d x̄ − B_d ū
            let ci = &fd - &ad * &xbar[i] - &bd * &ubar[i];
            for kr in 0..nx {
                a_eq[(row + kr, ix(i + 1) + kr)] = 1.0; // +I x_{i+1}
                for kc in 0..nx {
                    a_eq[(row + kr, ix(i) + kc)] = -ad[(kr, kc)]; // −A_d x_i
                }
                for kc in 0..nu {
                    a_eq[(row + kr, iu(i) + kc)] = -bd[(kr, kc)]; // −B_d u_i
                }
                a_eq[(row + kr, iv(i) + kr)] = -1.0; // −ν_i
                b_eq[row + kr] = ci[kr];
            }
            row += nx;
        }
        // E3 terminal (pin leading term_dims components of x_n)
        for k in 0..term_dims {
            a_eq[(row + k, ix(n) + k)] = 1.0;
            b_eq[row + k] = self.x_goal[k];
        }

        // ---- inequality constraints (NonnegativeCone): A_ineq z ≤ b_ineq ----
        // L1: v − t ≤ 0, −v − t ≤ 0 ; thrust box: u ≤ u_max, −u ≤ −u_min ;
        // trust region: (u−ū) ≤ r, −(u−ū) ≤ r, (x−x̄) ≤ r, −(x−x̄) ≤ r.
        let n_l1 = 2 * vsz;
        let n_box = 2 * usz;
        let n_tr_u = 2 * usz;
        let n_tr_x = 2 * xsz;
        let n_ineq = n_l1 + n_box + n_tr_u + n_tr_x;
        let mut a_in = DMatrix::<f64>::zeros(n_ineq, nz);
        let mut b_in = vec![0.0; n_ineq];
        let mut rr = 0;
        // L1 upper: v − t ≤ 0
        for i in 0..n {
            for k in 0..nx {
                a_in[(rr, iv(i) + k)] = 1.0;
                a_in[(rr, it(i) + k)] = -1.0;
                b_in[rr] = 0.0;
                rr += 1;
            }
        }
        // L1 lower: −v − t ≤ 0
        for i in 0..n {
            for k in 0..nx {
                a_in[(rr, iv(i) + k)] = -1.0;
                a_in[(rr, it(i) + k)] = -1.0;
                b_in[rr] = 0.0;
                rr += 1;
            }
        }
        // thrust box: u ≤ u_max
        for i in 0..n {
            for k in 0..nu {
                a_in[(rr, iu(i) + k)] = 1.0;
                b_in[rr] = self.u_max[k];
                rr += 1;
            }
        }
        // thrust box: −u ≤ −u_min
        for i in 0..n {
            for k in 0..nu {
                a_in[(rr, iu(i) + k)] = -1.0;
                b_in[rr] = -self.u_min[k];
                rr += 1;
            }
        }
        // trust region on u: (u−ū) ≤ r, −(u−ū) ≤ r
        for i in 0..n {
            for k in 0..nu {
                a_in[(rr, iu(i) + k)] = 1.0;
                b_in[rr] = ubar[i][k] + r;
                rr += 1;
            }
        }
        for i in 0..n {
            for k in 0..nu {
                a_in[(rr, iu(i) + k)] = -1.0;
                b_in[rr] = -ubar[i][k] + r;
                rr += 1;
            }
        }
        // trust region on x: (x−x̄) ≤ r, −(x−x̄) ≤ r
        for i in 0..=n {
            for k in 0..nx {
                a_in[(rr, ix(i) + k)] = 1.0;
                b_in[rr] = xbar[i][k] + r;
                rr += 1;
            }
        }
        for i in 0..=n {
            for k in 0..nx {
                a_in[(rr, ix(i) + k)] = -1.0;
                b_in[rr] = -xbar[i][k] + r;
                rr += 1;
            }
        }

        // ---- stack [A_eq; A_ineq], solve ----
        let mut a_all = DMatrix::<f64>::zeros(n_eq + n_ineq, nz);
        a_all.view_mut((0, 0), (n_eq, nz)).copy_from(&a_eq);
        a_all.view_mut((n_eq, 0), (n_ineq, nz)).copy_from(&a_in);
        let mut b_all = b_eq.clone();
        b_all.extend_from_slice(&b_in);

        let p_csc = csc_upper(&p);
        let a_csc = csc_dense(&a_all);
        let cones = [SupportedConeT::ZeroConeT(n_eq), SupportedConeT::NonnegativeConeT(n_ineq)];
        let settings = DefaultSettingsBuilder::default().verbose(false).max_iter(200).build().unwrap();
        let mut solver = DefaultSolver::new(&p_csc, &q, &a_csc, &b_all, &cones, settings).unwrap();
        solver.solve();
        let z = &solver.solution.x;

        let mut xc = Vec::with_capacity(n + 1);
        for i in 0..=n {
            xc.push(DVector::from_iterator(nx, (0..nx).map(|k| z[ix(i) + k])));
        }
        let mut uc = Vec::with_capacity(n);
        for i in 0..n {
            uc.push(DVector::from_iterator(nu, (0..nu).map(|k| z[iu(i) + k])));
        }
        let v_l1: f64 = (0..n).map(|i| (0..nx).map(|k| z[iv(i) + k].abs()).sum::<f64>()).sum();
        let l_new = self.fuel(&uc) + self.lambda * v_l1;
        (xc, uc, l_new, v_l1)
    }
}

/// Upper-triangular CSC of a dense symmetric matrix (clarabel wants `P` upper-triangular).
fn csc_upper(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.ncols();
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..=j {
            if p[(i, j)] != 0.0 {
                rowval.push(i);
                nzval.push(p[(i, j)]);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// Dense `m×n` matrix to CSC (column-major, zeros dropped).
fn csc_dense(a: &DMatrix<f64>) -> CscMatrix<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..m {
            let v = a[(i, j)];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- 2-D rocket soft-landing (scaled units): x = [px,py,vx,vy,m], u = [Tx,Ty] ----
    const G: f64 = 1.0; // scaled gravity
    const ALPHA: f64 = 0.05; // mass depletion per unit thrust magnitude

    fn rocket_f(x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        let m = x[4].max(0.2);
        let tn = (u[0] * u[0] + u[1] * u[1] + 1e-9).sqrt();
        DVector::from_vec(vec![x[2], x[3], u[0] / m, u[1] / m - G, -ALPHA * tn])
    }

    fn rocket_jac(x: &DVector<f64>, u: &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        let m = x[4].max(0.2);
        let tn = (u[0] * u[0] + u[1] * u[1] + 1e-9).sqrt();
        let mut a = DMatrix::zeros(5, 5);
        a[(0, 2)] = 1.0;
        a[(1, 3)] = 1.0;
        a[(2, 4)] = -u[0] / (m * m);
        a[(3, 4)] = -u[1] / (m * m);
        let mut b = DMatrix::zeros(5, 2);
        b[(2, 0)] = 1.0 / m;
        b[(3, 1)] = 1.0 / m;
        b[(4, 0)] = -ALPHA * u[0] / tn;
        b[(4, 1)] = -ALPHA * u[1] / tn;
        (a, b)
    }

    fn straight_line(x0: &DVector<f64>, xg: &DVector<f64>, n: usize) -> Vec<DVector<f64>> {
        (0..=n).map(|i| x0 + (xg - x0) * (i as f64 / n as f64)).collect()
    }

    #[test]
    fn the_analytic_jacobian_matches_finite_differences() {
        let x = DVector::from_vec(vec![-2.0, 3.0, 0.5, -0.4, 0.9]);
        let u = DVector::from_vec(vec![0.3, 1.1]);
        let (a, b) = rocket_jac(&x, &u);
        let eps = 1e-6;
        for j in 0..5 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps;
            xm[j] -= eps;
            let fd = (rocket_f(&xp, &u) - rocket_f(&xm, &u)) / (2.0 * eps);
            for i in 0..5 {
                assert!((a[(i, j)] - fd[i]).abs() < 1e-5, "A[{i},{j}] {} vs fd {}", a[(i, j)], fd[i]);
            }
        }
        for j in 0..2 {
            let mut up = u.clone();
            let mut um = u.clone();
            up[j] += eps;
            um[j] -= eps;
            let fd = (rocket_f(&x, &up) - rocket_f(&x, &um)) / (2.0 * eps);
            for i in 0..5 {
                assert!((b[(i, j)] - fd[i]).abs() < 1e-5, "B[{i},{j}] {} vs fd {}", b[(i, j)], fd[i]);
            }
        }
    }

    #[test]
    fn a_linear_system_converges_in_one_step_with_zero_defect() {
        // For LINEAR dynamics the linearization is exact, so SCvx must reach a zero-defect feasible
        // trajectory immediately — a correctness check on the whole subproblem machinery.
        // Horizon generous enough to be comfortably feasible, and a large initial trust region so it does
        // not bind — then the exactness of the linearization must show up as one-solve convergence.
        let n = 20;
        let f = |x: &DVector<f64>, u: &DVector<f64>| DVector::from_vec(vec![x[2], x[3], u[0], u[1]]);
        let jac = |_x: &DVector<f64>, _u: &DVector<f64>| {
            let mut a = DMatrix::zeros(4, 4);
            a[(0, 2)] = 1.0;
            a[(1, 3)] = 1.0;
            (a, DMatrix::from_row_slice(4, 2, &[0., 0., 0., 0., 1., 0., 0., 1.]))
        };
        let x0 = DVector::from_vec(vec![-3.0, 2.0, 0.0, 0.0]);
        let xg = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let prob = ScvxProblem {
            nx: 4, nu: 2, n, dt: 0.2,
            x_init: x0.clone(), x_goal: xg.clone(), terminal_free_tail: 0,
            u_min: DVector::from_vec(vec![-5.0, -5.0]), u_max: DVector::from_vec(vec![5.0, 5.0]),
            fuel_weight: 1.0, lambda: 1e4,
        };
        let xs = straight_line(&x0, &xg, n);
        let us = vec![DVector::zeros(2); n];
        let opts = ScvxOpts { r0: 20.0, ..Default::default() };
        let rep = prob.solve(&f, &jac, xs, us, opts);
        assert!(rep.final_defect < 1e-5, "linear system should reach ~zero defect: {}", rep.final_defect);
        assert!(rep.iterations <= 3, "linear system should converge almost immediately: {} iters", rep.iterations);
    }

    // The 2-D rocket soft-landing problem (a comfortably-feasible instance) shared by the tests below.
    fn rocket_problem() -> (ScvxProblem, Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let n = 30;
        let x0 = DVector::from_vec(vec![-3.0, 4.0, 0.3, 0.0, 1.0]);
        let xg = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]); // mass (index 4) left free
        let prob = ScvxProblem {
            nx: 5, nu: 2, n, dt: 0.15,
            x_init: x0.clone(), x_goal: xg.clone(), terminal_free_tail: 1,
            u_min: DVector::from_vec(vec![-6.0, 0.0]), u_max: DVector::from_vec(vec![6.0, 10.0]),
            fuel_weight: 1.0, lambda: 5e3,
        };
        let xs = straight_line(&x0, &xg, n); // dynamically-infeasible straight-line guess
        let us = vec![DVector::from_vec(vec![0.0, G]); n]; // hover-ish control guess
        (prob, xs, us)
    }

    fn rocket_opts() -> ScvxOpts {
        ScvxOpts { r0: 1.5, max_iter: 150, ..Default::default() }
    }

    #[test]
    fn scvx_lands_the_rocket_from_an_infeasible_guess() {
        // THE HEADLINE. Start from a dynamically-infeasible straight line; SCvx must produce a trajectory
        // that (a) satisfies the nonlinear dynamics (defect → 0), (b) hits the pad with zero velocity, and
        // (c) obeys the thrust box — all while the fuel proxy stays finite.
        let (prob, xs, us) = rocket_problem();
        let init_defect = prob.defect(&rocket_f, &xs, &us);
        let rep = prob.solve(&rocket_f, &rocket_jac, xs, us, rocket_opts());

        assert!(init_defect > 0.1, "the straight-line guess should be clearly infeasible: {init_defect}");
        assert!(rep.converged, "SCvx should report convergence");
        assert!(rep.final_defect < 1e-3, "SCvx should drive the dynamics defect to ~0: {}", rep.final_defect);
        let xn = rep.xs.last().unwrap();
        // landed: position and velocity at the pad
        assert!(xn[0].abs() < 1e-2 && xn[1].abs() < 1e-2, "should land on the pad: pos ({},{})", xn[0], xn[1]);
        assert!(xn[2].abs() < 1e-2 && xn[3].abs() < 1e-2, "should touch down at rest: vel ({},{})", xn[2], xn[3]);
        // thrust box respected
        for u in &rep.us {
            assert!(u[0] >= -6.0 - 1e-4 && u[0] <= 6.0 + 1e-4 && u[1] >= -1e-4 && u[1] <= 10.0 + 1e-4, "thrust in box: {u:?}");
        }
        assert!(rep.final_fuel.is_finite() && rep.final_fuel > 0.0, "fuel finite and positive");
    }

    #[test]
    fn the_virtual_controls_vanish_at_convergence() {
        // The exact-penalty ℓ₁ term drives the virtual controls to zero — at convergence the linear model
        // needs no slack, which is exactly dynamic feasibility (recursive feasibility, no artificial
        // infeasibility left).
        let (prob, xs, us) = rocket_problem();
        let rep = prob.solve(&rocket_f, &rocket_jac, xs, us, rocket_opts());
        assert!(rep.final_virtual < 1e-3, "virtual controls should vanish at convergence: {}", rep.final_virtual);
    }

    #[test]
    fn the_defect_decreases_over_the_iterations() {
        // SCvx makes monotone-in-spirit progress: the trajectory's dynamic infeasibility at the end is far
        // below where it started, and the trust radius adapts (it is not frozen).
        let (prob, xs, us) = rocket_problem();
        let rep = prob.solve(&rocket_f, &rocket_jac, xs, us, rocket_opts());
        let first = rep.defect_history.first().unwrap();
        let last = rep.defect_history.last().unwrap();
        assert!(last < &(first * 0.01), "defect should fall by >100×: {first} → {last}");
        let rmin = rep.radius_history.iter().cloned().fold(f64::INFINITY, f64::min);
        let rmax = rep.radius_history.iter().cloned().fold(0.0, f64::max);
        assert!(rmax > rmin, "the trust radius should adapt, not stay frozen: [{rmin}, {rmax}]");
    }
}
