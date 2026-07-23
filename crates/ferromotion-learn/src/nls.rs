//! **Differentiable nonlinear least squares** — optimization as a layer, the theseus-class
//! capability (upstream frozen since 2024) in pure Rust.
//!
//! Solve `θ* = argmin_θ ½‖r(θ, φ)‖²` by damped Gauss–Newton, then differentiate the *solution*
//! with respect to the outer parameters `φ` by **implicit differentiation** of the optimality
//! condition: at the optimum `g(θ*, φ) = Jᵀr = 0`, so `dθ*/dφ = −(JᵀJ)⁻¹ Jᵀ ∂r/∂φ` with the
//! Gauss–Newton Hessian — theseus's default, and **exact whenever the optimum has zero residual**
//! (IK to a reachable target, noise-free fitting); at nonzero-residual optima it drops the
//! `Σ rᵢ∇²rᵢ` term and is the standard small-residual approximation (tested to stay within a
//! small band of the true finite-difference sensitivity). No unrolled solver iterations; each
//! outer parameter costs one already-factored solve plus one dual pass.
//!
//! Residuals are written ONCE, generically over [`Real`] — the same code evaluates at `f64` for
//! solving and at [`Dual`] for every Jacobian (`∂r/∂θ`, one forward pass per inner variable) and
//! every sensitivity seed (`∂r/∂φ`, one pass per outer parameter). Forward-mode scaling is the
//! honest trade for the small-to-medium `θ` this layer targets (IK, fitting, model correction);
//! it composes directly with [`ferromotion_core::gendyn`] — an IK layer through the real FK is a
//! ten-line residual (see tests). Pure Rust, wasm-clean.

use crate::dual::Dual;
use ferromotion_core::gendyn::{cholesky_solve, Real};

/// A residual family evaluated at any scalar: inner variables `theta`, outer parameters `phi`.
pub trait Residual {
    fn eval<T: Real>(&self, theta: &[T], phi: &[T]) -> Vec<T>;
}

/// Solver options: LM damping is adapted multiplicatively; iteration stops on gradient norm.
#[derive(Clone, Copy, Debug)]
pub struct NlsOptions {
    pub max_iters: usize,
    pub grad_tol: f64,
    pub damping0: f64,
}

impl Default for NlsOptions {
    fn default() -> Self {
        Self { max_iters: 60, grad_tol: 1e-10, damping0: 1e-4 }
    }
}

/// The solved layer: the optimum, its cost, and the machinery for exact outer sensitivities.
pub struct NlsSolution {
    pub theta: Vec<f64>,
    pub cost: f64,
    pub iters: usize,
    /// `JᵀJ` at the optimum (row-major n×n), kept for the implicit-differentiation solves.
    jtj: Vec<f64>,
    /// `J` at the optimum (row-major m×n).
    jac: Vec<f64>,
    n_res: usize,
}

fn dual_theta(theta: &[f64], seed: usize) -> Vec<Dual> {
    theta.iter().enumerate().map(|(i, &v)| if i == seed { Dual::var(v) } else { Dual::constant(v) }).collect()
}

fn consts(v: &[f64]) -> Vec<Dual> {
    v.iter().map(|&x| Dual::constant(x)).collect()
}

/// Jacobian `∂r/∂θ` (row-major m×n) by one dual pass per inner variable.
fn jacobian<R: Residual>(res: &R, theta: &[f64], phi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = theta.len();
    let phic = consts(phi);
    let r0: Vec<f64> = res.eval(&consts(theta), &phic).iter().map(|d| d.re).collect();
    let m = r0.len();
    let mut jac = vec![0.0; m * n];
    for j in 0..n {
        let rj = res.eval(&dual_theta(theta, j), &phic);
        assert_eq!(rj.len(), m, "residual count must not depend on θ");
        for i in 0..m {
            jac[i * n + j] = rj[i].eps;
        }
    }
    (r0, jac)
}

/// Solve the damped Gauss–Newton problem and return the differentiable solution layer.
pub fn solve<R: Residual>(res: &R, theta0: &[f64], phi: &[f64], opts: NlsOptions) -> NlsSolution {
    let n = theta0.len();
    let mut theta = theta0.to_vec();
    let mut damping = opts.damping0;
    let cost_of = |th: &[f64]| -> f64 {
        res.eval(&consts(th), &consts(phi)).iter().map(|d| d.re * d.re).sum::<f64>() * 0.5
    };
    let mut cost = cost_of(&theta);
    let mut iters = 0;
    for _ in 0..opts.max_iters {
        iters += 1;
        let (r0, jac) = jacobian(res, &theta, phi);
        let m = r0.len();
        // g = Jᵀ r, H = JᵀJ (+ λ diag)
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n * n];
        for i in 0..m {
            for a in 0..n {
                g[a] += jac[i * n + a] * r0[i];
                for b in 0..n {
                    h[a * n + b] += jac[i * n + a] * jac[i * n + b];
                }
            }
        }
        if g.iter().map(|v| v.abs()).fold(0.0, f64::max) < opts.grad_tol {
            break;
        }
        // LM: try the damped step; shrink damping on success, grow on failure
        let mut stepped = false;
        for _ in 0..20 {
            let mut hd = h.clone();
            for a in 0..n {
                hd[a * n + a] += damping * (1.0 + h[a * n + a]);
            }
            let step = cholesky_solve(&hd, &g, n);
            let cand: Vec<f64> = theta.iter().zip(&step).map(|(t, s)| t - s).collect();
            let c_new = cost_of(&cand);
            if c_new.is_finite() && c_new <= cost {
                theta = cand;
                cost = c_new;
                damping = (damping * 0.3).max(1e-12);
                stepped = true;
                break;
            }
            damping *= 10.0;
        }
        if !stepped {
            break;
        }
    }
    let (r0, jac) = jacobian(res, &theta, phi);
    let m = r0.len();
    let mut jtj = vec![0.0; n * n];
    for i in 0..m {
        for a in 0..n {
            for b in 0..n {
                jtj[a * n + b] += jac[i * n + a] * jac[i * n + b];
            }
        }
    }
    // tiny Tikhonov so the factorization is defined even at flat directions — reported honestly
    for a in 0..n {
        jtj[a * n + a] += 1e-12;
    }
    NlsSolution { theta, cost, iters, jtj, jac, n_res: m }
}

impl NlsSolution {
    /// Exact sensitivity `dθ*/dφ_k` by implicit differentiation: `−(JᵀJ)⁻¹ Jᵀ ∂r/∂φ_k`, with
    /// `∂r/∂φ_k` from one dual pass seeded in the outer parameter.
    pub fn dtheta_dphi<R: Residual>(&self, res: &R, phi: &[f64], k: usize) -> Vec<f64> {
        let n = self.theta.len();
        let phid: Vec<Dual> =
            phi.iter().enumerate().map(|(i, &v)| if i == k { Dual::var(v) } else { Dual::constant(v) }).collect();
        let rk = res.eval(&consts(&self.theta), &phid);
        assert_eq!(rk.len(), self.n_res);
        let mut jtr = vec![0.0; n];
        for (i, r) in rk.iter().enumerate() {
            for a in 0..n {
                jtr[a] += self.jac[i * n + a] * r.eps;
            }
        }
        cholesky_solve(&self.jtj, &jtr, n).iter().map(|v| -v).collect()
    }

    /// Chain an outer loss gradient through the layer: given `∂L/∂θ*`, return `∂L/∂φ` (one
    /// factored solve plus one dual pass per outer parameter).
    pub fn backward<R: Residual>(&self, res: &R, phi: &[f64], dl_dtheta: &[f64]) -> Vec<f64> {
        let n = self.theta.len();
        // adjoint: w = (JᵀJ)⁻¹ ∂L/∂θ ;  ∂L/∂φ_k = −wᵀ Jᵀ ∂r/∂φ_k
        let w = cholesky_solve(&self.jtj, dl_dtheta, n);
        let jt_w: Vec<f64> = (0..self.n_res).map(|i| (0..n).map(|a| self.jac[i * n + a] * w[a]).sum()).collect();
        (0..phi.len())
            .map(|k| {
                let phid: Vec<Dual> = phi
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if i == k { Dual::var(v) } else { Dual::constant(v) })
                    .collect();
                let rk = res.eval(&consts(&self.theta), &phid);
                -rk.iter().zip(&jt_w).map(|(r, w)| r.eps * w).sum::<f64>()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linear least squares has a CLOSED FORM for both the optimum and the sensitivity:
    /// θ* = (AᵀA)⁻¹Aᵀy and dθ*/dy = (AᵀA)⁻¹Aᵀ. The layer must reproduce both exactly.
    struct LinearFit {
        a: Vec<f64>, // m×n row-major
        m: usize,
        n: usize,
    }
    impl Residual for LinearFit {
        fn eval<T: Real>(&self, theta: &[T], phi: &[T]) -> Vec<T> {
            (0..self.m)
                .map(|i| {
                    let mut acc = T::from_f64(0.0) - phi[i];
                    for j in 0..self.n {
                        acc = acc + T::from_f64(self.a[i * self.n + j]) * theta[j];
                    }
                    acc
                })
                .collect()
        }
    }

    #[test]
    fn linear_problem_matches_the_closed_form_solution_and_sensitivity() {
        let (m, n) = (7usize, 3usize);
        let a: Vec<f64> = (0..m * n).map(|i| ((i * 7 + 3) % 11) as f64 * 0.3 - 1.0).collect();
        let y: Vec<f64> = (0..m).map(|i| (i as f64 * 0.9).sin()).collect();
        let prob = LinearFit { a: a.clone(), m, n };
        let sol = solve(&prob, &vec![0.0; n], &y, NlsOptions::default());

        // closed form via normal equations
        let mut ata = vec![0.0; n * n];
        let mut aty = vec![0.0; n];
        for i in 0..m {
            for p in 0..n {
                aty[p] += a[i * n + p] * y[i];
                for q in 0..n {
                    ata[p * n + q] += a[i * n + p] * a[i * n + q];
                }
            }
        }
        let closed = cholesky_solve(&ata, &aty, n);
        for (got, want) in sol.theta.iter().zip(&closed) {
            assert!((got - want).abs() < 1e-8, "θ*: {got} vs {want}");
        }
        // dθ*/dy_k column: (AᵀA)⁻¹ Aᵀ e_k
        for k in [0usize, 3, 6] {
            let col_want = {
                let atk: Vec<f64> = (0..n).map(|p| a[k * n + p]).collect();
                cholesky_solve(&ata, &atk, n)
            };
            let col_got = sol.dtheta_dphi(&prob, &y, k);
            for (g, w) in col_got.iter().zip(&col_want) {
                assert!((g - w).abs() < 1e-8, "dθ/dy[{k}]: {g} vs {w}");
            }
        }
    }

    /// Nonlinear (exponential fit): the implicit sensitivity must match finite differences of
    /// RE-SOLVING the whole problem — the strong general oracle.
    struct ExpFit {
        xs: Vec<f64>,
    }
    impl Residual for ExpFit {
        fn eval<T: Real>(&self, theta: &[T], phi: &[T]) -> Vec<T> {
            // r_i = a·exp(b·x_i) − y_i  with θ = (a, b), φ = y
            self.xs
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let bx = theta[1] * T::from_f64(x);
                    // exp via tanh-free identity: e^z = (1+tanh(z/2))/(1−tanh(z/2))
                    let t = (bx * T::from_f64(0.5)).tanh();
                    let ez = (T::from_f64(1.0) + t) / (T::from_f64(1.0) - t);
                    theta[0] * ez - phi[i]
                })
                .collect()
        }
    }

    #[test]
    fn nonlinear_sensitivity_matches_finite_differences_of_resolving() {
        let xs: Vec<f64> = (0..9).map(|i| i as f64 * 0.25).collect();
        // ZERO-residual data (generated exactly by the model): at r* = 0 the Gauss–Newton
        // sensitivity is the EXACT sensitivity, so tolerances are tight.
        let y: Vec<f64> = xs.iter().map(|&x| 1.7 * (0.6 * x).exp()).collect();
        let prob = ExpFit { xs: xs.clone() };
        let opts = NlsOptions { max_iters: 120, ..Default::default() };
        let sol = solve(&prob, &[1.0, 0.1], &y, opts);
        assert!(sol.cost < 1e-16, "exact fit must converge to zero residual: cost {}", sol.cost);

        let eps = 1e-6;
        for k in [0usize, 4, 8] {
            let got = sol.dtheta_dphi(&prob, &y, k);
            let (mut yp, mut ym) = (y.clone(), y.clone());
            yp[k] += eps;
            ym[k] -= eps;
            let sp = solve(&prob, &sol.theta, &yp, opts);
            let sm = solve(&prob, &sol.theta, &ym, opts);
            for j in 0..2 {
                let want = (sp.theta[j] - sm.theta[j]) / (2.0 * eps);
                assert!((got[j] - want).abs() < 1e-5 * want.abs().max(1.0), "dθ{j}/dy{k}: {} vs {want}", got[j]);
            }
        }

        // NOISY data (nonzero residual at the optimum): the Gauss–Newton Hessian drops the
        // Σ rᵢ∇²rᵢ term, so the implicit sensitivity is an APPROXIMATION — the same one theseus
        // ships as default. Verify it stays within a small relative band of the true (FD) value.
        let yn: Vec<f64> = xs.iter().map(|&x| 1.7 * (0.6 * x).exp() + 0.03 * (x * 5.0).sin()).collect();
        let soln = solve(&prob, &[1.0, 0.1], &yn, opts);
        for k in [0usize, 8] {
            let got = soln.dtheta_dphi(&prob, &yn, k);
            let (mut yp, mut ym) = (yn.clone(), yn.clone());
            yp[k] += eps;
            ym[k] -= eps;
            let sp = solve(&prob, &soln.theta, &yp, opts);
            let sm = solve(&prob, &soln.theta, &ym, opts);
            for j in 0..2 {
                let want = (sp.theta[j] - sm.theta[j]) / (2.0 * eps);
                assert!((got[j] - want).abs() < 5e-3 * want.abs().max(1.0), "GN-approx band dθ{j}/dy{k}: {} vs {want}", got[j]);
            }
        }

        // backward(): chain an outer loss L = Σ θ² and compare to FD of the composed pipeline
        let dl_dtheta: Vec<f64> = sol.theta.iter().map(|t| 2.0 * t).collect();
        let gphi = sol.backward(&prob, &y, &dl_dtheta);
        for k in [2usize, 6] {
            let (mut yp, mut ym) = (y.clone(), y.clone());
            yp[k] += eps;
            ym[k] -= eps;
            let lp: f64 = solve(&prob, &sol.theta, &yp, opts).theta.iter().map(|t| t * t).sum();
            let lm: f64 = solve(&prob, &sol.theta, &ym, opts).theta.iter().map(|t| t * t).sum();
            let want = (lp - lm) / (2.0 * eps);
            assert!((gphi[k] - want).abs() < 1e-4 * want.abs().max(1.0), "dL/dy{k}: {} vs {want}", gphi[k]);
        }
    }

    /// The robotics payoff: IK as a differentiable layer THROUGH the real generic-scalar FK.
    /// θ = joint angles, φ = the Cartesian target; dθ*/dφ answers "how do the joints move as the
    /// target moves" — exactly, from the optimality condition, verified against FD re-solves.
    struct IkResidual {
        robot: ferromotion_core::Robot,
        inertia: Vec<ferromotion_core::LinkInertia>,
    }
    impl Residual for IkResidual {
        fn eval<T: Real>(&self, theta: &[T], phi: &[T]) -> Vec<T> {
            use ferromotion_core::gendyn::V3;
            let m = ferromotion_core::gendyn::GenModel::<T>::from_robot(&self.robot, &self.inertia, [0.0, 0.0, 0.0]);
            // tool point 0.25 along the last link's z — gives the WRIST joint position leverage
            let (r, p) = m.fk_frame(theta);
            let tool = r.mul_v(V3::from_f64([0.0, 0.0, 0.25]));
            let pt = p.add(tool);
            vec![pt.0[0] - phi[0], pt.0[1] - phi[1], pt.0[2] - phi[2]]
        }
    }

    #[test]
    fn ik_as_a_differentiable_layer_through_the_real_fk() {
        use ferromotion_core::{Iso, Joint, LinkInertia, Robot};
        use nalgebra::{Matrix3, Translation3, UnitQuaternion, Vector3};
        let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
        let robot = Robot {
            joints: vec![
                Joint::revolute(mk(0.1), Vector3::z()),
                Joint::revolute(mk(0.3), Vector3::y()),
                Joint::revolute(mk(0.3), Vector3::y()),
            ],
            ee_offset: Iso::identity(),
        };
        let inertia = vec![
            LinkInertia { mass: 1.0, com: Vector3::zeros(), inertia: Matrix3::identity() * 0.01 };
            3
        ];
        let prob = IkResidual { robot, inertia };
        let target = [0.25, 0.1, 0.45];
        let opts = NlsOptions { max_iters: 200, ..Default::default() };
        let sol = solve(&prob, &[0.3, -0.5, 0.8], &target, opts);
        assert!(sol.cost < 1e-16, "IK must reach the target: cost {}", sol.cost);

        let eps = 1e-6;
        for k in 0..3 {
            let got = sol.dtheta_dphi(&prob, &target, k);
            let (mut tp, mut tm) = (target, target);
            tp[k] += eps;
            tm[k] -= eps;
            let sp = solve(&prob, &sol.theta, &tp, opts);
            let sm = solve(&prob, &sol.theta, &tm, opts);
            for j in 0..3 {
                let want = (sp.theta[j] - sm.theta[j]) / (2.0 * eps);
                assert!((got[j] - want).abs() < 1e-4 * want.abs().max(1.0), "dq{j}/dtarget{k}: {} vs {want}", got[j]);
            }
        }
    }
}
