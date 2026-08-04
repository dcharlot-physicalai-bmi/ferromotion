//! **What a generative policy's sampling error costs a closed loop** — P1's base case, and four corrections to
//! the conjecture it was supposed to confirm.
//!
//! The flagship conjecture of the generative-policy agenda is that a per-step action error `η` costs closed-loop
//! regret `≲ η/λ`, horizon-free, with `λ` the contraction rate. The linear-quadratic case admits an *exact*
//! answer, and it says the conjecture is wrong in three separate ways and incomplete in a fourth. This module
//! is that answer, and every quantity in it is computable **before a policy is trained**.
//!
//! Setup: a linear plant `x⁺ = Ax + Bu + w`, quadratic cost, a stabilising expert `u = −Kx`, and a policy that
//! reproduces the expert up to a residual `u = −Kx + e`. The residual's three cases behave completely
//! differently, which is the first thing the conjecture misses.
//!
//! # 1. The exponent is two, not one
//!
//! [`regret_variance`] is exact: `tr[(BᵀPB + R)Σₑ]` with `P` the closed-loop cost Gramian. That is **quadratic**
//! in `η`, because expanding a smooth cost around the expert's trajectory kills the first-order term whenever
//! the error is zero-mean. The conjectured linear form is right for a *Lipschitz* cost — a task that either
//! completes or does not — and wrong for a smooth one. So the exponent is set by the cost class, not the policy,
//! and any general statement has to carry the cost class as a hypothesis:
//!
//! | cost | leading term | exponent in `η` |
//! |---|---|---|
//! | Lipschitz (task success, distance-to-goal) | `L·E‖Δx‖` | 1 |
//! | smooth quadratic (tracking, energy) | `½E[Δxᵀ∇²c Δx]` | **2** |
//! | smooth, but the error is *biased* | first order survives | 1 in the bias |
//!
//! # 2. Bias and variance are not interchangeable
//!
//! At equal `η`, a systematic bias costs `Θ(‖b‖²/λ²)` where zero-mean noise costs `Θ(tr Σₑ/λ)` — worse by a
//! factor `1/λ`. A policy whose score error averages out and one whose does not are not the same policy with
//! different luck. [`regret_bias`] is the exact expression.
//!
//! # 3. The sharp constant is a directional gain, not `1/λ`
//!
//! `‖BᵀPB + R‖` is the squared `H₂` gain of the error-to-cost channel. The `1/λ` bound is attained **only when
//! the error direction excites the slow closed-loop mode**; when it does not, the true constant is smaller by
//! orders of magnitude. A theorem in `λ` alone is loose where the error is benignly aligned and — worse — gives
//! no warning where a small `η` points straight at the slow mode. [`h2_gain`] computes the real constant.
//!
//! # 4. A state-dependent error has a cliff
//!
//! An error proportional to the *state* rather than an offset — which is what a systematically mis-fit score
//! produces — destabilises the loop once its gain passes [`stability_margin`], the `H∞` margin
//! `1/supω‖(e^{jω}I − A_K)⁻¹B‖`. Below it the horizon-free bound holds; above it the cost grows as `e^{Θ(H)}` and
//! no horizon-free reasoning applies. The margin is a property of the plant and the expert, computable in
//! advance.
//!
//! And a plumbing note that decides which generative results are usable at all: the composition must run
//! through `W₂`. A total-variation bound does not control `E‖e‖²`, because two laws can be TV-close while the
//! learner's rare actions are arbitrarily large — and a quadratic cost charges for exactly those.

use nalgebra::{Complex, DMatrix, DVector};

/// A linear-quadratic loop: plant, cost, and a stabilising expert gain.
#[derive(Clone, Debug)]
pub struct LqLoop {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    /// The expert's feedback gain, so the expert is `u = −Kx`.
    pub k: DMatrix<f64>,
}

impl LqLoop {
    /// The closed loop `A_K = A − BK`.
    pub fn closed_loop(&self) -> DMatrix<f64> {
        &self.a - &self.b * &self.k
    }

    /// Spectral radius of the closed loop.
    pub fn rho(&self) -> f64 {
        self.closed_loop().complex_eigenvalues().iter().fold(0.0f64, |m, l| m.max(l.norm()))
    }

    /// The per-step contraction rate `λ = −log ρ(A_K)`. `None` if the expert does not stabilise.
    pub fn contraction_rate(&self) -> Option<f64> {
        let rho = self.rho();
        (rho > 0.0 && rho < 1.0).then(|| -rho.ln())
    }

    /// The **closed-loop cost Gramian** `P`, the unique solution of `P = M + A_Kᵀ P A_K` with
    /// `M = Q + KᵀRK`. This is the object that turns an action error into a cost, and it carries the loop's
    /// directional structure — which is why `1/λ` is only a bound on what it does.
    pub fn cost_gramian(&self) -> Option<DMatrix<f64>> {
        let m = &self.q + self.k.transpose() * &self.r * &self.k;
        ferromotion_core::solve_lyapunov_discrete(&self.closed_loop(), &m)
    }

    /// **Exact regret of a zero-mean sampling error**: `tr[(BᵀPB + R)Σₑ]`.
    ///
    /// No horizon appears. A generative policy's regret on a contracting linear loop is a *constant*, and it is
    /// quadratic in the per-step error rather than linear.
    pub fn regret_variance(&self, sigma_e: &DMatrix<f64>) -> Option<f64> {
        let p = self.cost_gramian()?;
        let bt_p_b = self.b.transpose() * &p * &self.b;
        Some(((bt_p_b + &self.r) * sigma_e).trace())
    }

    /// **Exact regret of a constant action bias** `e ≡ b`: the loop settles at `x̄ = (I − A_K)⁻¹Bb` and pays
    /// `x̄ᵀQx̄ + ūᵀRū` forever.
    ///
    /// Compare the scaling with [`regret_variance`](Self::regret_variance): this one carries `(I − A_K)⁻¹`
    /// *squared*, hence `1/λ²`, while the variance term carries `P ~ 1/λ`.
    pub fn regret_bias(&self, bias: &DVector<f64>) -> Option<f64> {
        let n = self.a.nrows();
        let x_bar = (DMatrix::identity(n, n) - self.closed_loop()).lu().solve(&(&self.b * bias))?;
        let u_bar = -(&self.k * &x_bar) + bias;
        Some((x_bar.transpose() * &self.q * &x_bar)[0] + (u_bar.transpose() * &self.r * &u_bar)[0])
    }

    /// The **sharp constant**: `‖BᵀPB + R‖`, the squared `H₂` gain of the error-to-cost channel.
    ///
    /// This is the number the conjecture should have been stated in. It is a *directional* gain, so two loops
    /// with the same `λ` can differ in it by orders of magnitude depending on whether `B` excites the slow mode.
    pub fn h2_gain(&self) -> Option<f64> {
        let p = self.cost_gramian()?;
        let m = self.b.transpose() * &p * &self.b + &self.r;
        Some(spectral_norm_sym(&m))
    }

    /// The **rigorous** `1/λ`-style bound on the sharp constant, using `1 − ρ²` rather than its small-margin
    /// approximation: `κ²‖M‖‖B‖²/(1 − ρ²) + ‖R‖`.
    ///
    /// This is a genuine upper bound on [`h2_gain`](Self::h2_gain). Its ratio to the true constant is how much a
    /// statement in the spectral rate alone gives away, and on a loop whose error direction misses the slow mode
    /// that ratio is large.
    pub fn h2_gain_bound(&self, kappa: f64) -> Option<f64> {
        let rho = self.rho();
        if !(0.0..1.0).contains(&rho) {
            return None;
        }
        let m = &self.q + self.k.transpose() * &self.r * &self.k;
        Some(kappa * kappa * spectral_norm_sym(&m) * self.b.norm().powi(2) / (1.0 - rho * rho) + spectral_norm_sym(&self.r))
    }

    /// The `κ²‖M‖‖B‖²/(2λ) + ‖R‖` form as usually written — an **asymptotic estimate, not a bound**.
    ///
    /// It comes from approximating `1 − ρ²` by `2λ`, which holds only as `ρ → 1`. At `ρ = 0.9` the two differ by
    /// about 11% in the wrong direction, so the expression sits *below* the true constant and is not a bound
    /// there. Worth having beside [`h2_gain_bound`](Self::h2_gain_bound) precisely because the difference is easy
    /// to miss: the source presents it with `≲` and `≈`, and the approximation is where the `≈` is doing work.
    pub fn h2_gain_lambda_estimate(&self, kappa: f64) -> Option<f64> {
        let lam = self.contraction_rate()?;
        let m = &self.q + self.k.transpose() * &self.r * &self.k;
        Some(kappa * kappa * spectral_norm_sym(&m) * self.b.norm().powi(2) / (2.0 * lam) + spectral_norm_sym(&self.r))
    }

    /// The **`H∞` stability margin**: the smallest `‖ΔK‖` that can destabilise the loop,
    /// `1/supω‖(e^{jω}I − A_K)⁻¹B‖`.
    ///
    /// A state-proportional error below this is safe forever; above it the loop is unstable and the
    /// finite-horizon cost grows exponentially. This is the linear image of the exponential-compounding lower
    /// bound the whole agenda is built around, and it locates the transition exactly — from the plant and the
    /// expert, before any policy exists.
    pub fn stability_margin(&self, grid: usize) -> Option<f64> {
        let a_k = self.closed_loop();
        let n = a_k.nrows();
        let m = self.b.ncols();
        let n_grid = grid.max(16);
        let mut worst = 0.0f64;
        for i in 0..n_grid {
            let w = std::f64::consts::PI * i as f64 / (n_grid - 1) as f64;
            let z = Complex::new(w.cos(), w.sin());
            // (zI − A_K) as a complex matrix, solved against B
            let lhs = DMatrix::<Complex<f64>>::from_fn(n, n, |r, c| if r == c { z - Complex::new(a_k[(r, c)], 0.0) } else { -Complex::new(a_k[(r, c)], 0.0) });
            let rhs = DMatrix::<Complex<f64>>::from_fn(n, m, |r, c| Complex::new(self.b[(r, c)], 0.0));
            let g = lhs.lu().solve(&rhs)?;
            worst = worst.max(complex_spectral_norm(&g));
        }
        (worst > 0.0).then(|| 1.0 / worst)
    }

    /// The closed loop under a state-proportional error `e = ΔK x`, i.e. `A_K + BΔK`. Its spectral radius
    /// crossing one is what [`stability_margin`](Self::stability_margin) predicts.
    pub fn perturbed_rho(&self, delta_k: &DMatrix<f64>) -> f64 {
        (self.closed_loop() + &self.b * delta_k).complex_eigenvalues().iter().fold(0.0f64, |m, l| m.max(l.norm()))
    }

    /// **Henrici's departure from normality**, normalised: `√(‖A_K‖_F² − Σ|λᵢ|²) / ‖A_K‖_F`.
    ///
    /// The normalisation is not cosmetic. The raw departure has the units of the matrix, so it **conflates scale
    /// with non-normality** — placing a pole at 0.5 needs more gain than placing one at 0.99, which inflates
    /// `‖A_K‖_F` and makes a well-conditioned loop look more defective than a defective one. Comparing two
    /// families that way gets the ranking backwards, which is exactly what happened here before the divide.
    pub fn departure_from_normality(&self) -> f64 {
        let a_k = self.closed_loop();
        let frob_sq = a_k.iter().map(|v| v * v).sum::<f64>();
        let eig_sq = a_k.complex_eigenvalues().iter().map(|l| l.norm_sqr()).sum::<f64>();
        (frob_sq - eig_sq).max(0.0).sqrt() / frob_sq.sqrt().max(1e-300)
    }

    /// The **eigenvector condition number** `κ` of a 2-state closed loop — the quantity the corrections to P1
    /// actually need, since it is what multiplies the bound and inflates the exponents.
    ///
    /// Computed exactly for `2×2` by solving the characteristic quadratic and forming the eigenvectors, so a
    /// defective loop returns `INFINITY` rather than a large finite number that invites being ignored. `None`
    /// for other sizes, because nothing here should silently substitute a proxy for it.
    pub fn eigenvector_condition(&self) -> Option<f64> {
        let m = self.closed_loop();
        if m.nrows() != 2 {
            return None;
        }
        let (a, b, c, d) = (m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
        let disc = (a - d) * (a - d) + 4.0 * b * c;
        if disc < 1e-14 * (a * a + d * d + b * b + c * c).max(1e-300) {
            return Some(f64::INFINITY); // repeated eigenvalue: defective unless already diagonal
        }
        if disc < 0.0 {
            return Some(f64::INFINITY); // a complex pair; the real eigenvector basis does not exist
        }
        let sq = disc.sqrt();
        let (l1, l2) = (0.5 * (a + d + sq), 0.5 * (a + d - sq));
        // eigenvector of a 2x2 for eigenvalue l: (b, l - a) if b != 0, else (l - d, c)
        let vec_for = |l: f64| if b.abs() > 1e-300 { (b, l - a) } else { (l - d, c) };
        let (v1, v2) = (vec_for(l1), vec_for(l2));
        let norm = |v: (f64, f64)| (v.0 * v.0 + v.1 * v.1).sqrt();
        let (n1, n2) = (norm(v1), norm(v2));
        if n1 < 1e-300 || n2 < 1e-300 {
            return Some(f64::INFINITY);
        }
        let mut vm = DMatrix::zeros(2, 2);
        vm[(0, 0)] = v1.0 / n1;
        vm[(1, 0)] = v1.1 / n1;
        vm[(0, 1)] = v2.0 / n2;
        vm[(1, 1)] = v2.1 / n2;
        let sv = vm.singular_values();
        let (hi, lo) = (sv.iter().cloned().fold(0.0f64, f64::max), sv.iter().cloned().fold(f64::INFINITY, f64::min));
        Some(if lo > 1e-300 { hi / lo } else { f64::INFINITY })
    }
}

/// Ackermann pole placement for a 2-state single-input system. `p2 = None` places a **defective** double pole at
/// `p1`, which is how a non-normal closed loop is produced deliberately.
pub fn place_two(a: &DMatrix<f64>, b: &DMatrix<f64>, p1: f64, p2: Option<f64>) -> Option<DMatrix<f64>> {
    if a.nrows() != 2 || b.ncols() != 1 {
        return None;
    }
    let p2 = p2.unwrap_or(p1);
    let mut c = DMatrix::zeros(2, 2);
    c.set_column(0, &b.column(0));
    c.set_column(1, &(a * b).column(0));
    let phi = a * a - (p1 + p2) * a + (p1 * p2) * DMatrix::identity(2, 2);
    let sol = c.lu().solve(&phi)?; // C⁻¹ φ(A)
    Some(DMatrix::from_row_slice(1, 2, &[sol[(1, 0)], sol[(1, 1)]]))
}

/// The discrete-time LQR gain for `(A, B, Q, R)`, and the associated Riccati solution.
pub fn lqr_gain(a: &DMatrix<f64>, b: &DMatrix<f64>, q: &DMatrix<f64>, r: &DMatrix<f64>) -> DMatrix<f64> {
    crate::dlqr(a, b, q, r)
}

/// Largest eigenvalue magnitude of a symmetric matrix: its spectral norm.
fn spectral_norm_sym(m: &DMatrix<f64>) -> f64 {
    let sym = (m + m.transpose()) * 0.5;
    sym.symmetric_eigen().eigenvalues.iter().fold(0.0f64, |acc, l| acc.max(l.abs()))
}

/// Spectral norm of a complex matrix, as `√λ_max(GᴴG)`.
///
/// `GᴴG` is Hermitian, and its eigenvalues are obtained from the real symmetric embedding
/// `[[Re, −Im], [Im, Re]]`, whose spectrum is the Hermitian one doubled — so the largest is the same number.
fn complex_spectral_norm(g: &DMatrix<Complex<f64>>) -> f64 {
    let m = g.ncols();
    let gh_g = g.adjoint() * g;
    let mut real = DMatrix::zeros(2 * m, 2 * m);
    for i in 0..m {
        for j in 0..m {
            let z = gh_g[(i, j)];
            real[(i, j)] = z.re;
            real[(i + m, j + m)] = z.re;
            real[(i, j + m)] = -z.im;
            real[(i + m, j)] = z.im;
        }
    }
    spectral_norm_sym(&real).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The double integrator of the source note: `dt = 0.1`, unit state cost, input cost `0.1`, LQR expert.
    fn double_integrator() -> LqLoop {
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let k = lqr_gain(&a, &b, &q, &r);
        LqLoop { a, b, q, r, k }
    }

    /// **Reproducing the reference figures.** The source note reports, for this exact system, `λ = 0.10628282`,
    /// `ρ = 0.89917031`, and an exact regret of `0.013461266967` at `η = 0.3`. Matching those to many digits from
    /// an independent implementation is the strongest check available on the whole chain — the Riccati solve, the
    /// Lyapunov solve, and the regret formula together.
    #[test]
    fn the_exact_regret_reproduces_the_reference_value() {
        let loop_ = double_integrator();
        let lam = loop_.contraction_rate().expect("the expert stabilises");
        let rho = loop_.rho();
        let eta = 0.3;
        let sigma_e = DMatrix::from_row_slice(1, 1, &[eta * eta]);
        let regret = loop_.regret_variance(&sigma_e).expect("the Gramian exists");

        eprintln!("double integrator: rho {rho:.10} (reference 0.8991703059), lambda {lam:.10} (reference 0.1062828232)");
        eprintln!("exact regret at eta = {eta}: {regret:.12} (reference 0.013461266967)");
        assert!((rho - 0.899_170_305_888_774_6).abs() < 1e-8, "spectral radius: {rho:.12}");
        assert!((lam - 0.106_282_823_198_502).abs() < 1e-8, "contraction rate: {lam:.12}");
        assert!((regret - 0.013_461_266_967_080_068).abs() < 1e-9, "regret: {regret:.12}");
    }

    /// **The `H∞` margin reproduces the reference value**, and it really is where the loop destabilises: the
    /// spectral radius under a state-proportional error crosses one there. The source note locates the critical
    /// gain numerically at `2.6` against a predicted `2.5857`.
    #[test]
    fn the_stability_margin_reproduces_the_reference_and_predicts_destabilisation() {
        let loop_ = double_integrator();
        let margin = loop_.stability_margin(4096).expect("a margin exists");
        eprintln!("H-infinity margin {margin:.10} (reference 2.5857008967)");
        assert!((margin - 2.585_700_896_659_864).abs() < 1e-6, "margin: {margin:.10}");

        // locate the critical gain by bisection on the spectral radius, in the direction the margin is about
        let dir = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let rho_at = |alpha: f64| loop_.perturbed_rho(&(&dir * alpha));
        assert!(rho_at(0.5 * margin) < 1.0, "well inside the margin the loop is stable");
        let (mut lo, mut hi) = (0.5 * margin, 20.0);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if rho_at(mid) < 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let critical = 0.5 * (lo + hi);
        eprintln!("critical gain by bisection {critical:.6}, margin {margin:.6}, ratio {:.4}", critical / margin);
        assert!(critical >= margin - 1e-6, "the margin must be a lower bound on the critical gain");
        assert!(critical / margin < 1.05, "and a tight one: {:.4}x", critical / margin);
    }

    /// **The exponent is two.** Sweeping `η` and fitting a log-log slope to the exact regret gives 2, not 1. This
    /// is the correction that matters most for reading a training curve: halving the action error quarters the
    /// regret on a smooth cost, and only halves it on a Lipschitz one.
    #[test]
    fn the_regret_exponent_in_the_action_error_is_two() {
        let loop_ = double_integrator();
        let pts: Vec<(f64, f64)> = [0.05f64, 0.1, 0.2, 0.4, 0.8]
            .iter()
            .map(|&eta| (eta, loop_.regret_variance(&DMatrix::from_row_slice(1, 1, &[eta * eta])).unwrap()))
            .collect();
        let slope = log_log_slope(&pts);
        eprintln!("regret vs eta over a 16x range: log-log slope {slope:.6} (reference 2.0000)");
        assert!((slope - 2.0).abs() < 1e-9, "the exponent must be exactly 2, got {slope}");
        // and the constant is the H2 gain, exactly
        let g = loop_.h2_gain().unwrap();
        let predicted = g * 0.09;
        let actual = loop_.regret_variance(&DMatrix::from_row_slice(1, 1, &[0.09])).unwrap();
        assert!((predicted - actual).abs() < 1e-12, "single-input: the H2 gain IS the constant, {predicted} vs {actual}");
    }

    /// **Bias and variance scale differently in the stability margin**, so they are not interchangeable at equal
    /// `η`. Sweeping the closed-loop rate by pole placement: the bias regret grows like `1/λ²` and the noise
    /// regret like `1/λ`, so the ratio between them grows without bound as the margin shrinks.
    #[test]
    fn a_systematic_bias_costs_a_further_factor_of_one_over_lambda() {
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let eta = 0.3;

        let mut ratios = Vec::new();
        eprintln!("      rho    lambda    variance regret   bias regret     ratio");
        for &rho in &[0.5f64, 0.7, 0.85, 0.93, 0.98, 0.99] {
            let k = place_two(&a, &b, rho, Some(0.5)).unwrap();
            let l = LqLoop { a: a.clone(), b: b.clone(), q: q.clone(), r: r.clone(), k };
            let lam = l.contraction_rate().unwrap();
            let var = l.regret_variance(&DMatrix::from_row_slice(1, 1, &[eta * eta])).unwrap();
            let bias = l.regret_bias(&DVector::from_row_slice(&[eta])).unwrap();
            eprintln!("      {rho:.2}   {lam:.5}   {var:>14.6}   {bias:>11.6}   {:>8.3}", bias / var);
            ratios.push((lam, var, bias));
        }
        // the ratio must grow as the margin shrinks - that is the whole asymmetry
        let first = ratios.first().unwrap();
        let last = ratios.last().unwrap();
        assert!(last.2 / last.1 > 100.0 * (first.2 / first.1), "the bias-to-variance ratio must blow up as lambda shrinks");

        // and the exponents themselves: bias ~ 1/lambda^2, variance ~ 1/lambda
        let inv_lam: Vec<(f64, f64)> = ratios.iter().map(|(l, v, _)| (1.0 / l, *v)).collect();
        let inv_lam_bias: Vec<(f64, f64)> = ratios.iter().map(|(l, _, b)| (1.0 / l, *b)).collect();
        let (sv, sb) = (log_log_slope(&inv_lam), log_log_slope(&inv_lam_bias));
        eprintln!("log-log slopes against 1/lambda: variance {sv:.3}, bias {sb:.3} (reference: bias 1.86, theory 2)");
        assert!(sb > sv + 0.5, "the bias exponent must exceed the variance exponent: {sb} vs {sv}");
        assert!(sb > 1.4, "and be near 2, got {sb}");
    }

    /// **Non-normality changes the exponents, not just the constants.** Placing both poles together makes the
    /// closed loop defective; the departure from normality grows and the bias exponent inflates well past 2. A
    /// bound stated in the spectral rate alone is then wrong in its exponent, not merely loose.
    #[test]
    fn a_defective_closed_loop_inflates_the_exponents() {
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let eta = 0.3;

        let family = |second: Option<f64>| {
            let mut pts_b = Vec::new();
            let mut dep = 0.0f64;
            let mut kap = 0.0f64;
            for &rho in &[0.7f64, 0.85, 0.93, 0.98, 0.99] {
                let k = place_two(&a, &b, rho, second).unwrap();
                let l = LqLoop { a: a.clone(), b: b.clone(), q: q.clone(), r: r.clone(), k };
                let lam = l.contraction_rate().unwrap();
                pts_b.push((1.0 / lam, l.regret_bias(&DVector::from_row_slice(&[eta])).unwrap()));
                dep = dep.max(l.departure_from_normality());
                kap = kap.max(l.eigenvector_condition().unwrap_or(f64::INFINITY));
            }
            (log_log_slope(&pts_b), dep, kap)
        };
        let (slope_ok, dep_ok, kap_ok) = family(Some(0.5));
        let (slope_bad, dep_bad, kap_bad) = family(None);
        eprintln!("separated poles: bias slope {slope_ok:.3}, normalised departure {dep_ok:.4}, eigenvector cond {kap_ok:.3e}");
        eprintln!("defective poles: bias slope {slope_bad:.3}, normalised departure {dep_bad:.4}, eigenvector cond {kap_bad:.3e}");
        assert!(kap_bad > 1e3 * kap_ok, "the defective family must be vastly worse conditioned: {kap_bad:.2e} vs {kap_ok:.2e}");
        // The normalised departure ranks these two families BACKWARDS (0.70 for the defective one against 0.88
        // for the separated one), so it is not a substitute for the eigenvector condition number. Recorded
        // because Henrici's measure is the convenient one to reach for and it does not track kappa here: the
        // defective loop's repeated eigenvalue contributes more to the eigenvalue sum, shrinking the normalised
        // departure even as the eigenvector basis collapses.
        assert!(dep_ok > dep_bad, "the departure measure is expected to invert here; if it stops, this note is stale");
        assert!(slope_bad > slope_ok + 0.5, "its exponent must inflate: {slope_bad} vs {slope_ok}");
        assert!(slope_bad > 3.0, "the source note measures 3.72 here, got {slope_bad}");
    }

    /// **The `1/λ` bound is loose by a factor set by modal alignment.** On a scalar plant the error necessarily
    /// excites the only mode and the bound is nearly attained; on the double integrator the error direction
    /// barely touches the slow eigenvector and the true constant is orders of magnitude smaller. Same `λ`, same
    /// **The `1/lambda` growth is real but DIRECTIONAL.**
    ///
    /// This is the correction that matters most for a general theorem. On a scalar plant the error direction
    /// cannot avoid the only mode, and the sharp constant grows like `1/λ` cleanly. On the double integrator the
    /// error direction barely overlaps the slow eigenvector, and over the same range of `λ` the constant is
    /// **flat** — the `1/λ` growth appears only at far smaller margins. Same `λ`, same `η`, constants differing
    /// by orders of magnitude according to modal alignment.
    #[test]
    fn the_one_over_lambda_growth_appears_only_when_the_error_excites_the_slow_mode() {
        let rhos = [0.7f64, 0.85, 0.93, 0.98, 0.99];
        let scal = |rho: f64| LqLoop {
            a: DMatrix::from_row_slice(1, 1, &[1.0]),
            b: DMatrix::from_row_slice(1, 1, &[1.0]),
            q: DMatrix::identity(1, 1),
            r: DMatrix::from_row_slice(1, 1, &[0.1]),
            k: DMatrix::from_row_slice(1, 1, &[1.0 - rho]),
        };
        let scalar_pts: Vec<(f64, f64)> = rhos.iter().map(|&rho| { let l = scal(rho); (1.0 / l.contraction_rate().unwrap(), l.h2_gain().unwrap()) }).collect();

        let dt = 0.1;
        let (a, b) = (DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]), DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]));
        let di_pts: Vec<(f64, f64)> = rhos
            .iter()
            .map(|&rho| {
                let k = place_two(&a, &b, rho, Some(0.5)).unwrap();
                let l = LqLoop { a: a.clone(), b: b.clone(), q: DMatrix::identity(2, 2), r: DMatrix::from_row_slice(1, 1, &[0.1]), k };
                (1.0 / l.contraction_rate().unwrap(), l.h2_gain().unwrap())
            })
            .collect();

        let (ss, sd) = (log_log_slope(&scalar_pts), log_log_slope(&di_pts));
        eprintln!("H2 gain vs 1/lambda: scalar slope {ss:.3} (reference 0.983), double integrator slope {sd:.3} (reference: flat)");
        assert!((ss - 1.0).abs() < 0.15, "the scalar plant must show the 1/lambda growth cleanly, got {ss:.3}");
        assert!(sd.abs() < 0.25, "the double integrator must be flat over this range, got {sd:.3}");
        assert!(ss > sd + 0.7, "and the two must differ by the whole exponent: {ss:.3} vs {sd:.3}");

        // The rigorous bound holds in both cases, and is TIGHT where the error excites the only mode.
        let sc = scal(0.9);
        for (name, l) in [("scalar", sc.clone()), ("double integrator", double_integrator())] {
            let (g, bd) = (l.h2_gain().unwrap(), l.h2_gain_bound(1.0).unwrap());
            eprintln!("   {name}: true H2 gain {g:.4}, rigorous bound {bd:.4}, looseness {:.3}x", bd / g);
            assert!(bd >= g - 1e-12, "the rigorous bound must hold for {name}");
        }

        // And the 1/(2 lambda) form as usually written is an ESTIMATE, not a bound: at rho = 0.9 it sits BELOW
        // the true constant, because 1 - rho^2 = 0.19 while 2 lambda = 0.211.
        let (est, truth) = (sc.h2_gain_lambda_estimate(1.0).unwrap(), sc.h2_gain().unwrap());
        eprintln!("   the 1/(2 lambda) ESTIMATE is {est:.4} against a true constant of {truth:.4} - below it, so not a bound");
        assert!(est < truth, "the estimate should undershoot at rho = 0.9: {est} vs {truth}");
        let near = scal(0.999);
        let (e2, b2) = (near.h2_gain_lambda_estimate(1.0).unwrap(), near.h2_gain_bound(1.0).unwrap());
        eprintln!("   at rho = 0.999 the estimate {e2:.1} and the rigorous bound {b2:.1} agree to {:.2}%", 100.0 * (b2 - e2).abs() / b2);
        assert!((b2 - e2).abs() / b2 < 0.02, "the two must agree as rho -> 1, the regime it was derived for");
    }

    /// **The closed form against simulation, with common random numbers.**
    ///
    /// The regret being measured is `0.0135` against an absolute cost of order `1`, so an independent Monte Carlo
    /// on each policy would need on the order of `10⁸` steps to resolve it. Driving the expert and the learner
    /// with the **same process noise** cancels the shared term exactly and leaves only the difference, which is
    /// what makes the measurement possible at all. The source note reports 2.1% agreement on this system.
    #[test]
    fn monte_carlo_with_common_random_numbers_confirms_the_closed_form() {
        let l = double_integrator();
        let eta = 0.3;
        let sigma_w = 0.05f64; // process-noise standard deviation, shared between the two runs
        let predicted = l.regret_variance(&DMatrix::from_row_slice(1, 1, &[eta * eta])).unwrap();

        // deterministic Gaussian stream, so the test cannot flake
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut normal = || {
            let mut u = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            let (u1, u2) = (u().max(1e-12), u());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };

        let a_k = l.closed_loop();
        let m = &l.q + l.k.transpose() * &l.r * &l.k;
        let (steps, burn) = (4_000_000usize, 20_000usize);
        // expert and learner side by side under identical w_t
        let (mut xe, mut xl) = (DVector::zeros(2), DVector::zeros(2));
        let (mut ce, mut cl) = (0.0f64, 0.0f64);
        for t in 0..steps {
            let w = DVector::from_row_slice(&[sigma_w * normal(), sigma_w * normal()]);
            let e = eta * normal();
            if t >= burn {
                // cost of the expert: x' M x. cost of the learner: x' M x + 2 e' R (-Kx) ... expand exactly
                ce += (xe.transpose() * &m * &xe)[0];
                let ul = -(&l.k * &xl) + DVector::from_row_slice(&[e]);
                cl += (xl.transpose() * &l.q * &xl)[0] + (ul.transpose() * &l.r * &ul)[0];
            }
            xe = &a_k * &xe + &w;
            xl = &a_k * &xl + &l.b * DVector::from_row_slice(&[e]) + &w;
        }
        let n = (steps - burn) as f64;
        let measured = cl / n - ce / n;
        let rel = (measured - predicted).abs() / predicted;
        eprintln!("closed form {predicted:.8}, coupled Monte Carlo {measured:.8} over {} steps: {:.2}% apart (reference 2.1%)", steps - burn, 100.0 * rel);
        assert!(rel < 0.06, "the closed form must match simulation: {predicted} vs {measured} ({:.1}%)", 100.0 * rel);
    }

    /// Least-squares slope of `log y` against `log x`.
    fn log_log_slope(pts: &[(f64, f64)]) -> f64 {
        let n = pts.len() as f64;
        let (lx, ly): (Vec<f64>, Vec<f64>) = pts.iter().map(|(x, y)| (x.ln(), y.ln())).unzip();
        let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
        let num: f64 = lx.iter().zip(&ly).map(|(x, y)| (x - mx) * (y - my)).sum();
        let den: f64 = lx.iter().map(|x| (x - mx) * (x - mx)).sum();
        num / den
    }
}
