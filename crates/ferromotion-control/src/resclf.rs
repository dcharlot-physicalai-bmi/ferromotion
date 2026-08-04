//! **Rapidly exponentially stabilizing control Lyapunov functions** — the real-time bridge between a
//! reduced-order certificate and a full-order robot.
//!
//! The problem this solves is specific. A hybrid-zero-dynamics gait ([`hzd`](crate::hzd)) is certified on a
//! reduced manifold, and the impact that supplies its contraction is *expansive* off that manifold: a
//! perturbation transverse to the orbit is amplified by some `μ > 1` at every footfall. Continuous feedback
//! has to out-contract that between footfalls, and ordinary exponential stability gives no handle on
//! whether it does — the rate is whatever it happens to be.
//!
//! A **RES-CLF** makes the rate a design parameter. It is a control Lyapunov function on the output
//! coordinates `η` carrying a knob `ε`, satisfying
//!
//! * `c₁‖η‖² ≤ V_ε(η) ≤ (c₂/ε²)‖η‖²`, and
//! * a controller exists with `V̇_ε ≤ −(c₃/ε)V_ε`,
//!
//! so the convergence rate `c₃/ε` can be made arbitrarily fast by shrinking `ε`, at the price of gain.
//! Ames-Galloway-Sreenath-Grizzle's theorem is then: if the reduced orbit is exponentially stable *and* a
//! RES-CLF exists, then for `ε` small enough **any** Lipschitz controller in the CLF-descent set stabilises
//! the full-order hybrid orbit. The rapid continuous contraction dominates the impact expansion.
//!
//! "Small enough" is not vague. This module's [`ResClf::c3`] is precisely the constant that
//! [`hybrid_certificate`](ferromotion_core::hybrid_certificate) already consumes to report `ε̄ = c₃T/(2 ln μ)`
//! — the largest `ε` at which the per-step composition still contracts. The two halves were built
//! separately and the test at the bottom of this file closes the loop between them: pick `ε` below `ε̄` and
//! the certificate holds, pick it above and it fails, with the crossover where the formula says.
//!
//! The descent condition is realised as the **CLF-QP**, `min ½‖u‖²` subject to one linear inequality. With a
//! single constraint the solution is a projection onto a half-space and needs no solver, which is why this
//! runs at control rates.

use nalgebra::{DMatrix, DVector};

/// A rapidly exponentially stabilizing control Lyapunov function for output dynamics
/// `η̇ = F η + G u`, built from the solution of the associated Riccati equation.
#[derive(Clone, Debug)]
pub struct ResClf {
    /// The Riccati solution defining the quadratic form.
    pub p: DMatrix<f64>,
    /// Lower sandwich constant, `λ_min(P)`.
    pub c1: f64,
    /// Upper sandwich constant, so the upper bound is `c₂/ε²`.
    pub c2: f64,
    /// Decay constant, so the guaranteed rate is `c₃/ε`. **This is the constant a hybrid certificate
    /// consumes**, and the reason the two modules compose.
    pub c3: f64,
    f: DMatrix<f64>,
    g: DMatrix<f64>,
    /// Number of output coordinates, so `η = [y; ẏ]` has dimension `2·outputs`.
    outputs: usize,
}

impl ResClf {
    /// Build a RES-CLF for a set of `outputs` relative-degree-two virtual constraints.
    ///
    /// The output dynamics are the double integrator `η̇ = Fη + Gu` with `η = [y; ẏ]`, which is what
    /// relative degree two means: the input reaches the output through exactly two derivatives, so after
    /// feedback linearisation every such constraint looks like this regardless of the robot underneath.
    /// `q` weights the state in the Riccati equation and sets the trade between rate and gain.
    ///
    /// `None` if `q` is not the right size or the Riccati iteration fails to converge.
    pub fn double_integrator(outputs: usize, q: &DMatrix<f64>) -> Option<ResClf> {
        let n = 2 * outputs;
        if outputs == 0 || q.nrows() != n || q.ncols() != n {
            return None;
        }
        let mut f = DMatrix::zeros(n, n);
        f.view_mut((0, outputs), (outputs, outputs)).copy_from(&DMatrix::identity(outputs, outputs));
        let mut g = DMatrix::zeros(n, outputs);
        g.view_mut((outputs, 0), (outputs, outputs)).copy_from(&DMatrix::identity(outputs, outputs));
        Self::new(f, g, q.clone())
    }

    /// Build a RES-CLF for general output dynamics `η̇ = Fη + Gu` with unit input weight.
    pub fn new(f: DMatrix<f64>, g: DMatrix<f64>, q: DMatrix<f64>) -> Option<ResClf> {
        let n = f.nrows();
        if f.ncols() != n || g.nrows() != n || q.nrows() != n || q.ncols() != n {
            return None;
        }
        let outputs = g.ncols();
        let p = solve_care(&f, &g, &q)?;
        let pe = p.clone().symmetric_eigen();
        let c1 = pe.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        let c2 = pe.eigenvalues.iter().cloned().fold(0.0f64, f64::max);
        if c1 <= 0.0 {
            return None; // not positive definite: not a Lyapunov function
        }
        // The closed loop gives V̇ = −ηᵀQη ≤ −λ_min(Q)/λ_max(P) · V, so that ratio is the decay constant.
        let qe = q.clone().symmetric_eigen();
        let q_min = qe.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        if q_min <= 0.0 {
            return None;
        }
        Some(ResClf { p, c1, c2, c3: q_min / c2, f, g, outputs })
    }

    /// `I_ε = diag(ε⁻¹ I, I)`, the scaling that turns a fixed quadratic form into a family with a rate knob.
    fn i_eps(&self, eps: f64) -> DMatrix<f64> {
        let n = self.p.nrows();
        let mut m = DMatrix::identity(n, n);
        for i in 0..self.outputs.min(n) {
            m[(i, i)] = 1.0 / eps;
        }
        m
    }

    /// `V_ε(η) = ηᵀ I_ε P I_ε η`.
    pub fn value(&self, eta: &DVector<f64>, eps: f64) -> f64 {
        let ie = self.i_eps(eps);
        let z = &ie * eta;
        (z.transpose() * &self.p * &z)[0]
    }

    /// The sandwich bounds `(c₁, c₂/ε²)` for this `ε`: `c₁‖η‖² ≤ V_ε(η) ≤ (c₂/ε²)‖η‖²`.
    pub fn bounds(&self, eps: f64) -> (f64, f64) {
        (self.c1, self.c2 / (eps * eps))
    }

    /// The guaranteed decay rate `c₃/ε`. Shrinking `ε` makes this arbitrarily large, which is the "rapidly"
    /// in the name and the whole mechanism for dominating an expansive impact.
    pub fn rate(&self, eps: f64) -> f64 {
        self.c3 / eps
    }

    /// The convergence envelope `‖η(t)‖ ≤ ε⁻¹√(c₂/c₁) e^{−c₃t/(2ε)} ‖η(0)‖`.
    ///
    /// Note the two competing effects of shrinking `ε`: the exponential gets faster, and the **prefactor
    /// `1/ε` gets larger**. A small `ε` buys asymptotic speed at the cost of a bigger transient overshoot,
    /// which is the real reason not to take `ε` to zero.
    pub fn envelope(&self, eta0_norm: f64, eps: f64, t: f64) -> f64 {
        (1.0 / eps) * (self.c2 / self.c1).sqrt() * (-self.c3 * t / (2.0 * eps)).exp() * eta0_norm
    }

    /// The **CLF-QP**: the minimum-norm input satisfying `L_F V_ε + L_G V_ε · u ≤ −(c₃/ε)V_ε`.
    ///
    /// One inequality, so the answer is a projection onto a half-space: leave `u` at zero if the constraint
    /// is already met by the drift, otherwise take the shortest step onto the boundary. That closed form is
    /// what lets the descent condition be enforced inside a control loop rather than offline.
    ///
    /// Returns `None` only when the constraint cannot be met at any input, i.e. `L_G V_ε = 0` while the
    /// drift violates it — the RES-CLF has lost authority over the output.
    pub fn clf_qp(&self, eta: &DVector<f64>, eps: f64) -> Option<DVector<f64>> {
        let ie = self.i_eps(eps);
        // V = eta' M eta with M = I_e P I_e, so dV = 2 eta' M (F eta + G u)
        let m = &ie * &self.p * &ie;
        let meta = &m * eta;
        let lf = 2.0 * (meta.transpose() * &self.f * eta)[0];
        let lg = 2.0 * (self.g.transpose() * &meta); // gradient of V̇ in u
        let slack = lf + self.rate(eps) * self.value(eta, eps);
        if slack <= 0.0 {
            return Some(DVector::zeros(self.g.ncols())); // drift already satisfies the descent
        }
        let lg_sq = lg.norm_squared();
        if lg_sq < 1e-30 {
            return None; // no authority: the condition cannot be enforced
        }
        Some(-(slack / lg_sq) * lg)
    }

    /// Whether an `eps` is small enough for the continuous contraction to dominate a per-step impact
    /// expansion of `mu` over a step of duration `t_step`: the condition `μ e^{−c₃T/(2ε)} < 1`.
    ///
    /// The threshold is `ε̄ = c₃T/(2 ln μ)`, the same quantity
    /// [`hybrid_certificate`](ferromotion_core::hybrid_certificate) reports. A non-expansive impact
    /// (`μ ≤ 1`) admits every `ε`.
    pub fn dominates_impact(&self, mu: f64, t_step: f64, eps: f64) -> bool {
        if mu <= 1.0 {
            return true;
        }
        mu * (-self.c3 * t_step / (2.0 * eps)).exp() < 1.0
    }

    /// The largest `ε` for which [`dominates_impact`](Self::dominates_impact) holds: `ε̄ = c₃T/(2 ln μ)`.
    /// `None` when the impact is non-expansive, since then no bound is needed.
    pub fn eps_bar(&self, mu: f64, t_step: f64) -> Option<f64> {
        if mu <= 1.0 + 1e-9 {
            return None;
        }
        Some(self.c3 * t_step / (2.0 * mu.ln()))
    }
}

/// Solve the continuous algebraic Riccati equation `FᵀP + PF − PGGᵀP + Q = 0` by the Kleinman iteration:
/// repeatedly solve a Lyapunov equation for the current gain. Each step is a linear solve and the iteration
/// converges quadratically once it is near the solution.
fn solve_care(f: &DMatrix<f64>, g: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = f.nrows();
    // Kleinman needs a *stabilizing* initial gain and diverges without one — `K = Gᵀ` is the obvious guess
    // and it is not stabilizing even for a double integrator, where it leaves an eigenvalue at the origin
    // and makes the first Lyapunov solve singular. Bootstrap from a discrete LQR gain on a finely
    // discretised copy instead, which is stabilizing for the continuous system whenever the pair is
    // controllable.
    let dt = 1e-3;
    let ad = DMatrix::identity(n, n) + f * dt;
    let bd = g * dt;
    let m = g.ncols();
    let mut k = crate::dlqr(&ad, &bd, &(q * dt), &(DMatrix::identity(m, m) * dt));
    let mut p = DMatrix::identity(n, n);
    for _ in 0..300 {
        let acl = f - g * &k;
        // solve Aclᵀ P + P Acl + (Q + KᵀK) = 0
        let rhs = q + k.transpose() * &k;
        let p_new = solve_continuous_lyapunov(&acl, &rhs)?;
        let change = (&p_new - &p).norm() / p_new.norm().max(1e-30);
        p = p_new;
        k = g.transpose() * &p;
        if change < 1e-13 {
            // verify the residual rather than trusting convergence of the iterate
            let res = f.transpose() * &p + &p * f - &p * g * g.transpose() * &p + q;
            return if res.norm() / p.norm().max(1e-30) < 1e-8 { Some(p) } else { None };
        }
    }
    None
}

/// Solve `Aᵀ X + X A + C = 0` for symmetric `X`, by the Kronecker form. `A` must be stable.
fn solve_continuous_lyapunov(a: &DMatrix<f64>, c: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let n2 = n * n;
    // vec(AᵀX + XA) = (I ⊗ Aᵀ + Aᵀ ⊗ I) vec(X), with column-major vec
    let mut m = DMatrix::zeros(n2, n2);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // (I ⊗ Aᵀ): block (j,j) gets Aᵀ
                m[(j * n + i, j * n + k)] += a[(k, i)];
                // (Aᵀ ⊗ I): block (i,k) of the outer index
                m[(j * n + i, k * n + i)] += a[(k, j)];
            }
        }
    }
    let rhs = DVector::from_iterator(n2, c.iter().map(|v| -*v));
    let x = m.lu().solve(&rhs)?;
    let out = DMatrix::from_iterator(n, n, x.iter().cloned());
    Some((&out + out.transpose()) * 0.5) // symmetrise away round-off
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clf(outputs: usize) -> ResClf {
        ResClf::double_integrator(outputs, &DMatrix::identity(2 * outputs, 2 * outputs)).expect("RES-CLF")
    }

    /// The Riccati solution really solves the Riccati equation. Checking the residual rather than the
    /// iterate, because a converged iterate of a wrong recursion is still wrong.
    #[test]
    fn the_riccati_solution_has_a_vanishing_residual() {
        let c = clf(2);
        let n = 4;
        let mut f = DMatrix::zeros(n, n);
        f.view_mut((0, 2), (2, 2)).copy_from(&DMatrix::identity(2, 2));
        let mut g = DMatrix::zeros(n, 2);
        g.view_mut((2, 0), (2, 2)).copy_from(&DMatrix::identity(2, 2));
        let q = DMatrix::identity(n, n);
        let res = f.transpose() * &c.p + &c.p * &f - &c.p * &g * g.transpose() * &c.p + &q;
        eprintln!("CARE residual {:.2e}, c1 {:.4}, c2 {:.4}, c3 {:.4}", res.norm(), c.c1, c.c2, c.c3);
        assert!(res.norm() < 1e-8, "Riccati residual {:.2e}", res.norm());
        assert!(c.c1 > 0.0 && c.c3 > 0.0, "P must be positive definite with a positive decay constant");
    }

    /// **The sandwich bounds**, checked over many directions and several `ε`. These are what make the
    /// function a RES-CLF rather than just a Lyapunov function, and the `1/ε²` on the upper bound is the
    /// price paid for the rate.
    #[test]
    fn the_sandwich_bounds_hold_at_every_eps() {
        let c = clf(2);
        for &eps in &[1.0f64, 0.5, 0.1, 0.02] {
            let (lo, hi) = c.bounds(eps);
            assert!(hi > lo, "the sandwich must not invert");
            let mut tight_lo = f64::INFINITY;
            let mut tight_hi = 0.0f64;
            for k in 0..400 {
                // deterministic spread of directions over the 4-sphere
                let t = k as f64 * 0.618_033_988_749_895;
                let eta = DVector::from_row_slice(&[(t * 1.0).sin(), (t * 2.0).cos(), (t * 3.0).sin(), (t * 5.0).cos()]);
                let nn = eta.norm_squared();
                let v = c.value(&eta, eps);
                assert!(v >= lo * nn - 1e-9, "lower bound violated at eps {eps}: V {v} < c1||n||^2 {}", lo * nn);
                assert!(v <= hi * nn + 1e-9, "upper bound violated at eps {eps}: V {v} > (c2/eps^2)||n||^2 {}", hi * nn);
                tight_lo = tight_lo.min(v / nn);
                tight_hi = tight_hi.max(v / nn);
            }
            eprintln!("eps {eps:>5}: bounds [{lo:.4}, {hi:.4}], attained [{tight_lo:.4}, {tight_hi:.4}]");
        }
    }

    /// The CLF-QP returns the minimum-norm input meeting the descent condition. Verified two ways: the
    /// constraint is satisfied, and no smaller input satisfies it.
    #[test]
    fn the_clf_qp_is_the_minimum_norm_input_that_achieves_descent() {
        let c = clf(1);
        let eps = 0.2;
        // moving *away* from the origin, so the drift violates the descent condition and the QP is active.
        // (At [0.4, -0.9] the drift already satisfies it and the correct answer is zero effort, which is
        // checked separately below.)
        let eta = DVector::from_row_slice(&[0.4, 0.9]);
        let u = c.clf_qp(&eta, eps).unwrap();
        assert!(u.norm() > 0.0, "this state needs input; if not, the test has stopped exercising the QP");

        // recompute the constraint independently of the solver
        let vdot = |uu: &DVector<f64>| {
            let ie = c.i_eps(eps);
            let m = &ie * &c.p * &ie;
            let etad = &c.f * &eta + &c.g * uu;
            2.0 * (eta.transpose() * &m * &etad)[0]
        };
        let target = -c.rate(eps) * c.value(&eta, eps);
        eprintln!("CLF-QP: |u| = {:.4}, Vdot = {:.4}, required <= {:.4}", u.norm(), vdot(&u), target);
        assert!(vdot(&u) <= target + 1e-8, "the descent condition must hold: {} > {}", vdot(&u), target);

        // minimum norm: scaling the solution down breaks the constraint
        let shrunk = &u * 0.99;
        assert!(vdot(&shrunk) > target + 1e-12, "a smaller input should not satisfy it, so the solution is on the boundary");
        // and where the drift already achieves descent, the minimum-norm answer is exactly zero effort —
        // the QP asks for no more authority than the condition needs
        for easy in [[0.0, 0.0], [0.4, -0.9]] {
            let e = DVector::from_row_slice(&easy);
            assert_eq!(c.clf_qp(&e, eps).unwrap().norm(), 0.0, "no input needed at {easy:?}");
        }
    }

    /// **The envelope, against simulation.** Run the QP-controlled output dynamics and check the trajectory
    /// stays inside the promised bound — and that shrinking `ε` really does converge faster, which is the
    /// property the whole construction exists to provide.
    #[test]
    fn the_closed_loop_stays_inside_the_envelope_and_speeds_up_as_eps_shrinks() {
        let c = clf(1);
        let dt = 1e-5;
        let mut settle_times = Vec::new();
        for &eps in &[0.5f64, 0.2, 0.05] {
            let mut eta = DVector::from_row_slice(&[1.0, 0.0]);
            let n0 = eta.norm();
            let mut settled = f64::INFINITY;
            for k in 0..400_000 {
                let t = k as f64 * dt;
                let u = c.clf_qp(&eta, eps).unwrap();
                let etad = &c.f * &eta + &c.g * &u;
                eta += etad * dt;
                let bound = c.envelope(n0, eps, t);
                assert!(eta.norm() <= bound + 1e-6, "left the envelope at eps {eps}, t {t}: ||eta|| {} > {bound}", eta.norm());
                if settled.is_infinite() && eta.norm() < 0.01 * n0 {
                    settled = t;
                }
            }
            eprintln!("eps {eps:>5}: reached 1% of the initial deviation at t = {settled:.4} s (rate c3/eps = {:.2})", c.rate(eps));
            settle_times.push(settled);
        }
        for w in settle_times.windows(2) {
            assert!(w[1] < w[0], "a smaller eps must converge faster: {:?}", settle_times);
        }
    }

    /// **The loop closed across crates.** The RES-CLF supplies `c₃`; the engine's hybrid certificate
    /// consumes it and reports `ε̄`. Neither knew about the other when it was written, so agreement here is
    /// a real cross-check — and the crossover has to be *sharp*, since it is a strict inequality.
    #[test]
    fn the_resclf_rate_is_the_constant_the_hybrid_certificate_needs() {
        let c = clf(1);
        let t_step = 0.4;
        // an expansive impact: a transverse perturbation grows by 1.6 at each footfall
        let mu = 1.6f64;
        let p = DMatrix::identity(2, 2);
        let reset = DMatrix::from_row_slice(2, 2, &[mu, 0.0, 0.0, 0.5]);

        let eps_bar = c.eps_bar(mu, t_step).expect("an expansive impact has a threshold");
        // the engine's own formula, reached through the certificate rather than restated
        let cert = ferromotion_core::hybrid_certificate(&p, &reset, c.c3, t_step, eps_bar).unwrap();
        let engine_bar = cert.eps_bar.expect("the engine reports a threshold too");
        eprintln!("RES-CLF c3 = {:.4}, eps_bar = {eps_bar:.6}; engine eps_bar = {engine_bar:.6}", c.c3);
        assert!((eps_bar - engine_bar).abs() / engine_bar < 1e-12, "the two modules must agree on eps_bar: {eps_bar} vs {engine_bar}");

        // below the threshold the composition contracts, above it does not, and the crossover is sharp
        for (factor, want) in [(0.5, true), (0.9, true), (1.1, false), (2.0, false)] {
            let eps = eps_bar * factor;
            let got = c.dominates_impact(mu, t_step, eps);
            let chi = ferromotion_core::hybrid_certificate(&p, &reset, c.c3, t_step, eps).unwrap().chi;
            eprintln!("   eps = {factor} * eps_bar: dominates = {got}, engine chi = {chi:.4}");
            assert_eq!(got, want, "at {factor} x eps_bar the domination should be {want}");
            assert_eq!(chi < 1.0, want, "and the engine's chi must agree at {factor} x eps_bar");
        }
        // a non-expansive impact needs no threshold at all
        assert!(c.eps_bar(1.0, t_step).is_none());
        assert!(c.dominates_impact(0.9, t_step, 100.0), "a contracting impact admits any rate");
    }
}
