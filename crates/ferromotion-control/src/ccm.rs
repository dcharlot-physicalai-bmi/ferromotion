//! **Control Contraction Metrics** (Manchester & Slotine, IEEE TAC 2017) — a *certificate of exponential
//! tracking* for a feedback law, the tracking-stability analog of the safety certificate a CBF gives. A
//! constant contraction metric `M ≻ 0` certifies that *any two* closed-loop trajectories converge in the
//! Riemannian distance `‖x₁ − x₂‖_M` at rate `λ`, so tracking *any* feasible reference is exponentially
//! stable — not just regulation to an equilibrium.
//!
//! The key identity that keeps this wasm-clean (no SDP): for a linear closed loop `A_cl = A + BK`, the
//! contraction LMI `A_clᵀM + M A_cl ⪯ −2λM` holds for some `M ≻ 0` **iff `A_cl + λI` is Hurwitz**, and then
//! `M` is exactly the Lyapunov certificate of the *shifted* matrix — a Kronecker linear solve
//! ([`ferromotion_core::lyapunov`]). Verified by the LMI residual and by simulating that the Riemannian
//! error contracts at rate ≥ λ. Pure `nalgebra` → WASM-clean.

use ferromotion_core::lyapunov;
use nalgebra::{DMatrix, DVector};

/// A synthesized contraction metric + differential feedback for `ẋ = A x + B u`, `u = K(x − x_ref) + u_ref`.
#[derive(Clone, Debug)]
pub struct Ccm {
    pub m: DMatrix<f64>, // the metric, M ≻ 0
    pub k: DMatrix<f64>, // differential feedback gain
    pub lambda: f64,     // certified contraction rate
}

impl Ccm {
    /// Synthesize the contraction metric for closed loop `A + BK` at rate `lambda`: `M` is the Lyapunov
    /// certificate of `A_cl + λI`. Returns `None` if the closed loop does not contract that fast (i.e.
    /// `A_cl` has an eigenvalue with real part ≥ −λ).
    pub fn synthesize(a: &DMatrix<f64>, b: &DMatrix<f64>, k: &DMatrix<f64>, lambda: f64) -> Option<Ccm> {
        let a_cl = a + b * k;
        let shifted = &a_cl + DMatrix::<f64>::identity(a.nrows(), a.nrows()) * lambda;
        let m = lyapunov(&shifted)?; // M ≻ 0 with (A_cl+λI)ᵀM + M(A_cl+λI) = −I
        Some(Ccm { m, k: k.clone(), lambda })
    }

    /// The tracking control `u = K(x − x_ref) + u_ref`.
    pub fn control(&self, x: &DVector<f64>, x_ref: &DVector<f64>, u_ref: &DVector<f64>) -> DVector<f64> {
        &self.k * (x - x_ref) + u_ref
    }

    /// Riemannian distance `‖x₁ − x₂‖_M = √((x₁−x₂)ᵀ M (x₁−x₂))`.
    pub fn riemannian_dist(&self, x1: &DVector<f64>, x2: &DVector<f64>) -> f64 {
        let d = x1 - x2;
        (d.dot(&(&self.m * &d))).max(0.0).sqrt()
    }

    /// The contraction-LMI residual `max eig(A_clᵀM + M A_cl + 2λM)` — negative ⇒ the certificate holds.
    pub fn contraction_residual(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let a_cl = a + b * &self.k;
        let lmi = a_cl.transpose() * &self.m + &self.m * &a_cl + &self.m * (2.0 * self.lambda);
        let sym = (&lmi + lmi.transpose()) * 0.5;
        sym.symmetric_eigen().eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
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

    // Double integrator with a stabilizing feedback (poles at −1, −2 ⇒ contraction rate up to 1).
    fn system() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let a = dm(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = dm(2, 1, &[0.0, 1.0]);
        let k = dm(1, 2, &[-2.0, -3.0]); // A+BK = [[0,1],[-2,-3]], eigenvalues −1, −2
        (a, b, k)
    }

    #[test]
    fn the_metric_certifies_the_contraction_lmi() {
        // THE INVARIANT. At a feasible rate the LMI A_clᵀM + M A_cl + 2λM ⪯ 0 and M ≻ 0.
        let (a, b, k) = system();
        let ccm = Ccm::synthesize(&a, &b, &k, 0.8).expect("should contract at rate 0.8 (< 1)");
        assert!(ccm.m.clone().symmetric_eigen().eigenvalues.iter().all(|&e| e > 1e-9), "M must be PD");
        assert!(ccm.contraction_residual(&a, &b) < 1e-6, "LMI residual should be ≤ 0: {}", ccm.contraction_residual(&a, &b));
    }

    #[test]
    fn too_fast_a_rate_is_infeasible() {
        // The slowest closed-loop pole is at −1, so no constant metric contracts faster than λ = 1.
        let (a, b, k) = system();
        assert!(Ccm::synthesize(&a, &b, &k, 1.5).is_none(), "λ=1.5 exceeds the −1 pole ⇒ infeasible");
    }

    /// ⛔ **The reference was pinned at the origin, so nothing here checked reference handling at all.**
    ///
    /// `xr` started at `[0, 0]` with `ur = [0]`, and its update `xr += (A·xr + B·ur)·dt` is identically
    /// zero, so it stayed bit-exactly `[0, 0]` for all 5000 steps. That is pure regulation to the origin,
    /// precisely the case the module says the CCM guarantee goes beyond, and the comment claimed a moving
    /// reference was being simulated. With `x_ref` and `u_ref` exact zeros, `control` mutated to `K·x` —
    /// dropping BOTH the reference subtraction and the feedforward — passed bit-exactly, as did
    /// `riemannian_dist` mutated to ignore its second argument.
    ///
    /// The reference is now the feasible trajectory `xr(t) = [sin t, cos t]`, which the double integrator
    /// follows under `ur(t) = −sin t`. Both trajectories are advanced by the SAME explicit Euler step, so
    /// the error obeys `e_{n+1} = (I + A_cl·dt)·e_n` exactly and the envelope assertion is unchanged; what
    /// changed is that `x_ref` and `u_ref` are now non-zero at every step.
    #[test]
    fn the_riemannian_error_contracts_at_the_certified_rate() {
        // THE HEADLINE. Under u = K(x−x_ref) + u_ref, the Riemannian error ‖e‖_M decays at least as fast
        // as e^{−λt} — the exponential-tracking guarantee, for ANY feasible reference.
        let (a, b, k) = system();
        let lambda = 0.8;
        let ccm = Ccm::synthesize(&a, &b, &k, lambda).unwrap();
        let dt = 1e-3;
        // A genuinely moving, genuinely feasible reference: xr = [sin t, cos t] needs ur = −sin t, since
        // A·xr + B·ur = [xr₂, ur] and d/dt[sin t, cos t] = [cos t, −sin t].
        let mut xr = dv(&[0.0, 1.0]);
        let mut x = &xr + dv(&[0.4, -0.3]); // the same initial tracking error as before
        let d0 = ccm.riemannian_dist(&x, &xr);
        let (mut xr_travel, mut ur_peak) = (0.0f64, 0.0f64);
        for step in 1..=5000 {
            let t = (step - 1) as f64 * dt;
            let ur = dv(&[-t.sin()]);
            let u = ccm.control(&x, &xr, &ur);
            let xr_prev = xr.clone();
            x += (&a * &x + &b * &u) * dt;
            xr += (&a * &xr + &b * &ur) * dt;
            xr_travel += (&xr - &xr_prev).norm();
            ur_peak = ur_peak.max(ur.norm());
            // the Riemannian error must stay under the certified exponential envelope
            let t = step as f64 * dt;
            let d = ccm.riemannian_dist(&x, &xr);
            assert!(d <= d0 * (-lambda * t).exp() * 1.02, "error {d} exceeded the e^{{−λt}} envelope at t={t}");
        }
        // and it actually shrinks a lot
        assert!(ccm.riemannian_dist(&x, &xr) < 0.05 * d0, "error should decay substantially");
        // ⛔ AND THE REFERENCE ACTUALLY MOVED, which is the assertion whose absence made every mutation of
        // `control`'s reference handling invisible.
        eprintln!("  reference travelled {xr_travel:.3} in state space, peak ‖u_ref‖ {ur_peak:.3}, final xr = [{:.3}, {:.3}]", xr[0], xr[1]);
        assert!(xr_travel > 1.0, "the reference must be a moving trajectory, it travelled {xr_travel}");
        assert!(ur_peak > 0.5, "the feedforward must be exercised, peak ‖u_ref‖ was {ur_peak}");
    }
}
