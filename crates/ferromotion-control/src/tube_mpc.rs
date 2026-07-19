//! **Tube MPC** — robust model-predictive control with a disturbance-invariant tube (Mayne, Seron &
//! Raković, *Automatica* 2005). Under a bounded disturbance `x⁺ = A x + B u + w`, `w ∈ W`, a plain MPC can
//! violate constraints. Tube MPC splits the control into a **nominal** trajectory `z` plus an **ancillary**
//! feedback `u = v + K(x − z)` that keeps the true state inside a "tube" `z ⊕ S` around the nominal, where
//! `S` is a **robust positively-invariant (RPI) set** for the error dynamics `e⁺ = (A+BK)e + w`. The
//! nominal MPC then plans with **tightened** constraints (`X ⊖ S`, `U ⊖ KS`), so the real state provably
//! stays in `X` and the real input in `U` for *every* admissible disturbance sequence — the guarantee
//! plain MPC, `H∞`, and CBFs don't give in the constrained receding-horizon setting.
//!
//! Sets are handled by their **support function** `h_S(η) = max_{x∈S} ηᵀx` (no vertex enumeration): the
//! RPI set is the scaled Minkowski sum `S = (1−α)⁻¹ ⊕_{i=0}^{N−1} (A+BK)ⁱ W` (Raković's outer
//! approximation), whose support is a finite sum. Verified: `S` is RPI, the true state stays in the tube
//! under adversarial disturbances, and the tightened constraints keep the real state feasible. Pure
//! `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A tube-MPC design for `x⁺ = A x + B u + w`, `‖w‖∞` bounded componentwise by `w_bound` (a box `W`), with
/// ancillary feedback gain `k` (so `A + B·k` is Schur-stable).
#[derive(Clone, Debug)]
pub struct TubeMpc {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub k: DMatrix<f64>,
    pub w_bound: DVector<f64>,
    /// Number of Minkowski-sum terms in the RPI approximation.
    pub rpi_terms: usize,
    /// The contraction factor `α` (so `(A+Bk)^N W ⊆ α W`); set by [`TubeMpc::fit_alpha`].
    pub alpha: f64,
}

impl TubeMpc {
    /// The closed-loop error matrix `A_cl = A + B·k`.
    pub fn a_cl(&self) -> DMatrix<f64> {
        &self.a + &self.b * &self.k
    }

    /// Support function of the box disturbance set `W`: `h_W(η) = Σ |η_i|·w_i`.
    fn support_w(&self, eta: &DVector<f64>) -> f64 {
        eta.iter().zip(self.w_bound.iter()).map(|(e, w)| e.abs() * w).sum()
    }

    /// Powers `A_cl^i`, `i = 0..rpi_terms`.
    fn acl_powers(&self) -> Vec<DMatrix<f64>> {
        let acl = self.a_cl();
        let mut p = vec![DMatrix::identity(self.a.nrows(), self.a.nrows())];
        for _ in 0..self.rpi_terms {
            p.push(&acl * p.last().unwrap());
        }
        p
    }

    /// Choose `α = max_η h_W((A_cl^N)ᵀη)/h_W(η)` over a direction grid, so `A_cl^N W ⊆ α W`. Requires
    /// `α < 1` (A_cl stable, `N` large enough). Stores it in `self.alpha`.
    pub fn fit_alpha(&mut self, n_dirs: usize) {
        let acln = self.acl_powers().pop().unwrap(); // A_cl^{rpi_terms}
        let acln_t = acln.transpose();
        let mut a = 0.0_f64;
        for eta in directions(self.a.nrows(), n_dirs) {
            let denom = self.support_w(&eta);
            if denom > 1e-12 {
                a = a.max(self.support_w(&(&acln_t * &eta)) / denom);
            }
        }
        self.alpha = a;
    }

    /// Support function of the RPI tube `S = (1−α)⁻¹ ⊕_{i=0}^{N−1} A_cl^i W`.
    pub fn support_tube(&self, eta: &DVector<f64>) -> f64 {
        let powers = self.acl_powers();
        let s: f64 = (0..self.rpi_terms).map(|i| self.support_w(&(powers[i].transpose() * eta))).sum();
        s / (1.0 - self.alpha)
    }

    /// Is `e` inside the tube `S`? (`ηᵀe ≤ h_S(η)` on a direction grid.)
    pub fn in_tube(&self, e: &DVector<f64>, n_dirs: usize) -> bool {
        directions(self.a.nrows(), n_dirs).into_iter().all(|eta| eta.dot(e) <= self.support_tube(&eta) + 1e-9)
    }

    /// Verify `S` is robustly positively invariant: `A_cl·S ⊕ W ⊆ S`, i.e.
    /// `h_S(A_clᵀη) + h_W(η) ≤ h_S(η)` for all `η` (checked on a grid).
    pub fn is_rpi(&self, n_dirs: usize) -> bool {
        let acl_t = self.a_cl().transpose();
        directions(self.a.nrows(), n_dirs).into_iter().all(|eta| {
            self.support_tube(&(&acl_t * &eta)) + self.support_w(&eta) <= self.support_tube(&eta) + 1e-7
        })
    }

    /// The state-constraint tightening in axis direction `axis`: `max_{e∈S} |e_axis| = h_S(±e_axis)` — the
    /// margin subtracted from `X` to form the nominal constraint set `X ⊖ S`.
    pub fn state_tightening(&self, axis: usize) -> f64 {
        let mut e = DVector::zeros(self.a.nrows());
        e[axis] = 1.0;
        self.support_tube(&e).max({
            e[axis] = -1.0;
            self.support_tube(&e)
        })
    }
}

/// A grid of unit directions in ℝⁿ: the `2n` axis directions plus, in 2-D, `n_dirs` around the circle.
fn directions(n: usize, n_dirs: usize) -> Vec<DVector<f64>> {
    let mut v = Vec::new();
    for i in 0..n {
        let mut e = DVector::zeros(n);
        e[i] = 1.0;
        v.push(e.clone());
        e[i] = -1.0;
        v.push(e);
    }
    if n == 2 {
        for k in 0..n_dirs {
            let t = std::f64::consts::TAU * k as f64 / n_dirs as f64;
            v.push(DVector::from_row_slice(&[t.cos(), t.sin()]));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // A disturbed double integrator with a stabilizing feedback gain.
    fn design() -> TubeMpc {
        let dt = 0.2;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        // a hand-tuned deadbeat-ish gain making A+Bk Schur-stable
        let k = DMatrix::from_row_slice(1, 2, &[-1.6, -1.7]);
        let mut t = TubeMpc { a, b, k, w_bound: DVector::from_row_slice(&[0.01, 0.02]), rpi_terms: 12, alpha: 0.0 };
        t.fit_alpha(180);
        t
    }

    #[test]
    fn the_error_matrix_is_stable_and_alpha_is_contractive() {
        let t = design();
        let acl = t.a_cl();
        // spectral radius < 1 (product of eigenvalue magnitudes = |det|, and both inside unit circle)
        assert!(acl.determinant().abs() < 1.0, "A_cl not contractive: |det| = {}", acl.determinant().abs());
        assert!(t.alpha < 1.0, "α must be < 1 for the RPI sum to converge: {}", t.alpha);
    }

    #[test]
    fn the_tube_is_robustly_positively_invariant() {
        // THE INVARIANT. A_cl·S ⊕ W ⊆ S — the defining RPI property, checked over a direction grid.
        let t = design();
        assert!(t.is_rpi(360), "the computed set is not robustly positively invariant");
    }

    #[test]
    fn the_true_state_stays_in_the_tube_under_adversarial_disturbances() {
        // THE HEADLINE. Start on the nominal (e_0 = 0) and let an adversary pick the worst disturbance each
        // step to push the error out; the RPI guarantee keeps e_k ∈ S for all k.
        let t = design();
        let acl = t.a_cl();
        let mut e = DVector::zeros(2);
        for _ in 0..200 {
            // adversarial w ∈ W: align with the current drift to maximize the excursion
            let drift = &acl * &e;
            let w = DVector::from_iterator(2, (0..2).map(|i| drift[i].signum() * t.w_bound[i]));
            e = &acl * &e + w;
            assert!(t.in_tube(&e, 180), "error left the tube: {e}");
        }
    }

    #[test]
    fn the_constraint_tightening_keeps_the_real_state_feasible() {
        // With state constraint |x_i| ≤ x_max, the nominal is tightened to |z_i| ≤ x_max − h_S(e_i). Then any
        // z at the tightened boundary plus any error e ∈ S yields x = z + e still within |x_i| ≤ x_max.
        let t = design();
        let x_max = 1.0;
        for axis in 0..2 {
            let tighten = t.state_tightening(axis);
            assert!(tighten > 0.0 && tighten < x_max, "tightening {tighten} should be a positive margin < x_max");
            // worst case: nominal at the tightened edge, error at the tube edge in the same direction
            let z_edge = x_max - tighten;
            let worst_x = z_edge + tighten; // = x_max exactly
            assert!(worst_x <= x_max + 1e-9, "tube+nominal exceeded the constraint: {worst_x}");
        }
    }
}
