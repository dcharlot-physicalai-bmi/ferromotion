//! H∞ (sub)optimal state-feedback synthesis — the robust-control reference point.
//!
//! For the continuous-time plant `ẋ = A·x + B₁·w + B₂·u`, `z = C·x + D·u` and a performance
//! level `γ`, the state-feedback H∞ controller `u = −K·x` (with `K = R_uu⁻¹·B₂ᵀ·X`) makes the
//! closed loop internally stable and bounds the disturbance-to-error map: `‖T_zw‖∞ < γ`. The gain
//! comes from the stabilizing `X ⪰ 0` of the H∞ algebraic Riccati equation
//!
//! ```text
//!   AᵀX + XA + X(γ⁻²·B₁B₁ᵀ − B₂·R_uu⁻¹·B₂ᵀ)X + CᵀC = 0 ,   R_uu = DᵀD .
//! ```
//!
//! Under the clean-problem assumptions `DᵀD = I` and `DᵀC = 0` (the typical setup where `C`/`D`
//! stack a state penalty and a unit control penalty), this collapses to the textbook form
//! `AᵀX + XA + X(γ⁻²B₁B₁ᵀ − B₂B₂ᵀ)X + CᵀC = 0` with `K = B₂ᵀX`. We solve the ARE by integrating
//! the matrix Riccati flow `dX/dτ = AᵀX + XA − X R X + Q` (with `R = B₂R_uu⁻¹B₂ᵀ − γ⁻²B₁B₁ᵀ`,
//! `Q = CᵀC`) from `X(0) = 0` to steady state with RK4. When `γ` is feasible the flow converges
//! monotonically to the stabilizing `X ⪰ 0`; when it is not, the solution exhibits finite escape,
//! which we detect and report as infeasible (`None`). Pure Rust + `nalgebra`, so WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A synthesized continuous-time H∞ state-feedback controller: `u = −K·x`.
#[derive(Clone, Debug)]
pub struct Hinf {
    /// State-feedback gain `K = R_uu⁻¹·B₂ᵀ·X` (m×n).
    pub k: DMatrix<f64>,
    /// Stabilizing Riccati solution `X ⪰ 0` (n×n, symmetric).
    pub x: DMatrix<f64>,
    /// Performance level this controller was synthesized at.
    pub gamma: f64,
}

/// Riccati right-hand side `ℛ(X) = AᵀX + XA − X·R·X + Q` (all references, owned result).
fn ric_rhs(at: &DMatrix<f64>, a: &DMatrix<f64>, r: &DMatrix<f64>, q: &DMatrix<f64>, x: &DMatrix<f64>) -> DMatrix<f64> {
    at * x + x * a - (x * r) * x + q
}

/// Integrate the matrix Riccati flow `dX/dτ = ℛ(X)` from `X = 0` to steady state via RK4.
///
/// Returns the stabilizing `X` (residual `‖ℛ(X)‖_F` driven below `tol`) or `None` if the flow
/// escapes to infinity (γ infeasible) or fails to settle within the step budget.
fn solve_care_flow(a: &DMatrix<f64>, r: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let at = a.transpose();

    // Conservative, scale-aware step: the linearized flow rate is O(‖A‖ + ‖R‖·‖X‖ + ‖Q‖).
    let scale = a.norm() + r.norm() + q.norm() + 1.0;
    let dt = (0.2 / scale).min(0.05);

    let tol = 1e-10; // fixed-point of the RK4 map is exactly ℛ(X) = 0.
    let accept = 1e-8; // good enough to certify + polish the gain from.
    let blowup = 1e8; // finite-escape ⇒ infeasible γ.
    let max_steps = 400_000;

    let mut x = DMatrix::<f64>::zeros(n, n);
    let mut best = (f64::INFINITY, x.clone());

    for _ in 0..max_steps {
        let k1 = ric_rhs(&at, a, r, q, &x);
        let k2 = ric_rhs(&at, a, r, q, &(&x + &k1 * (dt * 0.5)));
        let k3 = ric_rhs(&at, a, r, q, &(&x + &k2 * (dt * 0.5)));
        let k4 = ric_rhs(&at, a, r, q, &(&x + &k3 * dt));
        let mut xn = &x + (&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) * (dt / 6.0);
        // Keep the iterate symmetric (the true solution is; RK4 preserves it up to rounding).
        xn = (&xn + xn.transpose()) * 0.5;

        if xn.iter().any(|v| !v.is_finite()) || xn.norm() > blowup {
            return None; // finite escape: no bounded stabilizing solution at this γ.
        }

        let res = ric_rhs(&at, a, r, q, &xn).norm();
        if res < best.0 {
            best = (res, xn.clone());
        }
        x = xn;
        if res < tol {
            return Some(x);
        }
    }

    // Ran out of steps: accept only if we clearly settled onto an equilibrium.
    if best.0 < accept {
        Some(best.1)
    } else {
        None
    }
}

impl Hinf {
    /// Solve the H∞ ARE at level `γ` and return `(K, X)`, or `None` if no stabilizing `X ⪰ 0`
    /// exists at that `γ` (infeasible), if `DᵀD` is singular, or if the dimensions are inconsistent.
    ///
    /// Shapes: `A` (n×n), `B₁` (n×p), `B₂` (n×m), `C` (q×n), `D` (q×m), `γ > 0`.
    pub fn synthesize(
        a: &DMatrix<f64>,
        b1: &DMatrix<f64>,
        b2: &DMatrix<f64>,
        c: &DMatrix<f64>,
        d: &DMatrix<f64>,
        gamma: f64,
    ) -> Option<(DMatrix<f64>, DMatrix<f64>)> {
        let n = a.nrows();
        // Dimension sanity — bail out cleanly rather than panic on a caller's mismatch.
        if a.ncols() != n
            || b1.nrows() != n
            || b2.nrows() != n
            || c.ncols() != n
            || d.nrows() != c.nrows()
            || d.ncols() != b2.ncols()
            || !(gamma > 0.0)
        {
            return None;
        }

        // R_uu = DᵀD (control weight); must be invertible. Clean case: R_uu = I ⇒ K = B₂ᵀX.
        let ruu = d.transpose() * d;
        let ruu_inv = ruu.try_inverse()?;

        // Q = CᵀC ; R = B₂ R_uu⁻¹ B₂ᵀ − γ⁻² B₁B₁ᵀ  (so the ARE is AᵀX + XA − XRX + Q = 0).
        let q = c.transpose() * c;
        let r_ctrl = b2 * &ruu_inv * b2.transpose();
        let r_dist = (b1 * b1.transpose()) * (1.0 / (gamma * gamma));
        let r = &r_ctrl - &r_dist;

        let x = solve_care_flow(a, &r, &q)?;

        // Certify the two defining properties: X ⪰ 0 and the actual closed loop A − B₂K Hurwitz.
        let x_sym = (&x + x.transpose()) * 0.5;
        let min_eig = x_sym.clone().symmetric_eigen().eigenvalues.min();
        if min_eig < -1e-7 {
            return None;
        }
        let k = &ruu_inv * b2.transpose() * &x; // m×n
        let a_cl = a - b2 * &k;
        let stable = a_cl.complex_eigenvalues().iter().all(|e| e.re < -1e-9);
        if !stable {
            return None;
        }

        Some((k, x_sym))
    }

    /// Convenience wrapper: synthesize and bundle the gain, Riccati solution, and `γ`.
    pub fn new(
        a: &DMatrix<f64>,
        b1: &DMatrix<f64>,
        b2: &DMatrix<f64>,
        c: &DMatrix<f64>,
        d: &DMatrix<f64>,
        gamma: f64,
    ) -> Option<Self> {
        let (k, x) = Self::synthesize(a, b1, b2, c, d, gamma)?;
        Some(Self { k, x, gamma })
    }

    /// Find a near-minimal feasible `γ` by bisection on `[gamma_lo, gamma_hi]`.
    ///
    /// Feasibility is monotone in `γ` (a larger level is always easier), so we maintain the
    /// invariant `gamma_lo` infeasible / `gamma_hi` feasible and shrink the bracket to width `tol`,
    /// returning the feasible (upper) endpoint. Returns `None` if `gamma_hi` is itself infeasible.
    /// If `gamma_lo` is already feasible it is returned directly (nothing lower is bracketed).
    pub fn gamma_bisection(
        a: &DMatrix<f64>,
        b1: &DMatrix<f64>,
        b2: &DMatrix<f64>,
        c: &DMatrix<f64>,
        d: &DMatrix<f64>,
        gamma_lo: f64,
        gamma_hi: f64,
        tol: f64,
    ) -> Option<f64> {
        let feasible = |g: f64| Self::synthesize(a, b1, b2, c, d, g).is_some();
        if !feasible(gamma_hi) {
            return None;
        }
        if feasible(gamma_lo) {
            return Some(gamma_lo);
        }
        let (mut lo, mut hi) = (gamma_lo, gamma_hi);
        let tol = tol.max(1e-12);
        // Bounded iteration count; log2 of the initial bracket over tol, with a hard cap.
        for _ in 0..200 {
            if hi - lo <= tol {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if feasible(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }

    /// Apply the control law `u = −K·x`.
    pub fn control(&self, x: &[f64]) -> Vec<f64> {
        (&self.k * DVector::from_row_slice(x)).iter().map(|v| -v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Double integrator with a matched (velocity-channel) disturbance at half the control
    // authority. z = [x₁; x₂; 0; u] penalizes both states (Q = I) and the control (R_uu = I),
    // so DᵀD = I and DᵀC = 0 — the clean case. The disturbance being matched at strength 0.5
    // puts the H∞ feasibility threshold at γ_min = 0.5.
    fn double_integrator() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b1 = DMatrix::from_row_slice(2, 1, &[0.0, 0.5]);
        let b2 = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        // C: rows 0,1 penalize state; rows 2,3 are the control-penalty rows (zero in C).
        let c = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // D: only the last row carries the unit control penalty ⇒ DᵀD = [1], DᵀC = [0 0].
        let d = DMatrix::from_row_slice(4, 1, &[0.0, 0.0, 0.0, 1.0]);
        (a, b1, b2, c, d)
    }

    fn are_residual(
        a: &DMatrix<f64>,
        b1: &DMatrix<f64>,
        b2: &DMatrix<f64>,
        c: &DMatrix<f64>,
        d: &DMatrix<f64>,
        x: &DMatrix<f64>,
        gamma: f64,
    ) -> f64 {
        let ruu_inv = (d.transpose() * d).try_inverse().unwrap();
        let q = c.transpose() * c;
        let disturb = (b1 * b1.transpose()) * (1.0 / (gamma * gamma));
        let control = b2 * &ruu_inv * b2.transpose();
        // AᵀX + XA + X(γ⁻²B₁B₁ᵀ − B₂R_uu⁻¹B₂ᵀ)X + CᵀC.
        (a.transpose() * x + x * a + (x * (&disturb - &control)) * x + q).norm()
    }

    #[test]
    fn synthesizes_stabilizing_psd_solution() {
        let (a, b1, b2, c, d) = double_integrator();
        let gamma = 5.0; // clearly feasible (threshold ≈ 0.5).
        let hinf = Hinf::new(&a, &b1, &b2, &c, &d, gamma).expect("γ = 5 should be feasible");

        // (1) Closed loop A − B₂K is Hurwitz.
        let a_cl = &a - &b2 * &hinf.k;
        for e in a_cl.complex_eigenvalues().iter() {
            assert!(e.re < 0.0, "closed-loop eigenvalue not in the open LHP: {e:?}");
        }

        // (2) X ⪰ 0 (symmetric, non-negative spectrum) and the ARE residual is ≈ 0.
        let min_eig = hinf.x.clone().symmetric_eigen().eigenvalues.min();
        assert!(min_eig > -1e-8, "X not PSD, min eigenvalue = {min_eig}");
        assert!((&hinf.x - hinf.x.transpose()).norm() < 1e-10, "X not symmetric");
        let res = are_residual(&a, &b1, &b2, &c, &d, &hinf.x, gamma);
        assert!(res < 1e-6, "ARE residual too large: {res}");
    }

    #[test]
    fn infeasible_gamma_returns_none() {
        let (a, b1, b2, c, d) = double_integrator();
        // Below the matched-disturbance threshold of 0.5 there is no stabilizing X ⪰ 0.
        assert!(Hinf::synthesize(&a, &b1, &b2, &c, &d, 0.2).is_none());
    }

    #[test]
    fn gamma_bisection_brackets_the_threshold() {
        let (a, b1, b2, c, d) = double_integrator();
        // Endpoints straddle the threshold (≈ 0.5): 0.2 infeasible, 5.0 feasible.
        assert!(Hinf::synthesize(&a, &b1, &b2, &c, &d, 0.2).is_none());
        assert!(Hinf::synthesize(&a, &b1, &b2, &c, &d, 5.0).is_some());

        let g = Hinf::gamma_bisection(&a, &b1, &b2, &c, &d, 0.2, 5.0, 1e-3)
            .expect("bracket contains a feasible level");

        // The returned γ is feasible, well below the loose upper endpoint, and dropping 10% below
        // it makes synthesis fail — i.e. we landed just above the true minimal γ.
        assert!(Hinf::synthesize(&a, &b1, &b2, &c, &d, g).is_some(), "returned γ = {g} infeasible");
        assert!(g < 1.0, "bisection did not descend toward the threshold: γ = {g}");
        assert!(
            Hinf::synthesize(&a, &b1, &b2, &c, &d, g * 0.9).is_none(),
            "γ just below the near-minimal level {g} should be infeasible"
        );
    }

    #[test]
    fn closed_loop_attenuates_disturbance_below_gamma() {
        // The H∞ guarantee: with zero initial state, the induced L2 gain from w to z is < γ, so
        // for any disturbance ‖z‖₂ / ‖w‖₂ ≤ γ. Drive the synthesized closed loop with a
        // deterministic Gaussian disturbance and check that energy ratio honestly.
        let (a, b1, b2, c, d) = double_integrator();
        let gamma = 5.0;
        let hinf = Hinf::new(&a, &b1, &b2, &c, &d, gamma).unwrap();
        let a_cl = &a - &b2 * &hinf.k; // ẋ = A_cl x + B₁ w

        // Seeded LCG → Box–Muller Gaussian, matching the house RNG.
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next_gauss = || {
            let mut u01 = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((rng >> 11) as f64) / ((1u64 << 53) as f64)
            };
            let u1 = u01().max(1e-12);
            let u2 = u01();
            (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
        };

        let dt = 1e-3;
        let mut x = DVector::from_row_slice(&[0.0, 0.0]); // zero initial state.
        let (mut energy_z, mut energy_w) = (0.0f64, 0.0f64);
        for _ in 0..40_000 {
            let w = next_gauss();
            let u = -(&hinf.k * &x); // u = −Kx (m-vector)
            let bw = &b1 * DVector::from_row_slice(&[w]);
            // z = C x + D u.
            let z = &c * &x + &d * &u;
            energy_z += z.norm_squared() * dt;
            energy_w += w * w * dt;
            // Forward Euler on the stable LTI closed loop.
            let xdot = &a_cl * &x + bw;
            x += xdot * dt;
        }
        let l2_gain = (energy_z / energy_w).sqrt();
        assert!(l2_gain.is_finite(), "closed loop diverged");
        assert!(l2_gain < gamma, "L2 gain {l2_gain} should be below γ = {gamma}");
    }
}
