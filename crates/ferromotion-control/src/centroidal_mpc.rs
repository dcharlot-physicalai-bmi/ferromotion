//! Walking-pattern MPC on the Linear-Inverted-Pendulum / cart-table model — the receding-horizon
//! sibling of the Riccati preview controller in [`crate::zmp`]. Per horizontal axis the state is
//! `[com, com_vel, com_acc]`, the control is CoM jerk, and the ZMP output is `p = com − (z/g)·com̈`.
//! Each control period we solve a condensed QP (states eliminated, à la [`crate::mpc`]) that
//! minimizes ZMP tracking error to a reference (footstep centers) plus jerk regularization, with
//! the ZMP hard-constrained to the support polygon (a per-axis box).
//!
//! The QP is written directly in the ZMP output as the decision variable. The jerk sequence `U`
//! and the predicted ZMP sequence `P` are in bijection given the current state,
//! `P = P₀(x₀) + Sₚ·U` with `Sₚ` square lower-triangular and invertible (its diagonal is the
//! one-step ZMP response `C·B ≠ 0`), so `U = Sₚ⁻¹(P − P₀)`. Re-parametrizing in `P` makes the
//! support-polygon limits *literal* box bounds on the decision variable — solvable with the shared
//! [`crate::qp::solve_box_qp`] backend — and makes the tracking Hessian the well-conditioned
//! identity, pushing the model's ill-conditioning into a single up-front triangular inverse.
//! Jerk regularization (mapped through `Sₚ⁻¹`) is what supplies the stabilizing preview: a CoM
//! that rides the LIPM's divergent mode needs exponentially growing jerk over the horizon, so even
//! a tiny jerk weight strongly penalizes it. Pure `nalgebra` + `clarabel`, WASM-clean.

use crate::qp::solve_box_qp;
use crate::zmp::CartState;
use nalgebra::{DMatrix, DVector};

/// Single-axis cart-table walking MPC (instantiate one per horizontal axis). Build it once with
/// [`CentroidalMpc::new`] — the constructor precomputes the condensed prediction/cost matrices —
/// then call [`control`](CentroidalMpc::control) each control period and roll the cart-table with
/// [`roll`](CentroidalMpc::roll).
#[derive(Clone, Debug)]
pub struct CentroidalMpc {
    /// Constant CoM height `z` (m).
    pub com_height: f64,
    /// Control period (s).
    pub dt: f64,
    /// Number of predicted ZMP samples (preview horizon = `horizon·dt`).
    pub horizon: usize,
    /// Gravity magnitude (m/s²).
    pub g: f64,
    /// Weight on ZMP tracking error.
    pub w_track: f64,
    /// Weight on CoM jerk (Kajita-style small regularizer that supplies the preview action).
    pub w_jerk: f64,
    zg: f64,                     // z/g, the LIPM cart-table coefficient
    csx: DMatrix<f64>,           // N×3: free ZMP response, P₀ = csx·x₀
    m: DMatrix<f64>,             // N×N = Sₚ⁻¹: jerk sequence U = m·(P − P₀)
    two_wjerk_mtm: DMatrix<f64>, // N×N = 2·w_jerk·MᵀM (the state-dependent linear-term coupling)
    h: DMatrix<f64>,             // N×N condensed Hessian in ZMP space (constant)
}

impl CentroidalMpc {
    /// Build with the usual defaults: `g = 9.81`, ZMP-tracking weight `1`, jerk weight `1e-6`
    /// (the Kajita ZMP-preview tradeoff — tight tracking, jerk only as a preview regularizer).
    pub fn new(com_height: f64, dt: f64, horizon: usize) -> Self {
        Self::with_weights(com_height, dt, horizon, 9.81, 1.0, 1e-6)
    }

    /// Build with explicit gravity and cost weights.
    pub fn with_weights(com_height: f64, dt: f64, horizon: usize, g: f64, w_track: f64, w_jerk: f64) -> Self {
        assert!(horizon >= 1, "horizon must be ≥ 1");
        let n = horizon;
        let zg = com_height / g;
        let dt2 = dt * dt;

        // Constant-jerk cart-table: xₖ₊₁ = A·xₖ + B·uₖ, ZMP pₖ = C·xₖ, C = [1, 0, −z/g].
        #[rustfmt::skip]
        let a = DMatrix::from_row_slice(3, 3, &[
            1.0, dt,  0.5 * dt2,
            0.0, 1.0, dt,
            0.0, 0.0, 1.0,
        ]);
        let b = DMatrix::from_column_slice(3, 1, &[dt * dt2 / 6.0, 0.5 * dt2, dt]);
        let crow = DMatrix::from_row_slice(1, 3, &[1.0, 0.0, -zg]);

        // Powers A⁰…Aᴺ.
        let mut apow = Vec::with_capacity(n + 1);
        apow.push(DMatrix::<f64>::identity(3, 3));
        for _ in 0..n {
            let next = &a * apow.last().unwrap();
            apow.push(next);
        }

        // One-step-and-beyond ZMP impulse response cab[i] = C·Aⁱ·B (the diagonals of Sₚ).
        let cab: Vec<f64> = (0..n).map(|i| (&crow * &apow[i] * &b)[(0, 0)]).collect();

        // Free ZMP response csx (row k = C·Aᵏ⁺¹), and the ZMP prediction map Sₚ (lower-triangular
        // Toeplitz: Sₚ[k,j] = C·Aᵏ⁻ʲ·B = cab[k−j] for j ≤ k). Pₖ = ZMP one-plus-k steps ahead.
        let mut csx = DMatrix::zeros(n, 3);
        let mut su_p = DMatrix::zeros(n, n);
        for k in 0..n {
            let row = &crow * &apow[k + 1];
            for c in 0..3 {
                csx[(k, c)] = row[(0, c)];
            }
            for j in 0..=k {
                su_p[(k, j)] = cab[k - j];
            }
        }

        // M = Sₚ⁻¹ (well-defined: Sₚ is triangular with nonzero diagonal cab[0] = C·B).
        let m = su_p.try_inverse().expect("cart-table ZMP map Sₚ is invertible (C·B ≠ 0)");
        let mtm = m.transpose() * &m;
        let two_wjerk_mtm = &mtm * (2.0 * w_jerk);

        // Condensed cost in ZMP space: J = w_track‖P − Pref‖² + w_jerk‖M(P − P₀)‖²
        //   = ½PᵀHP + gᵀP + const,  H = 2·w_track·I + 2·w_jerk·MᵀM,
        //   g = −2·w_track·Pref − 2·w_jerk·MᵀM·P₀.  H is constant → precompute here.
        let mut h = &two_wjerk_mtm + DMatrix::<f64>::identity(n, n) * (2.0 * w_track);
        h = (&h + &h.transpose()) * 0.5; // symmetrize for the QP backend

        Self { com_height, dt, horizon, g, w_track, w_jerk, zg, csx, m, two_wjerk_mtm, h }
    }

    /// Realized ZMP of a cart-table state: `p = com − (z/g)·com̈`.
    pub fn zmp(&self, s: &CartState) -> f64 {
        s.pos - self.zg * s.acc
    }

    /// Advance the cart-table one control period under constant jerk `u` (exact integration).
    pub fn roll(&self, s: &CartState, u: f64) -> CartState {
        let dt = self.dt;
        CartState {
            pos: s.pos + dt * s.vel + 0.5 * dt * dt * s.acc + (dt * dt * dt / 6.0) * u,
            vel: s.vel + dt * s.acc + 0.5 * dt * dt * u,
            acc: s.acc + dt * u,
        }
    }

    /// Optimal CoM jerk to apply this period (receding horizon: apply it, roll, re-solve).
    ///
    /// `state` is the current cart-table state `x₀`. `zmp_ref` is the upcoming reference-ZMP window
    /// (footstep centers): `zmp_ref[k]` is the desired ZMP for the ZMP sample `k+1` steps ahead;
    /// windows longer than `horizon` are truncated and shorter ones are clamped to the last sample.
    /// `zmp_lo`/`zmp_hi` are the support-polygon bounds for this axis; the returned jerk keeps the
    /// realized ZMP inside `[zmp_lo, zmp_hi]` at every predicted step.
    pub fn control(&self, state: &CartState, zmp_ref: &[f64], zmp_lo: f64, zmp_hi: f64) -> f64 {
        assert!(!zmp_ref.is_empty(), "zmp_ref must be non-empty");
        assert!(zmp_lo <= zmp_hi, "empty support polygon: {zmp_lo} > {zmp_hi}");
        let n = self.horizon;

        // Free ZMP response over the horizon.
        let x0 = DVector::from_row_slice(&[state.pos, state.vel, state.acc]);
        let p0 = &self.csx * &x0;

        // Reference window (clamp-extended to the last sample).
        let last = zmp_ref.len() - 1;
        let pref = DVector::from_iterator(n, (0..n).map(|k| zmp_ref[k.min(last)]));

        // Linear term g = −2·w_track·Pref − 2·w_jerk·MᵀM·P₀.
        let glin = &pref * (-2.0 * self.w_track) - &self.two_wjerk_mtm * &p0;
        let g_slice: Vec<f64> = glin.iter().cloned().collect();

        // Support polygon becomes the literal box on the ZMP decision variable.
        let lo = vec![zmp_lo; n];
        let hi = vec![zmp_hi; n];
        let p_opt = DVector::from_vec(solve_box_qp(&self.h, &g_slice, &lo, &hi));

        // Recover the jerk sequence and apply the first input. Because M is lower-triangular with
        // M[0,0] = 1/(C·B), the first jerk exactly places next-step ZMP at p_opt[0] ∈ [lo, hi].
        (&self.m * (&p_opt - &p0))[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference ZMP as a footstep shift: 0 before `step_at`, `x_ref` after.
    fn footstep_ref(i: i64, step_at: i64, x_ref: f64) -> f64 {
        if i >= step_at {
            x_ref
        } else {
            0.0
        }
    }

    #[test]
    fn zmp_stays_in_support_and_com_shifts_over_the_new_footstep() {
        // 1-D cart-table. The reference ZMP steps from 0 to x_ref (a footstep to the side). We run
        // the MPC in closed loop on the cart-table and assert (a) the realized ZMP never leaves the
        // support polygon and (b) the CoM comes to rest directly above the new footstep.
        let (dt, zc) = (0.02, 0.8);
        let horizon = 100; // 2.0 s preview (~7 LIPM time constants)
        let mpc = CentroidalMpc::new(zc, dt, horizon);

        let x_ref = 0.10;
        let (zmp_lo, zmp_hi) = (-0.05, 0.15); // support polygon straddling both footsteps
        let step_at: i64 = 100; // reference jumps at t = 2.0 s
        let total: i64 = 500; // 10 s of simulation

        let mut st = CartState::default();
        let eps = 1e-4; // clarabel constraint tolerance
        let mut moved_zmp = false;

        for t in 0..total {
            // Window of upcoming footstep centers: zmp_ref[k] targets the ZMP k+1 steps ahead.
            let window: Vec<f64> =
                (0..horizon as i64).map(|k| footstep_ref(t + 1 + k, step_at, x_ref)).collect();
            let u = mpc.control(&st, &window, zmp_lo, zmp_hi);
            st = mpc.roll(&st, u);

            let p = mpc.zmp(&st);
            assert!(p.is_finite(), "non-finite ZMP at t={t}");
            assert!(
                p >= zmp_lo - eps && p <= zmp_hi + eps,
                "ZMP {p} left the support polygon [{zmp_lo}, {zmp_hi}] at t={t}"
            );
            if p.abs() > 0.02 {
                moved_zmp = true;
            }
        }

        // The controller actually did work — the ZMP swung out to shift the CoM, it didn't sit at 0.
        assert!(moved_zmp, "ZMP never moved; controller was inert");

        // After the transient the CoM rests directly above the new footstep (com→x_ref, and the
        // realized ZMP = com there since the cart-table is at rest).
        assert!((st.pos - x_ref).abs() < 0.02, "CoM {} did not settle over footstep {x_ref}", st.pos);
        assert!(st.vel.abs() < 0.02, "CoM velocity not settled: {}", st.vel);
        assert!(st.acc.abs() < 0.05, "CoM acceleration not settled: {}", st.acc);
        assert!((mpc.zmp(&st) - x_ref).abs() < 0.02, "settled ZMP {} vs footstep {x_ref}", mpc.zmp(&st));
    }

    #[test]
    fn tight_support_polygon_saturates_the_zmp() {
        // A support polygon narrower than the target footstep: the ZMP must clamp to the edge and
        // never exceed it, yet the CoM still drifts toward the (unreachable-by-ZMP) reference.
        let (dt, zc) = (0.02, 0.8);
        let horizon = 80;
        let mpc = CentroidalMpc::new(zc, dt, horizon);

        let x_ref = 0.30;
        let (zmp_lo, zmp_hi) = (-0.02, 0.08); // deliberately tighter than x_ref
        let step_at: i64 = 50;
        let total: i64 = 400;

        let mut st = CartState::default();
        let mut max_zmp = f64::NEG_INFINITY;
        for t in 0..total {
            let window: Vec<f64> =
                (0..horizon as i64).map(|k| footstep_ref(t + 1 + k, step_at, x_ref)).collect();
            let u = mpc.control(&st, &window, zmp_lo, zmp_hi);
            st = mpc.roll(&st, u);
            let p = mpc.zmp(&st);
            max_zmp = max_zmp.max(p);
            assert!(p <= zmp_hi + 1e-4, "ZMP {p} exceeded tight upper bound {zmp_hi} at t={t}");
            assert!(p >= zmp_lo - 1e-4, "ZMP {p} below tight lower bound {zmp_lo} at t={t}");
        }
        // The constraint was actually active (ZMP pressed up against its edge).
        assert!(max_zmp > zmp_hi - 5e-3, "upper ZMP bound was never approached: max {max_zmp}");
        // The CoM advanced toward the reference even though the ZMP could not reach it.
        assert!(st.pos > 0.05, "CoM failed to advance under the saturated ZMP: {}", st.pos);
    }

    #[test]
    fn zero_reference_keeps_the_cart_table_at_rest() {
        let mpc = CentroidalMpc::new(0.8, 0.02, 60);
        let mut st = CartState::default();
        for _ in 0..200 {
            let u = mpc.control(&st, &vec![0.0; 60], -0.1, 0.1);
            assert!(u.abs() < 1e-6, "nonzero jerk on a zero reference at rest: {u}");
            st = mpc.roll(&st, u);
        }
        assert!(st.pos.abs() < 1e-9 && st.vel.abs() < 1e-9 && st.acc.abs() < 1e-9, "state drifted: {st:?}");
    }
}
