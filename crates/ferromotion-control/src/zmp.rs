//! Legged-balance primitives on the 3D-LIPM / cart-table model: the capture point and a
//! ZMP preview controller (Kajita 2003). The preview controller precomputes optimal gains by
//! solving the discrete Riccati equation for the incremental cart-table servo, then generates a
//! CoM (jerk-integrated) trajectory whose realized ZMP `p = pos − (z/g)·acc` tracks a reference
//! ZMP, using a known window of upcoming reference. Pure `nalgebra`, WASM-clean.

use nalgebra::{DMatrix, DVector};

/// Instantaneous capture point (a.k.a. divergent component of motion), per axis:
/// `ξ = x + ẋ·√(z/g)`. Stepping to `ξ` brings the LIPM to rest. `com_height` is the constant
/// CoM height `z`, `g` the gravity magnitude (e.g. 9.81).
pub fn capture_point(com_xy: [f64; 2], com_vel_xy: [f64; 2], com_height: f64, g: f64) -> [f64; 2] {
    let w = (com_height / g).sqrt(); // 1/ω, the LIPM time constant
    [com_xy[0] + com_vel_xy[0] * w, com_xy[1] + com_vel_xy[1] * w]
}

/// Cart-table state for one axis: CoM position, velocity, acceleration. The control input is jerk.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CartState {
    pub pos: f64,
    pub vel: f64,
    pub acc: f64,
}

/// A ZMP preview controller for a single axis (instantiate one per axis). The builder
/// [`ZmpPreview::new`] precomputes the LQR feedback gains (`gi`, `gx`) and the preview gains
/// (`gd`) from the discrete Riccati solution of the incremental servo; [`ZmpPreview::step`]
/// advances the cart-table one control period given the upcoming reference window.
#[derive(Clone, Debug)]
pub struct ZmpPreview {
    dt: f64,
    zc: f64,
    g: f64,
    /// Integral (of ZMP tracking error) gain.
    gi: f64,
    /// State-feedback gain on `[pos, vel, acc]`.
    gx: [f64; 3],
    /// Preview feedforward gains on reference increments `Δp_ref` over the horizon.
    gd: Vec<f64>,
}

/// Running state for a single-axis [`ZmpPreview`] loop. Start from `default()` with a reference
/// that begins at the current CoM/ZMP so the increments start clean.
#[derive(Clone, Copy, Debug, Default)]
pub struct PreviewState {
    /// Current cart-table state `xₖ`.
    pub cart: CartState,
    prev_cart: CartState, // xₖ₋₁, for Δxₖ
    prev_u: f64,          // uₖ₋₁, since the servo is incremental in the input
}

impl ZmpPreview {
    /// Build a preview controller. `dt` = control period (s), `zc` = CoM height (m), `g` = gravity
    /// magnitude, `n_preview` = number of future reference samples used (horizon = `n_preview·dt`).
    pub fn new(dt: f64, zc: f64, g: f64, n_preview: usize) -> Self {
        // Cart-table (constant-jerk) discretization: xₖ₊₁ = A·xₖ + B·uₖ, ZMP pₖ = C·xₖ.
        let (dt2h, dt3_6, zg) = (0.5 * dt * dt, dt * dt * dt / 6.0, zc / g);
        // C = [1, 0, −z/g]; C·A and C·B for the augmented (incremental) system.
        let ca = [1.0, dt, dt2h - zg]; // C·A (1×3)
        let cb = dt3_6 - zg * dt; //       C·B (scalar)

        // Augmented incremental servo state X = [e; Δx] ∈ ℝ⁴ (e = output error, Δx = state incr):
        //   Xₖ₊₁ = Φ·Xₖ + G·Δuₖ + Gr·Δp_ref,   Φ = [1, C·A; 0, A],  G = [C·B; B],  Gr = [−1; 0].
        #[rustfmt::skip]
        let phi = DMatrix::from_row_slice(4, 4, &[
            1.0, ca[0], ca[1], ca[2],
            0.0, 1.0,   dt,    dt2h,
            0.0, 0.0,   1.0,   dt,
            0.0, 0.0,   0.0,   1.0,
        ]);
        let gmat = DMatrix::from_column_slice(4, 1, &[cb, dt3_6, dt2h, dt]);
        let gr = DVector::from_row_slice(&[-1.0, 0.0, 0.0, 0.0]);

        // Cost: Σ qe·e² + r·Δu². qe on the tracking error, small r for tight tracking (Kajita).
        let qe = 1.0;
        let r = 1e-6;
        let mut qtil = DMatrix::<f64>::zeros(4, 4);
        qtil[(0, 0)] = qe;

        // Discrete Riccati by value iteration (same recursion as `lqr::dlqr`, scalar input).
        let mut p = qtil.clone();
        let phit = phi.transpose();
        let gt = gmat.transpose();
        for _ in 0..200_000 {
            let gtp = &gt * &p; // 1×4
            let denom = r + (&gtp * &gmat)[(0, 0)];
            let k = (&gtp * &phi) / denom; // 1×4 = S·GᵀPΦ
            let phitpg = &phit * (&p * &gmat); // 4×1
            let p_next = &phit * (&p * &phi) - &phitpg * &k + &qtil;
            let done = (&p_next - &p).norm() < 1e-12;
            p = p_next;
            if done {
                break;
            }
        }

        // Final feedback gain K = S·GᵀPΦ = [gi | gx].
        let gtp = &gt * &p;
        let denom = r + (&gtp * &gmat)[(0, 0)];
        let s = 1.0 / denom;
        let k = (&gtp * &phi) * s; // 1×4
        let gi = k[(0, 0)];
        let gx = [k[(0, 1)], k[(0, 2)], k[(0, 3)]];

        // Preview gains: f(l) = S·Gᵀ·(Φcᵀ)^{l−1}·P·Gr, Φc = Φ − G·K, for l = 1..=n_preview.
        let phic = &phi - &gmat * &k; // 4×4
        let phic_t = phic.transpose();
        let mut gd = Vec::with_capacity(n_preview);
        let mut v = &p * &gr; // v₁ = P·Gr (4×1)
        for _ in 0..n_preview {
            let f = s * (&gt * &v)[(0, 0)];
            gd.push(f);
            v = &phic_t * &v;
        }

        Self { dt, zc, g, gi, gx, gd }
    }

    /// Number of future reference samples the controller previews.
    pub fn n_preview(&self) -> usize {
        self.gd.len()
    }

    /// Realized ZMP for a cart-table state: `p = pos − (z/g)·acc`.
    pub fn zmp(&self, s: &CartState) -> f64 {
        s.pos - (self.zc / self.g) * s.acc
    }

    /// Advance one control period. `zmp_ref` is the upcoming reference-ZMP window with `zmp_ref[0]`
    /// the current step and `zmp_ref[l]` the reference `l` steps ahead (length ≥ `n_preview()+1`
    /// for full preview; shorter windows are clamped to the last sample). Mutates `st` to the next
    /// state and returns the jerk `u` applied this step.
    pub fn step(&self, st: &mut PreviewState, zmp_ref: &[f64]) -> f64 {
        assert!(!zmp_ref.is_empty(), "zmp_ref must be non-empty");
        let get = |i: usize| zmp_ref[i.min(zmp_ref.len() - 1)];

        // Current ZMP tracking error and state increment.
        let e = self.zmp(&st.cart) - get(0);
        let dpos = st.cart.pos - st.prev_cart.pos;
        let dvel = st.cart.vel - st.prev_cart.vel;
        let dacc = st.cart.acc - st.prev_cart.acc;

        // Preview feedforward on reference increments Δp_ref(k+l) = ref[l] − ref[l−1].
        let mut ff = 0.0;
        for (l, &gdl) in self.gd.iter().enumerate() {
            ff += gdl * (get(l + 1) - get(l));
        }

        // Incremental servo: Δu = −gi·e − gx·Δx − Σ gd·Δp_ref; integrate the input.
        let du = -self.gi * e - (self.gx[0] * dpos + self.gx[1] * dvel + self.gx[2] * dacc) - ff;
        let u = st.prev_u + du;

        // Cart-table update (exact under constant jerk over dt).
        let (dt, x) = (self.dt, st.cart);
        let next = CartState {
            pos: x.pos + dt * x.vel + 0.5 * dt * dt * x.acc + (dt * dt * dt / 6.0) * u,
            vel: x.vel + dt * x.acc + 0.5 * dt * dt * u,
            acc: x.acc + dt * u,
        };
        st.prev_cart = st.cart;
        st.cart = next;
        st.prev_u = u;
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_point_matches_closed_form() {
        // ξ = x + ẋ·√(z/g). Known state, per axis.
        let (z, g) = (0.8, 9.81);
        let xi = capture_point([0.1, -0.05], [0.2, 0.3], z, g);
        let w = (z / g).sqrt();
        assert!((xi[0] - (0.1 + 0.2 * w)).abs() < 1e-12, "ξx = {}", xi[0]);
        assert!((xi[1] - (-0.05 + 0.3 * w)).abs() < 1e-12, "ξy = {}", xi[1]);
        // Explicit numeric sanity: √(0.8/9.81) ≈ 0.285565.
        assert!((xi[0] - 0.157113).abs() < 1e-5, "ξx numeric = {}", xi[0]);
    }

    #[test]
    fn capture_point_at_rest_is_the_com() {
        let xi = capture_point([0.3, -0.2], [0.0, 0.0], 0.8, 9.81);
        assert_eq!(xi, [0.3, -0.2]);
    }

    #[test]
    fn preview_tracks_a_step_zmp_reference() {
        // 1-D cart-table (x axis). Reference ZMP steps from 0 to x_ref partway through.
        let (dt, zc, g) = (0.005, 0.8, 9.81);
        let n_preview = 200; // 1.0 s of preview
        let ctrl = ZmpPreview::new(dt, zc, g, n_preview);
        assert_eq!(ctrl.n_preview(), n_preview);

        let x_ref = 0.10;
        let total = 1400; // 7 s of simulation
        let step_at = 600; // reference jumps at t = 3.0 s (beyond the preview horizon at t=0)
        // Full reference including the preview tail so windows are always fully populated.
        let refs: Vec<f64> =
            (0..total + n_preview + 2).map(|k| if k >= step_at { x_ref } else { 0.0 }).collect();

        let zg = zc / g;
        let mut st = PreviewState::default();
        let mut zmp_after_settle = Vec::new();
        for k in 0..total {
            ctrl.step(&mut st, &refs[k..]);
            let zmp = st.cart.pos - zg * st.cart.acc;
            // Before the step enters the preview window (k + n_preview < step_at), the reference
            // is identically zero across the window, so the state stays exactly at the origin.
            if k < step_at - n_preview {
                assert!(zmp.abs() < 1e-9, "pre-preview ZMP drifted: {zmp} at k={k}");
            }
            // Well after the transition, collect realized ZMP to check tracking.
            if k > 1300 {
                zmp_after_settle.push(zmp);
            }
        }

        // Realized ZMP tracks the new reference to small error after the transient.
        for zmp in &zmp_after_settle {
            assert!((zmp - x_ref).abs() < 3e-3, "settled ZMP {zmp} vs ref {x_ref}");
        }
        // CoM converges directly above the new ZMP (pos → x_ref, acc → 0).
        assert!((st.cart.pos - x_ref).abs() < 5e-3, "CoM pos {} vs {x_ref}", st.cart.pos);
        assert!(st.cart.acc.abs() < 5e-2, "CoM accel not settled: {}", st.cart.acc);
        assert!(st.cart.vel.abs() < 5e-3, "CoM vel not settled: {}", st.cart.vel);
    }
}
