//! **Clohessy–Wiltshire rendezvous** — linearized relative orbital dynamics in the Hill/LVLH frame of a
//! chaser about a target on a circular reference orbit (Clohessy & Wiltshire, 1960). The relative state
//! `[x, y, z, ẋ, ẏ, ż]` (x radial, y along-track, z cross-track) obeys the CW equations
//! `ẍ − 2n·ẏ − 3n²·x = 0`, `ÿ + 2n·ẋ = 0`, `z̈ + n²·z = 0`, with mean motion `n`. They admit a **closed-form
//! state-transition matrix** `Φ(t)`, so proximity operations — station-keeping, fly-around, and the classic
//! **two-impulse rendezvous** — reduce to linear algebra on `Φ`.
//!
//! This opens the spacecraft rendezvous / proximity-ops domain (the crate had orbital *transfer* instruments
//! but no *relative* motion). Verified: `Φ(t)` matches numerical integration of the CW ODE to integrator
//! precision, `Φ(0) = I`, the cross-track channel is simple-harmonic with period `2π/n`, and the two-impulse
//! solution drives the chaser from an initial offset to the target with the arrival velocity nulled. Pure
//! `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};

/// A Clohessy–Wiltshire model for a circular reference orbit of mean motion `n` (rad/s).
#[derive(Clone, Copy, Debug)]
pub struct Cw {
    pub n: f64,
}

impl Cw {
    /// Mean motion from the orbit radius `a` (m) and gravitational parameter `mu` (m³/s²): `n = √(μ/a³)`.
    pub fn from_orbit(a: f64, mu: f64) -> Self {
        Cw { n: (mu / (a * a * a)).sqrt() }
    }

    /// The closed-form CW state-transition matrix `Φ(t)`, mapping `[x,y,z,ẋ,ẏ,ż]` at time 0 to time `t`.
    pub fn stm(&self, t: f64) -> Matrix6<f64> {
        let n = self.n;
        let (s, c) = ((n * t).sin(), (n * t).cos());
        let nt = n * t;
        let mut phi = Matrix6::zeros();
        // position rows
        // x
        phi[(0, 0)] = 4.0 - 3.0 * c;
        phi[(0, 3)] = s / n;
        phi[(0, 4)] = 2.0 * (1.0 - c) / n;
        // y
        phi[(1, 0)] = 6.0 * (s - nt);
        phi[(1, 1)] = 1.0;
        phi[(1, 3)] = 2.0 * (c - 1.0) / n;
        phi[(1, 4)] = (4.0 * s - 3.0 * nt) / n;
        // z
        phi[(2, 2)] = c;
        phi[(2, 5)] = s / n;
        // velocity rows
        // ẋ
        phi[(3, 0)] = 3.0 * n * s;
        phi[(3, 3)] = c;
        phi[(3, 4)] = 2.0 * s;
        // ẏ
        phi[(4, 0)] = 6.0 * n * (c - 1.0);
        phi[(4, 3)] = -2.0 * s;
        phi[(4, 4)] = 4.0 * c - 3.0;
        // ż
        phi[(5, 2)] = -n * s;
        phi[(5, 5)] = c;
        phi
    }

    /// Propagate a relative state `[r; v]` forward by `t` via the STM.
    pub fn propagate(&self, state: &Vector6<f64>, t: f64) -> Vector6<f64> {
        self.stm(t) * state
    }

    /// The 3×3 position-from-initial-velocity block `Φ_rv(t)` (top-right of `Φ`), whose inverse solves the
    /// two-point boundary-value (Lambert-like) problem in the CW frame.
    fn phi_rv(&self, t: f64) -> Matrix3<f64> {
        self.stm(t).fixed_view::<3, 3>(0, 3).into_owned()
    }

    /// The 3×3 position-from-initial-position block `Φ_rr(t)` (top-left of `Φ`).
    fn phi_rr(&self, t: f64) -> Matrix3<f64> {
        self.stm(t).fixed_view::<3, 3>(0, 0).into_owned()
    }

    /// **Two-impulse rendezvous.** Given the chaser's current relative position `r0` and velocity `v0`, a
    /// desired relative position `rf`, and a time-of-flight `tof`, return `(dv0, dv1)`: the departure burn
    /// that puts the chaser on a transfer reaching `rf` at `t = tof`, and the arrival burn that nulls the
    /// relative velocity there (station-keeping). Returns `None` if `Φ_rv(tof)` is singular (e.g. `tof` a
    /// multiple of the orbital period, where the boundary-value problem degenerates).
    pub fn two_impulse_rendezvous(&self, r0: &Vector3<f64>, v0: &Vector3<f64>, rf: &Vector3<f64>, tof: f64) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let rv = self.phi_rv(tof);
        let rv_inv = rv.try_inverse()?;
        // velocity needed just after the departure burn to arrive at rf
        let v0_plus = rv_inv * (rf - self.phi_rr(tof) * r0);
        let dv0 = v0_plus - v0;
        // arrival velocity before the second burn: Φ_vr r0 + Φ_vv v0_plus
        let phi = self.stm(tof);
        let phi_vr = phi.fixed_view::<3, 3>(3, 0).into_owned();
        let phi_vv = phi.fixed_view::<3, 3>(3, 3).into_owned();
        let vf_minus = phi_vr * r0 + phi_vv * v0_plus;
        let dv1 = -vf_minus; // null it
        Some((dv0, dv1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // RK4 integration of the CW ODE — the independent oracle for the STM.
    fn cw_deriv(n: f64, s: &Vector6<f64>) -> Vector6<f64> {
        let (x, _y, z, vx, vy, vz) = (s[0], s[1], s[2], s[3], s[4], s[5]);
        Vector6::new(vx, vy, vz, 3.0 * n * n * x + 2.0 * n * vy, -2.0 * n * vx, -n * n * z)
    }

    fn rk4(n: f64, s0: &Vector6<f64>, t: f64, steps: usize) -> Vector6<f64> {
        let h = t / steps as f64;
        let mut s = *s0;
        for _ in 0..steps {
            let k1 = cw_deriv(n, &s);
            let k2 = cw_deriv(n, &(s + 0.5 * h * k1));
            let k3 = cw_deriv(n, &(s + 0.5 * h * k2));
            let k4 = cw_deriv(n, &(s + h * k3));
            s += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        s
    }

    #[test]
    fn the_stm_matches_numerical_integration() {
        // THE ORACLE. The closed-form Φ(t) must reproduce a high-resolution RK4 of the CW ODE.
        let cw = Cw { n: 0.0011 }; // ~LEO mean motion (period ~95 min)
        let s0 = Vector6::new(50.0, -120.0, 30.0, 0.2, -0.1, 0.05);
        for &t in &[100.0, 800.0, 2000.0, 4000.0] {
            let analytic = cw.propagate(&s0, t);
            let numeric = rk4(cw.n, &s0, t, 20000);
            assert!((analytic - numeric).norm() < 1e-4, "Φ vs RK4 mismatch at t={t}: {} vs {}", analytic, numeric);
        }
    }

    #[test]
    fn the_stm_is_identity_at_zero() {
        let cw = Cw { n: 0.0011 };
        assert!((cw.stm(0.0) - Matrix6::identity()).norm() < 1e-12);
    }

    #[test]
    fn the_cross_track_channel_is_simple_harmonic() {
        // z decouples: z̈ + n²z = 0 ⇒ period 2π/n. After one full period the cross-track state returns.
        let cw = Cw { n: 0.0011 };
        let period = std::f64::consts::TAU / cw.n;
        let s0 = Vector6::new(0.0, 0.0, 40.0, 0.0, 0.0, 0.3);
        let after = cw.propagate(&s0, period);
        assert!((after[2] - s0[2]).abs() < 1e-6 && (after[5] - s0[5]).abs() < 1e-6, "cross-track should be periodic: {} {}", after[2], after[5]);
    }

    #[test]
    fn two_impulse_rendezvous_reaches_the_target_with_zero_relative_velocity() {
        // THE HEADLINE. From a 150 m along-track / 40 m radial offset, plan a half-orbit transfer to the
        // origin (the target). Apply dv0, coast, apply dv1 — and land on the target, at rest relative to it.
        let cw = Cw { n: 0.0011 };
        let r0 = Vector3::new(40.0, 150.0, -20.0);
        let v0 = Vector3::new(0.0, 0.0, 0.0);
        let rf = Vector3::zeros();
        let period = std::f64::consts::TAU / cw.n;
        let tof = 0.37 * period; // avoid the singular multiples of the period
        let (dv0, dv1) = cw.two_impulse_rendezvous(&r0, &v0, &rf, tof).expect("Φ_rv invertible");

        // fly it: state just after departure burn
        let mut s = Vector6::new(r0.x, r0.y, r0.z, v0.x + dv0.x, v0.y + dv0.y, v0.z + dv0.z);
        s = cw.propagate(&s, tof);
        // position must be the target
        assert!(Vector3::new(s[0], s[1], s[2]).norm() < 1e-6, "should arrive at the target: {} {} {}", s[0], s[1], s[2]);
        // arrival burn nulls the relative velocity
        let v_after = Vector3::new(s[3] + dv1.x, s[4] + dv1.y, s[5] + dv1.z);
        assert!(v_after.norm() < 1e-9, "arrival velocity should be nulled: {}", v_after.norm());
    }

    #[test]
    fn from_orbit_recovers_the_expected_mean_motion() {
        // A 400 km LEO: a = 6778 km, μ = 3.986e14. Period should be ~92.5 min.
        let cw = Cw::from_orbit(6.778e6, 3.986e14);
        let period_min = std::f64::consts::TAU / cw.n / 60.0;
        assert!((period_min - 92.5).abs() < 1.0, "LEO period ~92.5 min, got {period_min}");
    }
}
