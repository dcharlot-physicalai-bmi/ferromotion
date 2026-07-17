//! **Marine-craft dynamics + line-of-sight guidance** (Fossen, *Handbook of Marine Craft
//! Hydrodynamics and Motion Control*) — the standard 6-DOF model for surface and underwater vehicles,
//! opening the marine domain alongside the aerial [`crate::quadrotor`] model.
//!
//! In the body frame, with pose `η = [p; Θ]` (NED position + roll-pitch-yaw) and velocity
//! `ν = [v; ω]` (surge/sway/heave, roll/pitch/yaw rates):
//!
//! ```text
//!   M ν̇ + C(ν) ν + D(ν) ν + g(η) = τ,     η̇ = J(η) ν
//! ```
//!
//! `M = Mᵀ ≻ 0` is rigid-body + added mass; `C(ν) = −C(ν)ᵀ` is Coriolis–centripetal (built from `M`,
//! so it does no work); `D(ν) ⪰ 0` is linear + quadratic hydrodynamic damping (dissipative); `g(η)` is
//! the gravity/buoyancy restoring vector; `J(η)` maps body velocity to the NED frame. Those structural
//! properties are the model's backbone and are asserted in the tests. Under-actuated path following is
//! closed with **line-of-sight guidance**, which steers the heading to drive cross-track error to zero.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Matrix6, Rotation3, Vector3, Vector6};

/// Skew-symmetric cross-product matrix `S(a)` with `S(a) b = a × b`.
fn skew(a: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -a.z, a.y, a.z, 0.0, -a.x, -a.y, a.x, 0.0)
}

/// A 6-DOF marine craft: mass (with added mass), damping, and restoring parameters.
#[derive(Clone, Debug)]
pub struct MarineCraft {
    /// Total inertia `M = M_RB + M_A` (symmetric, positive-definite).
    pub m: Matrix6<f64>,
    /// Linear damping `D_l` (positive-definite).
    pub d_lin: Matrix6<f64>,
    /// Quadratic-damping coefficients per DOF: `D_n(ν) = diag(d_quad_i · |ν_i|)`.
    pub d_quad: Vector6<f64>,
    /// Weight `W = m g` and buoyancy `B = ρ g ∇`.
    pub weight: f64,
    pub buoyancy: f64,
    /// Centre of gravity and centre of buoyancy, in the body frame.
    pub r_g: Vector3<f64>,
    pub r_b: Vector3<f64>,
}

impl MarineCraft {
    /// Coriolis–centripetal matrix built from `M` by Fossen's skew-symmetric parameterization, so
    /// `C(ν) = −C(ν)ᵀ` and hence `νᵀ C(ν) ν = 0` (Coriolis forces do no work) for any `M = Mᵀ`.
    pub fn coriolis(&self, nu: Vector6<f64>) -> Matrix6<f64> {
        let (v, w) = (nu.fixed_rows::<3>(0).into_owned(), nu.fixed_rows::<3>(3).into_owned());
        let m11 = self.m.fixed_view::<3, 3>(0, 0).into_owned();
        let m12 = self.m.fixed_view::<3, 3>(0, 3).into_owned();
        let m21 = self.m.fixed_view::<3, 3>(3, 0).into_owned();
        let m22 = self.m.fixed_view::<3, 3>(3, 3).into_owned();
        let a = m11 * v + m12 * w; // "linear momentum"
        let b = m21 * v + m22 * w; // "angular momentum"
        let mut c = Matrix6::zeros();
        c.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(a)));
        c.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-skew(a)));
        c.fixed_view_mut::<3, 3>(3, 3).copy_from(&(-skew(b)));
        c
    }

    /// Total hydrodynamic damping `D(ν) = D_l + diag(d_quad_i |ν_i|)` (positive-definite ⇒ dissipative).
    pub fn damping(&self, nu: Vector6<f64>) -> Matrix6<f64> {
        let mut d = self.d_lin;
        for i in 0..6 {
            d[(i, i)] += self.d_quad[i] * nu[i].abs();
        }
        d
    }

    /// Gravity/buoyancy restoring generalized force `g(η)` (Fossen eq. 4.6), a function of orientation.
    pub fn restoring(&self, roll: f64, pitch: f64) -> Vector6<f64> {
        let (w, b) = (self.weight, self.buoyancy);
        let (sphi, cphi) = (roll.sin(), roll.cos());
        let (sth, cth) = (pitch.sin(), pitch.cos());
        let (xg, yg, zg) = (self.r_g.x, self.r_g.y, self.r_g.z);
        let (xb, yb, zb) = (self.r_b.x, self.r_b.y, self.r_b.z);
        Vector6::new(
            (w - b) * sth,
            -(w - b) * cth * sphi,
            -(w - b) * cth * cphi,
            -(yg * w - yb * b) * cth * cphi + (zg * w - zb * b) * cth * sphi,
            (zg * w - zb * b) * sth + (xg * w - xb * b) * cth * cphi,
            -(xg * w - xb * b) * cth * sphi - (yg * w - yb * b) * sth,
        )
    }

    /// Body→NED velocity transform `J(η)`: rotation for the linear part, Euler-rate matrix `T(Θ)` for
    /// the angular part (roll-pitch-yaw / ZYX).
    pub fn jacobian(&self, roll: f64, pitch: f64, yaw: f64) -> Matrix6<f64> {
        let r = Rotation3::from_euler_angles(roll, pitch, yaw).into_inner();
        let (sphi, cphi) = (roll.sin(), roll.cos());
        let (cth, tth) = (pitch.cos(), pitch.tan());
        let t = Matrix3::new(
            1.0, sphi * tth, cphi * tth, //
            0.0, cphi, -sphi, //
            0.0, sphi / cth, cphi / cth,
        );
        let mut j = Matrix6::zeros();
        j.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        j.fixed_view_mut::<3, 3>(3, 3).copy_from(&t);
        j
    }

    /// Body-frame acceleration `ν̇ = M⁻¹ (τ − C(ν)ν − D(ν)ν − g(η))`.
    pub fn accel(&self, nu: Vector6<f64>, roll: f64, pitch: f64, tau: Vector6<f64>) -> Vector6<f64> {
        let rhs = tau - self.coriolis(nu) * nu - self.damping(nu) * nu - self.restoring(roll, pitch);
        self.m.try_inverse().expect("mass matrix invertible") * rhs
    }

    /// Kinetic energy `½ νᵀ M ν` (the storage function whose decay proves passivity).
    pub fn kinetic_energy(&self, nu: Vector6<f64>) -> f64 {
        0.5 * (nu.transpose() * self.m * nu)[0]
    }

    /// One semi-implicit Euler step of the coupled kinematics + dynamics. `eta = [x,y,z,φ,θ,ψ]`.
    pub fn step(&self, eta: Vector6<f64>, nu: Vector6<f64>, tau: Vector6<f64>, dt: f64) -> (Vector6<f64>, Vector6<f64>) {
        let acc = self.accel(nu, eta[3], eta[4], tau);
        let nu2 = nu + acc * dt;
        let eta2 = eta + self.jacobian(eta[3], eta[4], eta[5]) * nu2 * dt;
        (eta2, nu2)
    }
}

/// **Line-of-sight guidance** for path following: the heading that steers a vehicle onto the straight
/// line through `p0` with direction angle `path_angle`. Cross-track error `e` is driven to zero by
/// aiming a look-ahead distance `delta` down the path: `ψ_d = path_angle − atan(e / delta)`.
pub fn los_heading(pos: Vector3<f64>, p0: Vector3<f64>, path_angle: f64, delta: f64) -> (f64, f64) {
    let (dx, dy) = (pos.x - p0.x, pos.y - p0.y);
    // Cross-track error: perpendicular distance to the path (positive = left of the path direction).
    let e = -(dx * path_angle.sin()) + dy * path_angle.cos();
    let psi_d = path_angle - (e / delta).atan();
    (psi_d, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A neutrally-buoyant, bottom-heavy AUV: diagonal mass+added-mass, linear+quadratic damping,
    /// centre of gravity below the centre of buoyancy (metacentric restoring in roll/pitch).
    fn auv() -> MarineCraft {
        let m = Matrix6::from_diagonal(&Vector6::new(30.0, 40.0, 40.0, 2.0, 8.0, 8.0));
        let d_lin = Matrix6::from_diagonal(&Vector6::new(12.0, 20.0, 20.0, 4.0, 6.0, 6.0));
        let d_quad = Vector6::new(20.0, 35.0, 35.0, 2.0, 5.0, 5.0);
        MarineCraft {
            m,
            d_lin,
            d_quad,
            weight: 300.0,
            buoyancy: 300.0,        // neutrally buoyant
            r_g: Vector3::new(0.0, 0.0, 0.05), // CG below CB (z down in NED) ⇒ righting moment
            r_b: Vector3::new(0.0, 0.0, 0.0),
        }
    }

    #[test]
    fn mass_matrix_is_symmetric_positive_definite() {
        let c = auv();
        assert!((c.m - c.m.transpose()).norm() < 1e-12, "M must be symmetric");
        let eig = c.m.symmetric_eigenvalues();
        assert!(eig.iter().all(|&e| e > 0.0), "M must be positive-definite: {eig:?}");
    }

    #[test]
    fn coriolis_is_skew_symmetric_and_does_no_work() {
        // THE BACKBONE PROPERTY. C(ν) = −C(ν)ᵀ for any ν, so νᵀC(ν)ν = 0 — Coriolis forces are
        // work-conserving, which is what lets the damping alone dissipate energy.
        let c = auv();
        for nu in [
            Vector6::new(1.0, -0.5, 0.3, 0.2, -0.1, 0.4),
            Vector6::new(-2.0, 1.5, 0.0, -0.3, 0.6, -0.2),
        ] {
            let cm = c.coriolis(nu);
            assert!((cm + cm.transpose()).norm() < 1e-12, "C must be skew-symmetric");
            let work = (nu.transpose() * cm * nu)[0];
            assert!(work.abs() < 1e-12, "Coriolis must do no work: νᵀCν = {work}");
        }
    }

    #[test]
    fn damping_is_dissipative() {
        let c = auv();
        for nu in [Vector6::new(1.2, -0.8, 0.5, 0.3, -0.4, 0.6), Vector6::new(-2.0, 0.0, 1.0, 0.0, 0.5, 0.0)] {
            let power = (nu.transpose() * c.damping(nu) * nu)[0];
            assert!(power > 0.0, "damping must remove energy: νᵀDν = {power}");
        }
    }

    #[test]
    fn energy_dissipates_to_rest_with_no_input() {
        // Passivity in action: no thrust, level attitude (no restoring), so ½νᵀMν must fall
        // monotonically to zero as the hydrodynamic damping bleeds off the kinetic energy.
        let c = auv();
        let (mut eta, mut nu) = (Vector6::zeros(), Vector6::new(2.0, 1.0, -0.5, 0.4, 0.0, 0.3));
        let mut e_prev = c.kinetic_energy(nu);
        for _ in 0..50_000 {
            // hold attitude level so restoring stays zero, isolating the damping
            eta[3] = 0.0;
            eta[4] = 0.0;
            let (_e2, n2) = c.step(eta, nu, Vector6::zeros(), 1e-3);
            nu = n2;
            let e = c.kinetic_energy(nu);
            assert!(e <= e_prev + 1e-9, "kinetic energy must not increase: {e} > {e_prev}");
            e_prev = e;
        }
        assert!(c.kinetic_energy(nu) < 1e-4, "should coast to rest: {}", c.kinetic_energy(nu));
    }

    #[test]
    fn decoupled_surge_matches_the_first_order_analytic_response() {
        // Pure surge with linear damping is m_u u̇ = τ − d_u u ⇒ u(t) = (τ/d_u)(1 − e^{−(d_u/m_u)t}).
        // Use tiny velocities so the quadratic term is negligible, and drive only surge.
        let mut c = auv();
        c.d_quad = Vector6::zeros(); // isolate the linear response
        let (m_u, d_u, thrust) = (c.m[(0, 0)], c.d_lin[(0, 0)], 0.5);
        let (mut eta, mut nu) = (Vector6::zeros(), Vector6::zeros());
        let tau = Vector6::new(thrust, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (dt, steps) = (1e-4, 20_000); // t = 2 s
        for _ in 0..steps {
            let (_e, n) = c.step(eta, nu, tau, dt);
            nu = n;
            eta[3] = 0.0;
            eta[4] = 0.0; // keep level (no restoring/coupling)
        }
        let t = steps as f64 * dt;
        let analytic = thrust / d_u * (1.0 - (-(d_u / m_u) * t).exp());
        assert!((nu[0] - analytic).abs() < 2e-3, "surge {} vs analytic {analytic}", nu[0]);
        // Motion stayed in surge only (no spurious coupling into other DOFs).
        assert!(nu.fixed_rows::<5>(1).norm() < 1e-9, "surge thrust leaked into other DOFs");
    }

    #[test]
    fn restoring_rights_a_bottom_heavy_vehicle() {
        // Zero at upright; a righting couple opposing any roll (CG below CB). Perturb roll and let it
        // settle — the restoring moment plus roll damping should return it toward level.
        let c = auv();
        assert!(c.restoring(0.0, 0.0).norm() < 1e-12, "restoring must vanish at upright");
        // The restoring vector sits on the LHS and is subtracted in `accel`, so a righting response is
        // a *negative* roll acceleration at a positive roll (from rest, no thrust).
        let ang_acc = c.accel(Vector6::zeros(), 0.3, 0.0, Vector6::zeros())[3];
        assert!(ang_acc < 0.0, "a positive roll should induce a righting (negative) angular acceleration: {ang_acc}");
        // simulate a released roll perturbation
        let (mut eta, mut nu): (Vector6<f64>, Vector6<f64>) = (Vector6::zeros(), Vector6::zeros());
        eta[3] = 0.4; // 0.4 rad roll
        let mut max_roll: f64 = eta[3].abs();
        for _ in 0..40_000 {
            let (e2, n2) = c.step(eta, nu, Vector6::zeros(), 1e-3);
            eta = e2;
            nu = n2;
            max_roll = max_roll.max(eta[3].abs());
        }
        assert!(eta[3].abs() < 0.05, "roll did not settle toward upright: {}", eta[3]);
        assert!(max_roll >= 0.4 - 1e-9, "sanity: started at 0.4 rad roll");
    }

    #[test]
    fn los_guidance_drives_cross_track_error_to_zero() {
        // An under-actuated surface craft: constant surge thrust, heading closed by LOS guidance
        // toward the x-axis path. The cross-track error (y) must converge to zero.
        let c = auv();
        // planar craft: no heave/roll/pitch dynamics excited; strong yaw authority for heading control.
        let (mut eta, mut nu): (Vector6<f64>, Vector6<f64>) =
            (Vector6::new(0.0, 3.0, 0.0, 0.0, 0.0, 1.2), Vector6::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.0));
        let (p0, path_angle, delta) = (Vector3::zeros(), 0.0, 2.0);
        let e0: f64 = eta[1].abs();
        let mut e_final = e0;
        for _ in 0..60_000 {
            let (psi_d, e) = los_heading(Vector3::new(eta[0], eta[1], 0.0), p0, path_angle, delta);
            // simple heading autopilot → yaw moment; constant surge thrust
            let yaw_err = {
                let mut d = psi_d - eta[5];
                while d > std::f64::consts::PI { d -= std::f64::consts::TAU }
                while d < -std::f64::consts::PI { d += std::f64::consts::TAU }
                d
            };
            let tau = Vector6::new(6.0, 0.0, 0.0, 0.0, 0.0, 8.0 * yaw_err - 4.0 * nu[5]);
            let (e2, n2) = c.step(eta, nu, tau, 1e-3);
            eta = e2;
            nu = n2;
            eta[2] = 0.0; // constrain to the horizontal plane
            eta[3] = 0.0;
            eta[4] = 0.0;
            e_final = e;
        }
        assert!(e_final.abs() < 0.15, "cross-track error did not converge: {e_final} (started {e0})");
        assert!(e0 > 1.0, "sanity: started well off the path");
    }
}
