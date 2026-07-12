//! Admittance control and hybrid force/position control.
//!
//! Admittance renders a virtual second-order mechanics `M·ẍ + D·ẋ + K·(x − x_ref) = F_ext`:
//! given a measured external wrench it integrates a *compliant* reference motion, so a stiff
//! position-controlled manipulator behaves like a programmable spring-damper. Hybrid
//! force/position control (Raibert–Craig) uses a diagonal selection matrix `S` to split the task
//! frame into position-regulated axes (`S = 1`) and force-regulated axes (`S = 0`), producing one
//! Cartesian command wrench that cleanly partitions the two subspaces.

use nalgebra::Vector3;

/// Virtual second-order admittance: `M·ẍ + D·ẋ + K·(x − x_ref) = F_ext`.
///
/// Isotropic scalar gains over a 3-D Cartesian point. `step` advances the compliant reference by
/// one semi-implicit Euler tick and returns the new commanded position/velocity.
#[derive(Clone, Copy, Debug)]
pub struct Admittance {
    /// Virtual mass (inertia). Must be > 0.
    pub m: f64,
    /// Virtual damping.
    pub d: f64,
    /// Virtual stiffness (spring toward `x_ref`).
    pub k: f64,
}

impl Admittance {
    pub fn new(m: f64, d: f64, k: f64) -> Self {
        Self { m, d, k }
    }

    /// Advance the compliant reference by `dt` under external wrench `f_ext`.
    ///
    /// `x`/`xdot` are the current commanded pose/velocity, `x_ref` the (stiff) equilibrium the
    /// spring pulls toward. Returns `(x_next, xdot_next)`. Semi-implicit Euler is used for the
    /// same unconditional stability the closed-loop robot tests rely on.
    pub fn step(
        &self,
        dt: f64,
        x: Vector3<f64>,
        xdot: Vector3<f64>,
        x_ref: Vector3<f64>,
        f_ext: Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        // M·ẍ = F_ext − D·ẋ − K·(x − x_ref)  ⇒  ẍ = (…)/M.
        let xddot = (f_ext - self.d * xdot - self.k * (x - x_ref)) / self.m;
        let xdot_next = xdot + xddot * dt;
        let x_next = x + xdot_next * dt;
        (x_next, xdot_next)
    }
}

/// Hybrid force/position control with a diagonal selection matrix.
///
/// Each Cartesian axis is either position-controlled (`selection = 1`) or force-controlled
/// (`selection = 0`). The command wrench is
/// `w = S·(Kp·(x_des − x) − Kd·ẋ) + (I − S)·(f_des + Kf·(f_des − f_meas) − Kd·ẋ)`,
/// so the position law acts only on the position subspace and the force law only on its
/// complement — the two never fight over an axis.
#[derive(Clone, Copy, Debug)]
pub struct HybridForcePosition {
    /// Position (stiffness) gain on position-controlled axes.
    pub kp: f64,
    /// Velocity damping (applied on every axis for stability).
    pub kd: f64,
    /// Force-feedback gain on force-controlled axes (closes the loop around `f_des`).
    pub kf: f64,
}

impl HybridForcePosition {
    pub fn new(kp: f64, kd: f64, kf: f64) -> Self {
        Self { kp, kd, kf }
    }

    /// Cartesian command wrench.
    ///
    /// `selection` holds 1.0 for position-controlled axes and 0.0 for force-controlled axes.
    /// `x`/`xdot` are the measured pose/velocity, `x_des` the position target, `f_des` the desired
    /// contact wrench, and `f_meas` the measured contact wrench.
    pub fn command(
        &self,
        selection: Vector3<f64>,
        x: Vector3<f64>,
        xdot: Vector3<f64>,
        x_des: Vector3<f64>,
        f_des: Vector3<f64>,
        f_meas: Vector3<f64>,
    ) -> Vector3<f64> {
        let mut w = Vector3::zeros();
        for i in 0..3 {
            let s = selection[i];
            let pos = self.kp * (x_des[i] - x[i]) - self.kd * xdot[i];
            let force = f_des[i] + self.kf * (f_des[i] - f_meas[i]) - self.kd * xdot[i];
            // S selects position, (1 − S) selects force — a clean partition for s ∈ {0, 1}.
            w[i] = s * pos + (1.0 - s) * force;
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admittance_relaxes_to_spring_law() {
        // Under a constant external force the compliant reference settles at x_ref + F/K.
        let adm = Admittance::new(1.0, 8.0, 50.0);
        let x_ref = Vector3::new(0.2, -0.1, 0.4);
        let f_ext = Vector3::new(5.0, -3.0, 2.0);
        let (mut x, mut xdot) = (x_ref, Vector3::zeros());
        let dt = 1e-3;
        for _ in 0..20000 {
            let (xn, vn) = adm.step(dt, x, xdot, x_ref, f_ext);
            x = xn;
            xdot = vn;
        }
        let expected = x_ref + f_ext / adm.k;
        assert!((x - expected).norm() < 1e-4, "x = {x:?}, expected {expected:?}");
        // Well-damped: velocity has died out at steady state.
        assert!(xdot.norm() < 1e-4, "residual velocity {xdot:?}");
    }

    #[test]
    fn admittance_is_stable_and_monotone_enough() {
        // Over/under-damped choice should not blow up; displacement magnitude stays bounded by a
        // small multiple of the static spring deflection (no growing oscillation).
        let adm = Admittance::new(2.0, 20.0, 100.0);
        let x_ref = Vector3::zeros();
        let f_ext = Vector3::new(10.0, 0.0, 0.0);
        let static_defl = (f_ext / adm.k).norm();
        let (mut x, mut xdot) = (x_ref, Vector3::zeros());
        let dt = 1e-3;
        let mut peak = 0.0f64;
        for _ in 0..40000 {
            let (xn, vn) = adm.step(dt, x, xdot, x_ref, f_ext);
            x = xn;
            xdot = vn;
            peak = peak.max(x.norm());
        }
        assert!(peak < 2.0 * static_defl, "overshoot too large: peak {peak}, static {static_defl}");
        assert!((x - f_ext / adm.k).norm() < 1e-4, "did not settle: x = {x:?}");
    }

    #[test]
    fn admittance_zero_force_holds_reference() {
        // No external force ⇒ the reference should stay pinned at x_ref.
        let adm = Admittance::new(1.0, 5.0, 30.0);
        let x_ref = Vector3::new(0.3, 0.3, 0.0);
        let (mut x, mut xdot) = (x_ref, Vector3::zeros());
        for _ in 0..5000 {
            let (xn, vn) = adm.step(1e-3, x, xdot, x_ref, Vector3::zeros());
            x = xn;
            xdot = vn;
        }
        assert!((x - x_ref).norm() < 1e-9, "reference drifted to {x:?}");
    }

    #[test]
    fn hybrid_partitions_position_and_force() {
        // Axis 0: position-controlled (free space). Axes 1,2: force-controlled against a spring
        // environment f_meas = k_env · x. A unit point mass per axis is driven by the command
        // wrench minus the environment reaction on the force axes.
        let ctrl = HybridForcePosition::new(200.0, 30.0, 1.5);
        let selection = Vector3::new(1.0, 0.0, 0.0);
        let x_des = Vector3::new(0.5, 0.0, 0.0); // only axis 0 is meaningful
        let f_des = Vector3::new(0.0, 4.0, -3.0); // only axes 1,2 are meaningful
        let k_env = 500.0;
        let mass = 1.0;
        let dt = 1e-4;

        let (mut x, mut xdot) = (Vector3::zeros(), Vector3::zeros());
        for _ in 0..200_000 {
            // Environment contact wrench (only where in contact = the force axes have x ≠ 0).
            let f_meas = Vector3::new(0.0, k_env * x[1], k_env * x[2]);
            let w = ctrl.command(selection, x, xdot, x_des, f_des, f_meas);
            // Point-mass dynamics per axis: m·ẍ = w − f_env, where f_env acts on force axes only.
            let f_env = Vector3::new(0.0, k_env * x[1], k_env * x[2]);
            let xddot = (w - f_env) / mass;
            xdot += xddot * dt;
            x += xdot * dt;
        }

        // Position axis reached its target.
        assert!((x[0] - x_des[0]).abs() < 1e-3, "position axis at {}, want {}", x[0], x_des[0]);
        // Force axes: measured contact wrench converged to the desired force.
        let f_meas = Vector3::new(0.0, k_env * x[1], k_env * x[2]);
        assert!((f_meas[1] - f_des[1]).abs() < 1e-2, "force axis 1: {}, want {}", f_meas[1], f_des[1]);
        assert!((f_meas[2] - f_des[2]).abs() < 1e-2, "force axis 2: {}, want {}", f_meas[2], f_des[2]);
    }

    #[test]
    fn hybrid_command_selects_cleanly() {
        // With no motion, the command on a position axis is the pure position law and on a force
        // axis the pure force law — no cross-contamination.
        let ctrl = HybridForcePosition::new(100.0, 10.0, 2.0);
        let selection = Vector3::new(1.0, 0.0, 1.0);
        let x = Vector3::new(0.0, 0.0, 0.0);
        let xdot = Vector3::zeros();
        let x_des = Vector3::new(0.1, 99.0, -0.2);
        let f_des = Vector3::new(99.0, 5.0, 99.0);
        let f_meas = Vector3::new(99.0, 1.0, 99.0);
        let w = ctrl.command(selection, x, xdot, x_des, f_des, f_meas);
        // Axis 0 (position): kp * (0.1 − 0) = 10.0; f_des ignored.
        assert!((w[0] - 100.0 * 0.1).abs() < 1e-12);
        // Axis 1 (force): f_des + kf*(f_des − f_meas) = 5 + 2*(5−1) = 13; x_des ignored.
        assert!((w[1] - (5.0 + 2.0 * (5.0 - 1.0))).abs() < 1e-12);
        // Axis 2 (position): kp * (−0.2) = −20.
        assert!((w[2] - 100.0 * -0.2).abs() < 1e-12);
    }
}
