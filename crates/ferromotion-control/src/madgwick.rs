//! **Madgwick AHRS filter** (Madgwick, 2010) — the de-facto orientation estimator on drones, IMUs, and
//! wearables. It fuses a **gyroscope** (accurate short-term rotation, but its integration drifts) with an
//! **accelerometer** (a noisy but drift-free gravity reference) into an attitude **quaternion**. Where a
//! plain complementary filter ([`crate::ComplementaryFilter`]) blends two rate estimates with a fixed
//! weight, Madgwick corrects the gyro-integrated quaternion with one step of **gradient descent** on the
//! error between the measured gravity direction and the direction the current orientation predicts — an
//! analytic Jacobian makes that step a handful of multiplies, so it runs at kHz on a microcontroller.
//!
//! The update is `q̇ = ½ q ⊗ (0, ω) − β · ∇f/‖∇f‖`, where `f` measures the gravity-alignment error. `β`
//! trades gyro trust against accelerometer trust. Verified: a static, level accelerometer drives the
//! estimate to level from any initial orientation; a tilted static accelerometer converges to the true
//! roll/pitch; and with a biased gyro the accelerometer correction holds the tilt while pure integration
//! drifts away. Pure `nalgebra` → WASM-clean.

use nalgebra::Vector3;

/// A Madgwick IMU (gyro + accelerometer) filter holding the orientation quaternion `(w, x, y, z)`.
#[derive(Clone, Copy, Debug)]
pub struct Madgwick {
    /// Orientation quaternion `[w, x, y, z]` (sensor relative to earth).
    pub q: [f64; 4],
    /// Filter gain: higher trusts the accelerometer more (faster convergence, more noise).
    pub beta: f64,
}

impl Madgwick {
    /// A level filter with the given gain.
    pub fn new(beta: f64) -> Self {
        Madgwick { q: [1.0, 0.0, 0.0, 0.0], beta }
    }

    /// One IMU update from gyro `ω` (rad/s) and accelerometer `a` (any units — only its direction is used)
    /// over timestep `dt`.
    pub fn update(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) {
        let [q0, q1, q2, q3] = self.q;
        // rate of change from the gyroscope: q̇ = ½ q ⊗ (0, ω)
        let (gx, gy, gz) = (gyro.x, gyro.y, gyro.z);
        let mut dq = [
            0.5 * (-q1 * gx - q2 * gy - q3 * gz),
            0.5 * (q0 * gx + q2 * gz - q3 * gy),
            0.5 * (q0 * gy - q1 * gz + q3 * gx),
            0.5 * (q0 * gz + q1 * gy - q2 * gx),
        ];

        // accelerometer correction (skip if the reading is degenerate)
        let anorm = accel.norm();
        if anorm > 1e-9 {
            let a = accel / anorm;
            // objective f: predicted gravity direction minus measured
            let f = [
                2.0 * (q1 * q3 - q0 * q2) - a.x,
                2.0 * (q0 * q1 + q2 * q3) - a.y,
                2.0 * (0.5 - q1 * q1 - q2 * q2) - a.z,
            ];
            // Jacobianᵀ · f  (the gradient ∇f), 4-vector
            let mut grad = [
                -2.0 * q2 * f[0] + 2.0 * q1 * f[1],
                2.0 * q3 * f[0] + 2.0 * q0 * f[1] - 4.0 * q1 * f[2],
                -2.0 * q0 * f[0] + 2.0 * q3 * f[1] - 4.0 * q2 * f[2],
                2.0 * q1 * f[0] + 2.0 * q2 * f[1],
            ];
            let gn = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2] + grad[3] * grad[3]).sqrt();
            if gn > 1e-12 {
                for k in 0..4 {
                    grad[k] /= gn;
                    dq[k] -= self.beta * grad[k];
                }
            }
        }

        // integrate and normalize
        let mut q = [q0 + dq[0] * dt, q1 + dq[1] * dt, q2 + dq[2] * dt, q3 + dq[3] * dt];
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        for qi in &mut q {
            *qi /= n;
        }
        self.q = q;
    }

    /// Roll (rotation about x), radians.
    pub fn roll(&self) -> f64 {
        let [w, x, y, z] = self.q;
        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y))
    }

    /// Pitch (rotation about y), radians.
    pub fn pitch(&self) -> f64 {
        let [w, x, y, z] = self.q;
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_static_accelerometer_drives_the_estimate_level() {
        // THE ORACLE. Gravity along +z, gyro silent: from a tilted initial quaternion the filter converges
        // to level (roll = pitch = 0).
        let mut f = Madgwick { q: [0.92, 0.3, 0.2, 0.1], beta: 0.5 };
        let up = Vector3::new(0.0, 0.0, 1.0);
        for _ in 0..4000 {
            f.update(Vector3::zeros(), up, 0.01);
        }
        assert!(f.roll().abs() < 1e-2 && f.pitch().abs() < 1e-2, "should level out: roll {} pitch {}", f.roll(), f.pitch());
    }

    #[test]
    fn a_tilted_static_accelerometer_converges_to_the_true_roll() {
        // A sensor rolled by φ measures gravity as (0, sinφ, cosφ) in its own frame; the filter must recover
        // roll = φ.
        let phi = 0.3_f64;
        let accel = Vector3::new(0.0, phi.sin(), phi.cos());
        let mut f = Madgwick::new(0.5);
        for _ in 0..5000 {
            f.update(Vector3::zeros(), accel, 0.01);
        }
        assert!((f.roll() - phi).abs() < 1e-2, "roll should converge to {phi}: got {}", f.roll());
        assert!(f.pitch().abs() < 1e-2, "pitch should stay 0: {}", f.pitch());
    }

    #[test]
    fn the_accelerometer_correction_arrests_gyro_drift() {
        // THE HEADLINE. A constant gyro bias makes pure integration drift without bound; the Madgwick
        // accelerometer term holds the tilt near truth. Compare against a bias-only integrator.
        let bias = Vector3::new(0.02, -0.015, 0.0); // rad/s of phantom rotation
        let up = Vector3::new(0.0, 0.0, 1.0);
        let mut fused = Madgwick::new(0.3);
        // pure-integration reference (β = 0 ⇒ no correction)
        let mut drifting = Madgwick::new(0.0);
        for _ in 0..3000 {
            fused.update(bias, up, 0.01);
            drifting.update(bias, up, 0.01);
        }
        let fused_tilt = fused.roll().hypot(fused.pitch());
        let drift_tilt = drifting.roll().hypot(drifting.pitch());
        assert!(fused_tilt < 0.05, "fused estimate should stay near level: {fused_tilt}");
        assert!(drift_tilt > 3.0 * fused_tilt, "pure integration should drift far more: {drift_tilt} vs {fused_tilt}");
    }
}
