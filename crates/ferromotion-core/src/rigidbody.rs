//! Symplectic free-rigid-body integrator (maximal coordinates: orientation quaternion + body
//! angular momentum) — the structural piece Dojo's variational integrator provides that a serial
//! RNEA chain doesn't. Rotational dynamics are integrated with the **implicit midpoint rule on the
//! body angular momentum** `Π̇ = Π × (J⁻¹Π) + τ`; because kinetic energy `½ΠᵀJ⁻¹Π` and `‖Π‖²` are
//! quadratic, implicit midpoint conserves them essentially exactly — no secular energy/momentum
//! drift, unlike explicit Euler. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

/// A free rigid body with a diagonal inertia (principal moments) in its body frame.
#[derive(Clone, Debug)]
pub struct RigidBody {
    pub orientation: UnitQuaternion<f64>,
    /// Angular momentum in the body frame, `Π = J·ω`.
    pub angular_momentum: Vector3<f64>,
    /// Principal moments of inertia `(Jx, Jy, Jz)`.
    pub inertia: Vector3<f64>,
}

impl RigidBody {
    /// Construct from a body angular velocity.
    pub fn from_omega(orientation: UnitQuaternion<f64>, omega_body: Vector3<f64>, inertia: Vector3<f64>) -> Self {
        Self { orientation, angular_momentum: omega_body.component_mul(&inertia), inertia }
    }

    fn inv_inertia(&self) -> Vector3<f64> {
        Vector3::new(1.0 / self.inertia.x, 1.0 / self.inertia.y, 1.0 / self.inertia.z)
    }

    /// Body angular velocity `ω = J⁻¹·Π`.
    pub fn omega_body(&self) -> Vector3<f64> {
        self.angular_momentum.component_mul(&self.inv_inertia())
    }

    /// Rotational kinetic energy `½ ωᵀ J ω` (conserved for a free body).
    pub fn energy(&self) -> f64 {
        0.5 * self.omega_body().dot(&self.angular_momentum)
    }

    /// Angular momentum in the world frame `R·Π` (conserved for a free body).
    pub fn angular_momentum_world(&self) -> Vector3<f64> {
        self.orientation * self.angular_momentum
    }

    /// Advance one step under a constant body-frame torque (`torque_body = 0` ⇒ free tumble).
    pub fn step(&mut self, dt: f64, torque_body: Vector3<f64>) {
        let jinv = self.inv_inertia();
        // Implicit midpoint: solve Π₁ = Π₀ + dt·(Π_mid × J⁻¹Π_mid + τ), Π_mid = (Π₀+Π₁)/2.
        let mut pi1 = self.angular_momentum;
        for _ in 0..40 {
            let pim = (self.angular_momentum + pi1) * 0.5;
            let om = pim.component_mul(&jinv);
            let next = self.angular_momentum + (pim.cross(&om) + torque_body) * dt;
            if (next - pi1).norm() < 1e-14 {
                pi1 = next;
                break;
            }
            pi1 = next;
        }
        let pim = (self.angular_momentum + pi1) * 0.5;
        let om_mid = pim.component_mul(&jinv);
        // Orientation update on SO(3) via the exponential map (unit-quaternion preserving).
        self.orientation *= UnitQuaternion::from_scaled_axis(om_mid * dt);
        self.angular_momentum = pi1;
    }

    /// Rotation matrix of the current orientation.
    pub fn rotation(&self) -> Matrix3<f64> {
        *self.orientation.to_rotation_matrix().matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_tumble_conserves_energy_and_angular_momentum() {
        // Asymmetric body spun near its intermediate axis (Dzhanibekov regime).
        let mut rb = RigidBody::from_omega(
            UnitQuaternion::identity(),
            Vector3::new(0.05, 4.0, 0.05),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let (e0, l0) = (rb.energy(), rb.angular_momentum_world().norm());
        let dt = 0.01;
        let mut tumbled = false;
        for _ in 0..3000 {
            rb.step(dt, Vector3::zeros());
            if rb.omega_body().x.abs() > 0.5 {
                tumbled = true; // the spin axis genuinely wandered (nontrivial dynamics)
            }
        }
        let (e1, l1) = (rb.energy(), rb.angular_momentum_world().norm());
        // Quadratic invariants ⇒ implicit midpoint conserves them to ~solver tolerance.
        assert!(((e1 - e0) / e0).abs() < 1e-6, "energy drifted: {e0} → {e1}");
        assert!(((l1 - l0) / l0).abs() < 1e-6, "angular momentum drifted: {l0} → {l1}");
        assert!((rb.orientation.norm() - 1.0).abs() < 1e-9, "quaternion left the unit sphere");
        assert!(tumbled, "expected nontrivial tumbling of the spin axis");
    }

    #[test]
    fn torque_about_a_principal_axis_spins_up_linearly() {
        // Constant torque about z on a body initially at rest: ωz(t) ≈ (τ/Jz)·t.
        let mut rb = RigidBody::from_omega(UnitQuaternion::identity(), Vector3::zeros(), Vector3::new(1.0, 1.0, 2.0));
        let (dt, tau) = (0.001, 0.5);
        for _ in 0..1000 {
            rb.step(dt, Vector3::new(0.0, 0.0, tau));
        }
        let expected = tau / rb.inertia.z * (1000.0 * dt); // = 0.25
        assert!((rb.omega_body().z - expected).abs() < 1e-6, "ωz = {} vs {expected}", rb.omega_body().z);
    }
}
