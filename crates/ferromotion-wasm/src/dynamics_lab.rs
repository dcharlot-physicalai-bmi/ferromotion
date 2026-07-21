//! **Dynamics lab** — the rig behind the "equations of motion" lesson. It integrates a real **double
//! pendulum** using ferromotion-core's rigid-body dynamics: every step calls [`forward_dynamics`] (which
//! builds the joint-space inertia matrix `M(q)`, the gravity/bias forces, and solves `M q̈ = τ − bias`), so
//! the reader watches the *actual* equation of motion `M(q) q̈ + C(q,q̇) q̇ + G(q) = τ` play out, and can read
//! `M(q)` and `G(q)` live as the arm swings. Two links, planar, gravity along −z; no motor torque (`τ = 0`),
//! so it is a free swing — the classic chaotic double pendulum.

use ferromotion_core::{forward_dynamics, from_urdf_full, gravity_vector, mass_matrix, LinkInertia, Robot};
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

const URDF: &str = r#"<robot name="dpend">
  <link name="base"/>
  <link name="l1"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.083"/></inertial></link>
  <link name="l2"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.083"/></inertial></link>
  <link name="tool"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-6.3" upper="6.3" effort="100" velocity="100"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-6.3" upper="6.3" effort="100" velocity="100"/></joint>
  <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="1 0 0" rpy="0 0 0"/></joint>
</robot>"#;

const G: f64 = 9.81;
const L: f64 = 1.0; // link length
const MASS: f64 = 1.0;

#[wasm_bindgen]
pub struct DynamicsLab {
    robot: Robot,
    inertia: Vec<LinkInertia>,
    q: [f64; 2],
    qd: [f64; 2],
    t: f64,
}

#[wasm_bindgen]
impl DynamicsLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> DynamicsLab {
        let (robot, inertia) = from_urdf_full(URDF, "base", "tool").expect("valid URDF");
        // start raised, so gravity does visible work
        DynamicsLab { robot, inertia, q: [0.4, 0.6], qd: [0.0, 0.0], t: 0.0 }
    }

    pub fn set_state(&mut self, q0: f64, q1: f64, qd0: f64, qd1: f64) {
        self.q = [q0, q1];
        self.qd = [qd0, qd1];
        self.t = 0.0;
    }

    /// Advance the free swing by `dt` with semi-implicit (symplectic) Euler, using the real forward dynamics
    /// `q̈ = M(q)⁻¹(τ − bias)` with `τ = 0`. Sub-steps for stability.
    pub fn step(&mut self, dt: f64) {
        let sub = 8;
        let h = dt / sub as f64;
        for _ in 0..sub {
            let qdd = forward_dynamics(
                &self.robot,
                &self.inertia,
                &self.q,
                &self.qd,
                &[0.0, 0.0],
                Vector3::new(0.0, 0.0, -G),
            );
            for (i, &a) in qdd.iter().enumerate() {
                self.qd[i] += a * h;
                self.q[i] += self.qd[i] * h;
            }
        }
        self.t += dt;
    }

    // --- joint state ---
    pub fn q0(&self) -> f64 { self.q[0] }
    pub fn q1(&self) -> f64 { self.q[1] }
    pub fn qd0(&self) -> f64 { self.qd[0] }
    pub fn qd1(&self) -> f64 { self.qd[1] }
    pub fn time(&self) -> f64 { self.t }

    // --- the equation of motion, computed live ---
    /// Joint-space inertia (mass) matrix entry (row, col), `M(q)`.
    pub fn mass(&self, r: usize, c: usize) -> f64 {
        mass_matrix(&self.robot, &self.inertia, &self.q)[(r, c)]
    }
    /// Gravity/bias generalized-force entry `G(q)[i]`.
    pub fn gravity(&self, i: usize) -> f64 {
        gravity_vector(&self.robot, &self.inertia, &self.q, Vector3::new(0.0, 0.0, -G))[i]
    }
    /// Joint acceleration `q̈[i]` right now (free swing).
    pub fn accel(&self, i: usize) -> f64 {
        forward_dynamics(&self.robot, &self.inertia, &self.q, &self.qd, &[0.0, 0.0], Vector3::new(0.0, 0.0, -G))[i]
    }

    // --- planar draw geometry (base at origin; screen = world x, and +down = gravity −z) ---
    pub fn joint1_x(&self) -> f64 { L * self.q[0].cos() }
    pub fn joint1_y(&self) -> f64 { L * self.q[0].sin() }
    pub fn tip_x(&self) -> f64 { self.joint1_x() + L * (self.q[0] + self.q[1]).cos() }
    pub fn tip_y(&self) -> f64 { self.joint1_y() + L * (self.q[0] + self.q[1]).sin() }

    /// Total mechanical energy `T + U` (for the next lesson; here a live readout). `T = ½ q̇ᵀ M q̇`,
    /// `U = Σ mᵢ g zᵢ` with world-up `+z`.
    pub fn energy(&self) -> f64 {
        let m = mass_matrix(&self.robot, &self.inertia, &self.q);
        let qd = nalgebra::DVector::from_row_slice(&self.qd);
        let ke = 0.5 * (qd.transpose() * &m * &qd)[(0, 0)];
        // COM world-z heights: link1 COM at 0.5 along link1; link2 COM at joint1 + 0.5 along link2
        let z1 = -0.5 * L * self.q[0].sin();
        let z2 = -L * self.q[0].sin() - 0.5 * L * (self.q[0] + self.q[1]).sin();
        let pe = MASS * G * z1 + MASS * G * z2;
        ke + pe
    }
}

impl Default for DynamicsLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_matrix_is_symmetric_positive_definite() {
        let lab = DynamicsLab::new();
        // M symmetric
        assert!((lab.mass(0, 1) - lab.mass(1, 0)).abs() < 1e-12, "M must be symmetric");
        // diagonal positive, and det > 0 (SPD for a real robot)
        let det = lab.mass(0, 0) * lab.mass(1, 1) - lab.mass(0, 1) * lab.mass(1, 0);
        assert!(lab.mass(0, 0) > 0.0 && lab.mass(1, 1) > 0.0 && det > 0.0, "M must be SPD");
    }

    #[test]
    fn a_free_swing_starts_falling_under_gravity() {
        // Raised from rest, the arm must begin to move (nonzero acceleration) — gravity does work.
        let mut lab = DynamicsLab::new();
        lab.set_state(0.3, 0.2, 0.0, 0.0);
        assert!(lab.accel(0).abs() > 1e-3, "gravity should accelerate the raised arm");
        let e0 = lab.energy();
        for _ in 0..50 {
            lab.step(0.01);
        }
        assert!(lab.qd0().abs() + lab.qd1().abs() > 1e-2, "the arm should be swinging");
        // symplectic Euler roughly conserves energy over a short horizon
        assert!((lab.energy() - e0).abs() < 0.5, "energy should stay near its start: {} vs {e0}", lab.energy());
    }
}
