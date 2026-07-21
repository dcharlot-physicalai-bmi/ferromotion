//! **DeLaN lab** — the rig behind the "learn the structure, not the map" lesson. It trains a real
//! [`ferromotion_learn::Delan`] on inverse-dynamics samples from ferromotion-core's own double pendulum, then
//! lets the reader inspect any configuration `q` and compare the network's learned mass matrix `M(q)` against
//! the true one from recursive Newton–Euler. Two things to see: as training proceeds the learned `M(q)`
//! converges to the truth, and at *every* configuration — even before training — `M(q)` is symmetric
//! positive-definite, because the network outputs a Cholesky factor. Structure the black box cannot break.

use ferromotion_core::{from_urdf_full, inverse_dynamics, mass_matrix, LinkInertia, Robot};
use ferromotion_learn::Delan;
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

const L: f64 = 1.0;

#[wasm_bindgen]
pub struct DelanLab {
    robot: Robot,
    inertia: Vec<LinkInertia>,
    delan: Delan,
    q: Vec<[f64; 2]>,
    qd: Vec<[f64; 2]>,
    qdd: Vec<[f64; 2]>,
    tau: Vec<[f64; 2]>,
    cfg: [f64; 2],
    epochs: u32,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[wasm_bindgen]
impl DelanLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> DelanLab {
        let (robot, inertia) = from_urdf_full(URDF, "base", "tool").expect("valid URDF");
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut s = 987u64;
        let mut rnd = |lo: f64, hi: f64| lo + (splitmix64(&mut s) as f64 / u64::MAX as f64) * (hi - lo);
        let (mut q, mut qd, mut qdd, mut tau) = (vec![], vec![], vec![], vec![]);
        for _ in 0..200 {
            let qi = [rnd(-2.0, 2.0), rnd(-2.0, 2.0)];
            let qdi = [rnd(-1.5, 1.5), rnd(-1.5, 1.5)];
            let qddi = [rnd(-2.0, 2.0), rnd(-2.0, 2.0)];
            let t = inverse_dynamics(&robot, &inertia, &qi, &qdi, &qddi, g);
            q.push(qi);
            qd.push(qdi);
            qdd.push(qddi);
            tau.push([t[0], t[1]]);
        }
        DelanLab { robot, inertia, delan: Delan::new(20, 5), q, qd, qdd, tau, cfg: [0.3, 0.7], epochs: 0 }
    }

    /// Train the DeLaN for `n` epochs; returns the training loss.
    pub fn train(&mut self, n: u32) -> f64 {
        let loss = self.delan.train(&self.q, &self.qd, &self.qdd, &self.tau, n as usize, 3e-3);
        self.epochs += n;
        loss
    }

    pub fn epochs(&self) -> u32 {
        self.epochs
    }

    /// Set the configuration to inspect.
    pub fn set_config(&mut self, q0: f64, q1: f64) {
        self.cfg = [q0, q1];
    }

    /// Learned mass-matrix entry: 0 → M00, 1 → M01, 2 → M11.
    pub fn m_hat(&self, i: usize) -> f64 {
        self.delan.mass(&self.cfg)[i]
    }
    /// True mass-matrix entry from ferromotion's recursive Newton–Euler.
    pub fn m_true(&self, i: usize) -> f64 {
        let m = mass_matrix(&self.robot, &self.inertia, &self.cfg);
        [m[(0, 0)], m[(0, 1)], m[(1, 1)]][i]
    }
    /// Max absolute error between learned and true M(q).
    pub fn m_error(&self) -> f64 {
        (0..3).map(|i| (self.m_hat(i) - self.m_true(i)).abs()).fold(0.0, f64::max)
    }
    /// Smaller eigenvalue of the learned M(q) — positive iff M is positive-definite.
    pub fn m_min_eig(&self) -> f64 {
        let m = self.delan.mass(&self.cfg);
        let (a, b, d) = (m[0], m[1], m[2]);
        let tr = a + d;
        let disc = ((a - d) * (a - d) + 4.0 * b * b).sqrt();
        0.5 * (tr - disc)
    }

    // planar draw geometry at the inspected config
    pub fn joint1_x(&self) -> f64 { L * self.cfg[0].cos() }
    pub fn joint1_y(&self) -> f64 { L * self.cfg[0].sin() }
    pub fn tip_x(&self) -> f64 { self.joint1_x() + L * (self.cfg[0] + self.cfg[1]).cos() }
    pub fn tip_y(&self) -> f64 { self.joint1_y() + L * (self.cfg[0] + self.cfg[1]).sin() }
}

impl Default for DelanLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_mass_matrix_is_always_spd_and_converges_with_training() {
        let mut lab = DelanLab::new();
        // SPD even before training (Cholesky structure)
        assert!(lab.m_min_eig() > 0.0, "M should be positive-definite pre-training");
        let e_before = lab.m_error();
        lab.train(400);
        let e_after = lab.m_error();
        assert!(lab.m_min_eig() > 0.0, "M stays positive-definite");
        assert!(e_after < e_before, "training should move M toward the truth: {e_before} → {e_after}");
    }
}
