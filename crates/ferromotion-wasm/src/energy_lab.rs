//! **Energy lab** — the rig behind the "why integrators drift" lesson. It runs the *same* frictionless
//! pendulum from the *same* raised start under two time-steppers and tracks the total mechanical energy of
//! each: **explicit (forward) Euler**, which systematically *injects* energy until the swing spirals outward,
//! and **semi-implicit / symplectic Euler**, a structure-preserving integrator whose energy error stays in a
//! tight bounded band forever. A frictionless system must conserve energy; a black-box learner rolled out over
//! a long horizon drifts exactly like explicit Euler, which is why the course later reaches for
//! structure-preserving (variational) integrators — and for architectures (Hamiltonian/Lagrangian nets) that
//! bake conservation in. A clean single pendulum (non-chaotic) makes the contrast unambiguous. Same
//! ferromotion-core dynamics as [`crate::DynamicsLab`]; only the update rule differs.

use ferromotion_core::{forward_dynamics, from_urdf_full, mass_matrix, LinkInertia, Robot};
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

const URDF: &str = r#"<robot name="pend">
  <link name="base"/>
  <link name="l1"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.083"/></inertial></link>
  <link name="tool"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-100" upper="100" effort="100" velocity="1000"/></joint>
  <joint name="jt" type="fixed"><parent link="l1"/><child link="tool"/><origin xyz="1 0 0" rpy="0 0 0"/></joint>
</robot>"#;

const G: f64 = 9.81;
const L: f64 = 1.0;
const MASS: f64 = 1.0;
const Q0: f64 = 1.0; // raised start angle (rad from horizontal)
const HIST: usize = 320;

#[wasm_bindgen]
pub struct EnergyLab {
    robot: Robot,
    inertia: Vec<LinkInertia>,
    qe: f64,
    qde: f64, // explicit-Euler state
    qs: f64,
    qds: f64, // symplectic-Euler state
    t: f64,
    e0: f64,
    hist_e: Vec<f64>,
    hist_s: Vec<f64>,
}

fn accel(robot: &Robot, inertia: &[LinkInertia], q: f64, qd: f64) -> f64 {
    forward_dynamics(robot, inertia, &[q], &[qd], &[0.0], Vector3::new(0.0, 0.0, -G))[0]
}

fn energy(robot: &Robot, inertia: &[LinkInertia], q: f64, qd: f64) -> f64 {
    let m = mass_matrix(robot, inertia, &[q])[(0, 0)];
    let ke = 0.5 * m * qd * qd;
    let z = -0.5 * L * q.sin(); // COM world-z height (up = +z)
    ke + MASS * G * z
}

#[wasm_bindgen]
impl EnergyLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> EnergyLab {
        let (robot, inertia) = from_urdf_full(URDF, "base", "tool").expect("valid URDF");
        let e0 = energy(&robot, &inertia, Q0, 0.0);
        EnergyLab { robot, inertia, qe: Q0, qde: 0.0, qs: Q0, qds: 0.0, t: 0.0, e0, hist_e: vec![e0], hist_s: vec![e0] }
    }

    /// Reset both integrators to the same raised start.
    pub fn reset(&mut self) {
        self.qe = Q0;
        self.qde = 0.0;
        self.qs = Q0;
        self.qds = 0.0;
        self.t = 0.0;
        self.hist_e = vec![self.e0];
        self.hist_s = vec![self.e0];
    }

    /// Advance both integrators one step of size `dt`. Explicit Euler moves position with the OLD velocity;
    /// symplectic Euler updates velocity first, then moves position with the NEW velocity — the only
    /// difference, and the whole story.
    pub fn step(&mut self, dt: f64) {
        // explicit (forward) Euler — injects energy
        let ae = accel(&self.robot, &self.inertia, self.qe, self.qde);
        let qde_old = self.qde;
        self.qde += ae * dt;
        self.qe += qde_old * dt;
        // symplectic (semi-implicit) Euler — bounded energy error
        let as_ = accel(&self.robot, &self.inertia, self.qs, self.qds);
        self.qds += as_ * dt;
        self.qs += self.qds * dt;

        self.t += dt;
        push_capped(&mut self.hist_e, energy(&self.robot, &self.inertia, self.qe, self.qde));
        push_capped(&mut self.hist_s, energy(&self.robot, &self.inertia, self.qs, self.qds));
    }

    pub fn time(&self) -> f64 {
        self.t
    }
    pub fn e0(&self) -> f64 {
        self.e0
    }
    /// Angle of the symplectic pendulum (for drawing the swinging arm).
    pub fn angle_symplectic(&self) -> f64 {
        self.qs
    }
    pub fn angle_explicit(&self) -> f64 {
        self.qe
    }
    /// Current total energy under explicit Euler (finite even if large).
    pub fn energy_explicit(&self) -> f64 {
        let e = energy(&self.robot, &self.inertia, self.qe, self.qde);
        if e.is_finite() { e } else { 1e9 }
    }
    /// Current total energy under symplectic Euler.
    pub fn energy_symplectic(&self) -> f64 {
        energy(&self.robot, &self.inertia, self.qs, self.qds)
    }

    pub fn n_hist(&self) -> usize {
        self.hist_e.len()
    }
    pub fn hist_explicit(&self, i: usize) -> f64 {
        let e = self.hist_e.get(i).copied().unwrap_or(self.e0);
        if e.is_finite() { e } else { 1e9 }
    }
    pub fn hist_symplectic(&self, i: usize) -> f64 {
        self.hist_s.get(i).copied().unwrap_or(self.e0)
    }
}

fn push_capped(v: &mut Vec<f64>, x: f64) {
    v.push(x);
    if v.len() > HIST {
        v.remove(0);
    }
}

impl Default for EnergyLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_euler_gains_energy_while_symplectic_stays_bounded() {
        // THE HEADLINE. From the same frictionless start, over a long horizon explicit Euler drifts far from
        // the initial energy while symplectic Euler stays in a tight band — the definition of a
        // structure-preserving integrator.
        let mut lab = EnergyLab::new();
        let e0 = lab.e0();
        for _ in 0..2000 {
            lab.step(0.02);
        }
        let drift_exp = (lab.energy_explicit() - e0).abs();
        let drift_sym = (lab.energy_symplectic() - e0).abs();
        assert!(drift_exp > 1.0, "explicit Euler should gain energy: drift {drift_exp}");
        assert!(drift_sym < 0.1, "symplectic energy should stay in a tight band: drift {drift_sym}");
    }

    #[test]
    fn both_integrators_start_at_the_same_energy() {
        let lab = EnergyLab::new();
        assert!((lab.energy_explicit() - lab.e0()).abs() < 1e-12);
        assert!((lab.energy_symplectic() - lab.e0()).abs() < 1e-12);
    }
}
