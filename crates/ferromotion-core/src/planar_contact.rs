//! Full-Dojo integration, milestone 1: a **planar rigid body with frictional multi-point contact**.
//! Config `(x, z, θ)`, generalized velocity `(vx, vz, ω)`, mass matrix `diag(m, m, I)`. Each body-frame
//! contact point contributes a normal + tangent row to the contact Jacobian (encoding the rotational
//! coupling `v_point = v + ω × r`), and one step = free integrate → frictional contact solve
//! ([`crate::solve_contacts_friction`], the SOC Coulomb model) → symplectic pose update. This assembles
//! the contact + friction + integrator pieces into a working rigid-body-with-contact simulator (the
//! fully-implicit interior-point coupling is the next Dojo refinement). Pure Rust → WASM-clean.

use crate::{solve_contacts_friction, FrictionContact};
use nalgebra::{DMatrix, DVector};

/// A planar rigid body (in the vertical x–z plane) with body-frame contact points and a floor at z = 0.
#[derive(Clone, Debug)]
pub struct PlanarBody {
    pub x: f64,
    pub z: f64,
    pub theta: f64,
    pub vx: f64,
    pub vz: f64,
    pub omega: f64,
    pub mass: f64,
    pub inertia: f64,
    /// Contact points in the body frame.
    pub contacts_body: Vec<[f64; 2]>,
    pub mu: f64,
}

impl PlanarBody {
    /// World position of body-frame point `p`.
    fn to_world(&self, p: [f64; 2]) -> [f64; 2] {
        let (c, s) = (self.theta.cos(), self.theta.sin());
        [self.x + c * p[0] - s * p[1], self.z + s * p[0] + c * p[1]]
    }

    /// Smallest floor gap over all contact points (negative ⇒ penetration).
    pub fn min_gap(&self) -> f64 {
        self.contacts_body.iter().map(|&p| self.to_world(p)[1]).fold(f64::INFINITY, f64::min)
    }

    /// Kinetic energy (for settling checks).
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * (self.vx * self.vx + self.vz * self.vz) + 0.5 * self.inertia * self.omega * self.omega
    }

    /// Advance one step under gravity `g` (downward). Resolves all frictional contacts jointly.
    pub fn step(&mut self, dt: f64, g: f64) {
        let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[self.mass, self.mass, self.inertia]));
        // Free generalized velocity after gravity (acts at the COM → no torque).
        let v_free = DVector::from_row_slice(&[self.vx, self.vz - g * dt, self.omega]);

        let (c, s) = (self.theta.cos(), self.theta.sin());
        let contacts: Vec<FrictionContact> = self
            .contacts_body
            .iter()
            .map(|&p| {
                let rx = c * p[0] - s * p[1]; // COM→point vector (world)
                let rz = s * p[0] + c * p[1];
                FrictionContact {
                    // point velocity: vz + ω·rx (normal, +z); vx − ω·rz (tangent, +x).
                    jn: DVector::from_row_slice(&[0.0, 1.0, rx]),
                    jt: vec![DVector::from_row_slice(&[1.0, 0.0, -rz])],
                    phi: self.z + rz, // world height of the contact point
                    mu: self.mu,
                }
            })
            .collect();

        let sol = solve_contacts_friction(&m, &v_free, &contacts, dt);
        self.vx = sol.v_next[0];
        self.vz = sol.v_next[1];
        self.omega = sol.v_next[2];
        self.x += self.vx * dt;
        self.z += self.vz * dt;
        self.theta += self.omega * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box of full width 2w, full height 2h, centered at the COM (unit mass).
    fn box_body(w: f64, h: f64, x: f64, z: f64, theta: f64) -> PlanarBody {
        let mass = 1.0;
        PlanarBody {
            x,
            z,
            theta,
            vx: 0.0,
            vz: 0.0,
            omega: 0.0,
            mass,
            inertia: mass * ((2.0 * w).powi(2) + (2.0 * h).powi(2)) / 12.0,
            contacts_body: vec![[-w, -h], [w, -h], [w, h], [-w, h]],
            mu: 0.6,
        }
    }

    #[test]
    fn flat_box_drops_and_rests_on_the_floor() {
        let mut b = box_body(0.5, 0.25, 0.0, 1.0, 0.0);
        let (dt, g) = (0.005, 9.81);
        let mut worst_pen: f64 = 0.0;
        for _ in 0..800 {
            b.step(dt, g);
            worst_pen = worst_pen.max(-b.min_gap());
        }
        assert!(worst_pen < 3e-3, "penetrated the floor by {worst_pen}");
        assert!(b.kinetic_energy() < 1e-3, "did not come to rest, KE = {}", b.kinetic_energy());
        assert!((b.z - 0.25).abs() < 5e-3, "did not settle at half-height: z = {}", b.z);
        assert!(b.theta.abs() < 1e-2, "should stay flat: θ = {}", b.theta);
    }

    #[test]
    fn tilted_box_settles_without_penetrating() {
        // Dropped tilted: it lands on a corner, the contact torque rights it, and it dissipates to rest.
        let mut b = box_body(0.5, 0.25, 0.0, 1.0, 0.25);
        let theta0 = b.theta;
        let (dt, g) = (0.004, 9.81);
        let mut worst_pen: f64 = 0.0;
        for _ in 0..1500 {
            b.step(dt, g);
            worst_pen = worst_pen.max(-b.min_gap());
        }
        assert!(worst_pen < 5e-3, "penetrated the floor by {worst_pen}");
        assert!(b.kinetic_energy() < 5e-3, "did not settle, KE = {}", b.kinetic_energy());
        assert!(b.theta.abs() < theta0 + 1e-2, "should not tip further than it started: θ = {}", b.theta);
        assert!(b.theta.abs() < std::f64::consts::FRAC_PI_2, "tipped over: θ = {}", b.theta);
    }
}
