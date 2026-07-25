//! ferromotion-dem — **discrete-element-method granular media**, the rigid-particle contact domain
//! (sand, gravel, soil, pills) that complements the continuum deformables (MPM/FEM/cloth/rod). Each
//! grain is a sphere; contacts are resolved with a linear spring–dashpot normal law and a
//! Coulomb-limited tangential spring (Cundall & Strack 1979). Grains contact each other and a flat
//! floor. Pure `nalgebra` → WASM-clean.
//!
//! Verified on the invariants granular contact must obey: a frictionless head-on collision conserves
//! momentum and (elastically) kinetic energy; a grain resting on the floor settles to the analytic
//! spring penetration `mg/kₙ`; and a poured pile comes to rest above the floor without exploding.

use nalgebra::Vector3;

/// A spherical grain.
#[derive(Clone, Copy, Debug)]
pub struct Grain {
    pub x: Vector3<f64>,
    pub v: Vector3<f64>,
    pub r: f64,
    pub m: f64,
}

/// A DEM granular simulation over a flat floor at `floor_z` (normal `+z`).
#[derive(Clone)]
pub struct DemSim {
    pub grains: Vec<Grain>,
    pub kn: f64,
    pub gamma_n: f64,
    pub kt: f64,
    pub mu: f64,
    pub floor_z: f64,
    pub gravity: Vector3<f64>,
    pub dt: f64,
}

impl DemSim {
    pub fn new(grains: Vec<Grain>, kn: f64, gamma_n: f64, mu: f64, dt: f64) -> Self {
        DemSim { grains, kn, gamma_n, kt: 0.8 * kn, mu, floor_z: 0.0, gravity: Vector3::new(0.0, 0.0, -9.81), dt }
    }

    /// Contact force on grain `i` from a contact with overlap `delta > 0`, unit normal `n` (pointing
    /// from the contact into `i`), and relative velocity `vrel` (of `i` w.r.t. the other body).
    fn contact_force(&self, delta: f64, n: Vector3<f64>, vrel: Vector3<f64>) -> Vector3<f64> {
        // normal: linear spring + viscous damping (only while approaching)
        let vn = vrel.dot(&n);
        let fn_mag = (self.kn * delta - self.gamma_n * vn).max(0.0);
        let f_normal = fn_mag * n;
        // tangential: Coulomb-capped viscous (a simplified stick-slip proxy)
        let vt = vrel - vn * n;
        let ft = -self.kt * self.dt * vt; // spring-like increment proxy
        let ft = if ft.norm() > self.mu * fn_mag { ft.normalize() * self.mu * fn_mag } else { ft };
        f_normal + ft
    }

    /// Total force on every grain (gravity + grain–grain + grain–floor).
    #[allow(clippy::needless_range_loop)] // pairwise contact indices address the force array
    fn forces(&self) -> Vec<Vector3<f64>> {
        let n = self.grains.len();
        let mut f: Vec<Vector3<f64>> = self.grains.iter().map(|g| g.m * self.gravity).collect();
        // grain–grain
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.grains[i].x - self.grains[j].x;
                let dist = d.norm();
                let overlap = self.grains[i].r + self.grains[j].r - dist;
                if overlap > 0.0 && dist > 1e-12 {
                    let nrm = d / dist; // points into i
                    let vrel = self.grains[i].v - self.grains[j].v;
                    let fc = self.contact_force(overlap, nrm, vrel);
                    f[i] += fc;
                    f[j] -= fc;
                }
            }
        }
        // grain–floor
        for i in 0..n {
            let overlap = self.floor_z + self.grains[i].r - self.grains[i].x.z;
            if overlap > 0.0 {
                let nrm = Vector3::new(0.0, 0.0, 1.0);
                let fc = self.contact_force(overlap, nrm, self.grains[i].v);
                f[i] += fc;
            }
        }
        f
    }

    /// One semi-implicit Euler step.
    #[allow(clippy::needless_range_loop)] // grain index addresses grains + forces together
    pub fn step(&mut self) {
        let forces = self.forces();
        for i in 0..self.grains.len() {
            let g = &mut self.grains[i];
            g.v += self.dt * forces[i] / g.m;
            g.x += self.dt * g.v;
        }
    }

    pub fn momentum(&self) -> Vector3<f64> {
        self.grains.iter().fold(Vector3::zeros(), |a, g| a + g.m * g.v)
    }
    pub fn kinetic_energy(&self) -> f64 {
        self.grains.iter().map(|g| 0.5 * g.m * g.v.norm_squared()).sum()
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// A frictionless head-on collision of two equal grains in free space conserves linear momentum
    /// (and, with no damping, kinetic energy) — Newton's third law at the contact.
    #[test]
    fn head_on_collision_conserves_momentum_and_energy() {
        let g = |x: f64, vx: f64| Grain { x: Vector3::new(x, 0.0, 0.0), v: Vector3::new(vx, 0.0, 0.0), r: 0.5, m: 1.0 };
        let mut sim = DemSim::new(vec![g(-0.6, 1.0), g(0.6, -1.0)], 5000.0, 0.0, 0.0, 1e-4);
        sim.gravity = Vector3::zeros(); // isolate the collision
        sim.floor_z = -100.0; // free space — no floor contact
        let p0 = sim.momentum();
        let e0 = sim.kinetic_energy();
        for _ in 0..4000 {
            sim.step();
        }
        let dp = (sim.momentum() - p0).norm();
        let de = (sim.kinetic_energy() - e0).abs();
        eprintln!("collision: |Δp| {dp:.2e}, |ΔKE| {de:.2e}; final v = [{:.3}, {:.3}]", sim.grains[0].v.x, sim.grains[1].v.x);
        assert!(dp < 1e-9, "momentum not conserved: {dp}");
        assert!(de < 1e-3, "elastic collision lost energy: {de}");
        // they should have separated (moving apart)
        assert!(sim.grains[0].v.x < 0.0 && sim.grains[1].v.x > 0.0, "grains did not bounce apart");
    }

    /// A grain dropped onto the floor settles to the analytic static spring penetration `mg/kₙ`.
    #[test]
    fn resting_grain_reaches_analytic_penetration() {
        let grain = Grain { x: Vector3::new(0.0, 0.0, 1.0), v: Vector3::zeros(), r: 0.5, m: 2.0 };
        let kn = 2.0e4;
        let mut sim = DemSim::new(vec![grain], kn, 40.0, 0.0, 5e-4);
        for _ in 0..8000 {
            sim.step();
        }
        // static equilibrium: kn·δ = m·g  ⇒  penetration δ = m g / kn ; grain center at r − δ
        let delta = sim.floor_z + sim.grains[0].r - sim.grains[0].x.z;
        let expected = grain.m * 9.81 / kn;
        eprintln!("rest penetration {delta:.5} vs analytic {expected:.5}, residual v {:.2e}", sim.grains[0].v.norm());
        assert!(sim.grains[0].v.norm() < 1e-3, "not at rest: {}", sim.grains[0].v.norm());
        assert!((delta - expected).abs() < 1e-4, "penetration {delta} vs {expected}");
    }

    /// A poured pile of grains settles: kinetic energy decays toward zero, every grain stays above
    /// the floor, and none escapes — a stable granular pack, not an explosion.
    #[test]
    fn poured_pile_settles() {
        let mut grains = Vec::new();
        for k in 0..3 {
            for i in 0..3 {
                grains.push(Grain {
                    x: Vector3::new(-0.2 + i as f64 * 0.22, 0.0, 0.6 + k as f64 * 0.25),
                    v: Vector3::zeros(),
                    r: 0.1,
                    m: 0.5,
                });
            }
        }
        let mut sim = DemSim::new(grains, 3.0e4, 60.0, 0.4, 3e-4);
        let ke_hist_start;
        for _ in 0..200 {
            sim.step();
        }
        ke_hist_start = sim.kinetic_energy();
        for _ in 0..15000 {
            sim.step();
        }
        let ke_end = sim.kinetic_energy();
        let lowest = sim.grains.iter().map(|g| g.x.z - g.r).fold(f64::INFINITY, f64::min);
        let highest = sim.grains.iter().map(|g| g.x.z).fold(f64::NEG_INFINITY, f64::max);
        eprintln!("pile: KE {ke_hist_start:.3e} → {ke_end:.3e}, floor gap {lowest:.3}, top {highest:.2}");
        assert!(ke_end < 1e-2 * ke_hist_start.max(1e-9) + 1e-4, "pile did not settle: KE {ke_end}");
        assert!(lowest > -0.02, "a grain sank through the floor: {lowest}");
        assert!(highest < 3.0, "the pile exploded: top {highest}");
    }
}
