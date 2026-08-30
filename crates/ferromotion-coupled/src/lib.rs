//! ferromotion-coupled — **two-way multiphysics coupling** between a volumetric soft body
//! (ferromotion-fem) and granular media (ferromotion-dem). This is the SoftMAC-class capability —
//! solvers that were each verified in isolation now interact through a shared contact so grains pile
//! *on* a deforming soft body, a soft body sinks *into* grains, and each pushes back on the other.
//!
//! The coupling is a symmetric penalty contact: a FEM surface vertex is treated as a small sphere,
//! and every FEM-vertex ↔ grain overlap applies an **equal-and-opposite** spring–dashpot force to
//! both bodies (Newton's third law). The integrator owns gravity and the floor uniformly and steps
//! both bodies together. Because every internal and contact force is equal-and-opposite, the coupled
//! system conserves total linear momentum exactly — the oracle a correct two-way coupling must pass.
//! Pure `nalgebra` → WASM-clean.

use ferromotion_dem::DemSim;
use ferromotion_fem::FemSim;
use nalgebra::Vector3;

mod grasp;
pub use grasp::GraspFemSim;

/// A coupled soft-body + granular simulation over an optional shared floor.
pub struct CoupledFemDem {
    pub fem: FemSim,
    pub dem: DemSim,
    /// Effective contact radius of a FEM vertex (as a sphere) against a grain.
    pub vert_r: f64,
    pub k_couple: f64,
    pub gravity: Vector3<f64>,
    pub floor: Option<f64>,
}

impl CoupledFemDem {
    pub fn new(mut fem: FemSim, mut dem: DemSim, vert_r: f64, k_couple: f64) -> Self {
        // the coupled integrator owns gravity + floor; silence the sub-sims' own copies
        fem.gravity = Vector3::zeros();
        fem.floor = None;
        dem.gravity = Vector3::zeros();
        dem.floor_z = f64::NEG_INFINITY;
        CoupledFemDem { fem, dem, vert_r, k_couple, gravity: Vector3::new(0.0, 0.0, -9.81), floor: None }
    }

    /// One semi-implicit step of the coupled system.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let dt = self.fem.dt;
        let nv = self.fem.n_verts();
        let ng = self.dem.grains.len();

        // internal forces
        let mut ff = self.fem.forces(); // elastic
        let mut gf = self.dem.pair_forces(); // grain–grain

        // gravity (uniform)
        for i in 0..nv {
            ff[i] += self.fem.mass * self.gravity;
        }
        for i in 0..ng {
            gf[i] += self.dem.grains[i].m * self.gravity;
        }

        // shared floor penalty
        if let Some(fz) = self.floor {
            let gamma = 0.7 * (self.dem.kn * self.dem.grains.first().map(|g| g.m).unwrap_or(1.0)).sqrt();
            for i in 0..nv {
                let pen = fz + self.vert_r - self.fem.x[i].z;
                if pen > 0.0 {
                    let vn = self.fem.v[i].z.min(0.0);
                    ff[i].z += self.dem.kn * pen - gamma * vn;
                }
            }
            for i in 0..ng {
                let pen = fz + self.dem.grains[i].r - self.dem.grains[i].x.z;
                if pen > 0.0 {
                    let vn = self.dem.grains[i].v.z.min(0.0);
                    gf[i].z += self.dem.kn * pen - gamma * vn;
                }
            }
        }

        // ---- cross-contact: FEM vertex (sphere vert_r) ↔ grain (sphere r) ----
        for a in 0..nv {
            let xv = self.fem.x[a];
            let vv = self.fem.v[a];
            for b in 0..ng {
                let d = xv - self.dem.grains[b].x;
                let dist = d.norm();
                let overlap = self.vert_r + self.dem.grains[b].r - dist;
                if overlap > 0.0 && dist > 1e-12 {
                    let nrm = d / dist; // into the vertex
                    let vrel = vv - self.dem.grains[b].v;
                    let vn = vrel.dot(&nrm);
                    let gamma = 0.7 * (self.k_couple * self.fem.mass).sqrt();
                    let fmag = (self.k_couple * overlap - gamma * vn.min(0.0)).max(0.0);
                    let fc = fmag * nrm;
                    ff[a] += fc;
                    gf[b] -= fc; // equal and opposite
                }
            }
        }

        // ---- integrate (semi-implicit Euler), pins held ----
        let inv_m = 1.0 / self.fem.mass;
        let fdamp = self.fem.damping_rate;
        for i in 0..nv {
            if self.fem.pinned[i] {
                self.fem.v[i] = Vector3::zeros();
                continue;
            }
            self.fem.v[i] = (self.fem.v[i] + dt * ff[i] * inv_m) / (1.0 + fdamp * dt);
            self.fem.x[i] += dt * self.fem.v[i];
        }
        for i in 0..ng {
            let m = self.dem.grains[i].m;
            self.dem.grains[i].v += dt * gf[i] / m;
            let v = self.dem.grains[i].v;
            self.dem.grains[i].x += dt * v;
        }
    }

    /// Total linear momentum of the coupled system.
    pub fn momentum(&self) -> Vector3<f64> {
        let pf: Vector3<f64> = self.fem.v.iter().map(|v| self.fem.mass * v).sum();
        let pd: Vector3<f64> = self.dem.grains.iter().map(|g| g.m * g.v).sum();
        pf + pd
    }

    pub fn kinetic_energy(&self) -> f64 {
        let kf: f64 = self.fem.v.iter().map(|v| 0.5 * self.fem.mass * v.norm_squared()).sum();
        let kd: f64 = self.dem.grains.iter().map(|g| 0.5 * g.m * g.v.norm_squared()).sum();
        kf + kd
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use ferromotion_dem::Grain;

    fn cube_and_grains() -> (FemSim, DemSim) {
        let fem = FemSim::box_grid(2, 2, 2, 0.25, 0.4, 3.0e3, 1.5e3, 2e-4);
        let grains: Vec<Grain> = (0..8)
            .map(|k| Grain {
                x: Vector3::new(1.2 + (k % 2) as f64 * 0.16, (k / 2) as f64 * 0.16 - 0.1, 0.2 + (k % 3) as f64 * 0.14),
                v: Vector3::zeros(),
                r: 0.09,
                m: 0.3,
            })
            .collect();
        let dem = DemSim::new(grains, 4.0e4, 60.0, 0.4, 2e-4);
        (fem, dem)
    }

    /// **The two-way-coupling oracle.** In free space (no gravity, no floor) a soft body and a
    /// cluster of grains collide; because every internal and cross-contact force is equal-and-
    /// opposite, total linear momentum is conserved to machine precision.
    #[test]
    fn coupled_collision_conserves_momentum() {
        let (fem, dem) = cube_and_grains();
        let mut sim = CoupledFemDem::new(fem, dem, 0.06, 3.0e4);
        sim.gravity = Vector3::zeros();
        sim.floor = None;
        // send the soft body and the grains toward each other
        for v in sim.fem.v.iter_mut() {
            *v = Vector3::new(1.5, 0.0, 0.0);
        }
        for g in sim.dem.grains.iter_mut() {
            g.v = Vector3::new(-1.0, 0.0, 0.0);
        }
        let p0 = sim.momentum();
        for _ in 0..3000 {
            sim.step();
        }
        let drift = (sim.momentum() - p0).norm();
        eprintln!("coupled FEM↔DEM collision: |Δp| {drift:.2e} over 3000 steps");
        assert!(drift < 1e-9, "two-way coupling did not conserve momentum: {drift}");
    }

    /// Grains poured onto a soft body on the floor settle: the coupled system comes to rest, grains
    /// stay above the soft body / floor, nothing tunnels or explodes.
    #[test]
    fn grains_settle_on_soft_body() {
        let fem = FemSim::box_grid(3, 3, 1, 0.22, 0.5, 4.0e3, 2.0e3, 2e-4);
        // grains dropped just above the slab
        let grains: Vec<Grain> = (0..9)
            .map(|k| Grain {
                x: Vector3::new(0.1 + (k % 3) as f64 * 0.22, 0.1 + (k / 3) as f64 * 0.22, 0.55),
                v: Vector3::zeros(),
                r: 0.08,
                m: 0.25,
            })
            .collect();
        let dem = DemSim::new(grains, 4.0e4, 80.0, 0.5, 2e-4);
        let mut sim = CoupledFemDem::new(fem, dem, 0.08, 4.0e4);
        sim.floor = Some(0.0);
        sim.fem.damping_rate = 154.639_175_257_732; // old per-step 0.03 at dt = 2e-4; the slab dissipates its own elastic vibration
        // pin the slab's base so it acts as a compliant mat
        let zmin = sim.fem.x.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
        for i in 0..sim.fem.n_verts() {
            if sim.fem.x[i].z < zmin + 1e-6 {
                sim.fem.pinned[i] = true;
            }
        }
        for _ in 0..300 {
            sim.step();
        }
        let ke_mid = sim.kinetic_energy();
        for _ in 0..12000 {
            sim.step();
        }
        let ke_end = sim.kinetic_energy();
        let lowest_grain = sim.dem.grains.iter().map(|g| g.x.z - g.r).fold(f64::INFINITY, f64::min);
        let highest = sim.dem.grains.iter().map(|g| g.x.z).fold(f64::NEG_INFINITY, f64::max);
        eprintln!("grains on soft body: KE {ke_mid:.3e} → {ke_end:.3e}, lowest grain gap {lowest_grain:.3}, top {highest:.2}");
        assert!(ke_end < 0.05 * ke_mid.max(1e-9) + 1e-3, "coupled system did not settle: KE {ke_end}");
        assert!(lowest_grain > -0.05, "a grain tunneled below the floor: {lowest_grain}");
        assert!(highest < 2.0, "the coupled system exploded: top {highest}");
    }
}
