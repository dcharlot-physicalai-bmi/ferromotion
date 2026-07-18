//! **XPBD — Extended Position-Based Dynamics** (Macklin, Müller & Chentanez, MIG 2016; "small-steps"
//! Macklin et al. 2019): a substepped constraint-projection integrator that adds **compliance** (an
//! inverse stiffness) to classic PBD. Compliance decouples a constraint's stiffness from the iteration
//! count and time step — the same material stiffness gives the *same* equilibrium regardless of how many
//! substeps you take, which plain PBD famously cannot do. One framework then covers rigid, cloth, soft,
//! and rod constraints; it is cheap and branch-light, ideal for the browser/WASM labs where a full
//! IPC/FEM solve is too heavy.
//!
//! Here: point masses with distance constraints. Per substep, positions are predicted under gravity, each
//! constraint contributes `Δλ = (−C − α̃λ)/(w + α̃)` with `α̃ = compliance/h²`, and velocities are read back
//! from the position change. Verified: the compliant hang matches the analytic Hookean elongation
//! `mg·compliance`, that equilibrium is **step-count-independent** (the "extended" property), the stiff
//! limit is rigid, and a free bar conserves momentum. Pure `nalgebra` → WASM-clean.

use nalgebra::Vector3;

/// A point mass. `inv_mass = 0` pins it (an anchor).
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub x: Vector3<f64>,
    pub v: Vector3<f64>,
    pub inv_mass: f64,
    prev: Vector3<f64>,
}

impl Particle {
    pub fn new(x: Vector3<f64>, inv_mass: f64) -> Self {
        Particle { x, v: Vector3::zeros(), inv_mass, prev: x }
    }
}

/// A distance constraint `‖x_i − x_j‖ = rest` with a given `compliance` (`0` ⇒ rigid; larger ⇒ softer).
#[derive(Clone, Copy, Debug)]
pub struct DistanceConstraint {
    pub i: usize,
    pub j: usize,
    pub rest: f64,
    pub compliance: f64,
}

/// An XPBD solver over point masses and distance constraints.
#[derive(Clone, Debug)]
pub struct XpbdSolver {
    pub particles: Vec<Particle>,
    pub constraints: Vec<DistanceConstraint>,
    pub gravity: Vector3<f64>,
    /// Substeps per [`Self::step`] call (more ⇒ stiffer-feeling, but XPBD keeps the *equilibrium* fixed).
    pub substeps: usize,
    /// Constraint-projection iterations per substep (more ⇒ closer to the exact compliant equilibrium).
    pub iters: usize,
    /// Per-substep velocity damping in `[0,1)` (0 = none).
    pub damping: f64,
}

impl XpbdSolver {
    /// Advance one frame of duration `dt` (split into `substeps` XPBD substeps).
    pub fn step(&mut self, dt: f64) {
        let h = dt / self.substeps as f64;
        for _ in 0..self.substeps {
            // predict (external force then integrate — no damping here, so gravity is unbiased)
            for p in &mut self.particles {
                if p.inv_mass > 0.0 {
                    p.v += self.gravity * h;
                }
                p.prev = p.x;
                p.x += p.v * h;
            }
            // solve constraints (compliant projection), λ accumulates across iters and resets each substep
            let mut lambda = vec![0.0; self.constraints.len()];
            for _ in 0..self.iters.max(1) {
                for (ci, c) in self.constraints.iter().enumerate() {
                    let (pi, pj) = (self.particles[c.i], self.particles[c.j]);
                    let w = pi.inv_mass + pj.inv_mass;
                    if w == 0.0 {
                        continue;
                    }
                    let d = pi.x - pj.x;
                    let len = d.norm();
                    if len < 1e-12 {
                        continue;
                    }
                    let n = d / len;
                    let cval = len - c.rest;
                    let a_tilde = c.compliance / (h * h);
                    let dlambda = (-cval - a_tilde * lambda[ci]) / (w + a_tilde);
                    lambda[ci] += dlambda;
                    let corr = n * dlambda;
                    self.particles[c.i].x += corr * pi.inv_mass;
                    self.particles[c.j].x -= corr * pj.inv_mass;
                }
            }
            // read velocities back from the position change, then damp (removes settling energy only)
            for p in &mut self.particles {
                p.v = (p.x - p.prev) / h;
                p.v *= 1.0 - self.damping;
            }
        }
    }

    /// Run `frames` frames of duration `dt` (convenience for settling to equilibrium).
    pub fn simulate(&mut self, dt: f64, frames: usize) {
        for _ in 0..frames {
            self.step(dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    // A mass hung from a fixed anchor by one compliant distance constraint; settle under gravity.
    fn hang(compliance: f64, substeps: usize, rest: f64) -> XpbdSolver {
        XpbdSolver {
            particles: vec![Particle::new(v(0.0, 0.0, 0.0), 0.0), Particle::new(v(0.0, -rest, 0.0), 1.0)],
            constraints: vec![DistanceConstraint { i: 0, j: 1, rest, compliance }],
            gravity: v(0.0, -9.81, 0.0),
            substeps,
            iters: 20,
            damping: 0.05,
        }
    }

    #[test]
    fn the_compliant_hang_matches_the_hookean_elongation() {
        // At rest, a spring of stiffness k = 1/compliance stretches by ΔL = m·g / k = m·g·compliance.
        let compliance = 0.01;
        let rest = 1.0;
        let mut s = hang(compliance, 4, rest);
        s.simulate(0.01, 4000);
        let elong = -s.particles[1].x.y - rest; // hangs below the anchor
        let expect = 1.0 * 9.81 * compliance; // m·g·compliance
        assert!((elong - expect).abs() < 1e-3, "elongation {elong} vs Hookean {expect}");
    }

    #[test]
    fn the_equilibrium_is_step_count_independent() {
        // THE "EXTENDED" PROPERTY. Plain PBD's stiffness depends on substep count; XPBD's compliance does
        // not — the settled elongation is the same at 1 substep and at 16.
        let (c, rest) = (0.01, 1.0);
        let mut a = hang(c, 1, rest);
        let mut b = hang(c, 16, rest);
        a.simulate(0.01, 4000);
        b.simulate(0.01, 4000);
        let ea = -a.particles[1].x.y - rest;
        let eb = -b.particles[1].x.y - rest;
        assert!((ea - eb).abs() < 1e-3, "equilibrium depends on substeps: {ea} (n=1) vs {eb} (n=16)");
    }

    #[test]
    fn the_stiff_limit_is_rigid() {
        // compliance → 0 ⇒ a rigid rod: essentially no stretch under load.
        let rest = 1.0;
        let mut s = hang(1e-7, 8, rest);
        s.simulate(0.01, 3000);
        let elong = (-s.particles[1].x.y - rest).abs();
        assert!(elong < 1e-3, "a stiff constraint should barely stretch: {elong}");
    }

    #[test]
    fn a_free_bar_conserves_linear_momentum() {
        // Two masses joined by a stretched rod, no gravity, given equal-and-opposite kicks: total momentum
        // stays zero as the rod oscillates (internal constraint forces cancel).
        let mut s = XpbdSolver {
            particles: vec![
                { let mut p = Particle::new(v(-0.6, 0.0, 0.0), 1.0); p.v = v(-0.5, 0.2, 0.0); p },
                { let mut p = Particle::new(v(0.6, 0.0, 0.0), 1.0); p.v = v(0.5, -0.2, 0.0); p },
            ],
            constraints: vec![DistanceConstraint { i: 0, j: 1, rest: 1.0, compliance: 0.001 }],
            gravity: v(0.0, 0.0, 0.0),
            substeps: 4,
            iters: 4,
            damping: 0.0,
        };
        let p0: Vector3<f64> = s.particles.iter().map(|p| p.v / p.inv_mass).sum();
        s.simulate(0.005, 200);
        let p1: Vector3<f64> = s.particles.iter().map(|p| p.v / p.inv_mass).sum();
        assert!((p1 - p0).norm() < 1e-9, "momentum drifted: {p0:?} → {p1:?}");
    }
}
