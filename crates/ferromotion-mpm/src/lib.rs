//! ferromotion-mpm — a **differentiable 2D Material Point Method** (MLS-MPM) for soft, elastic, and
//! granular material — the solid-mechanics companion to the Aquarium fluid solver, in the spirit of
//! DiffTaichi / Genesis.
//!
//! Particles carry mass, velocity, an affine velocity field `C` (APIC), and a deformation gradient
//! `F`; a background grid mediates the forces. Each step is the MLS-MPM transfer: **P2G** scatters
//! momentum and the internal-stress affine to the grid via quadratic B-splines, the grid velocity is
//! updated (gravity, walls), and **G2P** gathers it back, advecting the particles and evolving `F`.
//! The constitutive model is neo-Hookean, whose Kirchhoff stress `τ = μ(FFᵀ − I) + λ ln(J) I` is
//! smooth in `F` — so the whole pipeline is differentiable.
//!
//! Because `μ, λ ∝ E`, the stress is **linear in Young's modulus** (`∂τ/∂E = τ/E`), and within a
//! step the B-spline weights are fixed — giving an exact analytic gradient of an outcome w.r.t. the
//! material stiffness, which we check against finite differences to machine precision. Verified
//! physics: momentum conservation, free-fall, and a dropped elastic block that deforms and stays
//! bounded. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix2, Vector2};

pub mod adjoint;
pub use adjoint::{AdjointResult, SoftBody, Tape, Var};

/// A material point.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub x: Vector2<f64>,
    pub v: Vector2<f64>,
    /// APIC affine velocity field.
    pub c: Matrix2<f64>,
    /// Deformation gradient.
    pub f: Matrix2<f64>,
}

impl Particle {
    /// A particle at rest (F = I) at position `x`.
    pub fn at(x: Vector2<f64>) -> Self {
        Self { x, v: Vector2::zeros(), c: Matrix2::zeros(), f: Matrix2::identity() }
    }
}

/// An MLS-MPM simulation on the unit square `[0,1]²`.
#[derive(Clone)]
pub struct MpmSim {
    pub n: usize,
    pub dt: f64,
    pub gravity: Vector2<f64>,
    /// Poisson ratio and Young's modulus.
    pub nu: f64,
    pub e: f64,
    pub mass: f64,
    pub vol: f64,
    /// Enforce box walls (else the domain is open).
    pub walls: bool,
    pub particles: Vec<Particle>,
}

/// One P2G → grid → G2P transfer. Returns the advanced particles, the total post-step kinetic
/// energy `KE`, and its analytic derivative `∂KE/∂E` w.r.t. Young's modulus (weights are fixed within
/// the step, and `τ ∝ E`, so this is exact).
fn transfer(sim: &MpmSim) -> (Vec<Particle>, f64, f64) {
    let n = sim.n;
    let inv_dx = n as f64;
    let dx = 1.0 / inv_dx;
    let nnode = n + 1;
    let idx = |ix: i64, iy: i64| (ix as usize) * nnode + (iy as usize);
    let inrange = |ix: i64, iy: i64| ix >= 0 && ix <= n as i64 && iy >= 0 && iy <= n as i64;

    let mut gm = vec![0.0f64; nnode * nnode];
    let mut gv = vec![Vector2::zeros(); nnode * nnode];
    let mut dgv = vec![Vector2::zeros(); nnode * nnode]; // ∂(grid velocity)/∂E

    let (mu, la) = lame(sim.e, sim.nu);
    let (mu_h, la_h) = (mu / sim.e, la / sim.e); // Lamé per unit E (τ = E·τ̂)
    let eye = Matrix2::identity();
    let scale = -sim.dt * sim.vol * 4.0 * inv_dx * inv_dx;

    // Updated F per particle (F changes only here, in P2G).
    let mut new_f = Vec::with_capacity(sim.particles.len());

    // ---- P2G ----
    for p in &sim.particles {
        let f = (eye + sim.dt * p.c) * p.f;
        new_f.push(f);
        let j = f.determinant();
        let tau_hat = mu_h * (f * f.transpose() - eye) + la_h * j.ln() * eye; // τ/E
        let affine = scale * (sim.e * tau_hat) + sim.mass * p.c;
        let daffine = scale * tau_hat; // ∂affine/∂E

        let (base, w) = kernel(p.x * inv_dx);
        for i in 0..3 {
            for jj in 0..3 {
                let (ix, iy) = (base.0 + i as i64, base.1 + jj as i64);
                if !inrange(ix, iy) {
                    continue;
                }
                let weight = w[i].x * w[jj].y;
                let dpos = (Vector2::new(i as f64, jj as f64) - (p.x * inv_dx - Vector2::new(base.0 as f64, base.1 as f64))) * dx;
                let k = idx(ix, iy);
                gm[k] += weight * sim.mass;
                gv[k] += weight * (sim.mass * p.v + affine * dpos);
                dgv[k] += weight * (daffine * dpos);
            }
        }
    }

    // ---- grid update ----
    for ix in 0..nnode {
        for iy in 0..nnode {
            let k = ix * nnode + iy;
            if gm[k] <= 0.0 {
                continue;
            }
            gv[k] /= gm[k];
            dgv[k] /= gm[k];
            gv[k] += sim.dt * sim.gravity; // gravity has no E-dependence
            if sim.walls {
                let b = 3i64;
                if (ix as i64) < b && gv[k].x < 0.0 {
                    gv[k].x = 0.0;
                    dgv[k].x = 0.0;
                }
                if ix as i64 > n as i64 - b && gv[k].x > 0.0 {
                    gv[k].x = 0.0;
                    dgv[k].x = 0.0;
                }
                if (iy as i64) < b && gv[k].y < 0.0 {
                    gv[k].y = 0.0;
                    dgv[k].y = 0.0;
                }
                if iy as i64 > n as i64 - b && gv[k].y > 0.0 {
                    gv[k].y = 0.0;
                    dgv[k].y = 0.0;
                }
            }
        }
    }

    // ---- G2P ----
    let mut out = Vec::with_capacity(sim.particles.len());
    let (mut ke, mut dke) = (0.0, 0.0);
    for (pi, p) in sim.particles.iter().enumerate() {
        let (base, w) = kernel(p.x * inv_dx);
        let mut new_v = Vector2::zeros();
        let mut new_c = Matrix2::zeros();
        let mut dnew_v = Vector2::zeros();
        for i in 0..3 {
            for jj in 0..3 {
                let (ix, iy) = (base.0 + i as i64, base.1 + jj as i64);
                if !inrange(ix, iy) {
                    continue;
                }
                let weight = w[i].x * w[jj].y;
                let dpos = Vector2::new(i as f64, jj as f64) - (p.x * inv_dx - Vector2::new(base.0 as f64, base.1 as f64));
                let k = idx(ix, iy);
                new_v += weight * gv[k];
                new_c += 4.0 * inv_dx * weight * gv[k] * dpos.transpose();
                dnew_v += weight * dgv[k];
            }
        }
        let x = p.x + sim.dt * new_v;
        out.push(Particle { x, v: new_v, c: new_c, f: new_f[pi] });
        ke += 0.5 * sim.mass * new_v.norm_squared();
        dke += sim.mass * new_v.dot(&dnew_v);
    }
    (out, ke, dke)
}

/// Lamé parameters from Young's modulus and Poisson ratio.
fn lame(e: f64, nu: f64) -> (f64, f64) {
    (e / (2.0 * (1.0 + nu)), e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
}

/// Quadratic B-spline: base node and the three per-axis weights, given grid-space position.
fn kernel(xg: Vector2<f64>) -> ((i64, i64), [Vector2<f64>; 3]) {
    let base = ((xg.x - 0.5).floor(), (xg.y - 0.5).floor());
    let fx = xg - Vector2::new(base.0, base.1);
    let w = [
        0.5 * (Vector2::repeat(1.5) - fx).map(|v| v * v),
        Vector2::repeat(0.75) - (fx - Vector2::repeat(1.0)).map(|v| v * v),
        0.5 * (fx - Vector2::repeat(0.5)).map(|v| v * v),
    ];
    ((base.0 as i64, base.1 as i64), w)
}

impl MpmSim {
    /// Advance one timestep.
    pub fn step(&mut self) {
        self.particles = transfer(self).0;
    }

    /// Total kinetic energy after one step (without mutating), and its analytic `∂/∂E`.
    pub fn ke_and_dke_de(&self) -> (f64, f64) {
        let (_, ke, dke) = transfer(self);
        (ke, dke)
    }

    /// Center of mass.
    pub fn com(&self) -> Vector2<f64> {
        self.particles.iter().map(|p| p.x).sum::<Vector2<f64>>() / self.particles.len() as f64
    }

    /// Total linear momentum.
    pub fn momentum(&self) -> Vector2<f64> {
        self.mass * self.particles.iter().map(|p| p.v).sum::<Vector2<f64>>()
    }

    /// Total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.particles.iter().map(|p| p.v.norm_squared()).sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square block of particles (4 per cell) filling `[x0,x0+s]×[y0,y0+s]`.
    fn block(n: usize, x0: f64, y0: f64, s: f64) -> (Vec<Particle>, f64, f64) {
        let dx = 1.0 / n as f64;
        let step = dx * 0.5;
        let mut ps = Vec::new();
        let mut yy = y0;
        while yy < y0 + s {
            let mut xx = x0;
            while xx < x0 + s {
                ps.push(Particle::at(Vector2::new(xx, yy)));
                xx += step;
            }
            yy += step;
        }
        let vol = step * step;
        (ps, vol, vol) // density 1 → mass = vol
    }

    #[test]
    fn momentum_is_conserved_without_gravity() {
        let (ps, vol, mass) = block(20, 0.4, 0.4, 0.2);
        let mut sim = MpmSim { n: 20, dt: 1e-3, gravity: Vector2::zeros(), nu: 0.2, e: 50.0, mass, vol, walls: false, particles: ps };
        // Give a uniform drift + a stretch so there is internal stress but zero net external force.
        for p in &mut sim.particles {
            p.v = Vector2::new(0.5, -0.3);
            p.f = Matrix2::new(1.2, 0.0, 0.0, 1.0 / 1.2);
        }
        let p0 = sim.momentum();
        for _ in 0..40 {
            sim.step();
        }
        let drift = (sim.momentum() - p0).norm() / p0.norm();
        assert!(drift < 1e-6, "momentum not conserved: relative drift {drift:.2e}");
    }

    #[test]
    fn free_particles_fall_at_g() {
        let (ps, vol, mass) = block(20, 0.4, 0.5, 0.15);
        let g = Vector2::new(0.0, -9.81);
        let mut sim = MpmSim { n: 20, dt: 1e-3, gravity: g, nu: 0.2, e: 50.0, mass, vol, walls: false, particles: ps };
        let steps = 30;
        for _ in 0..steps {
            sim.step();
        }
        // COM velocity ≈ g·(steps·dt); check the mean vertical velocity.
        let vy = sim.particles.iter().map(|p| p.v.y).sum::<f64>() / sim.particles.len() as f64;
        let expected = g.y * steps as f64 * sim.dt;
        assert!((vy - expected).abs() < 1e-6, "free fall vy {vy} ≠ g·t {expected}");
    }

    #[test]
    fn dropped_elastic_block_deforms_and_stays_bounded() {
        let (ps, vol, mass) = block(24, 0.4, 0.55, 0.2);
        let mut sim = MpmSim { n: 24, dt: 5e-4, gravity: Vector2::new(0.0, -9.81), nu: 0.2, e: 80.0, mass, vol, walls: true, particles: ps };
        let y0 = sim.com().y;
        for _ in 0..1200 {
            sim.step();
        }
        // Nothing exploded, nothing tunneled through the floor, and it fell.
        for p in &sim.particles {
            assert!(p.x.iter().all(|c| c.is_finite()), "particle blew up: {:?}", p.x);
            assert!(p.x.y > -0.02 && p.x.x > -0.02 && p.x.x < 1.02, "left the box: {:?}", p.x);
        }
        assert!(sim.com().y < y0, "block did not fall");
        assert!(sim.kinetic_energy() < 10.0, "energy blew up: KE = {}", sim.kinetic_energy());
    }

    #[test]
    fn material_gradient_matches_finite_difference() {
        // Analytic ∂KE/∂E of one step vs central FD — the differentiable-MPM check. A pre-stretched,
        // moving block makes the stress (and hence the E-sensitivity) substantial.
        let (ps, vol, mass) = block(20, 0.4, 0.4, 0.2);
        let mut base = MpmSim { n: 20, dt: 1e-3, gravity: Vector2::zeros(), nu: 0.2, e: 50.0, mass, vol, walls: false, particles: ps };
        for (k, p) in base.particles.iter_mut().enumerate() {
            p.f = Matrix2::new(1.25, 0.05, 0.0, 0.85);
            p.v = Vector2::new(0.2 * ((k % 3) as f64 - 1.0), -0.1);
        }
        let (_, dke) = base.ke_and_dke_de();

        let eps = 1e-3;
        let ke_at = |e: f64| {
            let mut s = base.clone();
            s.e = e;
            s.ke_and_dke_de().0
        };
        let fd = (ke_at(base.e + eps) - ke_at(base.e - eps)) / (2.0 * eps);
        let rel = (dke - fd).abs() / fd.abs().max(1e-12);
        eprintln!("MPM ∂KE/∂E: analytic={dke:.6e}, fd={fd:.6e}, rel_err={rel:.2e}");
        assert!(dke.abs() > 1e-6, "gradient trivially zero — test not exercising stress");
        assert!(rel < 1e-4, "material gradient wrong: analytic {dke} vs fd {fd}");
    }
}
