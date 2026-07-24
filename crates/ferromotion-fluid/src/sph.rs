//! **Weakly-compressible SPH** (Honest Fluids — stage 5, the Lagrangian paradigm). Where the MAC
//! and lattice solvers are Eulerian (fields on a fixed grid), this is DualSPHysics-class smoothed
//! particle hydrodynamics: the fluid IS the particles, so free surfaces and violent flows (dam
//! breaks, splashes) come for free. Same house discipline — verified against what SPH must obey,
//! not asserted:
//!
//! - **Exact momentum conservation.** The pairwise pressure/viscosity forces are antisymmetric
//!   (`f_ij = −f_ji`), so a free blob conserves total linear momentum to machine precision.
//! - **Kernel partition of unity.** On a uniform lattice the summation density recovers ρ₀ — the
//!   density estimator and cubic-spline kernel are self-consistent.
//! - **Hydrostatic equilibrium.** A settled column reproduces the hydrostatic pressure profile
//!   `p(z) = ρ₀ g (H − z)` (fit slope = −ρ₀g), the physics oracle.
//!
//! Cubic-spline kernel (Monaghan), Tait equation of state, Monaghan artificial viscosity, symplectic
//! Euler. O(N²) neighbor search — the verification scale is a few hundred particles, so no spatial
//! hash is needed; the physics, not the throughput, is the point here.

use std::f64::consts::PI;

/// A 2-D weakly-compressible SPH fluid. The first `n_fluid` particles are fluid (integrated); the
/// rest are frozen dynamic-boundary particles (DBC) that carry density/pressure to repel the fluid.
pub struct Sph {
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub rho: Vec<f64>,
    pub n_fluid: usize,
    pub mass: f64,
    pub h: f64,
    pub rho0: f64,
    pub c0: f64,
    pub gamma: f64,
    pub gravity: f64,
    pub alpha: f64, // artificial-viscosity coefficient
}

impl Sph {
    /// Cubic-spline kernel value at separation `r` (2-D normalization `10/(7πh²)`).
    fn w(&self, r: f64) -> f64 {
        let ad = 10.0 / (7.0 * PI * self.h * self.h);
        let q = r / self.h;
        if q < 1.0 {
            ad * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        } else if q < 2.0 {
            ad * 0.25 * (2.0 - q).powi(3)
        } else {
            0.0
        }
    }

    /// Kernel gradient magnitude factor `dW/dr` (so `∇_i W = (dW/dr)·r̂_ij`, `r̂_ij = (x_i−x_j)/r`).
    fn dwdr(&self, r: f64) -> f64 {
        let ad = 10.0 / (7.0 * PI * self.h * self.h);
        let q = r / self.h;
        let dwdq = if q < 1.0 {
            -3.0 * q + 2.25 * q * q
        } else if q < 2.0 {
            -0.75 * (2.0 - q).powi(2)
        } else {
            0.0
        };
        ad * dwdq / self.h
    }

    fn pressure(&self, rho: f64) -> f64 {
        let b = self.rho0 * self.c0 * self.c0 / self.gamma;
        b * ((rho / self.rho0).powf(self.gamma) - 1.0)
    }

    /// Density by summation over all particles (fluid + boundary).
    pub fn compute_density(&mut self) {
        let n = self.pos.len();
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                let dx = self.pos[i][0] - self.pos[j][0];
                let dy = self.pos[i][1] - self.pos[j][1];
                s += self.mass * self.w((dx * dx + dy * dy).sqrt());
            }
            self.rho[i] = s;
        }
    }

    /// Acceleration on every fluid particle: symmetric pressure gradient + Monaghan artificial
    /// viscosity + gravity. Returns per-fluid-particle `[ax, ay]`.
    #[allow(clippy::needless_range_loop)] // pairwise (i, j) index arithmetic over several arrays
    fn accelerations(&self) -> Vec<[f64; 2]> {
        let n = self.pos.len();
        let p: Vec<f64> = (0..n).map(|i| self.pressure(self.rho[i])).collect();
        let mut acc = vec![[0.0, 0.0]; self.n_fluid];
        for i in 0..self.n_fluid {
            let (mut ax, mut ay) = (0.0, 0.0);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = self.pos[i][0] - self.pos[j][0];
                let dy = self.pos[i][1] - self.pos[j][1];
                let r2 = dx * dx + dy * dy;
                let r = r2.sqrt();
                if r >= 2.0 * self.h || r < 1e-12 {
                    continue;
                }
                let grad = self.dwdr(r);
                let (gx, gy) = (grad * dx / r, grad * dy / r);
                // symmetric pressure term
                let pterm = p[i] / (self.rho[i] * self.rho[i]) + p[j] / (self.rho[j] * self.rho[j]);
                ax -= self.mass * pterm * gx;
                ay -= self.mass * pterm * gy;
                // Monaghan artificial viscosity (only for approaching pairs)
                let dvx = self.vel[i][0] - self.vel[j][0];
                let dvy = self.vel[i][1] - self.vel[j][1];
                let vr = dvx * dx + dvy * dy;
                if vr < 0.0 {
                    let mu = self.h * vr / (r2 + 0.01 * self.h * self.h);
                    let rho_bar = 0.5 * (self.rho[i] + self.rho[j]);
                    let pi = -self.alpha * self.c0 * mu / rho_bar;
                    ax -= self.mass * pi * gx;
                    ay -= self.mass * pi * gy;
                }
            }
            ay -= self.gravity;
            acc[i] = [ax, ay];
        }
        acc
    }

    /// One symplectic-Euler step: density → forces → kick → drift (fluid only; boundary frozen).
    #[allow(clippy::needless_range_loop)] // index i addresses vel/pos/acc together
    pub fn step(&mut self, dt: f64) {
        self.compute_density();
        let acc = self.accelerations();
        for i in 0..self.n_fluid {
            self.vel[i][0] += dt * acc[i][0];
            self.vel[i][1] += dt * acc[i][1];
            self.pos[i][0] += dt * self.vel[i][0];
            self.pos[i][1] += dt * self.vel[i][1];
        }
    }

    /// Total linear momentum of the fluid (Σ m v).
    pub fn momentum(&self) -> [f64; 2] {
        let mut m = [0.0, 0.0];
        for i in 0..self.n_fluid {
            m[0] += self.mass * self.vel[i][0];
            m[1] += self.mass * self.vel[i][1];
        }
        m
    }

    /// CFL-limited timestep (acoustic).
    pub fn cfl_dt(&self) -> f64 {
        0.25 * self.h / self.c0
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// Build a rectangular block of fluid particles on a lattice of spacing `dx`.
    fn block(x0: f64, y0: f64, nx: usize, ny: usize, dx: f64) -> Vec<[f64; 2]> {
        let mut v = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                v.push([x0 + i as f64 * dx, y0 + j as f64 * dx]);
            }
        }
        v
    }

    fn make(pos: Vec<[f64; 2]>, n_fluid: usize, dx: f64, rho0: f64, c0: f64, g: f64) -> Sph {
        let n = pos.len();
        Sph {
            vel: vec![[0.0, 0.0]; n],
            rho: vec![rho0; n],
            n_fluid,
            mass: rho0 * dx * dx,
            h: 1.3 * dx,
            rho0,
            c0,
            gamma: 7.0,
            gravity: g,
            alpha: 0.2,
            pos,
        }
    }

    /// The summation density recovers ρ₀ on a uniform lattice — kernel + estimator are consistent.
    #[test]
    fn kernel_partition_recovers_density() {
        let dx = 0.02;
        // A patch large enough that the center particle has a full 2h neighborhood.
        let pos = block(0.0, 0.0, 21, 21, dx);
        let n = pos.len();
        let mut s = make(pos, n, dx, 1.0, 10.0, 0.0);
        s.compute_density();
        let center = 10 * 21 + 10; // interior particle
        let rel = (s.rho[center] - 1.0).abs() / 1.0;
        eprintln!("SPH kernel density at interior particle: {:.5} (rel err {:.2e})", s.rho[center], rel);
        assert!(rel < 0.02, "summation density off by {rel}");
    }

    /// A free blob with random-ish velocities conserves total linear momentum to machine precision —
    /// the pairwise forces are exactly antisymmetric.
    #[test]
    fn free_blob_conserves_momentum() {
        let dx = 0.02;
        let pos = block(0.0, 0.0, 12, 12, dx);
        let n = pos.len();
        let mut s = make(pos, n, dx, 1.0, 10.0, 0.0); // no gravity, no boundaries
        // Seed a swirl so pressure + viscosity forces are both active.
        for (i, v) in s.vel.iter_mut().enumerate() {
            let a = i as f64 * 0.7;
            *v = [0.05 * a.sin(), 0.05 * (a * 1.3).cos()];
        }
        let p0 = s.momentum();
        let dt = s.cfl_dt();
        for _ in 0..300 {
            s.step(dt);
        }
        let p1 = s.momentum();
        let drift = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        eprintln!("SPH momentum drift over 300 steps: {drift:.2e}");
        assert!(drift < 1e-12, "momentum not conserved: drift {drift}");
    }

    /// A fluid column in a box settles to hydrostatic equilibrium: the pressure profile is linear in
    /// depth with slope −ρ₀g. Boundary particles (floor + side walls) repel via DBC.
    #[test]
    fn column_settles_to_hydrostatic() {
        let dx = 0.025;
        let (rho0, g) = (1.0f64, 1.0f64);
        let col_h = 0.5f64;
        let c0 = 10.0 * (g * col_h).sqrt(); // weak compressibility (Ma ~ 0.1)

        // Fluid column [0,0.5]×[0,0.5]; then 3 boundary layers on floor and both side walls.
        let mut pos = block(0.0, 0.0, 20, 20, dx);
        let n_fluid = pos.len();
        let x_lo = -3.0 * dx;
        for i in 0..26 {
            for k in 1..=3 {
                pos.push([-(k as f64) * dx + 0.0, (i as f64) * dx - 3.0 * dx]); // left wall
                pos.push([0.5 - dx + (k as f64) * dx, (i as f64) * dx - 3.0 * dx]); // right wall
            }
        }
        for i in 0..26 {
            for k in 1..=3 {
                pos.push([x_lo + i as f64 * dx, -(k as f64) * dx]); // floor
            }
        }
        let mut s = make(pos, n_fluid, dx, rho0, c0, g);
        s.alpha = 0.5; // extra damping so it settles in a reasonable number of steps

        let dt = s.cfl_dt();
        for _ in 0..8000 {
            s.step(dt);
        }
        s.compute_density();

        // Fit p(z) = a·z + b over settled fluid particles; the slope a should be ≈ −ρ₀g.
        let (mut sz, mut sp, mut szz, mut szp, mut nn) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..s.n_fluid {
            let z = s.pos[i][1];
            let p = s.pressure(s.rho[i]);
            if z > 0.1 && z < 0.45 {
                // interior band: skip free surface (noisy) and floor layer
                sz += z;
                sp += p;
                szz += z * z;
                szp += z * p;
                nn += 1.0;
            }
        }
        let slope = (nn * szp - sz * sp) / (nn * szz - sz * sz);
        let rel = (slope - (-rho0 * g)).abs() / (rho0 * g);
        eprintln!("SPH hydrostatic dp/dz = {slope:.4} (expected {:.4}, rel {rel:.2e})", -rho0 * g);
        assert!(rel < 0.15, "hydrostatic pressure slope off by {rel}");
    }
}
