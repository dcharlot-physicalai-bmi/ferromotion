//! **Lattice-Boltzmann (D2Q9, BGK)** — the second solver of the Honest Fluids ingestion: the
//! method behind FluidX3D-class throughput, here as a wasm-clean CPU reference implementation
//! verified against analytic solutions and cross-checked against the MAC projection solver on the
//! canonical cavity benchmark (two independent discretizations, one Ghia table). The fabric GPU
//! path (wgpu compute — hardware-open where the incumbents are OpenCL/CUDA-bound) builds on this
//! reference; correctness first, throughput after.
//!
//! Collision is single-relaxation-time BGK: `ν_lattice = (τ − ½)/3` (lattice units, `c_s² = ⅓`).
//! Boundaries: fully periodic (the Taylor–Green case) or a closed box of half-way bounce-back
//! walls with a momentum-corrected moving lid (`f_opp = f_i − 6·w_i·ρ·(c_i·u_lid)` at the top) —
//! the standard cavity setup. Working in lattice units keeps the scheme exact; verification maps
//! to physics through the usual scalings (diffusive scaling for convergence studies, so the
//! O(Ma²) compressibility error shrinks with the grid).

/// D2Q9 lattice velocities, weights, and opposite-direction table.
const CX: [i32; 9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const CY: [i32; 9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const W: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
const OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Boundary handling for the box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LbmBc {
    /// Fully periodic in x and y.
    Periodic,
    /// No-slip bounce-back walls; the top wall translates at `lid_u` (lattice units).
    Cavity { lid_u: f64 },
}

/// A D2Q9 BGK lattice-Boltzmann solver on an `nx × ny` lattice.
pub struct LbmD2Q9 {
    pub nx: usize,
    pub ny: usize,
    pub tau: f64,
    bc: LbmBc,
    f: Vec<f64>,    // nx*ny*9, index (x*ny + y)*9 + k
    tmp: Vec<f64>,  // streaming target
}

impl LbmD2Q9 {
    /// Lattice with relaxation time `tau` (`ν = (τ − ½)/3`), initialized at rest, unit density.
    pub fn new(nx: usize, ny: usize, tau: f64, bc: LbmBc) -> Self {
        assert!(tau > 0.5, "τ must exceed ½ (positive viscosity)");
        let mut f = vec![0.0; nx * ny * 9];
        for c in f.chunks_mut(9) {
            c.copy_from_slice(&W);
        }
        Self { nx, ny, tau, bc, tmp: f.clone(), f }
    }

    /// Lattice kinematic viscosity `(τ − ½)/3`.
    pub fn nu(&self) -> f64 {
        (self.tau - 0.5) / 3.0
    }

    /// Set the velocity field from a function of lattice coordinates (equilibrium initialization
    /// at unit density) — initial conditions for verification cases.
    pub fn set_velocity(&mut self, fu: impl Fn(f64, f64) -> (f64, f64)) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                let (ux, uy) = fu(x as f64, y as f64);
                let base = (x * self.ny + y) * 9;
                for k in 0..9 {
                    self.f[base + k] = feq(k, 1.0, ux, uy);
                }
            }
        }
    }

    /// Macroscopic density and velocity at a lattice site.
    pub fn macroscopic(&self, x: usize, y: usize) -> (f64, f64, f64) {
        let base = (x * self.ny + y) * 9;
        let mut rho = 0.0;
        let mut mx = 0.0;
        let mut my = 0.0;
        for k in 0..9 {
            let fk = self.f[base + k];
            rho += fk;
            mx += fk * CX[k] as f64;
            my += fk * CY[k] as f64;
        }
        (rho, mx / rho, my / rho)
    }

    /// Total mass on the lattice (conserved exactly by collision; by streaming under periodic and
    /// plain bounce-back — the moving lid exchanges momentum, not mass).
    pub fn total_mass(&self) -> f64 {
        self.f.iter().sum()
    }

    /// One collide-and-stream step.
    pub fn step(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        let om = 1.0 / self.tau;
        // collide in place
        for x in 0..nx {
            for y in 0..ny {
                let base = (x * ny + y) * 9;
                let mut rho = 0.0;
                let mut mx = 0.0;
                let mut my = 0.0;
                for k in 0..9 {
                    let fk = self.f[base + k];
                    rho += fk;
                    mx += fk * CX[k] as f64;
                    my += fk * CY[k] as f64;
                }
                let (ux, uy) = (mx / rho, my / rho);
                for k in 0..9 {
                    let eq = feq(k, rho, ux, uy);
                    self.f[base + k] += om * (eq - self.f[base + k]);
                }
            }
        }
        // stream into tmp with boundary handling
        match self.bc {
            LbmBc::Periodic => {
                for x in 0..nx {
                    for y in 0..ny {
                        let base = (x * ny + y) * 9;
                        for k in 0..9 {
                            let xn = (x as i32 + CX[k]).rem_euclid(nx as i32) as usize;
                            let yn = (y as i32 + CY[k]).rem_euclid(ny as i32) as usize;
                            self.tmp[(xn * ny + yn) * 9 + k] = self.f[base + k];
                        }
                    }
                }
            }
            LbmBc::Cavity { lid_u } => {
                for x in 0..nx {
                    for y in 0..ny {
                        let base = (x * ny + y) * 9;
                        for k in 0..9 {
                            let xn = x as i32 + CX[k];
                            let yn = y as i32 + CY[k];
                            if xn < 0 || xn >= nx as i32 || yn < 0 || yn >= ny as i32 {
                                // half-way bounce-back; the top wall carries the lid momentum
                                let mut fb = self.f[base + k];
                                if yn >= ny as i32 {
                                    let rho = 1.0; // standard first-order wall-density closure
                                    fb -= 6.0 * W[k] * rho * (CX[k] as f64 * lid_u);
                                }
                                self.tmp[base + OPP[k]] = fb;
                            } else {
                                self.tmp[((xn as usize) * ny + yn as usize) * 9 + k] = self.f[base + k];
                            }
                        }
                    }
                }
            }
        }
        core::mem::swap(&mut self.f, &mut self.tmp);
    }

    /// `u_x/lid_u` along the vertical centerline (cavity verification), as `(y_normalized, u)`.
    pub fn centerline_u(&self) -> Vec<(f64, f64)> {
        let x = self.nx / 2;
        let lid = match self.bc {
            LbmBc::Cavity { lid_u } => lid_u,
            LbmBc::Periodic => 1.0,
        };
        (0..self.ny)
            .map(|y| {
                let (_, ux, _) = self.macroscopic(x, y);
                ((y as f64 + 0.5) / self.ny as f64, ux / lid)
            })
            .collect()
    }
}

/// D2Q9 equilibrium distribution.
fn feq(k: usize, rho: f64, ux: f64, uy: f64) -> f64 {
    let cu = CX[k] as f64 * ux + CY[k] as f64 * uy;
    let uu = ux * ux + uy * uy;
    W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)
}

#[cfg(test)]
mod verification {
    use super::*;
    use std::f64::consts::PI;

    /// Taylor–Green on the periodic lattice under DIFFUSIVE scaling (u₀ ∝ 1/N, so the O(Ma²)
    /// compressibility error shrinks with the grid): the decay must track the analytic rate and
    /// the velocity-field error must converge at ~2nd order.
    #[test]
    fn taylor_green_periodic_decay_and_order() {
        let run = |n: usize| -> f64 {
            let tau = 0.8; // ν = 0.1 lattice
            let u0 = 4.0 / n as f64; // diffusive scaling
            let mut l = LbmD2Q9::new(n, n, tau, LbmBc::Periodic);
            let k = 2.0 * PI / n as f64;
            l.set_velocity(|x, y| (u0 * (k * x).sin() * (k * y).cos(), -u0 * (k * x).cos() * (k * y).sin()));
            let nu = l.nu();
            // decay to ~60% amplitude
            let t_end = (0.5f64.ln() * -1.0) / (2.0 * nu * k * k);
            let steps = t_end.round() as usize;
            for _ in 0..steps {
                l.step();
            }
            let decay = (-2.0 * nu * k * k * steps as f64).exp();
            let mut err = 0.0f64;
            for x in 0..n {
                for y in 0..n {
                    let (_, ux, uy) = l.macroscopic(x, y);
                    let want_x = u0 * (k * x as f64).sin() * (k * y as f64).cos() * decay;
                    let want_y = -u0 * (k * x as f64).cos() * (k * y as f64).sin() * decay;
                    err = err.max((ux - want_x).abs().max((uy - want_y).abs()));
                }
            }
            err / u0 // relative to the initial amplitude
        };
        let e32 = run(32);
        let e64 = run(64);
        let order = (e32 / e64).log2();
        assert!(e32 < 0.02, "N=32 relative error {e32}");
        assert!(order > 1.6, "convergence order {order:.2} (e32={e32:.2e}, e64={e64:.2e})");
        eprintln!("LBM Taylor–Green: e32 {e32:.3e}, e64 {e64:.3e}, observed order {order:.2}");
    }

    /// Mass is conserved to machine precision — periodic AND bounce-back cavity.
    #[test]
    fn mass_is_conserved_exactly() {
        for bc in [LbmBc::Periodic, LbmBc::Cavity { lid_u: 0.1 }] {
            let mut l = LbmD2Q9::new(48, 48, 0.7, bc);
            l.set_velocity(|x, y| (0.05 * (0.1 * x).sin(), 0.05 * (0.13 * y).cos()));
            let m0 = l.total_mass();
            for _ in 0..500 {
                l.step();
            }
            let drift = (l.total_mass() - m0).abs() / m0;
            assert!(drift < 1e-12, "{bc:?}: mass drift {drift:.2e}");
        }
    }

    /// The cross-oracle: lid-driven cavity at Re = 100 vs the same Ghia table the MAC solver is
    /// verified against — two independent discretizations of the same physics, one benchmark.
    #[test]
    fn cavity_re100_matches_ghia_cross_oracle() {
        const GHIA: &[(f64, f64)] = &[
            (0.0547, -0.03717),
            (0.1016, -0.06434),
            (0.2813, -0.15662),
            (0.4531, -0.21090),
            (0.5000, -0.20581),
            (0.6172, -0.13641),
            (0.7344, 0.00332),
            (0.8516, 0.23151),
            (0.9531, 0.68717),
            (0.9766, 0.84123),
        ];
        let n = 96;
        let lid = 0.1; // lattice Ma = 0.1·√3 ≈ 0.17 — modest compressibility
        let nu = lid * n as f64 / 100.0; // Re = 100
        let tau = 3.0 * nu + 0.5;
        let mut l = LbmD2Q9::new(n, n, tau, LbmBc::Cavity { lid_u: lid });
        // run to steady state: ~40 box-transit times
        let steps = (40.0 * n as f64 / lid) as usize;
        for _ in 0..steps {
            l.step();
        }
        let profile = l.centerline_u();
        let interp = |yq: f64| -> f64 {
            for w in profile.windows(2) {
                let ((y0, u0), (y1, u1)) = (w[0], w[1]);
                if (y0..=y1).contains(&yq) {
                    return u0 + (u1 - u0) * (yq - y0) / (y1 - y0);
                }
            }
            profile.last().unwrap().1
        };
        let mut worst = 0.0f64;
        for &(y, u_ref) in GHIA {
            worst = worst.max((interp(y) - u_ref).abs());
        }
        assert!(worst < 0.03, "LBM cavity vs Ghia: worst deviation {worst:.4}");
        eprintln!("LBM cavity Re=100 vs Ghia: worst centerline deviation {worst:.4}");
    }
}
