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
//!
//! **Differentiable by construction** (Honest Fluids stage 3): the whole solver is generic over
//! [`ferromotion_core::gendyn::Real`] — collide is a rational polynomial, streaming a
//! permutation, bounce-back linear — so a dual number seeded in `τ`, the lid speed, or any
//! initial-condition parameter flows through the *entire simulation* and emerges as an exact
//! gradient of any observable. [`LbmD2Q9`] (= `GenLbm<f64>`) keeps the plain API. The verified
//! payoff (tests): identifying the lattice viscosity from observed decay data by gradient — the
//! calibration pattern that is differentiability's real job.

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

use ferromotion_core::gendyn::Real;

/// Boundary handling for the box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LbmBc<T = f64> {
    /// Fully periodic in x and y.
    Periodic,
    /// No-slip bounce-back walls; the top wall translates at `lid_u` (lattice units).
    Cavity { lid_u: T },
}

/// A D2Q9 BGK lattice-Boltzmann solver on an `nx × ny` lattice, generic over the scalar
/// (`GenLbm<f64>` for plain numerics — aliased as [`LbmD2Q9`]; a dual type for exact gradients).
pub struct GenLbm<T = f64> {
    pub nx: usize,
    pub ny: usize,
    pub tau: T,
    bc: LbmBc<T>,
    f: Vec<T>,    // nx*ny*9, index (x*ny + y)*9 + k
    tmp: Vec<T>,  // streaming target
}

/// The plain-`f64` lattice.
pub type LbmD2Q9 = GenLbm<f64>;

impl<T: Real> GenLbm<T> {
    /// Lattice with relaxation time `tau` (`ν = (τ − ½)/3`), initialized at rest, unit density.
    pub fn new(nx: usize, ny: usize, tau: T, bc: LbmBc<T>) -> Self {
        let mut f = vec![T::from_f64(0.0); nx * ny * 9];
        for c in f.chunks_mut(9) {
            for (k, v) in c.iter_mut().enumerate() {
                *v = T::from_f64(W[k]);
            }
        }
        Self { nx, ny, tau, bc, tmp: f.clone(), f }
    }

    /// Lattice kinematic viscosity `(τ − ½)/3`.
    pub fn nu(&self) -> T {
        (self.tau - T::from_f64(0.5)) / T::from_f64(3.0)
    }

    /// Set the velocity field from a function of lattice coordinates (equilibrium initialization
    /// at unit density) — initial conditions for verification cases.
    pub fn set_velocity(&mut self, fu: impl Fn(f64, f64) -> (T, T)) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                let (ux, uy) = fu(x as f64, y as f64);
                let base = (x * self.ny + y) * 9;
                for k in 0..9 {
                    self.f[base + k] = feq(k, T::from_f64(1.0), ux, uy);
                }
            }
        }
    }

    /// Macroscopic density and velocity at a lattice site.
    pub fn macroscopic(&self, x: usize, y: usize) -> (T, T, T) {
        let base = (x * self.ny + y) * 9;
        let mut rho = T::from_f64(0.0);
        let mut mx = T::from_f64(0.0);
        let mut my = T::from_f64(0.0);
        for k in 0..9 {
            let fk = self.f[base + k];
            rho = rho + fk;
            mx = mx + fk * T::from_f64(CX[k] as f64);
            my = my + fk * T::from_f64(CY[k] as f64);
        }
        (rho, mx / rho, my / rho)
    }

    /// Total mass on the lattice (conserved exactly by collision; by streaming under periodic and
    /// plain bounce-back — the moving lid exchanges momentum, not mass).
    pub fn total_mass(&self) -> T {
        let mut m = T::from_f64(0.0);
        for &v in &self.f {
            m = m + v;
        }
        m
    }

    /// One collide-and-stream step.
    pub fn step(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        let om = T::from_f64(1.0) / self.tau;
        // collide in place
        for x in 0..nx {
            for y in 0..ny {
                let base = (x * ny + y) * 9;
                let mut rho = T::from_f64(0.0);
                let mut mx = T::from_f64(0.0);
                let mut my = T::from_f64(0.0);
                for k in 0..9 {
                    let fk = self.f[base + k];
                    rho = rho + fk;
                    mx = mx + fk * T::from_f64(CX[k] as f64);
                    my = my + fk * T::from_f64(CY[k] as f64);
                }
                let (ux, uy) = (mx / rho, my / rho);
                for k in 0..9 {
                    let eq = feq(k, rho, ux, uy);
                    self.f[base + k] = self.f[base + k] + om * (eq - self.f[base + k]);
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
                                    // standard first-order wall-density closure (ρ_wall = 1)
                                    fb = fb - T::from_f64(6.0 * W[k] * CX[k] as f64) * lid_u;
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
    pub fn centerline_u(&self) -> Vec<(f64, T)> {
        let x = self.nx / 2;
        let lid = match self.bc {
            LbmBc::Cavity { lid_u } => lid_u,
            LbmBc::Periodic => T::from_f64(1.0),
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
fn feq<T: Real>(k: usize, rho: T, ux: T, uy: T) -> T {
    let cu = T::from_f64(CX[k] as f64) * ux + T::from_f64(CY[k] as f64) * uy;
    let uu = ux * ux + uy * uy;
    T::from_f64(W[k]) * rho
        * (T::from_f64(1.0) + T::from_f64(3.0) * cu + T::from_f64(4.5) * cu * cu - T::from_f64(1.5) * uu)
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

#[cfg(test)]
mod gradients {
    use super::*;
    use ferromotion_learn::Dual;
    use std::f64::consts::PI;

    fn tg_lattice<T: Real>(n: usize, tau: T, u0: T) -> GenLbm<T> {
        let mut l = GenLbm::new(n, n, tau, LbmBc::Periodic);
        let k = 2.0 * PI / n as f64;
        l.set_velocity(|x, y| {
            (
                u0 * T::from_f64((k * x).sin() * (k * y).cos()),
                T::from_f64(-1.0) * u0 * T::from_f64((k * x).cos() * (k * y).sin()),
            )
        });
        l
    }

    /// A probe observable after S steps, as a function of τ.
    fn probe<T: Real>(tau: T, steps: usize) -> T {
        let mut l = tg_lattice(24, tau, T::from_f64(0.08));
        for _ in 0..steps {
            l.step();
        }
        let (_, ux, _) = l.macroscopic(5, 11);
        ux
    }

    /// d(observable)/dτ through the ENTIRE simulation — dual vs central finite differences.
    #[test]
    fn dual_gradient_through_the_simulation_matches_fd() {
        let (tau, steps, eps) = (0.8f64, 60usize, 1e-6);
        let got = probe(Dual { re: tau, eps: 1.0 }, steps).eps;
        let want = (probe(tau + eps, steps) - probe(tau - eps, steps)) / (2.0 * eps);
        assert!(
            (got - want).abs() < 1e-7 * want.abs().max(1e-3),
            "d(probe)/dτ: dual {got} vs FD {want}"
        );
        // and through the lid boundary condition too
        let lid_probe = |lid: Dual| -> Dual {
            let mut l = GenLbm::new(24, 24, Dual::constant(0.7), LbmBc::Cavity { lid_u: lid });
            for _ in 0..80 {
                l.step();
            }
            l.macroscopic(12, 20).1
        };
        let lid = 0.08;
        let got = lid_probe(Dual { re: lid, eps: 1.0 }).eps;
        let f = |l: f64| -> f64 {
            let mut s = GenLbm::new(24, 24, 0.7, LbmBc::Cavity { lid_u: l });
            for _ in 0..80 {
                s.step();
            }
            s.macroscopic(12, 20).1
        };
        let want = (f(lid + eps) - f(lid - eps)) / (2.0 * eps);
        assert!((got - want).abs() < 1e-7 * want.abs().max(1e-6), "d/d(lid): dual {got} vs FD {want}");
    }

    /// The payoff — viscosity identification from observed decay: generate an energy trace with a
    /// hidden true τ*, start 20% wrong, and let exact dual gradients through the simulation pull τ
    /// onto the truth. Differentiability's real job, on a real solver.
    #[test]
    fn viscosity_identifies_from_decay_by_gradient() {
        let (n, steps_per_obs, n_obs) = (24usize, 25usize, 6usize);
        let energy_trace = |tau: f64| -> Vec<f64> {
            let mut l = tg_lattice(n, tau, 0.08);
            let mut out = Vec::new();
            for _ in 0..n_obs {
                for _ in 0..steps_per_obs {
                    l.step();
                }
                let mut e = 0.0;
                for x in 0..n {
                    for y in 0..n {
                        let (_, ux, uy) = l.macroscopic(x, y);
                        e += ux * ux + uy * uy;
                    }
                }
                out.push(e);
            }
            out
        };
        let tau_true = 0.85;
        let obs = energy_trace(tau_true);

        // loss and its exact gradient at a candidate τ, via one dual run
        let loss_grad = |tau: f64| -> (f64, f64) {
            let mut l = tg_lattice(n, Dual { re: tau, eps: 1.0 }, Dual::constant(0.08));
            let (mut loss, mut grad) = (0.0, 0.0);
            for ob in &obs {
                for _ in 0..steps_per_obs {
                    l.step();
                }
                let mut e = Dual::constant(0.0);
                for x in 0..n {
                    for y in 0..n {
                        let (_, ux, uy) = l.macroscopic(x, y);
                        e = e + ux * ux + uy * uy;
                    }
                }
                let r = e.re - ob;
                loss += r * r;
                grad += 2.0 * r * e.eps;
            }
            (loss, grad)
        };

        let mut tau = 0.68; // 20% wrong
        let mut lr = 0.5;
        let (mut best_loss, _) = loss_grad(tau);
        for _ in 0..60 {
            let (_, g) = loss_grad(tau);
            let cand = (tau - lr * g).max(0.55);
            let (cl, _) = loss_grad(cand);
            if cl < best_loss {
                tau = cand;
                best_loss = cl;
                lr *= 1.2;
            } else {
                lr *= 0.5;
            }
            if best_loss < 1e-18 {
                break;
            }
        }
        let rel = (tau - tau_true).abs() / tau_true;
        assert!(rel < 0.01, "τ identification: {tau} vs {tau_true} (rel {rel:.4})");
        eprintln!("viscosity identified: τ {tau:.5} vs true {tau_true} (rel err {rel:.2e})");
    }
}
