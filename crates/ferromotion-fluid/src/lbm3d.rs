//! **D3Q19 lattice-Boltzmann** (Honest Fluids — the lattice into three dimensions). Same BGK
//! scheme as the verified [`crate::lbm`] D2Q9, on the 19-velocity 3-D stencil, generic over
//! [`ferromotion_core::gendyn::Real`] so it inherits the differentiability of its 2-D sibling
//! (`GenLbm3<f64>` for plain numerics, a dual type for exact gradients).
//!
//! Verified against a genuinely 3-D closed-form solution: a **shear wave** `u = (u₀ sin(kz), 0, 0)`
//! on a periodic box. Because the velocity points in x but varies only in z, the nonlinear term
//! `u·∇u` vanishes identically and the field decays by pure momentum diffusion,
//! `u_x(z,t) = u₀ sin(kz) e^{−νk²t}` — an exact oracle, unlike the full (turbulent) 3-D
//! Taylor–Green vortex. Plus exact mass conservation and a continuum cross-check against the 2-D
//! solver on a z-invariant flow.

use ferromotion_core::gendyn::Real;

// D3Q19 velocity set. 0 = rest; 1–6 = faces; 7–18 = edges.
#[rustfmt::skip]
pub(crate) const CX: [i32; 19] = [0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0];
#[rustfmt::skip]
pub(crate) const CY: [i32; 19] = [0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1];
#[rustfmt::skip]
pub(crate) const CZ: [i32; 19] = [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1];
/// Weights: 1/3 (rest), 1/18 (6 faces), 1/36 (12 edges).
#[rustfmt::skip]
pub(crate) const W: [f64; 19] = [
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
];
/// Opposite direction of each velocity (for half-way bounce-back walls).
#[rustfmt::skip]
pub(crate) const OPP: [usize; 19] = [0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15, 18,17];

/// Boundary condition for the 3-D lattice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lbm3Bc<T = f64> {
    /// Fully periodic in x, y, z.
    Periodic,
    /// No-slip bounce-back on the z = 0 and z = nz−1 walls; the top (`z = nz−1`) wall translates in
    /// +x at `lid_u` (a 3-D lid-driven cavity, periodic in x and y).
    Lid { lid_u: T },
}

/// A D3Q19 BGK lattice-Boltzmann solver on an `nx × ny × nz` lattice, generic over the scalar.
pub struct GenLbm3<T = f64> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub tau: T,
    bc: Lbm3Bc<T>,
    f: Vec<T>,
    tmp: Vec<T>,
}

/// The plain-`f64` 3-D lattice.
pub type LbmD3Q19 = GenLbm3<f64>;

/// D3Q19 equilibrium distribution.
pub(crate) fn feq3<T: Real>(k: usize, rho: T, ux: T, uy: T, uz: T) -> T {
    let cu = T::from_f64(CX[k] as f64) * ux + T::from_f64(CY[k] as f64) * uy + T::from_f64(CZ[k] as f64) * uz;
    let uu = ux * ux + uy * uy + uz * uz;
    T::from_f64(W[k]) * rho
        * (T::from_f64(1.0) + T::from_f64(3.0) * cu + T::from_f64(4.5) * cu * cu - T::from_f64(1.5) * uu)
}

impl<T: Real> GenLbm3<T> {
    /// Lattice with relaxation time `tau` (`ν = (τ − ½)/3`), initialized at rest, unit density.
    pub fn new(nx: usize, ny: usize, nz: usize, tau: T, bc: Lbm3Bc<T>) -> Self {
        let mut f = vec![T::from_f64(0.0); nx * ny * nz * 19];
        for c in f.chunks_mut(19) {
            for (k, v) in c.iter_mut().enumerate() {
                *v = T::from_f64(W[k]);
            }
        }
        Self { nx, ny, nz, tau, bc, tmp: f.clone(), f }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        ((x * self.ny + y) * self.nz + z) * 19
    }

    /// Lattice kinematic viscosity `(τ − ½)/3`.
    pub fn nu(&self) -> T {
        (self.tau - T::from_f64(0.5)) / T::from_f64(3.0)
    }

    /// Set the velocity field from a function of lattice coordinates (equilibrium init, unit ρ).
    pub fn set_velocity(&mut self, fu: impl Fn(f64, f64, f64) -> (T, T, T)) {
        for x in 0..self.nx {
            for y in 0..self.ny {
                for z in 0..self.nz {
                    let (ux, uy, uz) = fu(x as f64, y as f64, z as f64);
                    let base = self.idx(x, y, z);
                    for k in 0..19 {
                        self.f[base + k] = feq3(k, T::from_f64(1.0), ux, uy, uz);
                    }
                }
            }
        }
    }

    /// Macroscopic density and velocity at a lattice site.
    pub fn macroscopic(&self, x: usize, y: usize, z: usize) -> (T, T, T, T) {
        let base = self.idx(x, y, z);
        let (mut rho, mut mx, mut my, mut mz) = (T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0));
        for k in 0..19 {
            let fk = self.f[base + k];
            rho = rho + fk;
            mx = mx + fk * T::from_f64(CX[k] as f64);
            my = my + fk * T::from_f64(CY[k] as f64);
            mz = mz + fk * T::from_f64(CZ[k] as f64);
        }
        (rho, mx / rho, my / rho, mz / rho)
    }

    /// Total mass on the lattice (conserved exactly by collision and by streaming under periodic
    /// and plain bounce-back — the moving lid exchanges momentum, not mass).
    pub fn total_mass(&self) -> T {
        let mut m = T::from_f64(0.0);
        for &v in &self.f {
            m = m + v;
        }
        m
    }

    /// One collide-and-stream step.
    pub fn step(&mut self) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let om = T::from_f64(1.0) / self.tau;
        // collide in place
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let base = self.idx(x, y, z);
                    let (mut rho, mut mx, mut my, mut mz) =
                        (T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0));
                    for k in 0..19 {
                        let fk = self.f[base + k];
                        rho = rho + fk;
                        mx = mx + fk * T::from_f64(CX[k] as f64);
                        my = my + fk * T::from_f64(CY[k] as f64);
                        mz = mz + fk * T::from_f64(CZ[k] as f64);
                    }
                    let (ux, uy, uz) = (mx / rho, my / rho, mz / rho);
                    for k in 0..19 {
                        let eq = feq3(k, rho, ux, uy, uz);
                        self.f[base + k] = self.f[base + k] + om * (eq - self.f[base + k]);
                    }
                }
            }
        }
        // stream into tmp
        let lid = match self.bc {
            Lbm3Bc::Lid { lid_u } => Some(lid_u),
            Lbm3Bc::Periodic => None,
        };
        let flat = |x: usize, y: usize, z: usize| ((x * ny + y) * nz + z) * 19;
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let base = flat(x, y, z);
                    for k in 0..19 {
                        let xn = (x as i32 + CX[k]).rem_euclid(nx as i32) as usize; // x, y always periodic
                        let yn = (y as i32 + CY[k]).rem_euclid(ny as i32) as usize;
                        let zn = z as i32 + CZ[k];
                        match lid {
                            None => {
                                let zn = zn.rem_euclid(nz as i32) as usize;
                                self.tmp[flat(xn, yn, zn) + k] = self.f[base + k];
                            }
                            Some(lid_u) => {
                                if zn < 0 || zn >= nz as i32 {
                                    // half-way bounce-back; the top wall carries the lid momentum
                                    let mut fb = self.f[base + k];
                                    if zn >= nz as i32 {
                                        fb = fb - T::from_f64(6.0 * W[k] * CX[k] as f64) * lid_u;
                                    }
                                    self.tmp[base + OPP[k]] = fb;
                                } else {
                                    self.tmp[flat(xn, yn, zn as usize) + k] = self.f[base + k];
                                }
                            }
                        }
                    }
                }
            }
        }
        core::mem::swap(&mut self.f, &mut self.tmp);
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::lbm::{GenLbm, LbmBc};
    use std::f64::consts::PI;

    /// Max |u_x − exact| / u₀ for the shear wave after `steps`, on an `n`³ box.
    fn shear_wave_error(n: usize, tau: f64, u0: f64, steps: usize) -> f64 {
        let k = 2.0 * PI / n as f64;
        let mut l = GenLbm3::new(n, n, n, tau, Lbm3Bc::Periodic);
        l.set_velocity(|_, _, z| (u0 * (k * z).sin(), 0.0, 0.0));
        for _ in 0..steps {
            l.step();
        }
        let nu = (tau - 0.5) / 3.0;
        let decay = (-nu * k * k * steps as f64).exp();
        let mut err = 0.0f64;
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let (_, ux, _, _) = l.macroscopic(x, y, z);
                    let want = u0 * (k * z as f64).sin() * decay;
                    err = err.max((ux - want).abs());
                }
            }
        }
        err / u0
    }

    /// The shear wave decays at the analytic diffusion rate, and the field error converges at ~2nd
    /// order under diffusive scaling (u₀ ∝ 1/n so the O(Ma²) compressibility error shrinks too).
    #[test]
    fn shear_wave_matches_analytic_and_converges() {
        let tau = 0.8;
        // Fixed diffusion time t = ν k² · steps held ~constant across resolutions: steps ∝ n².
        let e16 = shear_wave_error(16, tau, 0.04, 200);
        let e32 = shear_wave_error(32, tau, 0.02, 800);
        let order = (e16 / e32).log2();
        eprintln!("D3Q19 shear wave: e16 {e16:.3e}  e32 {e32:.3e}  order {order:.2}");
        assert!(e16 < 5e-3, "coarse shear-wave error too large: {e16}");
        assert!(order > 1.7, "shear-wave convergence order {order} < 1.7");
    }

    /// Mass is conserved to round-off under periodic streaming.
    #[test]
    fn mass_is_conserved() {
        let n = 12;
        let mut l = GenLbm3::new(n, n, n, 0.9, Lbm3Bc::Periodic);
        let k = 2.0 * PI / n as f64;
        l.set_velocity(|x, _, z| (0.03 * (k * z).sin(), 0.02 * (k * x).cos(), 0.0));
        let m0 = l.total_mass();
        for _ in 0..300 {
            l.step();
        }
        let rel = (l.total_mass() - m0).abs() / m0.abs(); // ~ε per site accumulated; total mass ≈ n³
        eprintln!("D3Q19 relative mass drift over 300 steps: {rel:.2e}");
        assert!(rel < 1e-12, "mass not conserved: relative drift {rel}");
    }

    /// Continuum cross-oracle: a z-invariant 2-D Taylor–Green flow run on the D3Q19 lattice must
    /// agree with the verified D2Q9 solver on the same flow (both converge to the same NS limit).
    #[test]
    fn z_invariant_flow_matches_2d_solver() {
        let n = 32;
        let tau = 0.8;
        let (u0, k) = (0.03, 2.0 * PI / n as f64);
        let init2 = |x: f64, y: f64| (u0 * (k * x).sin() * (k * y).cos(), -u0 * (k * x).cos() * (k * y).sin());

        let mut l3 = GenLbm3::new(n, n, 4, tau, Lbm3Bc::Periodic);
        l3.set_velocity(|x, y, _| {
            let (ux, uy) = init2(x, y);
            (ux, uy, 0.0)
        });
        let mut l2 = GenLbm::new(n, n, tau, LbmBc::Periodic);
        l2.set_velocity(init2);

        let steps = 150;
        for _ in 0..steps {
            l3.step();
            l2.step();
        }
        let mut err = 0.0f64;
        for x in 0..n {
            for y in 0..n {
                let (_, ux3, uy3, uz3) = l3.macroscopic(x, y, 0);
                let (_, ux2, uy2) = l2.macroscopic(x, y);
                err = err.max((ux3 - ux2).abs().max((uy3 - uy2).abs()).max(uz3.abs()));
            }
        }
        eprintln!("D3Q19 vs D2Q9 on z-invariant TG: max |Δu| {err:.2e} (rel {:.2e})", err / u0);
        // Different velocity sets => not bit-identical, but the same NS flow to sub-% agreement,
        // and the z-component must stay exactly zero (no spurious 3-D motion).
        assert!(err / u0 < 5e-3, "3-D vs 2-D disagreement {}", err / u0);
    }
}

#[cfg(test)]
mod gradients {
    use super::*;
    use ferromotion_learn::Dual;
    use std::f64::consts::PI;

    /// A shear-wave probe after `steps`, as a function of τ — the 3-D lattice carried on a dual so
    /// the derivative flows through every collide+stream. Confirms the generic-`Real` D3Q19 is
    /// differentiable, exactly as its 2-D sibling.
    fn probe<T: Real>(tau: T, steps: usize) -> T {
        let n = 16;
        let k = 2.0 * PI / n as f64;
        let mut l = GenLbm3::new(n, n, n, tau, Lbm3Bc::Periodic);
        l.set_velocity(|_, _, z| (T::from_f64(0.03 * (k * z).sin()), T::from_f64(0.0), T::from_f64(0.0)));
        for _ in 0..steps {
            l.step();
        }
        l.macroscopic(3, 3, 4).1
    }

    /// d(probe)/dτ through the entire 3-D simulation — dual vs central finite differences.
    #[test]
    fn dual_gradient_through_the_3d_simulation_matches_fd() {
        let (tau, steps, eps) = (0.8f64, 80usize, 1e-6);
        let got = probe(Dual { re: tau, eps: 1.0 }, steps).eps;
        let want = (probe(tau + eps, steps) - probe(tau - eps, steps)) / (2.0 * eps);
        eprintln!("D3Q19 d(probe)/dτ: dual {got:.6e}  fd {want:.6e}");
        assert!(
            (got - want).abs() < 1e-6 * want.abs().max(1e-3),
            "d(probe)/dτ: dual {got} vs FD {want}"
        );
    }
}
