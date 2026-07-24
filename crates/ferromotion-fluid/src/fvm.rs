//! **Finite-volume advection–diffusion** (Honest Fluids — stage 6, the FVM beachhead). This is the
//! SU2/OpenFOAM discretization family, ingested with the same oracle discipline: a cell-centered,
//! flux-form solver on a structured periodic grid. Two exact oracles, both defining properties of
//! the method:
//!
//! - **Discrete conservation.** The scheme is flux-form — every face flux is *added* to one cell and
//!   *subtracted* from its neighbor — so `Σ φ` changes only through domain-boundary fluxes. Under
//!   periodic boundaries it is conserved to machine precision. This is *the* reason FVM dominates
//!   CFD, and it is an exact test.
//! - **Second-order accuracy.** Central face reconstruction converges at order 2 against the exact
//!   advection–diffusion of a plane wave, `φ = sin(k·x − ωt)·e^{−D|k|²t}` (no manufactured source
//!   needed — this is the analytic solution). First-order upwind is shown, honestly, to be only
//!   order 1 — the scheme knows its own accuracy.
//!
//! Explicit RK-free time march; central or upwind advective flux; central diffusive flux. Kept in
//! the cell-Péclet-stable regime for the central scheme.

/// Advective face reconstruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Flux {
    /// Central (2nd-order, dispersive; stable when cell Péclet < 2).
    Central,
    /// First-order upwind (diffusive, unconditionally sign-stable).
    Upwind,
}

/// A cell-centered finite-volume field on an `n × n` periodic grid, transporting a scalar under a
/// constant velocity with constant diffusivity.
pub struct Fvm {
    pub n: usize,
    pub h: f64,
    pub ux: f64,
    pub uy: f64,
    pub d: f64,
    pub flux: Flux,
    phi: Vec<f64>,
}

impl Fvm {
    /// A grid on the unit square with velocity `(ux, uy)`, diffusivity `d`.
    pub fn new(n: usize, ux: f64, uy: f64, d: f64, flux: Flux) -> Self {
        Fvm { n, h: 1.0 / n as f64, ux, uy, d, flux, phi: vec![0.0; n * n] }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        (i % self.n) * self.n + (j % self.n)
    }

    /// Initialize `φ` from a function of cell-center coordinates.
    pub fn set(&mut self, f: impl Fn(f64, f64) -> f64) {
        for i in 0..self.n {
            for j in 0..self.n {
                self.phi[i * self.n + j] = f((i as f64 + 0.5) * self.h, (j as f64 + 0.5) * self.h);
            }
        }
    }

    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.phi[self.idx(i, j)]
    }

    /// Total scalar `Σ φ · cell-area` (conserved under periodic BC).
    pub fn total(&self) -> f64 {
        self.phi.iter().sum::<f64>() * self.h * self.h
    }

    /// Advective flux at a face between cell `a` (upstream side) and `b`, with face velocity `vel`.
    fn adv_face(&self, a: f64, b: f64, vel: f64) -> f64 {
        match self.flux {
            Flux::Central => vel * 0.5 * (a + b),
            Flux::Upwind => {
                if vel >= 0.0 {
                    vel * a
                } else {
                    vel * b
                }
            }
        }
    }

    /// One explicit time step of size `dt` (flux-form; periodic).
    pub fn step(&mut self, dt: f64) {
        let (n, h, d) = (self.n, self.h, self.d);
        let mut next = self.phi.clone();
        for i in 0..n {
            for j in 0..n {
                let c = self.phi[i * n + j];
                let e = self.phi[self.idx(i + 1, j)];
                let w = self.phi[self.idx(i + n - 1, j)];
                let nth = self.phi[self.idx(i, j + 1)];
                let s = self.phi[self.idx(i, j + n - 1)];
                // face advective fluxes (F_e uses (c,e); F_w uses (w,c)); flux-form divergence
                let fe = self.adv_face(c, e, self.ux);
                let fw = self.adv_face(w, c, self.ux);
                let fn_ = self.adv_face(c, nth, self.uy);
                let fs = self.adv_face(s, c, self.uy);
                let adv = (fe - fw + fn_ - fs) / h;
                // central diffusive flux divergence (Laplacian)
                let lap = (e + w + nth + s - 4.0 * c) / (h * h);
                next[i * n + j] = c + dt * (-adv + d * lap);
            }
        }
        self.phi = next;
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use std::f64::consts::PI;

    /// Exact plane-wave solution of ∂φ/∂t + u·∇φ = D∇²φ on the periodic unit square.
    fn exact(x: f64, y: f64, t: f64, ux: f64, uy: f64, d: f64) -> f64 {
        let k = 2.0 * PI;
        let phase = k * (x - ux * t) + k * (y - uy * t);
        phase.sin() * (-d * (k * k + k * k) * t).exp()
    }

    fn run(n: usize, flux: Flux, ux: f64, uy: f64, d: f64, t_end: f64) -> (Fvm, f64) {
        let mut f = Fvm::new(n, ux, uy, d, flux);
        f.set(|x, y| exact(x, y, 0.0, ux, uy, d));
        // stable dt: advective CFL + diffusive limit, generous safety factor.
        let h = 1.0 / n as f64;
        let dt = 0.2 * (h / (ux.abs() + uy.abs() + 1e-9)).min(h * h / (4.0 * d));
        let steps = (t_end / dt).ceil() as usize;
        let dt = t_end / steps as f64;
        for _ in 0..steps {
            f.step(dt);
        }
        // L2 error vs exact
        let mut se = 0.0;
        for i in 0..n {
            for j in 0..n {
                let e = exact((i as f64 + 0.5) * h, (j as f64 + 0.5) * h, t_end, ux, uy, d);
                se += (f.at(i, j) - e).powi(2);
            }
        }
        (f, (se / (n * n) as f64).sqrt())
    }

    /// Flux-form ⇒ `Σφ` conserved to machine precision under periodic boundaries.
    #[test]
    fn flux_form_conserves_the_scalar() {
        let mut f = Fvm::new(48, 0.6, 0.4, 0.02, Flux::Central);
        f.set(|x, y| (2.0 * PI * x).sin() * (2.0 * PI * y).cos() + 1.0); // mean 1 + a wave
        let t0 = f.total();
        let dt = 0.2 * f.h / 1.0;
        for _ in 0..500 {
            f.step(dt);
        }
        let drift = (f.total() - t0).abs();
        eprintln!("FVM scalar drift over 500 steps: {drift:.2e}");
        assert!(drift < 1e-12, "flux-form did not conserve: drift {drift}");
    }

    /// Central reconstruction converges at order ≈ 2 against the exact advection–diffusion wave.
    #[test]
    fn central_flux_is_second_order() {
        let (ux, uy, d, t) = (0.5, 0.3, 0.04, 0.1); // cell Péclet < 2 at these grids
        let (_, e32) = run(32, Flux::Central, ux, uy, d, t);
        let (_, e64) = run(64, Flux::Central, ux, uy, d, t);
        let order = (e32 / e64).log2();
        eprintln!("FVM central: e32 {e32:.3e}  e64 {e64:.3e}  order {order:.2}");
        assert!(order > 1.8, "central flux not 2nd order: {order}");
    }

    /// First-order upwind is honestly only order ≈ 1 — the scheme knows its own accuracy.
    #[test]
    fn upwind_flux_is_first_order() {
        let (ux, uy, d, t) = (0.5, 0.3, 0.04, 0.1);
        let (_, e32) = run(32, Flux::Upwind, ux, uy, d, t);
        let (_, e64) = run(64, Flux::Upwind, ux, uy, d, t);
        let order = (e32 / e64).log2();
        eprintln!("FVM upwind: e32 {e32:.3e}  e64 {e64:.3e}  order {order:.2}");
        assert!(order > 0.8 && order < 1.5, "upwind should be ~1st order, got {order}");
    }
}
