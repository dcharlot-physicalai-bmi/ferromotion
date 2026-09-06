//! **Spectral Navier–Stokes** (Honest Fluids — stage 7, the operators become a solver). A 2-D
//! incompressible vorticity–streamfunction solver on a periodic box, built on the [`crate::spectral`]
//! FFT: the streamfunction Poisson solve and every derivative are spectral (exact), the nonlinear
//! advection is evaluated pseudo-spectrally with 2/3-rule dealiasing, and the viscous term is
//! integrated *exactly* by an integrating factor — so the only time-discretization error is in the
//! nonlinear term (IF-RK2, 2nd order).
//!
//! Verified against the exact Taylor–Green decay: for the single-mode TG field the nonlinear term
//! `u·∇ω` vanishes identically, so a correct spectral solver reproduces `ω = 2k·sin(kx)sin(ky)·
//! e^{−2νk²t}` to machine precision — no numerical dissipation, no phase error.

use crate::spectral::{fft2, wavenumbers, Cplx};

/// A 2-D incompressible spectral solver, state held as the vorticity spectrum `ω̂` on an `n × n`
/// grid over `[0, L)²`.
pub struct SpectralNs {
    pub n: usize,
    pub l: f64,
    pub nu: f64,
    w_hat: Vec<Cplx>,
    kx: Vec<f64>,
    dealias: Vec<bool>,
}

impl SpectralNs {
    /// Build from an initial vorticity field (row-major `n×n`, `n` a power of two).
    pub fn new(omega0: &[f64], n: usize, l: f64, nu: f64) -> Self {
        let mut w_hat: Vec<Cplx> = omega0.iter().map(|&x| Cplx::new(x, 0.0)).collect();
        fft2(&mut w_hat, n, false);
        let k = wavenumbers(n, l);
        // 2/3-rule dealiasing mask: zero modes with |k| beyond 2/3 of the Nyquist.
        let kmax = 2.0 * std::f64::consts::PI * (n as f64 / 2.0) / l;
        let cut = (2.0 / 3.0) * kmax;
        let mut dealias = vec![false; n * n];
        for i in 0..n {
            for j in 0..n {
                dealias[i * n + j] = k[i].abs() <= cut && k[j].abs() <= cut;
            }
        }
        SpectralNs { n, l, nu, w_hat, kx: k, dealias }
    }

    /// Nonlinear term spectrum `N̂ = −FFT(u·∇ω)`, dealiased. `u = ∂ψ/∂y`, `v = −∂ψ/∂x`, `∇²ψ = −ω`.
    #[allow(clippy::needless_range_loop)] // spectral index i addresses coupled mode arrays
    fn nonlinear(&self, w_hat: &[Cplx]) -> Vec<Cplx> {
        let (n, k) = (self.n, &self.kx);
        let mut u = vec![Cplx::new(0.0, 0.0); n * n];
        let mut v = vec![Cplx::new(0.0, 0.0); n * n];
        let mut wx = vec![Cplx::new(0.0, 0.0); n * n];
        let mut wy = vec![Cplx::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                let k2 = k[i] * k[i] + k[j] * k[j];
                // ψ̂ = ω̂ / k²  (from ∇²ψ = −ω ⇒ −k²ψ̂ = −ω̂)
                let psi = if k2 == 0.0 { Cplx::new(0.0, 0.0) } else { w_hat[idx].mul(Cplx::new(1.0 / k2, 0.0)) };
                // û = i k_y ψ̂ ; v̂ = −i k_x ψ̂
                u[idx] = psi.mul(Cplx::new(0.0, k[j]));
                v[idx] = psi.mul(Cplx::new(0.0, -k[i]));
                // ω̂_x = i k_x ω̂ ; ω̂_y = i k_y ω̂
                wx[idx] = w_hat[idx].mul(Cplx::new(0.0, k[i]));
                wy[idx] = w_hat[idx].mul(Cplx::new(0.0, k[j]));
            }
        }
        fft2(&mut u, n, true);
        fft2(&mut v, n, true);
        fft2(&mut wx, n, true);
        fft2(&mut wy, n, true);
        // physical-space product u·∇ω
        let mut nl: Vec<Cplx> = (0..n * n).map(|i| Cplx::new(u[i].re * wx[i].re + v[i].re * wy[i].re, 0.0)).collect();
        fft2(&mut nl, n, false);
        // N̂ = −(dealiased advection)
        for i in 0..n * n {
            if self.dealias[i] {
                nl[i] = Cplx::new(-nl[i].re, -nl[i].im);
            } else {
                nl[i] = Cplx::new(0.0, 0.0);
            }
        }
        nl
    }

    /// Integrating factor `e^{−ν k² dt}` per mode.
    fn if_factor(&self, dt: f64) -> Vec<f64> {
        let (n, k) = (self.n, &self.kx);
        let mut e = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                e[i * n + j] = (-self.nu * (k[i] * k[i] + k[j] * k[j]) * dt).exp();
            }
        }
        e
    }

    /// One IF-RK2 step: viscosity exact via the integrating factor, nonlinear term 2nd-order.
    #[allow(clippy::needless_range_loop)] // per-mode update over coupled spectral arrays
    pub fn step(&mut self, dt: f64) {
        let e = self.if_factor(dt);
        let n1 = self.nonlinear(&self.w_hat);
        // predictor: ŵ* = e·(ŵ + dt·N1)
        let mut w_star = vec![Cplx::new(0.0, 0.0); self.n * self.n];
        for i in 0..self.n * self.n {
            let p = self.w_hat[i].add(n1[i].mul(Cplx::new(dt, 0.0)));
            w_star[i] = p.mul(Cplx::new(e[i], 0.0));
        }
        let n2 = self.nonlinear(&w_star);
        // corrector: ŵⁿ⁺¹ = e·ŵ + (dt/2)(e·N1 + N2)
        for i in 0..self.n * self.n {
            let ew = self.w_hat[i].mul(Cplx::new(e[i], 0.0));
            let en1 = n1[i].mul(Cplx::new(e[i], 0.0));
            let corr = en1.add(n2[i]).mul(Cplx::new(dt / 2.0, 0.0));
            self.w_hat[i] = ew.add(corr);
        }
    }

    /// Current vorticity field in physical space (row-major `n×n`).
    pub fn vorticity(&self) -> Vec<f64> {
        let mut a = self.w_hat.clone();
        fft2(&mut a, self.n, true);
        a.iter().map(|c| c.re).collect()
    }

    /// Enstrophy `½Σω²` (a decay monitor).
    pub fn enstrophy(&self) -> f64 {
        self.vorticity().iter().map(|w| 0.5 * w * w).sum()
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use std::f64::consts::PI;

    /// Taylor–Green decay: the exact solution the spectral solver must reproduce to round-off, since
    /// its nonlinear term is identically zero. `ω = 2k sin(kx)sin(ky) e^{−2νk²t}`.
    #[test]
    fn taylor_green_decays_exactly() {
        let n = 32;
        let l = 2.0 * PI;
        let k = 2.0 * PI / l; // one wavelength across the box → k = 1
        let nu = 0.05;
        let omega0: Vec<f64> = {
            let mut v = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let (x, y) = (i as f64 * l / n as f64, j as f64 * l / n as f64);
                    v[i * n + j] = 2.0 * k * (k * x).sin() * (k * y).sin();
                }
            }
            v
        };
        let mut ns = SpectralNs::new(&omega0, n, l, nu);
        let dt = 0.01;
        let steps = 100;
        for _ in 0..steps {
            ns.step(dt);
        }
        let t = dt * steps as f64;
        let decay = (-2.0 * nu * k * k * t).exp();
        let w = ns.vorticity();
        let mut err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let (x, y) = (i as f64 * l / n as f64, j as f64 * l / n as f64);
                let exact = 2.0 * k * (k * x).sin() * (k * y).sin() * decay;
                err = err.max((w[i * n + j] - exact).abs());
            }
        }
        eprintln!("spectral NS Taylor–Green: max error {err:.2e} at t={t} (decay {decay:.4})");
        assert!(err < 1e-9, "spectral NS did not reproduce TG decay: {err}");
    }

    /// **The advection term, against an analytic reference, because nothing else here touches it.**
    ///
    /// ⛔ **Both verification tests above use the single-mode Taylor–Green field, whose nonlinear term is
    /// identically zero** — the doc on the first one says so itself. Every assertion in them is satisfied
    /// by the integrating factor `e^{−νk²dt}` alone, so replacing the physical-space product `u·∇ω` with
    /// ZERO, turning this Navier–Stokes solver into a decoupled heat equation, passed both. Worse, all
    /// four active modes of that field share `|k|² = 2k²`, so scaling `ψ̂` by any function of `k²` keeps
    /// the cancellation exact and even a wrong Poisson solve passes. The two `dmd.rs` tests do drive this
    /// solver with a two-mode field, but they compare DMD against the solver's own snapshots, which is the
    /// solver against itself.
    ///
    /// This uses a stream function with two modes at different `|k|`, so nothing cancels, and compares one
    /// step at `ν = 0` against `−u·∇ω` derived analytically from that stream function rather than from any
    /// code under test:
    ///
    /// ```text
    ///   ψ  = A sin x sin y + B sin 2x cos y
    ///   ω  = −∇²ψ = 2A sin x sin y + 5B sin 2x cos y
    ///   u  =  ∂ψ/∂y =  A sin x cos y − B sin 2x sin y
    ///   v  = −∂ψ/∂x = −A cos x sin y − 2B cos 2x cos y
    /// ```
    ///
    /// The residual must be first order in `dt`, since IF-RK2 gives `(ω₁−ω₀)/dt = N(ω₀) + O(dt)`. Checking
    /// the ORDER is what separates "the formula agrees" from "the tolerance is loose".
    #[test]
    fn the_advection_term_matches_an_analytic_two_mode_reference() {
        let n = 64;
        let l = 2.0 * PI;
        let (aa, bb) = (1.0f64, 0.7f64);
        let at = |i: usize, j: usize| (i as f64 * l / n as f64, j as f64 * l / n as f64);
        let omega = |x: f64, y: f64| 2.0 * aa * x.sin() * y.sin() + 5.0 * bb * (2.0 * x).sin() * y.cos();
        // −u·∇ω, the right-hand side at ν = 0
        let rhs = |x: f64, y: f64| {
            let u = aa * x.sin() * y.cos() - bb * (2.0 * x).sin() * y.sin();
            let v = -aa * x.cos() * y.sin() - 2.0 * bb * (2.0 * x).cos() * y.cos();
            let wx = 2.0 * aa * x.cos() * y.sin() + 10.0 * bb * (2.0 * x).cos() * y.cos();
            let wy = 2.0 * aa * x.sin() * y.cos() - 5.0 * bb * (2.0 * x).sin() * y.sin();
            -(u * wx + v * wy)
        };

        let mut w0 = vec![0.0; n * n];
        let mut exact = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let (x, y) = at(i, j);
                w0[i * n + j] = omega(x, y);
                exact[i * n + j] = rhs(x, y);
            }
        }
        let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 1.0, "this field must have a substantial advection term, max |u·∇ω| = {scale}");

        // relative residual of one inviscid step against the analytic right-hand side
        let residual = |dt: f64| -> f64 {
            let mut ns = SpectralNs::new(&w0, n, l, 0.0);
            ns.step(dt);
            let w1 = ns.vorticity();
            (0..n * n).fold(0.0f64, |m, i| m.max(((w1[i] - w0[i]) / dt - exact[i]).abs())) / scale
        };
        let (r_coarse, r_fine) = (residual(1e-3), residual(1e-4));
        eprintln!("  advection vs analytic: max |u·∇ω| = {scale:.3}, residual {r_coarse:.3e} at dt=1e-3, {r_fine:.3e} at dt=1e-4 (ratio {:.2})", r_coarse / r_fine);
        assert!(r_fine < 1e-2, "one inviscid step must reproduce the analytic advection: {r_fine:.3e}");
        let order = r_coarse / r_fine;
        assert!((5.0..20.0).contains(&order), "the residual must be FIRST ORDER in dt, ratio was {order:.2} for a 10x step reduction");

        // And 2-D Euler advection conserves enstrophy while moving the field a long way, which is the
        // pair of facts a zeroed advection term cannot produce: it would conserve enstrophy by doing
        // nothing at all.
        let mut ns = SpectralNs::new(&w0, n, l, 0.0);
        let z0 = ns.enstrophy();
        for _ in 0..2000 {
            ns.step(1e-3);
        }
        let w = ns.vorticity();
        let drift = (ns.enstrophy() / z0 - 1.0).abs();
        let moved = (0..n * n).fold(0.0f64, |m, i| m.max((w[i] - w0[i]).abs())) / scale;
        eprintln!("  inviscid rollout: enstrophy drift {drift:.2e} over 2 s, field moved {moved:.3} of the advection scale");
        assert!(drift < 1e-3, "inviscid 2-D advection must conserve enstrophy, drifted {drift:.2e}");
        assert!(moved > 0.1, "the field must actually be advected, it moved only {moved:.3e}");
    }

    /// Enstrophy decays monotonically (viscous dissipation) and tracks the analytic `e^{−4νk²t}`
    /// rate (enstrophy ∝ ω²).
    #[test]
    fn enstrophy_decays_at_the_analytic_rate() {
        let n = 32;
        let l = 2.0 * PI;
        let (k, nu) = (1.0, 0.05);
        let omega0: Vec<f64> = {
            let mut v = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let (x, y) = (i as f64 * l / n as f64, j as f64 * l / n as f64);
                    v[i * n + j] = 2.0 * k * (k * x).sin() * (k * y).sin();
                }
            }
            v
        };
        let mut ns = SpectralNs::new(&omega0, n, l, nu);
        let e0 = ns.enstrophy();
        let dt = 0.01;
        let steps = 50;
        for _ in 0..steps {
            ns.step(dt);
        }
        let t = dt * steps as f64;
        let ratio = ns.enstrophy() / e0;
        let expected = (-4.0 * nu * k * k * t).exp();
        eprintln!("spectral NS enstrophy ratio {ratio:.5} vs analytic {expected:.5}");
        assert!((ratio - expected).abs() / expected < 1e-6, "enstrophy decay rate off: {ratio} vs {expected}");
    }
}
