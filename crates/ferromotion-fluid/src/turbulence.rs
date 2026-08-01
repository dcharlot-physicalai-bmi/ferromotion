//! **Turbulence closures** — the eddy-viscosity models that let the momentum equation carry the
//! unresolved turbulent stresses instead of resolving every eddy. Two families, matching the standard
//! CFD stack (and XCALibre's): an algebraic **LES Smagorinsky** subgrid model, and a two-equation
//! **RANS k-ω** (Wilcox) model. Both produce a turbulent (eddy) viscosity `νₜ` that a solver adds to
//! the molecular viscosity in its diffusion term, `ν_eff = ν + νₜ`.
//!
//! Fields are on a collocated 2-D grid, row-major with `x` fastest: `idx(i, j) = j·nx + i`, uniform
//! spacing `h`. The models are verified against analytic oracles: the strain-rate magnitude of a pure
//! shear, the vanishing of `νₜ` under rigid rotation / uniform flow (only the symmetric strain drives
//! it, not vorticity), Galilean invariance, and the closed-form free decay of homogeneous k-ω. Pure
//! Rust → WASM-clean.

/// Velocity gradients `(∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)` at cell `(i, j)` by central differences with a
/// clamped stencil at the boundary (exact for the affine test fields).
#[inline]
fn grads(u: &[f64], v: &[f64], i: usize, j: usize, nx: usize, ny: usize, h: f64) -> (f64, f64, f64, f64) {
    let idx = |i: usize, j: usize| j * nx + i;
    let (ip, im) = ((i + 1).min(nx - 1), i.saturating_sub(1));
    let (jp, jm) = ((j + 1).min(ny - 1), j.saturating_sub(1));
    let sx = (ip - im) as f64 * h;
    let sy = (jp - jm) as f64 * h;
    let du_dx = (u[idx(ip, j)] - u[idx(im, j)]) / sx;
    let dv_dx = (v[idx(ip, j)] - v[idx(im, j)]) / sx;
    let du_dy = (u[idx(i, jp)] - u[idx(i, jm)]) / sy;
    let dv_dy = (v[idx(i, jp)] - v[idx(i, jm)]) / sy;
    (du_dx, du_dy, dv_dx, dv_dy)
}

/// `‖S‖² = 2·S_ij·S_ij` at cell `(i, j)`, the strain-rate invariant that drives both closures, with
/// `S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)`.
#[inline]
pub fn strain_sq(u: &[f64], v: &[f64], i: usize, j: usize, nx: usize, ny: usize, h: f64) -> f64 {
    let (du_dx, du_dy, dv_dx, dv_dy) = grads(u, v, i, j, nx, ny, h);
    2.0 * du_dx * du_dx + 2.0 * dv_dy * dv_dy + (du_dy + dv_dx).powi(2)
}

/// Strain-rate magnitude `‖S‖ = √(2 S_ij S_ij)` field over the whole grid.
pub fn strain_rate_magnitude(u: &[f64], v: &[f64], nx: usize, ny: usize, h: f64) -> Vec<f64> {
    let mut s = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            s[j * nx + i] = strain_sq(u, v, i, j, nx, ny, h).sqrt();
        }
    }
    s
}

/// **LES Smagorinsky** subgrid eddy viscosity `νₜ = (C_s·Δ)²·‖S‖`, filter width `Δ = h`. `cs ≈ 0.1–0.17`
/// (0.17 is the classic homogeneous-isotropic value; 0.1 is typical for shear/channel flow). Zero
/// wherever the flow has no strain (uniform translation or rigid rotation) — the algebraic closure.
pub fn smagorinsky_eddy_viscosity(u: &[f64], v: &[f64], nx: usize, ny: usize, h: f64, cs: f64) -> Vec<f64> {
    let cd2 = (cs * h).powi(2);
    let mut nut = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            nut[j * nx + i] = cd2 * strain_sq(u, v, i, j, nx, ny, h).sqrt();
        }
    }
    nut
}

/// **RANS k-ω** (Wilcox) two-equation turbulence model on a periodic collocated grid. Transports the
/// turbulent kinetic energy `k` and the specific dissipation rate `ω`; the eddy viscosity is `νₜ = k/ω`.
/// Production `P = νₜ‖S‖²` feeds `k` (and, since `(ω/k)·νₜ = 1`, feeds `ω` as `α‖S‖²`); destruction is
/// `β*·kω` and `β·ω²`. With no gradients and no strain the equations reduce to the closed-form homogeneous
/// free decay — the oracle it is verified against.
#[derive(Clone)]
pub struct KOmega {
    pub nx: usize,
    pub ny: usize,
    pub h: f64,
    /// Molecular (kinematic) viscosity.
    pub nu: f64,
    pub k: Vec<f64>,
    pub w: Vec<f64>,
    // Wilcox (2006) closure coefficients.
    pub beta_star: f64,
    pub alpha: f64,
    pub beta: f64,
    pub sigma: f64,
    pub sigma_star: f64,
}

impl KOmega {
    /// A uniform field initialized to `(k0, w0)`.
    pub fn new(nx: usize, ny: usize, h: f64, nu: f64, k0: f64, w0: f64) -> Self {
        KOmega {
            nx,
            ny,
            h,
            nu,
            k: vec![k0; nx * ny],
            w: vec![w0; nx * ny],
            beta_star: 0.09,
            alpha: 5.0 / 9.0,
            beta: 3.0 / 40.0,
            sigma: 0.5,
            sigma_star: 0.5,
        }
    }

    /// Eddy viscosity `νₜ = k/ω` per cell (floored `ω` for safety).
    pub fn eddy_viscosity(&self) -> Vec<f64> {
        self.k.iter().zip(&self.w).map(|(&k, &w)| k / w.max(1e-10)).collect()
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Advance `k` and `ω` one step `dt` under the velocity field `(u, v)` (advection + production +
    /// destruction + turbulent diffusion). Periodic boundaries; `k`, `ω` floored positive.
    pub fn step(&mut self, u: &[f64], v: &[f64], dt: f64) {
        let (nx, ny, h) = (self.nx, self.ny, self.h);
        let wrap = |a: isize, n: usize| ((a % n as isize + n as isize) % n as isize) as usize;
        let (mut kn, mut wn) = (self.k.clone(), self.w.clone());
        for j in 0..ny {
            for i in 0..nx {
                let c = self.idx(i, j);
                let (ie, iw) = (wrap(i as isize + 1, nx), wrap(i as isize - 1, nx));
                let (jn, js) = (wrap(j as isize + 1, ny), wrap(j as isize - 1, ny));
                let nut = self.k[c] / self.w[c].max(1e-10);
                let s2 = strain_sq(u, v, i, j, nx, ny, h);
                let p = nut * s2; // production of k

                // central advection −U·∇φ
                let ddx = |f: &[f64]| (f[self.idx(ie, j)] - f[self.idx(iw, j)]) / (2.0 * h);
                let ddy = |f: &[f64]| (f[self.idx(i, jn)] - f[self.idx(i, js)]) / (2.0 * h);
                let adv_k = -(u[c] * ddx(&self.k) + v[c] * ddy(&self.k));
                let adv_w = -(u[c] * ddx(&self.w) + v[c] * ddy(&self.w));

                // turbulent diffusion ∇·((ν+σφ·νₜ)∇φ), constant-coefficient-per-cell Laplacian
                let lap = |f: &[f64]| (f[self.idx(ie, j)] + f[self.idx(iw, j)] + f[self.idx(i, jn)] + f[self.idx(i, js)] - 4.0 * f[c]) / (h * h);
                let diff_k = (self.nu + self.sigma_star * nut) * lap(&self.k);
                let diff_w = (self.nu + self.sigma * nut) * lap(&self.w);

                let dk = adv_k + p - self.beta_star * self.k[c] * self.w[c] + diff_k;
                let dw = adv_w + self.alpha * s2 - self.beta * self.w[c] * self.w[c] + diff_w;
                kn[c] = (self.k[c] + dt * dk).max(1e-10);
                wn[c] = (self.w[c] + dt * dw).max(1e-10);
            }
        }
        self.k = kn;
        self.w = wn;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // build a field with a closure f(x,y)
    fn field(nx: usize, ny: usize, h: f64, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        let mut a = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                a[j * nx + i] = f(i as f64 * h, j as f64 * h);
            }
        }
        a
    }

    #[test]
    fn strain_rate_matches_pure_shear() {
        // u = γ·y, v = 0 → ‖S‖ = γ everywhere (the canonical simple shear)
        let (nx, ny, h, gamma) = (12, 12, 0.1, 2.3);
        let u = field(nx, ny, h, |_x, y| gamma * y);
        let v = vec![0.0; nx * ny];
        let s = strain_rate_magnitude(&u, &v, nx, ny, h);
        let worst = s.iter().fold(0.0f64, |m, &si| m.max((si - gamma).abs()));
        eprintln!("pure-shear strain: worst |‖S‖ − γ| {worst:.3e}");
        assert!(worst < 1e-12, "strain-rate magnitude wrong for pure shear: {worst}");
    }

    #[test]
    fn smagorinsky_vanishes_for_rotation_and_translation_and_is_galilean() {
        let (nx, ny, h, cs) = (16usize, 16usize, 0.05f64, 0.17f64);
        let cd2 = (cs * h).powi(2);
        // pure shear u=γy → νₜ = (Cs h)²·γ
        let gamma = 3.1;
        let ush = field(nx, ny, h, |_x, y| gamma * y);
        let zero = vec![0.0; nx * ny];
        let nut = smagorinsky_eddy_viscosity(&ush, &zero, nx, ny, h, cs);
        let worst_shear = nut.iter().fold(0.0f64, |m, &x| m.max((x - cd2 * gamma).abs()));
        // rigid rotation u=−Ωy, v=Ωx → no strain → νₜ = 0
        let om = 1.7;
        let ur = field(nx, ny, h, |_x, y| -om * y);
        let vr = field(nx, ny, h, |x, _y| om * x);
        let nut_rot = smagorinsky_eddy_viscosity(&ur, &vr, nx, ny, h, cs);
        let worst_rot = nut_rot.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        // uniform translation → νₜ = 0
        let nut_uni = smagorinsky_eddy_viscosity(&vec![4.2; nx * ny], &vec![-1.3; nx * ny], nx, ny, h, cs);
        let worst_uni = nut_uni.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        // Galilean invariance: add a constant to the shear → νₜ unchanged
        let ushg = field(nx, ny, h, |_x, y| gamma * y + 5.0);
        let nut_g = smagorinsky_eddy_viscosity(&ushg, &vec![2.0; nx * ny], nx, ny, h, cs);
        let worst_gal = nut.iter().zip(&nut_g).fold(0.0f64, |m, (&a, &b)| m.max((a - b).abs()));
        eprintln!("Smagorinsky: shear err {worst_shear:.2e}, rotation νₜ {worst_rot:.2e}, uniform νₜ {worst_uni:.2e}, Galilean Δ {worst_gal:.2e}");
        assert!(worst_shear < 1e-12, "shear νₜ off: {worst_shear}");
        assert!(worst_rot < 1e-12, "rotation should give zero eddy viscosity: {worst_rot}");
        assert!(worst_uni < 1e-12, "uniform flow should give zero eddy viscosity: {worst_uni}");
        assert!(worst_gal < 1e-12, "eddy viscosity not Galilean invariant: {worst_gal}");
    }

    #[test]
    fn komega_free_decay_matches_analytic() {
        // homogeneous, quiescent (U=0), uniform (k,ω): dk/dt=−β*kω, dω/dt=−βω² has the closed form
        // ω(t)=ω₀/(1+βω₀t), k(t)=k₀(1+βω₀t)^(−β*/β). Verify the source terms reproduce it.
        let (nx, ny, h) = (8, 8, 0.1);
        let (k0, w0) = (0.5, 10.0);
        let mut m = KOmega::new(nx, ny, h, 1e-3, k0, w0);
        let (zero, dt, steps) = (vec![0.0; nx * ny], 1e-4, 20000);
        for _ in 0..steps {
            m.step(&zero, &zero, dt);
        }
        let t = dt * steps as f64;
        let w_exact = w0 / (1.0 + m.beta * w0 * t);
        let k_exact = k0 * (1.0 + m.beta * w0 * t).powf(-m.beta_star / m.beta);
        let (kc, wc) = (m.k[0], m.w[0]);
        let rel_k = (kc - k_exact).abs() / k_exact;
        let rel_w = (wc - w_exact).abs() / w_exact;
        // uniformity preserved (no spurious spatial variation)
        let kvar = m.k.iter().fold(0.0f64, |a, &x| a.max((x - kc).abs()));
        eprintln!("k-ω free decay @ t={t:.2}: k {kc:.5} vs {k_exact:.5} (rel {rel_k:.2e}), ω {wc:.4} vs {w_exact:.4} (rel {rel_w:.2e}), spatial var {kvar:.2e}");
        assert!(rel_k < 2e-3, "k free-decay off analytic: {rel_k}");
        assert!(rel_w < 2e-3, "ω free-decay off analytic: {rel_w}");
        assert!(kvar < 1e-12, "homogeneous field developed spatial variation: {kvar}");
    }

    #[test]
    fn komega_produces_turbulence_under_shear() {
        // under sustained mean shear, production is positive → k (and the eddy viscosity) grow from a
        // small seed, and νₜ = k/ω stays positive and finite.
        let (nx, ny, h) = (16, 16, 0.05);
        let gamma = 20.0;
        let u = field(nx, ny, h, |_x, y| gamma * y);
        let v = vec![0.0; nx * ny];
        let mut m = KOmega::new(nx, ny, h, 1e-3, 1e-4, 50.0); // tiny seed of turbulence
        let k_start = m.k[nx / 2 + (ny / 2) * nx];
        for _ in 0..2000 {
            m.step(&u, &v, 1e-4);
        }
        let c = nx / 2 + (ny / 2) * nx;
        let nut = m.eddy_viscosity()[c];
        eprintln!("k-ω under shear: k {:.4e} → {:.4e}, νₜ {:.4e}", k_start, m.k[c], nut);
        assert!(m.k.iter().all(|x| x.is_finite()) && m.w.iter().all(|x| x.is_finite()), "k-ω blew up");
        assert!(m.k[c] > 1.5 * k_start, "shear production did not grow k (production should exceed destruction): {} → {}", k_start, m.k[c]);
        assert!(nut > 0.0 && nut.is_finite(), "eddy viscosity should be positive and finite: {nut}");
    }
}
