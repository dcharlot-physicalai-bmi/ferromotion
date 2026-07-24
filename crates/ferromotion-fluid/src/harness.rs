//! **The surrogate honesty harness** (Honest Fluids — stage 8, the doctrine flagship). The field's
//! own 2026 consensus is that verification is the weak layer: NVIDIA's surrogate tooling now ships
//! "don't trust the model, test the physics" checks, vanilla PINNs are consensus-unfit for forward
//! turbulence, and FNO spectral bias is a known, actively-patched failure mode. This is that audit
//! as a first-class artifact — a surrogate and its receipts ship together.
//!
//! The point a plain error metric misses: **two predictions can have identical MSE against the
//! ground truth while one is physically valid and the other is a fluent lie.** The harness reads
//! the physics directly — the incompressibility residual (`∇·u`), a spectral-roughness proxy for
//! the high-frequency energy FNO surrogates smear, and the field energy — so it passes the honest
//! prediction and flags the cheat, at equal MSE. Verified below on exactly that adversarial pair.

/// A cell-centered velocity field on an `n × n` periodic grid (unit square).
#[derive(Clone)]
pub struct FlowField {
    pub n: usize,
    pub h: f64,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

impl FlowField {
    /// Sample a continuous field `(x,y) ↦ (u,v)` at cell centers.
    pub fn sample(n: usize, f: impl Fn(f64, f64) -> (f64, f64)) -> FlowField {
        let h = 1.0 / n as f64;
        let (mut u, mut v) = (vec![0.0; n * n], vec![0.0; n * n]);
        for i in 0..n {
            for j in 0..n {
                let (uu, vv) = f((i as f64 + 0.5) * h, (j as f64 + 0.5) * h);
                u[i * n + j] = uu;
                v[i * n + j] = vv;
            }
        }
        FlowField { n, h, u, v }
    }

    #[inline]
    fn ix(&self, i: usize, j: usize) -> usize {
        (i % self.n) * self.n + (j % self.n)
    }

    /// Add another field scaled by `s` (for building perturbed predictions).
    pub fn add_scaled(&mut self, other: &FlowField, s: f64) {
        for k in 0..self.u.len() {
            self.u[k] += s * other.u[k];
            self.v[k] += s * other.v[k];
        }
    }

    /// L2 norm of the field (√Σ(u²+v²)).
    pub fn norm(&self) -> f64 {
        self.u.iter().zip(&self.v).map(|(a, b)| a * a + b * b).sum::<f64>().sqrt()
    }

    /// RMS velocity difference from another field (the plain error metric that can be fooled).
    pub fn rms_diff(&self, other: &FlowField) -> f64 {
        let n = self.u.len();
        let se: f64 = (0..n).map(|k| (self.u[k] - other.u[k]).powi(2) + (self.v[k] - other.v[k]).powi(2)).sum();
        (se / n as f64).sqrt()
    }
}

/// The physics receipts computed on a predicted field. Every incompressible flow must have a small
/// divergence; a smooth resolved flow has bounded roughness. A surrogate that scores well on error
/// but violates these is a fluent lie.
#[derive(Clone, Copy, Debug)]
pub struct Receipts {
    /// RMS of the discrete divergence `∂u/∂x + ∂v/∂y` (incompressibility — 0 for a real flow).
    pub divergence_rms: f64,
    /// Discrete-Laplacian energy normalized by field energy — an FFT-free proxy for high-frequency
    /// (spectral-tail) content; a noise-injecting or ringing surrogate spikes this.
    pub roughness: f64,
    /// Kinetic energy `½Σ(u²+v²)h²`.
    pub energy: f64,
}

/// Compute the receipts for a field.
pub fn audit(f: &FlowField) -> Receipts {
    let (n, h) = (f.n, f.h);
    let (mut sdiv, mut srough, mut su2) = (0.0, 0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            let uc = f.u[i * n + j];
            let vc = f.v[i * n + j];
            let ue = f.u[f.ix(i + 1, j)];
            let uw = f.u[f.ix(i + n - 1, j)];
            let vn = f.v[f.ix(i, j + 1)];
            let vs = f.v[f.ix(i, j + n - 1)];
            let div = (ue - uw) / (2.0 * h) + (vn - vs) / (2.0 * h);
            sdiv += div * div;
            // discrete Laplacian magnitude (roughness numerator)
            let un = f.u[f.ix(i, j + 1)];
            let us = f.u[f.ix(i, j + n - 1)];
            let lap_u = (ue + uw + un + us - 4.0 * uc) / (h * h);
            let vv_e = f.v[f.ix(i + 1, j)];
            let vv_w = f.v[f.ix(i + n - 1, j)];
            let lap_v = (vv_e + vv_w + vn + vs - 4.0 * vc) / (h * h);
            srough += lap_u * lap_u + lap_v * lap_v;
            su2 += uc * uc + vc * vc;
        }
    }
    let m = (n * n) as f64;
    Receipts {
        divergence_rms: (sdiv / m).sqrt(),
        roughness: (srough / m).sqrt() / ((su2 / m).sqrt() + 1e-12),
        energy: 0.5 * su2 * h * h,
    }
}

/// A pass/fail verdict against reference thresholds (calibrated from the ground-truth field).
#[derive(Clone, Copy, Debug)]
pub struct Verdict {
    pub passes: bool,
    pub divergence_rms: f64,
    pub roughness: f64,
}

/// Grade a prediction: it passes only if its physics receipts stay within `tol×` the reference
/// field's own receipts. Error metrics are deliberately NOT consulted — the physics is the judge.
pub fn grade(pred: &FlowField, reference: &Receipts, tol: f64) -> Verdict {
    let r = audit(pred);
    let passes = r.divergence_rms <= reference.divergence_rms.max(1e-9) * tol
        && r.roughness <= reference.roughness.max(1e-9) * tol;
    Verdict { passes, divergence_rms: r.divergence_rms, roughness: r.roughness }
}

#[cfg(test)]
mod verification {
    use super::*;
    use std::f64::consts::PI;

    // A divergence-free Taylor–Green field (the ground truth): u = sin(kx)cos(ky), v = −cos(kx)sin(ky).
    fn truth(n: usize) -> FlowField {
        let k = 2.0 * PI;
        FlowField::sample(n, |x, y| ((k * x).sin() * (k * y).cos(), -(k * x).cos() * (k * y).sin()))
    }

    /// Normalize a perturbation field to a target L2 norm (so competing predictions share an MSE).
    fn scaled_to(mut f: FlowField, target: f64) -> FlowField {
        let s = target / f.norm();
        for k in 0..f.u.len() {
            f.u[k] *= s;
            f.v[k] *= s;
        }
        f
    }

    /// **The core claim: equal MSE, opposite physics — the harness tells them apart.** An honest
    /// prediction (truth + a divergence-free mode) and a cheat (truth + a curl-free gradient field
    /// that injects divergence), scaled to the SAME error budget. MSE cannot distinguish them; the
    /// divergence receipt does.
    #[test]
    fn harness_catches_the_divergence_cheat_at_equal_mse() {
        let n = 64;
        let k = 2.0 * PI;
        let t = truth(n);
        let reference = audit(&t);
        let budget = 0.15 * t.norm();

        // honest perturbation: another divergence-free TG mode
        let honest_p = scaled_to(FlowField::sample(n, |x, y| ((2.0 * k * x).sin() * (2.0 * k * y).cos(), -(2.0 * k * x).cos() * (2.0 * k * y).sin())), budget);
        // cheat perturbation: a pure gradient ∇φ, φ = cos(kx)cos(ky) → (−k sin(kx)cos(ky), −k cos(kx)sin(ky)); all divergence, no curl
        let cheat_p = scaled_to(FlowField::sample(n, |x, y| (-(k * x).sin() * (k * y).cos(), -(k * x).cos() * (k * y).sin())), budget);

        let mut honest = t.clone();
        honest.add_scaled(&honest_p, 1.0);
        let mut cheat = t.clone();
        cheat.add_scaled(&cheat_p, 1.0);

        let mse_h = honest.rms_diff(&t);
        let mse_c = cheat.rms_diff(&t);
        let vh = grade(&honest, &reference, 3.0);
        let vc = grade(&cheat, &reference, 3.0);
        eprintln!("equal-MSE pair: honest MSE {mse_h:.4} div {:.3e} pass={}  |  cheat MSE {mse_c:.4} div {:.3e} pass={}",
            vh.divergence_rms, vh.passes, vc.divergence_rms, vc.passes);

        // Same error budget to within a few percent...
        assert!((mse_h - mse_c).abs() / mse_h < 0.05, "MSEs not comparable: {mse_h} vs {mse_c}");
        // ...yet the harness passes the honest prediction and flags the cheat.
        assert!(vh.passes, "honest prediction wrongly flagged");
        assert!(!vc.passes, "harness FAILED to catch the divergence cheat");
        assert!(vc.divergence_rms > 5.0 * vh.divergence_rms, "cheat's divergence not clearly larger");
    }

    /// The same story for the spectral-bias failure mode: a high-frequency noise cheat has the same
    /// MSE as a smooth honest perturbation, but the roughness receipt spikes.
    #[test]
    fn harness_catches_the_spectral_cheat_at_equal_mse() {
        let n = 64;
        let k = 2.0 * PI;
        let t = truth(n);
        let reference = audit(&t);
        let budget = 0.1 * t.norm();

        let honest_p = scaled_to(FlowField::sample(n, |x, y| ((2.0 * k * x).sin() * (2.0 * k * y).cos(), -(2.0 * k * x).cos() * (2.0 * k * y).sin())), budget);
        // high-frequency cheat: mode 10 — same energy, wildly rougher
        let cheat_p = scaled_to(FlowField::sample(n, |x, y| ((10.0 * k * x).sin() * (10.0 * k * y).cos(), -(10.0 * k * x).cos() * (10.0 * k * y).sin())), budget);

        let mut honest = t.clone();
        honest.add_scaled(&honest_p, 1.0);
        let mut cheat = t.clone();
        cheat.add_scaled(&cheat_p, 1.0);

        let mse_h = honest.rms_diff(&t);
        let mse_c = cheat.rms_diff(&t);
        let vh = grade(&honest, &reference, 3.0);
        let vc = grade(&cheat, &reference, 3.0);
        eprintln!("equal-MSE pair: honest MSE {mse_h:.4} rough {:.2e} pass={}  |  cheat MSE {mse_c:.4} rough {:.2e} pass={}",
            vh.roughness, vh.passes, vc.roughness, vc.passes);
        assert!((mse_h - mse_c).abs() / mse_h < 0.05, "MSEs not comparable");
        assert!(vh.passes, "honest prediction wrongly flagged");
        assert!(!vc.passes, "harness FAILED to catch the spectral cheat");
        assert!(vc.roughness > 5.0 * vh.roughness, "cheat's roughness not clearly larger");
    }
}
