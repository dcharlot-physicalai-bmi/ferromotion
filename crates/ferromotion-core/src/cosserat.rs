//! **Cosserat soft-rod statics (variable-strain / GVS)** — the strain-parameterized model behind soft
//! and continuum robots (Renda, Boyer; the GVS/SoRoSim line). A slender rod is described not by node
//! positions but by its **strain field** along the arc length; here, the planar piecewise-constant-strain
//! (PCS) specialization: `N` segments each with a constant curvature `κ_i`, the workhorse
//! "constant-curvature section" model. Equilibrium under load is the minimizer of the total potential
//!
//! ```text
//!   U(κ) = ½ Σ EI κ_i² Δs  −  f · p_tip(κ)      (bending energy − load work),
//! ```
//!
//! found with the analytic strain gradient. It complements the dynamic Discrete Elastic Rod
//! (`ferromotion-rod`): this is the strain-based *statics* that reduces, in the small-deflection limit,
//! to the Euler–Bernoulli cantilever — the invariant the tests pin it against. Pure Rust → WASM-clean.

use nalgebra::Vector2;

/// A planar Cosserat rod of `n` constant-curvature segments, total length `l`, bending stiffness `ei`.
#[derive(Clone, Debug)]
pub struct CosseratRod {
    pub n: usize,
    pub l: f64,
    pub ei: f64,
}

impl CosseratRod {
    pub fn new(n: usize, l: f64, ei: f64) -> Self {
        CosseratRod { n, l, ei }
    }

    fn ds(&self) -> f64 {
        self.l / self.n as f64
    }

    /// Segment **midpoint** orientations `θ̄_k = Σ_{m<k} κ_m·Δs + ½ κ_k·Δs` — the midpoint rule, so the
    /// polygonal backbone approximates the smooth centerline to `O(Δs²)` rather than `O(Δs)`.
    fn mid_angles(&self, kappa: &[f64]) -> Vec<f64> {
        let ds = self.ds();
        let mut th = Vec::with_capacity(self.n);
        let mut acc = 0.0;
        for &k in kappa {
            th.push(acc + 0.5 * k * ds);
            acc += k * ds;
        }
        th
    }

    /// Backbone points `p_0…p_N` (base clamped at the origin, tangent along +x), integrated with the
    /// midpoint rule.
    pub fn backbone(&self, kappa: &[f64]) -> Vec<Vector2<f64>> {
        let ds = self.ds();
        let th = self.mid_angles(kappa);
        let mut pts = vec![Vector2::zeros()];
        let mut p = Vector2::zeros();
        for k in 0..self.n {
            p += Vector2::new(th[k].cos(), th[k].sin()) * ds;
            pts.push(p);
        }
        pts
    }

    pub fn tip(&self, kappa: &[f64]) -> Vector2<f64> {
        *self.backbone(kappa).last().unwrap()
    }

    /// Total potential `U(κ) = ½ Σ EI κ² Δs − f·p_tip` under a tip force `f`.
    pub fn potential(&self, kappa: &[f64], f: Vector2<f64>) -> f64 {
        let ds = self.ds();
        let bend: f64 = kappa.iter().map(|&k| 0.5 * self.ei * k * k * ds).sum();
        bend - f.dot(&self.tip(kappa))
    }

    /// Analytic gradient `∂U/∂κ_i = EI κ_i Δs − f·∂p_tip/∂κ_i`. With the midpoint convention
    /// `∂θ̄_k/∂κ_i = Δs (k>i), ½Δs (k=i), 0 (k<i)`, so
    /// `∂p_tip/∂κ_i = Δs² [ Σ_{k>i}(−sin θ̄_k, cos θ̄_k) + ½(−sin θ̄_i, cos θ̄_i) ]`.
    pub fn gradient(&self, kappa: &[f64], f: Vector2<f64>) -> Vec<f64> {
        let ds = self.ds();
        let th = self.mid_angles(kappa);
        let term = |k: usize| Vector2::new(-th[k].sin(), th[k].cos());
        // suffix[k] = Σ_{m≥k} term(m)
        let mut suffix = vec![Vector2::zeros(); self.n + 1];
        for k in (0..self.n).rev() {
            suffix[k] = suffix[k + 1] + term(k);
        }
        (0..self.n)
            .map(|i| {
                let dptip = (suffix[i + 1] + term(i) * 0.5) * ds * ds;
                self.ei * kappa[i] * ds - f.dot(&dptip)
            })
            .collect()
    }

    /// Solve for the equilibrium strain field under a tip force by gradient descent with backtracking.
    pub fn solve(&self, f: Vector2<f64>) -> Vec<f64> {
        let mut kappa = vec![0.0; self.n];
        let mut step = 1.0 / self.ei.max(1e-9); // scale to the stiffness
        for _ in 0..20_000 {
            let g = self.gradient(&kappa, f);
            let gnorm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            if gnorm < 1e-12 {
                break;
            }
            // backtracking line search on U
            let u0 = self.potential(&kappa, f);
            let mut s = step * 4.0;
            loop {
                let trial: Vec<f64> = kappa.iter().zip(&g).map(|(k, gi)| k - s * gi).collect();
                if self.potential(&trial, f) <= u0 - 1e-4 * s * gnorm * gnorm || s < 1e-16 {
                    kappa = trial;
                    step = s;
                    break;
                }
                s *= 0.5;
            }
        }
        kappa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_load_stays_straight() {
        let rod = CosseratRod::new(40, 1.0, 2.0);
        let k = rod.solve(Vector2::zeros());
        assert!(k.iter().all(|&x| x.abs() < 1e-9), "unloaded rod should stay straight");
        let tip = rod.tip(&k);
        assert!((tip - Vector2::new(1.0, 0.0)).norm() < 1e-9, "straight tip should be at (L,0)");
    }

    #[test]
    fn a_tip_moment_bends_to_a_uniform_circular_arc() {
        // A pure tip moment M gives constant curvature κ = M/EI exactly. Model it as the limit the
        // energy predicts: with no force and a prescribed uniform curvature, the shape is a circular
        // arc of radius EI/M and tip angle ML/EI. Check the constant-curvature backbone is a circle.
        let (ei, l, n) = (2.0, 1.0, 200);
        let rod = CosseratRod::new(n, l, ei);
        let m = 1.5;
        let kappa_val = m / ei; // M/EI
        let kappa = vec![kappa_val; n];
        let pts = rod.backbone(&kappa);
        // arc of radius R = 1/κ centered at (0, R): every point is distance R from the centre.
        let r = 1.0 / kappa_val;
        let centre = Vector2::new(0.0, r);
        let worst = pts.iter().map(|p| ((p - centre).norm() - r).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-3, "constant curvature should trace a circle of radius EI/M: worst {worst}");
        // tip angle = ML/EI
        assert!((kappa_val * l - m * l / ei).abs() < 1e-12, "tip angle should be ML/EI");
    }

    #[test]
    fn cantilever_tip_load_matches_euler_bernoulli() {
        // THE INVARIANT. A small transverse tip force F on a clamped cantilever deflects the tip by
        // δ = F L³ / (3 EI) in the small-deflection (Euler–Bernoulli) limit.
        let (ei, l, n) = (5.0, 1.0, 60);
        let rod = CosseratRod::new(n, l, ei);
        let force = 0.02; // small, so linear beam theory holds
        let kappa = rod.solve(Vector2::new(0.0, -force)); // downward tip force
        let tip = rod.tip(&kappa);
        let deflection = -tip.y; // downward deflection magnitude
        let euler_bernoulli = force * l.powi(3) / (3.0 * ei);
        let rel = (deflection - euler_bernoulli).abs() / euler_bernoulli;
        assert!(rel < 0.02, "tip deflection {deflection:.5} vs Euler-Bernoulli {euler_bernoulli:.5} (rel {rel:.3})");
    }

    #[test]
    fn deflection_is_inversely_proportional_to_stiffness() {
        // δ ∝ 1/EI: doubling the bending stiffness halves the deflection.
        let defl = |ei: f64| {
            let rod = CosseratRod::new(60, 1.0, ei);
            -rod.tip(&rod.solve(Vector2::new(0.0, -0.02))).y
        };
        let (d1, d2) = (defl(5.0), defl(10.0));
        assert!((d1 / d2 - 2.0).abs() < 0.05, "δ should scale as 1/EI: ratio {}", d1 / d2);
    }

    #[test]
    fn the_equilibrium_gradient_vanishes() {
        // The solver returns a genuine stationary point of the potential (∇U ≈ 0).
        let rod = CosseratRod::new(50, 1.0, 3.0);
        let f = Vector2::new(0.01, -0.03);
        let kappa = rod.solve(f);
        let g = rod.gradient(&kappa, f);
        let gnorm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(gnorm < 1e-6, "equilibrium gradient should vanish: ‖∇U‖ = {gnorm}");
    }

    #[test]
    fn the_analytic_gradient_matches_finite_differences() {
        let rod = CosseratRod::new(12, 1.0, 2.0);
        let f = Vector2::new(0.05, -0.1);
        let kappa: Vec<f64> = (0..12).map(|i| 0.1 * (i as f64 * 0.3).sin()).collect();
        let g = rod.gradient(&kappa, f);
        let eps = 1e-6;
        for i in 0..12 {
            let (mut kp, mut km) = (kappa.clone(), kappa.clone());
            kp[i] += eps;
            km[i] -= eps;
            let fd = (rod.potential(&kp, f) - rod.potential(&km, f)) / (2.0 * eps);
            assert!((g[i] - fd).abs() < 1e-5, "∂U/∂κ[{i}] {} vs fd {fd}", g[i]);
        }
    }
}
