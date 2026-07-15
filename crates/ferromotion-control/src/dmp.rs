//! **Dynamic Movement Primitives** (Ijspeert, Schaal et al.) — the workhorse representation for
//! learning a motion from a *single* demonstration and replaying it to new goals.
//!
//! A DMP is a stable spring-damper toward the goal, perturbed by a learned forcing term:
//!
//! ```text
//!   τ ż = α_z(β_z(g − y) − z) + f(x),   τ ẏ = z        (transformation system)
//!   τ ẋ = −α_x x,  x(0) = 1                            (canonical system: phase, not time)
//!   f(x) = [Σ ψ_i(x) w_i / Σ ψ_i(x)] · x · (g − y₀)    (Gaussian basis in phase)
//! ```
//!
//! The design is the point. The forcing term is gated by `x`, which decays to zero — so `f` vanishes
//! and the system *provably* reduces to a stable spring-damper at `g`: **goal convergence is
//! structural, whatever the learned weights say**. Phase (not time) drives the shape, so `τ` rescales
//! duration without touching it; the `(g − y₀)` factor scales it spatially. Weights are fit from one
//! demonstration by locally weighted regression. Pure Rust → WASM-clean.

/// A one-dimensional discrete DMP.
#[derive(Clone, Debug)]
pub struct Dmp {
    pub alpha_z: f64,
    pub beta_z: f64,
    pub alpha_x: f64,
    /// Learned basis weights.
    pub w: Vec<f64>,
    /// Basis centers (in phase) and widths.
    centers: Vec<f64>,
    widths: Vec<f64>,
    /// Demonstration start/goal (used for spatial scaling).
    pub y0: f64,
    pub g: f64,
    pub tau: f64,
}

impl Dmp {
    /// A DMP with `n_basis` Gaussians, laid out at evenly-spaced *times* mapped into phase.
    pub fn new(n_basis: usize) -> Self {
        let alpha_x = 1.0;
        let centers: Vec<f64> = (0..n_basis).map(|i| (-alpha_x * i as f64 / (n_basis - 1).max(1) as f64).exp()).collect();
        // Widths so neighbours overlap.
        let mut widths = vec![0.0; n_basis];
        for i in 0..n_basis {
            let d = if i + 1 < n_basis { centers[i + 1] - centers[i] } else { centers[i] - centers[i - 1] };
            widths[i] = 1.0 / (0.65 * d).powi(2);
        }
        Self { alpha_z: 25.0, beta_z: 25.0 / 4.0, alpha_x, w: vec![0.0; n_basis], centers, widths, y0: 0.0, g: 1.0, tau: 1.0 }
    }

    /// Normalized Gaussian basis activations at phase `x`.
    fn psi(&self, x: f64) -> Vec<f64> {
        (0..self.w.len()).map(|i| (-self.widths[i] * (x - self.centers[i]).powi(2)).exp()).collect()
    }

    fn forcing(&self, x: f64, y0: f64, g: f64) -> f64 {
        let psi = self.psi(x);
        let denom: f64 = psi.iter().sum::<f64>().max(1e-12);
        let num: f64 = psi.iter().zip(&self.w).map(|(p, w)| p * w).sum();
        num / denom * x * (g - y0)
    }

    /// Learn the weights from a position demonstration sampled at `dt` (locally weighted regression).
    pub fn fit(&mut self, demo: &[f64], dt: f64) {
        let n = demo.len();
        self.tau = (n - 1) as f64 * dt;
        self.y0 = demo[0];
        self.g = demo[n - 1];

        // Finite-difference velocity/acceleration of the demonstration.
        let vel: Vec<f64> = (0..n)
            .map(|t| {
                if t == 0 || t == n - 1 { 0.0 } else { (demo[t + 1] - demo[t - 1]) / (2.0 * dt) }
            })
            .collect();
        let acc: Vec<f64> = (0..n)
            .map(|t| {
                if t == 0 || t == n - 1 { 0.0 } else { (vel[t + 1] - vel[t - 1]) / (2.0 * dt) }
            })
            .collect();

        // Target forcing implied by the demo: f = τ²ÿ − α_z(β_z(g−y) − τẏ).
        let scale = self.g - self.y0;
        let (mut num, mut den) = (vec![0.0; self.w.len()], vec![0.0; self.w.len()]);
        for t in 0..n {
            let x = (-self.alpha_x * (t as f64 * dt) / self.tau).exp(); // phase
            let f_target = self.tau * self.tau * acc[t] - self.alpha_z * (self.beta_z * (self.g - demo[t]) - self.tau * vel[t]);
            let xi = x * scale; // the regressor the weights multiply
            for (i, p) in self.psi(x).iter().enumerate() {
                num[i] += p * xi * f_target;
                den[i] += p * xi * xi;
            }
        }
        for i in 0..self.w.len() {
            self.w[i] = if den[i].abs() > 1e-12 { num[i] / den[i] } else { 0.0 };
        }
    }

    /// Integrate the DMP to a (possibly new) start/goal and time constant.
    pub fn rollout(&self, y0: f64, g: f64, tau: f64, dt: f64, steps: usize) -> Vec<f64> {
        let (mut x, mut y, mut z) = (1.0, y0, 0.0);
        let mut out = Vec::with_capacity(steps + 1);
        out.push(y);
        for _ in 0..steps {
            let f = self.forcing(x, y0, g);
            let zd = (self.alpha_z * (self.beta_z * (g - y) - z) + f) / tau;
            z += zd * dt;
            y += (z / tau) * dt;
            x += (-self.alpha_x * x / tau) * dt;
            out.push(y);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A demonstration: a smooth 0→1 move with a bump that vanishes at both ends.
    fn demo(n: usize) -> (Vec<f64>, f64) {
        let t_total = 1.0;
        let dt = t_total / (n - 1) as f64;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let s = i as f64 / (n - 1) as f64;
                3.0 * s * s - 2.0 * s * s * s + 0.2 * (std::f64::consts::TAU * s).sin()
            })
            .collect();
        (y, dt)
    }

    #[test]
    fn reproduces_the_demonstration() {
        let (y, dt) = demo(501);
        let mut d = Dmp::new(30);
        d.fit(&y, dt);
        let out = d.rollout(d.y0, d.g, d.tau, dt, y.len() - 1);
        let err = y.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(err < 0.02, "DMP did not reproduce the demo: max error {err:.4}");
    }

    #[test]
    fn always_converges_to_the_goal_even_a_new_one() {
        // The forcing term is gated by the decaying phase, so the system provably ends at g.
        let (y, dt) = demo(501);
        let mut d = Dmp::new(30);
        d.fit(&y, dt);
        for &g in &[1.0, 2.5, -0.7] {
            // Roll out well past the demo duration so the phase has decayed.
            let out = d.rollout(d.y0, g, d.tau, dt, 2000);
            let end = *out.last().unwrap();
            assert!((end - g).abs() < 1e-3, "did not converge to goal {g}: ended at {end}");
        }
    }

    #[test]
    fn temporal_scaling_preserves_the_shape() {
        // τ rescales duration only: the trajectory in *normalized* time is invariant.
        let (y, dt) = demo(501);
        let mut d = Dmp::new(30);
        d.fit(&y, dt);
        let n = y.len() - 1;
        let fast = d.rollout(d.y0, d.g, d.tau, dt, n);
        let slow = d.rollout(d.y0, d.g, 2.0 * d.tau, dt, 2 * n); // twice as long, same shape
        let mut err = 0.0f64;
        for i in 0..=n {
            err = err.max((fast[i] - slow[2 * i]).abs()); // compare at equal normalized time
        }
        assert!(err < 0.02, "temporal scaling changed the shape: max error {err:.4}");
    }

    #[test]
    fn spatial_scaling_scales_the_shape() {
        // The (g − y₀) factor scales the learned shape with the goal displacement.
        let (y, dt) = demo(501);
        let mut d = Dmp::new(30);
        d.fit(&y, dt);
        let n = y.len() - 1;
        let base = d.rollout(d.y0, d.g, d.tau, dt, n);
        let big = d.rollout(d.y0, d.y0 + 2.0 * (d.g - d.y0), d.tau, dt, n);
        // Peak deviation from the straight start→goal line should roughly double.
        let peak = |tr: &[f64], g: f64| {
            tr.iter().enumerate().map(|(i, v)| (v - (d.y0 + (g - d.y0) * i as f64 / n as f64)).abs()).fold(0.0f64, f64::max)
        };
        let (p1, p2) = (peak(&base, d.g), peak(&big, d.y0 + 2.0 * (d.g - d.y0)));
        assert!(p1 > 1e-3, "baseline has no shape to scale");
        assert!((p2 / p1 - 2.0).abs() < 0.35, "shape did not scale with goal displacement: ratio {:.2}", p2 / p1);
    }
}
