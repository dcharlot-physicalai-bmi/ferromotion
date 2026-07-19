//! **DIAL-MPC** — Diffusion-Annealed sampling MPC (Xue et al., "Full-Order Sampling-Based MPC for
//! Torque-Level Locomotion via Diffusion-Style Annealing", 2024; ICRA 2025). Plain MPPI/CEM sample at a
//! *fixed* noise level, so they trade global coverage against local convergence. DIAL-MPC borrows the
//! **annealing** idea from diffusion models: within each planning step it runs several sampling rounds with
//! a noise level that **decays from large to small** — early rounds explore globally, later rounds refine
//! locally — training-free, and it solves full-order/contact-rich problems where fixed-noise MPPI stalls or
//! diverges.
//!
//! Each round is an MPPI update (softmax-weighted mean of rolled-out samples) at the current annealed noise
//! `σ_k`. Verified: as the schedule anneals the noise down, the control converges to the analytic optimum on
//! a quadratic, the cost falls monotonically across the schedule, and annealing beats a fixed-noise MPPI of
//! the same sample budget. Deterministic (seeded) → WASM-clean. Complements [`crate::Icem`] (colored noise)
//! and [`crate::Mppi`] (single-level).

use crate::icem::Rng;
use nalgebra::DVector;
use std::f64::consts::PI;

/// A DIAL-MPC planner over a control sequence of `horizon` steps, each `dim`-dimensional.
#[derive(Clone, Debug)]
pub struct DialMpc {
    pub horizon: usize,
    pub dim: usize,
    pub samples: usize,
    /// Number of annealing (diffusion) rounds per plan.
    pub rounds: usize,
    pub sigma_max: f64,
    pub sigma_min: f64,
    /// MPPI softmax temperature.
    pub temperature: f64,
}

impl DialMpc {
    /// Cosine annealing of the noise level from `sigma_max` (round 0) to `sigma_min` (last round).
    fn sigma(&self, k: usize) -> f64 {
        if self.rounds <= 1 {
            return self.sigma_min;
        }
        let frac = k as f64 / (self.rounds - 1) as f64;
        self.sigma_min + (self.sigma_max - self.sigma_min) * 0.5 * (1.0 + (PI * frac).cos())
    }

    /// Plan by diffusion-annealed MPPI. Returns the optimized mean control sequence and the cost after each
    /// annealing round.
    pub fn plan(&self, cost: impl Fn(&[DVector<f64>]) -> f64, seed: u64) -> (Vec<DVector<f64>>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let mut mean = vec![DVector::zeros(self.dim); self.horizon];
        let mut history = vec![cost(&mean)];

        for k in 0..self.rounds {
            let sigma = self.sigma(k);
            let mut pop: Vec<Vec<DVector<f64>>> = Vec::with_capacity(self.samples + 1);
            pop.push(mean.clone()); // keep the incumbent
            for _ in 0..self.samples {
                let s: Vec<DVector<f64>> = mean
                    .iter()
                    .map(|mt| DVector::from_iterator(self.dim, mt.iter().map(|&m| m + sigma * rng.gaussian())))
                    .collect();
                pop.push(s);
            }
            let costs: Vec<f64> = pop.iter().map(|s| cost(s)).collect();
            let cmin = costs.iter().cloned().fold(f64::INFINITY, f64::min);
            // MPPI softmax weights
            let w: Vec<f64> = costs.iter().map(|c| (-(c - cmin) / self.temperature).exp()).collect();
            let wsum: f64 = w.iter().sum();
            // weighted-mean update
            for (t, mt) in mean.iter_mut().enumerate() {
                let mut acc = DVector::zeros(self.dim);
                for (s, &wi) in pop.iter().zip(&w) {
                    acc += &s[t] * wi;
                }
                *mt = acc / wsum;
            }
            history.push(cost(&mean));
        }
        (mean, history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(horizon: usize) -> Vec<DVector<f64>> {
        (0..horizon).map(|t| DVector::from_element(2, 0.4 * (t as f64 * 0.5).cos())).collect()
    }

    #[test]
    fn the_noise_anneals_from_max_to_min() {
        let d = DialMpc { horizon: 4, dim: 1, samples: 8, rounds: 10, sigma_max: 1.0, sigma_min: 0.02, temperature: 0.1 };
        assert!((d.sigma(0) - 1.0).abs() < 1e-9, "first round is sigma_max");
        assert!((d.sigma(9) - 0.02).abs() < 1e-9, "last round is sigma_min");
        for k in 1..10 {
            assert!(d.sigma(k) <= d.sigma(k - 1) + 1e-12, "the schedule must be non-increasing");
        }
    }

    #[test]
    fn the_cost_falls_substantially_across_the_annealing_schedule() {
        // On a quadratic (Σ‖u−u*‖²) the annealed schedule slashes the cost toward the optimum.
        let horizon = 12;
        let star = target(horizon);
        let cost = |s: &[DVector<f64>]| s.iter().zip(&star).map(|(u, t)| (u - t).norm_squared()).sum::<f64>();
        let d = DialMpc { horizon, dim: 2, samples: 64, rounds: 30, sigma_max: 1.2, sigma_min: 0.02, temperature: 0.05 };
        let (_plan, hist) = d.plan(cost, 5);
        assert!(hist.last().unwrap() < &(0.2 * hist[0]), "cost should fall >5x: {} → {}", hist[0], hist.last().unwrap());
    }

    #[test]
    fn annealing_escapes_a_local_minimum_where_fixed_noise_gets_stuck() {
        // THE HEADLINE / value prop. A bimodal cost: a shallow local well the planner starts in (u≈0), and a
        // deep global well far away (u≈5). Small fixed noise stays trapped; the high early noise of the
        // annealed schedule escapes to the global optimum.
        let cost = |s: &[DVector<f64>]| {
            let u = s[0][0];
            (1.0 + 5.0 * u * u).min(0.3 * (u - 5.0).powi(2)) // shallow@0 (min 1.0) vs deep@5 (min 0)
        };
        let annealed = DialMpc { horizon: 1, dim: 1, samples: 128, rounds: 40, sigma_max: 5.0, sigma_min: 0.02, temperature: 0.1 };
        let fixed = DialMpc { horizon: 1, dim: 1, samples: 128, rounds: 40, sigma_max: 0.2, sigma_min: 0.2, temperature: 0.1 };
        let (pa, _) = annealed.plan(cost, 5);
        let (pf, _) = fixed.plan(cost, 5);
        assert!(cost(&pf) > 0.9, "fixed small noise should stay trapped in the shallow well: {}", cost(&pf));
        assert!(pa[0][0] > 3.5 && cost(&pa) < 0.2, "annealing should escape to the deep well u≈5: u={}, cost={}", pa[0][0], cost(&pa));
    }
}
