//! TrajectoryBundles (REx Lab) — gradient-free "bundle" trajectory optimization. Around a nominal
//! control sequence we sample `num_samples` perturbations δ, roll each out through a user-supplied
//! discrete dynamics closure, and score it with a user cost closure. From the bundle of
//! (δ, cost) pairs we fit a *local linear model* of cost vs. δ (a least-squares gradient estimate
//! `g ≈ argmin_g Σ (cᵢ − c₀ − gᵀδᵢ)²`) and take a *trust-region* step
//! `Δ = argmin gᵀΔ + λ‖Δ‖²  s.t.  ‖Δ‖∞ ≤ trust` — a small box QP solved via [`crate::qp::solve_box_qp`].
//! The nominal is updated only when the step actually lowers the rolled-out cost, and the trust
//! radius adapts (grow on accept, shrink on reject), so the returned cost history is monotonically
//! non-increasing. Pure Rust + `nalgebra` → WASM-clean; every perturbation comes from a seeded LCG
//! + Box–Muller, so a run is bit-for-bit reproducible in its seed.
//!
//! The optimizer is dynamics-agnostic: `optimize` takes closures `dynamics(state, control) -> next`
//! and `cost(state, control) -> f64`, so it drives any discrete system, not just a robot model.
//! Convention: a rollout accumulates `cost(xₜ, uₜ)` for `t = 0..horizon` and then a terminal
//! `cost(x_horizon, 0)` so the final state is scored.

use nalgebra::{DMatrix, DVector};

/// Gradient-free bundle trajectory optimizer.
pub struct TrajectoryBundle {
    /// Number of control steps in a rollout (states run `x_0 … x_horizon`).
    pub horizon: usize,
    /// Timestep (s) — passed through for the caller's dynamics closure; the optimizer itself is
    /// unit-agnostic and never reads it.
    pub dt: f64,
    /// Width of the control vector at each step.
    pub control_dim: usize,
    /// Number of perturbed control-sequence samples drawn per iteration (the "bundle" size K).
    pub num_samples: usize,
    /// Std-dev of the per-element Gaussian control perturbation.
    pub sigma: f64,
    /// Number of bundle-refinement iterations.
    pub iters: usize,
    /// Trust-region regularization λ on the step (`gᵀΔ + λ‖Δ‖²`); must be > 0 (keeps the QP PD).
    pub lambda: f64,
    /// Initial trust radius: the ∞-norm cap `‖Δ‖∞ ≤ trust` on a single update.
    pub trust: f64,
    /// LCG state; advanced by every draw so the run is fully deterministic in its seed.
    pub rng: u64,
}

/// Ridge added to the least-squares normal equations so the gradient fit stays well-posed.
const FIT_RIDGE: f64 = 1e-6;
/// Trust-radius growth factor on an accepted step and shrink factor on a rejected one.
const TRUST_GROW: f64 = 1.5;
const TRUST_SHRINK: f64 = 0.5;
/// Trust-radius clamp so it neither explodes nor collapses to exactly zero.
const TRUST_MIN: f64 = 1e-8;

impl TrajectoryBundle {
    /// Next uniform sample in `[0, 1)` from the LCG (Knuth/MMIX multiplier & increment).
    fn next_u01(&mut self) -> f64 {
        self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.rng >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// One zero-mean, unit-variance Gaussian draw via Box–Muller.
    fn next_gauss(&mut self) -> f64 {
        let mut u1 = self.next_u01();
        if u1 < 1e-12 {
            u1 = 1e-12; // keep ln() finite
        }
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// Pad/truncate an arbitrary nominal to a flat `horizon * control_dim` control vector.
    fn flat_nominal(&self, nominal: &[Vec<f64>]) -> Vec<f64> {
        let m = self.control_dim;
        let mut out = vec![0.0; self.horizon * m];
        for t in 0..self.horizon {
            for i in 0..m {
                out[t * m + i] = nominal.get(t).and_then(|u| u.get(i)).copied().unwrap_or(0.0);
            }
        }
        out
    }

    /// Accumulated cost of a flat control sequence rolled out from `x0` through `dynamics`, scored
    /// by `cost`. Running cost `cost(xₜ, uₜ)` for `t = 0..horizon`, plus a terminal `cost(x_H, 0)`.
    /// A non-finite state or cost aborts the rollout with `f64::MAX`.
    fn rollout_flat<D, C>(&self, dynamics: &D, cost: &C, x0: &[f64], u: &[f64]) -> f64
    where
        D: Fn(&[f64], &[f64]) -> Vec<f64>,
        C: Fn(&[f64], &[f64]) -> f64,
    {
        let m = self.control_dim;
        let mut x = x0.to_vec();
        let mut total = 0.0;
        for t in 0..self.horizon {
            let ut = &u[t * m..t * m + m];
            let c = cost(&x, ut);
            if !c.is_finite() {
                return f64::MAX;
            }
            total += c;
            x = dynamics(&x, ut);
            if !x.iter().all(|v| v.is_finite()) {
                return f64::MAX;
            }
        }
        let ct = cost(&x, &vec![0.0; m]); // terminal state score (zero control)
        if !ct.is_finite() {
            return f64::MAX;
        }
        total + ct
    }

    /// Rolled-out cost of a `horizon × control_dim` control sequence (public convenience for scoring
    /// a plan, e.g. to compare against the returned `best_controls`).
    pub fn rollout_cost<D, C>(
        &self,
        dynamics: &D,
        cost: &C,
        x0: &[f64],
        controls: &[Vec<f64>],
    ) -> f64
    where
        D: Fn(&[f64], &[f64]) -> Vec<f64>,
        C: Fn(&[f64], &[f64]) -> f64,
    {
        self.rollout_flat(dynamics, cost, x0, &self.flat_nominal(controls))
    }

    /// Optimize a control sequence for the discrete system `dynamics` under `cost`, starting the
    /// initial state at `x0` and warm-starting the plan from `nominal` (padded/truncated to
    /// `horizon × control_dim`; missing entries → 0).
    ///
    /// Returns `(best_controls, cost_history)`: the refined `horizon × control_dim` plan and the
    /// best rolled-out cost after each iteration (length `iters + 1`; `[0]` = the starting nominal's
    /// cost). The history is monotonically non-increasing — a candidate is adopted only if it
    /// lowers the rolled-out cost.
    pub fn optimize<D, C>(
        &mut self,
        dynamics: D,
        cost: C,
        x0: &[f64],
        nominal: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<f64>)
    where
        D: Fn(&[f64], &[f64]) -> Vec<f64>,
        C: Fn(&[f64], &[f64]) -> f64,
    {
        let m = self.control_dim;
        let dim = self.horizon * m; // full control-vector dimension M

        let mut best = self.flat_nominal(nominal);
        let mut best_cost = self.rollout_flat(&dynamics, &cost, x0, &best);
        let mut trust = self.trust.max(TRUST_MIN);
        let lambda = self.lambda.max(1e-9);

        let mut history = Vec::with_capacity(self.iters + 1);
        history.push(best_cost);

        // Trust-region QP Hessian: min gᵀΔ + λ‖Δ‖²  ==  min ½Δᵀ(2λI)Δ + gᵀΔ, so H = 2λI.
        let h = DMatrix::<f64>::identity(dim, dim) * (2.0 * lambda);

        for _ in 0..self.iters {
            // Sample the bundle: K perturbations δ and their rolled-out costs.
            let k = self.num_samples;
            let mut deltas = vec![0.0f64; k * dim]; // row s = δ_s (K × M)
            let mut dc = vec![0.0f64; k]; // cᵢ − c₀
            for s in 0..k {
                let mut cand = best.clone();
                for j in 0..dim {
                    let d = self.sigma * self.next_gauss();
                    deltas[s * dim + j] = d;
                    cand[j] += d;
                }
                let c = self.rollout_flat(&dynamics, &cost, x0, &cand);
                // Guard against a diverged sample poisoning the linear fit.
                dc[s] = if c.is_finite() && best_cost.is_finite() { c - best_cost } else { 0.0 };
            }

            // Least-squares gradient: g = argmin Σ (cᵢ − c₀ − gᵀδᵢ)²  ⇒  (DᵀD + εI) g = Dᵀ Δc.
            let d_mat = DMatrix::from_row_slice(k, dim, &deltas);
            let dt = d_mat.transpose();
            let dtd = &dt * &d_mat + DMatrix::<f64>::identity(dim, dim) * FIT_RIDGE;
            let dc_vec = DVector::from_vec(dc);
            let dtc = &dt * &dc_vec;
            let g = match dtd.try_inverse() {
                Some(inv) => &inv * &dtc,
                None => dtc, // fall back to the (unnormalized) descent direction Dᵀ Δc
            };

            // Trust-region step via the box QP: ‖Δ‖∞ ≤ trust.
            let g_vec: Vec<f64> = g.iter().copied().collect();
            let lo = vec![-trust; dim];
            let hi = vec![trust; dim];
            let step = crate::qp::solve_box_qp(&h, &g_vec, &lo, &hi);

            // Evaluate the candidate; accept only on improvement, then adapt the trust radius.
            let candidate: Vec<f64> = (0..dim).map(|j| best[j] + step[j]).collect();
            let cand_cost = self.rollout_flat(&dynamics, &cost, x0, &candidate);
            if cand_cost.is_finite() && cand_cost < best_cost - 1e-12 {
                best = candidate;
                best_cost = cand_cost;
                trust *= TRUST_GROW;
            } else {
                trust = (trust * TRUST_SHRINK).max(TRUST_MIN);
            }
            history.push(best_cost);
        }

        // Reshape the flat plan back to horizon × control_dim.
        let controls: Vec<Vec<f64>> =
            (0..self.horizon).map(|t| best[t * m..t * m + m].to_vec()).collect();
        (controls, history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// Discrete double integrator `x⁺ = A·x + B·u` (state `[pos, vel]`, scalar force `u`).
    fn double_integrator(
        dt: f64,
    ) -> (impl Fn(&[f64], &[f64]) -> Vec<f64>, impl Fn(&[f64], &[f64]) -> f64) {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, dt]);
        let dynamics = move |x: &[f64], u: &[f64]| {
            let xv = DVector::from_row_slice(x);
            let uv = DVector::from_row_slice(u);
            let nx = &a * &xv + &b * &uv;
            nx.iter().copied().collect::<Vec<f64>>()
        };
        // Regulate toward the origin: penalize position hard, velocity lightly, control faintly.
        let cost =
            |x: &[f64], u: &[f64]| x[0] * x[0] + 0.1 * x[1] * x[1] + 0.001 * u[0] * u[0];
        (dynamics, cost)
    }

    fn make_bundle(seed: u64) -> TrajectoryBundle {
        TrajectoryBundle {
            horizon: 20,
            dt: 0.1,
            control_dim: 1,
            num_samples: 64,
            sigma: 0.1,
            iters: 80,
            lambda: 0.5,
            trust: 0.5,
            rng: seed,
        }
    }

    /// Roll `controls` through the discrete dynamics and return the final state.
    fn final_state<D>(dynamics: &D, x0: &[f64], controls: &[Vec<f64>]) -> Vec<f64>
    where
        D: Fn(&[f64], &[f64]) -> Vec<f64>,
    {
        let mut x = x0.to_vec();
        for u in controls {
            x = dynamics(&x, u);
        }
        x
    }

    #[test]
    fn regulates_a_double_integrator() {
        let (dynamics, cost) = double_integrator(0.1);
        let mut opt = make_bundle(0x1234_5678_9abc_def0);
        let x0 = vec![1.0, 0.0];
        let nominal = vec![vec![0.0; 1]; opt.horizon];

        let (controls, history) = opt.optimize(&dynamics, &cost, &x0, &nominal);

        // Cost history is monotonically non-increasing (bundle refinement never regresses).
        assert_eq!(history.len(), opt.iters + 1);
        for w in history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "cost history increased: {history:?}");
        }
        // Refinement actually lowered the cost substantially.
        assert!(
            *history.last().unwrap() < 0.5 * history[0],
            "cost barely improved: {} -> {}",
            history[0],
            history.last().unwrap()
        );

        // The rolled-out final state is much closer to the origin than the do-nothing nominal's
        // (which just sits at pos = 1).
        let nom_final = final_state(&dynamics, &x0, &nominal);
        let opt_final = final_state(&dynamics, &x0, &controls);
        let nom_dist = (nom_final[0].powi(2) + nom_final[1].powi(2)).sqrt();
        let opt_dist = (opt_final[0].powi(2) + opt_final[1].powi(2)).sqrt();
        assert!(nom_dist > 0.9, "sanity: nominal should stay near pos 1, got {nom_dist}");
        assert!(
            opt_dist < 0.5 * nom_dist,
            "final state not substantially closer to goal: {nom_dist} -> {opt_dist} (state {opt_final:?})"
        );
    }

    #[test]
    fn is_deterministic_in_its_seed() {
        let (dynamics, cost) = double_integrator(0.1);
        let x0 = vec![1.0, 0.0];
        let nominal = vec![vec![0.0; 1]; 20];

        let mut a = make_bundle(42);
        let mut b = make_bundle(42);
        let (ca, ha) = a.optimize(&dynamics, &cost, &x0, &nominal);
        let (cb, hb) = b.optimize(&dynamics, &cost, &x0, &nominal);
        assert_eq!(ca, cb, "same-seed control plans diverged");
        assert_eq!(ha, hb, "same-seed cost histories diverged");

        // A different seed generally yields a different refined plan.
        let mut c = make_bundle(9001);
        let (cc, _) = c.optimize(&dynamics, &cost, &x0, &nominal);
        assert_ne!(ca, cc, "different seeds gave identical plans (unexpected)");
    }

    #[test]
    fn rollout_cost_matches_optimized_plan() {
        // The returned best_controls should score no worse than the starting nominal under the
        // public rollout_cost helper (consistency between optimize and rollout_cost).
        let (dynamics, cost) = double_integrator(0.1);
        let mut opt = make_bundle(7);
        let x0 = vec![1.0, 0.0];
        let nominal = vec![vec![0.0; 1]; opt.horizon];

        let nom_cost = opt.rollout_cost(&dynamics, &cost, &x0, &nominal);
        let (controls, history) = opt.optimize(&dynamics, &cost, &x0, &nominal);
        let plan_cost = opt.rollout_cost(&dynamics, &cost, &x0, &controls);

        assert!((plan_cost - *history.last().unwrap()).abs() < 1e-9, "rollout_cost != history tail");
        assert!(plan_cost <= nom_cost, "optimized plan worse than nominal: {nom_cost} -> {plan_cost}");
    }
}
