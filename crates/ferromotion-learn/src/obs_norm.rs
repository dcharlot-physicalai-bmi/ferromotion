//! **Observation normalization** — the transform that has to travel with the weights.
//!
//! A policy net sees whatever units the environment reports. A joint angle of order 1 alongside a rate of order
//! 30 saturates the first `tanh` layer on the rate alone, and the policy cannot see the error it is meant to
//! null. That is not hypothetical: the actuator bench in this workspace would not learn until its observations
//! were divided by hand inside the environment.
//!
//! **Hand-normalising in the environment is the wrong place**, for three reasons that are all failure modes:
//!
//! * Every environment author has to know to do it, and nothing checks that they did.
//! * The constants are invisible to the checkpoint, so a deployed policy has no idea what scaling it was
//!   trained under.
//! * Change a sensor's units and a trained checkpoint silently means something different, with no error.
//!
//! So the transform is estimated during training, applied before the policy sees anything, and **shipped in the
//! checkpoint** — [`ferromotion_policy::Policy`] already has `obs_mean` and `obs_std` fields and applies them in
//! `act`. Until now [`to_deployable`](crate::GaussianPolicy::to_deployable) always shipped them empty, because
//! nothing computed any. This closes that loop.
//!
//! # Welford, not sum-of-squares
//!
//! The naive estimator accumulates `Σx` and `Σx²` and forms `E[x²] − E[x]²`. That subtracts two nearly-equal
//! large numbers whenever the mean is large relative to the spread — which is exactly an observation like a
//! position reported about a large setpoint, or a temperature in kelvin. [`ObsNorm`] uses Welford's online
//! update instead.
//!
//! Measured on a channel offset by `1e9` with a spread of `1e-3`: the true variance is `3.995e-6`, Welford
//! returns it to a relative error of **4.3e-6**, and the naive form returns **6272** — a relative error of
//! `1.6e9`, fifteen orders worse. Note Welford is *not* exact here either, and cannot be: an offset-to-spread
//! ratio of `1e12` leaves `f64` about six digits on the deviations before they are squared. The claim is the
//! contrast, not perfection, and an earlier version of this comment overstated both halves of it.
//!
//! # The statistics freeze at export
//!
//! A running estimate that keeps moving after training stops means the deployed transform is not the trained
//! one. That is the same class of divergence as an action map applied on one side and not the other. Export
//! takes a snapshot, and [`ObsNorm::snapshot`] is the only way the numbers leave.

/// A per-dimension running mean and variance over observations, by Welford's online algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct ObsNorm {
    /// Per-dimension running mean.
    mean: Vec<f64>,
    /// Per-dimension sum of squared deviations from the running mean (Welford's `M2`).
    m2: Vec<f64>,
    /// Samples seen.
    count: u64,
    /// Floor on the standard deviation used for scaling, so a constant channel does not divide by zero.
    epsilon: f64,
}

impl ObsNorm {
    /// A fresh normalizer over `dim` channels, with a standard-deviation floor of `1e-8`.
    pub fn new(dim: usize) -> ObsNorm {
        ObsNorm { mean: vec![0.0; dim], m2: vec![0.0; dim], count: 0, epsilon: 1e-8 }
    }

    /// As [`new`](ObsNorm::new), with an explicit standard-deviation floor.
    ///
    /// The floor is what a **constant** channel divides by. A sensor that never moves has zero variance, and
    /// scaling by it would produce infinities from a channel that carries no information at all; the floor
    /// turns that into a passthrough of the (zero) deviation instead.
    pub fn with_epsilon(dim: usize, epsilon: f64) -> Option<ObsNorm> {
        // Finiteness first, so the ordering comparison is on a real number and NaN cannot slip past it. Same
        // shape as `BoxSpace::new` and `SnCurve::basquin`.
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return None;
        }
        Some(ObsNorm { mean: vec![0.0; dim], m2: vec![0.0; dim], count: 0, epsilon })
    }

    /// Channels.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Samples seen.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Fold one observation in. Ignores an observation of the wrong length rather than panicking mid-rollout,
    /// and reports it by leaving the count unchanged.
    pub fn update(&mut self, obs: &[f64]) {
        if obs.len() != self.dim() || obs.iter().any(|v| !v.is_finite()) {
            return;
        }
        self.count += 1;
        let n = self.count as f64;
        for d in 0..self.dim() {
            // Welford: the mean is corrected by the residual over the new count, and M2 accumulates the product
            // of the residuals BEFORE and AFTER that correction. Never forms a sum of squares.
            let delta = obs[d] - self.mean[d];
            self.mean[d] += delta / n;
            let delta2 = obs[d] - self.mean[d];
            self.m2[d] += delta * delta2;
        }
    }

    /// Fold in every observation of a batch.
    pub fn update_batch(&mut self, batch: &[Vec<f64>]) {
        for o in batch {
            self.update(o);
        }
    }

    /// Per-dimension mean. Zeros before any sample.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Per-dimension **population** variance, `M2/n`.
    ///
    /// Population rather than sample (`M2/(n−1)`): the estimate is used to scale, not to do inference, and the
    /// two differ by a factor no scaling cares about. Zeros with fewer than two samples.
    pub fn variance(&self) -> Vec<f64> {
        if self.count < 2 {
            return vec![0.0; self.dim()];
        }
        self.m2.iter().map(|m| m / self.count as f64).collect()
    }

    /// Per-dimension standard deviation, floored at `epsilon`.
    pub fn std(&self) -> Vec<f64> {
        self.variance().iter().map(|v| v.sqrt().max(self.epsilon)).collect()
    }

    /// Normalize an observation: `(x − mean)/std`.
    ///
    /// **Identity before two samples.** With one sample the variance is undefined and the mean is that sample,
    /// so normalizing would map the only observation seen to exactly zero and hand the policy a constant. A
    /// passthrough is the honest answer until there is a spread to divide by.
    pub fn normalize(&self, obs: &[f64]) -> Vec<f64> {
        if self.count < 2 || obs.len() != self.dim() {
            return obs.to_vec();
        }
        let sd = self.std();
        (0..self.dim()).map(|d| (obs[d] - self.mean[d]) / sd[d]).collect()
    }

    /// Normalize a whole batch.
    pub fn normalize_batch(&self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        batch.iter().map(|o| self.normalize(o)).collect()
    }

    /// The `(mean, std)` pair to ship in a checkpoint, **frozen** at the moment of the call.
    ///
    /// Returns empty vectors before two samples, which is what a `Policy` interprets as "no transform" — the
    /// same passthrough [`normalize`](ObsNorm::normalize) applies, so a checkpoint exported early behaves
    /// identically to training rather than applying a half-formed transform.
    pub fn snapshot(&self) -> (Vec<f64>, Vec<f64>) {
        if self.count < 2 {
            return (Vec::new(), Vec::new());
        }
        (self.mean.clone(), self.std())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-pass mean and population variance, the reference the online form must match.
    fn two_pass(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        (mean, var)
    }

    /// The naive estimator, kept only to demonstrate that it fails where Welford does not.
    fn naive(data: &[f64]) -> (f64, f64) {
        let n = data.len() as f64;
        let s1: f64 = data.iter().sum();
        let s2: f64 = data.iter().map(|x| x * x).sum();
        (s1 / n, s2 / n - (s1 / n) * (s1 / n))
    }

    #[test]
    fn welford_matches_a_two_pass_computation() {
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        for n in [2usize, 5, 50, 5000] {
            let data: Vec<f64> = (0..n).map(|_| 40.0 * next()).collect();
            let mut norm = ObsNorm::new(1);
            for x in &data {
                norm.update(&[*x]);
            }
            let (m, v) = two_pass(&data);
            assert!((norm.mean()[0] - m).abs() < 1e-12 * m.abs().max(1.0), "n={n} mean");
            assert!((norm.variance()[0] - v).abs() < 1e-10 * v.abs().max(1.0), "n={n} variance");
            assert_eq!(norm.count(), n as u64);
        }
    }

    #[test]
    fn the_naive_estimator_fails_where_welford_does_not() {
        // The control that justifies the algorithm choice rather than asserting it. A channel with a large
        // offset and a small spread is not exotic — it is a position about a setpoint, or a temperature in
        // kelvin — and the naive form subtracts two nearly-equal large numbers.
        let offset = 1e9;
        let data: Vec<f64> = (0..1000).map(|k| offset + (k % 7) as f64 * 1e-3).collect();
        let (_, v_true) = two_pass(&data);
        assert!(v_true > 0.0, "the fixture must have real spread, got {v_true:.3e}");

        let (_, v_naive) = naive(&data);
        let mut norm = ObsNorm::new(1);
        for x in &data {
            norm.update(&[*x]);
        }
        let v_welford = norm.variance()[0];

        // Welford is ACCURATE, not exact, and asserting exactness here was wrong: with an offset of 1e9 against
        // a spread of 1e-3 the ratio is 1e12, so f64 has about six digits left on the deviations before they are
        // squared. Measured 4.3e-6 relative, and a 1e-9 bound demanded a precision the format cannot deliver.
        let err_welford = (v_welford - v_true).abs() / v_true;
        assert!(err_welford < 1e-4, "Welford relative error {err_welford:.3e} on this fixture");

        // The naive form loses the answer entirely: measured 6272 against a true 3.995e-6.
        let err_naive = (v_naive - v_true).abs() / v_true;
        assert!(err_naive > 1.0, "the naive form must visibly fail, relative error only {err_naive:.3e}");

        // THE CLAIM IS THE CONTRAST, so assert that rather than either number alone. Measured 1.6e9 against
        // 4.3e-6, which is fifteen orders.
        assert!(
            err_naive / err_welford > 1e6,
            "the two must differ by orders of magnitude: naive {err_naive:.3e} vs Welford {err_welford:.3e}"
        );
    }

    #[test]
    fn normalizing_its_own_data_gives_zero_mean_and_unit_variance() {
        // The operational property: after the transform the policy sees channels on one scale.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        // Three channels on wildly different scales, which is the case that motivates this.
        let data: Vec<Vec<f64>> =
            (0..4000).map(|_| vec![0.5 * next(), 30.0 * next(), 1e-3 * next()]).collect();
        let mut norm = ObsNorm::new(3);
        norm.update_batch(&data);
        let z = norm.normalize_batch(&data);

        for d in 0..3 {
            let col: Vec<f64> = z.iter().map(|r| r[d]).collect();
            let (m, v) = two_pass(&col);
            assert!(m.abs() < 1e-9, "channel {d} normalised mean {m:.3e}");
            assert!((v - 1.0).abs() < 1e-9, "channel {d} normalised variance {v}");
        }
        // And the raw channels really did differ by orders of magnitude, or the test proves nothing.
        let raw_sd = norm.std();
        assert!(raw_sd[1] / raw_sd[0] > 10.0 && raw_sd[0] / raw_sd[2] > 10.0, "raw scales: {raw_sd:?}");
    }

    #[test]
    fn it_is_a_passthrough_until_there_is_a_spread_to_divide_by() {
        // With one sample, normalising would map the only observation seen to exactly zero and hand the policy
        // a constant. A passthrough is the honest answer.
        let mut norm = ObsNorm::new(2);
        assert_eq!(norm.normalize(&[3.0, -7.0]), vec![3.0, -7.0], "no samples: identity");
        norm.update(&[3.0, -7.0]);
        assert_eq!(norm.normalize(&[3.0, -7.0]), vec![3.0, -7.0], "one sample: identity");
        assert_eq!(norm.snapshot(), (Vec::new(), Vec::new()), "and nothing to ship");
        norm.update(&[5.0, -3.0]);
        assert_ne!(norm.normalize(&[3.0, -7.0]), vec![3.0, -7.0], "two samples: it transforms");
        let (m, s) = norm.snapshot();
        assert_eq!(m.len(), 2);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn a_constant_channel_divides_by_the_floor_rather_than_by_zero() {
        // A sensor that never moves carries no information; it must not produce infinities.
        let mut norm = ObsNorm::new(2);
        for k in 0..100 {
            norm.update(&[7.0, k as f64]);
        }
        assert_eq!(norm.variance()[0], 0.0, "the constant channel has zero variance");
        assert_eq!(norm.std()[0], 1e-8, "and its std is the floor");
        let z = norm.normalize(&[7.0, 50.0]);
        assert!(z.iter().all(|v| v.is_finite()), "no infinities: {z:?}");
        assert_eq!(z[0], 0.0, "a constant channel normalises to exactly zero deviation");

        // A custom floor is honoured, and a non-positive one is refused.
        let n2 = ObsNorm::with_epsilon(1, 0.5).expect("positive floor");
        assert!(ObsNorm::with_epsilon(1, 0.0).is_none());
        assert!(ObsNorm::with_epsilon(1, -1.0).is_none());
        assert_eq!(n2.dim(), 1);
    }

    #[test]
    fn malformed_observations_are_ignored_rather_than_poisoning_the_estimate() {
        // A NaN folded in would make the mean NaN forever and silently destroy every later observation. Mid
        // rollout is the worst place to panic, so these are dropped and the count shows it.
        let mut norm = ObsNorm::new(2);
        norm.update(&[1.0, 2.0]);
        norm.update(&[3.0, 4.0]);
        let before = (norm.mean().to_vec(), norm.count());
        norm.update(&[f64::NAN, 1.0]);
        norm.update(&[f64::INFINITY, 1.0]);
        norm.update(&[1.0]); // wrong length
        norm.update(&[1.0, 2.0, 3.0]); // wrong length
        assert_eq!((norm.mean().to_vec(), norm.count()), before, "nothing malformed may land");
        assert!(norm.std().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_snapshot_is_frozen_and_does_not_track_later_updates() {
        // A running estimate that keeps moving after export means the deployed transform is not the trained
        // one — the same divergence class as an action map applied on one side only.
        let mut norm = ObsNorm::new(1);
        for k in 0..50 {
            norm.update(&[k as f64]);
        }
        let (m0, s0) = norm.snapshot();
        for k in 0..5000 {
            norm.update(&[1e6 + k as f64]);
        }
        let (m1, s1) = norm.snapshot();
        assert_ne!(m0, m1, "the live estimate must have moved, or this proves nothing");
        // The earlier snapshot is unaffected by later updates: it is a value, not a view.
        assert!((m0[0] - 24.5).abs() < 1e-12, "the frozen mean stays put, got {}", m0[0]);
        assert!(s0[0] > 0.0 && s1[0] > s0[0]);
    }
}
