//! **Welford online mean & covariance** — numerically-stable, single-pass running statistics for streaming
//! vector data, plus a **parallel merge** (Chan's formula) to combine accumulators. Computing a mean and
//! covariance by accumulating `Σx` and `Σxxᵀ` and subtracting loses catastrophic precision when the data
//! sits far from the origin (a common case: sensor readings around a large bias); Welford's incremental
//! update keeps full accuracy in one pass with `O(d²)` state and no stored history. In physical AI this is
//! the standard tool for **observation / reward normalization** in RL policies, streaming sensor-noise
//! characterization, and adaptive filters that track slowly-varying signal statistics.
//!
//! Verified: the online mean and covariance match a batch computation to machine precision; it stays exact
//! where a naive sum-of-squares would lose precision (data offset by `10⁸`); and merging two accumulators
//! equals accumulating all the data together. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// Streaming mean and covariance of `d`-dimensional samples (Welford).
#[derive(Clone, Debug)]
pub struct RunningStats {
    n: usize,
    mean: DVector<f64>,
    // co-moment accumulator M2 = Σ (xᵢ − mean)(xᵢ − mean)ᵀ (updated incrementally)
    m2: DMatrix<f64>,
}

impl RunningStats {
    /// An empty accumulator over `d`-dimensional samples.
    pub fn new(d: usize) -> Self {
        RunningStats { n: 0, mean: DVector::zeros(d), m2: DMatrix::zeros(d, d) }
    }

    /// Number of samples seen.
    pub fn count(&self) -> usize {
        self.n
    }

    /// Ingest one sample (Welford update).
    pub fn update(&mut self, x: &DVector<f64>) {
        self.n += 1;
        let delta = x - &self.mean;
        self.mean += &delta / self.n as f64;
        let delta2 = x - &self.mean; // after updating the mean
        self.m2 += &delta * delta2.transpose();
    }

    /// The running mean.
    pub fn mean(&self) -> DVector<f64> {
        self.mean.clone()
    }

    /// The sample covariance `M2 / (n − 1)` (returns zeros for fewer than 2 samples).
    pub fn covariance(&self) -> DMatrix<f64> {
        if self.n < 2 {
            DMatrix::zeros(self.mean.len(), self.mean.len())
        } else {
            &self.m2 / (self.n as f64 - 1.0)
        }
    }

    /// The sample variance of each component (the covariance diagonal).
    pub fn variance(&self) -> DVector<f64> {
        self.covariance().diagonal()
    }

    /// Merge another accumulator into this one (Chan's parallel algorithm) — the result equals having fed
    /// both streams to a single accumulator.
    pub fn merge(&mut self, other: &RunningStats) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }
        let (na, nb) = (self.n as f64, other.n as f64);
        let n = na + nb;
        let delta = &other.mean - &self.mean;
        self.mean += &delta * (nb / n);
        self.m2 += &other.m2 + &delta * delta.transpose() * (na * nb / n);
        self.n += other.n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(data: &[DVector<f64>]) -> (DVector<f64>, DMatrix<f64>) {
        let n = data.len();
        let d = data[0].len();
        let mean = data.iter().fold(DVector::zeros(d), |a, x| a + x) / n as f64;
        let mut cov = DMatrix::zeros(d, d);
        for x in data {
            let e = x - &mean;
            cov += &e * e.transpose();
        }
        (mean, cov / (n as f64 - 1.0))
    }

    fn sample_data(offset: f64) -> Vec<DVector<f64>> {
        let mut seed = 1u64;
        let mut u = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5 };
        (0..200).map(|_| DVector::from_row_slice(&[offset + u(), offset + 2.0 * u(), offset - u()])).collect()
    }

    #[test]
    fn the_online_stats_match_the_batch_computation() {
        // THE ORACLE. Streaming Welford equals the batch mean/covariance.
        let data = sample_data(0.0);
        let mut rs = RunningStats::new(3);
        for x in &data {
            rs.update(x);
        }
        let (bm, bc) = batch(&data);
        assert!((rs.mean() - bm).norm() < 1e-12, "mean mismatch");
        assert!((rs.covariance() - bc).norm() < 1e-12, "covariance mismatch");
        assert_eq!(rs.count(), 200);
    }

    #[test]
    fn it_beats_the_naive_sum_of_squares_at_a_huge_offset() {
        // Welford's reason for being. Offset the data by 1e8. The true (variance-scale) covariance is the
        // stable centered `batch`. Welford tracks it closely; the naive Σxxᵀ − (Σx)²/n catastrophically
        // cancels (2e18 − 2e18) and is wildly wrong.
        let data = sample_data(1e8);
        let (_bm, truth) = batch(&data);
        let mut rs = RunningStats::new(3);
        for x in &data {
            rs.update(x);
        }
        let welford_rel = (rs.covariance() - &truth).norm() / truth.norm();

        // naive unstable estimator
        let n = data.len() as f64;
        let sum = data.iter().fold(DVector::zeros(3), |a, x| a + x);
        let mut sumsq = DMatrix::zeros(3, 3);
        for x in &data {
            sumsq += x * x.transpose();
        }
        let mean_n = &sum / n;
        let naive = (sumsq - &mean_n * mean_n.transpose() * n) / (n - 1.0);
        let naive_rel = (naive - &truth).norm() / truth.norm();

        assert!(welford_rel < 1e-6, "Welford should stay accurate: rel {welford_rel}");
        assert!(naive_rel > 100.0 * welford_rel, "naive should be far worse: naive {naive_rel} vs welford {welford_rel}");
    }

    #[test]
    fn merging_accumulators_equals_one_combined_stream() {
        let data = sample_data(3.0);
        let mut whole = RunningStats::new(3);
        for x in &data {
            whole.update(x);
        }
        let (mut a, mut b) = (RunningStats::new(3), RunningStats::new(3));
        for x in &data[..70] {
            a.update(x);
        }
        for x in &data[70..] {
            b.update(x);
        }
        a.merge(&b);
        assert!((a.mean() - whole.mean()).norm() < 1e-10, "merged mean");
        assert!((a.covariance() - whole.covariance()).norm() < 1e-10, "merged covariance");
        assert_eq!(a.count(), whole.count());
    }
}
