//! **Bootstrap particle filter (SIR)** — the sequential-Monte-Carlo estimator the filter stack
//! lacked (it had EKF/UKF/InEKF/MSCKF, all Gaussian). Particles represent an arbitrary posterior, so
//! the filter tracks strongly nonlinear / multimodal systems where a Gaussian linearization fails.
//! Predict propagates each particle through the dynamics plus process noise; update reweights by the
//! measurement likelihood; systematic resampling fights weight degeneracy. Deterministic noise
//! (splitmix64 Box–Muller) → reproducible. `nalgebra` state → consistent with the rest of the stack.

use nalgebra::DVector;

/// A weighted particle set approximating a state posterior.
pub struct ParticleFilter {
    pub particles: Vec<DVector<f64>>,
    pub weights: Vec<f64>,
    rng: u64,
}

impl ParticleFilter {
    /// Initialize `n` particles from a diagonal Gaussian `mean ± std`.
    pub fn new(n: usize, mean: &DVector<f64>, std: &DVector<f64>, seed: u64) -> Self {
        let mut pf = ParticleFilter { particles: Vec::with_capacity(n), weights: vec![1.0 / n as f64; n], rng: seed };
        for _ in 0..n {
            let mut x = mean.clone();
            for d in 0..mean.len() {
                x[d] += std[d] * pf.gauss();
            }
            pf.particles.push(x);
        }
        pf
    }

    fn u01(&mut self) -> f64 {
        self.rng = self.rng.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) as f64) / (u64::MAX as f64)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.u01().max(1e-12);
        let u2 = self.u01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Propagate every particle through `f` and add diagonal Gaussian process noise `proc_std`.
    pub fn predict(&mut self, f: impl Fn(&DVector<f64>) -> DVector<f64>, proc_std: &DVector<f64>) {
        for i in 0..self.particles.len() {
            let mut x = f(&self.particles[i]);
            for d in 0..x.len() {
                x[d] += proc_std[d] * self.gauss();
            }
            self.particles[i] = x;
        }
    }

    /// Reweight by the Gaussian measurement likelihood `p(z | h(x))` with diagonal covariance
    /// `meas_var`, then normalize.
    pub fn update(&mut self, z: &DVector<f64>, h: impl Fn(&DVector<f64>) -> DVector<f64>, meas_var: &DVector<f64>) {
        let mut sum = 0.0;
        for i in 0..self.particles.len() {
            let hz = h(&self.particles[i]);
            let mut loglik = 0.0;
            for d in 0..z.len() {
                let e = z[d] - hz[d];
                loglik += -0.5 * e * e / meas_var[d];
            }
            self.weights[i] *= loglik.exp();
            sum += self.weights[i];
        }
        if sum <= 0.0 {
            let w = 1.0 / self.particles.len() as f64;
            self.weights.iter_mut().for_each(|x| *x = w);
        } else {
            self.weights.iter_mut().for_each(|x| *x /= sum);
        }
    }

    /// Weighted posterior mean.
    pub fn mean(&self) -> DVector<f64> {
        let dim = self.particles[0].len();
        let mut m = DVector::zeros(dim);
        for i in 0..self.particles.len() {
            m += self.weights[i] * &self.particles[i];
        }
        m
    }

    /// Effective sample size `1/Σwᵢ²`.
    pub fn ess(&self) -> f64 {
        1.0 / self.weights.iter().map(|w| w * w).sum::<f64>()
    }

    /// Systematic resampling — draw a new equally-weighted particle set (call when ESS is low).
    #[allow(clippy::needless_range_loop)] // CDF build + comb-sweep are inherently index-based
    pub fn resample(&mut self) {
        let n = self.particles.len();
        let mut cdf = vec![0.0; n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += self.weights[i];
            cdf[i] = acc;
        }
        let start = self.u01() / n as f64;
        let mut out = Vec::with_capacity(n);
        let mut j = 0;
        for i in 0..n {
            let u = start + i as f64 / n as f64;
            while j < n - 1 && u > cdf[j] {
                j += 1;
            }
            out.push(self.particles[j].clone());
        }
        self.particles = out;
        self.weights = vec![1.0 / n as f64; n];
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    /// The Gordon–Salmond–Smith (1993) nonlinear-growth benchmark — the canonical PF test, where the
    /// bimodal `y = x²/20` measurement defeats a Gaussian filter. The PF posterior mean tracks the
    /// hidden state with far lower error than the measurement-free prior baseline.
    #[test]
    fn particle_filter_tracks_gordon_model() {
        let dvn = |v: f64| DVector::from_vec(vec![v]);
        // ground-truth simulation with fixed deterministic noise
        let mut rng = 0xABCDu64;
        let mut rn = || {
            rng = rng.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            let u1 = (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)).max(1e-12);
            rng = rng.wrapping_add(0x9E3779B97F4A7C15);
            let mut z2 = rng;
            z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            let u2 = ((z2 ^ (z2 >> 31)) as f64) / (u64::MAX as f64);
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let steps = 60;
        let (proc_std, meas_std) = (3.16f64, 1.0f64); // Q=10, R=1
        let mut truth = 0.1;
        let mut ys = Vec::new();
        let mut truths = Vec::new();
        let f_at = |x: f64, k: usize| 0.5 * x + 25.0 * x / (1.0 + x * x) + 8.0 * (1.2 * k as f64).cos();
        for k in 1..=steps {
            truth = f_at(truth, k) + proc_std * rn();
            let y = truth * truth / 20.0 + meas_std * rn();
            truths.push(truth);
            ys.push(y);
        }

        let mut pf = ParticleFilter::new(600, &dvn(0.1), &dvn(2.0), 0x1234);
        let mut se_pf = 0.0;
        let mut se_prior = 0.0;
        let mut prior = 0.1; // measurement-free baseline (pure propagation of the mean)
        for k in 1..=steps {
            pf.predict(|x| dvn(f_at(x[0], k)), &dvn(proc_std));
            pf.update(&dvn(ys[k - 1]), |x| dvn(x[0] * x[0] / 20.0), &dvn(meas_std * meas_std));
            if pf.ess() < 300.0 {
                pf.resample();
            }
            let est = pf.mean()[0];
            se_pf += (est - truths[k - 1]).powi(2);
            prior = f_at(prior, k);
            se_prior += (prior - truths[k - 1]).powi(2);
        }
        let rmse_pf = (se_pf / steps as f64).sqrt();
        let rmse_prior = (se_prior / steps as f64).sqrt();
        eprintln!("PF RMSE {rmse_pf:.3} vs measurement-free prior RMSE {rmse_prior:.3}");
        assert!(rmse_pf < 0.6 * rmse_prior, "PF not clearly better than the prior: {rmse_pf} vs {rmse_prior}");
        assert!(rmse_pf < 8.0, "PF RMSE too large: {rmse_pf}");
    }
}
