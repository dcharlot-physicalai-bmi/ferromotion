//! **Covariance steering** (Chen, Georgiou & Pavon, IEEE TAC 2016; discrete-time batch form after
//! Bakolas / Goldshtein–Tsiotras) — drive not just the *mean* of a stochastic system but its **whole
//! Gaussian distribution** to a prescribed terminal one. Deterministic optimal control regulates the
//! mean; chance-constrained MPC only tightens constraints; covariance steering is the missing primitive in
//! between — "arrive at the goal with covariance exactly `Σ_f`" (e.g. a tight grasp pose, a docking gate).
//!
//! For a controllable linear system `x_{k+1} = A x_k + B u_k`, an affine state-feedback policy
//! `u_k = ū_k + [L]_k (x_0 − μ_0)` reaches an exact terminal distribution `𝒩(μ_f, Σ_f)` in closed form:
//! stacking the horizon gives `x_N = Aᴺ x_0 + G·U` with controllability matrix `G`, so the feed-forward
//! `Ū = G⁺(μ_f − Aᴺ μ_0)` places the mean and the gain `L = G⁺(Φ − Aᴺ)` with `Φ = Σ_f^{1/2} Σ_0^{-1/2}`
//! makes the closed-loop map `Φ` satisfy `Φ Σ_0 Φᵀ = Σ_f` exactly. Verified against that analytic terminal
//! covariance and by Monte-Carlo. Pure `nalgebra`, deterministic → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A discrete linear system `x⁺ = A x + B u` over `horizon` steps.
#[derive(Clone, Debug)]
pub struct CovarianceSteering {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub horizon: usize,
}

/// The steering policy: per-step feed-forward `ū_k` and the stacked feedback gain `L` (`Nm × n`) applied to
/// the initial deviation `x_0 − μ_0`.
#[derive(Clone, Debug)]
pub struct SteeringPolicy {
    pub ubar: DVector<f64>, // stacked (N·m)
    pub gain: DMatrix<f64>, // N·m × n
}

/// Symmetric matrix square root of an SPD matrix via its eigendecomposition.
fn sym_sqrt(m: &DMatrix<f64>) -> DMatrix<f64> {
    let e = m.clone().symmetric_eigen();
    let d = DMatrix::from_diagonal(&e.eigenvalues.map(|v| v.max(0.0).sqrt()));
    &e.eigenvectors * d * e.eigenvectors.transpose()
}
/// Symmetric inverse square root of an SPD matrix.
fn sym_inv_sqrt(m: &DMatrix<f64>) -> DMatrix<f64> {
    let e = m.clone().symmetric_eigen();
    let d = DMatrix::from_diagonal(&e.eigenvalues.map(|v| 1.0 / v.max(1e-12).sqrt()));
    &e.eigenvectors * d * e.eigenvectors.transpose()
}

impl CovarianceSteering {
    fn n(&self) -> usize {
        self.a.nrows()
    }
    fn m(&self) -> usize {
        self.b.ncols()
    }

    /// `Aᴺ`.
    fn a_pow_n(&self) -> DMatrix<f64> {
        let mut p = DMatrix::identity(self.n(), self.n());
        for _ in 0..self.horizon {
            p = &self.a * p;
        }
        p
    }

    /// The reachability matrix `G = [Aᴺ⁻¹B, …, AB, B]` (`n × N·m`) so that `x_N = Aᴺ x_0 + G·U`.
    fn reachability(&self) -> DMatrix<f64> {
        let (n, m, big_n) = (self.n(), self.m(), self.horizon);
        let mut g = DMatrix::zeros(n, big_n * m);
        // block k (control u_k) is A^{N-1-k} B
        let mut apow = vec![DMatrix::<f64>::identity(n, n)];
        for _ in 0..big_n {
            apow.push(&self.a * apow.last().unwrap());
        }
        for k in 0..big_n {
            let blk = &apow[big_n - 1 - k] * &self.b;
            g.view_mut((0, k * m), (n, m)).copy_from(&blk);
        }
        g
    }

    /// Compute the covariance-steering policy from `𝒩(μ_0, Σ_0)` to `𝒩(μ_f, Σ_f)`.
    pub fn steer(&self, mu0: &DVector<f64>, sigma0: &DMatrix<f64>, muf: &DVector<f64>, sigmaf: &DMatrix<f64>) -> SteeringPolicy {
        let an = self.a_pow_n();
        let g = self.reachability();
        let gpinv = g.clone().pseudo_inverse(1e-10).expect("reachability pseudo-inverse");
        // mean steering: Ū = G⁺(μ_f − Aᴺ μ_0)
        let ubar = &gpinv * (muf - &an * mu0);
        // covariance steering: Φ = Σ_f^{1/2} Σ_0^{-1/2}, then L = G⁺(Φ − Aᴺ)
        let phi = sym_sqrt(sigmaf) * sym_inv_sqrt(sigma0);
        let gain = &gpinv * (&phi - &an);
        SteeringPolicy { ubar, gain }
    }

    /// The closed-loop transition `Φ = Aᴺ + G·L` the policy induces (`x_N − μ_f = Φ(x_0 − μ_0)`).
    pub fn closed_loop(&self, policy: &SteeringPolicy) -> DMatrix<f64> {
        self.a_pow_n() + self.reachability() * &policy.gain
    }

    /// The analytic terminal covariance `Φ Σ_0 Φᵀ`.
    pub fn terminal_covariance(&self, policy: &SteeringPolicy, sigma0: &DMatrix<f64>) -> DMatrix<f64> {
        let phi = self.closed_loop(policy);
        &phi * sigma0 * phi.transpose()
    }

    /// Roll out a single initial state `x0` under the policy, returning `x_N`.
    pub fn rollout(&self, policy: &SteeringPolicy, mu0: &DVector<f64>, x0: &DVector<f64>) -> DVector<f64> {
        let m = self.m();
        let dev = x0 - mu0;
        let mut x = x0.clone();
        for k in 0..self.horizon {
            let u = policy.ubar.rows(k * m, m) + policy.gain.view((k * m, 0), (m, self.n())) * &dev;
            x = &self.a * &x + &self.b * u;
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A controllable double integrator: x = [p, v], u = [a].
    fn sys(dt: f64, horizon: usize) -> CovarianceSteering {
        CovarianceSteering {
            a: DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]),
            b: DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]),
            horizon,
        }
    }

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    #[test]
    fn the_terminal_mean_and_covariance_are_hit_exactly() {
        // THE INVARIANT. The analytic closed-loop reaches μ_f and Σ_f exactly.
        let s = sys(0.1, 20);
        let mu0 = dv(&[0.0, 0.0]);
        let sigma0 = DMatrix::from_diagonal(&dv(&[0.04, 0.09])); // wide, correlated-free
        let muf = dv(&[1.0, 0.0]);
        let sigmaf = DMatrix::from_row_slice(2, 2, &[0.01, 0.002, 0.002, 0.005]); // tight, tilted target
        let pol = s.steer(&mu0, &sigma0, &muf, &sigmaf);

        // mean: μ_N = Aᴺ μ0 + G ū
        let mu_n = s.a_pow_n() * &mu0 + s.reachability() * &pol.ubar;
        assert!((&mu_n - &muf).norm() < 1e-9, "terminal mean off: {mu_n}");
        // covariance: Φ Σ0 Φᵀ = Σf
        let cov_n = s.terminal_covariance(&pol, &sigma0);
        assert!((&cov_n - &sigmaf).abs().max() < 1e-8, "terminal covariance off: {cov_n}");
    }

    #[test]
    fn a_monte_carlo_ensemble_matches_the_target_distribution() {
        // THE HEADLINE. Draw an ensemble from 𝒩(μ0,Σ0), apply the feedback policy, and the terminal sample
        // mean/covariance match the target — the whole cloud is steered, not just its center.
        let s = sys(0.1, 15);
        let mu0 = dv(&[0.2, -0.1]);
        let sigma0 = DMatrix::from_diagonal(&dv(&[0.05, 0.05]));
        let muf = dv(&[1.5, 0.0]);
        let sigmaf = DMatrix::from_diagonal(&dv(&[0.004, 0.002]));
        let pol = s.steer(&mu0, &sigma0, &muf, &sigmaf);

        // deterministic Gaussian ensemble via a seeded LCG + Box–Muller
        let l0 = sym_sqrt(&sigma0);
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut samples = Vec::new();
        for _ in 0..8000 {
            let (u1, u2): (f64, f64) = (rng().max(1e-12), rng());
            let r = (-2.0 * u1.ln()).sqrt();
            let z = dv(&[r * (std::f64::consts::TAU * u2).cos(), r * (std::f64::consts::TAU * u2).sin()]);
            let x0 = &mu0 + &l0 * z;
            samples.push(s.rollout(&pol, &mu0, &x0));
        }
        let nsamp = samples.len() as f64;
        let mean: DVector<f64> = samples.iter().sum::<DVector<f64>>() / nsamp;
        let mut cov = DMatrix::zeros(2, 2);
        for x in &samples {
            let d = x - &mean;
            cov += &d * d.transpose();
        }
        cov /= nsamp;
        assert!((&mean - &muf).norm() < 0.03, "ensemble mean {mean} vs target {muf}");
        assert!((&cov - &sigmaf).abs().max() < 0.01, "ensemble covariance off:\n{cov}\nvs\n{sigmaf}");
    }

    #[test]
    fn it_can_inflate_as_well_as_shrink_uncertainty() {
        // Steering is bidirectional: a small initial covariance can be driven to a larger terminal one.
        let s = sys(0.1, 12);
        let mu0 = dv(&[0.0, 0.0]);
        let sigma0 = DMatrix::from_diagonal(&dv(&[0.001, 0.001]));
        let sigmaf = DMatrix::from_diagonal(&dv(&[0.05, 0.02]));
        let pol = s.steer(&mu0, &sigma0, &dv(&[0.5, 0.0]), &sigmaf);
        let cov_n = s.terminal_covariance(&pol, &sigma0);
        assert!((&cov_n - &sigmaf).abs().max() < 1e-7, "should inflate to the target covariance");
    }
}
