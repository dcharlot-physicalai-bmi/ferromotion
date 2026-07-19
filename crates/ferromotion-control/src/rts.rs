//! **Rauch–Tung–Striebel (RTS) smoother** (Rauch, Tung & Striebel, 1965) — the optimal *fixed-interval
//! smoother* for a linear-Gaussian system. A Kalman filter ([`crate::estimation`]) estimates each state from
//! **past** measurements only; the RTS smoother makes a **backward pass** that folds in **future**
//! measurements too, so every estimate is conditioned on the *entire* record. The result is the exact
//! posterior mean `E[xₖ | z₁:N]` — strictly more accurate and less uncertain than the forward filter, which
//! is why offline trajectory estimation, calibration, and batch SLAM all smooth rather than filter.
//!
//! Given the linear model `xₖ₊₁ = F xₖ + wₖ`, `zₖ = H xₖ + vₖ` with covariances `Q`, `R`, the forward pass
//! stores the predicted and filtered moments, and the backward recursion blends them through the smoother
//! gain `Cₖ = P̂ₖ Fᵀ (P⁻ₖ₊₁)⁻¹`. Verified: the smoothed trajectory has lower RMS error than the filtered one
//! on a noisy tracking problem; the smoothed covariance is tighter than the filtered covariance at interior
//! steps; and the backward pass leaves the final step unchanged (it already used all data). Pure `nalgebra`
//! → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A linear-Gaussian RTS smoother for `xₖ₊₁ = F xₖ + w`, `zₖ = H xₖ + v`, `Cov(w)=Q`, `Cov(v)=R`.
#[derive(Clone, Debug)]
pub struct RtsSmoother {
    pub f: DMatrix<f64>,
    pub h: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
}

/// The result of smoothing: filtered and smoothed means/covariances at every step.
#[derive(Clone, Debug)]
pub struct SmoothResult {
    pub filtered: Vec<DVector<f64>>,
    pub filtered_cov: Vec<DMatrix<f64>>,
    pub smoothed: Vec<DVector<f64>>,
    pub smoothed_cov: Vec<DMatrix<f64>>,
}

impl RtsSmoother {
    /// Run the forward Kalman filter over `measurements` (starting from prior `x0`, `p0`) and then the RTS
    /// backward pass. `measurements[k]` is the observation at step `k` (the prior is the state at step 0
    /// *before* the first measurement is applied at step 0).
    pub fn smooth(&self, x0: &DVector<f64>, p0: &DMatrix<f64>, measurements: &[DVector<f64>]) -> SmoothResult {
        let n = measurements.len();
        let dim = x0.len();
        let ident = DMatrix::<f64>::identity(dim, dim);

        // forward pass storing predicted (prior) and filtered (posterior) moments at each step
        let mut x_pred = Vec::with_capacity(n);
        let mut p_pred = Vec::with_capacity(n);
        let mut x_filt = Vec::with_capacity(n);
        let mut p_filt = Vec::with_capacity(n);

        let (mut xp, mut pp) = (x0.clone(), p0.clone());
        for (k, z) in measurements.iter().enumerate() {
            if k > 0 {
                // predict from the previous posterior
                xp = &self.f * &x_filt[k - 1];
                pp = &self.f * &p_filt[k - 1] * self.f.transpose() + &self.q;
            }
            x_pred.push(xp.clone());
            p_pred.push(pp.clone());
            // update with the measurement
            let s = &self.h * &pp * self.h.transpose() + &self.r;
            let k_gain = &pp * self.h.transpose() * s.clone().try_inverse().expect("innovation covariance invertible");
            let xf = &xp + &k_gain * (z - &self.h * &xp);
            let pf = (&ident - &k_gain * &self.h) * &pp;
            x_filt.push(xf);
            p_filt.push(pf);
        }

        // backward RTS pass
        let mut x_smooth = x_filt.clone();
        let mut p_smooth = p_filt.clone();
        for k in (0..n - 1).rev() {
            // smoother gain C = P̂ₖ Fᵀ (P⁻ₖ₊₁)⁻¹
            let c = &p_filt[k] * self.f.transpose() * p_pred[k + 1].clone().try_inverse().expect("predicted covariance invertible");
            x_smooth[k] = &x_filt[k] + &c * (&x_smooth[k + 1] - &x_pred[k + 1]);
            p_smooth[k] = &p_filt[k] + &c * (&p_smooth[k + 1] - &p_pred[k + 1]) * c.transpose();
        }

        SmoothResult { filtered: x_filt, filtered_cov: p_filt, smoothed: x_smooth, smoothed_cov: p_smooth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A constant-velocity 1-D tracker: state [position, velocity], measure position only.
    fn cv_system(dt: f64, q_scale: f64, r_var: f64) -> RtsSmoother {
        let f = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        // white-noise-acceleration process noise
        let q = DMatrix::from_row_slice(2, 2, &[dt.powi(3) / 3.0, dt.powi(2) / 2.0, dt.powi(2) / 2.0, dt]) * q_scale;
        let r = DMatrix::from_row_slice(1, 1, &[r_var]);
        RtsSmoother { f, h, q, r }
    }

    // deterministic Gaussian-ish noise (sum of uniforms), seedable
    fn noise(seed: &mut u64) -> f64 {
        let mut acc = 0.0;
        for _ in 0..12 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            acc += (*seed >> 40) as f64 / (1u64 << 24) as f64;
        }
        acc - 6.0 // ~N(0,1)
    }

    #[test]
    fn smoothing_beats_filtering_on_a_noisy_track() {
        // THE ORACLE. The RTS smoother, using the whole record, tracks the true trajectory more accurately
        // than the forward filter.
        let dt = 0.1;
        let sys = cv_system(dt, 0.02, 0.25);
        let steps = 60;
        // true constant-velocity trajectory + noisy position measurements
        let (mut pos, vel) = (0.0f64, 1.0f64);
        let mut seed = 12345u64;
        let mut truth = Vec::new();
        let mut meas = Vec::new();
        for _ in 0..steps {
            truth.push(pos);
            meas.push(DVector::from_row_slice(&[pos + 0.5 * noise(&mut seed)]));
            pos += vel * dt;
        }
        let x0 = DVector::from_row_slice(&[0.0, 0.0]);
        let p0 = DMatrix::identity(2, 2) * 10.0;
        let res = sys.smooth(&x0, &p0, &meas);
        let rms = |est: &[DVector<f64>]| -> f64 {
            (est.iter().zip(&truth).map(|(e, &t)| (e[0] - t).powi(2)).sum::<f64>() / steps as f64).sqrt()
        };
        let (filt_rms, smooth_rms) = (rms(&res.filtered), rms(&res.smoothed));
        assert!(smooth_rms < filt_rms, "smoothing should beat filtering: {smooth_rms} vs {filt_rms}");
    }

    #[test]
    fn smoothing_tightens_the_covariance_at_interior_steps() {
        // The smoothed covariance is ≤ the filtered covariance (it uses strictly more information).
        let dt = 0.1;
        let sys = cv_system(dt, 0.02, 0.25);
        let steps = 40;
        let mut seed = 999u64;
        let mut meas = Vec::new();
        let (mut pos, vel) = (0.0f64, 0.5f64);
        for _ in 0..steps {
            meas.push(DVector::from_row_slice(&[pos + 0.5 * noise(&mut seed)]));
            pos += vel * dt;
        }
        let res = sys.smooth(&DVector::from_row_slice(&[0.0, 0.0]), &(DMatrix::identity(2, 2) * 10.0), &meas);
        // interior step: smoothed variance trace should not exceed filtered
        for k in 0..steps - 1 {
            assert!(res.smoothed_cov[k].trace() <= res.filtered_cov[k].trace() + 1e-9, "step {k}: smoothed cov {} > filtered {}", res.smoothed_cov[k].trace(), res.filtered_cov[k].trace());
        }
    }

    #[test]
    fn the_last_step_is_unchanged_by_smoothing() {
        // The final filtered estimate already conditions on all data, so the backward pass leaves it alone.
        let dt = 0.2;
        let sys = cv_system(dt, 0.05, 0.5);
        let mut seed = 7u64;
        let meas: Vec<_> = (0..20).map(|k| DVector::from_row_slice(&[k as f64 * 0.3 + noise(&mut seed)])).collect();
        let res = sys.smooth(&DVector::from_row_slice(&[0.0, 0.0]), &(DMatrix::identity(2, 2) * 5.0), &meas);
        let last = res.filtered.len() - 1;
        assert!((&res.filtered[last] - &res.smoothed[last]).norm() < 1e-12, "last step should be identical");
    }
}
