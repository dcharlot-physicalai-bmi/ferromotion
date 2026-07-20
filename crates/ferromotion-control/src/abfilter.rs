//! **α-β and α-β-γ tracking filters** (the g-h and g-h-k filters). Fixed-gain steady-state trackers that
//! estimate a target's position, velocity (and, for α-β-γ, acceleration) from noisy **position**
//! measurements. They are the constant-gain limit of a Kalman filter for a constant-velocity / constant-
//! acceleration motion model: no covariance to propagate and no matrix inverse per step, just two (or three)
//! scalar gains — which is why radar trackers, embedded systems, and encoder/IMU smoothers reach for them
//! when a full Kalman filter ([`crate::estimation`]) is overkill.
//!
//! Each step **predicts** with the motion model then **corrects** by a fixed fraction of the measurement
//! residual: `x += α·r`, `v += (β/Δt)·r` (and `a += (2γ/Δt²)·r`). The α-β filter tracks constant velocity
//! with zero steady-state lag but *lags* under acceleration; α-β-γ removes that lag. Verified: α-β recovers
//! the velocity of a constant-velocity target and smooths measurement noise; α-β lags a constant-
//! acceleration target while α-β-γ tracks it lag-free. Pure Rust → WASM-clean.

/// An α-β (g-h) filter tracking position `x` and velocity `v`.
#[derive(Clone, Copy, Debug)]
pub struct AlphaBeta {
    pub alpha: f64,
    pub beta: f64,
    pub x: f64,
    pub v: f64,
    pub dt: f64,
}

impl AlphaBeta {
    /// A filter initialized at `x0` with zero velocity.
    pub fn new(alpha: f64, beta: f64, dt: f64, x0: f64) -> Self {
        AlphaBeta { alpha, beta, x: x0, v: 0.0, dt }
    }

    /// Ingest a position measurement `z`; returns the updated position estimate.
    pub fn update(&mut self, z: f64) -> f64 {
        let x_pred = self.x + self.v * self.dt;
        let r = z - x_pred; // residual
        self.x = x_pred + self.alpha * r;
        self.v += (self.beta / self.dt) * r;
        self.x
    }
}

/// An α-β-γ (g-h-k) filter tracking position, velocity, and acceleration.
#[derive(Clone, Copy, Debug)]
pub struct AlphaBetaGamma {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub x: f64,
    pub v: f64,
    pub a: f64,
    pub dt: f64,
}

impl AlphaBetaGamma {
    pub fn new(alpha: f64, beta: f64, gamma: f64, dt: f64, x0: f64) -> Self {
        AlphaBetaGamma { alpha, beta, gamma, x: x0, v: 0.0, a: 0.0, dt }
    }

    /// Ingest a position measurement `z`; returns the updated position estimate.
    pub fn update(&mut self, z: f64) -> f64 {
        let dt = self.dt;
        let x_pred = self.x + self.v * dt + 0.5 * self.a * dt * dt;
        let v_pred = self.v + self.a * dt;
        let r = z - x_pred;
        self.x = x_pred + self.alpha * r;
        self.v = v_pred + (self.beta / dt) * r;
        self.a += (2.0 * self.gamma / (dt * dt)) * r;
        self.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic small noise
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.2
    }

    #[test]
    fn alpha_beta_recovers_velocity_and_smooths_noise() {
        // THE ORACLE. A constant-velocity target measured with noise: the filter recovers the velocity and
        // its position estimate is closer to truth than the raw measurements.
        let dt = 0.1;
        let (x0, vel) = (0.0, 2.0);
        let mut f = AlphaBeta::new(0.5, 0.1, dt, x0);
        let mut seed = 1u64;
        let (mut sum_raw, mut sum_est, n) = (0.0, 0.0, 200);
        for k in 0..n {
            let truth = x0 + vel * (k as f64 * dt);
            let z = truth + noise(&mut seed);
            let est = f.update(z);
            if k > 50 {
                // after settling
                sum_raw += (z - truth).powi(2);
                sum_est += (est - truth).powi(2);
            }
        }
        // the fixed-gain velocity carries some residual measurement noise (inherent to β/Δt)
        assert!((f.v - vel).abs() < 0.1, "velocity should be recovered: {}", f.v);
        assert!(sum_est < sum_raw, "estimate should be smoother than raw: {sum_est} vs {sum_raw}");
    }

    #[test]
    fn alpha_beta_gamma_tracks_acceleration_without_lag() {
        // THE HEADLINE. On a constant-acceleration target the α-β filter develops a steady-state position
        // lag, while α-β-γ (which models acceleration) tracks it lag-free.
        let dt = 0.1;
        let acc = 1.0;
        let mut ab = AlphaBeta::new(0.6, 0.2, dt, 0.0);
        let mut abg = AlphaBetaGamma::new(0.6, 0.2, 0.05, dt, 0.0);
        let n = 300;
        let (mut ab_err, mut abg_err) = (0.0, 0.0);
        for k in 0..n {
            let truth = 0.5 * acc * (k as f64 * dt).powi(2);
            let e_ab = ab.update(truth); // noise-free to isolate the lag
            let e_abg = abg.update(truth);
            if k > 200 {
                ab_err += (e_ab - truth).abs();
                abg_err += (e_abg - truth).abs();
            }
        }
        assert!(abg_err < 0.05 * ab_err, "α-β-γ should track accel far better: {abg_err} vs {ab_err}");
        assert!((abg.a - acc).abs() < 0.05, "α-β-γ should recover the acceleration: {}", abg.a);
    }

    #[test]
    fn a_steady_measurement_converges_to_that_position() {
        // A constant measurement ⇒ the estimate converges to it and the velocity to zero.
        let mut f = AlphaBeta::new(0.4, 0.1, 0.1, 0.0);
        for _ in 0..500 {
            f.update(5.0);
        }
        assert!((f.x - 5.0).abs() < 1e-3 && f.v.abs() < 1e-3, "should settle at the measurement: x={} v={}", f.x, f.v);
    }
}
