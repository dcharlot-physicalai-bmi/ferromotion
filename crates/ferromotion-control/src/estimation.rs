//! State estimation — a linear Kalman filter, an extended (EKF), and an unscented (UKF) filter.
//!
//! All three are generic over dynamic-size state via `nalgebra` `DVector`/`DMatrix`, so they slot
//! onto any of the dynamics/control blocks in this crate. Nonlinear models are supplied as plain
//! closures `state → state` / `state → measurement`; the EKF takes either an analytic Jacobian or a
//! numerical one via [`numerical_jacobian`], and the UKF uses Van der Merwe scaled sigma points.
//! Pure `nalgebra`, WASM-clean.

use nalgebra::{DMatrix, DVector};

/// Kalman measurement correction shared by the linear KF and the EKF (Joseph form keeps `P`
/// symmetric positive-definite). `y` is the innovation, `h` the (linearized) measurement matrix.
fn joseph_correct(x: &mut DVector<f64>, p: &mut DMatrix<f64>, y: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
    let ht = h.transpose();
    let s = h * &*p * &ht + r; // innovation covariance
    let s_inv = s.try_inverse().expect("innovation covariance is singular");
    let k = &*p * &ht * s_inv; // Kalman gain
    *x += &k * y;
    let n = p.nrows();
    let ikh = DMatrix::identity(n, n) - &k * h;
    *p = &ikh * &*p * ikh.transpose() + &k * r * k.transpose();
}

/// Central-difference Jacobian of a vector map `f: ℝⁿ → ℝᵐ` at `x` (`m = out_dim`). Handy when an
/// analytic Jacobian for the EKF isn't available.
pub fn numerical_jacobian<F: Fn(&DVector<f64>) -> DVector<f64>>(f: &F, x: &DVector<f64>, out_dim: usize) -> DMatrix<f64> {
    let n = x.len();
    let eps = 1e-6;
    let mut j = DMatrix::zeros(out_dim, n);
    for i in 0..n {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let d = (f(&xp) - f(&xm)) / (2.0 * eps);
        j.set_column(i, &d);
    }
    j
}

/// Linear Kalman filter for `x' = F·x (+ w)`, `z = H·x (+ v)`.
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    pub x: DVector<f64>,
    pub p: DMatrix<f64>,
}

impl KalmanFilter {
    pub fn new(x0: DVector<f64>, p0: DMatrix<f64>) -> Self {
        Self { x: x0, p: p0 }
    }

    /// Time update: `x ← F·x`, `P ← F·P·Fᵀ + Q`.
    pub fn predict(&mut self, f: &DMatrix<f64>, q: &DMatrix<f64>) {
        self.x = f * &self.x;
        self.p = f * &self.p * f.transpose() + q;
    }

    /// Measurement update with observation `z`, matrix `H`, noise covariance `R`.
    pub fn update(&mut self, z: &DVector<f64>, h: &DMatrix<f64>, r: &DMatrix<f64>) {
        let y = z - h * &self.x;
        joseph_correct(&mut self.x, &mut self.p, &y, h, r);
    }

    pub fn state(&self) -> &DVector<f64> {
        &self.x
    }
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.p
    }
}

/// Extended Kalman filter: nonlinear `x' = f(x)`, `z = h(x)`, linearized about the current estimate.
#[derive(Clone, Debug)]
pub struct Ekf {
    pub x: DVector<f64>,
    pub p: DMatrix<f64>,
}

impl Ekf {
    pub fn new(x0: DVector<f64>, p0: DMatrix<f64>) -> Self {
        Self { x: x0, p: p0 }
    }

    /// Time update with an analytic process Jacobian `f_jac = ∂f/∂x` evaluated at the current state.
    pub fn predict<F: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, f: F, f_jac: &DMatrix<f64>, q: &DMatrix<f64>) {
        self.x = f(&self.x);
        self.p = f_jac * &self.p * f_jac.transpose() + q;
    }

    /// Time update that forms the process Jacobian numerically about the current state.
    pub fn predict_auto<F: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, f: F, q: &DMatrix<f64>) {
        let fj = numerical_jacobian(&f, &self.x, self.x.len());
        self.x = f(&self.x);
        self.p = &fj * &self.p * fj.transpose() + q;
    }

    /// Measurement update with an analytic measurement Jacobian `h_jac = ∂h/∂x`.
    pub fn update<H: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, z: &DVector<f64>, h: H, h_jac: &DMatrix<f64>, r: &DMatrix<f64>) {
        let y = z - h(&self.x);
        joseph_correct(&mut self.x, &mut self.p, &y, h_jac, r);
    }

    /// Measurement update that forms the measurement Jacobian numerically about the current state.
    pub fn update_auto<H: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, z: &DVector<f64>, h: H, r: &DMatrix<f64>) {
        let hj = numerical_jacobian(&h, &self.x, z.len());
        let y = z - h(&self.x);
        joseph_correct(&mut self.x, &mut self.p, &y, &hj, r);
    }

    pub fn state(&self) -> &DVector<f64> {
        &self.x
    }
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.p
    }
}

/// Unscented Kalman filter with Van der Merwe scaled sigma points. Propagates `2n+1` points through
/// the nonlinear model rather than linearizing, capturing mean/covariance to second order.
#[derive(Clone, Debug)]
pub struct Ukf {
    pub x: DVector<f64>,
    pub p: DMatrix<f64>,
    n: usize,
    gamma: f64, // √(n + λ), the sigma-point spread
    wm: Vec<f64>,
    wc: Vec<f64>,
}

impl Ukf {
    /// `alpha` (spread, ~1e-3..1), `beta` (2 is optimal for Gaussians), `kappa` (secondary scaling,
    /// often 0 or 3−n).
    pub fn new(x0: DVector<f64>, p0: DMatrix<f64>, alpha: f64, beta: f64, kappa: f64) -> Self {
        let n = x0.len();
        let nf = n as f64;
        let lambda = alpha * alpha * (nf + kappa) - nf;
        let denom = nf + lambda;
        let count = 2 * n + 1;
        let mut wm = vec![0.5 / denom; count];
        let mut wc = vec![0.5 / denom; count];
        wm[0] = lambda / denom;
        wc[0] = lambda / denom + (1.0 - alpha * alpha + beta);
        Self { x: x0, p: p0, n, gamma: denom.sqrt(), wm, wc }
    }

    /// `2n+1` sigma points about the current mean, spread by the matrix square root of `P`.
    fn sigma_points(&self) -> Vec<DVector<f64>> {
        let l = self.p.clone().cholesky().expect("covariance P is not positive-definite").l();
        let mut pts = Vec::with_capacity(2 * self.n + 1);
        pts.push(self.x.clone());
        for i in 0..self.n {
            let col = l.column(i) * self.gamma;
            pts.push(&self.x + &col);
            pts.push(&self.x - &col);
        }
        pts
    }

    /// Time update: push sigma points through `f`, recover the predicted mean and covariance `+ Q`.
    pub fn predict<F: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, f: F, q: &DMatrix<f64>) {
        let prop: Vec<DVector<f64>> = self.sigma_points().iter().map(|s| f(s)).collect();
        let mut xm = DVector::zeros(self.n);
        for (w, p) in self.wm.iter().zip(&prop) {
            xm += p.scale(*w);
        }
        let mut pc = q.clone();
        for (w, p) in self.wc.iter().zip(&prop) {
            let d = p - &xm;
            pc += (&d * d.transpose()).scale(*w);
        }
        self.x = xm;
        self.p = pc;
    }

    /// Measurement update: push sigma points through `h`, form innovation/cross covariances, correct.
    pub fn update<H: Fn(&DVector<f64>) -> DVector<f64>>(&mut self, z: &DVector<f64>, h: H, r: &DMatrix<f64>) {
        let m = z.len();
        let pts = self.sigma_points();
        let zpts: Vec<DVector<f64>> = pts.iter().map(|s| h(s)).collect();
        let mut zm = DVector::zeros(m);
        for (w, zp) in self.wm.iter().zip(&zpts) {
            zm += zp.scale(*w);
        }
        let mut s = r.clone(); // innovation covariance
        let mut pxz = DMatrix::zeros(self.n, m); // cross covariance
        for i in 0..pts.len() {
            let wc = self.wc[i];
            let dz = &zpts[i] - &zm;
            let dx = &pts[i] - &self.x;
            s += (&dz * dz.transpose()).scale(wc);
            pxz += (&dx * dz.transpose()).scale(wc);
        }
        let s_inv = s.clone().try_inverse().expect("innovation covariance is singular");
        let k = &pxz * &s_inv;
        self.x += &k * (z - &zm);
        self.p -= &k * &s * k.transpose();
    }

    pub fn state(&self) -> &DVector<f64> {
        &self.x
    }
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG → standard-normal samples (Box–Muller). No `rand`, fully reproducible.
    struct Lcg {
        s: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.s
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn numerical_jacobian_matches_analytic() {
        // f(x) = [x0^2 + x1, sin(x0)*x1]  →  J = [[2x0, 1], [cos(x0)x1, sin(x0)]]
        let f = |x: &DVector<f64>| DVector::from_vec(vec![x[0] * x[0] + x[1], x[0].sin() * x[1]]);
        let x = DVector::from_vec(vec![0.7, -1.3]);
        let j = numerical_jacobian(&f, &x, 2);
        let a = DMatrix::from_row_slice(2, 2, &[2.0 * 0.7, 1.0, 0.7_f64.cos() * -1.3, 0.7_f64.sin()]);
        assert!((j - a).norm() < 1e-6, "numerical Jacobian off");
    }

    #[test]
    fn kalman_tracks_double_integrator_better_than_raw() {
        // Constant-velocity model, position-only measurements. True system is a random-walk in
        // acceleration (the process noise), measured with additive noise.
        let dt = 0.1_f64;
        let f = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let sa = 0.4_f64; // process accel std
        let sa2 = sa * sa;
        let q = DMatrix::from_row_slice(
            2,
            2,
            &[sa2 * dt.powi(4) / 4.0, sa2 * dt.powi(3) / 2.0, sa2 * dt.powi(3) / 2.0, sa2 * dt.powi(2)],
        );
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let sz = 0.5; // measurement std
        let r = DMatrix::from_row_slice(1, 1, &[sz * sz]);

        let mut kf = KalmanFilter::new(DVector::from_vec(vec![0.0, 0.0]), DMatrix::identity(2, 2));
        let mut rng = Lcg::new(0x1234_5678_9abc_def0);
        let (mut pos, mut vel) = (0.0f64, 1.0f64);
        let (mut se_f, mut se_raw, mut count) = (0.0, 0.0, 0u32);

        for k in 0..500 {
            // ground truth
            vel += sa * rng.gauss() * dt;
            pos += vel * dt;
            // filter
            kf.predict(&f, &q);
            let z = pos + sz * rng.gauss();
            kf.update(&DVector::from_vec(vec![z]), &h, &r);
            if k >= 60 {
                se_f += (kf.state()[0] - pos).powi(2);
                se_raw += (z - pos).powi(2);
                count += 1;
            }
        }
        let rmse_f = (se_f / count as f64).sqrt();
        let rmse_raw = (se_raw / count as f64).sqrt();
        assert!(rmse_raw > 0.35 && rmse_raw < 0.65, "sanity: raw RMSE ≈ σ_z, got {rmse_raw}");
        assert!(rmse_f < rmse_raw, "filter {rmse_f} not better than raw {rmse_raw}");
        assert!(rmse_f < 0.6 * sz, "filter RMSE {rmse_f} not clearly below meas noise {sz}");
    }

    // Pendulum: θ̈ = −(g/L)·sinθ − c·θ̇. State x = [θ, ω]. Semi-implicit-flavoured explicit Euler.
    const DT: f64 = 0.02;
    const G_L: f64 = 9.81;
    const DAMP: f64 = 0.15;

    fn pend_step(x: &DVector<f64>) -> DVector<f64> {
        let (th, om) = (x[0], x[1]);
        DVector::from_vec(vec![th + om * DT, om + (-G_L * th.sin() - DAMP * om) * DT])
    }

    #[test]
    fn ekf_tracks_pendulum_angle_better_than_raw() {
        // Nonlinear dynamics, *linear* angle measurement — RMSE is directly comparable to σ.
        let q = DMatrix::from_row_slice(2, 2, &[1e-7, 0.0, 0.0, 4e-4]);
        let sm = 0.1; // angle-measurement std (rad)
        let r = DMatrix::from_row_slice(1, 1, &[sm * sm]);
        let h_fn = |x: &DVector<f64>| DVector::from_vec(vec![x[0]]);
        let h_jac = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        let mut ekf = Ekf::new(DVector::from_vec(vec![0.0, 0.0]), DMatrix::identity(2, 2) * 0.1);
        let mut rng = Lcg::new(0xdead_beef_0000_0001);
        let mut xt = DVector::from_vec(vec![0.8, 0.0]); // true state
        let (mut se_f, mut se_raw, mut count) = (0.0, 0.0, 0u32);

        for k in 0..800 {
            // ground truth with process (torque) noise on ω
            xt = pend_step(&xt);
            xt[1] += 0.02 * rng.gauss();
            // predict with analytic Jacobian evaluated at current estimate
            let xe = ekf.state().clone();
            let fj = DMatrix::from_row_slice(2, 2, &[1.0, DT, -G_L * xe[0].cos() * DT, 1.0 - DAMP * DT]);
            ekf.predict(pend_step, &fj, &q);
            // measure angle, correct
            let z = xt[0] + sm * rng.gauss();
            ekf.update(&DVector::from_vec(vec![z]), h_fn, &h_jac, &r);
            if k >= 100 {
                se_f += (ekf.state()[0] - xt[0]).powi(2);
                se_raw += (z - xt[0]).powi(2);
                count += 1;
            }
        }
        let rmse_f = (se_f / count as f64).sqrt();
        let rmse_raw = (se_raw / count as f64).sqrt();
        assert!(rmse_raw > 0.07 && rmse_raw < 0.13, "sanity: raw RMSE ≈ σ, got {rmse_raw}");
        assert!(rmse_f < rmse_raw, "EKF {rmse_f} not better than raw {rmse_raw}");
        assert!(rmse_f < 0.6 * sm, "EKF RMSE {rmse_f} not clearly below meas noise {sm}");
    }

    #[test]
    fn ukf_tracks_pendulum_through_nonlinear_measurement() {
        // Nonlinear dynamics AND a nonlinear measurement z = sin(θ) (bob horizontal offset, L=1).
        // Baseline: invert the measurement, θ̂_raw = asin(z). UKF fuses dynamics and should beat it.
        let q = DMatrix::from_row_slice(2, 2, &[1e-7, 0.0, 0.0, 4e-4]);
        let sm = 0.05; // measurement std on sin(θ)
        let r = DMatrix::from_row_slice(1, 1, &[sm * sm]);
        let h_fn = |x: &DVector<f64>| DVector::from_vec(vec![x[0].sin()]);

        // α=1, κ=3−n → the classic well-conditioned scaling (λ=1, positive spread & weights).
        let mut ukf = Ukf::new(DVector::from_vec(vec![0.0, 0.0]), DMatrix::identity(2, 2) * 0.1, 1.0, 2.0, 1.0);
        let mut rng = Lcg::new(0x00c0_ffee_1234_5678);
        let mut xt = DVector::from_vec(vec![0.8, 0.0]);
        let (mut se_f, mut se_raw, mut count) = (0.0, 0.0, 0u32);

        for k in 0..800 {
            xt = pend_step(&xt);
            xt[1] += 0.02 * rng.gauss();
            ukf.predict(pend_step, &q);
            let z = xt[0].sin() + sm * rng.gauss();
            ukf.update(&DVector::from_vec(vec![z]), h_fn, &r);
            if k >= 100 {
                let raw = z.clamp(-1.0, 1.0).asin();
                se_f += (ukf.state()[0] - xt[0]).powi(2);
                se_raw += (raw - xt[0]).powi(2);
                count += 1;
            }
        }
        let rmse_f = (se_f / count as f64).sqrt();
        let rmse_raw = (se_raw / count as f64).sqrt();
        assert!(rmse_f < rmse_raw, "UKF {rmse_f} not better than raw asin {rmse_raw}");
        assert!(rmse_f < 0.08, "UKF angle RMSE {rmse_f} not small");
    }
}
