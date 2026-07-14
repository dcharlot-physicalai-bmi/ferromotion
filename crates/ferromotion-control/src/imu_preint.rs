//! **IMU preintegration** on the `SO(3)` manifold — a clean-room implementation of Forster et al.,
//! *On-Manifold Preintegration for Real-Time Visual–Inertial Odometry* (T-RO 2017), the workhorse of
//! modern legged/visual-inertial state estimation.
//!
//! Between two keyframes, the high-rate gyro + accelerometer stream is compressed into three
//! **preintegrated measurements** — a relative rotation `ΔR`, velocity `Δv`, and position `Δp`, all in
//! the first keyframe's body frame and independent of gravity and of the initial state — so the
//! optimizer never re-integrates raw IMU inside its loop. First-order **bias Jacobians** let the
//! preintegral be corrected when the bias estimate changes, without re-integrating. The relative
//! state then reads `R_j = R_i ΔR`, `v_j = v_i + g Δt + R_i Δv`, `p_j = p_i + v_i Δt + ½ g Δt² + R_i Δp`.
//!
//! Fused with **leg-odometry** factors (a stance foot pins the relative base position via forward
//! kinematics), this corrects the accelerometer-bias drift that plagues IMU-only integration — the
//! basis of proprioceptive legged state estimation. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

fn hat(w: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

/// `SO(3)` exponential (Rodrigues).
pub fn exp_so3(w: Vector3<f64>) -> Matrix3<f64> {
    let t = w.norm();
    if t < 1e-9 {
        Matrix3::identity() + hat(w)
    } else {
        let k = hat(w / t);
        Matrix3::identity() + t.sin() * k + (1.0 - t.cos()) * k * k
    }
}

/// `SO(3)` right Jacobian.
pub fn right_jacobian(w: Vector3<f64>) -> Matrix3<f64> {
    let t = w.norm();
    if t < 1e-9 {
        Matrix3::identity() - 0.5 * hat(w)
    } else {
        let wx = hat(w);
        Matrix3::identity() - (1.0 - t.cos()) / (t * t) * wx + (t - t.sin()) / (t * t * t) * wx * wx
    }
}

/// A running IMU preintegration, linearized about biases `(bg, ba)`.
#[derive(Clone, Debug)]
pub struct ImuPreintegrator {
    pub dr: Matrix3<f64>,
    pub dv: Vector3<f64>,
    pub dp: Vector3<f64>,
    pub dt: f64,
    pub bg: Vector3<f64>,
    pub ba: Vector3<f64>,
    // First-order bias Jacobians.
    pub dr_dbg: Matrix3<f64>,
    pub dv_dbg: Matrix3<f64>,
    pub dv_dba: Matrix3<f64>,
    pub dp_dbg: Matrix3<f64>,
    pub dp_dba: Matrix3<f64>,
}

impl ImuPreintegrator {
    pub fn new(bg: Vector3<f64>, ba: Vector3<f64>) -> Self {
        Self {
            dr: Matrix3::identity(),
            dv: Vector3::zeros(),
            dp: Vector3::zeros(),
            dt: 0.0,
            bg,
            ba,
            dr_dbg: Matrix3::zeros(),
            dv_dbg: Matrix3::zeros(),
            dv_dba: Matrix3::zeros(),
            dp_dbg: Matrix3::zeros(),
            dp_dba: Matrix3::zeros(),
        }
    }

    /// Fold in one IMU sample (`gyro`, `accel`) over `dt`.
    pub fn integrate(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) {
        let w = gyro - self.bg;
        let a = accel - self.ba;
        let dr_inc = exp_so3(w * dt);
        let jr = right_jacobian(w * dt);
        let a_skew = hat(a);

        // Bias Jacobians (use the *current* ΔR and ΔR-bias-Jacobian, updated last).
        self.dp_dba += self.dv_dba * dt - 0.5 * self.dr * (dt * dt);
        self.dp_dbg += self.dv_dbg * dt - 0.5 * self.dr * a_skew * self.dr_dbg * (dt * dt);
        self.dv_dba += -self.dr * dt;
        self.dv_dbg += -self.dr * a_skew * self.dr_dbg * dt;
        self.dr_dbg = dr_inc.transpose() * self.dr_dbg - jr * dt;

        // States (ΔP uses old ΔV and ΔR; ΔV uses old ΔR; ΔR last).
        self.dp += self.dv * dt + 0.5 * self.dr * a * (dt * dt);
        self.dv += self.dr * a * dt;
        self.dr *= dr_inc;
        self.dt += dt;
    }

    /// Bias-corrected preintegrals for a new bias estimate `(bg, ba)` (first order).
    pub fn corrected(&self, bg: Vector3<f64>, ba: Vector3<f64>) -> (Matrix3<f64>, Vector3<f64>, Vector3<f64>) {
        let (dbg, dba) = (bg - self.bg, ba - self.ba);
        let dr = self.dr * exp_so3(self.dr_dbg * dbg);
        let dv = self.dv + self.dv_dbg * dbg + self.dv_dba * dba;
        let dp = self.dp + self.dp_dbg * dbg + self.dp_dba * dba;
        (dr, dv, dp)
    }

    /// Propagate a keyframe state `(R_i, v_i, p_i)` to the next using the preintegral and gravity.
    pub fn predict(&self, r_i: Matrix3<f64>, v_i: Vector3<f64>, p_i: Vector3<f64>, g: Vector3<f64>) -> (Matrix3<f64>, Vector3<f64>, Vector3<f64>) {
        let r_j = r_i * self.dr;
        let v_j = v_i + g * self.dt + r_i * self.dv;
        let p_j = p_i + v_i * self.dt + 0.5 * g * (self.dt * self.dt) + r_i * self.dp;
        (r_j, v_j, p_j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn log_so3(r: Matrix3<f64>) -> Vector3<f64> {
        let c = ((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
        let t = c.acos();
        if t < 1e-9 {
            Vector3::new(r[(2, 1)] - r[(1, 2)], r[(0, 2)] - r[(2, 0)], r[(1, 0)] - r[(0, 1)]) * 0.5
        } else {
            Vector3::new(r[(2, 1)] - r[(1, 2)], r[(0, 2)] - r[(2, 0)], r[(1, 0)] - r[(0, 1)]) * (t / (2.0 * t.sin()))
        }
    }

    // A synthetic IMU stream (gyro, specific-force accel) and its ground-truth direct integration.
    fn sim(steps: usize, dt: f64, g: Vector3<f64>, ba: Vector3<f64>) -> (Vec<(Vector3<f64>, Vector3<f64>)>, Matrix3<f64>, Vector3<f64>, Vector3<f64>) {
        let (mut r, mut v, mut p) = (Matrix3::identity(), Vector3::new(0.1, -0.2, 0.05), Vector3::zeros());
        let mut meas = Vec::new();
        for k in 0..steps {
            let tt = k as f64 * dt;
            let gyro = Vector3::new(0.3 * (0.7 * tt).sin(), 0.2, -0.15 * (0.5 * tt).cos());
            let a_world = Vector3::new(0.5 * (tt).cos(), 0.3, 0.4 * (1.3 * tt).sin());
            let accel_true = r.transpose() * (a_world - g); // specific force in body frame
            meas.push((gyro, accel_true + ba)); // measured accel carries the bias
            // Direct integration (same Euler scheme the preintegrator uses).
            p += v * dt + 0.5 * a_world * (dt * dt);
            v += a_world * dt;
            r *= exp_so3(gyro * dt);
        }
        (meas, r, v, p)
    }

    #[test]
    fn preintegration_reconstructs_direct_integration() {
        let (dt, g) = (1e-3, Vector3::new(0.0, 0.0, -9.81));
        let (meas, r_true, v_true, p_true) = sim(400, dt, g, Vector3::zeros());
        let mut pre = ImuPreintegrator::new(Vector3::zeros(), Vector3::zeros());
        for &(gyro, accel) in &meas {
            pre.integrate(gyro, accel, dt);
        }
        let (r0, v0, p0) = (Matrix3::identity(), Vector3::new(0.1, -0.2, 0.05), Vector3::zeros());
        let (rj, vj, pj) = pre.predict(r0, v0, p0, g);
        assert!(log_so3(rj.transpose() * r_true).norm() < 1e-9, "ΔR reconstruction off");
        assert!((vj - v_true).norm() < 1e-9, "Δv reconstruction off: {} ", (vj - v_true).norm());
        assert!((pj - p_true).norm() < 1e-9, "Δp reconstruction off: {}", (pj - p_true).norm());
    }

    #[test]
    fn bias_jacobians_match_reintegration() {
        let (dt, g) = (1e-3, Vector3::new(0.0, 0.0, -9.81));
        let (meas, ..) = sim(300, dt, g, Vector3::zeros());
        let reintegrate = |bg: Vector3<f64>, ba: Vector3<f64>| {
            let mut pre = ImuPreintegrator::new(bg, ba);
            for &(gyro, accel) in &meas {
                pre.integrate(gyro, accel, dt);
            }
            pre
        };
        let base = reintegrate(Vector3::zeros(), Vector3::zeros());
        let (dbg, dba) = (Vector3::new(1e-3, -2e-3, 5e-4), Vector3::new(-1e-3, 2e-3, 1e-3));
        // First-order correction vs a full re-integration at the perturbed bias.
        let (dr_c, dv_c, dp_c) = base.corrected(dbg, dba);
        let exact = reintegrate(dbg, dba);
        assert!(log_so3(dr_c.transpose() * exact.dr).norm() < 1e-5, "ΔR bias-Jacobian off");
        assert!((dv_c - exact.dv).norm() < 1e-5, "Δv bias-Jacobian off: {}", (dv_c - exact.dv).norm());
        assert!((dp_c - exact.dp).norm() < 1e-6, "Δp bias-Jacobian off: {}", (dp_c - exact.dp).norm());
    }

    #[test]
    fn leg_odometry_corrects_accelerometer_bias_drift() {
        // No rotation (gyro ≈ 0), an unknown accelerometer bias, and leg-odometry giving the true
        // per-interval displacement. Estimate the bias from the leg factors → IMU drift is corrected.
        let (dt, g) = (1e-3, Vector3::new(0.0, 0.0, -9.81));
        let ba_true = Vector3::new(0.08, -0.05, 0.03);
        let (k_int, m_int) = (200usize, 6usize);

        // Build M leg-odometry intervals; each contributes a preintegrated Δp Jacobian.
        let mut a_stack = DVector::zeros(3 * m_int); // residuals b
        let mut jac = nalgebra::DMatrix::zeros(3 * m_int, 3);
        let (mut r, mut v, mut p) = (Matrix3::identity(), Vector3::new(0.2, 0.0, 0.1), Vector3::zeros());
        for m in 0..m_int {
            let mut pre = ImuPreintegrator::new(Vector3::zeros(), Vector3::zeros());
            let (mut rt, mut vt, mut pt) = (r, v, p); // ground truth over the interval
            for k in 0..k_int {
                let tt = (m * k_int + k) as f64 * dt;
                let a_world = Vector3::new(0.4 * (0.6 * tt).cos(), 0.2 * (0.3 * tt).sin(), 0.3);
                let accel_meas = rt.transpose() * (a_world - g) + ba_true;
                pre.integrate(Vector3::zeros(), accel_meas, dt);
                pt += vt * dt + 0.5 * a_world * (dt * dt);
                vt += a_world * dt;
                let _ = &mut rt;
            }
            // Leg odometry measures the true interval displacement in the start-frame body coords.
            let dp_true = r.transpose() * (pt - p - v * pre.dt - 0.5 * g * (pre.dt * pre.dt));
            // Δp_pred(ba) ≈ Δp_pred(0) + dp_dba·ba ⇒ (Δp_true − Δp_pred0) = dp_dba·ba_true.
            let resid = dp_true - pre.dp;
            a_stack.rows_mut(3 * m, 3).copy_from(&resid);
            jac.view_mut((3 * m, 0), (3, 3)).copy_from(&pre.dp_dba);
            (r, v, p) = (rt, vt, pt);
        }
        // Least-squares bias estimate.
        let jtj = jac.transpose() * &jac;
        let sol = jtj.try_inverse().unwrap() * jac.transpose() * a_stack;
        let ba_est = Vector3::new(sol[0], sol[1], sol[2]);

        assert!((ba_est - ba_true).norm() < 1e-6, "bias estimate off: {ba_est:?} vs {ba_true:?}");

        // Drift check: integrating one interval with the estimated bias beats bias=0.
        let drift = |ba: Vector3<f64>| {
            let (mut vt, mut pt) = (Vector3::new(0.2, 0.0, 0.1), Vector3::zeros());
            let (mut vg, mut pg) = (vt, pt);
            for k in 0..k_int {
                let tt = k as f64 * dt;
                let a_world = Vector3::new(0.4 * (0.6 * tt).cos(), 0.2 * (0.3 * tt).sin(), 0.3);
                let accel_meas = a_world - g + ba_true; // R=I
                let a_est = accel_meas - ba + g; // corrected world accel
                pt += vt * dt + 0.5 * a_est * (dt * dt);
                vt += a_est * dt;
                pg += vg * dt + 0.5 * a_world * (dt * dt);
                vg += a_world * dt;
            }
            (pt - pg).norm()
        };
        assert!(drift(ba_est) < 1e-3 * drift(Vector3::zeros()).max(1e-9) + 1e-9, "estimated bias did not reduce drift");
    }
}
