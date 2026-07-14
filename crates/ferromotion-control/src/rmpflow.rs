//! **RMPflow** — Riemannian Motion Policies (Cheng et al., NVIDIA / WAFR 2018, T-RO 2021): reactive,
//! multi-task motion generation. Each task (reach a goal, avoid an obstacle, respect a joint limit)
//! is an **RMP** — a desired acceleration paired with a state-dependent **Riemannian metric** in its
//! own task space. Leaves are combined by a metric-weighted **pullback** to configuration space:
//!
//! ```text
//!   M_c = Σᵢ Jᵢᵀ Mᵢ Jᵢ ,   f_c = Σᵢ Jᵢᵀ (fᵢ − Mᵢ J̇ᵢ q̇) ,   q̈ = M_c⁺ f_c .
//! ```
//!
//! Because an obstacle's metric → ∞ as the distance → 0, that leaf dominates the pullback near the
//! obstacle — the robot bends around it while the attractor still pulls it to the goal, with no
//! explicit planner. Built on the `ferromotion-core` chain (real Jacobians). Pure `nalgebra` → WASM-clean.

use ferromotion_core::Robot;
use nalgebra::{DMatrix, DVector, Point3, Vector2, Vector3};

/// A reactive RMPflow controller for a planar arm: reach an end-effector goal while a set of control
/// points along the arm avoid a circular obstacle.
pub struct RmpArm<'a> {
    pub robot: &'a Robot,
    pub goal: Vector2<f64>,
    pub obstacle: Vector2<f64>,
    pub obstacle_r: f64,
    /// Control points `(frame, offset)` sampled along the links.
    pub control_points: Vec<(usize, Vector3<f64>)>,
    pub kp: f64,
    pub kd: f64,
    /// Obstacle influence distance and repulsion/damping gains.
    pub d0: f64,
    pub k_rep: f64,
    pub kd_obs: f64,
    pub w_attract: f64,
}

impl RmpArm<'_> {
    fn ee_xy(&self, q: &[f64]) -> Vector2<f64> {
        let p = self.robot.fk(q).translation.vector;
        Vector2::new(p.x, p.y)
    }

    fn cp_world(&self, q: &[f64], frame: usize, offset: Vector3<f64>) -> Vector2<f64> {
        let w = (self.robot.frame_pose(q, frame) * Point3::from(offset)).coords;
        Vector2::new(w.x, w.y)
    }

    /// Smallest control-point clearance to the obstacle surface (negative ⇒ collision).
    pub fn min_clearance(&self, q: &[f64]) -> f64 {
        self.control_points
            .iter()
            .map(|&(f, o)| (self.cp_world(q, f, o) - self.obstacle).norm() - self.obstacle_r)
            .fold(f64::INFINITY, f64::min)
    }

    /// RMPflow configuration-space acceleration at `(q, q̇)`.
    pub fn accel(&self, q: &[f64], qd: &[f64]) -> DVector<f64> {
        let n = self.robot.dof();
        let qd_v = DVector::from_row_slice(qd);
        let mut mc = DMatrix::<f64>::zeros(n, n);
        let mut fc = DVector::<f64>::zeros(n);
        let h = 1e-6;

        // Finite-difference J̇q̇ for a task Jacobian.
        let jdot_qd = |jac: &dyn Fn(&[f64]) -> DMatrix<f64>| -> DVector<f64> {
            let qp: Vec<f64> = (0..n).map(|i| q[i] + h * qd[i]).collect();
            let qm: Vec<f64> = (0..n).map(|i| q[i] - h * qd[i]).collect();
            (jac(&qp) - jac(&qm)) / (2.0 * h) * &qd_v
        };

        // --- attractor RMP on the end-effector (fk includes the tool offset) ---
        let ee = self.ee_xy(q);
        let jee_fn = |qq: &[f64]| {
            let p = self.robot.fk(qq).translation.vector;
            self.robot.point_jacobian(qq, self.robot.dof(), &p).rows(0, 2).into_owned()
        };
        let jee = jee_fn(q); // 2×n (DMatrix)
        let xdot = &jee * &qd_v; // 2
        let a = DVector::from_row_slice(&[
            self.kp * (self.goal.x - ee.x) - self.kd * xdot[0],
            self.kp * (self.goal.y - ee.y) - self.kd * xdot[1],
        ]);
        let m = DMatrix::<f64>::identity(2, 2) * self.w_attract;
        let jd_qd = jdot_qd(&jee_fn); // 2
        mc += jee.transpose() * &m * &jee;
        fc += jee.transpose() * (&m * (&a - &jd_qd));

        // --- obstacle-avoidance RMPs at every control point ---
        for &(frame, offset) in &self.control_points {
            let xy = self.cp_world(q, frame, offset);
            let diff = xy - self.obstacle;
            let dist = diff.norm();
            let d = dist - self.obstacle_r;
            if d >= self.d0 || d <= 0.0 {
                continue; // outside influence, or already touching
            }
            // 1D task: signed distance, increasing away from the obstacle. Its config Jacobian is
            // jd = n̂ᵀ Jₓ, so the config-space gradient column is Jₓᵀ n̂ (n-vector).
            let nfn = |qq: &[f64]| {
                let w2 = (self.robot.frame_pose(qq, frame) * Point3::from(offset)).coords;
                let dv = Vector2::new(w2.x, w2.y) - self.obstacle;
                let nh = dv / dv.norm();
                let jx2 = self.robot.point_jacobian(qq, frame, &Vector3::new(w2.x, w2.y, w2.z)).rows(0, 2).into_owned();
                jx2.transpose() * DVector::from_row_slice(&[nh.x, nh.y]) // n-vector = jdᵀ
            };
            let jd_col = nfn(q); // n
            let ddot = jd_col.dot(&qd_v);
            // Barrier metric → ∞ as d → 0; repulsion grows the distance; approach damping.
            let w = (1.0 / d - 1.0 / self.d0).powi(2);
            let a_obs = self.k_rep * (1.0 / d - 1.0 / self.d0) / (d * d) - self.kd_obs * ddot.min(0.0);
            // J̇_d q̇ (scalar) by finite difference of the gradient column.
            let qp: Vec<f64> = (0..n).map(|i| q[i] + h * qd[i]).collect();
            let qm: Vec<f64> = (0..n).map(|i| q[i] - h * qd[i]).collect();
            let jdq = ((nfn(&qp) - nfn(&qm)) / (2.0 * h)).dot(&qd_v);
            mc += w * &jd_col * jd_col.transpose();
            fc += &jd_col * (w * (a_obs - jdq));
        }

        // q̈ = M_c⁺ f_c (regularized).
        let reg = &mc + DMatrix::identity(n, n) * 1e-6;
        reg.try_inverse().map(|inv| inv * fc).unwrap_or_else(|| DVector::zeros(n))
    }

    /// Roll the policy out from `(q0, q̇0)`; returns the final `q` and the min clearance over the run.
    pub fn simulate(&self, q0: &[f64], dt: f64, steps: usize) -> (Vec<f64>, f64) {
        let n = self.robot.dof();
        let (mut q, mut qd) = (q0.to_vec(), vec![0.0; n]);
        let mut worst = self.min_clearance(&q);
        for _ in 0..steps {
            let a = self.accel(&q, &qd);
            for i in 0..n {
                qd[i] += a[i] * dt;
                q[i] += qd[i] * dt;
            }
            worst = worst.min(self.min_clearance(&q));
        }
        (q, worst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferromotion_core::{from_urdf_str, Robot};

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/><link name="l1"/><link name="l2"/><link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="1 0 0"/></joint>
    </robot>"#;

    fn control_points() -> Vec<(usize, Vector3<f64>)> {
        let mut cps = Vec::new();
        for s in [0.5, 1.0] {
            cps.push((1usize, Vector3::new(s, 0.0, 0.0))); // along link 1
        }
        for s in [0.5, 1.0] {
            cps.push((2usize, Vector3::new(s, 0.0, 0.0))); // along link 2 (1.0 = EE)
        }
        cps
    }

    #[test]
    fn reaches_goal_while_avoiding_obstacle() {
        let robot: Robot = from_urdf_str(ARM2, "base", "tool").unwrap();
        // Start folded up (+y); goal folded down (−y); obstacle sits on the +x path between them.
        let arm = RmpArm {
            robot: &robot,
            goal: Vector2::new(0.6, -1.4),
            obstacle: Vector2::new(1.5, 0.0),
            obstacle_r: 0.4,
            control_points: control_points(),
            kp: 8.0,
            kd: 5.0,
            d0: 0.5,
            k_rep: 0.5,
            kd_obs: 2.0,
            w_attract: 1.0,
        };
        let q0 = [0.9, 0.6];
        let goal_reachable = {
            // sanity: the goal is within reach (|goal| < 2)
            arm.goal.norm() < 2.0
        };
        assert!(goal_reachable);

        let (qf, worst) = arm.simulate(&q0, 2e-3, 6000);
        let ee = arm.ee_xy(&qf);
        let err = (ee - arm.goal).norm();
        eprintln!("RMPflow: ee={ee:?}, goal={:?}, err={err:.3}, min_clearance={worst:.3}", arm.goal);
        // Never collided (the obstacle metric blow-up guarantees clearance stays positive) …
        assert!(worst > 0.0, "arm collided with the obstacle: min clearance {worst:.3}");
        // … the obstacle was actually in play (clearance got small — avoidance was active) …
        assert!(worst < arm.d0, "obstacle never engaged (clearance {worst}); test not exercising avoidance");
        // … and the end-effector reached the goal.
        assert!(err < 0.1, "did not reach goal: EE error {err:.3}");
    }
}
