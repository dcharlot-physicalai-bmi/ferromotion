//! placo-style whole-body task-space IK: a weighted stack of Cartesian/posture tasks solved as one
//! velocity-level QP with **hard** joint-position limits, then integrated `q += q̇·dt`. This is the
//! classic placo / pink QP-IK niche — kinematics only, no dynamics.
//!
//! Each step minimizes `½ Σ wᵢ‖Jᵢ·q̇ − bᵢ‖² + λ‖q̇‖²` over joint velocities `q̇`, subject to the box
//! `q̇ ∈ [(lo−q)/dt, (hi−q)/dt] ∩ [−vmax, vmax]`. Mapping the joint-position limits into per-step
//! velocity bounds and solving with [`crate::qp::solve_box_qp`] makes the limits a *hard* constraint:
//! the returned `q̇` cannot carry any joint past its declared range in that step. That is the whole
//! point — unlike a soft [`ferromotion_core::JointLimitCost`] penalty, a joint pinned against its limit
//! stays exactly at the limit while the remaining DoF absorb as much of the task as they can.
//!
//! Distinct from [`crate::WholeBody`], which is the *acceleration*-level QP mapped to torques via
//! inverse dynamics. Here everything lives at the velocity/kinematics level.

use crate::qp::solve_box_qp;
use nalgebra::{DMatrix, DVector, Point3, Vector3};
use ferromotion_core::{Iso, Robot};

/// One task in the placo stack. All are least-squares terms; `weight` sets their relative priority
/// (a large weight ≈ a hard task, a small one ≈ a soft regularizer resolved in the null space).
#[derive(Clone, Debug)]
pub enum PlacoTask {
    /// Drive the point `frame`+`offset` toward `target` (3-DoF). `gain` shapes the closed-loop rate:
    /// the desired point velocity is `gain·(target − point)`.
    Position { frame: usize, offset: Vector3<f64>, target: Vector3<f64>, gain: f64, weight: f64 },
    /// Drive the end-effector pose (translation + orientation, 6-DoF) toward `target`. The desired
    /// twist is `−gain·[Δp; log(R·R_target⁻¹)]`.
    Pose { target: Iso, gain: f64, weight: f64 },
    /// Regularize toward a rest posture: pull `q̇` toward `rest − q` with the given `weight`.
    Posture { rest: Vec<f64>, weight: f64 },
}

/// A velocity-level, hard-limit whole-body IK solver over a stack of [`PlacoTask`]s.
#[derive(Clone, Debug)]
pub struct PlacoSolver {
    pub tasks: Vec<PlacoTask>,
    /// Integration / limit-mapping timestep (seconds).
    pub dt: f64,
    /// Per-joint velocity magnitude cap (rad/s or m/s), intersected with the position-limit bounds.
    pub vmax: f64,
    /// Tikhonov damping `λ` on `q̇` — keeps the QP strictly convex and the motion regular near
    /// singularities.
    pub damping: f64,
    /// Enforce joint-position limits as hard velocity bounds (the placo headline feature).
    pub use_limits: bool,
}

impl Default for PlacoSolver {
    fn default() -> Self {
        Self { tasks: Vec::new(), dt: 0.02, vmax: 3.0, damping: 1e-4, use_limits: true }
    }
}

/// Outcome of a multi-step [`PlacoSolver::solve`].
#[derive(Clone, Debug)]
pub struct PlacoResult {
    pub q: Vec<f64>,
    /// Worst-case Cartesian task error at the final configuration (max over position/pose tasks).
    pub error: f64,
    pub iters: usize,
    pub converged: bool,
}

impl PlacoSolver {
    /// The largest Cartesian task error at `q` (max over Position/Pose tasks; posture ignored).
    pub fn task_error(&self, robot: &Robot, q: &[f64]) -> f64 {
        let mut worst = 0.0f64;
        for t in &self.tasks {
            let e = match t {
                PlacoTask::Position { frame, offset, target, .. } => {
                    let p = (robot.frame_pose(q, *frame) * Point3::from(*offset)).coords;
                    (target - p).norm()
                }
                PlacoTask::Pose { target, .. } => {
                    let cur = robot.fk(q);
                    let dp = cur.translation.vector - target.translation.vector;
                    let dr = (cur.rotation * target.rotation.inverse()).scaled_axis();
                    (dp.norm().powi(2) + dr.norm().powi(2)).sqrt()
                }
                PlacoTask::Posture { .. } => continue,
            };
            worst = worst.max(e);
        }
        worst
    }

    /// Solve the one-step QP for the joint velocity `q̇` at configuration `q`. The returned velocity
    /// is guaranteed to lie inside the hard box bounds (position limits mapped through `dt`, and
    /// `±vmax`), so a single `q += q̇·dt` step cannot violate any declared joint limit.
    pub fn velocity(&self, robot: &Robot, q: &[f64]) -> Vec<f64> {
        let n = robot.dof();
        let mut h = DMatrix::<f64>::identity(n, n) * self.damping;
        let mut g = DVector::<f64>::zeros(n);

        for t in &self.tasks {
            match t {
                PlacoTask::Position { frame, offset, target, gain, weight } => {
                    let (gain, weight) = (*gain, *weight);
                    let p = (robot.frame_pose(q, *frame) * Point3::from(*offset)).coords;
                    let j = robot.point_jacobian(q, *frame, &p); // 3×n
                    let b = (target - p) * gain; // desired point velocity
                    h += weight * (j.transpose() * &j);
                    g -= weight * (j.transpose() * b);
                }
                PlacoTask::Pose { target, gain, weight } => {
                    let (gain, weight) = (*gain, *weight);
                    let cur = robot.fk(q);
                    let dp = cur.translation.vector - target.translation.vector;
                    let dr = (cur.rotation * target.rotation.inverse()).scaled_axis();
                    // Desired twist that drives the 6-DoF pose error to zero.
                    let b = DVector::from_row_slice(&[
                        -gain * dp.x, -gain * dp.y, -gain * dp.z,
                        -gain * dr.x, -gain * dr.y, -gain * dr.z,
                    ]);
                    let j = robot.jacobian(q); // 6×n, rows 0..3 linear, 3..6 angular
                    h += weight * (j.transpose() * &j);
                    g -= weight * (j.transpose() * b);
                }
                PlacoTask::Posture { rest, weight } => {
                    let weight = *weight;
                    // ½w‖q̇ − (rest − q)‖² → adds w·I to H and −w·(rest − q) to g.
                    for i in 0..n {
                        h[(i, i)] += weight;
                        g[i] -= weight * (rest[i] - q[i]);
                    }
                }
            }
        }
        h = 0.5 * (&h + &h.transpose()); // symmetrize for the QP backend

        // Hard box: intersect the velocity cap with the position-limit-derived per-step bounds.
        let (mut lo, mut hi) = (vec![-self.vmax; n], vec![self.vmax; n]);
        if self.use_limits {
            for (i, joint) in robot.joints.iter().enumerate() {
                if let Some((lower, upper)) = joint.limits {
                    hi[i] = hi[i].min((upper - q[i]) / self.dt);
                    lo[i] = lo[i].max((lower - q[i]) / self.dt);
                    if lo[i] > hi[i] {
                        // Only reachable if q already sits (numerically) outside the range; collapse
                        // the box to the nearer face so the step drives strictly back inside.
                        let mid = 0.5 * (lo[i] + hi[i]);
                        lo[i] = mid;
                        hi[i] = mid;
                    }
                }
            }
        }

        let g_lin: Vec<f64> = g.iter().cloned().collect();
        let mut qd = solve_box_qp(&h, &g_lin, &lo, &hi);
        // Defensive clamp: the interior-point solve satisfies the box only to its tolerance; project
        // back so the hard-limit guarantee holds exactly under floating point.
        for i in 0..n {
            qd[i] = qd[i].clamp(lo[i], hi[i]);
        }
        qd
    }

    /// One closed-loop step: solve the QP and integrate `q += q̇·dt`. Returns the new configuration.
    pub fn step(&self, robot: &Robot, q: &[f64]) -> Vec<f64> {
        let qd = self.velocity(robot, q);
        (0..robot.dof()).map(|i| q[i] + qd[i] * self.dt).collect()
    }

    /// Run `iters` steps from `q0`, stopping early once the worst Cartesian task error is below `tol`.
    pub fn solve(&self, robot: &Robot, q0: &[f64], iters: usize, tol: f64) -> PlacoResult {
        let mut q = q0.to_vec();
        let mut used = 0;
        for it in 0..iters {
            used = it + 1;
            if self.task_error(robot, &q) < tol {
                break;
            }
            q = self.step(robot, &q);
        }
        let error = self.task_error(robot, &q);
        PlacoResult { q, error, iters: used, converged: error < tol }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferromotion_core::from_urdf_str;

    const ARM6: &str = r#"<robot name="arm6"><link name="world"/><link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/><joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint><joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

    // Tool tip = frame 6 + offset (0,0,0.05).
    fn tool_offset() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, 0.05)
    }
    fn tool_point(robot: &Robot, q: &[f64]) -> Vector3<f64> {
        (robot.frame_pose(q, 6) * Point3::from(tool_offset())).coords
    }

    /// Assert every joint of `q` is within its declared limits (exact, up to floating point).
    fn assert_within_limits(robot: &Robot, q: &[f64]) {
        for (i, joint) in robot.joints.iter().enumerate() {
            if let Some((lo, hi)) = joint.limits {
                assert!(
                    q[i] >= lo - 1e-9 && q[i] <= hi + 1e-9,
                    "joint {i} = {} violated limits [{lo}, {hi}]",
                    q[i]
                );
            }
        }
    }

    #[test]
    fn reaches_a_reachable_target_within_loose_limits() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let q_ref = [0.3, -0.5, 0.6, 0.2, 0.4, -0.3];
        let target = tool_point(&robot, &q_ref);
        let solver = PlacoSolver {
            tasks: vec![PlacoTask::Position {
                frame: 6,
                offset: tool_offset(),
                target,
                gain: 4.0,
                weight: 1.0,
            }],
            ..PlacoSolver::default()
        };
        let mut q = vec![0.0; 6];
        for _ in 0..400 {
            q = solver.step(&robot, &q);
            assert_within_limits(&robot, &q); // never violate, every step
        }
        let err = (tool_point(&robot, &q) - target).norm();
        assert!(err < 1e-3, "did not reach reachable target: err = {err}");
    }

    #[test]
    fn hard_limit_never_violated_when_target_demands_it() {
        // Tighten the base yaw (joint index 0, axis z) to a narrow window, then command a target
        // that can only be reached with a large yaw. The solver MUST keep every joint in range at
        // every step while still driving the error down as far as the feasible set allows.
        let mut robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        robot.joints[0].limits = Some((-0.1, 0.1));

        // Target = tool pose at a yaw of ~1.0 rad (far outside the narrowed [-0.1, 0.1] window).
        let q_demand = [1.0, -0.5, 0.6, 0.2, 0.4, -0.3];
        let target = tool_point(&robot, &q_demand);

        let solver = PlacoSolver {
            tasks: vec![PlacoTask::Position {
                frame: 6,
                offset: tool_offset(),
                target,
                gain: 4.0,
                weight: 1.0,
            }],
            dt: 0.02,
            vmax: 3.0,
            damping: 1e-4,
            use_limits: true,
        };

        let mut q = vec![0.0; 6];
        let err0 = (tool_point(&robot, &q) - target).norm();
        for _ in 0..600 {
            q = solver.step(&robot, &q);
            // CORE PROPERTY: no joint ever leaves its declared limits, despite the demanding target.
            assert_within_limits(&robot, &q);
        }
        let err = (tool_point(&robot, &q) - target).norm();

        // The limit is genuinely active: the natural yaw is ~1.0 rad, so joint 0 is driven hard against
        // its narrowed [-0.1, 0.1] window and settles pinned just under it (a velocity-limited QP
        // approaches the bound asymptotically — pressed against, never over).
        assert!(q[0] > 0.095 && q[0] <= 0.1, "constrained joint should be pinned near its 0.1 limit, got {}", q[0]);
        // …and because this 6-DoF arm is redundant for a 3-DoF position task, it still reaches the
        // target using the other joints while joint 0 stays capped. That's the win: hard limits held
        // AND the task solved. (err0 is the initial error, kept for context.)
        let _ = err0;
        assert!(err < 1e-2, "solver should still reach the position target via redundancy: err = {err}");
    }

    #[test]
    fn six_dof_pose_task_and_posture_stack_stay_feasible() {
        // A full 6-DoF end-effector pose task plus a soft posture regularizer, with all joints kept
        // in range throughout. Verifies the pose Jacobian path and the multi-task stack.
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let q_ref = [0.2, -0.4, 0.5, 0.3, -0.2, 0.4];
        let target = robot.fk(&q_ref);
        let solver = PlacoSolver {
            tasks: vec![
                PlacoTask::Pose { target, gain: 3.0, weight: 1.0 },
                PlacoTask::Posture { rest: vec![0.0; 6], weight: 1e-3 },
            ],
            ..PlacoSolver::default()
        };
        let mut q = vec![0.0; 6];
        for _ in 0..600 {
            q = solver.step(&robot, &q);
            assert_within_limits(&robot, &q);
        }
        // Pose error (position + orientation) driven small.
        let cur = robot.fk(&q);
        let dp = (cur.translation.vector - target.translation.vector).norm();
        let dr = (cur.rotation * target.rotation.inverse()).scaled_axis().norm();
        assert!(dp < 5e-3 && dr < 5e-3, "pose not reached: dp = {dp}, dr = {dr}");
    }
}
