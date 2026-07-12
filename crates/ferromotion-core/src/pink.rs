//! Pink/mink-parity differential inverse kinematics — a per-step *velocity* QP over a stack of
//! weighted tasks, here on the shared ferromotion kinematics with `clarabel` as the pure-Rust backend.
//!
//! This generalizes [`crate::solve_diffik`] (a single 3-DoF position task) to the full Pink/mink
//! task set: 6-DoF **frame-pose** tasks (position *and* orientation), a **posture** task, and
//! joint-limit avoidance folded into the per-step box bounds. Each step solves
//!
//! ```text
//!   minimize  Σ_task wₜ‖Jₜ q̇ − gainₜ·eₜ‖²  +  λ‖q̇‖²
//!   over q̇,   subject to   lb ≤ q̇ ≤ ub
//! ```
//!
//! where `eₜ` is the task-space error (for a pose task, the twist `[Δpos; SO3-log(R·R_tgt⁻¹)]`),
//! `Jₜ` its Jacobian, and the box bounds intersect velocity limits with joint-position limits
//! mapped to per-step velocity. It then integrates `q += q̇·dt` and repeats. WASM-clean.

use crate::{pose_error, Iso, JointKind, Robot};
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::{DMatrix, DVector, Translation3, UnitQuaternion};

/// A single task's contribution to one velocity-QP step: a (pre-weighted) least-squares block
/// `‖J q̇ − v‖²` plus a scalar task-space error used to decide convergence.
pub trait PinkTask {
    /// Weighted block for this step: returns `(J, v)` such that the step minimizes `‖J q̇ − v‖²`.
    /// Weights are already folded in (rows scaled by `√weight`), so the solver just sums `JᵀJ`
    /// and `Jᵀv`.
    fn block(&self, robot: &Robot, q: &[f64]) -> (DMatrix<f64>, DVector<f64>);
    /// Task-space error magnitude at `q` (0 for purely-secondary tasks like posture). The solver
    /// stops when the largest task error falls below the tolerance.
    fn error(&self, robot: &Robot, q: &[f64]) -> f64;
}

/// Drive the pose of `frame` toward `target`. `frame` indexes the frame after the first `frame`
/// joints (`0..=dof`); `frame == dof` is the tool/end-effector frame (includes the tool offset).
/// Set `rot_gain == 0` for a position-only task, or `pos_gain == 0` for orientation-only.
#[derive(Clone, Copy, Debug)]
pub struct FramePoseTask {
    pub frame: usize,
    pub target: Iso,
    pub pos_gain: f64,
    pub rot_gain: f64,
    pub weight: f64,
}

impl FramePoseTask {
    pub fn new(frame: usize, target: Iso, pos_gain: f64, rot_gain: f64, weight: f64) -> Self {
        Self { frame, target, pos_gain, rot_gain, weight }
    }
}

impl PinkTask for FramePoseTask {
    fn block(&self, robot: &Robot, q: &[f64]) -> (DMatrix<f64>, DVector<f64>) {
        let n = robot.dof();
        let (pose, jac) = frame_jacobian(robot, q, self.frame);
        // Twist error e = [pos_cur − pos_tgt; log(R_cur·R_tgt⁻¹)]; the reducing velocity is −gain·e.
        let e = pose_error(&pose, &self.target);
        let sw = self.weight.max(0.0).sqrt();

        // Select controlled rows: position rows (0..3) if pos_gain>0, orientation rows (3..6) if rot_gain>0.
        let mut rows: Vec<(usize, f64)> = Vec::with_capacity(6);
        if self.pos_gain > 0.0 {
            for r in 0..3 {
                rows.push((r, -self.pos_gain * e[r]));
            }
        }
        if self.rot_gain > 0.0 {
            for r in 3..6 {
                rows.push((r, -self.rot_gain * e[r]));
            }
        }

        let k = rows.len();
        let mut jb = DMatrix::zeros(k, n);
        let mut vb = DVector::zeros(k);
        for (bi, &(r, desired)) in rows.iter().enumerate() {
            for c in 0..n {
                jb[(bi, c)] = sw * jac[(r, c)];
            }
            vb[bi] = sw * desired;
        }
        (jb, vb)
    }

    fn error(&self, robot: &Robot, q: &[f64]) -> f64 {
        let pose = frame_pose_of(robot, q, self.frame);
        let e = pose_error(&pose, &self.target);
        let mut s = 0.0;
        if self.pos_gain > 0.0 {
            s += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }
        if self.rot_gain > 0.0 {
            s += e[3] * e[3] + e[4] * e[4] + e[5] * e[5];
        }
        s.sqrt()
    }
}

/// Regularize toward a rest posture in the nullspace of the primary tasks — resolves redundant
/// DoF and keeps the solve well-conditioned. Secondary: contributes no convergence error.
#[derive(Clone, Debug)]
pub struct PostureTask {
    pub rest: Vec<f64>,
    pub gain: f64,
    pub weight: f64,
}

impl PostureTask {
    pub fn new(rest: Vec<f64>, gain: f64, weight: f64) -> Self {
        Self { rest, gain, weight }
    }
}

impl PinkTask for PostureTask {
    fn block(&self, robot: &Robot, q: &[f64]) -> (DMatrix<f64>, DVector<f64>) {
        // ½w‖q̇ − gain·(rest − q)‖²: J = √w·I, v = √w·gain·(rest − q).
        let n = robot.dof();
        let sw = self.weight.max(0.0).sqrt();
        let jb = DMatrix::<f64>::identity(n, n) * sw;
        let vb = DVector::from_fn(n, |i, _| {
            let rest_i = self.rest.get(i).copied().unwrap_or(0.0);
            sw * self.gain * (rest_i - q[i])
        });
        (jb, vb)
    }

    fn error(&self, _robot: &Robot, _q: &[f64]) -> f64 {
        0.0 // secondary task — never gates convergence
    }
}

/// A weighted stack of tasks solved together each step.
#[derive(Default)]
pub struct TaskStack {
    pub tasks: Vec<Box<dyn PinkTask>>,
}

impl TaskStack {
    pub fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    /// Builder-style push.
    pub fn with(mut self, task: Box<dyn PinkTask>) -> Self {
        self.tasks.push(task);
        self
    }

    pub fn push(&mut self, task: Box<dyn PinkTask>) {
        self.tasks.push(task);
    }

    /// Largest task error over the stack (secondary tasks report 0).
    fn error(&self, robot: &Robot, q: &[f64]) -> f64 {
        self.tasks.iter().map(|t| t.error(robot, q)).fold(0.0, f64::max)
    }
}

/// Options for [`solve_pink`].
#[derive(Clone, Debug)]
pub struct PinkOptions {
    pub dt: f64,
    pub vmax: f64,
    pub damping: f64,
    pub max_iters: usize,
    pub tol: f64,
    /// Also respect joint-position limits (mapped to per-step velocity bounds).
    pub use_limits: bool,
}

impl Default for PinkOptions {
    fn default() -> Self {
        Self { dt: 0.05, vmax: 3.0, damping: 1e-3, max_iters: 300, tol: 1e-6, use_limits: true }
    }
}

/// Result of a [`solve_pink`] run.
#[derive(Clone, Debug)]
pub struct PinkResult {
    pub q: Vec<f64>,
    pub error: f64,
    pub iters: usize,
    pub converged: bool,
}

/// Ergonomic wrapper: hold options once, reuse across solves.
#[derive(Clone, Debug, Default)]
pub struct PinkSolver {
    pub opts: PinkOptions,
}

impl PinkSolver {
    pub fn new(opts: PinkOptions) -> Self {
        Self { opts }
    }

    pub fn solve(&self, robot: &Robot, stack: &TaskStack, q0: &[f64]) -> PinkResult {
        solve_pink(robot, stack, q0, &self.opts)
    }
}

/// Build the upper-triangular CSC of a small dense symmetric matrix (clarabel wants P upper-tri).
fn csc_upper(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.ncols();
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..n {
        for i in 0..=j {
            rowval.push(i);
            nzval.push(p[(i, j)]);
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// Box-constraint matrix `[I; -I]` (2n×n) in CSC for `lb ≤ x ≤ ub`.
fn csc_box(n: usize) -> CscMatrix<f64> {
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::with_capacity(2 * n);
    let mut nzval = Vec::with_capacity(2 * n);
    colptr.push(0);
    for j in 0..n {
        rowval.push(j);
        nzval.push(1.0);
        rowval.push(n + j);
        nzval.push(-1.0);
        colptr.push(rowval.len());
    }
    CscMatrix::new(2 * n, n, colptr, rowval, nzval)
}

/// World pose of `frame` (`frame == dof` ⇒ tool frame, including the tool offset).
fn frame_pose_of(robot: &Robot, q: &[f64], frame: usize) -> Iso {
    if frame >= robot.dof() {
        robot.fk(q)
    } else {
        robot.frame_pose(q, frame)
    }
}

/// `(pose, 6×n geometric Jacobian)` for `frame`. For the tool frame (`frame >= dof`) this is
/// exactly [`Robot::fk`] and [`Robot::jacobian`]; interior frames extend the same construction,
/// counting only the joints that move the frame (the rest are zero columns).
fn frame_jacobian(robot: &Robot, q: &[f64], frame: usize) -> (Iso, DMatrix<f64>) {
    let n = robot.dof();
    if frame >= n {
        return (robot.fk(q), robot.jacobian(q));
    }
    let pose = robot.frame_pose(q, frame);
    let p_ref = pose.translation.vector;
    let mut jac = DMatrix::zeros(6, n);
    let mut t = Iso::identity();
    for (i, (j, &qi)) in robot.joints.iter().zip(q).enumerate() {
        if i >= frame {
            break;
        }
        let pre = t * j.origin; // this joint's frame, before applying qi
        let z = pre.rotation * j.axis.into_inner(); // joint axis in world
        let p = pre.translation.vector; // joint origin in world
        match j.kind {
            JointKind::Revolute => {
                let lin = z.cross(&(p_ref - p));
                jac.fixed_view_mut::<3, 1>(0, i).copy_from(&lin);
                jac.fixed_view_mut::<3, 1>(3, i).copy_from(&z);
            }
            JointKind::Prismatic => {
                jac.fixed_view_mut::<3, 1>(0, i).copy_from(&z);
            }
        }
        // Advance: pre · motion(qi) == frame_pose(q, i+1) (reconstructed from public fields).
        let motion = match j.kind {
            JointKind::Revolute => {
                Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&j.axis, qi))
            }
            JointKind::Prismatic => {
                Iso::from_parts(Translation3::from(j.axis.into_inner() * qi), UnitQuaternion::identity())
            }
        };
        t = pre * motion;
    }
    (pose, jac)
}

/// Differential IK à la Pink/mink: iterate velocity-QP steps over the task stack until the
/// primary tasks are met (or `max_iters`), integrating `q += q̇·dt` each step.
pub fn solve_pink(robot: &Robot, stack: &TaskStack, q0: &[f64], opts: &PinkOptions) -> PinkResult {
    let n = robot.dof();
    let mut q = q0.to_vec();
    let a_csc = csc_box(n);
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();

    let mut iters = 0;
    for it in 0..opts.max_iters {
        iters = it + 1;
        if stack.error(robot, &q) < opts.tol {
            break;
        }

        // Assemble P = Σ wₜ·JₜᵀJₜ + λI and linear term q_lin = −Σ wₜ·Jₜᵀvₜ (weights folded into blocks).
        let mut p = DMatrix::<f64>::identity(n, n) * opts.damping;
        let mut g = DVector::<f64>::zeros(n);
        for task in &stack.tasks {
            let (jb, vb) = task.block(robot, &q);
            if jb.nrows() == 0 {
                continue;
            }
            let jt = jb.transpose();
            p += &jt * &jb;
            g += &jt * &vb;
        }
        let q_lin: Vec<f64> = (0..n).map(|i| -g[i]).collect();

        // Box bounds: velocity limits intersected with position-limit-derived per-step bounds.
        let (mut ub, mut lb) = (vec![opts.vmax; n], vec![-opts.vmax; n]);
        if opts.use_limits {
            for (i, joint) in robot.joints.iter().enumerate() {
                if let Some((lo, hi)) = joint.limits {
                    ub[i] = ub[i].min((hi - q[i]) / opts.dt);
                    lb[i] = lb[i].max((lo - q[i]) / opts.dt);
                }
            }
        }
        // b = [ub; -lb] for [I; -I]·q̇ ≤ b.
        let mut b = ub.clone();
        b.extend(lb.iter().map(|v| -v));

        let p_csc = csc_upper(&p);
        let cones = [SupportedConeT::NonnegativeConeT(2 * n)];
        let mut solver = DefaultSolver::new(&p_csc, &q_lin, &a_csc, &b, &cones, settings.clone()).unwrap();
        solver.solve();
        let qd = &solver.solution.x;

        let mut moved = 0.0;
        for i in 0..n {
            q[i] += qd[i] * opts.dt;
            moved += (qd[i] * opts.dt).abs();
        }
        if moved < 1e-12 {
            break; // no feasible progress
        }
    }

    let err = stack.error(robot, &q);
    PinkResult { q, error: err, iters, converged: err < 1e-4 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_str, solve_diffik, DiffIkOptions, FrameTaskDef};
    use nalgebra::Vector3;

    const ARM6: &str = r#"<robot name="arm6"><link name="world"/><link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/><joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint><joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

    fn norm(q: &[f64]) -> f64 {
        q.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn reaches_full_6dof_pose() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let dof = robot.dof();
        assert_eq!(dof, 6);
        let q_ref = [0.3, -0.5, 0.6, 0.2, 0.4, -0.3];
        let target = robot.fk(&q_ref); // a full 6-DoF pose that is exactly reachable

        let stack = TaskStack::new().with(Box::new(FramePoseTask::new(dof, target, 2.0, 2.0, 1.0)));
        let res = solve_pink(&robot, &stack, &[0.0; 6], &PinkOptions::default());

        // Honest guarantee: the full twist error (position + orientation) is driven small.
        let e = pose_error(&robot.fk(&res.q), &target);
        assert!(e.norm() < 1e-2, "6-DoF pose not reached: twist err {} (iters {})", e.norm(), res.iters);
    }

    #[test]
    fn posture_biases_redundant_solution() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let dof = robot.dof();
        let q_ref = [0.2, -0.4, 0.6, 0.2, 0.3, -0.2];
        let target = robot.fk(&q_ref);
        let seed = [0.4; 6];

        let pose_task = || Box::new(FramePoseTask::new(dof, target, 2.0, 2.0, 1.0)) as Box<dyn PinkTask>;

        // Without the posture bias.
        let base = solve_pink(&robot, &TaskStack::new().with(pose_task()), &seed, &PinkOptions::default());
        // With a clearly-secondary posture task pulling toward the zero rest posture.
        let stack = TaskStack::new()
            .with(pose_task())
            .with(Box::new(PostureTask::new(vec![0.0; dof], 1.0, 2e-3)));
        let res = solve_pink(&robot, &stack, &seed, &PinkOptions::default());

        // Primary pose still met (posture is a soft, secondary weight — allow a cm/rad-level compromise).
        let e = pose_error(&robot.fk(&res.q), &target);
        assert!(e.norm() < 1.5e-2, "posture broke the pose task: twist err {}", e.norm());
        // ...and the redundant DoF are pulled toward the rest posture (‖q‖ smaller than without it).
        assert!(norm(&res.q) < norm(&base.q), "posture did not bias the solution: {} vs {}", norm(&res.q), norm(&base.q));
    }

    #[test]
    fn position_only_task_equals_diffik() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let dof = robot.dof();
        let q_ref = [0.2, -0.3, 0.5, 0.1, 0.2, -0.2];
        let target_pos = robot.fk(&q_ref).translation.vector;

        // Pink position-only task on the tool frame (rot_gain = 0 drops the orientation rows).
        let target = Iso::from_parts(Translation3::from(target_pos), UnitQuaternion::identity());
        let stack = TaskStack::new().with(Box::new(FramePoseTask::new(dof, target, 2.0, 0.0, 1.0)));
        let pink = solve_pink(&robot, &stack, &[0.0; 6], &PinkOptions::default());

        // The diffik primitive: same point (tool tip = frame `dof` + the 0.05 tool offset), same gain/opts.
        let tasks = vec![FrameTaskDef::new(dof, Vector3::new(0.0, 0.0, 0.05), target_pos, 2.0, 1.0)];
        let diff = solve_diffik(&robot, &tasks, &[0.0; 6], &DiffIkOptions::default());

        let tip_pink = robot.fk(&pink.q).translation.vector;
        let tip_diff = robot.fk(&diff.q).translation.vector;

        // Both reach the target...
        assert!((tip_pink - target_pos).norm() < 1e-3, "pink position-only missed: {}", (tip_pink - target_pos).norm());
        assert!((tip_diff - target_pos).norm() < 1e-3, "diffik missed: {}", (tip_diff - target_pos).norm());
        // ...and, since the assembled QP is identical each step, the trajectories coincide.
        assert!((tip_pink - tip_diff).norm() < 1e-6, "pink != diffik: {}", (tip_pink - tip_diff).norm());
    }
}
