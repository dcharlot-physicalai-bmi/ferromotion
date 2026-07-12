//! Motion retargeting — map a stream of observed keypoints (human demo, teleop, mocap) onto a
//! robot's joint motion. Each timestep is a nonlinear-least-squares solve over point/vector costs,
//! warm-started from the previous frame with a smoothness term for temporal coherence. This is
//! the "show it, don't program it" primitive, and the piece with no Rust-native equivalent today.

use crate::{solve, Cost, JointLimitCost, PointCost, PostureCost, Robot, SolveOptions, SolveResult, VectorCost};
use nalgebra::Vector3;

/// One correspondence: a point attached at `frame` (offset in that frame) that should track a
/// per-timestep target keypoint.
#[derive(Clone, Copy, Debug)]
pub struct FrameTask {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub weight: f64,
}

impl FrameTask {
    pub fn new(frame: usize, offset: Vector3<f64>, weight: f64) -> Self {
        Self { frame, offset, weight }
    }
}

/// Position-based retargeter: a fixed set of tasks, plus temporal smoothness and joint-limit terms.
#[derive(Clone, Debug)]
pub struct Retargeter {
    pub tasks: Vec<FrameTask>,
    /// Weight pulling each solve toward the previous configuration (velocity damping).
    pub smoothness: f64,
    /// Weight on the soft joint-limit hinge (0 disables).
    pub limit_weight: f64,
    pub opts: SolveOptions,
}

impl Retargeter {
    pub fn new(tasks: Vec<FrameTask>) -> Self {
        Self { tasks, smoothness: 0.01, limit_weight: 0.0, opts: SolveOptions::default() }
    }

    /// Solve one timestep. `targets` are aligned with `self.tasks`; `prev` is the warm-start.
    pub fn solve_frame(&self, robot: &Robot, targets: &[Vector3<f64>], prev: &[f64]) -> SolveResult {
        let mut costs: Vec<Box<dyn Cost>> = Vec::with_capacity(self.tasks.len() + 2);
        for (t, tgt) in self.tasks.iter().zip(targets) {
            costs.push(Box::new(PointCost::new(t.frame, t.offset, *tgt, t.weight)));
        }
        if self.smoothness > 0.0 {
            costs.push(Box::new(PostureCost::new(prev.to_vec(), self.smoothness)));
        }
        if self.limit_weight > 0.0 {
            costs.push(Box::new(JointLimitCost::new(self.limit_weight)));
        }
        solve(robot, &costs, prev, &self.opts)
    }

    /// Retarget a whole sequence, warm-starting each frame from the previous solution.
    pub fn solve_sequence(
        &self,
        robot: &Robot,
        targets_seq: &[Vec<Vector3<f64>>],
        q0: &[f64],
    ) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(targets_seq.len());
        let mut prev = q0.to_vec();
        for targets in targets_seq {
            let r = self.solve_frame(robot, targets, &prev);
            prev = r.q.clone();
            out.push(r.q);
        }
        out
    }
}

/// A vector correspondence: the robot vector from point A (`frame_a`+`offset_a`) to point B
/// (`frame_b`+`offset_b`) should track an observed target vector.
#[derive(Clone, Copy, Debug)]
pub struct VectorTask {
    pub frame_a: usize,
    pub offset_a: Vector3<f64>,
    pub frame_b: usize,
    pub offset_b: Vector3<f64>,
    pub weight: f64,
}

impl VectorTask {
    pub fn new(frame_a: usize, offset_a: Vector3<f64>, frame_b: usize, offset_b: Vector3<f64>, weight: f64) -> Self {
        Self { frame_a, offset_a, frame_b, offset_b, weight }
    }
}

/// Vector-based retargeter — the dex-retargeting "vector optimizer": match a set of keypoint
/// vectors (translation-invariant), which transfers a demonstration across differing morphologies
/// (human hand → robot hand). Per-frame warm-started solve with smoothness + joint limits.
#[derive(Clone, Debug)]
pub struct VectorRetargeter {
    pub tasks: Vec<VectorTask>,
    pub smoothness: f64,
    pub limit_weight: f64,
    pub opts: SolveOptions,
}

impl VectorRetargeter {
    pub fn new(tasks: Vec<VectorTask>) -> Self {
        Self { tasks, smoothness: 0.01, limit_weight: 0.0, opts: SolveOptions::default() }
    }

    /// Solve one frame: `targets` are the observed vectors, aligned with `self.tasks`.
    pub fn solve_frame(&self, robot: &Robot, targets: &[Vector3<f64>], prev: &[f64]) -> SolveResult {
        let mut costs: Vec<Box<dyn Cost>> = Vec::with_capacity(self.tasks.len() + 2);
        for (t, tgt) in self.tasks.iter().zip(targets) {
            costs.push(Box::new(VectorCost::new(t.frame_a, t.offset_a, t.frame_b, t.offset_b, *tgt, t.weight)));
        }
        if self.smoothness > 0.0 {
            costs.push(Box::new(PostureCost::new(prev.to_vec(), self.smoothness)));
        }
        if self.limit_weight > 0.0 {
            costs.push(Box::new(JointLimitCost::new(self.limit_weight)));
        }
        solve(robot, &costs, prev, &self.opts)
    }

    /// Retarget a whole sequence, warm-starting each frame from the previous solution.
    pub fn solve_sequence(&self, robot: &Robot, targets_seq: &[Vec<Vector3<f64>>], q0: &[f64]) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(targets_seq.len());
        let mut prev = q0.to_vec();
        for targets in targets_seq {
            let r = self.solve_frame(robot, targets, &prev);
            prev = r.q.clone();
            out.push(r.q);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_str;
    use nalgebra::Point3;

    // Same 6-DoF arm as the URDF tests, inlined.
    const ARM: &str = r#"<robot name="a"><link name="world"/><link name="base"/>
      <link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
      <joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

    #[test]
    fn retargets_a_trajectory_from_keypoints() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        // Track three points along the arm: mid, wrist, tool-tip.
        let tasks = vec![
            FrameTask::new(3, Vector3::zeros(), 1.0),
            FrameTask::new(5, Vector3::zeros(), 1.0),
            FrameTask::new(6, Vector3::new(0.0, 0.0, 0.05), 1.0), // tool tip
        ];
        let point = |q: &[f64], t: &FrameTask| (robot.frame_pose(q, t.frame) * Point3::from(t.offset)).coords;

        // Ground-truth smooth joint motion → synthesize the observed keypoint stream.
        let steps = 40;
        let mut truth = Vec::new();
        let mut targets_seq = Vec::new();
        for k in 0..steps {
            let s = k as f64 / steps as f64;
            let q: Vec<f64> = (0..6)
                .map(|i| 0.6 * (2.0 * std::f64::consts::PI * s + i as f64).sin())
                .collect();
            targets_seq.push(tasks.iter().map(|t| point(&q, t)).collect::<Vec<_>>());
            truth.push(q);
        }

        let retargeter = Retargeter::new(tasks.clone());
        let solved = retargeter.solve_sequence(&robot, &targets_seq, &truth[0]);

        // The meaningful check: reconstructed keypoints reproduce the observed stream every frame.
        let mut worst: f64 = 0.0;
        for (k, q) in solved.iter().enumerate() {
            for (t, tgt) in tasks.iter().zip(&targets_seq[k]) {
                worst = worst.max((point(q, t) - tgt).norm());
            }
        }
        // Sub-2mm keypoint reproduction across the whole motion; the small residual is the
        // smoothness term trading a little fidelity for temporal coherence, by design.
        assert!(worst < 2e-3, "worst keypoint reproduction error across trajectory: {worst}");
    }

    #[test]
    fn vector_retargeting_reproduces_keypoint_vectors() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let tasks = vec![
            VectorTask::new(0, Vector3::zeros(), 6, Vector3::new(0.0, 0.0, 0.05), 1.0),
            VectorTask::new(3, Vector3::zeros(), 6, Vector3::new(0.0, 0.0, 0.05), 1.0),
        ];
        let vecof = |q: &[f64], t: &VectorTask| {
            let a = (robot.frame_pose(q, t.frame_a) * Point3::from(t.offset_a)).coords;
            let b = (robot.frame_pose(q, t.frame_b) * Point3::from(t.offset_b)).coords;
            b - a
        };
        let q_true = [0.3, -0.4, 0.6, 0.2, 0.5, -0.2];
        let targets: Vec<Vector3<f64>> = tasks.iter().map(|t| vecof(&q_true, t)).collect();
        let mut rt = VectorRetargeter::new(tasks.clone());
        rt.smoothness = 0.0; // pure vector matching for this reproduction check
        let res = rt.solve_frame(&robot, &targets, &[0.0; 6]);
        let worst = tasks
            .iter()
            .zip(&targets)
            .fold(0.0_f64, |m, (t, tgt)| m.max((vecof(&res.q, t) - tgt).norm()));
        assert!(worst < 1e-3, "vector reproduction error {worst}");
    }

    #[test]
    fn vector_cost_matches_a_direction() {
        use crate::{solve, Cost, VectorCost};
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let q_true = [0.3, -0.4, 0.6, 0.2, 0.5, -0.2];
        // Target = the base→tool vector at q_true.
        let a = (robot.frame_pose(&q_true, 0) * Point3::from(Vector3::zeros())).coords;
        let b = (robot.frame_pose(&q_true, 6) * Point3::from(Vector3::new(0.0, 0.0, 0.05))).coords;
        let target = b - a;
        let costs: Vec<Box<dyn Cost>> = vec![Box::new(VectorCost::new(
            0, Vector3::zeros(), 6, Vector3::new(0.0, 0.0, 0.05), target, 1.0,
        ))];
        let res = solve(&robot, &costs, &[0.0; 6], &SolveOptions::default());
        assert!(res.error < 1e-4, "vector match residual {}", res.error);
    }
}
