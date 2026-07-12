//! Dexterous retargeting — a faithful Rust port of the three optimizers in the
//! `dex-retargeting` library, expressed on ferromotion's composable-cost spine. Each maps a stream of
//! observed human keypoints onto a robot's joint motion, one nonlinear-least-squares solve per
//! frame, warm-started from the previous solution with a smoothness term for temporal coherence.
//!
//! The three flavours, and when to reach for each:
//!
//! * [`PositionRetargeter`] — the *position optimizer*: pin a set of robot body points to human
//!   keypoint **positions**. Direct and precise when the human and robot share a frame and scale
//!   (a good default for teleop of an arm), but sensitive to morphology mismatch.
//! * [`VectorRetargeter`] — the *vector optimizer*: match a set of keypoint-**pair vectors**.
//!   Translation-invariant, so it transfers a demonstration across bodies of differing size and
//!   proportion (human hand → robot hand). The robust cross-morphology workhorse.
//! * [`DexPilotRetargeter`] — the DexPilot scheme: a curated set of fingertip-to-palm and
//!   fingertip-to-fingertip vectors whose weight **rises as the target vector shortens**, so the
//!   solve pulls digits into contact exactly when the human means them to touch.
//!
//! All three reuse the existing [`crate::Retargeter`] / [`crate::VectorRetargeter`] primitives (and
//! through them [`crate::PointCost`] / [`crate::VectorCost`] and [`crate::solve`]); this module adds
//! only the dex-retargeting *correspondence bookkeeping* — which robot points/vectors track which
//! human keypoint indices — and DexPilot's distance-based weighting.

use crate::{
    FrameTask, Retargeter as CoreRetargeter, Robot, SolveOptions, VectorRetargeter as CoreVectorRetargeter,
    VectorTask,
};
use nalgebra::Vector3;

/// A position correspondence: the robot point at `frame`+`offset` should track human keypoint
/// index `keypoint`, weighted by `weight`.
#[derive(Clone, Copy, Debug)]
pub struct PositionCorr {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub keypoint: usize,
    pub weight: f64,
}

impl PositionCorr {
    pub fn new(frame: usize, offset: Vector3<f64>, keypoint: usize, weight: f64) -> Self {
        Self { frame, offset, keypoint, weight }
    }
}

/// A vector correspondence: the robot vector from point A (`frame_a`+`offset_a`) to point B
/// (`frame_b`+`offset_b`) should track the human vector `keypoints[kp_b] − keypoints[kp_a]`.
#[derive(Clone, Copy, Debug)]
pub struct VectorCorr {
    pub frame_a: usize,
    pub offset_a: Vector3<f64>,
    pub frame_b: usize,
    pub offset_b: Vector3<f64>,
    pub kp_a: usize,
    pub kp_b: usize,
    pub weight: f64,
}

impl VectorCorr {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_a: usize,
        offset_a: Vector3<f64>,
        frame_b: usize,
        offset_b: Vector3<f64>,
        kp_a: usize,
        kp_b: usize,
        weight: f64,
    ) -> Self {
        Self { frame_a, offset_a, frame_b, offset_b, kp_a, kp_b, weight }
    }
}

// ---------------------------------------------------------------------------
// Position optimizer.
// ---------------------------------------------------------------------------

/// Position-based retargeter (dex-retargeting's `PositionOptimizer`): match a set of robot body
/// points to human keypoint positions, with temporal smoothness and optional joint limits.
#[derive(Clone, Debug)]
pub struct PositionRetargeter {
    pub corrs: Vec<PositionCorr>,
    /// Weight pulling each solve toward the previous configuration (velocity damping).
    pub smoothness: f64,
    /// Weight on the soft joint-limit hinge (0 disables).
    pub limit_weight: f64,
    pub opts: SolveOptions,
}

impl PositionRetargeter {
    pub fn new(corrs: Vec<PositionCorr>) -> Self {
        Self { corrs, smoothness: 0.01, limit_weight: 0.0, opts: SolveOptions::default() }
    }

    /// Solve one frame. `keypoints` is the full observed keypoint set; each correspondence reads
    /// the index it tracks. Returns the retargeted configuration.
    pub fn solve_frame(&self, robot: &Robot, keypoints: &[Vector3<f64>], prev: &[f64]) -> Vec<f64> {
        let tasks: Vec<FrameTask> =
            self.corrs.iter().map(|c| FrameTask::new(c.frame, c.offset, c.weight)).collect();
        let targets: Vec<Vector3<f64>> = self.corrs.iter().map(|c| keypoints[c.keypoint]).collect();
        let inner = CoreRetargeter {
            tasks,
            smoothness: self.smoothness,
            limit_weight: self.limit_weight,
            opts: self.opts,
        };
        inner.solve_frame(robot, &targets, prev).q
    }

    /// Retarget a whole keypoint stream, warm-starting each frame from the previous solution.
    pub fn solve_sequence(
        &self,
        robot: &Robot,
        keypoints_seq: &[Vec<Vector3<f64>>],
        q0: &[f64],
    ) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(keypoints_seq.len());
        let mut prev = q0.to_vec();
        for keypoints in keypoints_seq {
            let q = self.solve_frame(robot, keypoints, &prev);
            prev = q.clone();
            out.push(q);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Vector optimizer.
// ---------------------------------------------------------------------------

/// Vector-based retargeter (dex-retargeting's `VectorOptimizer`): match a set of keypoint-pair
/// vectors. Translation-invariant, so it transfers across differing morphologies.
#[derive(Clone, Debug)]
pub struct VectorRetargeter {
    pub corrs: Vec<VectorCorr>,
    pub smoothness: f64,
    pub limit_weight: f64,
    pub opts: SolveOptions,
}

impl VectorRetargeter {
    pub fn new(corrs: Vec<VectorCorr>) -> Self {
        Self { corrs, smoothness: 0.01, limit_weight: 0.0, opts: SolveOptions::default() }
    }

    /// The target vector each correspondence should reproduce, read from the keypoint set.
    fn targets(&self, keypoints: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        self.corrs.iter().map(|c| keypoints[c.kp_b] - keypoints[c.kp_a]).collect()
    }

    /// Solve one frame from the full observed keypoint set. Returns the retargeted configuration.
    pub fn solve_frame(&self, robot: &Robot, keypoints: &[Vector3<f64>], prev: &[f64]) -> Vec<f64> {
        let tasks: Vec<VectorTask> = self
            .corrs
            .iter()
            .map(|c| VectorTask::new(c.frame_a, c.offset_a, c.frame_b, c.offset_b, c.weight))
            .collect();
        let targets = self.targets(keypoints);
        let inner = CoreVectorRetargeter {
            tasks,
            smoothness: self.smoothness,
            limit_weight: self.limit_weight,
            opts: self.opts,
        };
        inner.solve_frame(robot, &targets, prev).q
    }

    /// Retarget a whole keypoint stream, warm-starting each frame from the previous solution.
    pub fn solve_sequence(
        &self,
        robot: &Robot,
        keypoints_seq: &[Vec<Vector3<f64>>],
        q0: &[f64],
    ) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(keypoints_seq.len());
        let mut prev = q0.to_vec();
        for keypoints in keypoints_seq {
            let q = self.solve_frame(robot, keypoints, &prev);
            prev = q.clone();
            out.push(q);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// DexPilot optimizer.
// ---------------------------------------------------------------------------

/// DexPilot retargeter: the same vector-matching machinery, but each correspondence's weight is a
/// function of the *target* vector length. As two keypoints that should touch draw close, the
/// weight rises toward [`w_close`](Self::w_close), so the solve prioritizes closing that gap
/// (contact) over the looser, far-apart vectors. This is what lets DexPilot reproduce precise
/// pinches and grasps rather than just an average hand shape.
///
/// The weight profile is a clamped linear ramp in the target length `L`:
/// `L ≤ project_dist → w_close`, `L ≥ escape_dist → w_far`, linear in between (with
/// `w_close ≥ w_far`, i.e. monotonically non-increasing in `L`). Each correspondence's own
/// `weight` field scales this profile, so per-vector emphasis still applies.
#[derive(Clone, Debug)]
pub struct DexPilotRetargeter {
    pub corrs: Vec<VectorCorr>,
    /// At or below this target length, apply the full close-contact weight.
    pub project_dist: f64,
    /// At or above this target length, apply the baseline far weight.
    pub escape_dist: f64,
    /// Weight when the two keypoints are meant to touch (short target vector).
    pub w_close: f64,
    /// Baseline weight when the keypoints are far apart.
    pub w_far: f64,
    pub smoothness: f64,
    pub limit_weight: f64,
    pub opts: SolveOptions,
}

impl DexPilotRetargeter {
    /// Construct with dex-retargeting-flavored defaults (distances in metres): full weight under
    /// 3 cm, baseline over 10 cm, a 5× emphasis on closing contacts.
    pub fn new(corrs: Vec<VectorCorr>) -> Self {
        Self {
            corrs,
            project_dist: 0.03,
            escape_dist: 0.10,
            w_close: 5.0,
            w_far: 1.0,
            smoothness: 0.01,
            limit_weight: 0.0,
            opts: SolveOptions::default(),
        }
    }

    /// The distance-based weight for a target vector of length `len` (before the per-correspondence
    /// scale). Monotonically non-increasing in `len`.
    pub fn weight_for(&self, len: f64) -> f64 {
        if len <= self.project_dist {
            self.w_close
        } else if len >= self.escape_dist {
            self.w_far
        } else {
            let t = (len - self.project_dist) / (self.escape_dist - self.project_dist);
            self.w_close + t * (self.w_far - self.w_close)
        }
    }

    /// Build the per-frame vector tasks (weights recomputed from the current target lengths) and
    /// their targets.
    fn build(&self, keypoints: &[Vector3<f64>]) -> (Vec<VectorTask>, Vec<Vector3<f64>>) {
        let mut tasks = Vec::with_capacity(self.corrs.len());
        let mut targets = Vec::with_capacity(self.corrs.len());
        for c in &self.corrs {
            let tv = keypoints[c.kp_b] - keypoints[c.kp_a];
            let w = c.weight * self.weight_for(tv.norm());
            tasks.push(VectorTask::new(c.frame_a, c.offset_a, c.frame_b, c.offset_b, w));
            targets.push(tv);
        }
        (tasks, targets)
    }

    /// Solve one frame from the full observed keypoint set. Returns the retargeted configuration.
    pub fn solve_frame(&self, robot: &Robot, keypoints: &[Vector3<f64>], prev: &[f64]) -> Vec<f64> {
        let (tasks, targets) = self.build(keypoints);
        let inner = CoreVectorRetargeter {
            tasks,
            smoothness: self.smoothness,
            limit_weight: self.limit_weight,
            opts: self.opts,
        };
        inner.solve_frame(robot, &targets, prev).q
    }

    /// Retarget a whole keypoint stream, warm-starting each frame from the previous solution.
    pub fn solve_sequence(
        &self,
        robot: &Robot,
        keypoints_seq: &[Vec<Vector3<f64>>],
        q0: &[f64],
    ) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(keypoints_seq.len());
        let mut prev = q0.to_vec();
        for keypoints in keypoints_seq {
            let q = self.solve_frame(robot, keypoints, &prev);
            prev = q.clone();
            out.push(q);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_str;
    use nalgebra::Point3;

    // 6-DoF spatial arm (no inertials — kinematics only). base "world", tip "tool", dof 6.
    const ARM6: &str = r#"<robot name="arm6"><link name="world"/><link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/><joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint><joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint><joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

    // The keypoint set used across all tests: four points rigidly attached along the SAME robot.
    // Self-retargeting against these is the honest test — a config that reproduces the robot's own
    // keypoints/vectors must exist (the truth config), so any residual is solver/smoothness error.
    const KP_SPECS: [(usize, [f64; 3]); 4] = [
        (0, [0.0, 0.0, 0.0]),  // kp0: base frame
        (3, [0.0, 0.0, 0.0]),  // kp1: mid-arm
        (5, [0.0, 0.0, 0.0]),  // kp2: wrist
        (6, [0.0, 0.0, 0.05]), // kp3: tool tip
    ];

    fn keypoints_of(robot: &Robot, q: &[f64]) -> Vec<Vector3<f64>> {
        KP_SPECS
            .iter()
            .map(|(f, o)| (robot.frame_pose(q, *f) * Point3::from(Vector3::new(o[0], o[1], o[2]))).coords)
            .collect()
    }

    // A smooth ground-truth joint trajectory of the arm itself.
    fn truth_trajectory(steps: usize) -> Vec<Vec<f64>> {
        (0..steps)
            .map(|k| {
                let s = k as f64 / steps as f64;
                (0..6)
                    .map(|i| 0.6 * (2.0 * std::f64::consts::PI * s + i as f64).sin())
                    .collect::<Vec<f64>>()
            })
            .collect()
    }

    fn point_at(robot: &Robot, q: &[f64], frame: usize, offset: Vector3<f64>) -> Vector3<f64> {
        (robot.frame_pose(q, frame) * Point3::from(offset)).coords
    }

    #[test]
    fn position_retarget_reconstructs_keypoints_over_trajectory() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let corrs = vec![
            PositionCorr::new(3, Vector3::zeros(), 1, 1.0),
            PositionCorr::new(5, Vector3::zeros(), 2, 1.0),
            PositionCorr::new(6, Vector3::new(0.0, 0.0, 0.05), 3, 1.0),
        ];
        let truth = truth_trajectory(40);
        let kp_seq: Vec<Vec<Vector3<f64>>> = truth.iter().map(|q| keypoints_of(&robot, q)).collect();

        let rt = PositionRetargeter::new(corrs.clone());
        let solved = rt.solve_sequence(&robot, &kp_seq, &truth[0]);

        let mut worst = 0.0_f64;
        for (k, q) in solved.iter().enumerate() {
            for c in &corrs {
                let p = point_at(&robot, q, c.frame, c.offset);
                worst = worst.max((p - kp_seq[k][c.keypoint]).norm());
            }
        }
        // Sub-2mm keypoint reproduction across the whole motion; the tiny residual is the
        // smoothness term trading a little fidelity for temporal coherence, by design.
        assert!(worst < 2e-3, "worst position keypoint error across trajectory: {worst}");
    }

    #[test]
    fn vector_retarget_reproduces_keypoint_vectors() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let corrs = vec![
            VectorCorr::new(0, Vector3::zeros(), 6, Vector3::new(0.0, 0.0, 0.05), 0, 3, 1.0),
            VectorCorr::new(3, Vector3::zeros(), 5, Vector3::zeros(), 1, 2, 1.0),
        ];
        let mut rt = VectorRetargeter::new(corrs.clone());
        rt.smoothness = 0.0; // pure vector matching for this reproduction check

        let truth = truth_trajectory(30);
        let kp_seq: Vec<Vec<Vector3<f64>>> = truth.iter().map(|q| keypoints_of(&robot, q)).collect();
        let solved = rt.solve_sequence(&robot, &kp_seq, &truth[0]);

        let vec_of = |q: &[f64], c: &VectorCorr| {
            point_at(&robot, q, c.frame_b, c.offset_b) - point_at(&robot, q, c.frame_a, c.offset_a)
        };
        let mut worst = 0.0_f64;
        for (k, q) in solved.iter().enumerate() {
            for c in &corrs {
                let target = kp_seq[k][c.kp_b] - kp_seq[k][c.kp_a];
                worst = worst.max((vec_of(q, c) - target).norm());
            }
        }
        // Translation-invariant reproduction: the reconstructed vectors track the target vectors.
        assert!(worst < 2e-3, "worst vector reproduction error across trajectory: {worst}");
    }

    #[test]
    fn dexpilot_weight_rises_as_target_shortens() {
        let rt = DexPilotRetargeter::new(vec![]);
        let w_touch = rt.weight_for(0.01); // digits touching
        let w_mid = rt.weight_for(0.06); // mid-range
        let w_far = rt.weight_for(0.5); // wide open
        assert!(w_touch > w_mid && w_mid > w_far, "not monotone: {w_touch} {w_mid} {w_far}");
        assert!((w_touch - rt.w_close).abs() < 1e-12, "close plateau");
        assert!((w_far - rt.w_far).abs() < 1e-12, "far plateau");
    }

    #[test]
    fn dexpilot_reproduces_its_vectors() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        // Fingertip-to-"palm" (tip→base) and a fingertip-to-fingertip analogue (wrist→mid).
        let corrs = vec![
            VectorCorr::new(0, Vector3::zeros(), 6, Vector3::new(0.0, 0.0, 0.05), 0, 3, 1.0),
            VectorCorr::new(3, Vector3::zeros(), 5, Vector3::zeros(), 1, 2, 1.0),
        ];
        let mut rt = DexPilotRetargeter::new(corrs.clone());
        rt.smoothness = 0.0;

        let truth = truth_trajectory(30);
        let kp_seq: Vec<Vec<Vector3<f64>>> = truth.iter().map(|q| keypoints_of(&robot, q)).collect();
        let solved = rt.solve_sequence(&robot, &kp_seq, &truth[0]);

        let vec_of = |q: &[f64], c: &VectorCorr| {
            point_at(&robot, q, c.frame_b, c.offset_b) - point_at(&robot, q, c.frame_a, c.offset_a)
        };
        let mut worst = 0.0_f64;
        for (k, q) in solved.iter().enumerate() {
            for c in &corrs {
                let target = kp_seq[k][c.kp_b] - kp_seq[k][c.kp_a];
                worst = worst.max((vec_of(q, c) - target).norm());
            }
        }
        assert!(worst < 2e-3, "worst DexPilot vector error across trajectory: {worst}");
    }

    #[test]
    fn retargeting_is_deterministic() {
        let robot = from_urdf_str(ARM6, "world", "tool").unwrap();
        let corrs = vec![
            PositionCorr::new(5, Vector3::zeros(), 2, 1.0),
            PositionCorr::new(6, Vector3::new(0.0, 0.0, 0.05), 3, 1.0),
        ];
        let rt = PositionRetargeter::new(corrs);
        let q = vec![0.2, -0.3, 0.4, 0.1, -0.2, 0.3];
        let kps = keypoints_of(&robot, &q);
        let a = rt.solve_frame(&robot, &kps, &[0.0; 6]);
        let b = rt.solve_frame(&robot, &kps, &[0.0; 6]);
        assert_eq!(a, b, "identical inputs must produce bit-identical output");
    }
}
