//! Composable costs — the PyRoki-shaped design: each cost contributes a residual block and
//! its Jacobian, decoupled from the solver. Stack any set of them and [`crate::solve`] does
//! the rest.

use crate::{pose_error, Iso, Robot};
use nalgebra::{DMatrix, DVector, Point3, Vector3};

/// A cost contributes `dim()` residual rows and their `dim()×dof` Jacobian.
pub trait Cost {
    fn dim(&self, robot: &Robot) -> usize;
    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64>;
    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64>;
}

/// Match the end-effector to a target pose (position + orientation), independently weighted.
#[derive(Clone, Copy, Debug)]
pub struct PoseCost {
    pub target: Iso,
    pub pos_weight: f64,
    pub rot_weight: f64,
}

impl PoseCost {
    pub fn new(target: Iso, pos_weight: f64, rot_weight: f64) -> Self {
        Self { target, pos_weight, rot_weight }
    }

    fn w(&self, row: usize) -> f64 {
        if row < 3 {
            self.pos_weight
        } else {
            self.rot_weight
        }
    }
}

impl Cost for PoseCost {
    fn dim(&self, _robot: &Robot) -> usize {
        6
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        let e = pose_error(&robot.fk(q), &self.target);
        DVector::from_fn(6, |r, _| e[r] * self.w(r))
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let mut j = robot.jacobian(q); // 6×n, and d(pose_error)/dq ≈ geometric Jacobian
        let n = j.ncols();
        for r in 0..6 {
            let w = self.w(r);
            for c in 0..n {
                j[(r, c)] *= w;
            }
        }
        j
    }
}

/// Match a point attached at frame `frame` (offset in that frame) to a world target position.
/// The building block of position-based motion retargeting.
#[derive(Clone, Copy, Debug)]
pub struct PointCost {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub target: Vector3<f64>,
    pub weight: f64,
}

impl PointCost {
    pub fn new(frame: usize, offset: Vector3<f64>, target: Vector3<f64>, weight: f64) -> Self {
        Self { frame, offset, target, weight }
    }

    fn world_point(&self, robot: &Robot, q: &[f64]) -> Vector3<f64> {
        (robot.frame_pose(q, self.frame) * Point3::from(self.offset)).coords
    }
}

impl Cost for PointCost {
    fn dim(&self, _robot: &Robot) -> usize {
        3
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        let d = self.weight * (self.world_point(robot, q) - self.target);
        DVector::from_row_slice(&[d.x, d.y, d.z])
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let wp = self.world_point(robot, q);
        self.weight * robot.point_jacobian(q, self.frame, &wp)
    }
}

/// Match the *vector* between two attached points to a target vector — translation-invariant,
/// the robust workhorse for cross-morphology retargeting (human hand/body → robot).
#[derive(Clone, Copy, Debug)]
pub struct VectorCost {
    pub frame_a: usize,
    pub offset_a: Vector3<f64>,
    pub frame_b: usize,
    pub offset_b: Vector3<f64>,
    pub target: Vector3<f64>,
    pub weight: f64,
}

impl VectorCost {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_a: usize,
        offset_a: Vector3<f64>,
        frame_b: usize,
        offset_b: Vector3<f64>,
        target: Vector3<f64>,
        weight: f64,
    ) -> Self {
        Self { frame_a, offset_a, frame_b, offset_b, target, weight }
    }

    fn points(&self, robot: &Robot, q: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
        let a = (robot.frame_pose(q, self.frame_a) * Point3::from(self.offset_a)).coords;
        let b = (robot.frame_pose(q, self.frame_b) * Point3::from(self.offset_b)).coords;
        (a, b)
    }
}

impl Cost for VectorCost {
    fn dim(&self, _robot: &Robot) -> usize {
        3
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        let (a, b) = self.points(robot, q);
        let d = self.weight * ((b - a) - self.target);
        DVector::from_row_slice(&[d.x, d.y, d.z])
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let (a, b) = self.points(robot, q);
        let ja = robot.point_jacobian(q, self.frame_a, &a);
        let jb = robot.point_jacobian(q, self.frame_b, &b);
        self.weight * (jb - ja)
    }
}

/// Soft joint-limit cost: a one-sided hinge penalty outside each joint's `(lower, upper)`.
/// Joints without declared limits contribute zero.
#[derive(Clone, Copy, Debug)]
pub struct JointLimitCost {
    pub weight: f64,
}

impl JointLimitCost {
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl Cost for JointLimitCost {
    fn dim(&self, robot: &Robot) -> usize {
        robot.dof()
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        DVector::from_fn(robot.dof(), |i, _| {
            let v = match robot.joints[i].limits {
                Some((lo, _)) if q[i] < lo => q[i] - lo,
                Some((_, hi)) if q[i] > hi => q[i] - hi,
                _ => 0.0,
            };
            self.weight * v
        })
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let n = robot.dof();
        let mut j = DMatrix::zeros(n, n);
        for i in 0..n {
            let active = match robot.joints[i].limits {
                Some((lo, _)) if q[i] < lo => true,
                Some((_, hi)) if q[i] > hi => true,
                _ => false,
            };
            if active {
                j[(i, i)] = self.weight;
            }
        }
        j
    }
}

/// Regularize toward a rest posture — keeps redundant DoF well-conditioned and near-natural.
#[derive(Clone, Debug)]
pub struct PostureCost {
    pub rest: Vec<f64>,
    pub weight: f64,
}

impl PostureCost {
    pub fn new(rest: Vec<f64>, weight: f64) -> Self {
        Self { rest, weight }
    }
}

impl Cost for PostureCost {
    fn dim(&self, _robot: &Robot) -> usize {
        self.rest.len()
    }

    fn residual(&self, _robot: &Robot, q: &[f64]) -> DVector<f64> {
        DVector::from_fn(self.rest.len(), |i, _| self.weight * (q[i] - self.rest[i]))
    }

    fn jacobian(&self, robot: &Robot, _q: &[f64]) -> DMatrix<f64> {
        let n = robot.dof();
        let mut j = DMatrix::zeros(self.rest.len(), n);
        for i in 0..self.rest.len().min(n) {
            j[(i, i)] = self.weight;
        }
        j
    }
}
