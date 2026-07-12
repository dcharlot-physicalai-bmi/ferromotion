//! Collision-avoidance costs, using the sphere model (as cuRobo does): approximate robot links
//! and obstacles as spheres so the signed distance — and its gradient — are smooth, which is what
//! makes gradient-based trajectory optimization converge cleanly. Pure `nalgebra`, WASM-clean.
//! General convex/mesh distance via `parry` is a documented future extension.

use crate::{Cost, Robot};
use nalgebra::{DMatrix, DVector, Point3, Vector3};

/// Push a robot sphere (center at `frame`+`offset`, radius `radius`) at least `margin` clear of a
/// world sphere obstacle. One-sided: zero when already clear.
#[derive(Clone, Copy, Debug)]
pub struct SphereCollisionCost {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub radius: f64,
    pub obstacle_center: Vector3<f64>,
    pub obstacle_radius: f64,
    pub margin: f64,
    pub weight: f64,
}

impl SphereCollisionCost {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame: usize,
        offset: Vector3<f64>,
        radius: f64,
        obstacle_center: Vector3<f64>,
        obstacle_radius: f64,
        margin: f64,
        weight: f64,
    ) -> Self {
        Self { frame, offset, radius, obstacle_center, obstacle_radius, margin, weight }
    }

    fn center(&self, robot: &Robot, q: &[f64]) -> Vector3<f64> {
        (robot.frame_pose(q, self.frame) * Point3::from(self.offset)).coords
    }

    /// Surface-to-surface clearance (negative ⇒ interpenetrating).
    fn clearance(&self, robot: &Robot, q: &[f64]) -> f64 {
        (self.center(robot, q) - self.obstacle_center).norm() - self.radius - self.obstacle_radius
    }
}

impl Cost for SphereCollisionCost {
    fn dim(&self, _robot: &Robot) -> usize {
        1
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        let viol = (self.margin - self.clearance(robot, q)).max(0.0);
        DVector::from_row_slice(&[self.weight * viol])
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let n = robot.dof();
        let mut row = DMatrix::zeros(1, n);
        let c = self.center(robot, q);
        let diff = c - self.obstacle_center;
        let d = diff.norm();
        if self.margin - self.clearance(robot, q) > 0.0 && d > 1e-9 {
            // residual = w·(margin − clearance), clearance = d − rsum, d(d)/dq = n̂ᵀ·Jp.
            let nhat = diff / d;
            let jp = robot.point_jacobian(q, self.frame, &c);
            for col in 0..n {
                row[(0, col)] = -self.weight * nhat.dot(&jp.column(col));
            }
        }
        row
    }
}

/// Keep a robot point (at `frame`+`offset`) on the `+normal` side of a world plane
/// (`normal·x = plane_offset`) with a `margin`. One-sided; useful for ground/table clearance.
#[derive(Clone, Copy, Debug)]
pub struct PlaneCollisionCost {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub plane_offset: f64,
    pub margin: f64,
    pub weight: f64,
}

impl PlaneCollisionCost {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame: usize,
        offset: Vector3<f64>,
        normal: Vector3<f64>,
        plane_offset: f64,
        margin: f64,
        weight: f64,
    ) -> Self {
        Self { frame, offset, normal: normal.normalize(), plane_offset, margin, weight }
    }

    fn point(&self, robot: &Robot, q: &[f64]) -> Vector3<f64> {
        (robot.frame_pose(q, self.frame) * Point3::from(self.offset)).coords
    }

    /// Signed clearance above the plane (negative ⇒ below it).
    fn clearance(&self, robot: &Robot, q: &[f64]) -> f64 {
        self.normal.dot(&self.point(robot, q)) - self.plane_offset
    }
}

impl Cost for PlaneCollisionCost {
    fn dim(&self, _robot: &Robot) -> usize {
        1
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        let viol = (self.margin - self.clearance(robot, q)).max(0.0);
        DVector::from_row_slice(&[self.weight * viol])
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let n = robot.dof();
        let mut row = DMatrix::zeros(1, n);
        if self.margin - self.clearance(robot, q) > 0.0 {
            let c = self.point(robot, q);
            let jp = robot.point_jacobian(q, self.frame, &c);
            for col in 0..n {
                row[(0, col)] = -self.weight * self.normal.dot(&jp.column(col));
            }
        }
        row
    }
}

/// Closest points between segment A(`a0`→`a1`) and segment B(`b0`→`b1`), returning the barycentric
/// coordinate `s` on A, the point on A, and the point on B. (Ericson, Real-Time Collision Detection.)
fn segment_segment(
    a0: Vector3<f64>, a1: Vector3<f64>, b0: Vector3<f64>, b1: Vector3<f64>,
) -> (f64, Vector3<f64>, Vector3<f64>) {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.dot(&d1);
    let e = d2.dot(&d2);
    let f = d2.dot(&r);
    let eps = 1e-12;
    let (mut s, t);
    if a <= eps && e <= eps {
        s = 0.0;
        t = 0.0;
    } else if a <= eps {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(&r);
        if e <= eps {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(&d2);
            let denom = a * e - b * b;
            s = if denom.abs() > eps { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut tt = (b * s + f) / e;
            if tt < 0.0 {
                tt = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if tt > 1.0 {
                tt = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            t = tt;
        }
    }
    (s, a0 + d1 * s, b0 + d2 * t)
}

/// Keep a robot **capsule** (a fattened segment between two attached points, radius `radius`) clear
/// of a world capsule obstacle (`obs_p0`→`obs_p1`, radius `obs_radius`). A sphere obstacle is a
/// degenerate capsule (`obs_p0 == obs_p1`); a robot sphere is `offset_a == offset_b`. Covers
/// link-vs-obstacle and link-vs-link self-collision with smooth, analytic gradients.
#[derive(Clone, Copy, Debug)]
pub struct CapsuleCollisionCost {
    pub frame_a: usize,
    pub offset_a: Vector3<f64>,
    pub frame_b: usize,
    pub offset_b: Vector3<f64>,
    pub radius: f64,
    pub obs_p0: Vector3<f64>,
    pub obs_p1: Vector3<f64>,
    pub obs_radius: f64,
    pub margin: f64,
    pub weight: f64,
}

impl CapsuleCollisionCost {
    fn endpoints(&self, robot: &Robot, q: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
        let a = (robot.frame_pose(q, self.frame_a) * Point3::from(self.offset_a)).coords;
        let b = (robot.frame_pose(q, self.frame_b) * Point3::from(self.offset_b)).coords;
        (a, b)
    }

    /// (surface clearance, barycentric s on the robot segment, contact normal robot→obstacle-away).
    fn geometry(&self, robot: &Robot, q: &[f64]) -> (f64, f64, Vector3<f64>) {
        let (a, b) = self.endpoints(robot, q);
        let (s, pc_r, pc_o) = segment_segment(a, b, self.obs_p0, self.obs_p1);
        let diff = pc_r - pc_o;
        let d = diff.norm();
        let clearance = d - self.radius - self.obs_radius;
        let nhat = if d > 1e-9 { diff / d } else { Vector3::z() };
        (clearance, s, nhat)
    }
}

impl Cost for CapsuleCollisionCost {
    fn dim(&self, _robot: &Robot) -> usize {
        1
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        let (clearance, _, _) = self.geometry(robot, q);
        DVector::from_row_slice(&[self.weight * (self.margin - clearance).max(0.0)])
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let n = robot.dof();
        let mut row = DMatrix::zeros(1, n);
        let (clearance, s, nhat) = self.geometry(robot, q);
        if self.margin - clearance > 0.0 {
            // Closest robot point moves as (1−s)·A + s·B (s held fixed to first order).
            let (a, b) = self.endpoints(robot, q);
            let ja = robot.point_jacobian(q, self.frame_a, &a);
            let jb = robot.point_jacobian(q, self.frame_b, &b);
            let jp = ja * (1.0 - s) + jb * s;
            for col in 0..n {
                row[(0, col)] = -self.weight * nhat.dot(&jp.column(col));
            }
        }
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_str, pose_error, Cost, PoseCost, SolveOptions, TrajectoryProblem};

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
    fn capsule_collision_jacobian_matches_finite_difference() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let q = [0.1, 0.6, -0.4, 0.3, 0.0, 0.1];
        let a = robot.frame_pose(&q, 3).translation.vector;
        let b = robot.frame_pose(&q, 6).translation.vector;
        let mid = (a + b) * 0.5;
        // A short obstacle capsule laterally offset from the link → interior closest point.
        let cost = CapsuleCollisionCost {
            frame_a: 3, offset_a: Vector3::zeros(), frame_b: 6, offset_b: Vector3::zeros(), radius: 0.03,
            obs_p0: mid + Vector3::new(0.05, -0.05, 0.0), obs_p1: mid + Vector3::new(0.05, 0.05, 0.0),
            obs_radius: 0.05, margin: 0.03, weight: 1.0,
        };
        assert!(cost.residual(&robot, &q)[0] > 0.0, "cost must be active for the FD check");
        let analytic = cost.jacobian(&robot, &q);
        let eps = 1e-7;
        for i in 0..6 {
            let mut qp = q;
            qp[i] += eps;
            let fd = (cost.residual(&robot, &qp)[0] - cost.residual(&robot, &q)[0]) / eps;
            assert!((analytic[(0, i)] - fd).abs() < 2e-3, "col {i}: analytic {} fd {}", analytic[(0, i)], fd);
        }
    }

    #[test]
    fn collision_jacobian_matches_finite_difference() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let q = [0.0, 0.5, -0.3, 0.2, 0.1, 0.0];
        let tool = robot.fk(&q).translation.vector;
        // Obstacle overlapping the tool so the cost is active.
        let cost = SphereCollisionCost::new(6, Vector3::new(0.0, 0.0, 0.05), 0.02, tool + Vector3::new(0.03, 0.0, 0.0), 0.05, 0.02, 1.0);
        let analytic = cost.jacobian(&robot, &q);
        let eps = 1e-7;
        for i in 0..6 {
            let mut qp = q;
            qp[i] += eps;
            let fd = (cost.residual(&robot, &qp)[0] - cost.residual(&robot, &q)[0]) / eps;
            assert!((analytic[(0, i)] - fd).abs() < 1e-3, "col {i}: analytic {} fd {}", analytic[(0, i)], fd);
        }
    }

    #[test]
    fn trajectory_routes_around_an_obstacle() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        // Tool swings through the x-z plane; midpoint passes near straight-up.
        let qa = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let qc = [0.0, -0.5, 0.0, 0.0, 0.0, 0.0];
        let (ta, tc) = (robot.fk(&qa), robot.fk(&qc));
        let tool_r = 0.02;
        let obs_c = Vector3::new(0.0, 0.0, 0.92); // sits above the midpoint tool position
        let obs_r = 0.08;
        let margin = 0.02;

        let steps = 25;
        let mut costs: Vec<Vec<Box<dyn Cost>>> = (0..steps).map(|_| Vec::new()).collect();
        costs[0].push(Box::new(PoseCost::new(ta, 100.0, 100.0)));
        costs[steps - 1].push(Box::new(PoseCost::new(tc, 100.0, 100.0)));
        for c in costs.iter_mut() {
            c.push(Box::new(SphereCollisionCost::new(
                6, Vector3::new(0.0, 0.0, 0.05), tool_r, obs_c, obs_r, margin, 50.0,
            )));
        }

        let prob = TrajectoryProblem {
            robot: &robot,
            costs,
            vel_weight: 0.4,
            opts: SolveOptions { max_iters: 600, ..SolveOptions::default() },
        };
        let init: Vec<Vec<f64>> = (0..steps)
            .map(|k| {
                let s = k as f64 / (steps - 1) as f64;
                (0..6).map(|i| qa[i] * (1.0 - s) + qc[i] * s).collect()
            })
            .collect();
        let res = prob.solve(&init);

        // Endpoints preserved.
        assert!(pose_error(&robot.fk(&res.qs[0]), &ta).norm() < 5e-3);
        assert!(pose_error(&robot.fk(&res.qs[steps - 1]), &tc).norm() < 5e-3);

        // Worst (most negative) surface clearance across a trajectory; the tool tip via fk is the
        // collision sphere center.
        let worst_clearance = |qs: &[Vec<f64>]| {
            qs.iter().fold(f64::INFINITY, |m, q| {
                let tool = robot.fk(q).translation.vector;
                m.min((tool - obs_c).norm() - tool_r - obs_r)
            })
        };
        let before = worst_clearance(&init);
        let after = worst_clearance(&res.qs);
        assert!(before < -0.01, "sanity: straight-line init should penetrate (was {before})");
        assert!(after > before + 0.02, "optimization should improve clearance ({before} → {after})");
        assert!(after > -5e-3, "optimized path should not meaningfully penetrate (clearance {after})");
    }
}
