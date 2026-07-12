//! Augmented-Lagrangian hard constraints. Soft costs only *trade off* against each other; some
//! things must actually hold (stay above the table, don't exceed a joint limit, keep clearance).
//! This wraps the least-squares solve in an outer AL loop: penalize inequality violations, lift
//! the penalty and update multipliers until every constraint `c(q) ≤ 0` holds to tolerance.
//!
//! A constraint is any [`Cost`] whose `residual` returns the *raw* signed values `cᵢ` (want ≤ 0)
//! and whose `jacobian` is `dc/dq` (ungated).

use crate::{Cost, Robot, SolveOptions};
use nalgebra::{DMatrix, DVector, Point3, Vector3};

/// Inequality constraint: keep a robot point on the `+normal` side of a plane, i.e.
/// `c(q) = (plane_offset + margin) − normal·point ≤ 0`.
#[derive(Clone, Copy, Debug)]
pub struct PlaneConstraint {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub normal: Vector3<f64>,
    pub plane_offset: f64,
    pub margin: f64,
}

impl PlaneConstraint {
    pub fn new(frame: usize, offset: Vector3<f64>, normal: Vector3<f64>, plane_offset: f64, margin: f64) -> Self {
        Self { frame, offset, normal: normal.normalize(), plane_offset, margin }
    }

    fn point(&self, robot: &Robot, q: &[f64]) -> Vector3<f64> {
        (robot.frame_pose(q, self.frame) * Point3::from(self.offset)).coords
    }
}

impl Cost for PlaneConstraint {
    fn dim(&self, _robot: &Robot) -> usize {
        1
    }

    fn residual(&self, robot: &Robot, q: &[f64]) -> DVector<f64> {
        let c = (self.plane_offset + self.margin) - self.normal.dot(&self.point(robot, q));
        DVector::from_row_slice(&[c])
    }

    fn jacobian(&self, robot: &Robot, q: &[f64]) -> DMatrix<f64> {
        let n = robot.dof();
        let mut row = DMatrix::zeros(1, n);
        let p = self.point(robot, q);
        let jp = robot.point_jacobian(q, self.frame, &p);
        for col in 0..n {
            row[(0, col)] = -self.normal.dot(&jp.column(col)); // dc/dq
        }
        row
    }
}

/// Options for [`solve_al`].
#[derive(Clone, Copy, Debug)]
pub struct AlOptions {
    pub max_outer: usize,
    pub mu0: f64,
    pub rho: f64,
    pub tol: f64,
    pub inner: SolveOptions,
}

impl Default for AlOptions {
    fn default() -> Self {
        Self { max_outer: 25, mu0: 10.0, rho: 10.0, tol: 1e-4, inner: SolveOptions::default() }
    }
}

/// Result of an augmented-Lagrangian solve.
#[derive(Clone, Debug)]
pub struct AlResult {
    pub q: Vec<f64>,
    /// Max inequality violation `max(0, cᵢ)` at the solution.
    pub max_violation: f64,
    pub outer_iters: usize,
}

fn con_dims(robot: &Robot, constraints: &[Box<dyn Cost>]) -> Vec<usize> {
    constraints.iter().map(|c| c.dim(robot)).collect()
}

/// Minimize the `objective` costs subject to `constraints` (each `c(q) ≤ 0`) from seed `q0`.
pub fn solve_al(
    robot: &Robot,
    objective: &[Box<dyn Cost>],
    constraints: &[Box<dyn Cost>],
    q0: &[f64],
    al: &AlOptions,
) -> AlResult {
    let cdims = con_dims(robot, constraints);
    let total_con: usize = cdims.iter().sum();
    let mut lambda = DVector::zeros(total_con);
    let mut mu = al.mu0;
    let mut q = q0.to_vec();
    let mut max_violation = f64::INFINITY;
    let mut outer = 0;

    for o in 0..al.max_outer {
        outer = o + 1;
        q = inner_solve(robot, objective, constraints, &cdims, mu, &lambda, &q, &al.inner);

        // Evaluate raw constraints, update multipliers, measure violation.
        let mut viol: f64 = 0.0;
        let mut off = 0;
        for (ci, c) in constraints.iter().enumerate() {
            let cv = c.residual(robot, &q);
            for i in 0..cdims[ci] {
                let val = cv[i];
                viol = viol.max(val.max(0.0));
                lambda[off + i] = (lambda[off + i] + mu * val).max(0.0);
            }
            off += cdims[ci];
        }
        max_violation = viol;
        if viol < al.tol {
            break;
        }
        mu = (mu * al.rho).min(1e10);
    }

    AlResult { q, max_violation, outer_iters: outer }
}

/// Dense LM minimizing `objective` residuals + the augmented-Lagrangian penalty on `constraints`.
fn inner_solve(
    robot: &Robot,
    objective: &[Box<dyn Cost>],
    constraints: &[Box<dyn Cost>],
    cdims: &[usize],
    mu: f64,
    lambda: &DVector<f64>,
    q0: &[f64],
    opts: &SolveOptions,
) -> Vec<f64> {
    let n = robot.dof();
    let sm = mu.sqrt();

    let assemble = |q: &[f64]| -> (DVector<f64>, DMatrix<f64>) {
        let obj_dim: usize = objective.iter().map(|c| c.dim(robot)).sum();
        let total = obj_dim + cdims.iter().sum::<usize>();
        let mut r = DVector::zeros(total);
        let mut j = DMatrix::zeros(total, n);
        let mut off = 0;
        for c in objective {
            let d = c.dim(robot);
            r.rows_mut(off, d).copy_from(&c.residual(robot, q));
            j.view_mut((off, 0), (d, n)).copy_from(&c.jacobian(robot, q));
            off += d;
        }
        let mut coff = 0;
        for (ci, c) in constraints.iter().enumerate() {
            let cv = c.residual(robot, q);
            let cj = c.jacobian(robot, q);
            for i in 0..cdims[ci] {
                let shifted = cv[i] + lambda[coff + i] / mu;
                if shifted > 0.0 {
                    r[off + i] = sm * shifted;
                    for col in 0..n {
                        j[(off + i, col)] = sm * cj[(i, col)];
                    }
                }
            }
            off += cdims[ci];
            coff += cdims[ci];
        }
        (r, j)
    };

    let mut q = DVector::from_row_slice(q0);
    let mut lambda_lm = opts.lambda0;
    let (mut r, _) = assemble(q.as_slice());
    let mut cost = r.norm_squared();

    'outer: for _ in 0..opts.max_iters {
        if r.norm() < opts.tol {
            break;
        }
        let (_, jm) = assemble(q.as_slice());
        let jt = jm.transpose();
        let jtj = &jt * &jm;
        let g = &jt * &r;
        loop {
            let mut a = jtj.clone();
            for d in 0..n {
                a[(d, d)] += lambda_lm;
            }
            let dq = match a.clone().cholesky() {
                Some(ch) => ch.solve(&g),
                None => a.lu().solve(&g).unwrap_or_else(|| DVector::zeros(n)),
            };
            let q_new = &q - &dq;
            let (r_new, _) = assemble(q_new.as_slice());
            let cost_new = r_new.norm_squared();
            if cost_new < cost {
                q = q_new;
                r = r_new;
                cost = cost_new;
                lambda_lm = (lambda_lm * 0.5).max(1e-12);
                break;
            }
            lambda_lm *= 3.0;
            if lambda_lm > 1e12 {
                break 'outer;
            }
        }
    }
    q.as_slice().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_str, Cost, PointCost};

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
    fn hard_constraint_holds_where_a_soft_penalty_would_not() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        // Objective pulls the tool to a target well BELOW a plane it must stay above.
        let target = Vector3::new(0.2, 0.0, 0.5);
        let objective: Vec<Box<dyn Cost>> =
            vec![Box::new(PointCost::new(6, Vector3::new(0.0, 0.0, 0.05), target, 1.0))];
        // Hard: tool tip must stay at or above z = 0.75.
        let plane_z = 0.75;
        let constraints: Vec<Box<dyn Cost>> = vec![Box::new(PlaneConstraint::new(
            6, Vector3::new(0.0, 0.0, 0.05), Vector3::z(), plane_z, 0.0,
        ))];

        let res = solve_al(&robot, &objective, &constraints, &[0.1, 0.2, 0.2, 0.0, 0.0, 0.0], &AlOptions::default());
        let tool_z = robot.fk(&res.q).translation.vector.z;

        assert!(res.max_violation < 1e-3, "constraint violated by {}", res.max_violation);
        assert!(tool_z > plane_z - 1e-3, "tool sank below the plane: z = {tool_z}");
        // The constraint is genuinely binding: the tool is pressed to the plane, not floating far above.
        assert!(tool_z < plane_z + 0.02, "constraint should be active; z = {tool_z}");
    }
}
