//! Trajectory optimization — optimize a whole motion `q₀…q_{T-1}` jointly, not frame-by-frame.
//!
//! Per-timestep costs (reach these waypoints, avoid this obstacle) plus velocity smoothness
//! coupling consecutive steps. That coupling makes the Gauss-Newton system **block-tridiagonal**,
//! so we solve each LM step with a block-Thomas elimination — linear in `T`, the sparse structure
//! PyRoki/jaxls exploit — using dense `nalgebra` blocks (WASM-clean, no external sparse solver).

use crate::{Cost, Robot, SolveOptions};
use nalgebra::{DMatrix, DVector};

/// A `T`-step trajectory problem over one robot. `costs[t]` applies at timestep `t`;
/// `vel_weight` penalizes `q_{t+1} − q_t` (velocity smoothness).
pub struct TrajectoryProblem<'a> {
    pub robot: &'a Robot,
    pub costs: Vec<Vec<Box<dyn Cost>>>,
    pub vel_weight: f64,
    pub opts: SolveOptions,
}

/// Result of a trajectory solve.
#[derive(Clone, Debug)]
pub struct TrajectoryResult {
    pub qs: Vec<Vec<f64>>,
    /// Final objective residual norm (nonzero by design — smoothness trades against tasks).
    pub error: f64,
    pub iters: usize,
    /// Whether the gradient settled (a proper optimality measure for a nonzero-objective problem).
    pub converged: bool,
}

impl<'a> TrajectoryProblem<'a> {
    pub fn new(robot: &'a Robot, costs: Vec<Vec<Box<dyn Cost>>>, vel_weight: f64) -> Self {
        Self { robot, costs, vel_weight, opts: SolveOptions::default() }
    }

    fn steps(&self) -> usize {
        self.costs.len()
    }

    fn total_cost(&self, xs: &[DVector<f64>]) -> f64 {
        let mut c = 0.0;
        for (t, q) in xs.iter().enumerate() {
            for cost in &self.costs[t] {
                c += cost.residual(self.robot, q.as_slice()).norm_squared();
            }
        }
        let wv2 = self.vel_weight * self.vel_weight;
        for t in 0..xs.len().saturating_sub(1) {
            c += wv2 * (&xs[t + 1] - &xs[t]).norm_squared();
        }
        c
    }

    /// Solve from an initial trajectory guess (`q_init[t]` per timestep).
    pub fn solve(&self, q_init: &[Vec<f64>]) -> TrajectoryResult {
        let n = self.robot.dof();
        let t_steps = self.steps();
        let wv2 = self.vel_weight * self.vel_weight;
        let mut xs: Vec<DVector<f64>> = q_init.iter().map(|q| DVector::from_row_slice(q)).collect();
        let mut lambda = self.opts.lambda0;
        let mut cost = self.total_cost(&xs);
        let mut gnorm = f64::INFINITY;
        let mut iters = 0;

        'outer: for it in 0..self.opts.max_iters {
            iters = it + 1;

            // Assemble the block-tridiagonal Gauss-Newton system: diagonal blocks `d`, super-diagonal
            // blocks `b` (H_{t,t+1}), gradient `g`.
            let mut d: Vec<DMatrix<f64>> = (0..t_steps).map(|_| DMatrix::zeros(n, n)).collect();
            let mut b: Vec<DMatrix<f64>> =
                (0..t_steps.saturating_sub(1)).map(|_| DMatrix::zeros(n, n)).collect();
            let mut g: Vec<DVector<f64>> = (0..t_steps).map(|_| DVector::zeros(n)).collect();

            for t in 0..t_steps {
                let q = xs[t].as_slice();
                for cost_i in &self.costs[t] {
                    let jt = cost_i.jacobian(self.robot, q).transpose();
                    d[t] += &jt * &jt.transpose();
                    g[t] += &jt * &cost_i.residual(self.robot, q);
                }
            }
            for t in 0..t_steps.saturating_sub(1) {
                let diff = &xs[t + 1] - &xs[t];
                for k in 0..n {
                    d[t][(k, k)] += wv2;
                    d[t + 1][(k, k)] += wv2;
                    b[t][(k, k)] -= wv2;
                }
                let gt = &diff * wv2;
                g[t] -= &gt;
                g[t + 1] += &gt;
            }

            gnorm = g.iter().map(|v| v.norm_squared()).sum::<f64>().sqrt();
            if gnorm < self.opts.tol {
                break;
            }

            // LM inner loop: damp diagonal, block-Thomas solve H·Δ = g, accept if cost drops.
            loop {
                let mut dd = d.clone();
                for dt in dd.iter_mut() {
                    for k in 0..n {
                        dt[(k, k)] += lambda;
                    }
                }

                // Forward elimination.
                let mut dprime: Vec<DMatrix<f64>> = Vec::with_capacity(t_steps);
                let mut gprime: Vec<DVector<f64>> = Vec::with_capacity(t_steps);
                dprime.push(dd[0].clone());
                gprime.push(g[0].clone());
                let mut ok = true;
                for t in 1..t_steps {
                    // x = D'_{t-1}⁻¹ · B_{t-1}; blocks are symmetric so xᵀ = B_{t-1}·D'_{t-1}⁻¹.
                    let x = match dprime[t - 1].clone().lu().solve(&b[t - 1]) {
                        Some(x) => x,
                        None => {
                            ok = false;
                            break;
                        }
                    };
                    let xt = x.transpose();
                    dprime.push(&dd[t] - &xt * &b[t - 1]);
                    gprime.push(&g[t] - &xt * &gprime[t - 1]);
                }
                if !ok {
                    lambda *= 3.0;
                    if lambda > 1e12 {
                        break 'outer;
                    }
                    continue;
                }

                // Back substitution.
                let mut delta: Vec<DVector<f64>> = vec![DVector::zeros(n); t_steps];
                match dprime[t_steps - 1].clone().lu().solve(&gprime[t_steps - 1]) {
                    Some(v) => delta[t_steps - 1] = v,
                    None => {
                        lambda *= 3.0;
                        if lambda > 1e12 {
                            break 'outer;
                        }
                        continue;
                    }
                }
                for t in (0..t_steps - 1).rev() {
                    let rhs = &gprime[t] - &b[t] * &delta[t + 1];
                    match dprime[t].clone().lu().solve(&rhs) {
                        Some(v) => delta[t] = v,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    lambda *= 3.0;
                    if lambda > 1e12 {
                        break 'outer;
                    }
                    continue;
                }

                let xs_new: Vec<DVector<f64>> = xs.iter().zip(&delta).map(|(x, dq)| x - dq).collect();
                let cost_new = self.total_cost(&xs_new);
                if cost_new < cost {
                    xs = xs_new;
                    cost = cost_new;
                    lambda = (lambda * 0.5).max(1e-12);
                    break;
                }
                lambda *= 3.0;
                if lambda > 1e12 {
                    break 'outer;
                }
            }
        }

        let qs = xs.iter().map(|x| x.as_slice().to_vec()).collect();
        TrajectoryResult { qs, error: cost.sqrt(), iters, converged: gnorm < 1e-3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_str, pose_error, Cost, PoseCost, SolveOptions};

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
    fn trajectory_through_waypoints_is_smooth() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let qa = [0.0; 6];
        let qb = [0.4, -0.5, 0.7, 0.2, 0.3, -0.4];
        let qc = [-0.3, 0.6, -0.5, 0.5, -0.2, 0.3];
        let (ta, tb, tc) = (robot.fk(&qa), robot.fk(&qb), robot.fk(&qc));

        let steps = 21;
        let mut costs: Vec<Vec<Box<dyn Cost>>> = (0..steps).map(|_| Vec::new()).collect();
        costs[0].push(Box::new(PoseCost::new(ta, 100.0, 100.0)));
        costs[steps / 2].push(Box::new(PoseCost::new(tb, 100.0, 100.0)));
        costs[steps - 1].push(Box::new(PoseCost::new(tc, 100.0, 100.0)));

        let prob = TrajectoryProblem {
            robot: &robot,
            costs,
            vel_weight: 0.5,
            opts: SolveOptions { max_iters: 400, ..SolveOptions::default() },
        };

        // Linear-interpolation warm start from qa to qc.
        let init: Vec<Vec<f64>> = (0..steps)
            .map(|k| {
                let s = k as f64 / (steps - 1) as f64;
                (0..6).map(|i| qa[i] * (1.0 - s) + qc[i] * s).collect()
            })
            .collect();

        let res = prob.solve(&init);
        assert!(res.converged, "gradient did not settle (err={}, iters={})", res.error, res.iters);

        // All three waypoint poses are hit.
        assert!(pose_error(&robot.fk(&res.qs[0]), &ta).norm() < 5e-3);
        assert!(pose_error(&robot.fk(&res.qs[steps / 2]), &tb).norm() < 5e-3);
        assert!(pose_error(&robot.fk(&res.qs[steps - 1]), &tc).norm() < 5e-3);

        // Motion is smooth: no large per-step joint jumps.
        let mut max_step: f64 = 0.0;
        for k in 0..steps - 1 {
            let d = (0..6).map(|i| (res.qs[k + 1][i] - res.qs[k][i]).powi(2)).sum::<f64>().sqrt();
            max_step = max_step.max(d);
        }
        assert!(max_step < 1.0, "max per-step joint motion {max_step}");
    }
}
