//! iLQR / DDP — nonlinear optimal control for a torque-controlled robot. We plan an open-loop
//! torque sequence that swings the arm to a goal state by iterating: forward-rollout the current
//! controls through `forward_dynamics` + semi-implicit Euler, linearize the discrete dynamics
//! about that rollout by finite differences, run an LQR-like backward Riccati pass to get
//! feedforward `k_t` and gains `K_t`, then take a backtracking line-search step. Gauss-Newton
//! (iLQR): we keep the linear dynamics term only, so no second-order dynamics tensor is needed.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, Vector3};
use ferromotion_core::{forward_dynamics, LinkInertia, Robot};

/// A finite-horizon optimal-control problem: swing the robot from a start state toward a goal
/// state `x_goal = [q_goal; qd_goal]`, trading state error against control effort.
///
/// Cost `J = Σ_{t=0}^{N-1} ½(xₜ−x_g)ᵀQ(xₜ−x_g) + ½ uₜᵀR uₜ + ½(x_N−x_g)ᵀQ_f(x_N−x_g)`, with
/// diagonal `Q = diag(w_q·Iₙ, w_qd·Iₙ)`, `Q_f = diag(w_qf·Iₙ, w_qdf·Iₙ)` and `R = r·Iₙ`.
/// The state is `x = [q; qd]` (2n) and the control is joint torque `u = τ` (n).
pub struct IlqrProblem<'a> {
    pub robot: &'a Robot,
    pub inertia: &'a [LinkInertia],
    pub gravity: Vector3<f64>,
    /// Number of control steps N (states run `x_0 … x_N`).
    pub horizon: usize,
    pub dt: f64,
    pub q_goal: Vec<f64>,
    pub qd_goal: Vec<f64>,
    /// Running state weights on position / velocity.
    pub w_q: f64,
    pub w_qd: f64,
    /// Control (torque) weight.
    pub r: f64,
    /// Terminal state weights on position / velocity.
    pub w_qf: f64,
    pub w_qdf: f64,
}

/// Result of an iLQR solve: the planned torque sequence and the state trajectory it produces.
#[derive(Clone, Debug)]
pub struct IlqrResult {
    /// Planned torques `u_0 … u_{N-1}` (length N, each of dof).
    pub taus: Vec<Vec<f64>>,
    /// Rolled-out joint positions `q_0 … q_N` (length N+1, each of dof).
    pub qs: Vec<Vec<f64>>,
    /// Rolled-out joint velocities `qd_0 … qd_N` (length N+1, each of dof).
    pub qds: Vec<Vec<f64>>,
    /// Cost after each accepted iteration (`costs[0]` = initial rollout cost); monotone decreasing.
    pub costs: Vec<f64>,
    pub converged: bool,
    pub iters: usize,
}

impl IlqrProblem<'_> {
    fn n(&self) -> usize {
        self.robot.dof()
    }

    /// Goal state `x_g = [q_goal; qd_goal]` (2n).
    fn x_goal(&self) -> DVector<f64> {
        let n = self.n();
        let mut xg = DVector::zeros(2 * n);
        for i in 0..n {
            xg[i] = self.q_goal[i];
            xg[n + i] = self.qd_goal[i];
        }
        xg
    }

    /// One discrete dynamics step `x_{t+1} = f(x_t, u_t)` via forward dynamics + semi-implicit
    /// Euler (`qd += qdd·dt` then `q += qd·dt`).
    fn step(&self, x: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        let n = self.n();
        let q: Vec<f64> = (0..n).map(|i| x[i]).collect();
        let qd: Vec<f64> = (0..n).map(|i| x[n + i]).collect();
        let tau: Vec<f64> = (0..n).map(|i| u[i]).collect();
        let qdd = forward_dynamics(self.robot, self.inertia, &q, &qd, &tau, self.gravity);
        let mut out = DVector::zeros(2 * n);
        for i in 0..n {
            let qd_next = qd[i] + qdd[i] * self.dt;
            out[n + i] = qd_next;
            out[i] = q[i] + qd_next * self.dt;
        }
        out
    }

    /// Roll the control sequence `u` forward from `x0`, returning the state trajectory (N+1 states).
    fn rollout(&self, x0: &DVector<f64>, u: &[DVector<f64>]) -> Vec<DVector<f64>> {
        let mut xs = Vec::with_capacity(self.horizon + 1);
        xs.push(x0.clone());
        for (t, ut) in u.iter().enumerate() {
            let next = self.step(&xs[t], ut);
            xs.push(next);
        }
        xs
    }

    /// Trajectory cost for the given states/controls.
    fn cost(&self, xs: &[DVector<f64>], u: &[DVector<f64>], qx: &DMatrix<f64>, qf: &DMatrix<f64>, rmat: &DMatrix<f64>) -> f64 {
        let xg = self.x_goal();
        let mut j = 0.0;
        for t in 0..self.horizon {
            let dx = &xs[t] - &xg;
            j += 0.5 * dx.dot(&(qx * &dx));
            j += 0.5 * u[t].dot(&(rmat * &u[t]));
        }
        let dxn = &xs[self.horizon] - &xg;
        j += 0.5 * dxn.dot(&(qf * &dxn));
        j
    }

    /// Finite-difference linearization of `f` about `(x, u)`: `A = ∂f/∂x` (2n×2n), `B = ∂f/∂u`
    /// (2n×n), by central differences.
    fn linearize(&self, x: &DVector<f64>, u: &DVector<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = self.n();
        let s = 2 * n;
        let eps = 1e-6;
        let mut a = DMatrix::zeros(s, s);
        for j in 0..s {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps;
            xm[j] -= eps;
            let col = (self.step(&xp, u) - self.step(&xm, u)) / (2.0 * eps);
            a.set_column(j, &col);
        }
        let mut b = DMatrix::zeros(s, n);
        for j in 0..n {
            let mut up = u.clone();
            let mut um = u.clone();
            up[j] += eps;
            um[j] -= eps;
            let col = (self.step(x, &up) - self.step(x, &um)) / (2.0 * eps);
            b.set_column(j, &col);
        }
        (a, b)
    }
}

/// Invert a (small) symmetric matrix, adding an increasing multiple of the identity until it is
/// numerically invertible (Levenberg-style regularization for the backward pass).
fn safe_inverse(m: &DMatrix<f64>) -> DMatrix<f64> {
    let nr = m.nrows();
    let mut reg = 1e-9;
    for _ in 0..24 {
        let cand = m + DMatrix::identity(nr, nr) * reg;
        if let Some(inv) = cand.try_inverse() {
            return inv;
        }
        reg *= 10.0;
    }
    DMatrix::identity(nr, nr)
}

/// Solve the iLQR problem starting from `(q0, qd0)`. Returns the optimized torque plan and the
/// state trajectory it rolls out to.
pub fn solve_ilqr(problem: &IlqrProblem, q0: &[f64], qd0: &[f64]) -> IlqrResult {
    let n = problem.n();
    let s = 2 * n;
    let nn = problem.horizon;

    // Cost matrices (diagonal).
    let mut qx_diag = DVector::zeros(s);
    let mut qf_diag = DVector::zeros(s);
    for i in 0..n {
        qx_diag[i] = problem.w_q;
        qx_diag[n + i] = problem.w_qd;
        qf_diag[i] = problem.w_qf;
        qf_diag[n + i] = problem.w_qdf;
    }
    let qx = DMatrix::from_diagonal(&qx_diag);
    let qf = DMatrix::from_diagonal(&qf_diag);
    let rmat = DMatrix::<f64>::identity(n, n) * problem.r;

    let xg = problem.x_goal();

    // Initial state and zero control sequence.
    let mut x0 = DVector::zeros(s);
    for i in 0..n {
        x0[i] = q0[i];
        x0[n + i] = qd0[i];
    }
    let mut u: Vec<DVector<f64>> = vec![DVector::zeros(n); nn];
    let mut xs = problem.rollout(&x0, &u);
    let mut j = problem.cost(&xs, &u, &qx, &qf, &rmat);

    let mut costs = vec![j];
    let tol = 1e-5;
    let max_iters = 200usize;
    let alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
    let mut converged = false;
    let mut iters = 0usize;

    for iter in 0..max_iters {
        iters = iter + 1;

        // (1) Linearize about the current rollout.
        let mut a_list = Vec::with_capacity(nn);
        let mut b_list = Vec::with_capacity(nn);
        for t in 0..nn {
            let (a, b) = problem.linearize(&xs[t], &u[t]);
            a_list.push(a);
            b_list.push(b);
        }

        // (2) Backward Riccati pass.
        let mut v_x = &qf * (&xs[nn] - &xg);
        let mut v_xx = qf.clone();
        let mut k_ff: Vec<DVector<f64>> = vec![DVector::zeros(n); nn];
        let mut k_fb: Vec<DMatrix<f64>> = vec![DMatrix::zeros(n, s); nn];
        for t in (0..nn).rev() {
            let a = &a_list[t];
            let b = &b_list[t];
            let at = a.transpose();
            let bt = b.transpose();

            let l_x = &qx * (&xs[t] - &xg); // 2n
            let l_u = &rmat * &u[t]; // n

            let q_x = &l_x + &at * &v_x; // 2n
            let q_u = &l_u + &bt * &v_x; // n
            let q_xx = &qx + &at * &v_xx * a; // 2n×2n
            let mut q_uu = &rmat + &bt * &v_xx * b; // n×n
            q_uu = (&q_uu + q_uu.transpose()) * 0.5;
            let q_ux = &bt * &v_xx * a; // n×2n

            let q_uu_inv = safe_inverse(&q_uu);
            let k = -&q_uu_inv * &q_u; // feedforward, n
            let big_k = -&q_uu_inv * &q_ux; // gain, n×2n

            let kt = big_k.transpose();
            v_x = &q_x + &kt * &q_uu * &k + &kt * &q_u + q_ux.transpose() * &k;
            v_xx = &q_xx + &kt * &q_uu * &big_k + &kt * &q_ux + q_ux.transpose() * &big_k;
            v_xx = (&v_xx + v_xx.transpose()) * 0.5;

            k_ff[t] = k;
            k_fb[t] = big_k;
        }

        // (3) Forward pass with backtracking line search.
        let mut accepted: Option<(Vec<DVector<f64>>, Vec<DVector<f64>>, f64)> = None;
        for &alpha in alphas.iter() {
            let mut xn = Vec::with_capacity(nn + 1);
            xn.push(x0.clone());
            let mut un = Vec::with_capacity(nn);
            let mut valid = true;
            for t in 0..nn {
                let dx = &xn[t] - &xs[t];
                let ut = &u[t] + &k_ff[t] * alpha + &k_fb[t] * &dx;
                let xt_next = problem.step(&xn[t], &ut);
                if xt_next.iter().any(|v| !v.is_finite()) {
                    valid = false;
                    break;
                }
                un.push(ut);
                xn.push(xt_next);
            }
            if !valid {
                continue;
            }
            let jn = problem.cost(&xn, &un, &qx, &qf, &rmat);
            if jn.is_finite() && jn < j {
                accepted = Some((xn, un, jn)); // first (largest) step that decreases cost
                break;
            }
        }

        match accepted {
            Some((xn, un, jn)) => {
                let improvement = j - jn;
                xs = xn;
                u = un;
                j = jn;
                costs.push(j);
                if improvement < tol {
                    converged = true;
                    break;
                }
            }
            None => {
                // No step reduced the cost — at a local optimum for this quadratic model.
                converged = true;
                break;
            }
        }
    }

    // Unpack the final trajectory.
    let taus: Vec<Vec<f64>> = u.iter().map(|ut| (0..n).map(|i| ut[i]).collect()).collect();
    let qs: Vec<Vec<f64>> = xs.iter().map(|xt| (0..n).map(|i| xt[i]).collect()).collect();
    let qds: Vec<Vec<f64>> = xs.iter().map(|xt| (0..n).map(|i| xt[n + i]).collect()).collect();

    IlqrResult { taus, qs, qds, costs, converged, iters }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use ferromotion_core::{forward_dynamics, from_urdf_full};

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0 0" rpy="0 0 0"/><mass value="1.5"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    fn swing_problem<'a>(robot: &'a Robot, inertia: &'a [LinkInertia]) -> IlqrProblem<'a> {
        IlqrProblem {
            robot,
            inertia,
            gravity: Vector3::new(0.0, 0.0, -9.81),
            horizon: 60,
            dt: 0.02,
            q_goal: vec![0.6, -0.8],
            qd_goal: vec![0.0, 0.0],
            w_q: 1.0,
            w_qd: 0.01,
            r: 0.001,
            w_qf: 1000.0,
            w_qdf: 100.0,
        }
    }

    #[test]
    fn ilqr_swings_arm_to_goal() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let problem = swing_problem(&robot, &inertia);
        let res = solve_ilqr(&problem, &[0.0, 0.0], &[0.0, 0.0]);

        // Shapes.
        assert_eq!(res.taus.len(), 60);
        assert_eq!(res.qs.len(), 61);
        assert_eq!(res.qds.len(), 61);

        // Cost decreased overall and never increased across accepted iterations.
        assert!(res.costs.len() >= 2, "expected at least one improving iteration");
        for w in res.costs.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "cost increased: {:?}", res.costs);
        }
        assert!(
            *res.costs.last().unwrap() < res.costs[0],
            "cost did not decrease: {:?}",
            res.costs
        );

        assert!(res.converged, "iLQR did not converge (iters {})", res.iters);

        // Final planned state is at the goal, at rest.
        let qf = res.qs.last().unwrap();
        let qdf = res.qds.last().unwrap();
        let perr = ((qf[0] - 0.6).powi(2) + (qf[1] + 0.8).powi(2)).sqrt();
        assert!(perr < 0.1, "final joint error {perr}, q = {qf:?}");
        let verr = (qdf[0].powi(2) + qdf[1].powi(2)).sqrt();
        assert!(verr < 0.6, "final joint velocity {verr}, qd = {qdf:?}");
    }

    #[test]
    fn planned_torques_are_dynamically_consistent() {
        // Independently close the loop: apply the planned torques through forward_dynamics with the
        // same semi-implicit Euler the planner used, and confirm we reproduce its state trajectory.
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let problem = swing_problem(&robot, &inertia);
        let res = solve_ilqr(&problem, &[0.0, 0.0], &[0.0, 0.0]);

        let dt = 0.02;
        let (mut q, mut qd) = (vec![0.0, 0.0], vec![0.0, 0.0]);
        for t in 0..res.taus.len() {
            let tau = &res.taus[t];
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let qf = res.qs.last().unwrap();
        for i in 0..2 {
            assert!((q[i] - qf[i]).abs() < 1e-9, "rollout mismatch at joint {i}: {} vs {}", q[i], qf[i]);
        }
        // And that re-simulated state lands at the goal.
        let perr = ((q[0] - 0.6).powi(2) + (q[1] + 0.8).powi(2)).sqrt();
        assert!(perr < 0.1, "re-simulated joint error {perr}, q = {q:?}");
    }
}
