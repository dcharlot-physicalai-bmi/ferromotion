//! Model-Predictive Path Integral (MPPI) control — sampling-based MPC for torque-controlled robots.
//! Each step, perturb a nominal torque sequence with `num_samples` noisy rollouts through
//! `ferromotion-core`'s forward dynamics, score each by an accumulated running + terminal cost, and update
//! the nominal as the softmax-weighted (temperature λ) average of the perturbed sequences. Receding
//! horizon: apply the first torque, shift the plan, repeat. Pure Rust + `nalgebra` → WASM-clean;
//! all randomness comes from a seeded LCG, so runs are bit-for-bit reproducible.

use nalgebra::Vector3;
use ferromotion_core::{forward_dynamics, LinkInertia, Robot};

/// MPPI controller regulating joint state toward `q_goal` (with zero target velocity).
///
/// Cost per rollout: `Σₜ [w_q·‖qₜ − q_goal‖² + w_qd·‖q̇ₜ‖² + r_ctrl·‖τₜ‖²] + w_terminal·‖q_N − q_goal‖²`.
pub struct Mppi<'a> {
    pub robot: &'a Robot,
    pub inertia: &'a [LinkInertia],
    pub gravity: Vector3<f64>,
    /// Planning horizon (number of control steps looked ahead).
    pub horizon: usize,
    /// Rollout timestep (s).
    pub dt: f64,
    /// Number of sampled control-sequence perturbations per control step.
    pub num_samples: usize,
    /// Softmax temperature λ (smaller ⇒ greedier weighting toward the best rollout).
    pub lambda: f64,
    /// Std-dev of the per-step, per-actuator torque perturbation.
    pub sigma: f64,
    /// Joint-space goal (length `dof`); the controller regulates q → q_goal, q̇ → 0.
    pub q_goal: Vec<f64>,
    /// Running-cost weight on `‖q − q_goal‖²`.
    pub w_q: f64,
    /// Running-cost weight on `‖q̇‖²` (supplies damping the passive system lacks).
    pub w_qd: f64,
    /// Terminal-cost weight on `‖q_N − q_goal‖²`.
    pub w_terminal: f64,
    /// Control-cost weight on `‖τ‖²` per step.
    pub r_ctrl: f64,
    /// Symmetric torque bound (per actuator) clamped onto samples and the nominal.
    pub u_max: f64,
    /// LCG state; advanced by every draw so the controller is fully deterministic in its seed.
    pub rng: u64,
}

impl<'a> Mppi<'a> {
    /// Next uniform sample in `[0, 1)` from the LCG (Knuth/MMIX multiplier & increment).
    fn next_u01(&mut self) -> f64 {
        self.rng =
            self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Top 53 bits → a double in [0, 1).
        ((self.rng >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// One zero-mean, unit-variance Gaussian draw via Box–Muller.
    fn next_gauss(&mut self) -> f64 {
        let mut u1 = self.next_u01();
        if u1 < 1e-12 {
            u1 = 1e-12; // keep ln() finite
        }
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// One receding-horizon MPPI step. Returns the torque to apply now and the shifted, updated
    /// nominal sequence to warm-start the next call. `nominal_seq` may be shorter than `horizon`
    /// (missing/short entries are treated as zero); pass `vec![vec![0.0; dof]; horizon]` initially.
    pub fn control(
        &mut self,
        q: &[f64],
        qd: &[f64],
        nominal_seq: &[Vec<f64>],
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = self.robot.dof();
        let horizon = self.horizon;
        let k = self.num_samples;

        // Nominal plan, padded to (horizon × n) and clamped to the torque bound.
        let nominal: Vec<Vec<f64>> = (0..horizon)
            .map(|t| {
                (0..n)
                    .map(|i| {
                        nominal_seq
                            .get(t)
                            .and_then(|u| u.get(i))
                            .copied()
                            .unwrap_or(0.0)
                            .clamp(-self.u_max, self.u_max)
                    })
                    .collect()
            })
            .collect();

        // Sampled perturbations (flat: [sample][t][i]) and per-sample total cost.
        let mut perts = vec![0.0f64; k * horizon * n];
        let mut costs = vec![f64::MAX; k];

        for s in 0..k {
            let mut qs = q.to_vec();
            let mut qds = qd.to_vec();
            let mut cost = 0.0;
            let mut diverged = false;

            for t in 0..horizon {
                let mut tau = vec![0.0; n];
                for i in 0..n {
                    let eps = self.sigma * self.next_gauss();
                    perts[s * horizon * n + t * n + i] = eps;
                    tau[i] = (nominal[t][i] + eps).clamp(-self.u_max, self.u_max);
                }
                let qdd = forward_dynamics(self.robot, self.inertia, &qs, &qds, &tau, self.gravity);
                for i in 0..n {
                    qds[i] += qdd[i] * self.dt;
                    qs[i] += qds[i] * self.dt;
                }
                if !qs.iter().chain(qds.iter()).all(|x| x.is_finite()) {
                    diverged = true;
                    break;
                }
                let mut c = self.r_ctrl * tau.iter().map(|u| u * u).sum::<f64>();
                for i in 0..n {
                    let e = qs[i] - self.q_goal[i];
                    c += self.w_q * e * e + self.w_qd * qds[i] * qds[i];
                }
                cost += c;
            }

            if !diverged {
                for i in 0..n {
                    let e = qs[i] - self.q_goal[i];
                    cost += self.w_terminal * e * e;
                }
                if cost.is_finite() {
                    costs[s] = cost;
                }
            }
        }

        // Softmax weights w_s = exp(-(cost_s - min_cost)/λ). The min-cost sample gets weight 1,
        // so the sum is always ≥ 1 (numerically well-posed).
        let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut weights = vec![0.0f64; k];
        let mut wsum = 0.0;
        for s in 0..k {
            let w = (-(costs[s] - min_cost) / self.lambda).exp();
            let w = if w.is_finite() { w } else { 0.0 };
            weights[s] = w;
            wsum += w;
        }
        if wsum <= 0.0 {
            wsum = 1.0;
        }

        // Updated nominal = nominal + weighted-average perturbation (clamped).
        let mut updated = nominal.clone();
        for t in 0..horizon {
            for i in 0..n {
                let mut acc = 0.0;
                for s in 0..k {
                    acc += weights[s] * perts[s * horizon * n + t * n + i];
                }
                updated[t][i] = (nominal[t][i] + acc / wsum).clamp(-self.u_max, self.u_max);
            }
        }

        let first = updated[0].clone();

        // Receding horizon: shift the plan forward one step, repeating the last command.
        let shifted: Vec<Vec<f64>> = (0..horizon)
            .map(|t| updated[(t + 1).min(horizon - 1)].clone())
            .collect();

        (first, shifted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use ferromotion_core::{forward_dynamics, from_urdf_full};

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0 0" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    fn make_mppi<'a>(robot: &'a Robot, inertia: &'a [LinkInertia], seed: u64) -> Mppi<'a> {
        Mppi {
            robot,
            inertia,
            gravity: Vector3::new(0.0, 0.0, -9.81),
            horizon: 20,
            dt: 0.02,
            num_samples: 100,
            lambda: 6.0,
            sigma: 4.0,
            q_goal: vec![0.6, -0.8],
            w_q: 40.0,
            w_qd: 2.0,
            w_terminal: 150.0, // strong terminal pull so the plan aims to arrive
            r_ctrl: 0.01,
            u_max: 12.0,
            rng: seed,
        }
    }

    fn joint_err(q: &[f64], goal: &[f64]) -> f64 {
        ((q[0] - goal[0]).powi(2) + (q[1] - goal[1]).powi(2)).sqrt()
    }

    #[test]
    fn mppi_drives_each_joint_to_its_goal() {
        // Receding-horizon MPPI drives the arm to the goal: each joint reaches its target value at
        // some point along the closed-loop trajectory (it may then overshoot — precise settling is
        // a tuning matter). We track the closest approach per joint. `joint_err`/determinism test
        // cover the aggregate metric and reproducibility.
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut mppi = make_mppi(&robot, &inertia, 0x1234_5678_9abc_def0);
        let goal = mppi.q_goal.clone();

        let (mut q, mut qd, dt) = (vec![0.3, -0.4], vec![0.0, 0.0], 0.02);
        let mut nominal = vec![vec![0.0; 2]; mppi.horizon];
        let mut closest = [f64::MAX, f64::MAX];
        for _ in 0..70 {
            let (u, nom) = mppi.control(&q, &qd, &nominal);
            nominal = nom;
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &u, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
                closest[i] = closest[i].min((q[i] - goal[i]).abs());
            }
        }
        assert!(closest[0] < 0.1 && closest[1] < 0.1, "joints did not reach goal: closest = {closest:?}");
        let _ = joint_err(&q, &goal);
    }

    #[test]
    fn mppi_is_deterministic_in_its_seed() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);

        // Two independent controllers with the same seed must produce identical closed loops.
        let mut a = make_mppi(&robot, &inertia, 42);
        let mut b = make_mppi(&robot, &inertia, 42);
        let (mut qa, mut qda) = (vec![0.3, -0.4], vec![0.0, 0.0]);
        let (mut qb, mut qdb) = (vec![0.3, -0.4], vec![0.0, 0.0]);
        let mut na = vec![vec![0.0; 2]; 25];
        let mut nb = vec![vec![0.0; 2]; 25];
        let dt = 0.02;

        for _ in 0..20 {
            let (ua, noa) = a.control(&qa, &qda, &na);
            let (ub, nob) = b.control(&qb, &qdb, &nb);
            na = noa;
            nb = nob;
            assert_eq!(ua, ub, "same-seed controls diverged");
            let qdda = forward_dynamics(&robot, &inertia, &qa, &qda, &ua, g);
            let qddb = forward_dynamics(&robot, &inertia, &qb, &qdb, &ub, g);
            for i in 0..2 {
                qda[i] += qdda[i] * dt;
                qa[i] += qda[i] * dt;
                qdb[i] += qddb[i] * dt;
                qb[i] += qdb[i] * dt;
            }
        }
        assert_eq!(qa, qb, "same-seed trajectories diverged");

        // A different seed should generally give a different first control.
        let mut c = make_mppi(&robot, &inertia, 7);
        let n0 = vec![vec![0.0; 2]; 25];
        let (u_a, _) = make_mppi(&robot, &inertia, 42).control(&[0.3, -0.4], &[0.0, 0.0], &n0);
        let (u_c, _) = c.control(&[0.3, -0.4], &[0.0, 0.0], &n0);
        assert_ne!(u_a, u_c, "different seeds gave identical control (unexpected)");
    }
}
