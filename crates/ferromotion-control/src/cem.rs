//! Cross-Entropy Method (CEM) sampling MPC — sibling to `mppi`, but a *distributional* planner.
//! Instead of a single softmax-weighted update, CEM maintains a diagonal Gaussian over every
//! control dimension × timestep (mean = nominal plan, per-element std). Each iteration it draws
//! `num_samples` control sequences from that Gaussian, rolls each out through `ferromotion-core`'s forward
//! dynamics + semi-implicit Euler, scores them by an accumulated running + terminal cost, keeps the
//! lowest-cost `elite_frac` fraction ("elites"), and refits the Gaussian's mean and std to those
//! elites. After `n_iters` refinements it returns the first control of the final mean (receding
//! horizon). Pure Rust + `nalgebra` → WASM-clean; all randomness is a seeded LCG + Box–Muller, so
//! runs are bit-for-bit reproducible.
//!
//! Elite retention: the previous iteration's elites are carried into the next iteration's candidate
//! pool. Because the state is fixed across a single `plan` call, a retained elite's rollout cost is
//! unchanged, so the new elite set is chosen from a superset that already contains the old elites —
//! making the **elite mean cost monotonically non-increasing across iterations** (the core CEM
//! refinement guarantee), not merely non-increasing in expectation.

use nalgebra::Vector3;
use ferromotion_core::{forward_dynamics, LinkInertia, Robot};

/// The full result of one receding-horizon CEM step.
#[derive(Clone, Debug, PartialEq)]
pub struct CemStep {
    /// First control of the final mean plan — the command to apply now.
    pub control: Vec<f64>,
    /// The final mean plan shifted forward one step (last command repeated); warm-starts the next
    /// `plan`/`control` call.
    pub nominal: Vec<Vec<f64>>,
    /// The final (un-shifted) mean plan, `horizon × dof`. `rollout_cost` this to score the plan.
    pub plan: Vec<Vec<f64>>,
    /// Elite mean cost after each CEM iteration (length `n_iters`); monotonically non-increasing.
    pub elite_costs: Vec<f64>,
}

/// Cross-Entropy Method sampling MPC regulating joint state toward `q_goal` (zero target velocity).
///
/// Cost per rollout (identical family to `Mppi`):
/// `Σₜ [w_q·‖qₜ − q_goal‖² + w_qd·‖q̇ₜ‖² + r_ctrl·‖τₜ‖²] + w_terminal·‖q_N − q_goal‖²`.
pub struct Cem<'a> {
    pub robot: &'a Robot,
    pub inertia: &'a [LinkInertia],
    pub gravity: Vector3<f64>,
    /// Planning horizon (number of control steps looked ahead).
    pub horizon: usize,
    /// Rollout timestep (s).
    pub dt: f64,
    /// Number of control sequences sampled per CEM iteration.
    pub num_samples: usize,
    /// Fraction of samples kept as elites each iteration (in `(0, 1]`).
    pub elite_frac: f64,
    /// Number of CEM refinement iterations per control step.
    pub n_iters: usize,
    /// Initial per-element std of the sampling Gaussian.
    pub sigma0: f64,
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
    /// Symmetric torque bound (per actuator) clamped onto samples and the mean.
    pub u_max: f64,
    /// LCG state; advanced by every draw so the controller is fully deterministic in its seed.
    pub rng: u64,
}

/// Lower floor on the refit std so the sampling Gaussian never fully collapses between iterations.
const MIN_STD: f64 = 1e-2;

impl<'a> Cem<'a> {
    /// Next uniform sample in `[0, 1)` from the LCG (Knuth/MMIX multiplier & increment).
    fn next_u01(&mut self) -> f64 {
        self.rng =
            self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

    /// Pad/truncate an arbitrary nominal to a `horizon × dof` plan, clamped to the torque bound.
    fn padded(&self, nominal: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.robot.dof();
        (0..self.horizon)
            .map(|t| {
                (0..n)
                    .map(|i| {
                        nominal
                            .get(t)
                            .and_then(|u| u.get(i))
                            .copied()
                            .unwrap_or(0.0)
                            .clamp(-self.u_max, self.u_max)
                    })
                    .collect()
            })
            .collect()
    }

    /// Accumulated running + terminal cost of a `horizon × dof` plan rolled out from `(q, qd)` via
    /// forward dynamics + semi-implicit Euler. Torques are clamped to `u_max`. Diverging rollouts
    /// (non-finite state) score `f64::MAX`.
    pub fn rollout_cost(&self, q: &[f64], qd: &[f64], plan: &[Vec<f64>]) -> f64 {
        self.rollout(q, qd, &self.padded(plan))
    }

    /// Cost of a plan already shaped `horizon × dof` and clamped (internal fast path).
    fn rollout(&self, q0: &[f64], qd0: &[f64], plan: &[Vec<f64>]) -> f64 {
        let n = self.robot.dof();
        let mut qs = q0.to_vec();
        let mut qds = qd0.to_vec();
        let mut cost = 0.0;

        for t in 0..self.horizon {
            let tau: Vec<f64> =
                (0..n).map(|i| plan[t][i].clamp(-self.u_max, self.u_max)).collect();
            let qdd = forward_dynamics(self.robot, self.inertia, &qs, &qds, &tau, self.gravity);
            for i in 0..n {
                qds[i] += qdd[i] * self.dt;
                qs[i] += qds[i] * self.dt;
            }
            if !qs.iter().chain(qds.iter()).all(|x| x.is_finite()) {
                return f64::MAX;
            }
            let mut c = self.r_ctrl * tau.iter().map(|u| u * u).sum::<f64>();
            for i in 0..n {
                let e = qs[i] - self.q_goal[i];
                c += self.w_q * e * e + self.w_qd * qds[i] * qds[i];
            }
            cost += c;
        }
        for i in 0..n {
            let e = qs[i] - self.q_goal[i];
            cost += self.w_terminal * e * e;
        }
        if cost.is_finite() {
            cost
        } else {
            f64::MAX
        }
    }

    /// One receding-horizon CEM step, returning the full [`CemStep`] (command, warm-start nominal,
    /// final mean plan, and the per-iteration elite-mean-cost trace).
    ///
    /// `nominal` seeds the initial mean (padded/clamped to `horizon × dof`; missing entries → 0);
    /// pass `vec![vec![0.0; dof]; horizon]` on the first call.
    pub fn plan(&mut self, q: &[f64], qd: &[f64], nominal: &[Vec<f64>]) -> CemStep {
        let n = self.robot.dof();
        let horizon = self.horizon;
        let k = self.num_samples;
        // Constant elite count across iterations (needed for the monotonicity guarantee).
        let n_elite = ((k as f64 * self.elite_frac).round() as usize).clamp(1, k);

        // Gaussian: mean initialised to the (clamped) nominal, std to sigma0 everywhere.
        let mut mean = self.padded(nominal);
        let mut std = vec![vec![self.sigma0.max(MIN_STD); n]; horizon];

        // Retained elites from the previous iteration: (cost, plan). Empty on iteration 0.
        let mut elites: Vec<(f64, Vec<Vec<f64>>)> = Vec::new();
        let mut elite_costs: Vec<f64> = Vec::with_capacity(self.n_iters);

        for _ in 0..self.n_iters {
            // Candidate pool = retained elites (unchanged cost, since state is fixed) + fresh draws.
            let mut pool: Vec<(f64, Vec<Vec<f64>>)> =
                Vec::with_capacity(elites.len() + k);
            pool.append(&mut elites);

            for _ in 0..k {
                let seq: Vec<Vec<f64>> = (0..horizon)
                    .map(|t| {
                        (0..n)
                            .map(|i| {
                                let u = mean[t][i] + std[t][i] * self.next_gauss();
                                u.clamp(-self.u_max, self.u_max)
                            })
                            .collect()
                    })
                    .collect();
                let cost = self.rollout(q, qd, &seq);
                pool.push((cost, seq));
            }

            // Keep the lowest-cost `n_elite`. Because the pool contains last iteration's `n_elite`
            // elites, the new elite total cost ≤ the old one ⇒ elite mean cost is non-increasing.
            pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
            pool.truncate(n_elite);

            let mean_cost = pool.iter().map(|(c, _)| *c).sum::<f64>() / (n_elite as f64);
            elite_costs.push(mean_cost);

            // Refit the Gaussian to the elites (per timestep × dimension).
            for t in 0..horizon {
                for i in 0..n {
                    let m = pool.iter().map(|(_, s)| s[t][i]).sum::<f64>() / (n_elite as f64);
                    let var = pool
                        .iter()
                        .map(|(_, s)| {
                            let d = s[t][i] - m;
                            d * d
                        })
                        .sum::<f64>()
                        / (n_elite as f64);
                    mean[t][i] = m;
                    std[t][i] = var.sqrt().max(MIN_STD);
                }
            }
            elites = pool;
        }

        let plan = mean; // final refined mean plan
        let control = plan.first().cloned().unwrap_or_else(|| vec![0.0; n]);
        // Receding horizon: shift forward one step, repeating the last command.
        let nominal_out: Vec<Vec<f64>> = (0..horizon)
            .map(|t| plan[(t + 1).min(horizon.saturating_sub(1))].clone())
            .collect();

        CemStep { control, nominal: nominal_out, plan, elite_costs }
    }

    /// Convenience wrapper mirroring [`Mppi::control`](crate::Mppi): the command to apply now and
    /// the shifted warm-start nominal.
    pub fn control(
        &mut self,
        q: &[f64],
        qd: &[f64],
        nominal: &[Vec<f64>],
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let step = self.plan(q, qd, nominal);
        (step.control, step.nominal)
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

    fn make_cem<'a>(robot: &'a Robot, inertia: &'a [LinkInertia], seed: u64) -> Cem<'a> {
        Cem {
            robot,
            inertia,
            gravity: Vector3::new(0.0, 0.0, -9.81),
            horizon: 15,
            dt: 0.02,
            num_samples: 150,
            elite_frac: 0.1,
            n_iters: 5,
            sigma0: 4.0,
            q_goal: vec![0.6, -0.8],
            w_q: 40.0,
            w_qd: 2.0,
            w_terminal: 150.0,
            r_ctrl: 0.01,
            u_max: 12.0,
            rng: seed,
        }
    }

    #[test]
    fn elite_mean_cost_is_monotonically_non_increasing() {
        // The core CEM refinement guarantee: with elite retention, the elite mean cost cannot go up
        // from one iteration to the next (the previous elites remain in the candidate pool).
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let mut cem = make_cem(&robot, &inertia, 0xC0FFEE);
        let q = [0.0, 0.0];
        let qd = [0.0, 0.0];
        let nominal = vec![vec![0.0; 2]; cem.horizon];
        let step = cem.plan(&q, &qd, &nominal);

        assert_eq!(step.elite_costs.len(), cem.n_iters);
        for w in step.elite_costs.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-9,
                "elite mean cost increased: {:?}",
                step.elite_costs
            );
        }
        // Refinement actually did something over the run.
        assert!(
            *step.elite_costs.last().unwrap() < step.elite_costs[0],
            "elite cost never improved: {:?}",
            step.elite_costs
        );
    }

    #[test]
    fn refined_plan_beats_the_zero_plan_when_motion_helps() {
        // Start away from the goal so staying put (τ ≡ 0) is costly and moving is rewarded. The
        // final refined mean plan must roll out cheaper than the do-nothing plan.
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let mut cem = make_cem(&robot, &inertia, 0x1234_5678);
        let q = [0.0, 0.0];
        let qd = [0.0, 0.0];
        let zero_plan = vec![vec![0.0; 2]; cem.horizon];

        let zero_cost = cem.rollout_cost(&q, &qd, &zero_plan);
        let step = cem.plan(&q, &qd, &zero_plan);
        let plan_cost = cem.rollout_cost(&q, &qd, &step.plan);

        assert!(
            plan_cost < zero_cost,
            "refined plan ({plan_cost}) not better than zero plan ({zero_cost})"
        );
        // Sanity: gravity acts along the joint (z) axes for this in-plane arm, so the zero plan just
        // sits at the start — its cost is dominated by the standing goal error.
        assert!(zero_cost.is_finite() && zero_cost > 0.0);
    }

    #[test]
    fn is_deterministic_in_its_seed() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let q = [0.1, -0.2];
        let qd = [0.0, 0.0];
        let nominal = vec![vec![0.0; 2]; 15];

        let mut a = make_cem(&robot, &inertia, 42);
        let mut b = make_cem(&robot, &inertia, 42);
        let sa = a.plan(&q, &qd, &nominal);
        let sb = b.plan(&q, &qd, &nominal);
        assert_eq!(sa, sb, "same-seed CEM steps diverged");

        // Different seed ⇒ (generally) a different plan.
        let mut c = make_cem(&robot, &inertia, 9001);
        let sc = c.plan(&q, &qd, &nominal);
        assert_ne!(sa.control, sc.control, "different seeds gave identical control");
    }

    #[test]
    fn closed_loop_reduces_goal_error() {
        // Honest closed-loop property (NOT tight settling): running the receding-horizon controller
        // for a while gets the arm closer to the goal than it started.
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut cem = make_cem(&robot, &inertia, 0xABCDEF);
        let goal = cem.q_goal.clone();

        let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 0.02);
        let start_err =
            ((q[0] - goal[0]).powi(2) + (q[1] - goal[1]).powi(2)).sqrt();
        let mut nominal = vec![vec![0.0; 2]; cem.horizon];
        for _ in 0..40 {
            let (u, nom) = cem.control(&q, &qd, &nominal);
            nominal = nom;
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &u, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let end_err = ((q[0] - goal[0]).powi(2) + (q[1] - goal[1]).powi(2)).sqrt();
        assert!(end_err < start_err, "goal error did not decrease: {start_err} -> {end_err}");
    }
}
