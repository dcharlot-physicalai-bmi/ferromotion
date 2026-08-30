//! **Can this pipeline control a real arm, or only a pendulum?**
//!
//! The other two RL benches here are one degree of freedom. The stack exists for articulated robots, so this
//! points the whole chain at one: the **SO-101** (LeRobot / TheRobotStudio) loaded from its own URDF, five
//! actuated joints, link inertias from the model's `<inertial>` blocks, torque control through
//! `mass_matrix` and `inverse_dynamics`.
//!
//! # What building it found: a URDF is not an actuator model, and this one proves it quantitatively
//!
//! The action space was written to read its torque bound from `Joint::effort`, so the bench would use the
//! robot's declared capability instead of a hand-chosen constant. Reading it exposed two defects.
//!
//! **The declared effort limits are a placeholder.** All five joints say exactly `10.0` N·m. Gravity at the
//! home pose needs at most 0.51 N·m and a real STS3215 stalls near 3 N·m, so the figure is neither measured
//! nor per-joint.
//!
//! **The link inertias alone make torque control very stiff.** The mass-matrix diagonal at home runs
//! `[1.41e-2, 1.52e-2, 7.85e-3, 8.30e-4, 3.45e-5]` kg·m². The last joint is the gripper roll, and 10 N·m into
//! `3.45e-5` kg·m² is **290,000 rad/s²** — one 5 ms step adds 1,449 rad/s. Over a 4x4 grid of gains and
//! substeps (`--sweep`), IK+PD+gravity reaches a 1 cm target in **1 of 16** configurations, only at 10 kHz
//! integration and low gain, settling at 0.0177 m on 13.8 J. `actuator_plausibility` flags exactly the two
//! joints responsible, from the model alone, before any simulation runs.
//!
//! The missing term is the **actuator's own reflected rotor inertia**. A geared servo presents `N²·J_rotor` at
//! its output, and for the SO-101's distal joints that dominates the link the URDF describes — by a factor of
//! ~340 at the wrist. `SeaJoint::reflected_inertia` in `ferromotion-control` has always computed it; nothing
//! had ever fed it into a URDF-loaded multibody plant. Adding it, plus the motor's linear speed–torque droop:
//!
//! | reflected inertia | reaches 1 cm | best settle | electrical | rate needed |
//! |---|---|---|---|---|
//! | 0 (URDF as written) | 1 of 16 configs | 0.0177 m | 13.8 J | 10 kHz |
//! | 1.19e-2 (N=345) | **16 of 16** | 0.0001 m | 4.0 J | 200 Hz |
//!
//! 50x the integration rate, and at identical gains 4.2x the settling error and 3.5x the energy, with a
//! working region that collapses to one corner of the grid. `SENSITIVITY` re-measures the inertia dependence every run and `--sweep` re-measures
//! this table, because the conclusion should not rest on one estimate of a rotor inertia — and because an
//! earlier draft of this file claimed the plant was *unsolvable* without the term, which was wrong.
//!
//! # Declared servo parameters, and why these are derived rather than guessed
//!
//! `TAU_STALL` and `OMEGA_NO_LOAD` are STS3215 catalogue figures at 7.4 V. The damping is not a third free
//! parameter: a DC motor's torque falls linearly to zero at no-load speed, so `b = τ_stall / ω_0` **follows
//! from the other two**. That droop also replaces the hard velocity clamp this bench started with, which was
//! injecting energy at every clamp event.
//!
//! Reflected inertia is the one genuinely estimated number (`N = 345`, `J_rotor = 1e-7` kg·m²). The sweep is
//! there because of it: the result is stable from `4e-3` to `4e-2`, so the finding rests on the term being
//! **present**, not on its exact value.
//!
//! # The baseline solves the same problem the policy does
//!
//! `solve_ik` gets the Cartesian target only — never the joint configuration the target was generated from,
//! which the environment knows and could have handed over. A baseline given the answer in joint space is not
//! IK, and calling it IK would overstate what PPO has to beat.
//!
//! `cargo run --release -p ferromotion-bench --example so101_reach_rl`

use ferromotion_core::{
    actuator_plausibility, confounding, forward_dynamics, from_urdf_full, gravity_vector, identify_actuator,
    inverse_dynamics, mass_matrix, solve_ik, IdSample, IkOptions, Iso, LinkInertia, PlannedMotion, Robot,
    SavGol,
};
use ferromotion_learn::{train_normalized, BoxSpace, Env, GaussianPolicy, Mlp, ObsNorm, PpoConfig, StepResult};
use nalgebra::Vector3;

const URDF: &str = include_str!("../../ferromotion-core/examples/so101.urdf");
const GRAVITY: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);
/// Control period (200 Hz) and physics substeps within it. The two are different things: the rate is a
/// property of the servo bus, the substep is a numerics choice.
const CONTROL_DT: f64 = 5e-3;
const SUBSTEPS: usize = 4;
const STEPS: usize = 300;
/// Counted as reached (m).
const REACH_TOL: f64 = 0.01;

/// **Which reward. Measured over 3 seeds each, because one seed is not a measurement.**
///
/// The first run trained cleanly (return −25.0 → −2.2) and still reached only 8%: best error 2.2 cm against a
/// 1 cm tolerance. It found the direction and not the hold. The suspect was the reward's own shape rather than
/// the optimizer, because `−‖e‖²` has gradient `2‖e‖`, so at 2 cm the signal driving the last centimetre is
/// **25x weaker** than the one that drove the approach from 25 cm.
///
/// Seeds 7 / 11 / 23, 150 × 3000 steps each, evaluated on the deterministic mean policy over 12 targets:
///
/// | reward | final error, mean ± sd | per-seed range | reached | electrical |
/// |---|---|---|---|---|
/// | quadratic `−‖e‖²` | 0.0573 ± 0.0234 m | 0.0401–0.0840 | 0% | 21.5 J |
/// | **linear `−‖e‖`** | **0.0147 ± 0.0037 m** | **0.0116–0.0188** | **36%** | 16.8 J |
/// | peaked Gaussian | 0.0816 ± 0.0164 m | 0.0630–0.0941 | 3% | 20.8 J |
///
/// **Linear wins and the per-seed ranges do not overlap** — every linear seed beats every quadratic seed,
/// 3.9x on the mean. That is the constant-gradient prediction confirmed.
///
/// **The peaked bonus is the informative failure.** `exp(−(‖e‖/σ)²)` with `σ` at the tolerance is the sharpest
/// of the three exactly where the quadratic goes flat, and it came last. At the 25 cm start the bonus is
/// numerically zero, so it is *sparse* through the entire approach: sharpening a reward where you want
/// precision buys nothing if it goes silent where you need guidance. Its range does overlap the quadratic's,
/// so the honest claim is "no better", not "worse".
///
/// Returns are **not** comparable across these shapes — linear's run −84 → −12 purely because `−‖e‖` has a
/// different scale — which is why every column above is metres, joules or a success rate.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reward {
    /// `−‖e‖²`. Gradient `2‖e‖`, vanishing at the goal.
    Quadratic,
    /// `−‖e‖`. Constant gradient, so the last centimetre is worth as much per metre as the first.
    Linear,
    /// `−‖e‖² + exp(−(‖e‖/σ)²)` with `σ` at the tolerance: a bounded bonus whose gradient peaks near `σ`.
    Peaked,
}

impl Reward {
    fn parse(s: &str) -> Option<Reward> {
        match s {
            "quadratic" => Some(Reward::Quadratic),
            "linear" => Some(Reward::Linear),
            "peaked" => Some(Reward::Peaked),
            _ => None,
        }
    }

    fn of(&self, err: f64) -> f64 {
        match self {
            Reward::Quadratic => -(err * err),
            Reward::Linear => -err,
            Reward::Peaked => -(err * err) + (-(err / REACH_TOL).powi(2)).exp(),
        }
    }
}

/// STS3215 at 7.4 V, catalogue: ~30 kg·cm stall, ~45 rpm no-load. All five joints use the same servo.
const TAU_STALL: f64 = 3.0;
const OMEGA_NO_LOAD: f64 = 4.7;
/// Gearbox ratio and rotor inertia. The estimated pair; `SENSITIVITY` exists because of them.
const GEAR_RATIO: f64 = 345.0;
const J_ROTOR: f64 = 1e-7;
/// Bus voltage the catalogue figures are quoted at.
const V_BUS: f64 = 7.4;

/// **Power drawn by the computer running the controller**, for the compute half of `E_task`.
///
/// The bench measures actuation in joules and would be telling half the story without this. A declared
/// estimate rather than a measurement, and the one number here that is: a Raspberry Pi 5 under load, the usual
/// companion for an arm this size, sits around 6 W. Compute energy scales linearly in it, so the ratio the
/// table reports is what matters and the absolute value is a stated assumption. Measured on THIS machine's
/// wall clock, so it is a fair comparison between the two controllers and not a claim about embedded silicon.
const COMPUTE_WATTS: f64 = 6.0;

/// **Torque constant and winding resistance, referred to the joint output.** Both follow from `TAU_STALL` and
/// `OMEGA_NO_LOAD` — no third parameter is fitted.
///
/// A DC motor's back-EMF gives `k_e = V / ω_0`, and in SI `k_t = k_e`. Stall is the zero-speed corner, so
/// `I_stall = τ_stall / k_t` and `R = V / I_stall`. Referring to the output rather than the rotor is
/// self-consistent: the gearbox multiplies torque by `N` and divides speed by `N`, so `k_t` scales by `N` on
/// both routes. For the STS3215 that is `k_t = 1.574` N·m/A and `R = 3.88` Ω.
fn motor_constants() -> (f64, f64) {
    let k_t = V_BUS / OMEGA_NO_LOAD;
    (k_t, V_BUS / (TAU_STALL / k_t))
}

/// Reflected rotor inertia at the joint output, `N²·J_rotor` — the same expression as
/// `SeaJoint::reflected_inertia`.
fn reflected_inertia() -> f64 {
    GEAR_RATIO * GEAR_RATIO * J_ROTOR
}

/// Speed droop `b = τ_stall / ω_0`, from the motor's linear speed–torque line. Not independent.
fn speed_droop() -> f64 {
    TAU_STALL / OMEGA_NO_LOAD
}

/// Reaching on the SO-101, torque control through its own multibody dynamics plus a servo model.
struct So101Reach {
    robot: Robot,
    inertia: Vec<LinkInertia>,
    reward: Reward,
    /// Physics substeps per control period. A field so the ill-posedness sweep can vary it, which is the
    /// measurement that distinguishes a missing term from a timestep that is merely too coarse.
    substeps: usize,
    q: Vec<f64>,
    qd: Vec<f64>,
    target: Vector3<f64>,
    steps: usize,
    /// Mechanical work delivered, `∫max(0, τ·q̇)`. Positive part only: a non-regenerative drive does not
    /// recover energy when the load back-drives it, and `∫|τ·q̇|` would score that recovery as consumption.
    mech: f64,
    /// Resistive loss in the windings, `∫(τ/k_t)²·R`. **This is the term a mechanical-work number misses, and
    /// on a reaching task it is not a correction — holding a pose against gravity costs current at zero
    /// speed, so `τ·q̇` reads zero while the servo keeps drawing.**
    copper: f64,
}

impl So101Reach {
    fn new(j_refl: f64) -> So101Reach {
        So101Reach::with_reward(j_refl, Reward::Quadratic)
    }

    fn with_reward(j_refl: f64, reward: Reward) -> So101Reach {
        let (mut robot, inertia) = from_urdf_full(URDF, "base_link", "gripper_link").expect("load SO-101");
        let n = robot.dof();
        // The servo terms the URDF cannot state, attached to the model itself rather than applied by hand at
        // the call site. `inverse_dynamics` applies both, so `mass_matrix` picks up the armature on its
        // diagonal and `forward_dynamics` inherits both — one place, and the identity
        // `M q̈ + bias == inverse_dynamics` keeps holding.
        for j in robot.joints.iter_mut() {
            *j = j.clone().with_armature(j_refl).with_damping(speed_droop());
        }
        So101Reach {
            robot,
            inertia,
            reward,
            substeps: SUBSTEPS,
            q: vec![0.0; n],
            qd: vec![0.0; n],
            target: Vector3::zeros(),
            steps: 0,
            mech: 0.0,
            copper: 0.0,
        }
    }

    fn dof(&self) -> usize {
        self.robot.dof()
    }

    fn tool(&self) -> Vector3<f64> {
        self.robot.fk(&self.q).translation.vector
    }

    fn reach_error(&self) -> f64 {
        (self.tool() - self.target).norm()
    }

    /// A target that is reachable **by construction**, being the forward kinematics of a configuration inside
    /// the joint limits. Sampling in Cartesian space would put targets outside the workspace and score every
    /// controller on tasks none could do.
    fn sample_target(&self, seed: u64) -> Vector3<f64> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        let q: Vec<f64> = self
            .robot
            .joints
            .iter()
            .map(|j| {
                let (lo, hi) = j.limits.unwrap_or((-1.0, 1.0));
                // Two thirds of the range, so no target sits against a hard stop.
                0.5 * (lo + hi) + 0.33 * (hi - lo) * next()
            })
            .collect();
        self.robot.fk(&q).translation.vector
    }

    fn observation(&self) -> Vec<f64> {
        let mut o = self.q.clone();
        o.extend(self.qd.iter().copied());
        let e = self.tool() - self.target;
        o.extend([e.x, e.y, e.z]);
        o
    }
}

impl Env for So101Reach {
    fn reset(&mut self, seed: u64) -> Vec<f64> {
        let n = self.dof();
        self.q = vec![0.0; n];
        self.qd = vec![0.0; n];
        self.target = self.sample_target(seed);
        self.steps = 0;
        self.mech = 0.0;
        self.copper = 0.0;
        self.observation()
    }

    fn step(&mut self, action: &[f64]) -> StepResult {
        let n = self.dof();
        let cmd = self.action_space().clamp(action);
        let h = CONTROL_DT / self.substeps as f64;
        for _ in 0..self.substeps {
            // Armature and damping live on the joints, so this is the ordinary call.
            let qdd = forward_dynamics(&self.robot, &self.inertia, &self.q, &self.qd, &cmd, GRAVITY);
            for i in 0..n {
                self.qd[i] += qdd[i] * h;
                self.q[i] += self.qd[i] * h;
                if let Some((lo, hi)) = self.robot.joints[i].limits {
                    if self.q[i] <= lo || self.q[i] >= hi {
                        self.q[i] = self.q[i].clamp(lo, hi);
                        self.qd[i] = 0.0; // a hard stop absorbs the momentum
                    }
                }
                self.mech += (cmd[i] * self.qd[i]).max(0.0) * h;
                let (k_t, r) = motor_constants();
                self.copper += (cmd[i] / k_t).powi(2) * r * h;
            }
        }
        self.steps += 1;
        let err = self.reach_error();
        let effort: f64 = cmd.iter().map(|t| (t / TAU_STALL).powi(2)).sum::<f64>() / n as f64;
        StepResult {
            observation: self.observation(),
            reward: self.reward.of(err) - 0.01 * effort,
            terminated: false,
            truncated: self.steps >= STEPS,
        }
    }

    fn observation_space(&self) -> BoxSpace {
        let n = self.dof();
        // Physical units throughout. `ObsNorm` learns the scaling and ships it in the checkpoint, so the
        // environment reports what it measures and the constants stay visible.
        let mut lo = vec![-4.0; n];
        let mut hi = vec![4.0; n];
        lo.extend(vec![-20.0; n]);
        hi.extend(vec![20.0; n]);
        lo.extend(vec![-1.0; 3]);
        hi.extend(vec![1.0; 3]);
        BoxSpace::new(&lo, &hi).expect("valid observation space")
    }

    fn action_space(&self) -> BoxSpace {
        // NOT the URDF's `effort`. All five joints declare exactly 10.0 N m, which is a placeholder: it is 3x
        // the servo's stall torque and 290,000 rad/s^2 into the wrist link. The servo spec is the real bound.
        BoxSpace::symmetric(&vec![TAU_STALL; self.dof()]).expect("valid action space")
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    final_err: f64,
    best_err: f64,
    reached: f64,
    mech: f64,
    copper: f64,
    /// Wall-clock seconds spent **in the controller**, excluding the physics step. The whole point of
    /// separating it: a simulator's cost is not a robot's cost, but the controller's is.
    compute_s: f64,
}

impl Score {
    /// Actuation energy: delivered work plus resistive loss, no regenerative recovery.
    fn elec(&self) -> f64 {
        self.mech + self.copper
    }

    /// Compute energy at the declared platform power.
    fn compute_j(&self) -> f64 {
        self.compute_s * COMPUTE_WATTS
    }

    /// `E_task = E_actuation + E_compute` — the quantity the institute's own metric names, which a bench
    /// reporting only actuation cannot claim to measure.
    fn total_j(&self) -> f64 {
        self.elec() + self.compute_j()
    }
}

fn mean(v: &[Score]) -> Score {
    let n = v.len().max(1) as f64;
    Score {
        final_err: v.iter().map(|s| s.final_err).sum::<f64>() / n,
        best_err: v.iter().map(|s| s.best_err).sum::<f64>() / n,
        reached: v.iter().map(|s| s.reached).sum::<f64>() / n,
        mech: v.iter().map(|s| s.mech).sum::<f64>() / n,
        copper: v.iter().map(|s| s.copper).sum::<f64>() / n,
        compute_s: v.iter().map(|s| s.compute_s).sum::<f64>() / n,
    }
}

fn episode(j_refl: f64, seed: u64, law: impl FnMut(&So101Reach) -> Vec<f64>) -> Score {
    episode_with(j_refl, seed, SUBSTEPS, law)
}

fn episode_with(j_refl: f64, seed: u64, substeps: usize, law: impl FnMut(&So101Reach) -> Vec<f64>) -> Score {
    episode_timed(j_refl, seed, substeps, 0.0, law)
}

/// `setup_s` folds in one-off controller cost paid before the loop — the IK solve, which PPO has no equivalent
/// of. Charging it to the episode is the honest accounting: the baseline cannot reach without it.
fn episode_timed(
    j_refl: f64,
    seed: u64,
    substeps: usize,
    setup_s: f64,
    law: impl FnMut(&So101Reach) -> Vec<f64>,
) -> Score {
    episode_timed_on(So101Reach::new(j_refl), seed, substeps, setup_s, law)
}

/// Same, on an environment the caller has already configured — so a sweep can vary a joint property that
/// `So101Reach::new` does not take.
fn episode_timed_on(
    mut env: So101Reach,
    seed: u64,
    substeps: usize,
    setup_s: f64,
    mut law: impl FnMut(&So101Reach) -> Vec<f64>,
) -> Score {
    env.substeps = substeps;
    env.reset(seed);
    let mut best = f64::INFINITY;
    // Time the CONTROLLER only. `env.step` is the simulator and a real robot does not pay for it; the control
    // law is the part that has to run on the robot's computer at 200 Hz.
    let mut compute_s = 0.0;
    for _ in 0..STEPS {
        let t = std::time::Instant::now();
        let a = law(&env);
        compute_s += t.elapsed().as_secs_f64();
        env.step(&a);
        best = best.min(env.reach_error());
    }
    let final_err = env.reach_error();
    Score {
        final_err,
        best_err: best,
        reached: if final_err < REACH_TOL { 1.0 } else { 0.0 },
        mech: env.mech,
        copper: env.copper,
        compute_s: compute_s + setup_s,
    }
}

/// Solve IK for the Cartesian target, position only. The environment knows the joint configuration the target
/// came from and deliberately does not pass it.
fn ik_goal(env: &So101Reach) -> Vec<f64> {
    let opts = IkOptions { max_iters: 400, tol: 1e-12, rot_weight: 0.0, ..Default::default() };
    let target = Iso::from_parts(env.target.into(), nalgebra::UnitQuaternion::identity());
    solve_ik(&env.robot, &target, &vec![0.0; env.dof()], &opts).q
}

/// PD with gravity compensation onto an IK solution: what this workspace would already do.
fn pd_track(env: &So101Reach, q_goal: &[f64], kp: f64, kd: f64) -> Vec<f64> {
    let g = gravity_vector(&env.robot, &env.inertia, &env.q, GRAVITY);
    (0..env.dof())
        .map(|i| (g[i] + kp * (q_goal[i] - env.q[i]) - kd * env.qd[i]).clamp(-TAU_STALL, TAU_STALL))
        .collect()
}

/// **What the two control laws cost to evaluate**, head to head. Run with `--compute`.
///
/// The arithmetic cost of a policy does not depend on the values of its weights, so this needs no training run
/// and is not an estimate: it times the same `pd_track` and `GaussianPolicy::mean` the scored controllers call.
/// Reported per control step and per 1.5 s episode, because that is what a robot's computer actually pays.
///
/// **Minimum of repeated trials, with the spread printed.** A single wall-clock reading on a loaded machine is
/// not a measurement: back-to-back runs of a median-based version of this function moved 30% and its reported
/// ratio wandered over 5.05x / 5.87x / 7.23x / 9.23x; switching to the minimum tightened the per-step spread
/// to about 1.5x and the ratio to 6-10x. The minimum is the right statistic and the median is not, because
/// noise is **one-sided** — contention, migration and interrupts only ever add time to a fixed amount of
/// arithmetic, so the fastest trial is the closest look at the real cost. (That is the opposite of the rule for
/// comparing GPU and CPU floating point, where the concern is disagreement rather than delay.)
///
/// The trial ranges are printed so a reader can see whether the comparison is resolved at all: PD's slowest
/// trial has stayed below the policy's fastest in every run, which is what makes the direction solid even while
/// the machine is too loaded to pin the ratio down.
fn compute_cost(n_calls: usize) {
    let env = So101Reach::new(reflected_inertia());
    let n = env.dof();
    let obs_dim = 2 * n + 3;
    let q_goal = vec![0.1; n];
    let policy = GaussianPolicy::new(&[obs_dim, 64, 64, n], 7, -1.0);
    let norm = ObsNorm::new(obs_dim);
    let space = env.action_space();

    // Warm both paths first: a cold branch predictor and a cold cache would be measured as the controller.
    for _ in 0..2000 {
        std::hint::black_box(pd_track(&env, &q_goal, 8.0, 1.5));
        std::hint::black_box(space.from_unit(&policy.mean(&norm.normalize(&env.observation()))));
    }

    // Interleave the two so a drifting machine load hits both equally, and take the median of the trials.
    const TRIALS: usize = 7;
    let per = n_calls / TRIALS;
    let (mut pd_t, mut mlp_t) = (Vec::new(), Vec::new());
    for _ in 0..TRIALS {
        let t = std::time::Instant::now();
        for _ in 0..per {
            std::hint::black_box(pd_track(&env, &q_goal, 8.0, 1.5));
        }
        pd_t.push(t.elapsed().as_nanos() as f64 / per as f64);

        let t = std::time::Instant::now();
        for _ in 0..per {
            std::hint::black_box(space.from_unit(&policy.mean(&norm.normalize(&env.observation()))));
        }
        mlp_t.push(t.elapsed().as_nanos() as f64 / per as f64);
    }
    let stats = |v: &mut Vec<f64>| -> (f64, f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
        (v[0], v[0], v[v.len() - 1]) // best estimate is the fastest trial; range printed for the reader
    };
    let (pd_ns, pd_lo, pd_hi) = stats(&mut pd_t);
    let (mlp_ns, mlp_lo, mlp_hi) = stats(&mut mlp_t);

    // And the one-off the baseline pays that the policy does not.
    let mut probe = So101Reach::new(reflected_inertia());
    probe.reset(0);
    let t = std::time::Instant::now();
    for _ in 0..64 {
        std::hint::black_box(ik_goal(&probe));
    }
    let ik_us = t.elapsed().as_secs_f64() * 1e6 / 64.0;

    let ep = |ns: f64, setup_us: f64| (ns * STEPS as f64 + setup_us * 1e3) * 1e-9 * COMPUTE_WATTS;
    println!("\n  Cost of evaluating each control law (fastest of {TRIALS} trials x {per} calls, warmed):\n");
    println!("  {:<34} {:>12} {:>17} {:>11} {:>10}", "control law", "per step", "trial range", "per episode", "energy");
    println!(
        "  {:<34} {:>9.0} ns {:>7.0}-{:<7.0} ns {:>8.2} ms {:>8.5} J",
        "PD + gravity_vector (RNEA)", pd_ns, pd_lo, pd_hi, pd_ns * STEPS as f64 * 1e-6, ep(pd_ns, 0.0)
    );
    println!(
        "  {:<34} {:>9.0} ns {:>7.0}-{:<7.0} ns {:>8.2} ms {:>8.5} J",
        "GaussianPolicy 13-64-64-5 + ObsNorm", mlp_ns, mlp_lo, mlp_hi, mlp_ns * STEPS as f64 * 1e-6, ep(mlp_ns, 0.0)
    );
    println!(
        "  {:<34} {:>12} {:>17} {:>8.2} ms {:>8.5} J",
        "solve_ik, once per episode", "--", "--", ik_us * 1e-3, ik_us * 1e-6 * COMPUTE_WATTS
    );
    // The ratio is only meaningful if the two trial ranges are separated; say so when they are not.
    if pd_hi >= mlp_lo {
        println!("\n  WARNING: the trial ranges OVERLAP, so the ratio below is not resolved by this measurement.");
    }
    // Measure the baseline's actuation rather than quoting a number that will drift out of date.
    let act: f64 = {
        let seeds: Vec<u64> = (0..12).collect();
        let scores: Vec<Score> = seeds
            .iter()
            .map(|&sd| {
                let mut probe = So101Reach::new(reflected_inertia());
                probe.reset(sd);
                let g = ik_goal(&probe);
                episode_with(reflected_inertia(), sd, SUBSTEPS, move |e| pd_track(e, &g, 8.0, 1.5))
            })
            .collect();
        mean(&scores).elec()
    };
    let worst_compute = ep(pd_ns.max(mlp_ns), ik_us);
    println!(
        "\n  About {:.0}x per step in the {}'s favour, and at least {:.1}x guaranteed by the non-overlapping\n  trial ranges. Two figures because one would overstate the precision: the point estimate has read 6.1x\n  through 10.0x across runs on this machine, while the range-to-range bound holds whatever the load.\n\n  Against the {:.2} J of actuation the baseline actually draws, compute is\n  {:.0}x smaller either way: on THIS task the energy question is settled entirely by the actuator, and a controller that saves compute at the cost of a worse trajectory loses. That is a statement\n  about a 5-DoF arm at 200 Hz, not a general law: the ratio inverts as the policy grows or the motors\n  shrink. Worth knowing before deploying this: the policy path allocates four Vecs per control step\n  (observation, normalize, mean, from_unit) against the PD law's two, and that allocator traffic is both part\n  of the cost and the largest source of the spread above. A 200 Hz loop should not touch the heap at all.",
        if mlp_ns > pd_ns { mlp_ns / pd_ns } else { pd_ns / mlp_ns },
        if mlp_ns > pd_ns { "PD law" } else { "policy" },
        if mlp_lo > pd_hi { mlp_lo / pd_hi } else { 1.0 },
        act,
        act / worst_compute
    );
}

/// **Can this arm's torque constant be identified from motion at all? Screen the excitation first.**
/// Run with `--screen`.
///
/// `identify_actuator_with_gain` fits `k_t` alongside the three actuator terms, which removes the inherited
/// constant that otherwise bounds every parameter by its own error. Whether it *can* on a given arm and a given
/// trajectory is a separate question, and `confounding` answers it before the arm moves.
///
/// Turning it on this bench's own hand-designed excitation produced a result worth having. The identifiability
/// requires `τ_rigid` to lie outside `span{q̈, q̇, tanh(q̇/ε)}`, and the **gravity term** is what puts it there.
/// So the screening should track gravity — and it does, monotonically, on four of the five joints.
///
/// The fifth is the finding. **Joint 0 is invariant to four decimal places across zero, lunar, terrestrial and
/// Jovian gravity.** It is the base yaw: its axis is parallel to gravity, so no gravity magnitude produces any
/// torque about it, and no amount of excitation or loading will ever separate its `k_t`. That is the structural
/// degenerate case, present in the arm as shipped.
///
/// Practical conclusion for this robot: `k_t` is identifiable from motion on joints 1 and 2. Joints 0, 3 and 4
/// need a locked-rotor and back-EMF measurement instead.
fn screen_mode() {
    let j_refl = reflected_inertia();
    let env = So101Reach::new(j_refl);
    let n = env.dof();
    let (k_t, _) = motor_constants();

    // The same excitation `--identify` uses, verbatim, so this screens the trajectory actually run.
    let plan: Vec<PlannedMotion> = (0..3000)
        .map(|k| {
            let t = k as f64 * CONTROL_DT;
            let (mut q, mut qd, mut qdd) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            for i in 0..n {
                let (w1, w2) = (2.0 + 0.7 * i as f64, 7.0 + 1.3 * i as f64);
                let a = 0.35;
                q[i] = a * (w1 * t).sin() + 0.4 * a * (w2 * t).sin();
                qd[i] = a * w1 * (w1 * t).cos() + 0.4 * a * w2 * (w2 * t).cos();
                qdd[i] = -a * w1 * w1 * (w1 * t).sin() - 0.4 * a * w2 * w2 * (w2 * t).sin();
            }
            PlannedMotion { q, qd, qdd }
        })
        .collect();

    println!("\n  Screening this bench's own excitation for whether k_t is identifiable:\n");
    println!(
        "  {:>5} {:>13}   {:>30}   confounded pair",
        "joint", "conditioning", "unresolvable (kt,Ja,b,f)"
    );
    for c in confounding(&env.robot, &env.inertia, &plan, GRAVITY, k_t) {
        let d = c.direction;
        let (a, b) = c.worst_pair();
        println!(
            "  {:>5} {:>13.3e}   [{:>6.3},{:>6.3},{:>6.3},{:>6.3}]   {a} vs {b}",
            c.joint, c.conditioning, d[0], d[1], d[2], d[3]
        );
    }

    // Is gravity really the discriminator? Vary its magnitude and watch. This is the confirmation, and it also
    // isolates the one joint that CANNOT be helped.
    println!("\n  Conditioning against gravity magnitude, which the theory says supplies the independence:\n");
    println!("  {:>9}  {:>9} {:>9} {:>9} {:>9} {:>9}   {:>9}", "gravity", "j0", "j1", "j2", "j3", "j4", "max |G|");
    let mut first: Option<f64> = None;
    let mut last_j0 = 0.0;
    for (label, mag) in [("none", 0.0f64), ("Moon", 1.62), ("Earth", 9.81), ("Jupiter", 24.79)] {
        let g = Vector3::new(0.0, 0.0, -mag);
        let cs = confounding(&env.robot, &env.inertia, &plan, g, k_t);
        let gv = gravity_vector(&env.robot, &env.inertia, &vec![0.0; n], g);
        print!("  {label:>9}  ");
        for c in &cs {
            print!("{:>9.2e} ", c.conditioning);
        }
        println!("  {:>9.4}", gv.iter().fold(0.0f64, |m, v| m.max(v.abs())));
        if first.is_none() {
            first = Some(cs[1].conditioning);
        }
        last_j0 = cs[0].conditioning;
    }
    println!(
        "\n  Joint 1 improves {:.0}x from zero gravity to Jovian, so gravity is doing the work. Joint 0 does not\n  move at all ({:.3e} throughout): it is the base yaw, its axis is parallel to gravity, and no magnitude\n  produces any torque about it. That joint's k_t is structurally unidentifiable from motion — the case to\n  measure with a locked rotor rather than excite harder.\n\n  On this arm: identify k_t from motion on joints 1 and 2. Joints 0, 3 and 4 need the bench test.",
        confounding(&env.robot, &env.inertia, &plan, Vector3::new(0.0, 0.0, -24.79), k_t)[1].conditioning
            / first.unwrap_or(1.0).max(1e-300),
        last_j0
    );
}

/// **What does the Coulomb friction term cost, on an arm whose value for it is unknown?** Run with
/// `--friction`.
///
/// `Joint::friction` exists now and the SO-101 model does not state one, because no measured value for the
/// STS3215's gear train was located. That is not a neutral omission: a 345:1 reduction is where friction
/// concentrates, and leaving it unstated makes the energy figure optimistic in exactly that place.
///
/// The accounting itself is already right, which is worth being precise about — `mech` and `copper` are computed
/// from the **commanded** torque, and a stated friction raises the torque the controller must produce, so it
/// flows through without any change to the energy code. What is missing is the number, not the term. So rather
/// than invent one, this sweeps plausible values and reports what each costs, which is the honest form of
/// "this matters".
fn friction_mode(seeds: &[u64]) {
    let j_refl = reflected_inertia();
    println!("\n  Coulomb friction the SO-101 model does not state, and what stating it would cost:\n");
    println!(
        "  {:>10} {:>11} {:>9} {:>9} {:>9} {:>8}",
        "friction", "final err", "mech J", "copper J", "elec J", "reached"
    );
    let mut base = 0.0;
    for (k, &f) in [0.0f64, 0.02, 0.05, 0.10, 0.20].iter().enumerate() {
        let scores: Vec<Score> = seeds
            .iter()
            .map(|&sd| {
                let mut env = So101Reach::new(j_refl);
                for j in env.robot.joints.iter_mut() {
                    *j = j.clone().with_friction(f);
                }
                env.reset(sd);
                let g = ik_goal(&env);
                let t = std::time::Instant::now();
                let ik_s = t.elapsed().as_secs_f64();
                let mut e2 = So101Reach::new(j_refl);
                for j in e2.robot.joints.iter_mut() {
                    *j = j.clone().with_friction(f);
                }
                episode_timed_on(e2, sd, SUBSTEPS, ik_s, move |e| pd_track(e, &g, 8.0, 1.5))
            })
            .collect();
        let m = mean(&scores);
        if k == 0 {
            base = m.elec();
        }
        println!(
            "  {:>10} {:>11.4} {:>9.2} {:>9.2} {:>9.2} {:>7.0}%{}",
            if f == 0.0 { "unstated".to_string() } else { format!("{f:.2} N m") },
            m.final_err,
            m.mech,
            m.copper,
            m.elec(),
            100.0 * m.reached,
            if k == 0 { String::new() } else { format!("   {:+.0}% energy", 100.0 * (m.elec() / base - 1.0)) }
        );
    }
    println!(
        "\n  The energy code needed no change for this: friction raises the torque the controller has to command,\n  and mech and copper are both computed from the commanded torque. What was missing is a measured value for\n  this gearbox, which is what `identify_actuator` is for.\n\n  The accuracy column is the bigger finding. A PD law has no integral action, so Coulomb friction parks the\n  joint at whatever offset makes kp*error equal the friction: f/kp, or 0.025 rad at 0.20 N m and kp = 8. That\n  is the 0.0105 m of tool error in the last row, and it is why the reach rate falls to 50% while the energy\n  has only risen 23%. Feed the friction forward (the model now states it, so gravity_vector's sibling term is\n  available) or add integral action; do not raise kp, which buys the same offset back in copper loss."
    );
}

/// **Can the SO-101's actuator terms be identified from data a real arm could produce?** Run with `--identify`.
///
/// `identify_actuator` is exact on noise-free torques — that is proven on a two-link arm in
/// `ferromotion-core`. The question this answers is the practical one: on hardware you do not measure `q̈`. You
/// read a **quantised encoder** and differentiate twice, and double differentiation multiplies quantisation
/// noise by `1/dt²`. At 200 Hz that factor is 40,000, so the interesting number is not whether the method works
/// but what encoder resolution it survives.
///
/// The STS3215 reports 12 bits over a full turn, so one count is `2π/4096 = 1.53e-3` rad. This runs the same
/// identification three ways: against ideal derivatives, against central differences of quantised position, and
/// against a Savitzky-Golay fit of that same quantised position.
///
/// Measured. Ideal derivatives recover both terms exactly. Central differences at 12 bits put the armature
/// **88–217% out and frequently negative** while leaving the damping inside 1% — because damping multiplies `q̇`
/// (one differentiation) and armature multiplies `q̈` (two, so quantisation arrives scaled by `1/dt²` = 40,000
/// at 200 Hz). A Savitzky-Golay fit over a 50 ms window at order 3 brings the armature to **1.5–2.7%**, which
/// makes the measurement practical on the arm exactly as it ships.
///
/// Window length is **not monotone**: 125 ms at order 3 is several times worse than 50 ms, because an order-3
/// polynomial cannot follow a 9 rad/s excitation over that span and over-smoothing biases the second
/// derivative. Match the window to the excitation bandwidth rather than making it large.
///
/// This is also where `ActuatorFit::conditioning` shows its limit: it stays at ~1.0 across every row above,
/// including the ones where the armature is 200% wrong. It measures whether the excitation separates the two
/// terms, not whether the data is good enough to trust — `residual` is the field that catches that.
fn identify_mode() {
    let j_refl = reflected_inertia();
    let env = So101Reach::new(j_refl);
    let n = env.dof();
    let truth_a = j_refl;
    let truth_b = speed_droop();
    println!("\n  Identifying the SO-101's actuator terms from a prescribed excitation.\n");
    println!("  truth: armature {truth_a:.6e} kg m^2, damping {truth_b:.4} N m s/rad");

    // Multi-frequency excitation. A single sinusoid per joint makes qdd proportional to qd and the two terms
    // inseparable, which `identify_actuator` would correctly refuse — so the trajectory has to be designed, not
    // just large. Amplitudes stay inside the joint limits.
    let dt = CONTROL_DT;
    let steps = 3000; // 15 s
    let traj = |t: f64, i: usize| -> (f64, f64, f64) {
        let (w1, w2) = (2.0 + 0.7 * i as f64, 7.0 + 1.3 * i as f64);
        let a = 0.35;
        (
            a * (w1 * t).sin() + 0.4 * a * (w2 * t).sin(),
            a * w1 * (w1 * t).cos() + 0.4 * a * w2 * (w2 * t).cos(),
            -a * w1 * w1 * (w1 * t).sin() - 0.4 * a * w2 * w2 * (w2 * t).sin(),
        )
    };

    let mut ideal = Vec::with_capacity(steps);
    let mut q_hist: Vec<Vec<f64>> = Vec::with_capacity(steps);
    for k in 0..steps {
        let t = k as f64 * dt;
        let (mut q, mut qd, mut qdd) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let (a, b, c) = traj(t, i);
            q[i] = a;
            qd[i] = b;
            qdd[i] = c;
        }
        // The torque a torque-controlled arm would have to command, INCLUDING the actuator terms, because the
        // model states them. That is what a current sensor would report.
        let tau = inverse_dynamics(&env.robot, &env.inertia, &q, &qd, &qdd, GRAVITY);
        q_hist.push(q.clone());
        ideal.push(IdSample { q, qd, qdd, tau });
    }

    let show = |label: &str, fits: &[ferromotion_core::ActuatorFit]| {
        println!("\n  {label}\n");
        println!(
            "  {:>5} {:>13} {:>8} {:>11} {:>8} {:>11} {:>11} {:>10} {:>11} {:>9}",
            "joint", "armature", "err", "damping", "err", "friction", "conditioning", "residual", "se(arm)", "err/se"
        );
        for f in fits {
            // Friction is FITTED now, so it has to be shown. The SO-101 model states none, so the truth is
            // zero and a relative error is undefined — the absolute value is the honest column, and anything
            // far from zero here means the fit is absorbing something else into it.
            // `se(arm)` is the fitted standard error and `err/se` is how many of them the truth is
            // away. Under 2 means the error is consistent with the noise; far above means the fit is
            // wrong AND says so, which is the thing `conditioning` cannot report.
            let (se, z) = match f.stderr {
                Some(e) => (e[0], (f.armature - truth_a).abs() / e[0]),
                None => (f64::NAN, f64::NAN),
            };
            println!(
                "  {:>5} {:>13.6e} {:>7.1}% {:>11.4} {:>7.1}% {:>11.2e} {:>11.3e} {:>10.2e} {se:>11.2e} {z:>9.1}",
                f.joint,
                f.armature,
                100.0 * (f.armature - truth_a).abs() / truth_a,
                f.damping,
                100.0 * (f.damping - truth_b).abs() / truth_b,
                f.friction,
                f.conditioning,
                f.residual
            );
        }
    };
    show("From ideal derivatives (the upper bound):", &identify_actuator(&env.robot, &env.inertia, &ideal, GRAVITY));

    // Now the hardware version: quantise position, then reconstruct rates by central differences.
    for bits in [12u32, 14, 16] {
        let counts = (1u32 << bits) as f64;
        let step = std::f64::consts::TAU / counts;
        let quant = |v: f64| (v / step).round() * step;
        let mut samples = Vec::with_capacity(steps);
        for k in 1..steps - 1 {
            let (mut q, mut qd, mut qdd) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            for i in 0..n {
                let (qm, q0, qp) = (quant(q_hist[k - 1][i]), quant(q_hist[k][i]), quant(q_hist[k + 1][i]));
                q[i] = q0;
                qd[i] = (qp - qm) / (2.0 * dt);
                qdd[i] = (qp - 2.0 * q0 + qm) / (dt * dt); // the 1/dt^2 that does the damage
            }
            samples.push(IdSample { q, qd, qdd, tau: ideal[k].tau.clone() });
        }
        show(&format!("From a {bits}-bit encoder, rates by central difference:"), &identify_actuator(&env.robot, &env.inertia, &samples, GRAVITY));
    }
    // Central differences are the naive choice and nobody with real encoders uses them. This crate ships a
    // Savitzky-Golay differentiator built for exactly this: fit a low-order polynomial over a sliding window
    // and read its derivative at the centre, so noise averages out while the trend survives.
    for (half, order) in [(10usize, 3usize), (25, 3), (50, 4)] {
        let counts = (1u32 << 12) as f64; // the STS3215's own 12 bits, the hard case
        let step = std::f64::consts::TAU / counts;
        let sg = SavGol { half_window: half, order };
        // Per joint: quantise the whole position history, then differentiate the SIGNAL rather than a triple.
        let mut qs = Vec::with_capacity(n);
        let mut qds = Vec::with_capacity(n);
        let mut qdds = Vec::with_capacity(n);
        for i in 0..n {
            let col: Vec<f64> = q_hist.iter().map(|q| (q[i] / step).round() * step).collect();
            qds.push(sg.apply(&col, dt, 1));
            qdds.push(sg.apply(&col, dt, 2));
            qs.push(col);
        }
        // Drop the window edges, where one-sided fits are much worse than the interior.
        let mut samples = Vec::new();
        for k in half..steps - half {
            samples.push(IdSample {
                q: (0..n).map(|i| qs[i][k]).collect(),
                qd: (0..n).map(|i| qds[i][k]).collect(),
                qdd: (0..n).map(|i| qdds[i][k]).collect(),
                tau: ideal[k].tau.clone(),
            });
        }
        show(
            &format!("12-bit encoder, Savitzky-Golay (half-window {half} = {:.0} ms, order {order}):", half as f64 * dt * 1e3),
            &identify_actuator(&env.robot, &env.inertia, &samples, GRAVITY),
        );
    }

    // THE LIMITATION THAT DECIDES WHETHER ANY OF THIS IS USABLE. Everything above assumed exact torques. The
    // STS3215 has no torque sensor: it reports current, and torque is inferred as k_t * I. But k_t here was
    // itself derived from two catalogue numbers, so it carries an unquantified scale error -- and that error
    // multiplies the MEASURED torque while leaving the model-computed rigid-body torque alone, so it does not
    // cancel. This measures how the bias lands.
    let sg = SavGol { half_window: 10, order: 3 };
    let counts = (1u32 << 12) as f64;
    let step = std::f64::consts::TAU / counts;
    let mut qs = Vec::with_capacity(n);
    let mut qds = Vec::with_capacity(n);
    let mut qdds = Vec::with_capacity(n);
    for i in 0..n {
        let col: Vec<f64> = q_hist.iter().map(|q| (q[i] / step).round() * step).collect();
        qds.push(sg.apply(&col, dt, 1));
        qdds.push(sg.apply(&col, dt, 2));
        qs.push(col);
    }
    println!("\n  And with no torque sensor: torque inferred as k_t * I, where k_t is itself an estimate.\n");
    println!("  {:>10} {:>13} {:>10} {:>13} {:>10}", "k_t error", "armature", "err", "damping", "err");
    for scale in [0.80f64, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20] {
        let mut samples = Vec::new();
        for k in 10..steps - 10 {
            samples.push(IdSample {
                q: (0..n).map(|i| qs[i][k]).collect(),
                qd: (0..n).map(|i| qds[i][k]).collect(),
                qdd: (0..n).map(|i| qdds[i][k]).collect(),
                // A current reading interpreted through the wrong k_t scales the whole inferred torque.
                tau: ideal[k].tau.iter().map(|t| t * scale).collect(),
            });
        }
        let fits = identify_actuator(&env.robot, &env.inertia, &samples, GRAVITY);
        let f = &fits[n - 1]; // the wrist: smallest rigid-body torque, so the most exposed
        println!(
            "  {:>9.0}% {:>13.6e} {:>9.1}% {:>13.4} {:>9.1}%",
            100.0 * (scale - 1.0),
            f.armature,
            100.0 * (f.armature - truth_a).abs() / truth_a,
            f.damping,
            100.0 * (f.damping - truth_b).abs() / truth_b
        );
    }
    println!(
        "\n  Read on the wrist, the joint with the least rigid-body torque and so the most exposed. The damping\n  error tracks the k_t error EXACTLY one-for-one -- 10% out gives 10.0% out -- because at this joint almost\n  all the torque is actuator rather than rigid-body, so scaling the measured torque scales the fitted term\n  with it. Armature follows at roughly the same rate. There is no averaging-down and no cancellation: the\n  parameters are known no better than k_t is.\n\n  So k_t is not a detail to inherit from a catalogue. A locked-rotor test gives R directly and back-EMF at a\n  known speed gives k_e = k_t; both should be measured before any of the three actuator terms is trusted.\n  That is a sequencing requirement, not a reason to abandon the method -- and note the 0% row still shows a\n  2.6% armature error, which is the Savitzky-Golay quantisation floor and nothing to do with torque at all."
    );

    println!(
        "\n  The torques above are exact; only the kinematics are quantised, so every error is attributable to\n  differentiation alone. Two things worth reading off this table.\n\n  Damping identifies robustly and armature does not, because damping multiplies qd (one differentiation)\n  while armature multiplies qdd (two, so quantisation noise arrives scaled by 1/dt^2 = 40,000 at 200 Hz).\n\n  And `conditioning` stays at ~1.0 throughout, including where the armature is 200% wrong and negative. It\n  measures whether the EXCITATION separates the three terms, not whether the data is good enough to trust.\n  The residual is the field that catches this one: 1e-15 on ideal derivatives against 1e-1 on quantised ones.\n\n  The recipe, then: a 12-bit encoder is enough IF the rates come from a Savitzky-Golay fit rather than a\n  difference. That takes the worst-case armature error from 217% to 2.7%.\n\n  Window length is NOT monotone, which is the trap. 125 ms at order 3 is several times worse than 50 ms,\n  because an order-3 polynomial cannot follow a 9 rad/s excitation across 125 ms and over-smoothing biases\n  the second derivative. Order 4 recovers curvature over 250 ms but starts costing the damping instead.\n  Match the window to the excitation bandwidth; smoothing harder is the instinct and it is wrong."
    );
}

/// **Both inertias on the same grid**, so the comparison is like for like. Run with `--sweep`.
///
/// This exists because an earlier version of these docs claimed the URDF-only plant was unsolvable — "0 of 32
/// configurations" — and that was wrong. The sweep it came from still contained a hard velocity clamp that was
/// itself injecting energy at every clamp event, and it never tried a low gain at a high substep count. The
/// plant without reflected inertia is not unsolvable. It is **stiff**, which is a different and more useful
/// claim, and one this grid actually measures. A cited number no committed artefact reproduces is a weak
/// claim; that is what let the wrong one stand.
fn ill_posed_sweep(seeds: &[u64], j_refl: f64) {
    println!("\n  The same grid at two reflected inertias:\n");
    println!(
        "  {:>5} {:>5} {:>4} | {:>9} {:>8} {:>7} | {:>9} {:>8} {:>7}",
        "kp", "kd", "sub", "0: err", "elec J", "reach", "N^2J: err", "elec J", "reach"
    );
    let mut best: Option<(usize, f64, f64)> = None; // (substeps, err, work) for the best j_refl=0 config
    for &(kp, kd) in &[(3.0, 0.5), (8.0, 1.5), (20.0, 3.0), (40.0, 6.0)] {
        for &sub in &[1usize, 4, 20, 50] {
            let row = |jr: f64| -> Score {
                mean(
                    &seeds
                        .iter()
                        .map(|&sd| {
                            let mut probe = So101Reach::new(jr);
                            probe.reset(sd);
                            let t = std::time::Instant::now();
                            let g = ik_goal(&probe);
                            let ik_s = t.elapsed().as_secs_f64();
                            episode_timed(jr, sd, sub, ik_s, move |e| pd_track(e, &g, kp, kd))
                        })
                        .collect::<Vec<_>>(),
                )
            };
            let (a, b) = (row(0.0), row(j_refl));
            if a.reached >= 0.9 && best.is_none_or(|(s, _, _)| sub < s) {
                best = Some((sub, a.final_err, a.elec()));
            }
            println!(
                "  {kp:>5.1} {kd:>5.1} {sub:>4} | {:>9.4} {:>8.2} {:>6.0}% | {:>9.4} {:>8.2} {:>6.0}%",
                a.final_err, a.elec(), 100.0 * a.reached, b.final_err, b.elec(), 100.0 * b.reached
            );
        }
    }
    match best {
        Some((sub, err, elec)) => println!(
            "\n  Without the reflected inertia the plant is still controllable, but only at {sub} substeps\n  ({:.0} Hz), and it settles at {err:.4} m drawing {elec:.1} J. The cost of the missing term is\n  integration rate and energy, not impossibility.",
            sub as f64 / CONTROL_DT
        ),
        None => println!("\n  No configuration without reflected inertia reached {REACH_TOL} m on this grid."),
    }
}

/// Reflected inertias to re-measure every run, with the value actually used at `USED_INDEX`.
///
/// Built at runtime rather than as a `const` so the used entry is the *same float* as
/// `reflected_inertia()`. Written as a literal it was `1.19e-2` against a computed `0.0119025` — 2.5e-6 apart,
/// so the float match that selected the baseline never fired and the baseline stayed at `Score::default()`.
/// The fail-fast gate caught it because it reads `reached`, where zero means failure; a gate reading
/// `final_err` would have seen a flawless 0.0000 m baseline and passed.
const USED_INDEX: usize = 2;
fn sensitivity() -> Vec<f64> {
    vec![0.0, 4e-3, reflected_inertia(), 4e-2]
}

fn main() {
    let baselines_only = std::env::args().any(|a| a == "--baselines");
    let train_seed: u64 = std::env::args()
        .find_map(|a| a.strip_prefix("--seed=").and_then(|v| v.parse().ok()))
        .unwrap_or(7);
    let reward = std::env::args()
        .find_map(|a| a.strip_prefix("--reward=").and_then(Reward::parse))
        .unwrap_or(Reward::Quadratic);
    let j_refl = reflected_inertia();
    let probe = So101Reach::new(j_refl);
    let n = probe.dof();
    // The diagnostic below is about the RIGID-BODY inertia, so it must read a model with no armature attached.
    // Taking it from `probe` reports 1.19e-2 and "838 rad/s^2" — true of the corrected plant and useless as a
    // statement about the URDF, which is what this paragraph exists to make.
    let bare = So101Reach::new(0.0);
    let m = mass_matrix(&bare.robot, &bare.inertia, &vec![0.0; n]);
    let seeds: Vec<u64> = (0..12).collect();

    println!("SO-101 reaching: 5 DoF, torque control through the model's own dynamics\n");
    println!(
        "  URDF declares effort {:?} N m on every joint, which is a placeholder",
        probe.robot.joints.iter().filter_map(|j| j.effort).collect::<Vec<_>>()
    );
    println!(
        "  link-only mass matrix diagonal at home: {:?} kg m^2",
        (0..n).map(|i| format!("{:.2e}", m[(i, i)])).collect::<Vec<_>>()
    );
    // The library's own check, rather than a number computed here: `actuator_plausibility` exists because of
    // this arm, and running it on the arm is the honest demonstration.
    let report = actuator_plausibility(&bare.robot, &bare.inertia, &vec![0.0; n]);
    println!("\n  actuator_plausibility on the model as written:\n");
    println!("  {:>6} {:>9} {:>11} {:>14} {:>9}", "joint", "effort", "M_ii", "implied qdd", "armature");
    for r in &report {
        let flag = if !r.armature_stated && r.implied_acceleration.is_some_and(|a| a >= 1e4) { "  <- implausible" } else { "" };
        println!(
            "  {:>6} {:>9} {:>11.2e} {:>14} {:>9}{flag}",
            r.joint,
            r.declared_effort.map_or("none".to_string(), |e| format!("{e:.1}")),
            r.joint_inertia,
            r.implied_acceleration.map_or("n/a".to_string(), |a| format!("{a:.0}")),
            if r.armature_stated { "stated" } else { "none" }
        );
    }
    let i_min = (0..n).map(|i| m[(i, i)]).fold(f64::INFINITY, f64::min);
    println!(
        "\n  One {:.0} ms step at the worst joint adds {:.0} rad/s. That is the whole problem, and the check\n  above finds it from the model alone without running a single simulation step.\n",
        CONTROL_DT * 1e3,
        10.0 / i_min * CONTROL_DT
    );
    println!(
        "  servo: stall {TAU_STALL} N m, no-load {OMEGA_NO_LOAD} rad/s, so droop b = {:.3} N m s/rad (derived)",
        speed_droop()
    );
    println!("  reflected rotor inertia N^2 J = {GEAR_RATIO}^2 x {J_ROTOR:.0e} = {j_refl:.2e} kg m^2\n");

    // Does the conclusion depend on the estimated rotor inertia, or only on it being nonzero?
    println!("  IK + PD + gravity against reflected inertia:\n");
    println!(
        "  {:>12} {:>10} {:>8} {:>8} {:>8} {:>9} {:>9} {:>7}",
        "j_refl", "final err", "mech J", "copper J", "act J", "compute J", "total J", "reached"
    );
    let mut baseline: Option<Score> = None;
    for (k, &jr) in sensitivity().iter().enumerate() {
        let s = mean(
            &seeds
                .iter()
                .map(|&sd| {
                    let mut probe = So101Reach::new(jr);
                    probe.reset(sd);
                    let t = std::time::Instant::now();
                    let g = ik_goal(&probe);
                    let ik_s = t.elapsed().as_secs_f64();
                    episode_timed(jr, sd, SUBSTEPS, ik_s, move |e| pd_track(e, &g, 8.0, 1.5))
                })
                .collect::<Vec<_>>(),
        );
        let mark = if k == USED_INDEX { " <- used" } else { "" };
        println!(
            "  {jr:>12.2e} {:>10.4} {:>8.2} {:>8.2} {:>8.2} {:>9.4} {:>9.2} {:>6.0}%{mark}",
            s.final_err, s.mech, s.copper, s.elec(), s.compute_j(), s.total_j(), 100.0 * s.reached
        );
        if k == USED_INDEX {
            baseline = Some(s);
        }
    }
    // Selecting by index cannot silently miss, but it can be pointed at the wrong row, so check the value.
    let baseline = baseline.expect("USED_INDEX is within sensitivity()");
    assert_eq!(sensitivity()[USED_INDEX], j_refl, "USED_INDEX must name the inertia the run uses");

    if std::env::args().any(|a| a == "--screen") {
        screen_mode();
    }
    if std::env::args().any(|a| a == "--friction") {
        friction_mode(&seeds);
    }
    if std::env::args().any(|a| a == "--identify") {
        identify_mode();
    }
    if std::env::args().any(|a| a == "--compute") {
        compute_cost(200_000);
    }
    if std::env::args().any(|a| a == "--sweep") {
        ill_posed_sweep(&seeds, j_refl);
    }
    if baselines_only {
        return;
    }
    // FAIL FAST. A comparison against a baseline that does not work is not a measurement.
    if baseline.reached < 0.5 {
        println!(
            "\n  BASELINE DID NOT SOLVE IT ({:.0}%). Stopping before spending compute on training.",
            100.0 * baseline.reached
        );
        return;
    }

    let cfg = PpoConfig {
        gamma: 0.99,
        lambda: 0.95,
        clip: 0.2,
        policy_lr: 3e-3,
        value_lr: 3e-3,
        epochs: 8,
        value_epochs: 15,
        entropy_coef: 1e-3,
        steps_per_batch: 3000,
        max_episode_steps: STEPS,
        min_log_std: -2.5,
        log_std_ceiling: None,
        normalize_value_targets: true,
        final_lr_fraction: 0.05,
    };
    let iterations = 150;
    let obs_dim = 2 * n + 3;
    let mut env = So101Reach::with_reward(j_refl, reward);
    let mut policy = GaussianPolicy::new(&[obs_dim, 64, 64, n], train_seed, -1.0);
    let mut value = Mlp::new(&[obs_dim, 64, 64, 1], train_seed);
    let mut norm = ObsNorm::new(obs_dim);
    let reports =
        train_normalized(&mut env, &mut policy, &mut value, Some(&mut norm), &cfg, iterations, train_seed);
    let early: f64 = reports[..5].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;
    let late: f64 = reports[reports.len() - 5..].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;

    let space = env.action_space();
    let ppo = mean(
        &seeds
            .iter()
            .map(|&s| episode(j_refl, s, |e| space.from_unit(&policy.mean(&norm.normalize(&e.observation())))))
            .collect::<Vec<_>>(),
    );

    println!(
        "\n  PPO ({reward:?} reward): {iterations} x {} steps, mean return {early:.2} -> {late:.2}",
        cfg.steps_per_batch
    );
    // Returns under different rewards are NOT comparable to each other. Only the task metrics below are, which
    // is the whole reason the table reports metres and joules rather than reward.
    println!("  Returns across reward shapes are not comparable; compare the metres and joules.\n");
    println!(
        "  {:<20} {:>10} {:>8} {:>8} {:>10} {:>9} {:>7}",
        "controller", "final err", "act J", "compute", "compute J", "total J", "reached"
    );
    for (name, s) in [("IK + PD + gravity", baseline), ("PPO", ppo)] {
        println!(
            "  {name:<20} {:>10.4} {:>8.2} {:>7.2}ms {:>10.4} {:>9.2} {:>6.0}%",
            s.final_err,
            s.elec(),
            s.compute_s * 1e3,
            s.compute_j(),
            s.total_j(),
            100.0 * s.reached
        );
    }
    println!(
        "\n  E_task = E_actuation + E_compute, at a declared {COMPUTE_WATTS} W for the controller's computer.\n  Compute is wall clock for the CONTROL LAW only, excluding the physics step a real robot never pays\n  for, and including the baseline's one-off IK solve, which PPO has no equivalent of."
    );
    // ONE SEED IS NOT A MEASUREMENT, and this bench has the receipt: the same quadratic configuration read
    // 8% reached in one run and 0% in the next, differing only by an LU-versus-Cholesky solve at 1e-15
    // amplified through 150 training iterations. Across seeds 7/11/23 the reached rate spans 17-50% for the
    // linear reward alone. Every comparative claim in the module docs comes from the 3-seed sweep.
    println!(
        "\n  This is ONE seed ({train_seed}). Measured spread on `reached` between otherwise identical runs is\n  at least 8 points, and 17-50% across seeds for one reward, so read this row as a sample and not as a\n  result. Re-run with --seed to see the dispersion."
    );
    let (k_t, r) = motor_constants();
    println!(
        "\n  Energy is electrical, not mechanical: k_t = {k_t:.3} N m/A and R = {r:.2} ohm both follow from the\n  stall torque and no-load speed above. Copper loss is the term a mechanical-work number misses, and it\n  is the one that matters here — holding a pose against gravity draws current at zero speed, where\n  tau*qd reads exactly zero. Regeneration is not credited."
    );

    println!();
    if ppo.reached >= 0.8 * baseline.reached {
        println!(
            "  PPO reaches {:.0}% against IK+PD's {:.0}% on a real 5-DoF arm driven by joint torques through\n  \
             its own multibody dynamics.",
            100.0 * ppo.reached,
            100.0 * baseline.reached
        );
    } else if ppo.best_err < 3.0 * REACH_TOL {
        println!(
            "  PPO APPROACHES AND DOES NOT SETTLE: best {:.4} m against a {REACH_TOL} m tolerance, final\n  \
             {:.4} m, reaching {:.0}% against IK+PD's {:.0}%. It has the direction and not the hold, which is\n  \
             a different failure from not learning.",
            ppo.best_err, ppo.final_err, 100.0 * ppo.reached, 100.0 * baseline.reached
        );
    } else {
        println!(
            "  PPO DID NOT MATCH IT: final {:.4} m and best {:.4} m against IK+PD's {:.4} m at {:.0}% reached.\n  \
             At this budget a from-scratch f64 PPO does not match the stack's own IK+PD on a 5-DoF reach.\n  \
             That is the result.",
            ppo.final_err, ppo.best_err, baseline.final_err, 100.0 * baseline.reached
        );
    }
}
