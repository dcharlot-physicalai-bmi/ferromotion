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
//! integration and low gain, settling at 0.0177 m on 20.4 J.
//!
//! The missing term is the **actuator's own reflected rotor inertia**. A geared servo presents `N²·J_rotor` at
//! its output, and for the SO-101's distal joints that dominates the link the URDF describes — by a factor of
//! ~340 at the wrist. `SeaJoint::reflected_inertia` in `ferromotion-control` has always computed it; nothing
//! had ever fed it into a URDF-loaded multibody plant. Adding it, plus the motor's linear speed–torque droop:
//!
//! | reflected inertia | reaches 1 cm | best settle | work | rate needed |
//! |---|---|---|---|---|
//! | 0 (URDF as written) | 1 of 16 configs | 0.0177 m | 20.4 J | 10 kHz |
//! | 1.19e-2 (N=345) | **16 of 16** | 0.0001 m | 2.1 J | 200 Hz |
//!
//! 50x the integration rate, ~10x the work, 22x the settling error, and a working region that collapses to one
//! corner of the grid. `SENSITIVITY` re-measures the inertia dependence every run and `--sweep` re-measures
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
    forward_dynamics, from_urdf_full, gravity_vector, mass_matrix, solve_ik, IkOptions, Iso, LinkInertia, Robot,
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
}

impl Score {
    /// Electrical energy drawn: delivered work plus resistive loss, with no regenerative recovery.
    fn elec(&self) -> f64 {
        self.mech + self.copper
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
    }
}

fn episode(j_refl: f64, seed: u64, law: impl FnMut(&So101Reach) -> Vec<f64>) -> Score {
    episode_with(j_refl, seed, SUBSTEPS, law)
}

fn episode_with(j_refl: f64, seed: u64, substeps: usize, mut law: impl FnMut(&So101Reach) -> Vec<f64>) -> Score {
    let mut env = So101Reach::new(j_refl);
    env.substeps = substeps;
    env.reset(seed);
    let mut best = f64::INFINITY;
    for _ in 0..STEPS {
        let a = law(&env);
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
                            let g = ik_goal(&{
                                let mut e = So101Reach::new(jr);
                                e.reset(sd);
                                e
                            });
                            episode_with(jr, sd, sub, move |e| pd_track(e, &g, kp, kd))
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
            "\n  Without the reflected inertia the plant is still controllable, but only at {sub} substeps\n               ({:.0} Hz), and it settles at {err:.4} m drawing {elec:.1} J. The cost of the missing term is\n               integration rate and energy, not impossibility.",
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
    let i_min = (0..n).map(|i| m[(i, i)]).fold(f64::INFINITY, f64::min);
    println!(
        "  10 N m into the smallest ({i_min:.2e}) is {:.0} rad/s^2 — one {:.0} ms step adds {:.0} rad/s\n",
        10.0 / i_min,
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
        "  {:>12} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "j_refl", "final err", "best err", "mech J", "copper J", "elec J", "reached"
    );
    let mut baseline: Option<Score> = None;
    for (k, &jr) in sensitivity().iter().enumerate() {
        let s = mean(
            &seeds
                .iter()
                .map(|&sd| {
                    let g = ik_goal(&{
                        let mut e = So101Reach::new(jr);
                        e.reset(sd);
                        e
                    });
                    episode(jr, sd, move |e| pd_track(e, &g, 8.0, 1.5))
                })
                .collect::<Vec<_>>(),
        );
        let mark = if k == USED_INDEX { " <- used" } else { "" };
        println!(
            "  {jr:>12.2e} {:>10.4} {:>10.4} {:>8.2} {:>8.2} {:>8.2} {:>7.0}%{mark}",
            s.final_err, s.best_err, s.mech, s.copper, s.elec(), 100.0 * s.reached
        );
        if k == USED_INDEX {
            baseline = Some(s);
        }
    }
    // Selecting by index cannot silently miss, but it can be pointed at the wrong row, so check the value.
    let baseline = baseline.expect("USED_INDEX is within sensitivity()");
    assert_eq!(sensitivity()[USED_INDEX], j_refl, "USED_INDEX must name the inertia the run uses");

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
        "  {:<22} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "controller", "final err", "best err", "mech J", "copper J", "elec J", "reached"
    );
    for (name, s) in [("IK + PD + gravity", baseline), ("PPO", ppo)] {
        println!(
            "  {name:<22} {:>10.4} {:>10.4} {:>8.2} {:>8.2} {:>8.2} {:>7.0}%",
            s.final_err, s.best_err, s.mech, s.copper, s.elec(), 100.0 * s.reached
        );
    }
    // ONE SEED IS NOT A MEASUREMENT, and this bench has the receipt: the same quadratic configuration read
    // 8% reached in one run and 0% in the next, differing only by an LU-versus-Cholesky solve at 1e-15
    // amplified through 150 training iterations. Across seeds 7/11/23 the reached rate spans 17-50% for the
    // linear reward alone. Every comparative claim in the module docs comes from the 3-seed sweep.
    println!(
        "\n  This is ONE seed ({train_seed}). Measured spread on `reached` between otherwise identical runs is\n           at least 8 points, and 17-50% across seeds for one reward, so read this row as a sample and not as a\n           result. Re-run with --seed to see the dispersion."
    );
    let (k_t, r) = motor_constants();
    println!(
        "\n  Energy is electrical, not mechanical: k_t = {k_t:.3} N m/A and R = {r:.2} ohm both follow from the\n           stall torque and no-load speed above. Copper loss is the term a mechanical-work number misses, and it\n           is the one that matters here — holding a pose against gravity draws current at zero speed, where\n           tau*qd reads exactly zero. Regeneration is not credited."
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
