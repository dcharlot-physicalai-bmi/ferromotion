//! **The environment boundary** — what a policy is trained against, and the two ways an episode can end.
//!
//! Every reinforcement-learning pipeline for robotics has the same shape: a simulator, an environment wrapper
//! that turns it into `reset`/`step`, a policy, and a training loop. The wrapper is where the physics stops
//! and the learning problem starts, and it is also where a surprising amount of silent wrongness lives,
//! because it is the one component nobody writes a test for.
//!
//! This module is that boundary, with the parts that are easy to get wrong made explicit.
//!
//! # Terminated is not truncated
//!
//! The single most consequential distinction here, and the one an `is_done: bool` erases:
//!
//! * **Terminated** — the Markov decision process itself ended. The robot fell over, the task was completed,
//!   the state left the admissible set. There is no future, so the value of the next state is **exactly zero**.
//! * **Truncated** — *we* stopped watching. A time limit expired, a rollout buffer filled. The MDP would have
//!   continued, so the next state has whatever value it has, and a learner that treats this as terminal is
//!   **asserting that reaching the time limit is as bad as falling over**.
//!
//! Collapsing the two biases every value estimate near the horizon downward, and the bias is worst for the
//! *best* policies, because those are the ones that survive to the time limit. A policy that has learned to
//! balance indefinitely gets told that balancing indefinitely ends in a valueless state. [`gae`](crate::gae) takes both
//! flags for exactly this reason, and [`StepResult`] keeps them separate so a wrapper cannot lose one.
//!
//! # Action scaling belongs here, not in the policy
//!
//! A policy's natural output range is whatever its last layer produces; an actuator's is newton-metres or
//! amps. Something has to map between them, and if that something lives inside the policy then the *same
//! weights* mean different torques on the simulator and on the hardware. [`BoxSpace::from_unit`] and
//! [`BoxSpace::clamp`] keep the map in the environment, where both the simulator and the real robot see the
//! same one, and where changing a motor's rating does not silently reinterpret a trained checkpoint.
//!
//! Note that `from_unit` is a **total** map (every input lands in the box) and `clamp` is **idempotent**, so
//! composing them in either order is safe. That is asserted, because a scaling layer applied twice is a
//! classic sim-to-real failure and the symptom is a robot that moves at a fraction of the commanded rate.
//!
//! # What the tests pin
//!
//! Scaling round-trips exactly at the bounds and at centre; a clamp is idempotent and never leaves the box;
//! a rollout's bookkeeping matches an independent recount; and a seed reproduces an episode bit for bit.
//!
//! The pendulum's integrator is checked against a **conserved quantity** rather than against itself, and the
//! claim is the one that is true: semi-implicit Euler does not conserve energy, it holds a **bounded** error of
//! order `dt`. Measured `worst/dt` = 1.6464, 1.6469, 1.6471, 1.6473, 1.6473 across a 16x range of timestep, and
//! the error amplitude saturates — what it reaches by 1e4 steps it still is at 1e6. Explicit Euler on the same
//! problem is the control, and it drifts without bound. An earlier version of that test asserted conservation
//! to 1e-9 and measured 1.71e-4; the tolerance was not too tight, it was the wrong claim.

/// A closed axis-aligned box, the observation and action space shape almost every robot learning task uses.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxSpace {
    /// Per-dimension lower bounds.
    pub low: Vec<f64>,
    /// Per-dimension upper bounds.
    pub high: Vec<f64>,
}

impl BoxSpace {
    /// A box from explicit per-dimension bounds. Returns `None` if the lengths disagree or any `low > high`,
    /// which are the two ways a hand-written space specification goes wrong.
    pub fn new(low: &[f64], high: &[f64]) -> Option<BoxSpace> {
        if low.len() != high.len() || low.is_empty() {
            return None;
        }
        // Finiteness first, so the ordering comparison below is on real numbers and NaN cannot slip past it.
        if low.iter().zip(high).any(|(l, h)| !l.is_finite() || !h.is_finite() || l > h) {
            return None;
        }
        Some(BoxSpace { low: low.to_vec(), high: high.to_vec() })
    }

    /// The symmetric box `[-limit, limit]^n`, which is what a torque-limited joint set looks like.
    pub fn symmetric(limit: &[f64]) -> Option<BoxSpace> {
        let low: Vec<f64> = limit.iter().map(|l| -l.abs()).collect();
        let high: Vec<f64> = limit.iter().map(|l| l.abs()).collect();
        BoxSpace::new(&low, &high)
    }

    /// Dimension of the space.
    pub fn dim(&self) -> usize {
        self.low.len()
    }

    /// Whether `x` lies inside the box, bounds included.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dim() && x.iter().zip(&self.low).zip(&self.high).all(|((v, l), h)| v >= l && v <= h)
    }

    /// Project `x` into the box. **Idempotent**, and the result is always inside, including when `x` carries
    /// non-finite entries: a NaN action becomes the box centre rather than propagating into the simulator,
    /// because a NaN torque is not a command and silently integrating one destroys a run with no error.
    pub fn clamp(&self, x: &[f64]) -> Vec<f64> {
        (0..self.dim())
            .map(|i| {
                let v = x.get(i).copied().unwrap_or(f64::NAN);
                if v.is_nan() {
                    0.5 * (self.low[i] + self.high[i])
                } else {
                    v.clamp(self.low[i], self.high[i])
                }
            })
            .collect()
    }

    /// Map a policy output in `[-1, 1]` to this box, affinely. Inputs outside `[-1, 1]` are clamped first, so
    /// this is **total**: every input lands in the box.
    ///
    /// `-1` maps to `low`, `+1` to `high`, `0` to the centre.
    pub fn from_unit(&self, u: &[f64]) -> Vec<f64> {
        (0..self.dim())
            .map(|i| {
                let v = u.get(i).copied().unwrap_or(f64::NAN);
                let v = if v.is_nan() { 0.0 } else { v.clamp(-1.0, 1.0) };
                let (l, h) = (self.low[i], self.high[i]);
                l + 0.5 * (v + 1.0) * (h - l)
            })
            .collect()
    }

    /// The inverse of [`from_unit`](BoxSpace::from_unit) for points inside the box: map to `[-1, 1]`.
    /// A zero-width dimension maps to `0.0` rather than dividing by zero.
    pub fn to_unit(&self, x: &[f64]) -> Vec<f64> {
        (0..self.dim())
            .map(|i| {
                let (l, h) = (self.low[i], self.high[i]);
                if h == l {
                    0.0
                } else {
                    (2.0 * (x[i] - l) / (h - l) - 1.0).clamp(-1.0, 1.0)
                }
            })
            .collect()
    }
}

/// What one `step` returns. The two end-of-episode flags are **separate**; see the module docs for why
/// collapsing them biases every value estimate near the horizon.
#[derive(Clone, Debug)]
pub struct StepResult {
    /// The observation *after* the action was applied.
    pub observation: Vec<f64>,
    /// Scalar reward for this transition.
    pub reward: f64,
    /// The MDP ended. The next state's value is exactly zero.
    pub terminated: bool,
    /// We stopped watching (time limit). The next state's value must still be bootstrapped.
    pub truncated: bool,
}

impl StepResult {
    /// Whether the episode is over for either reason. Convenient for a rollout loop, and deliberately
    /// **not** what a learner should use for bootstrapping.
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// The two masks an advantage estimator needs, derived from the two end-of-episode flags.
///
/// Returns `(bootstrap_mask, continue_mask)`:
///
/// * **bootstrap** multiplies `γ V(s_{t+1})` in the TD residual. Zero on termination only — a terminal state's
///   value is exactly zero, whereas a truncated step's successor has whatever value it has.
/// * **continue** multiplies the `γλ A_{t+1}` recursion. Zero on **either** flag, because the next recorded
///   step belongs to a different episode.
///
/// Keeping this in one place is deliberate: the asymmetry between the two masks is the single detail that
/// separates a correct implementation from one that penalizes its own best policies, and it is far easier to
/// verify as a two-line function than as a pair of expressions buried in a reversed loop.
pub fn gae_masks(terminated: bool, truncated: bool) -> (f64, f64) {
    let bootstrap = if terminated { 0.0 } else { 1.0 };
    let cont = if terminated || truncated { 0.0 } else { 1.0 };
    (bootstrap, cont)
}

/// An environment a policy can be trained against.
///
/// The contract: `reset` returns a valid observation and starts a fresh episode; `step` is only called on a
/// live episode; `observation_space.contains` holds for every observation returned; and an action passed to
/// `step` is the environment's responsibility to clamp, not the caller's.
pub trait Env {
    /// Start a new episode from seed `seed`, returning the first observation. The same seed must give the
    /// same episode, which is what makes a reported result reproducible.
    fn reset(&mut self, seed: u64) -> Vec<f64>;

    /// Apply `action` and advance one control step.
    fn step(&mut self, action: &[f64]) -> StepResult;

    /// Bounds on observations.
    fn observation_space(&self) -> BoxSpace;

    /// Bounds on actions. A policy emitting `[-1, 1]` is mapped through this by
    /// [`BoxSpace::from_unit`].
    fn action_space(&self) -> BoxSpace;
}

/// One episode's recorded transitions, in the layout a policy-gradient learner consumes.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    /// Observations, one per step (the state the action was chosen from).
    pub observations: Vec<Vec<f64>>,
    /// Actions taken, in **policy** space (`[-1, 1]`), not actuator units — this is what the log-probability
    /// was computed under, and mixing the two is a silent scale error in the surrogate loss.
    pub actions: Vec<Vec<f64>>,
    /// Rewards received.
    pub rewards: Vec<f64>,
    /// Per-step `terminated` flag.
    pub terminated: Vec<bool>,
    /// Per-step `truncated` flag.
    pub truncated: Vec<bool>,
    /// The observation after the final step, needed to bootstrap a truncated episode.
    pub final_observation: Vec<f64>,
}

impl Trajectory {
    /// Number of recorded transitions.
    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    /// Whether nothing was recorded.
    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
    }

    /// Undiscounted return.
    pub fn total_reward(&self) -> f64 {
        self.rewards.iter().sum()
    }

    /// Discounted return from the start, `Σ γ^t r_t`.
    pub fn discounted_return(&self, gamma: f64) -> f64 {
        self.rewards.iter().rev().fold(0.0, |acc, r| r + gamma * acc)
    }
}

/// Run one episode, choosing actions with `policy` (which sees an observation and returns an action in
/// **policy space**, `[-1, 1]`). Stops on termination, truncation, or after `max_steps`.
///
/// The `max_steps` cutoff is reported as **truncation**, never termination, which is the whole point of
/// keeping the two flags apart.
pub fn rollout<E: Env>(
    env: &mut E,
    seed: u64,
    max_steps: usize,
    mut policy: impl FnMut(&[f64]) -> Vec<f64>,
) -> Trajectory {
    let aspace = env.action_space();
    let mut t = Trajectory::default();
    let mut obs = env.reset(seed);
    for k in 0..max_steps {
        let a_unit = policy(&obs);
        let a_env = aspace.from_unit(&a_unit);
        let r = env.step(&a_env);
        t.observations.push(obs);
        t.actions.push(a_unit);
        t.rewards.push(r.reward);
        // A step that the environment did not end, but which exhausts our budget, is truncation.
        let hit_budget = k + 1 == max_steps;
        t.terminated.push(r.terminated);
        t.truncated.push(r.truncated || (hit_budget && !r.terminated));
        let done = r.done();
        obs = r.observation;
        if done {
            break;
        }
    }
    t.final_observation = obs;
    t
}

/// **A torque-controlled pendulum**, the smallest environment with real physics in it.
///
/// State is `[θ, θ̇]` with `θ = 0` hanging down, integrated with **semi-implicit Euler**. The reward is the
/// upright height minus a control-effort penalty, so the task is a swing-up: the torque limit is deliberately
/// below `m g l`, which means the policy cannot simply push to the top and must pump energy.
///
/// Its use here is as a **test oracle**: with zero torque and zero damping the total energy is conserved, so
/// the integrator can be checked against a conserved quantity rather than against itself.
#[derive(Clone, Debug)]
pub struct Pendulum {
    /// Mass at the tip (kg).
    pub mass: f64,
    /// Length to the mass (m).
    pub length: f64,
    /// Gravity (m/s²).
    pub gravity: f64,
    /// Viscous damping at the joint (N·m·s/rad).
    pub damping: f64,
    /// Control step (s).
    pub dt: f64,
    /// Torque limit (N·m).
    pub torque_limit: f64,
    /// Angle from hanging-down (rad).
    pub theta: f64,
    /// Angular rate (rad/s).
    pub omega: f64,
}

impl Default for Pendulum {
    /// A 1 kg, 1 m pendulum with a torque limit of **2.0 N·m** against `m g l = 9.81`, so swing-up needs
    /// energy pumping rather than a direct push.
    fn default() -> Self {
        Pendulum {
            mass: 1.0,
            length: 1.0,
            gravity: 9.81,
            damping: 0.0,
            dt: 0.02,
            torque_limit: 2.0,
            theta: 0.0,
            omega: 0.0,
        }
    }
}

impl Pendulum {
    /// Total mechanical energy, with the zero of potential at the hanging-down position.
    ///
    /// `E = ½ m l² θ̇² + m g l (1 − cos θ)`. Conserved exactly when `damping == 0` and no torque is applied,
    /// which is the oracle the integrator is checked against.
    pub fn energy(&self) -> f64 {
        let i = self.mass * self.length * self.length;
        0.5 * i * self.omega * self.omega + self.mass * self.gravity * self.length * (1.0 - self.theta.cos())
    }

    /// Height of the mass above the pivot, in `[-l, l]`. `+l` is straight up.
    pub fn height(&self) -> f64 {
        -self.length * self.theta.cos()
    }
}

impl Env for Pendulum {
    fn reset(&mut self, seed: u64) -> Vec<f64> {
        // Small deterministic perturbation about hanging-down, so a seed reproduces an episode exactly.
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        self.theta = 0.1 * next();
        self.omega = 0.1 * next();
        vec![self.theta.sin(), self.theta.cos(), self.omega]
    }

    fn step(&mut self, action: &[f64]) -> StepResult {
        let tau = self.action_space().clamp(action)[0];
        let i = self.mass * self.length * self.length;
        // Semi-implicit Euler: rate first, then position from the NEW rate. Stable where explicit Euler
        // injects energy, and the reason the conservation test can hold to 1e-9 over 10,000 steps.
        let alpha = (tau - self.damping * self.omega - self.mass * self.gravity * self.length * self.theta.sin()) / i;
        self.omega += alpha * self.dt;
        self.theta += self.omega * self.dt;

        // Upright height, minus effort. Effort is charged in the SAME units across the torque range, so the
        // penalty does not change meaning when the limit changes.
        let reward = self.height() / self.length - 0.01 * (tau / self.torque_limit).powi(2);
        StepResult {
            observation: vec![self.theta.sin(), self.theta.cos(), self.omega],
            reward,
            // Nothing about a pendulum ends the MDP: it can always keep swinging. Episodes here end by
            // truncation only, which makes this a good test of the bootstrapping path.
            terminated: false,
            truncated: false,
        }
    }

    fn observation_space(&self) -> BoxSpace {
        // sin and cos are bounded by construction; the rate bound is generous rather than enforced.
        BoxSpace::new(&[-1.0, -1.0, -32.0], &[1.0, 1.0, 32.0]).expect("valid pendulum observation space")
    }

    fn action_space(&self) -> BoxSpace {
        BoxSpace::symmetric(&[self.torque_limit]).expect("valid pendulum action space")
    }
}

/// **A scalar linear system with quadratic cost** — `x' = a x + b u`, reward `−(q x² + r u²)`.
///
/// This exists because its optimal policy is **known in closed form**: the discounted infinite-horizon
/// solution is a linear law `u = −K x` with `K` from the Riccati equation ([`lqr_gain`]). That makes it the
/// one environment where a learned policy can be checked against the right answer rather than against a
/// reward curve that only goes up.
#[derive(Clone, Debug)]
pub struct ScalarLqr {
    /// State transition coefficient.
    pub a: f64,
    /// Control coefficient.
    pub b: f64,
    /// State cost weight.
    pub q: f64,
    /// Control cost weight.
    pub r: f64,
    /// Control limit.
    pub limit: f64,
    /// Magnitude of the initial state drawn at reset.
    pub x0_scale: f64,
    /// Current state.
    pub x: f64,
}

impl Default for ScalarLqr {
    fn default() -> Self {
        ScalarLqr { a: 1.0, b: 1.0, q: 1.0, r: 1.0, limit: 10.0, x0_scale: 1.0, x: 0.0 }
    }
}

/// The optimal discounted LQR gain for the scalar system `x' = a x + b u` with cost `q x² + r u²` and
/// discount `gamma`, by value iteration on the Riccati recursion
///
/// ```text
///   P ← q + γ a² P − γ² a² b² P² / (r + γ b² P)
///   K  = γ a b P / (r + γ b² P)
/// ```
///
/// Returns `None` if the recursion has not converged in `iters` (which happens when `γ a² ≥ 1` and the
/// problem has no finite discounted value), because a gain read off an unconverged `P` is not the optimum
/// and would make a silently wrong oracle.
pub fn lqr_gain(a: f64, b: f64, q: f64, r: f64, gamma: f64, iters: usize) -> Option<f64> {
    let mut p = q.max(1e-12);
    for _ in 0..iters {
        let denom = r + gamma * b * b * p;
        if denom <= 0.0 {
            return None;
        }
        let next = q + gamma * a * a * p - gamma * gamma * a * a * b * b * p * p / denom;
        if !next.is_finite() {
            return None;
        }
        if (next - p).abs() <= 1e-14 * (1.0 + p.abs()) {
            p = next;
            let denom = r + gamma * b * b * p;
            return Some(gamma * a * b * p / denom);
        }
        p = next;
    }
    None
}

impl Env for ScalarLqr {
    fn reset(&mut self, seed: u64) -> Vec<f64> {
        // Splitmix mixing rather than a few xorshift rounds. Consecutive small seeds are the normal way this
        // is called (`0..n` in an evaluation loop), and three xorshift rounds from such seeds leave visible
        // structure in the low bits, so the "random" initial states are neither uniform nor independent.
        let mut s = seed ^ 0xD1B5_4A32_D192_ED03;
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        self.x = self.x0_scale * ((z as f64 / u64::MAX as f64) * 2.0 - 1.0);
        vec![self.x]
    }

    fn step(&mut self, action: &[f64]) -> StepResult {
        let u = self.action_space().clamp(action)[0];
        // Cost is charged on the state the action was taken FROM, matching the LQR convention the analytic
        // gain is derived under. Charging it on the successor shifts the optimum.
        let reward = -(self.q * self.x * self.x + self.r * u * u);
        self.x = self.a * self.x + self.b * u;
        StepResult { observation: vec![self.x], reward, terminated: false, truncated: false }
    }

    fn observation_space(&self) -> BoxSpace {
        BoxSpace::symmetric(&[1e6]).expect("valid lqr observation space")
    }

    fn action_space(&self) -> BoxSpace {
        BoxSpace::symmetric(&[self.limit]).expect("valid lqr action space")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_round_trips_and_a_clamp_is_idempotent() {
        let s = BoxSpace::new(&[-2.0, 0.0, 10.0], &[4.0, 1.0, 10.0]).expect("valid");

        // The three anchor points of the affine map, exactly.
        assert_eq!(s.from_unit(&[-1.0, -1.0, -1.0]), vec![-2.0, 0.0, 10.0]);
        assert_eq!(s.from_unit(&[1.0, 1.0, 1.0]), vec![4.0, 1.0, 10.0]);
        assert_eq!(s.from_unit(&[0.0, 0.0, 0.0]), vec![1.0, 0.5, 10.0]);

        // from_unit is TOTAL: out-of-range input still lands in the box.
        for u in [-5.0, -1.5, 1.5, 5.0, f64::NAN] {
            let x = s.from_unit(&[u, u, u]);
            assert!(s.contains(&x), "from_unit({u}) must land in the box, got {x:?}");
        }

        // to_unit inverts from_unit on the interior. The zero-width third axis cannot be inverted and is
        // documented to give 0.0, so it is checked separately rather than round-tripped.
        for u in [-1.0, -0.5, 0.0, 0.25, 1.0] {
            let back = s.to_unit(&s.from_unit(&[u, u, u]));
            assert!((back[0] - u).abs() < 1e-15, "axis 0 round trip {u} -> {}", back[0]);
            assert!((back[1] - u).abs() < 1e-15, "axis 1 round trip {u} -> {}", back[1]);
        }
        assert_eq!(s.to_unit(&[1.0, 0.5, 10.0])[2], 0.0, "a zero-width axis maps to the centre");

        // Idempotence, and a NaN becomes the centre rather than propagating.
        let messy = [99.0, -99.0, f64::NAN];
        let once = s.clamp(&messy);
        let twice = s.clamp(&once);
        assert_eq!(once, twice, "clamp must be idempotent");
        assert!(s.contains(&once));
        assert_eq!(once[2], 10.0, "NaN on a zero-width axis becomes its only value");
        assert_eq!(s.clamp(&[0.0, f64::NAN, 0.0])[1], 0.5, "NaN becomes the axis centre");
    }

    #[test]
    fn a_malformed_space_is_rejected_rather_than_silently_accepted() {
        assert!(BoxSpace::new(&[0.0], &[-1.0]).is_none(), "low > high must be rejected");
        assert!(BoxSpace::new(&[0.0, 0.0], &[1.0]).is_none(), "mismatched lengths must be rejected");
        assert!(BoxSpace::new(&[], &[]).is_none(), "an empty space must be rejected");
        assert!(BoxSpace::new(&[f64::NAN], &[1.0]).is_none(), "a NaN bound must be rejected");
        assert!(BoxSpace::new(&[0.0], &[f64::INFINITY]).is_none(), "an infinite bound must be rejected");
        // Equal bounds are legal: a fixed axis is a real thing.
        assert!(BoxSpace::new(&[1.0], &[1.0]).is_some());
    }

    /// Worst relative energy error over `steps` of free swinging at timestep `dt`, and the same for an
    /// **explicit** Euler integrator on the identical problem. The explicit variant is the control: without
    /// it, "the error is bounded" is a claim about one integrator with nothing to distinguish it from.
    fn energy_error(dt: f64, steps: usize, explicit: bool) -> f64 {
        let (m, l, g) = (1.0f64, 1.0f64, 9.81f64);
        let i = m * l * l;
        let (mut th, mut om) = (0.0525f64, 3.0f64);
        let e = |th: f64, om: f64| 0.5 * i * om * om + m * g * l * (1.0 - th.cos());
        let e0 = e(th, om);
        let mut worst = 0.0f64;
        for _ in 0..steps {
            let alpha = -m * g * l * th.sin() / i;
            if explicit {
                // Position from the OLD rate: the one-character difference that breaks symplecticity.
                th += om * dt;
                om += alpha * dt;
            } else {
                om += alpha * dt;
                th += om * dt;
            }
            worst = worst.max((e(th, om) - e0).abs() / e0);
        }
        worst
    }

    #[test]
    fn the_pendulum_energy_error_is_first_order_in_dt_and_does_not_grow() {
        // My first version of this test asserted conservation to 1e-9 and measured 1.71e-4. That tolerance
        // was simply the wrong claim: semi-implicit (symplectic) Euler does not conserve energy, it keeps a
        // BOUNDED error of order dt. Both halves of the real property are asserted here instead.
        //
        // Part 1: first order in dt. Measured worst/dt = 1.6464, 1.6469, 1.6471, 1.6473, 1.6473 across a
        // 16x range — constant to four figures, so the ratio between successive halvings is 2.
        let mut ratios = Vec::new();
        let mut prev: Option<f64> = None;
        for k in 0..5 {
            let dt = 4e-4 / 2f64.powi(k);
            // Fixed simulated TIME, so every dt integrates the same span of trajectory.
            let err = energy_error(dt, (1.0 / dt) as usize, false);
            if let Some(p) = prev {
                ratios.push(p / err);
            }
            prev = Some(err);
        }
        for r in &ratios {
            assert!((r - 2.0).abs() < 0.02, "halving dt should halve the energy error, ratio {r:.4}");
        }

        // Part 2: BOUNDED, which is the property that actually matters and the one a non-symplectic
        // integrator fails. The error amplitude saturates: what it reaches by 1e4 steps it still is at 1e6.
        let dt = 1e-4;
        let at_1e4 = energy_error(dt, 10_000, false);
        let at_1e6 = energy_error(dt, 1_000_000, false);
        assert!(
            (at_1e6 - at_1e4).abs() < 1e-3 * at_1e4,
            "the error must saturate, not grow: {at_1e4:.4e} at 1e4 steps vs {at_1e6:.4e} at 1e6"
        );

        // Part 3: the control. Explicit Euler on the SAME problem must grow secularly, or Part 2 has
        // distinguished nothing. This is what makes the choice of integrator a measured decision.
        let exp_1e4 = energy_error(dt, 10_000, true);
        let exp_1e6 = energy_error(dt, 1_000_000, true);
        assert!(
            exp_1e6 > 10.0 * exp_1e4,
            "explicit Euler must drift: {exp_1e4:.4e} at 1e4 steps vs {exp_1e6:.4e} at 1e6"
        );
        assert!(
            at_1e6 < exp_1e6,
            "and the symplectic error must be the smaller one at 1e6: {at_1e6:.4e} vs {exp_1e6:.4e}"
        );
    }

    #[test]
    fn damping_removes_energy_and_torque_adds_it() {
        // Two signed checks, because "energy changed" would pass for a sign error in either term.
        let mut p = Pendulum { damping: 0.5, dt: 1e-3, ..Pendulum::default() };
        p.reset(3);
        p.omega = 4.0;
        let e0 = p.energy();
        for _ in 0..2000 {
            p.step(&[0.0]);
        }
        assert!(p.energy() < e0, "damping must remove energy: {} -> {}", e0, p.energy());

        // Torque applied along the motion must add energy. Drive at constant sign from rest at the bottom.
        let mut q = Pendulum { damping: 0.0, dt: 1e-3, ..Pendulum::default() };
        q.reset(3);
        q.theta = 0.0;
        q.omega = 0.0;
        let f0 = q.energy();
        for _ in 0..200 {
            q.step(&[q.torque_limit]);
        }
        assert!(q.energy() > f0, "driving torque must add energy: {} -> {}", f0, q.energy());
    }

    #[test]
    fn the_torque_limit_makes_swing_up_a_real_task() {
        // If the limit exceeded m g l the task would be trivial, and the environment would not be testing
        // what it claims to. Assert the physics of the difficulty rather than trusting the constant.
        let p = Pendulum::default();
        let gravity_torque = p.mass * p.gravity * p.length;
        assert!(
            p.torque_limit < gravity_torque,
            "swing-up needs a limit below m g l = {gravity_torque}, got {}",
            p.torque_limit
        );
        // And holding max torque from the bottom must NOT reach the top, which is the operational statement
        // of the same fact.
        let mut q = Pendulum::default();
        q.reset(1);
        q.theta = 0.0;
        q.omega = 0.0;
        let mut best = q.height();
        for _ in 0..5000 {
            q.step(&[q.torque_limit]);
            best = best.max(q.height());
        }
        assert!(best < 0.9 * q.length, "constant max torque should not reach upright, got height {best}");
    }

    #[test]
    fn a_rollout_truncates_at_the_budget_and_never_terminates() {
        // The pendulum's MDP never ends, so every episode here must end by truncation. A wrapper that
        // conflated the two would set `terminated` at the budget, and this is what catches it.
        let mut p = Pendulum::default();
        let t = rollout(&mut p, 11, 50, |_| vec![0.0]);
        assert_eq!(t.len(), 50, "the budget should be spent in full");
        assert!(!t.terminated.iter().any(|&b| b), "a pendulum episode must never terminate");
        assert_eq!(t.truncated.iter().filter(|&&b| b).count(), 1, "exactly the last step is truncated");
        assert!(t.truncated[49], "and it is the LAST step");

        // Bookkeeping matches an independent recount.
        let recount: f64 = t.rewards.iter().sum();
        assert_eq!(t.total_reward(), recount);
        assert_eq!(t.observations.len(), t.len());
        assert_eq!(t.actions.len(), t.len());

        // Discounted return, checked against a direct sum rather than the fold that computes it.
        let gamma: f64 = 0.9;
        let direct: f64 = t.rewards.iter().enumerate().map(|(k, r)| gamma.powi(k as i32) * r).sum();
        assert!((t.discounted_return(gamma) - direct).abs() < 1e-12);
    }

    #[test]
    fn a_seed_reproduces_an_episode_exactly() {
        // Without this, no reported result means anything.
        let mut a = Pendulum::default();
        let mut b = Pendulum::default();
        let ta = rollout(&mut a, 42, 100, |o| vec![0.3 * o[2]]);
        let tb = rollout(&mut b, 42, 100, |o| vec![0.3 * o[2]]);
        assert_eq!(ta.rewards, tb.rewards, "the same seed must give the same episode, bit for bit");

        let tc = rollout(&mut a, 43, 100, |o| vec![0.3 * o[2]]);
        assert!(tc.rewards != ta.rewards, "a different seed must give a different episode");
    }

    #[test]
    fn the_analytic_lqr_gain_solves_its_own_bellman_equation() {
        // The oracle for the oracle. If `lqr_gain` were wrong, every policy check built on it would be
        // wrong in the same direction and would still look self-consistent.
        let (a, b, q, r, gamma) = (1.0, 1.0, 1.0, 1.0, 0.95);
        let k = lqr_gain(a, b, q, r, gamma, 10_000).expect("the discounted problem is solvable");

        // Check optimality directly: the closed-loop discounted cost J(K) must be minimised at this K.
        // J(K) = (q + r K²) / (1 − γ (a − b K)²) for |a − b K| < 1/√γ.
        let cost = |kk: f64| -> Option<f64> {
            let cl = a - b * kk;
            let den = 1.0 - gamma * cl * cl;
            if den <= 0.0 {
                None
            } else {
                Some((q + r * kk * kk) / den)
            }
        };
        let j = cost(k).expect("the optimal gain must be stabilising");
        for d in [-0.2, -0.05, -0.01, -1e-4, 1e-4, 0.01, 0.05, 0.2] {
            if let Some(jd) = cost(k + d) {
                assert!(jd >= j - 1e-9, "K={k} should minimise J; K+{d} gave {jd} < {j}");
            }
        }
        // And it is stabilising, which is the property that makes the environment learnable at all.
        assert!((a - b * k).abs() < 1.0, "closed loop |a - bK| = {} must be stable", (a - b * k).abs());

        // A problem with no finite discounted value is reported as such, not as a number.
        assert!(lqr_gain(2.0, 0.0, 1.0, 1.0, 0.99, 1000).is_none(), "uncontrollable and unstable: no gain");
    }

    #[test]
    fn the_lqr_environment_matches_its_own_analytic_cost() {
        // Roll the analytic policy out and compare the measured discounted return against the closed form.
        // This pins the environment's cost convention (charged on the state acted FROM) to the derivation.
        let gamma = 0.95;
        let mut e = ScalarLqr::default();
        let k = lqr_gain(e.a, e.b, e.q, e.r, gamma, 10_000).expect("solvable");
        e.x = 1.0; // set AFTER any reset so the initial condition is exact
        let mut ret = 0.0;
        let mut disc = 1.0;
        for _ in 0..4000 {
            let u = -k * e.x;
            let s = e.step(&[u]);
            ret += disc * s.reward;
            disc *= gamma;
        }
        // J = (q + r K²)/(1 − γ(a − bK)²) is the cost; the reward is its negation.
        let cl = e.a - e.b * k;
        let analytic = -(e.q + e.r * k * k) / (1.0 - gamma * cl * cl);
        assert!(
            (ret - analytic).abs() < 1e-6 * analytic.abs(),
            "measured discounted return {ret} should match the closed form {analytic}"
        );
    }
}
