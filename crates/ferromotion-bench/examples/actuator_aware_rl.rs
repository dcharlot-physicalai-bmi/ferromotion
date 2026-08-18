//! **What a torque command actually costs.**
//!
//! Twelve modules in this workspace model the chain between a commanded torque and a moving joint: the motor
//! drive, the winding that heats, the gearbox with clearance, the battery whose voltage sags, the fatigue that
//! load reversals accumulate. Every one is unit-tested in isolation. None appears in a reinforcement-learning
//! reward function, because a reward function sees position error and effort and nothing else.
//!
//! This bench wires them together and measures what the omission costs. Two policies, identical architecture,
//! hyperparameters and seed. One trained against an **ideal torque source** — the model an RL pipeline uses by
//! default. One trained against the **full actuator chain**. Both evaluated on the full chain, which is the only
//! environment here that stands in for hardware.
//!
//! Task cost is in the reward. Joules, peak winding temperature and fatigue damage are not. That is the point.
//!
//! # What it measures, with a 12-reach duty cycle on the sized joint
//!
//! ```text
//!   peak winding temperature        108.1 degC   (foldback at 100, limit 120)
//!   steps thermally derated          32.0 %
//!   steps inside the backlash         1.5 %
//!   energy per reach                 63.2 J
//!   reaches to fatigue failure       5558
//! ```
//!
//! A policy that maximised position error and effort drove the drive into **current foldback for a third of the
//! duty cycle**, which is authority the controller silently no longer had, and it was never told the winding
//! existed.
//!
//! **A more capable policy hits the limit harder, which is the sharper form of the point.** These numbers
//! replace an earlier run in which the observations were normalised by two constants chosen by hand inside the
//! environment (divide by π, divide by 20). Handing the transform to
//! [`ObsNorm`](ferromotion_learn::ObsNorm) instead moved the final return from **−53.6 to −48.7** (9.2%) on the
//! identical seed, budget and architecture, task cost from 43.3 to 40.8, and the fraction of derated steps from
//! 2.3% to **32.0%**. The better policy is not gentler on the hardware; it is worse, because it can now reach
//! harder and nothing in the reward charges it for the heat. A weak policy brushing a thermal limit is a
//! curiosity. A competent one living inside the foldback region for a third of its duty cycle is a design
//! problem.
//!
//! **A correction, recorded because the wrong number was published and then reasoned from.** An earlier version
//! of this comment gave that return improvement as "−70.6 to −48.7", i.e. 31%. The −70.6 belongs to the
//! *swing-up* bench's baseline, not to this one — two benches' numbers conflated. The real improvement here is
//! 9.2%, and the inflated figure was subsequently cited as evidence for spending a swing-up arm on observation
//! scaling. That arm was run and refuted anyway, so nothing downstream rests on it, but the lesson is that a
//! number quoted from memory across two experiments is a number to re-read from the log first.
//!
//! # Three limits on what this supports, stated because they bound the conclusion
//!
//! * **The absolute fatigue figure is not a claim about any gearbox.** It depends on a section modulus chosen so
//!   that 40 N m reads as 200 MPa. Only comparisons made with the same constant are meaningful.
//! * **The voltage limit never engages at 6:1.** Higher gearing raises back-EMF per unit load speed but lowers
//!   the current needed, so the thermal and voltage limits trade against each other; sizing for both at once is
//!   its own design problem and this joint is sized for the thermal one.
//! * **The ideal-versus-chain comparison is context, not attribution.** The two training returns are measured on
//!   different plants — the ideal one is easier, so its returns are higher by construction — and there is no
//!   common scale here on which "equally converged" is checkable. A real transfer claim needs both policies
//!   trained to convergence and evaluated on one plant over many seeds, which is a larger experiment.
//!
//! `cargo run --release -p ferromotion-bench --example actuator_aware_rl`

use ferromotion_control::{svpwm_voltage_limit, Backlash, Battery, MeanCorrection, MotorThermal, Pmsm, SnCurve};
use ferromotion_learn::{train_normalized, BoxSpace, Env, GaussianPolicy, Mlp, ObsNorm, PpoConfig, StepResult};
use std::cell::Cell;

/// Winding temperature at which the drive begins folding back current, and the limit it protects (deg C).
const T_FOLDBACK: f64 = 100.0;
const T_LIMIT: f64 = 120.0;
const AMBIENT: f64 = 25.0;

/// Telemetry the reward cannot see.
#[derive(Clone, Debug, Default)]
struct Telemetry {
    /// Net electrical energy drawn from the pack (J).
    joules: f64,
    /// Highest winding temperature reached (deg C).
    peak_winding: f64,
    /// Load-side torque at every control step, for rainflow counting.
    torque: Vec<f64>,
    /// Steps where the thermal foldback reduced the commanded current.
    derated: usize,
    /// Steps where the bus voltage, not the current limit, capped the torque.
    voltage_limited: usize,
    /// Steps where the gearbox was inside its backlash gap, transmitting nothing.
    in_backlash: usize,
}

/// A one-degree-of-freedom joint: a position task behind either an ideal torque source or the real chain.
#[derive(Clone, Debug)]
struct Joint {
    ideal: bool,
    motor: Pmsm,
    thermal: MotorThermal,
    battery: Battery,
    backlash: Backlash,
    /// Reduction ratio, motor to load.
    gear: f64,
    /// Load-side inertia (kg m^2).
    inertia: f64,
    /// Motor inertia referred to the load side (kg m^2).
    motor_inertia: f64,
    /// Load-side viscous damping (N m s/rad).
    damping: f64,
    /// Gravity torque amplitude at the load (N m).
    gravity: f64,
    /// Load-side torque limit (N m).
    torque_limit: f64,
    dt: f64,

    theta: f64,
    omega: f64,
    /// Motor angle referred to the load side, and its rate.
    theta_m: f64,
    omega_m: f64,
    target: f64,
    tau_prev: f64,
    steps: usize,
    max_steps: usize,
    tel: Telemetry,
    /// The previous step's DC current, for the sag estimate. A cell because `terminal_voltage` must be read
    /// while `self` is already mutably borrowed inside `through_chain`.
    last_current: Cell<f64>,
}

impl Joint {
    fn new(ideal: bool) -> Joint {
        // 7 pole pairs, 0.1 ohm, 0.5 mH, 0.02 Wb: k_t = 1.5 * 7 * 0.02 = 0.21 N m/A motor-side.
        let motor = Pmsm::surface(7.0, 0.1, 0.5e-3, 0.02);
        // Winding 20 J/K, housing 400 J/K, 1.2 K/W winding-to-housing, 2.5 K/W housing-to-air.
        let thermal = MotorThermal::new(0.1, 1.5, 400.0, 1.2, 2.5, AMBIENT);
        // 24 V nominal, 50 mohm internal, 2 Ah = 7200 C.
        let battery = Battery::lithium(7200.0, 0.05, 24.0);
        Joint {
            ideal,
            motor,
            thermal,
            battery,
            // 0.6 degrees of lost motion at the output, which is a good harmonic drive.
            backlash: Backlash::new(0.6f64.to_radians(), 0.0),
            // SIZED SO THE LIMITS BIND, which the first version was not. At 20:1 into a 0.05 kg m^2 load the
            // motor needed 0.7 A, dissipated 0.075 W, and the winding never left ambient — the bench reported
            // zero derated and zero voltage-limited steps and was measuring nothing at all.
            //
            // Working backwards instead: for the thermal foldback to engage inside a 3 s episode the winding
            // must rise ~75 K, so at a 4 J/K winding that needs ~100 W of copper loss, so 1.5*0.1*i^2 = 100
            // gives i ~ 26 A, so tau_motor = 0.21*26 = 5.5 N m, so at 6:1 the load must demand ~33 N m. That
            // is a knee-scale joint, and the numbers below are chosen to be it.
            gear: 6.0,
            inertia: 1.0,
            motor_inertia: 0.08,
            damping: 0.4,
            gravity: 28.0,
            torque_limit: 40.0,
            dt: 5e-3,
            theta: 0.0,
            omega: 0.0,
            theta_m: 0.0,
            omega_m: 0.0,
            target: 0.0,
            tau_prev: 0.0,
            steps: 0,
            max_steps: 200,
            tel: Telemetry::default(),
            last_current: Cell::new(0.0),
        }
    }

    /// Motor torque constant, N m per amp, motor-side.
    fn kt(&self) -> f64 {
        1.5 * self.motor.pole_pairs * self.motor.flux_linkage
    }

    /// Apply the full chain to a commanded load-side torque, returning the torque actually delivered.
    fn through_chain(&mut self, tau_cmd: f64) -> f64 {
        let kt = self.kt();
        let mut i_q = (tau_cmd / self.gear) / kt;

        // 1. Thermal foldback. A real drive derates before it destroys the winding, so the policy's authority
        //    shrinks exactly when it has been working hardest.
        let t = self.thermal.t_winding;
        if t > T_FOLDBACK {
            let scale = ((T_LIMIT - t) / (T_LIMIT - T_FOLDBACK)).clamp(0.0, 1.0);
            if scale < 1.0 {
                self.tel.derated += 1;
            }
            i_q *= scale;
        }

        // 2. Bus voltage. Back-EMF grows with speed and the pack sags under load, so the reachable torque falls
        //    as the joint moves faster and as the battery drains.
        let omega_motor = self.omega * self.gear;
        let omega_e = omega_motor * self.motor.pole_pairs;
        let v_bus = self.battery.terminal_voltage(self.tel_last_current());
        let v_max = svpwm_voltage_limit(v_bus);
        let (v_d, v_q) = self.motor.steady_voltage(0.0, i_q, omega_e);
        if (v_d * v_d + v_q * v_q).sqrt() > v_max {
            // Solve |R i + omega_e lambda| = v_max for the largest feasible i_q of the commanded sign.
            let back_emf = omega_e * self.motor.flux_linkage;
            let hi = (v_max - back_emf) / self.motor.r_s;
            let lo = (-v_max - back_emf) / self.motor.r_s;
            let clamped = i_q.clamp(lo.min(hi), lo.max(hi));
            if (clamped - i_q).abs() > 1e-12 {
                self.tel.voltage_limited += 1;
            }
            i_q = clamped;
        }

        // 3. Heat and energy from the current that survived both limits.
        self.motor.i_d = 0.0;
        self.motor.i_q = i_q;
        let tau_motor = kt * i_q;
        let tau_load = tau_motor * self.gear;
        let p_cu = self.motor.copper_loss();
        let p_mech = tau_motor * omega_motor;
        let p_elec = p_mech + p_cu;
        self.thermal.step(self.dt, self.motor.thermal_equivalent_current(), AMBIENT);
        self.tel.peak_winding = self.tel.peak_winding.max(self.thermal.t_winding);
        let i_dc = if v_bus > 1.0 { p_elec / v_bus } else { 0.0 };
        self.battery.step(self.dt, i_dc);
        self.tel.joules += p_elec * self.dt;
        self.last_current.set(i_dc);
        tau_load
    }

    /// The DC current the previous step drew, for the sag estimate. Interior mutability keeps `terminal_voltage`
    /// out of a borrow conflict with `&mut self`.
    fn tel_last_current(&self) -> f64 {
        self.last_current.get()
    }
}

impl Joint {
    fn wrap_pi(x: f64) -> f64 {
        let mut a = x;
        while a > std::f64::consts::PI {
            a -= 2.0 * std::f64::consts::PI;
        }
        while a < -std::f64::consts::PI {
            a += 2.0 * std::f64::consts::PI;
        }
        a
    }
}

impl Joint {
    /// Start a new reach WITHOUT clearing the winding temperature or the pack's charge.
    ///
    /// A single one-second reach cannot heat a motor to its limit — the first version of this bench reset
    /// everything each episode, peaked at 76 degC against a 100 degC foldback threshold, and therefore never
    /// engaged the limit it was built to measure. Thermal limits bind over **duty cycles**, so evaluation runs
    /// consecutive targets on one joint and lets the heat and the charge state accumulate.
    fn next_reach(&mut self, seed: u64) -> Vec<f64> {
        let u = Self::seed_to_unit(seed);
        self.theta = 0.0;
        self.omega = 0.0;
        self.theta_m = 0.0;
        self.omega_m = 0.0;
        self.target = u.signum() * (0.5 + 1.0 * u.abs());
        self.tau_prev = 0.0;
        self.steps = 0;
        self.backlash = Backlash::new(0.6f64.to_radians(), 0.0);
        self.observation()
    }

    fn seed_to_unit(seed: u64) -> f64 {
        let mut s = seed ^ 0x51_7C_C1_B7_27_22_0A_95;
        s = (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        s ^= s >> 31;
        (s as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
}

impl Env for Joint {
    fn reset(&mut self, seed: u64) -> Vec<f64> {
        let u = Self::seed_to_unit(seed);
        // A fresh joint: cold winding, full pack, a target 30 to 90 degrees away.
        self.theta = 0.0;
        self.omega = 0.0;
        self.theta_m = 0.0;
        self.omega_m = 0.0;
        self.target = u.signum() * (0.5 + 1.0 * u.abs());
        self.tau_prev = 0.0;
        self.steps = 0;
        self.thermal = MotorThermal::new(0.1, 1.5, 400.0, 1.2, 2.5, AMBIENT);
        self.battery = Battery::lithium(7200.0, 0.05, 24.0);
        self.backlash = Backlash::new(0.6f64.to_radians(), 0.0);
        self.motor.i_d = 0.0;
        self.motor.i_q = 0.0;
        self.last_current.set(0.0);
        self.tel = Telemetry { peak_winding: AMBIENT, ..Telemetry::default() };
        self.observation()
    }

    fn step(&mut self, action: &[f64]) -> StepResult {
        let tau_cmd = self.action_space().clamp(action)[0];

        let tau_load = if self.ideal { tau_cmd } else { self.through_chain(tau_cmd) };

        if self.ideal {
            // No drive, no gearbox: the commanded torque reaches the load directly.
            let err = Self::wrap_pi(self.theta - self.target);
            let _ = err;
            let accel =
                (tau_load - self.damping * self.omega - self.gravity * self.theta.sin()) / self.inertia;
            self.omega += accel * self.dt;
            self.theta += self.omega * self.dt;
        } else {
            // Two masses with a deadband BETWEEN THEM. `Backlash` compares its input against its own `output`
            // field, so that field must be the load angle for the deadband to be the physical one. My first
            // version called `update(theta_m)`, which let `output` become a filtered copy of the MOTOR angle —
            // the element was then measuring the motor against itself, the load was not in the loop at all, and
            // the reported time-in-gap (57% of the duty cycle) was an artifact of that.
            self.backlash.output = self.theta;
            let tau_t = self.backlash.transmitted_torque(self.theta_m, tau_load);
            if tau_t == 0.0 && tau_load != 0.0 {
                self.tel.in_backlash += 1;
            }
            let a_m = (tau_load - tau_t) / self.motor_inertia;
            self.omega_m += a_m * self.dt;
            self.theta_m += self.omega_m * self.dt;
            let a_l = (tau_t - self.damping * self.omega - self.gravity * self.theta.sin()) / self.inertia;
            self.omega += a_l * self.dt;
            self.theta += self.omega * self.dt;
        }

        self.tel.torque.push(tau_load);
        self.tau_prev = tau_cmd;
        self.steps += 1;

        let err = Self::wrap_pi(self.theta - self.target);
        // Position error and effort. Nothing about heat, energy or damage: that is the omission being measured.
        let reward = -(err * err) - 0.01 * (tau_cmd / self.torque_limit).powi(2);
        StepResult {
            observation: self.observation(),
            reward,
            terminated: false,
            truncated: self.steps >= self.max_steps,
        }
    }

    fn observation_space(&self) -> BoxSpace {
        // Generous bounds in physical units, since the normalizer rather than the space now sets the scale the
        // policy sees. The space still exists to document what the environment can report.
        BoxSpace::new(
            &[-std::f64::consts::PI, -60.0, -self.torque_limit],
            &[std::f64::consts::PI, 60.0, self.torque_limit],
        )
        .expect("valid observation space")
    }

    fn action_space(&self) -> BoxSpace {
        BoxSpace::symmetric(&[self.torque_limit]).expect("valid action space")
    }
}

impl Joint {
    fn observation(&self) -> Vec<f64> {
        // RAW PHYSICAL UNITS. An earlier version of this bench divided by pi and by 20 here, because feeding a
        // rate of order 30 alongside an error of order 1 saturates a tanh first layer on the rate alone and the
        // policy cannot see the error it is meant to null. That fix was in the wrong place: the constants were
        // invisible to the exported checkpoint, and changing a limit here would silently reinterpret a trained
        // policy. `ObsNorm` now learns the transform during training and ships it in the checkpoint, so an
        // environment's job is to report what it measures.
        vec![Self::wrap_pi(self.theta - self.target), self.omega, self.tau_prev]
    }
}

/// Evaluate a greedy policy on the real chain over several targets, returning averaged telemetry.
/// Evaluate a greedy policy, applying the SAME observation transform it trained under.
///
/// Passing raw observations to a policy trained on normalised ones would measure a domain mismatch rather than
/// the policy, which is the failure `to_deployable_normalized` exists to prevent on real hardware.
fn evaluate(policy: &GaussianPolicy, norm: &ObsNorm, episodes: u64) -> (f64, Telemetry, f64) {
    // PASCALS. The first version wrote 900.0 and 800.0, i.e. 900 Pa and 800 Pa, so every cycle exceeded the
    // ultimate strength and `damage` correctly returned infinity for all of them.
    let sn = SnCurve::basquin(900.0e6, 5.0).expect("valid S-N curve");
    let corr = MeanCorrection::Goodman { ultimate: 800.0e6 };
    let mut cost = 0.0;
    // ONE joint for the whole duty cycle: the winding stays hot and the pack stays drained between reaches.
    let mut env = Joint::new(false);
    env.reset(0);
    let max_steps = env.max_steps;
    for s in 0..episodes {
        env.next_reach(s);
        let mut obs = env.observation();
        let aspace = env.action_space();
        let mut ret = 0.0;
        for _ in 0..max_steps {
            let a = aspace.from_unit(&policy.mean(&norm.normalize(&obs)));
            let r = env.step(&a);
            ret += r.reward;
            let done = r.done();
            obs = r.observation;
            if done {
                break;
            }
        }
        cost += -ret;
    }
    // Nominal stress through a section modulus, so 40 N m reads as 200 MPa. The RATIO between the two policies
    // is what this supports; the absolute damage figure depends on this constant and is not a claim about any
    // real gearbox.
    let stress: Vec<f64> = env.tel.torque.iter().map(|t| t / 2.0e-7).collect();
    let damage = ferromotion_control::damage_from_history(&stress, &sn, corr);
    let n = episodes as f64;
    let mut agg = env.tel.clone();
    agg.joules /= n;
    (cost / n, agg, damage / n)
}

fn main() {
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
        max_episode_steps: 200,
        min_log_std: -2.5,
        log_std_ceiling: None,
        final_lr_fraction: 0.05,
    };
    let iterations = 60;

    println!("Actuator-aware RL: what a torque command costs\n");
    println!(
        "A 1-DOF position task. Both policies are [3, 32, 32, 1], seed 7, {iterations} PPO iterations of\n\
         {} steps. The reward contains position error and effort. It contains nothing about heat, energy or\n\
         fatigue. Both are then evaluated on the FULL chain.\n",
        cfg.steps_per_batch
    );

    let mut trained: Vec<(&str, GaussianPolicy, bool, f64, ObsNorm)> = Vec::new();
    for (label, ideal) in [("ideal torque source", true), ("full actuator chain", false)] {
        let mut env = Joint::new(ideal);
        let mut policy = GaussianPolicy::new(&[3, 32, 32, 1], 7, -1.0);
        let mut value = Mlp::new(&[3, 32, 32, 1], 7);
        let mut norm = ObsNorm::new(3);
        let reports =
            train_normalized(&mut env, &mut policy, &mut value, Some(&mut norm), &cfg, iterations, 7);
        let early: f64 = reports[..5].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;
        let late: f64 = reports[reports.len() - 5..].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;
        let improved = late > early;
        println!(
            "  trained against {label:<22} return {early:>9.2} -> {late:>9.2}  {}",
            if improved { "" } else { "  NO IMPROVEMENT" }
        );
        trained.push((label, policy, improved, late, norm));
    }

    println!("\nEvaluated on the full actuator chain, 12 targets each:\n");
    println!(
        "  {:<22} {:>10} {:>10} {:>9} {:>12} {:>8} {:>8} {:>8}",
        "trained against", "task cost", "joules", "peak degC", "damage/ep", "derate", "V-lim", "gap"
    );
    let mut rows = Vec::new();
    for (label, policy, improved, final_return, norm) in &trained {
        let (cost, tel, damage) = evaluate(policy, norm, 12);
        println!(
            "  {label:<22} {cost:>10.2} {:>10.1} {:>9.1} {damage:>12.3e} {:>8} {:>8} {:>8}",
            tel.joules, tel.peak_winding, tel.derated, tel.voltage_limited, tel.in_backlash
        );
        rows.push((*label, cost, tel.joules, damage, tel.derated, tel.voltage_limited, *improved, *final_return, tel.in_backlash, tel.peak_winding));
    }

    // A bench that cannot support its comparison must say so rather than print ratios. Three ways this
    // configuration can fail to measure anything, each checked:
    let binds = rows.iter().any(|r| r.4 > 0 || r.5 > 0);
    let finite = rows.iter().all(|r| r.3.is_finite() && r.3 > 0.0);
    let learned = rows.iter().all(|r| r.6);
    if !binds {
        println!(
            "\n  NO SATURATING LIMIT ENGAGED: no step was thermally derated or voltage-limited. The plants are\n\
             still not identical — backlash and the two-mass dynamics are active, see the `gap` column — but\n\
             the two limits this bench exists to exercise did not bind, so the columns above understate the\n\
             chain. Lengthen the duty cycle or raise the torque demand."
        );
    }
    if !finite {
        println!(
            "\n  DAMAGE COLUMN UNUSABLE: non-finite or zero. Non-finite means a cycle's mean stress reached\n\
             the Goodman ultimate, so check the stress conversion's units before believing any ratio."
        );
    }
    // The cross-policy comparison is NOT SUPPORTED by this design, and the reason is worth stating rather than
    // patching around. The two training returns are measured in DIFFERENT environments — the ideal plant is
    // easier, so its returns are higher by construction and the two numbers are not commensurable. Equalising
    // them would not fix it; there is no common scale on which "equally converged" is checkable here. A valid
    // sim-to-real transfer claim needs both policies trained to convergence and evaluated on one plant, which
    // is a different and larger experiment than this bench runs.
    //
    // So the ideal-trained row below is reported as CONTEXT, never as attribution.
    let parity = false;
    if !learned {
        println!(
            "\n  TRAINING DID NOT CONVERGE for at least one policy: its mean return did not improve over the\n\
             run. Comparing two policies that did not learn measures the initialisation, not the plant, so\n\
             the ratios below are withheld."
        );
    }
    let _ = parity;

    // THE RESULT THIS BENCH DOES SUPPORT, which needs no cross-policy comparison: what a policy that maximises
    // task reward alone does to the actuator it is driving.
    if binds && finite && learned {
        let chain = &rows[1];
        let total_steps = 12 * 200;
        let derate_frac = 100.0 * chain.4 as f64 / total_steps as f64;
        let gap_frac = 100.0 * chain.8 as f64 / total_steps as f64;
        let reps = if chain.3 > 0.0 { 1.0 / chain.3 } else { f64::INFINITY };
        println!("\n  The policy trained ON the chain, over a 12-reach duty cycle:\n");
        println!("    peak winding temperature   {:>10.1} degC   (foldback at {T_FOLDBACK}, limit {T_LIMIT})", chain.9);
        println!("    steps thermally derated    {derate_frac:>10.1} %");
        println!("    steps inside the backlash  {gap_frac:>10.1} %");
        println!("    energy per reach           {:>10.1} J", chain.2);
        println!("    reaches to fatigue failure {reps:>10.0}");
        println!(
            "\n  It maximised position error and effort. It was never told about the winding, the pack or the\n\
             gearbox, and it drove the drive into current foldback for {derate_frac:.0}% of the duty cycle — which\n\
             is authority the controller silently no longer had. `fatigue::damage` and `MotorThermal` put numbers\n\
             on that; the reward function cannot.\n\
             \n  The ideal-trained row above is CONTEXT ONLY. Its training return was measured on a different,\n\
             easier plant, so it is not commensurable with the chain-trained one and the difference in task cost\n\
             cannot be attributed to the plant. See the note in the source."
        );
    }
}
