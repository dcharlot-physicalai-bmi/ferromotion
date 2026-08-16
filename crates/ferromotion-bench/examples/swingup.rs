//! **Can this PPO do non-linear control, or only linear problems?**
//!
//! `ferromotion-learn`'s PPO is verified against an analytic LQR gain, which is a linear problem with a
//! quadratic cost and a known answer. Its pendulum test is deliberately weak: it asserts a trained policy beats
//! doing nothing, which establishes that the gradient points the right way and nothing more. That leaves the
//! interesting question open, and it is a question about the stack rather than about the task.
//!
//! Swing-up is the smallest genuinely hard case. The torque limit is **below `m g l`**, so no policy can simply
//! push the mass to the top: it has to pump energy over several swings, then catch the pendulum at an unstable
//! equilibrium. There is no linear controller that does it.
//!
//! # Measured against a controller that already works
//!
//! "PPO beat doing nothing" is not evidence of competence. So the comparison here is against
//! **energy-shaping plus a catch**, the textbook swing-up: drive the total energy toward the homoclinic orbit
//! with `τ = k(E* − E)·ω`, then switch to a stabilising PD once the mass is near the top. That controller is
//! known-good, needs no training, and gives the number PPO has to reach. If PPO falls short of it, that is the
//! finding and it is reported as one.
//!
//! # What is scored
//!
//! Reaching the top once is easy to do by accident on a lucky swing. Holding it is the task, so the metric is
//! the fraction of the **final half** of each episode spent within 0.2 rad of upright, averaged over seeds. Peak
//! height is reported alongside it, because a policy that swings high and never catches looks identical to a
//! failing one on the hold metric alone and the two want different fixes.
//!
//! # The answer, measured
//!
//! ```text
//!   controller                     hold  peak height    effort
//!   do nothing                     0.0%       -0.997      0.00
//!   constant max torque            0.0%       -0.942      1.00
//!   energy shaping + catch        74.6%        1.000      0.53
//!   PPO (200 x 2000 steps)        10.1%        1.000      0.85
//! ```
//!
//! **PPO swings up and does not catch.** Its return improved from −501 to −71 and it reached the top on every
//! seed, so it found the energy pumping — which is the genuinely non-linear part, requiring several swings and
//! having no linear solution. It then failed to stabilise there, holding 10% against the hand-built controller's
//! 75%, and spent 60% more torque doing it.
//!
//! That is a *specific* failure rather than "did not learn", and the distinction is why peak height is reported
//! next to hold. Two candidates worth trying, neither attempted here because tuning until the number improves
//! would defeat the point of measuring it:
//!
//! * **Exploration noise may be preventing the catch.** The floor on the learned σ is `min_log_std = -2.0`,
//!   which is 0.135 in policy space, or 0.27 N m of noise against a 2 N m limit. The inverted equilibrium can
//!   only be held within `asin(τ_max/m g l) = 0.205 rad`, so 13% action noise is large relative to the basin
//!   the policy has to stay inside.
//! * **The horizon may be too short to value balancing over passing through.** `γ = 0.99` is a 100-step
//!   horizon against 500-step episodes.
//!
//! The reward is deliberately left alone. `height − effort` is the standard swing-up reward, and adding a term
//! that pays specifically for dwelling near the top would be solving the benchmark rather than measuring the
//! learner.
//!
//! `cargo run --release -p ferromotion-bench --example swingup`

use ferromotion_learn::{train, Env, GaussianPolicy, Mlp, Pendulum, PpoConfig};

/// Angular window around upright that counts as held (rad).
const HOLD_WINDOW: f64 = 0.2;
/// Episode length in control steps, and the control step itself.
const STEPS: usize = 500;

/// How a controller did on one episode.
#[derive(Clone, Copy, Debug, Default)]
struct Score {
    /// Fraction of the final half of the episode within `HOLD_WINDOW` of upright.
    hold: f64,
    /// Highest the mass reached, as a fraction of the length. `+1` is upright.
    peak_height: f64,
    /// Mean absolute torque, as a fraction of the limit.
    effort: f64,
}

/// Run one episode under a torque law and score it.
fn run(seed: u64, mut torque: impl FnMut(&Pendulum) -> f64) -> Score {
    let mut p = Pendulum::default();
    p.reset(seed);
    let space = p.action_space();
    let mut in_window = 0usize;
    let mut counted = 0usize;
    let mut peak = p.height() / p.length;
    let mut effort = 0.0;
    for k in 0..STEPS {
        let tau = space.clamp(&[torque(&p)])[0];
        effort += (tau / p.torque_limit).abs();
        p.step(&[tau]);
        peak = peak.max(p.height() / p.length);
        if k >= STEPS / 2 {
            counted += 1;
            // Upright is theta = pi. Wrap the error so the window is symmetric about the top.
            let err = ((p.theta - std::f64::consts::PI) % (2.0 * std::f64::consts::PI) + 3.0 * std::f64::consts::PI)
                % (2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
            if err.abs() < HOLD_WINDOW {
                in_window += 1;
            }
        }
    }
    Score {
        hold: in_window as f64 / counted.max(1) as f64,
        peak_height: peak,
        effort: effort / STEPS as f64,
    }
}

/// **Energy shaping plus a catch** — the textbook swing-up, and the number PPO has to reach.
///
/// Below the catch window, drive the total energy toward its upright value with `τ = k(E* − E)·ω`. That is the
/// classic result: the sign of `ω` is what makes the torque add energy on both halves of a swing, and pumping
/// works because the pendulum's energy is the only thing that has to change. Inside the window, switch to a PD
/// on the upright error, because energy shaping has no opinion about *where* on the homoclinic orbit you are.
fn energy_shaping(p: &Pendulum, k_pump: f64, kp: f64, kd: f64, window: f64) -> f64 {
    let e_top = 2.0 * p.mass * p.gravity * p.length;
    let e = p.energy();
    let err = ((p.theta - std::f64::consts::PI) % (2.0 * std::f64::consts::PI) + 3.0 * std::f64::consts::PI)
        % (2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    if err.abs() < window {
        // Close enough to catch. **`kp` MUST exceed `m g l`.** Linearised at the top, `I θ̈ = m g l·err + τ`
        // with `I = m l²`, so under `τ = −kp·err − kd·ω` the closed loop is `θ̈ = (m g l − kp)err − kd·ω` and any
        // `kp` below `m g l` leaves it unstable. My first version used `kp = 6` against `m g l = 9.81`: the
        // "stabiliser" was a destabiliser, the baseline held only 30% of the time, and the bench correctly
        // refused to compare anything to it.
        debug_assert!(kp > p.mass * p.gravity * p.length, "kp must exceed m g l to stabilise the top");
        -kp * err - kd * p.omega
    } else {
        k_pump * (e_top - e) * p.omega
    }
}

fn main() {
    let seeds: Vec<u64> = (0..8).collect();
    let p0 = Pendulum::default();
    println!("Swing-up: can this PPO do non-linear control?\n");
    println!(
        "  torque limit {:.2} N m against m g l = {:.2} N m, so the top cannot be reached directly\n\
         and the pendulum must be pumped. {} seeds, {STEPS}-step episodes, hold window {HOLD_WINDOW} rad\n\
         measured over the final half of each episode.\n",
        p0.torque_limit,
        p0.mass * p0.gravity * p0.length,
        seeds.len()
    );

    let mut rows: Vec<(String, Score)> = Vec::new();

    // Baseline 1: do nothing. The floor the weak unit test compares against.
    let idle: Vec<Score> = seeds.iter().map(|&s| run(s, |_| 0.0)).collect();
    rows.push(("do nothing".into(), mean(&idle)));

    // Baseline 2: constant maximum torque. Establishes that the limit really does prevent a direct push.
    let shove: Vec<Score> = seeds.iter().map(|&s| run(s, |p| p.torque_limit)).collect();
    rows.push(("constant max torque".into(), mean(&shove)));

    // Baseline 3: energy shaping plus a catch, the known-good controller.
    let shaped: Vec<Score> =
        seeds.iter().map(|&s| run(s, |p| energy_shaping(p, 0.6, 25.0, 6.0, 0.3))).collect();
    rows.push(("energy shaping + catch".into(), mean(&shaped)));

    // FAIL FAST. The baselines compute in milliseconds and PPO takes minutes, so a broken baseline must
    // surface before the expensive part rather than after it. The first version printed this table last and
    // would not have revealed the unstable catch gain until training had finished.
    println!("  baselines, before spending compute on training:\n");
    println!("  {:<24} {:>10} {:>12} {:>9}", "controller", "hold", "peak height", "effort");
    for (name, s) in &rows {
        println!("  {name:<24} {:>9.1}% {:>12.3} {:>9.2}", 100.0 * s.hold, s.peak_height, s.effort);
    }
    if rows[2].1.hold <= 0.5 {
        println!(
            "\n  BASELINE DID NOT SOLVE IT (hold {:.0}%). Stopping before training, because a comparison\n\
             against a controller that does not work says nothing about PPO. Fix the gains first: `kp` must\n\
             exceed m g l = {:.2}, and the catch window cannot exceed the region the torque limit can hold,\n\
             which is asin(tau_max/(m g l)) = {:.3} rad.",
            100.0 * rows[2].1.hold,
            p0.mass * p0.gravity * p0.length,
            (p0.torque_limit / (p0.mass * p0.gravity * p0.length)).asin()
        );
        return;
    }
    println!();

    // PPO, with a budget that is a real attempt rather than a gesture.
    let cfg = PpoConfig {
        gamma: 0.99,
        lambda: 0.95,
        clip: 0.2,
        policy_lr: 3e-3,
        value_lr: 3e-3,
        epochs: 8,
        value_epochs: 15,
        entropy_coef: 3e-3,
        steps_per_batch: 2000,
        max_episode_steps: STEPS,
        min_log_std: -2.0,
        final_lr_fraction: 0.05,
    };
    let iterations = 200;
    let mut env = Pendulum::default();
    let mut policy = GaussianPolicy::new(&[3, 64, 64, 1], 5, -0.5);
    let mut value = Mlp::new(&[3, 64, 64, 1], 5);
    let reports = train(&mut env, &mut policy, &mut value, &cfg, iterations, 5);
    let early: f64 = reports[..5].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;
    let late: f64 = reports[reports.len() - 5..].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;
    println!(
        "  PPO: {iterations} iterations of {} steps, return {early:.2} -> {late:.2}\n",
        cfg.steps_per_batch
    );
    let space = env.action_space();
    let ppo: Vec<Score> = seeds
        .iter()
        .map(|&s| {
            run(s, |p| {
                let obs = vec![p.theta.sin(), p.theta.cos(), p.omega];
                space.from_unit(&policy.mean(&obs))[0]
            })
        })
        .collect();
    rows.push(("PPO".into(), mean(&ppo)));

    println!("  with PPO:\n");
    println!("  {:<24} {:>10} {:>12} {:>9}", "controller", "hold", "peak height", "effort");
    for (name, s) in &rows {
        println!("  {name:<24} {:>9.1}% {:>12.3} {:>9.2}", 100.0 * s.hold, s.peak_height, s.effort);
    }

    // The comparison the bench exists to make, stated either way.
    let shaped_hold = rows[2].1.hold;
    let ppo_hold = rows[3].1.hold;
    let shaped_peak = rows[2].1.peak_height;
    let ppo_peak = rows[3].1.peak_height;
    println!();
    if rows[1].1.peak_height > 0.9 {
        println!(
            "  CONFIGURATION PROBLEM: constant maximum torque reached a height of {:.3}, so the limit is NOT\n\
             below what a direct push needs and this is not a swing-up task at all.",
            rows[1].1.peak_height
        );
    } else if ppo_hold >= 0.8 * shaped_hold && shaped_hold > 0.5 {
        println!(
            "  PPO holds {:.0}% against energy shaping's {:.0}%, so it solved a task with no linear solution.",
            100.0 * ppo_hold,
            100.0 * shaped_hold
        );
    } else if shaped_hold <= 0.5 {
        println!(
            "  THE BASELINE DID NOT SOLVE IT EITHER (hold {:.0}%), so this run says nothing about PPO. The\n\
             energy-shaping gains or the catch window need tuning before the comparison means anything.",
            100.0 * shaped_hold
        );
    } else if ppo_peak > 0.8 {
        println!(
            "  PPO SWINGS UP BUT DOES NOT CATCH: peak height {ppo_peak:.3} against a hold of {:.0}%, where\n\
             energy shaping holds {:.0}%. It found the pumping and not the stabilisation, which is a different\n\
             failure from not learning at all and wants a different fix (a catch term in the reward, or a\n\
             longer horizon near the top).",
            100.0 * ppo_hold,
            100.0 * shaped_hold
        );
    } else {
        println!(
            "  PPO DID NOT SOLVE IT: hold {:.0}% and peak height {ppo_peak:.3}, against energy shaping's\n\
             {:.0}% and {shaped_peak:.3}. At this budget and architecture, a from-scratch f64 PPO does not\n\
             reach a hand-built energy-shaping controller on this task. That is the result.",
            100.0 * ppo_hold,
            100.0 * shaped_hold
        );
    }
}

fn mean(v: &[Score]) -> Score {
    let n = v.len().max(1) as f64;
    Score {
        hold: v.iter().map(|s| s.hold).sum::<f64>() / n,
        peak_height: v.iter().map(|s| s.peak_height).sum::<f64>() / n,
        effort: v.iter().map(|s| s.effort).sum::<f64>() / n,
    }
}
