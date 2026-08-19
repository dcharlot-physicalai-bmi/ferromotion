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
//!   controller                     hold  peak height    effort    value head
//!   do nothing                     0.0%       -0.997      0.00    --
//!   constant max torque            0.0%       -0.942      1.00    --
//!   energy shaping + catch        74.6%        1.000      0.53    --
//!   PPO baseline                  19.8%        1.000      0.89    working
//!   PPO long horizon              18.4%        1.000      0.53    working
//!   PPO quiet (sigma/16)           0.0%       -0.886      0.12    INERT
//!   PPO annealed sigma             8.9%        1.000      0.71    INERT
//!   PPO state-dependent sd         7.5%        1.000      0.95    INERT
//!   PPO normalized obs            12.2%        1.000      0.91    INERT
//! ```
//!
//! **PPO swings up and does not catch.** It reaches the top on every seed, so it finds the energy pumping — the
//! genuinely non-linear part, requiring several swings, with no linear solution. It then fails to stabilise
//! there, holding 20% against the hand-built controller's 75%.
//!
//! # The largest single effect was a defect, not a hypothesis
//!
//! Every arm in the first five was measured with a **value head that had learned nothing**: an `Mlp` fitting raw
//! returns of this magnitude scores an MSE equal to the target variance, which is exactly what predicting the
//! mean scores. GAE was therefore differencing against a useless baseline throughout. See
//! `normalize_value_targets` in `ferromotion-learn`.
//!
//! Fixing it moved the baseline from **10.1% to 19.8%** hold — a larger improvement than anything the five
//! deliberate arms produced. The value function was named in an earlier version of this file as untested
//! *surface*, not as a prediction, and it turned out to hold the biggest available gain. It still does not close
//! the gap.
//!
//! **AND IT REFUTES THE ONE POSITIVE RESULT THIS BENCH REPORTED.** The longer horizon (`γ` 0.99 → 0.997) was
//! recorded as "a real effect and four times too small", from 10.1% → 17.7%. Re-measured on a working value
//! head: baseline 19.8%, long horizon **18.4%** — no benefit, slightly worse. The horizon was compensating for
//! the broken baseline, not helping the task. A contributor that disappears when an unrelated defect is fixed
//! was never a contributor.
//!
//! # What each arm still supports
//!
//! * **Baseline and long horizon** are re-measured on a working value head. The horizon hypothesis is refuted.
//! * **The other four arms are NOT re-measured** and were all run under the defect. Their single-variable
//!   structure is intact — the defect applied equally to each arm and its own baseline — but any conclusion drawn
//!   from them is a conclusion about a configuration containing a known fault. Specifically:
//!   * "Exploration noise prevents the catch" was refuted *backwards* (σ/16 never swings up at all, peak −0.886).
//!     That is a large qualitative effect and unlikely to be an artifact, but it is unverified on the fix.
//!   * The annealed-σ, state-dependent-σ and normalized-observation arms each landed within a few points of
//!     their contemporaneous baseline. Those margins are smaller than the 9.7-point shift the value fix produced,
//!     so **none of them can be trusted either way** until re-run.
//!
//! # The standing lesson
//!
//! Six predictions have now been written into this file before their measurement. All six were wrong: five arms
//! refuted, and the one effect reported as real turned out to be an artifact of a defect elsewhere in the loop.
//! The only thing that produced a material gain was a component nobody had looked at, found by measuring it
//! rather than by reasoning about the task.
//!
//! Still not varied: budget and architecture. 200 iterations of a `[3, 64, 64, 1]` net in `f64` is a real attempt
//! and not obviously a sufficient one, and that is recorded as surface rather than as a prediction.
//!
//! The reward is left alone throughout, at the standard `height − effort`. Paying specifically for dwell near the
//! top would solve the benchmark rather than measure the learner.
//!
//! `cargo run --release -p ferromotion-bench --example swingup`

use ferromotion_learn::{train, train_normalized, Env, GaussianPolicy, Mlp, ObsNorm, Pendulum, PpoConfig};

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

    // Three PPO configurations, each differing from the first in ONE variable, so that the two hypotheses the
    // first run left open are tested rather than argued about:
    //
    //  - Quieter exploration. `min_log_std = -2.0` is 0.27 N m of action noise against a 2 N m limit, and the
    //    inverted equilibrium can only be held within asin(tau_max/m g l) = 0.205 rad. If the noise is what
    //    prevents the catch, lowering it should produce one.
    //  - A longer horizon. gamma = 0.99 is a 100-step horizon against 500-step episodes, which may not
    //    distinguish balancing at the top from passing through it.
    //
    // Everything else is held fixed, including the reward, which stays the standard `height - effort`.
    let iterations = 200;
    let mut returns: Vec<f64> = Vec::new();
    // Each tuple is (name, init_log_std, min_log_std, gamma, log_std_ceiling).
    //
    // `init_log_std` is here because the first version of this ablation varied only `min_log_std`, which is a
    // FLOOR: the policy starts at the initial sigma and the floor binds only if the learned sigma descends to
    // it. It never did, so lowering the floor changed nothing and the "quiet" arm was BIT-IDENTICAL to the
    // baseline. The variable I thought I was changing was inert while the output claimed a tested hypothesis.
    //
    // The fourth arm tests what the first three pointed at: coarse noise is REQUIRED to discover the pumping
    // (the quiet arm proves it, at hold 0% and peak -0.886) and too coarse to refine the catch. The phases are
    // separated in time, so a σ CEILING annealed from loose to tight can serve both. Note it must be a ceiling:
    // a floor cannot push σ down, only stop it collapsing.
    // The fifth arm is the surviving candidate after the annealed schedule was refuted: sigma that reads the
    // STATE rather than the iteration, so it can be coarse while swinging and fine near the top within a single
    // episode. That is the axis the refutation pointed at.
    // The sixth arm is from OUTSIDE the exploration family, which the first five exhausted. The pendulum's
    // observation is `[sin θ, cos θ, ω]` on a space of [-1,-1,-32] to [1,1,32] — a 32:1 mismatch between the
    // trigonometric channels and the rate, which saturates a tanh first layer on the rate alone. The actuator
    // bench in this workspace measured that exact defect costing 31% of its final return, so this is an
    // evidence-backed hypothesis rather than another guess.
    let variants: [(&str, f64, f64, f64, Option<(f64, f64)>, bool, bool); 6] = [
        ("PPO baseline", -0.5, -2.0, 0.99, None, false, false),
        ("PPO quiet (sigma/16)", -2.5, -3.5, 0.99, None, false, false),
        ("PPO long horizon", -0.5, -2.0, 0.997, None, false, false),
        ("PPO annealed sigma", -0.5, -6.0, 0.99, Some((-0.5, -3.5)), false, false),
        ("PPO state-dependent sd", -0.5, -6.0, 0.99, None, true, false),
        ("PPO normalized obs", -0.5, -2.0, 0.99, None, false, true),
    ];
    // Each arm costs about 40 minutes, so all four do not fit in one sitting. Optional positional args select
    // a subset by index; with none given, all run. Arm 0 must be included for the verdict's baseline reference,
    // and it reproduces exactly from its seed, so re-running it is cheap insurance rather than waste.
    let selected: Vec<usize> = {
        let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
        if args.is_empty() {
            (0..variants.len()).collect()
        } else {
            assert!(args.contains(&0), "arm 0 is the baseline the verdict compares against; include it");
            assert!(args.iter().all(|&i| i < variants.len()), "arm index out of range");
            args
        }
    };
    if selected.len() < variants.len() {
        println!(
            "  running arms {:?} of {} ({} skipped)\n",
            selected,
            variants.len(),
            variants.len() - selected.len()
        );
    }
    for (name, init_log_std, min_log_std, gamma, ceiling, state_dep, normalize) in
        selected.iter().map(|&i| variants[i])
    {
        let cfg = PpoConfig {
            gamma,
            lambda: 0.95,
            clip: 0.2,
            policy_lr: 3e-3,
            value_lr: 3e-3,
            epochs: 8,
            value_epochs: 15,
            entropy_coef: 3e-3,
            steps_per_batch: 2000,
            max_episode_steps: STEPS,
            min_log_std,
            log_std_ceiling: ceiling,
            normalize_value_targets: true,
            final_lr_fraction: 0.05,
        };
        let mut env = Pendulum::default();
        let mut policy = if state_dep {
            GaussianPolicy::new_state_dependent(&[3, 64, 64, 1], 5, init_log_std)
                .expect("init_log_std in range")
        } else {
            GaussianPolicy::new(&[3, 64, 64, 1], 5, init_log_std)
        };
        let mut value = Mlp::new(&[3, 64, 64, 1], 5);
        let mut norm = ObsNorm::new(3);
        let reports = if normalize {
            train_normalized(&mut env, &mut policy, &mut value, Some(&mut norm), &cfg, iterations, 5)
        } else {
            train(&mut env, &mut policy, &mut value, &cfg, iterations, 5)
        };
        let early: f64 = reports[..5].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;
        let late: f64 = reports[reports.len() - 5..].iter().map(|r| r.mean_return).sum::<f64>() / 5.0;
        println!(
            "  {name:<22} {iterations} x {} steps, sigma {:.4}{}, gamma {gamma}, return {early:.1} -> {late:.1}",
            cfg.steps_per_batch,
            init_log_std.exp(),
            if normalize {
                " normalized obs".to_string()
            } else if state_dep {
                " state-dependent".to_string()
            } else {
                match ceiling {
                    Some((a, b)) => format!(" ceiling {:.3} -> {:.3}", a.exp(), b.exp()),
                    None => format!(" floor {:.4}", min_log_std.exp()),
                }
            }
        );
        let space = env.action_space();
        let scores: Vec<Score> = seeds
            .iter()
            .map(|&s| {
                run(s, |p| {
                    let raw = vec![p.theta.sin(), p.theta.cos(), p.omega];
                    // The SAME transform the arm trained under, or this measures a domain mismatch rather
                    // than the policy. `ObsNorm::normalize` is a passthrough when nothing was estimated, so
                    // the unnormalised arms are unaffected.
                    let obs = norm.normalize(&raw);
                    space.from_unit(&policy.mean(&obs))[0]
                })
            })
            .collect();
        returns.push(late);
        rows.push((name.to_string(), mean(&scores)));
    }
    println!();

    println!("  with PPO:\n");
    println!("  {:<24} {:>10} {:>12} {:>9}", "controller", "hold", "peak height", "effort");
    for (name, s) in &rows {
        println!("  {name:<24} {:>9.1}% {:>12.3} {:>9.2}", 100.0 * s.hold, s.peak_height, s.effort);
    }

    // The comparison the bench exists to make, stated either way.
    let shaped_hold = rows[2].1.hold;
    let shaped_peak = rows[2].1.peak_height;
    // Judge on the BEST PPO variant, and say which one it was, so a win cannot be attributed to the wrong
    // change and a loss cannot be blamed on a configuration that was not the best available.
    let best = rows[3..]
        .iter()
        .max_by(|a, b| a.1.hold.partial_cmp(&b.1.hold).expect("hold is finite"))
        .expect("at least one PPO variant");
    let ppo_hold = best.1.hold;
    let ppo_peak = best.1.peak_height;
    let baseline_hold = rows[3].1.hold;
    println!("  best PPO variant: {} (hold {:.1}%)\n", best.0, 100.0 * ppo_hold);

    // PROVE THE ARMS DIFFER before interpreting any of them. A variant whose training return is identical to
    // the baseline's did not vary anything, and reporting it as a tested hypothesis is worse than not running
    // it: the output asserts a negative result that was never measured.
    let mut inert = Vec::new();
    for i in 1..returns.len() {
        if (returns[i] - returns[0]).abs() < 1e-9 {
            inert.push(rows[3 + i].0.as_str());
        }
    }
    if !inert.is_empty() {
        println!(
            "  ARM(S) DID NOT VARY: {} produced a training return identical to the baseline, so whatever they\n\
             were meant to change was inert and those hypotheses are UNTESTED, not refuted.\n",
            inert.join(", ")
        );
    }
    if ppo_hold > baseline_hold + 0.1 {
        println!(
            "  ONE OF THE TWO HYPOTHESES HELD: '{}' improved the hold from {:.1}% to {:.1}%, so the gap was\n\
             not a limit of the learner but of that setting.\n",
            best.0,
            100.0 * baseline_hold,
            100.0 * ppo_hold
        );
    } else if inert.is_empty() {
        println!(
            "  NO HYPOTHESIS HELD: the best of {} variants reached {:.1}% against the baseline's {:.1}%, and\n\
             every arm is confirmed to have varied, so none of them explains the missing catch.\n",
            rows.len() - 3,
            100.0 * ppo_hold,
            100.0 * baseline_hold
        );
    } else {
        println!(
            "  NO CONCLUSION: the best variant reached {:.1}% against {:.1}%, but at least one arm was inert,\n\
             so the set of hypotheses tested is incomplete.\n",
            100.0 * ppo_hold,
            100.0 * baseline_hold
        );
    }
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
