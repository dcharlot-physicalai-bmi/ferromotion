//! **R7.4 / M1 — does the predicted decomposition optimum survive simulation?**
//!
//! `decomposition.rs` derives an optimal granularity from a retry model. This runs the process it models: `k` skills
//! over a 500-step task, per-step failures, detection, containment checks on each handoff, and retries. Two questions:
//! does the closed form match the measured pass rate, and does the optimum move when retries are not free?

use ferromotion_control::{sampled_set_escape_rate, TaskDecomposition, Xorshift};

const HORIZON: usize = 500;
const Q: f64 = 0.002;
const DETECT: f64 = 0.9;
const DIM: usize = 4;
const BUDGET: usize = 20_000;
const TRIALS: usize = 200_000;

/// Run the task once. Returns `(succeeded, retries_used)`.
fn run_task(k: usize, monitored: bool, retry_cap: Option<usize>, rng: &mut Xorshift) -> (bool, usize) {
    let sub = HORIZON / k;
    let eps = sampled_set_escape_rate(DIM, BUDGET / k);
    let mut retries = 0usize;
    for _skill in 0..k {
        loop {
            // the skill runs its stretch; any step can fail
            let mut skill_failed = false;
            for _ in 0..sub {
                if rng.uniform() < Q {
                    skill_failed = true;
                    break;
                }
            }
            // the handoff can leave the downstream skill's certified inflow
            let breached = !skill_failed && rng.uniform() < eps;
            if !skill_failed && !breached {
                break; // this skill is done
            }
            // failed one way or the other - was it noticed?
            let detected = if breached { monitored } else { rng.uniform() < DETECT };
            if !detected {
                return (false, retries);
            }
            retries += 1;
            if retry_cap.is_some_and(|cap| retries > cap) {
                return (false, retries);
            }
        }
    }
    (true, retries)
}

fn measure(k: usize, monitored: bool, retry_cap: Option<usize>, rng: &mut Xorshift) -> (f64, f64) {
    let mut wins = 0usize;
    let mut total_retries = 0usize;
    for _ in 0..TRIALS {
        let (ok, r) = run_task(k, monitored, retry_cap, rng);
        wins += usize::from(ok);
        total_retries += r;
    }
    (wins as f64 / TRIALS as f64, total_retries as f64 / TRIALS as f64)
}

fn main() {
    let mut rng = Xorshift::new(0xD3C0_0000);
    let task = TaskDecomposition { horizon: HORIZON, per_step_failure: Q, detection: DETECT, dim: DIM, rollout_budget: BUDGET };

    println!("R7.4 / M1 - decomposition granularity, predicted vs simulated");
    println!("  {HORIZON} steps, per-step failure {Q}, detection {DETECT}, handoff dim {DIM}, {BUDGET} rollouts total");
    println!("  {TRIALS} simulated tasks per row, unlimited retries\n");

    for monitored in [false, true] {
        println!("  containment monitor: {}", if monitored { "ON" } else { "off" });
        println!("    {:>5}  {:>10}  {:>10}  {:>8}  {:>12}  {:>12}", "k", "predicted", "measured", "ratio", "retries pred", "retries meas");
        let mut worst = 0.0f64;
        for k in [1usize, 2, 4, 8, 16, 50, 100, 500] {
            let g = task.at(k, monitored).unwrap();
            let (m, r) = measure(k, monitored, None, &mut rng);
            println!("    {k:>5}  {:>10.4}  {m:>10.4}  {:>8.4}  {:>12.2}  {r:>12.2}", g.pass, m / g.pass, g.expected_retries);
            worst = worst.max((m - g.pass).abs());
        }
        // absolute deviation, because the sweep spans 0.89 down to 0.0000 and a ratio is meaningless at the bottom
        println!("    worst |predicted - measured| across the sweep: {worst:.5}");
        let best = task.optimal_granularity(500, monitored).unwrap();
        println!("    closed-form optimum: k = {} at {:.4}\n", best.skills, best.pass);
    }

    // --- retries are not free. A robot has finite time, so cap them.
    println!("  now with a RETRY CAP, which the closed form does not model:");
    for cap in [1usize, 3, 10] {
        println!("\n    retry cap {cap} (monitor ON):");
        println!("      {:>5}  {:>10}  {:>10}  {:>10}", "k", "uncapped", "capped", "loss");
        let mut best = (0usize, 0.0f64);
        for k in [1usize, 2, 4, 8, 16, 50, 100, 500] {
            let (unc, _) = measure(k, true, None, &mut rng);
            let (cap_m, _) = measure(k, true, Some(cap), &mut rng);
            println!("      {k:>5}  {unc:>10.4}  {cap_m:>10.4}  {:>10.4}", unc - cap_m);
            if cap_m > best.1 {
                best = (k, cap_m);
            }
        }
        println!("      best k under this cap: {} at {:.4}", best.0, best.1);
    }
}
