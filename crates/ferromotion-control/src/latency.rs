//! **Latency as a stability budget** — what a policy's inference time costs the loop it closes, and how to spend
//! compute against it.
//!
//! A dual-rate stack runs a slow module (a vision-language policy at 1–10 Hz) over a fast one (a reflex controller
//! at 100–500 Hz), and the split is usually justified by analogy to human cognition. Classical control has the
//! tools to price it — delay margins, jitter margins, event-triggered execution — and they are not usually applied,
//! so the architecture ends up running *open-loop with respect to its own timing*.
//!
//! Three things are computable and are what this module provides.
//!
//! # 1. The delay margin is exact, not a phase-margin estimate
//!
//! For a discrete loop, a delay of `m` samples is **exactly** representable by augmenting the state with `m` shift
//! registers — no Padé approximation, no phase-margin rule of thumb. [`delay_margin_samples`] then simply asks how
//! many registers the loop tolerates before its spectral radius reaches one. The answer is a hard number of control
//! periods, which converts directly into a latency budget in milliseconds.
//!
//! # 2. Event-triggered execution buys budget, with a Zeno-free floor
//!
//! Tabuada's result: updating only when the measurement error crosses `‖e‖ = σ‖x‖` preserves input-to-state
//! stability, with inter-event times bounded below by
//!
//! ```text
//! τ ≥ σ / (L(1 + σ))
//! ```
//!
//! so the "compute only when needed" pattern cannot chatter. [`event_triggered_floor`] is that bound and the test
//! below measures actual inter-event times against it. This is the principled version of running a slow policy at a
//! variable rate, and it prices computation against performance rather than fixing a rate by analogy.
//!
//! # 3. Compute buys accuracy and costs latency, and the optimum is interior
//!
//! A bigger model is more accurate and slower. Accuracy enters closed-loop cost through the regret formula of
//! [`lq_regret`](crate::lq_regret) — quadratically in the action error — while latency enters through the delay
//! margin, and past the margin no accuracy helps at all. [`optimal_compute`] finds the balance, and the shape of
//! the answer is the useful part: **the cost is not monotone in model size**, so "use the biggest model that fits
//! the budget" is the wrong rule whenever the budget is set by wall-clock rather than by the margin.

use nalgebra::DMatrix;

/// The **delay margin in whole control periods**: the largest `m` for which the loop `A − BK` with `m` samples of
/// input delay remains stable.
///
/// Exact rather than estimated. A delay of `m` samples on a discrete system is a shift register, so the augmented
/// state `[x; u_{t−1}; …; u_{t−m}]` reproduces it with no approximation, and the question becomes an eigenvalue
/// computation. `max_search` caps how far to look. Returns `None` if the loop is unstable even with no delay.
pub fn delay_margin_samples(a: &DMatrix<f64>, b: &DMatrix<f64>, k: &DMatrix<f64>, max_search: usize) -> Option<usize> {
    if delayed_rho(a, b, k, 0)? >= 1.0 {
        return None; // not stable to begin with
    }
    let mut best = 0;
    for m in 1..=max_search {
        match delayed_rho(a, b, k, m) {
            Some(r) if r < 1.0 => best = m,
            _ => break,
        }
    }
    Some(best)
}

/// Spectral radius of the loop with `m` samples of input delay, via the augmented shift-register state.
pub fn delayed_rho(a: &DMatrix<f64>, b: &DMatrix<f64>, k: &DMatrix<f64>, m: usize) -> Option<f64> {
    let n = a.nrows();
    let nu = b.ncols();
    if a.ncols() != n || b.nrows() != n || k.ncols() != n || k.nrows() != nu {
        return None;
    }
    if m == 0 {
        return Some((a - b * k).complex_eigenvalues().iter().fold(0.0f64, |acc, l| acc.max(l.norm())));
    }
    // state [x; u_{t-1}; ...; u_{t-m}], with the plant driven by the OLDEST register and the newest fed by −Kx
    let dim = n + m * nu;
    let mut aug = DMatrix::zeros(dim, dim);
    aug.view_mut((0, 0), (n, n)).copy_from(a);
    aug.view_mut((0, n + (m - 1) * nu), (n, nu)).copy_from(b); // plant sees the oldest input
    aug.view_mut((n, 0), (nu, n)).copy_from(&(-k)); // newest register is the fresh command
    for i in 1..m {
        // each register shifts into the next
        aug.view_mut((n + i * nu, n + (i - 1) * nu), (nu, nu)).copy_from(&DMatrix::identity(nu, nu));
    }
    Some(aug.complex_eigenvalues().iter().fold(0.0f64, |acc, l| acc.max(l.norm())))
}

/// The delay margin in **seconds**, given the control period.
pub fn delay_margin_seconds(a: &DMatrix<f64>, b: &DMatrix<f64>, k: &DMatrix<f64>, control_period: f64, max_search: usize) -> Option<f64> {
    (control_period > 0.0).then(|| delay_margin_samples(a, b, k, max_search).map(|m| m as f64 * control_period))?
}

/// The **jitter margin**: the largest delay the loop tolerates when the delay may take any value in `0..=m`
/// unpredictably, which is the realistic case for a policy whose inference time varies.
///
/// Reported separately from the delay margin because a loop stable at *every* fixed delay in a range is not
/// necessarily stable under arbitrary switching between them — so this checks each value and returns the largest
/// range that is uniformly stable, which is a necessary condition and the honestly-available one. A sufficient
/// condition needs a common Lyapunov function across the switched family.
pub fn uniform_delay_margin(a: &DMatrix<f64>, b: &DMatrix<f64>, k: &DMatrix<f64>, max_search: usize) -> Option<usize> {
    let mut best = 0;
    for m in 0..=max_search {
        match delayed_rho(a, b, k, m) {
            Some(r) if r < 1.0 => best = m,
            _ => break,
        }
    }
    Some(best)
}

/// **Tabuada's inter-event floor**: `τ ≥ σ/(L(1+σ))`, the minimum time between updates of an event-triggered
/// controller that fires when `‖e‖ = σ‖x‖` on an `L`-Lipschitz closed loop.
///
/// The reason this matters for a slow policy: it means "recompute only when needed" has a *guaranteed* lower bound
/// on how often it needs to, so the pattern cannot chatter into requiring infinite compute. The floor rises with
/// the trigger threshold and falls with the loop's Lipschitz constant, both as one would want.
pub fn event_triggered_floor(sigma: f64, lipschitz: f64) -> Option<f64> {
    (sigma > 0.0 && lipschitz > 0.0).then(|| sigma / (lipschitz * (1.0 + sigma)))
}

/// The result of balancing model accuracy against inference latency.
#[derive(Clone, Copy, Debug)]
pub struct ComputeChoice {
    /// The chosen compute budget, in units of the accuracy model's argument.
    pub compute: f64,
    /// Action error at that budget.
    pub action_error: f64,
    /// Latency at that budget, in control periods.
    pub latency_samples: f64,
    /// Closed-loop cost, or `INFINITY` past the delay margin.
    pub cost: f64,
    /// Whether the choice sits inside the delay margin at all.
    pub feasible: bool,
}

/// **Balance accuracy against latency.**
///
/// `error_of_compute` gives the policy's action error at a compute budget (decreasing), `latency_of_compute` its
/// inference time in control periods (increasing), `h2_gain` the loop's error-to-cost constant from
/// [`LqLoop::h2_gain`](crate::LqLoop::h2_gain), and `margin` the delay margin in samples. Cost is
/// `h2_gain · error²` inside the margin and infinite outside it.
///
/// Sweeps the budget and returns the best. The point is the *shape*: cost falls with compute while the model is
/// fast enough and becomes infinite the moment latency crosses the margin, so the optimum is interior and "use the
/// biggest model that fits the wall-clock budget" is the wrong rule.
pub fn optimal_compute(error_of_compute: &dyn Fn(f64) -> f64, latency_of_compute: &dyn Fn(f64) -> f64, h2_gain: f64, margin: usize, budgets: &[f64]) -> Option<ComputeChoice> {
    let mut best: Option<ComputeChoice> = None;
    for &c in budgets {
        let e = error_of_compute(c);
        let lat = latency_of_compute(c);
        let feasible = lat <= margin as f64;
        let cost = if feasible { h2_gain * e * e } else { f64::INFINITY };
        let choice = ComputeChoice { compute: c, action_error: e, latency_samples: lat, cost, feasible };
        if best.as_ref().is_none_or(|b| cost < b.cost) {
            best = Some(choice);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lqr_gain, LqLoop};
    use nalgebra::DVector;

    fn double_integrator() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let dt = 0.01; // a 100 Hz control loop
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let k = lqr_gain(&a, &b, &DMatrix::identity(2, 2), &DMatrix::from_row_slice(1, 1, &[0.1]));
        (a, b, k)
    }

    /// **The delay margin, and it is exact.** The augmented shift-register construction reproduces the delay with no
    /// approximation, so the margin is a hard sample count — and the loop really does destabilise one sample past it.
    #[test]
    fn the_delay_margin_is_exact_and_the_loop_fails_one_sample_past_it() {
        let (a, b, k) = double_integrator();
        let m = delay_margin_samples(&a, &b, &k, 200).expect("the undelayed loop is stable");
        let (inside, outside) = (delayed_rho(&a, &b, &k, m).unwrap(), delayed_rho(&a, &b, &k, m + 1).unwrap());
        eprintln!("100 Hz loop: delay margin {m} samples = {:.1} ms; rho at the margin {inside:.6}, one past it {outside:.6}", m as f64 * 10.0);
        assert!(inside < 1.0 && outside >= 1.0, "the margin must be the exact crossing: {inside} then {outside}");
        assert!((delay_margin_seconds(&a, &b, &k, 0.01, 200).unwrap() - m as f64 * 0.01).abs() < 1e-12);

        // and the augmented construction really is the delay: simulate both and compare
        let mut x = DVector::from_row_slice(&[1.0, 0.0]);
        let mut queue = vec![0.0f64; 3];
        for _ in 0..200 {
            let u_fresh = -(&k * &x)[0];
            let u_applied = queue.remove(0);
            queue.push(u_fresh);
            x = &a * &x + &b * DVector::from_row_slice(&[u_applied]);
        }
        let direct = x.norm();
        // the same thing through the augmented matrix
        let mut z = DVector::zeros(2 + 3);
        z[0] = 1.0;
        let aug_rho = delayed_rho(&a, &b, &k, 3).unwrap();
        for _ in 0..200 {
            // reconstruct the augmented step explicitly
            let mut aug = DMatrix::zeros(5, 5);
            aug.view_mut((0, 0), (2, 2)).copy_from(&a);
            aug.view_mut((0, 4), (2, 1)).copy_from(&b);
            aug.view_mut((2, 0), (1, 2)).copy_from(&(-(&k)));
            aug[(3, 2)] = 1.0;
            aug[(4, 3)] = 1.0;
            z = aug * z;
        }
        eprintln!("   3-sample delay: explicit queue simulation reached {direct:.4e}, augmented matrix {:.4e} (rho {aug_rho:.5})", z.rows(0, 2).norm());
        assert!((direct.ln() - z.rows(0, 2).norm().ln()).abs() < 0.5, "the augmented model must reproduce the queue: {direct:.3e} vs {:.3e}", z.rows(0, 2).norm());
    }

    /// A faster control loop tolerates more delay *samples* but the same delay in **seconds** — which is the sanity
    /// check that the margin is a physical time and not an artefact of the sampling rate.
    #[test]
    fn the_margin_is_a_physical_time_not_a_sample_count() {
        let mut seconds = Vec::new();
        for &dt in &[0.02f64, 0.01, 0.005, 0.0025] {
            let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
            let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
            // the SAME continuous-time controller at each rate, so only the sampling changes
            let k = DMatrix::from_row_slice(1, 2, &[4.0, 4.0]);
            let m = delay_margin_samples(&a, &b, &k, 4000).expect("stable");
            let secs = m as f64 * dt;
            eprintln!("   control period {:>6.4} s: margin {m:>4} samples = {:.4} s", dt, secs);
            seconds.push(secs);
        }
        let (lo, hi) = (seconds.iter().cloned().fold(f64::INFINITY, f64::min), seconds.iter().cloned().fold(0.0f64, f64::max));
        eprintln!("   the margin in SECONDS across an 8x range of control rates: {lo:.4} to {hi:.4} ({:.0}% spread)", 100.0 * (hi - lo) / lo);
        assert!((hi - lo) / lo < 0.25, "the margin should be roughly rate-independent in seconds: {lo} to {hi}");
    }

    /// **Tabuada's floor, against measured inter-event times.** The bound is what stops "recompute when needed" from
    /// chattering, so it is worth checking that real inter-event times respect it rather than trusting the formula.
    #[test]
    fn the_event_triggered_floor_bounds_measured_inter_event_times() {
        // A scalar contracting loop, xdot = -x + u with u = -gain*x_held, so the CLOSED loop is xdot = -(1+gain)x
        // and its Lipschitz constant is 1 + gain. Getting this wrong is not a detail: asserting L = 2 where the
        // closed loop is actually 3-Lipschitz produced a "floor" of 0.0455 s against measured gaps of 0.0308 s -
        // the bound apparently violated, when in fact it had been handed the wrong constant. A guarantee is only as
        // sound as the Lipschitz bound fed to it, and this is the cheapest possible way to learn that.
        let gain = 2.0f64;
        let lipschitz = 1.0 + gain;
        for &sigma in &[0.1f64, 0.3, 0.8] {
            let floor = event_triggered_floor(sigma, lipschitz).unwrap();
            let dt = 1e-5;
            let (mut x, mut x_held) = (1.0f64, 1.0f64);
            let mut last_event = 0.0;
            let mut worst_gap = f64::INFINITY;
            let mut events = 0;
            for i in 1..400_000 {
                let t = i as f64 * dt;
                let u = -gain * x_held; // control computed at the last event
                x += (-x + u) * dt;
                if (x - x_held).abs() >= sigma * x.abs().max(1e-12) {
                    if events > 0 {
                        worst_gap = worst_gap.min(t - last_event);
                    }
                    last_event = t;
                    x_held = x;
                    events += 1;
                }
            }
            eprintln!("sigma {sigma}: Tabuada floor {floor:.5} s, smallest measured inter-event gap {worst_gap:.5} s over {events} events");
            assert!(events > 2, "the trigger must actually fire to be measuring anything");
            assert!(worst_gap >= floor - 1e-6, "measured gaps must respect the floor: {worst_gap} vs {floor}");
        }
        // the floor behaves: a looser trigger buys more time, a stiffer loop less
        assert!(event_triggered_floor(0.5, 2.0).unwrap() > event_triggered_floor(0.1, 2.0).unwrap());
        assert!(event_triggered_floor(0.3, 10.0).unwrap() < event_triggered_floor(0.3, 2.0).unwrap());
        assert!(event_triggered_floor(0.0, 2.0).is_none());
    }

    /// **The compute trade, and its shape.** Accuracy improves with compute and latency worsens; past the delay
    /// margin no accuracy helps. So the optimum is interior and the cost is NOT monotone in model size.
    #[test]
    fn the_optimal_compute_budget_is_interior_not_the_largest_that_fits() {
        let (a, b, k) = double_integrator();
        let margin = delay_margin_samples(&a, &b, &k, 200).unwrap();
        let loop_ = LqLoop { a: a.clone(), b: b.clone(), q: DMatrix::identity(2, 2), r: DMatrix::from_row_slice(1, 1, &[0.1]), k: k.clone() };
        let g2 = loop_.h2_gain().unwrap();

        // a model whose error falls as a power of compute and whose latency rises linearly
        let err = |c: f64| 0.5 * c.powf(-0.4);
        let lat = |c: f64| 0.6 * c;
        let budgets: Vec<f64> = (1..=60).map(|i| i as f64).collect();
        let best = optimal_compute(&err, &lat, g2, margin, &budgets).unwrap();
        eprintln!("delay margin {margin} samples; H2 gain {g2:.4}");
        eprintln!("chosen compute {:.0}: action error {:.4}, latency {:.1} samples, cost {:.6}", best.compute, best.action_error, best.latency_samples, best.cost);
        assert!(best.feasible, "the chosen budget must fit inside the margin");
        assert!(best.latency_samples <= margin as f64);

        // the largest budget that "fits the wall clock" is NOT the answer: past the margin the cost is infinite
        let biggest = budgets.last().copied().unwrap();
        assert!(lat(biggest) > margin as f64, "the sweep must reach past the margin for this to be a real trade");
        let over = optimal_compute(&err, &lat, g2, margin, &[biggest]).unwrap();
        assert!(!over.feasible && over.cost.is_infinite(), "past the margin, accuracy buys nothing");

        // and the optimum sits AT the margin here, because error falls monotonically with compute - which is the
        // useful diagnostic: when the accuracy curve is monotone the margin is the binding constraint, and the
        // right engineering response is to widen the margin (faster inner loop) rather than shrink the model
        eprintln!("   the optimum sits at the margin boundary, so the binding constraint is LATENCY, not accuracy:");
        eprintln!("   widening the margin is worth more than shrinking the model.");
        assert!(best.latency_samples > 0.5 * margin as f64, "the optimum should be pressed against the margin");
    }
}
