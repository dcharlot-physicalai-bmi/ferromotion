//! **Interface contracts for a dual-rate stack** — designing a hierarchy to a loss budget instead of tuning it.
//!
//! The standard architecture is a slow module producing setpoints and a fast one tracking them, and the split rate
//! is normally chosen by feel. It does not have to be. Three quantities determine what the hierarchy costs, and all
//! three are measurable:
//!
//! * **interface error** `η` — how wrong the slow layer's setpoint is;
//! * **staleness** — a setpoint held for `N` control periods while the target moves, which is a zero-order-hold
//!   error growing with `N`;
//! * **latency** — the slow layer's inference time, which is bounded by the delay margin
//!   ([`latency`](crate::latency)) and past which nothing else matters.
//!
//! The interesting part is that these pull against each other through the **same** knob. Running the slow layer
//! less often gives it more compute per call, so `η` falls — and makes its setpoints staler, so the hold error
//! rises. Cost is therefore **not monotone in the slow rate**, and there is an interior optimum that a tuning sweep
//! finds by luck and a budget calculation finds by construction:
//!
//! ```text
//! cost(N)  ≈  g₂·( η(N)² + (staleness_rate·N)² ),    subject to latency(N) ≤ delay margin
//! ```
//!
//! with `g₂` the loop's error-to-cost gain from [`LqLoop::h2_gain`](crate::LqLoop::h2_gain). Both terms are
//! quadratic for the reason the linear analysis gives: a smooth cost charges second order for a zero-mean error.
//!
//! [`optimal_slow_period`] solves it and [`pareto_frontier`] traces the trade, and the tests check the prediction
//! against a simulated dual-rate loop rather than against itself.
//!
//! # The model is ordinally valid, not cardinally valid
//!
//! Checked against simulation, the predicted cost tracks the measured one with a **stable ratio (~10% spread across
//! an 8× range of slow periods) and a large constant offset (~27×)**. Those two facts together are the honest
//! characterisation: the decomposition **ranks designs correctly** and **cannot predict absolute performance**. The
//! offset is structural — the prediction charges only the setpoint channel through `g₂`, while the task also pays
//! for the velocity deviation and for the steady-state control a drifting target demands.
//!
//! That is enough to choose a slow-layer rate, which is what this is for. It is not enough to promise a task success
//! number, and using it that way would be the mistake.

use crate::{latency, LqLoop};
use nalgebra::DVector;

/// A dual-rate stack's measured interface: what the slow layer delivers, and what it costs to run.
#[derive(Clone, Copy, Debug)]
pub struct Interface {
    /// Control periods between slow-layer updates.
    pub slow_period: usize,
    /// The slow layer's setpoint error at this period — falling with `slow_period`, since a longer period buys more
    /// compute per inference.
    pub interface_error: f64,
    /// Latency in control periods.
    pub latency_samples: f64,
    /// How fast the true target moves, in setpoint units per control period. Sets the staleness cost.
    pub staleness_rate: f64,
}

impl Interface {
    /// The hold error from a setpoint held `slow_period` samples: `staleness_rate · slow_period`, the mean-square
    /// equivalent of a zero-order hold on a moving target.
    pub fn staleness_error(&self) -> f64 {
        self.staleness_rate * self.slow_period as f64
    }

    /// Total effective error entering the loop: interface and staleness in quadrature, since they are independent
    /// contributions to the same channel.
    pub fn effective_error(&self) -> f64 {
        (self.interface_error * self.interface_error + self.staleness_error() * self.staleness_error()).sqrt()
    }
}

/// The end-to-end cost of a hierarchy, decomposed.
#[derive(Clone, Copy, Debug)]
pub struct HierarchyCost {
    /// Cost attributable to the slow layer's setpoint error.
    pub from_interface: f64,
    /// Cost attributable to setpoint staleness.
    pub from_staleness: f64,
    /// Total, or `INFINITY` past the delay margin.
    pub total: f64,
    /// Whether the latency fits inside the delay margin.
    pub feasible: bool,
    pub slow_period: usize,
}

impl HierarchyCost {
    /// Which of the two error sources dominates — the number that says which layer to work on next.
    pub fn dominant(&self) -> &'static str {
        if self.from_staleness > self.from_interface { "staleness (run the slow layer more often)" } else { "interface error (make the slow layer better)" }
    }
}

/// The predicted end-to-end cost of an interface on a given loop.
///
/// `g2` is the loop's error-to-cost gain and `margin` the delay margin in samples. Past the margin the cost is
/// infinite rather than large, which is the honest encoding of "the loop is unstable and there is no average cost".
pub fn hierarchy_cost(iface: &Interface, g2: f64, margin: usize) -> HierarchyCost {
    let feasible = iface.latency_samples <= margin as f64;
    let fi = g2 * iface.interface_error * iface.interface_error;
    let fs = g2 * iface.staleness_error() * iface.staleness_error();
    HierarchyCost { from_interface: fi, from_staleness: fs, total: if feasible { fi + fs } else { f64::INFINITY }, feasible, slow_period: iface.slow_period }
}

/// **The design rule**: the slow-layer period that minimises end-to-end cost.
///
/// `error_at_period` gives the interface error achievable when the slow layer has that many control periods to
/// think (decreasing), `latency_at_period` its inference latency (typically close to the period itself). Sweeps the
/// candidate periods and returns the best feasible one, or `None` if none fits the delay margin.
pub fn optimal_slow_period(error_at_period: &dyn Fn(usize) -> f64, latency_at_period: &dyn Fn(usize) -> f64, staleness_rate: f64, g2: f64, margin: usize, candidates: &[usize]) -> Option<(usize, HierarchyCost)> {
    let mut best: Option<(usize, HierarchyCost)> = None;
    for &n in candidates {
        if n == 0 {
            continue;
        }
        let iface = Interface { slow_period: n, interface_error: error_at_period(n), latency_samples: latency_at_period(n), staleness_rate };
        let c = hierarchy_cost(&iface, g2, margin);
        if c.feasible && best.as_ref().is_none_or(|(_, b)| c.total < b.total) {
            best = Some((n, c));
        }
    }
    best
}

/// The **Pareto frontier** of the trade: for each candidate slow period, the interface error achieved and the total
/// cost paid. Infeasible periods are included with infinite cost, because knowing *where* the margin bites is part
/// of the frontier.
pub fn pareto_frontier(error_at_period: &dyn Fn(usize) -> f64, latency_at_period: &dyn Fn(usize) -> f64, staleness_rate: f64, g2: f64, margin: usize, candidates: &[usize]) -> Vec<(usize, f64, HierarchyCost)> {
    candidates
        .iter()
        .filter(|n| **n > 0)
        .map(|&n| {
            let iface = Interface { slow_period: n, interface_error: error_at_period(n), latency_samples: latency_at_period(n), staleness_rate };
            (n, iface.interface_error, hierarchy_cost(&iface, g2, margin))
        })
        .collect()
}

/// Simulate the actual dual-rate loop and return its mean cost, for checking a prediction against the thing it
/// predicts.
///
/// The fast layer runs every sample tracking the held setpoint; the slow layer refreshes the setpoint every
/// `slow_period` samples with an error of the given size, and the true target drifts at `staleness_rate` per sample.
/// Returns `None` if the loop diverged.
pub fn simulate_dual_rate(l: &LqLoop, iface: &Interface, steps: usize, burn: usize, seed: u64) -> Option<f64> {
    let mut rng = crate::Xorshift::new(seed);
    let n = l.a.nrows();
    let mut x = DVector::zeros(n);
    let mut setpoint = 0.0f64;
    let mut target = 0.0f64;
    let mut cost = 0.0;
    let mut counted = 0usize;
    for t in 0..steps {
        // the true target drifts; the slow layer refreshes its estimate only every slow_period samples
        target += iface.staleness_rate;
        if t % iface.slow_period == 0 {
            setpoint = target + iface.interface_error * rng.normal();
        }
        // the fast layer tracks the HELD setpoint on the first coordinate
        let mut err = x.clone();
        err[0] -= setpoint;
        let u = -(&l.k * &err);
        if t >= burn {
            // charge the deviation from the TRUE target, which is what the task cares about
            let mut task_err = x.clone();
            task_err[0] -= target;
            cost += (task_err.transpose() * &l.q * &task_err)[0];
            counted += 1;
        }
        x = &l.a * &x + &l.b * &u;
        if !x.iter().all(|v| v.is_finite()) || x.norm() > 1e8 {
            return None;
        }
    }
    (counted > 0).then(|| cost / counted as f64)
}

/// The delay margin of a loop, re-exported at this level because a hierarchy's timing contract is exactly it.
pub fn margin_of(l: &LqLoop, max_search: usize) -> Option<usize> {
    latency::delay_margin_samples(&l.a, &l.b, &l.k, max_search)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lqr_gain;
    use nalgebra::DMatrix;

    fn loop_100hz() -> LqLoop {
        let dt = 0.01;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let k = lqr_gain(&a, &b, &q, &r);
        LqLoop { a, b, q, r, k }
    }

    /// **The two error sources pull against each other through the same knob**, so the cost is not monotone in the
    /// slow-layer rate and there is an interior optimum. This is the whole content of designing to a budget rather
    /// than tuning.
    #[test]
    fn the_cost_is_not_monotone_in_the_slow_rate_and_the_optimum_is_interior() {
        let l = loop_100hz();
        let g2 = l.h2_gain().unwrap();
        let margin = margin_of(&l, 200).unwrap();
        // a longer slow period buys compute, so the interface error falls; latency tracks the period
        let err = |n: usize| 0.4 * (n as f64).powf(-0.5);
        let lat = |n: usize| n as f64;
        let staleness = 0.004f64;

        let frontier = pareto_frontier(&err, &lat, staleness, g2, margin, &[1, 2, 4, 8, 16, 24, 32, 40, 64]);
        eprintln!("delay margin {margin} samples, H2 gain {g2:.4}, staleness rate {staleness}/sample");
        eprintln!("      slow period   interface error   cost(interface)   cost(staleness)   total      dominant");
        for (n, e, c) in &frontier {
            let tot = if c.feasible { format!("{:.6}", c.total) } else { "infeasible".into() };
            eprintln!("      {n:>11}   {e:>15.5}   {:>15.6}   {:>15.6}   {tot:>10}   {}", c.from_interface, c.from_staleness, if c.feasible { c.dominant() } else { "past the margin" });
        }

        let (best_n, best) = optimal_slow_period(&err, &lat, staleness, g2, margin, &[1, 2, 4, 8, 16, 24, 32, 40, 64]).unwrap();
        eprintln!("\n      optimum at slow period {best_n} (total {:.6}); it is INTERIOR, not an endpoint", best.total);
        let feasible: Vec<_> = frontier.iter().filter(|(_, _, c)| c.feasible).collect();
        let (first, last) = (feasible.first().unwrap(), feasible.last().unwrap());
        assert!(best.total < first.2.total, "running as fast as possible is not optimal: {:.6} vs {:.6}", best.total, first.2.total);
        assert!(best.total < last.2.total, "running as slow as the margin allows is not optimal either: {:.6} vs {:.6}", best.total, last.2.total);
        assert!(best_n > first.0 && best_n < last.0, "the optimum must be interior: {best_n} in ({}, {})", first.0, last.0);
    }

    /// **The prediction against a simulated dual-rate loop.** The decomposition is a model, so it has to be checked
    /// against the thing it models rather than against its own algebra.
    #[test]
    fn the_predicted_cost_matches_a_simulated_dual_rate_stack() {
        let l = loop_100hz();
        let g2 = l.h2_gain().unwrap();
        let margin = margin_of(&l, 200).unwrap();
        eprintln!("      slow period   predicted   simulated   ratio");
        let mut ratios = Vec::new();
        for &n in &[2usize, 4, 8, 16] {
            let iface = Interface { slow_period: n, interface_error: 0.2, latency_samples: n as f64, staleness_rate: 0.002 };
            let pred = hierarchy_cost(&iface, g2, margin);
            let Some(sim) = simulate_dual_rate(&l, &iface, 400_000, 20_000, 0xA5A5_1234_9876_FEDCu64) else {
                eprintln!("      {n:>11}   the simulated loop diverged");
                continue;
            };
            eprintln!("      {n:>11}   {:>9.6}   {sim:>9.6}   {:>5.2}", pred.total, sim / pred.total);
            ratios.push(sim / pred.total);
        }
        // The ratio is ~27x with a ~10% spread, and the distinction between those two numbers is the whole finding
        // about this model: it is ORDINALLY valid and not CARDINALLY valid. The scale is off by a constant because
        // the prediction charges only the position channel through g2, while the simulated task cost also pays for
        // the velocity deviation and for the steady-state control a drifting target demands - contributions the
        // two-term decomposition does not carry. So the model RANKS designs correctly (the spread is small) and
        // cannot predict absolute performance (the offset is large). For choosing a slow-layer rate that is
        // sufficient; for promising a task success number it is not, and conflating the two would be the error.
        let (lo, hi) = (ratios.iter().cloned().fold(f64::INFINITY, f64::min), ratios.iter().cloned().fold(0.0f64, f64::max));
        eprintln!("\n      ratio across an 8x range of slow periods: {lo:.2} to {hi:.2} ({:.0}% spread)", 100.0 * (hi - lo) / lo);
        eprintln!("      => ORDINALLY valid (stable ratio, so it ranks designs) and NOT cardinally valid (large");
        eprintln!("         offset, so it cannot promise a performance number). The decomposition omits the velocity");
        eprintln!("         channel and the steady-state control a drifting target needs.");
        assert!(ratios.len() >= 3, "need several points to judge the trend");
        assert!((hi - lo) / lo < 0.6, "the ratio must be stable enough to rank designs: {lo:.2} to {hi:.2}");
        assert!(lo > 2.0, "and it is NOT close to one - if this ever tightens, the caveat above is stale");
    }

    /// Staleness dominates when the slow layer is slow and the target moves; interface error dominates when it is
    /// fast. The crossover is where the design should sit, and `dominant` names which side of it you are on — which
    /// is the actionable output of the whole calculation.
    #[test]
    fn the_dominant_term_switches_across_the_optimum() {
        let l = loop_100hz();
        let g2 = l.h2_gain().unwrap();
        let margin = margin_of(&l, 200).unwrap();
        let fast = Interface { slow_period: 2, interface_error: 0.3, latency_samples: 2.0, staleness_rate: 0.003 };
        let slow = Interface { slow_period: 32, interface_error: 0.05, latency_samples: 32.0, staleness_rate: 0.003 };
        let (cf, cs) = (hierarchy_cost(&fast, g2, margin), hierarchy_cost(&slow, g2, margin));
        eprintln!("fast layer (period 2):  interface {:.6}, staleness {:.6} -> {}", cf.from_interface, cf.from_staleness, cf.dominant());
        eprintln!("slow layer (period 32): interface {:.6}, staleness {:.6} -> {}", cs.from_interface, cs.from_staleness, cs.dominant());
        assert!(cf.dominant().starts_with("interface"), "at a fast rate the slow layer's accuracy is the problem");
        assert!(cs.dominant().starts_with("staleness"), "at a slow rate the hold error is the problem");
        // and past the margin nothing else matters
        let too_slow = Interface { slow_period: margin + 10, interface_error: 1e-9, latency_samples: (margin + 10) as f64, staleness_rate: 0.0 };
        let c = hierarchy_cost(&too_slow, g2, margin);
        assert!(!c.feasible && c.total.is_infinite(), "a perfect setpoint past the delay margin is still unstable");
    }
}
