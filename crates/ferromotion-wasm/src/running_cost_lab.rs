//! **"It runs" is not "it runs in time"** — the on-device lab behind the cost-of-running lesson.
//!
//! Three costs that a correctness test cannot see, because a passing test says nothing about any of them:
//!
//! 1. **The deadline.** A policy that infers in 90 ms cannot drive a 100 Hz loop, and the failure is not an exception:
//!    the fast loop runs out of actions and something has to be sent to the motors anyway.
//!    [`ferromotion_policy::ChunkClock`] owns the two rates and reports which of three states it is in.
//! 2. **The allocations.** Allocating inside a control tick is a latency spike waiting for the wrong moment. The fix
//!    only counts if the numbers do not move, so the live check here is **bit-identity** between the allocating and
//!    workspace dynamics paths, compared as raw bit patterns rather than with a tolerance.
//! 3. **The benchmark's own honesty.** A timing whose spread is dominated by the machine is not a measurement of the
//!    code, and [`ferromotion_bench::Measurement::trustworthy`] is the gate that says so.
//!
//! Allocation *counting* is deliberately not done here. It needs a `#[global_allocator]`, and installing one would
//! tax every other lab on the page to instrument this one. The counts quoted in the lesson were measured natively;
//! what this lab checks live is the property that makes them safe to act on.

use ferromotion_bench::{median_absolute_deviation, quantile, Measurement};
use ferromotion_core::{
    forward_dynamics, forward_dynamics_in, DynamicsWorkspace, Joint, JointKind, LinkInertia, Robot,
};
use ferromotion_policy::{ChunkClock, ClockHealth, Integrator};
use nalgebra::{Matrix3, Vector3};
use wasm_bindgen::prelude::*;

/// Control rate of the fast loop, Hz. 100 Hz is the usual floor for a manipulator.
const CONTROL_HZ: f64 = 100.0;
/// Actions per inferred chunk.
const CHUNK: usize = 20;
/// Action dimension.
const ADIM: usize = 6;
/// Actions the clock keeps in hand before it will accept a new chunk.
const SOFT: usize = 4;
/// Delay margin, in control samples, that the plant tolerates before the stability argument lapses.
const MARGIN: usize = 6;

#[wasm_bindgen]
pub struct RunningCostLab {
    /// Policy inference time, seconds.
    inference: f64,
    /// Relative jitter injected into the synthetic timing samples, for the trustworthiness panel.
    jitter: f64,
    dof: usize,
}

#[wasm_bindgen]
impl RunningCostLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RunningCostLab {
        RunningCostLab { inference: 0.03, jitter: 0.02, dof: 6 }
    }

    /// Inference time in milliseconds, which is how a practitioner actually thinks about it.
    pub fn set_inference_ms(&mut self, ms: f64) {
        self.inference = (ms.max(0.0) / 1000.0).min(1.0);
    }

    pub fn inference_ms(&self) -> f64 {
        self.inference * 1000.0
    }

    /// Relative jitter on the synthetic timing samples, 0 to 1.
    pub fn set_jitter(&mut self, j: f64) {
        self.jitter = j.clamp(0.0, 1.0);
    }

    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    pub fn control_period_ms(&self) -> f64 {
        1000.0 / CONTROL_HZ
    }

    // ---------------------------------------------------------------- 1. the deadline
    fn clock(&self) -> Option<ChunkClock> {
        ChunkClock::new(CHUNK, ADIM, 1.0 / CONTROL_HZ, SOFT, MARGIN)
    }

    /// **Actions frozen by the inference latency.** Not a tuning parameter: `ceil(inference / control_period)`, capped
    /// at the chunk length. These are the actions already committed to the motors before a new chunk can land.
    pub fn frozen(&self) -> f64 {
        self.clock().map_or(f64::NAN, |c| c.frozen_for(self.inference) as f64)
    }

    /// Latency in control samples, **uncapped**, so the margin check can still see an over-long inference.
    pub fn latency_samples(&self) -> f64 {
        self.clock().map_or(f64::NAN, |c| c.latency_samples(self.inference) as f64)
    }

    pub fn margin_samples(&self) -> f64 {
        MARGIN as f64
    }

    /// Run the clock for `ticks` control steps against a constant-velocity field and report how many ticks had **no
    /// action to send**. This is the number that matters: a gap is a motor command that had to be invented.
    pub fn gaps_over(&self, ticks: usize) -> f64 {
        let Some(mut clock) = self.clock() else { return f64::NAN };
        let field = |_a: &[f64], _t: f64| vec![0.1; CHUNK * ADIM];
        let a0 = vec![0.0; CHUNK * ADIM];
        let mut gaps = 0usize;
        for _ in 0..ticks.min(4000) {
            if clock.queued() <= SOFT {
                clock.deliver(&field, &a0, self.inference, 4, Integrator::Heun);
            }
            if clock.tick().is_none() {
                gaps += 1;
            }
        }
        gaps as f64
    }

    /// **Health**: `0 = Nominal, 1 = BeyondMargin, 2 = Starved, -1 = unavailable`. Beyond-margin and starved are
    /// independent failures: the first withdraws the stability claim while the stream continues, the second means the
    /// stream stopped.
    pub fn health_code(&self) -> i32 {
        let Some(mut clock) = self.clock() else { return -1 };
        let field = |_a: &[f64], _t: f64| vec![0.1; CHUNK * ADIM];
        let a0 = vec![0.0; CHUNK * ADIM];
        // Warm the clock the way the loop would, then read its state.
        for _ in 0..40 {
            if clock.queued() <= SOFT {
                clock.deliver(&field, &a0, self.inference, 4, Integrator::Heun);
            }
            let _ = clock.tick();
        }
        match clock.state(self.inference).health {
            ClockHealth::Nominal => 0,
            ClockHealth::BeyondMargin { .. } => 1,
            ClockHealth::Starved => 2,
        }
    }

    /// Seconds of slack the queue represents at the control rate. Negative-going slack is the early warning.
    pub fn slack_ms(&self) -> f64 {
        let Some(mut clock) = self.clock() else { return f64::NAN };
        let field = |_a: &[f64], _t: f64| vec![0.1; CHUNK * ADIM];
        let a0 = vec![0.0; CHUNK * ADIM];
        for _ in 0..40 {
            if clock.queued() <= SOFT {
                clock.deliver(&field, &a0, self.inference, 4, Integrator::Heun);
            }
            let _ = clock.tick();
        }
        clock.slack_seconds() * 1000.0
    }

    // ---------------------------------------------------------------- 2. the allocations
    fn arm(&self) -> (Robot, Vec<LinkInertia>) {
        let joints: Vec<Joint> = (0..self.dof)
            .map(|i| Joint {
                origin: nalgebra::Isometry3::translation(0.0, 0.0, 0.12),
                axis: nalgebra::Unit::new_normalize(if i % 2 == 0 { Vector3::z() } else { Vector3::y() }),
                kind: JointKind::Revolute,
                limits: Some((-2.5, 2.5)),
            })
            .collect();
        let inertia: Vec<LinkInertia> = (0..self.dof)
            .map(|_| LinkInertia {
                mass: 1.4,
                com: Vector3::new(0.0, 0.0, 0.06),
                inertia: Matrix3::from_diagonal(&Vector3::new(0.01, 0.01, 0.004)),
            })
            .collect();
        (Robot { joints, ee_offset: nalgebra::Isometry3::identity() }, inertia)
    }

    /// **Bit-identity between the allocating and workspace dynamics paths.**
    ///
    /// Compared with `to_bits()`, not with a tolerance. An optimisation that changes the last bit is a different
    /// simulator, and "close enough" is how a reproducibility claim quietly dies.
    pub fn workspace_is_bit_identical(&self) -> bool {
        let (robot, inertia) = self.arm();
        let n = robot.dof();
        let q: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        let qd: Vec<f64> = (0..n).map(|i| 0.05 * (i as f64)).collect();
        let tau: Vec<f64> = (0..n).map(|i| 0.3 - 0.02 * (i as f64)).collect();
        let g = Vector3::new(0.0, 0.0, -9.81);

        let a = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
        let mut ws = DynamicsWorkspace::new(n);
        let b = forward_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &tau, g).to_vec();
        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Largest absolute difference between the two paths, so a learner sees `0` rather than being told it.
    pub fn workspace_max_diff(&self) -> f64 {
        let (robot, inertia) = self.arm();
        let n = robot.dof();
        let q: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        let qd: Vec<f64> = (0..n).map(|i| 0.05 * (i as f64)).collect();
        let tau: Vec<f64> = (0..n).map(|i| 0.3 - 0.02 * (i as f64)).collect();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let a = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
        let mut ws = DynamicsWorkspace::new(n);
        let b = forward_dynamics_in(&mut ws, &robot, &inertia, &q, &qd, &tau, g).to_vec();
        a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Allocations per `forward_dynamics` call before and after the workspace, measured natively with a counting
    /// allocator. Quoted rather than re-measured, because instrumenting the browser's allocator would tax every lab.
    pub fn allocs_before(&self) -> f64 {
        35.0
    }

    pub fn allocs_after(&self) -> f64 {
        1.0
    }

    // ---------------------------------------------------------------- 3. the benchmark's honesty
    /// Synthetic timing samples with the chosen relative jitter, deterministic so the readout is stable.
    fn samples(&self) -> Vec<f64> {
        let base = 8e-6;
        let n = 21usize;
        (0..n)
            .map(|i| {
                // A deterministic zig-zag plus one occasional large excursion, which is what a real machine does.
                let phase = (i as f64 * 2.399963).sin();
                let spike = if i % 7 == 3 { 3.0 } else { 0.0 };
                base * (1.0 + self.jitter * (phase + spike))
            })
            .map(|v| v.max(1e-9))
            .collect()
    }

    fn measurement(&self) -> Measurement {
        let mut s = self.samples();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        Measurement {
            median: quantile(&s, 0.5),
            mad: median_absolute_deviation(&s),
            p10: quantile(&s, 0.10),
            p90: quantile(&s, 0.90),
            min: s.first().copied().unwrap_or(f64::NAN),
            samples: s.len(),
            iters_per_sample: 1000,
        }
    }

    pub fn median_ns(&self) -> f64 {
        self.measurement().median * 1e9
    }

    /// Median absolute deviation, reported instead of a standard deviation because one paused sample moves a standard
    /// deviation and does not move this.
    pub fn mad_ns(&self) -> f64 {
        self.measurement().mad * 1e9
    }

    /// `(p90 - p10) / median`. Above about 0.1 the run is dominated by the machine rather than the code.
    pub fn relative_spread(&self) -> f64 {
        self.measurement().relative_spread()
    }

    /// **Whether this number may be quoted.** The gate that stops performance folklore.
    pub fn trustworthy(&self) -> bool {
        self.measurement().trustworthy()
    }

    /// The library's own one-line report, verdict attached so it cannot be dropped in transit.
    pub fn report(&self) -> String {
        self.measurement().report("forward_dynamics (6 dof)")
    }
}

impl Default for RunningCostLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The deadline, pinned.** Frozen actions must equal `ceil(inference / control_period)`, because a frozen prefix
    /// that is a guess rather than a measurement lets a new chunk contradict motions already sent to the motors.
    #[test]
    fn frozen_actions_are_a_measurement_of_the_latency() {
        let mut lab = RunningCostLab::new();
        for (ms, want) in [(1.0, 1.0), (10.0, 1.0), (11.0, 2.0), (30.0, 3.0), (95.0, 10.0)] {
            lab.set_inference_ms(ms);
            let got = lab.frozen();
            eprintln!("inference {ms:>5.1} ms at {} ms/tick -> {got} actions frozen", lab.control_period_ms());
            assert_eq!(got, want, "at {ms} ms");
        }
    }

    /// A fast policy must never starve the loop, and a slow one must be reported rather than silently papered over.
    #[test]
    fn a_fast_policy_never_gaps_and_a_slow_one_is_reported() {
        let mut lab = RunningCostLab::new();
        lab.set_inference_ms(20.0);
        let gaps = lab.gaps_over(400);
        eprintln!("20 ms inference: {gaps} gaps over 400 ticks, health {}, slack {:.2} ms", lab.health_code(), lab.slack_ms());
        assert_eq!(gaps, 0.0, "a 20 ms policy should keep a 100 Hz loop fed");
        assert_eq!(lab.health_code(), 0, "and report Nominal");

        // Past the delay margin the stream can survive while the stability argument does not, and that is a distinct
        // state rather than a failure.
        lab.set_inference_ms(90.0);
        eprintln!(
            "90 ms inference: latency {} samples against a margin of {} -> health {}",
            lab.latency_samples(),
            lab.margin_samples(),
            lab.health_code()
        );
        assert!(lab.latency_samples() > lab.margin_samples(), "90 ms should exceed the margin");
        assert_ne!(lab.health_code(), 0, "and the clock must not call that Nominal");
    }

    /// **The allocation fix has to be free.** Bit-identity, not a tolerance: an optimisation that moves the last bit is
    /// a different simulator.
    #[test]
    fn the_workspace_path_is_bit_identical() {
        let lab = RunningCostLab::new();
        eprintln!(
            "workspace vs allocating forward_dynamics: max diff {:.1e}, bit-identical {}",
            lab.workspace_max_diff(),
            lab.workspace_is_bit_identical()
        );
        assert!(lab.workspace_is_bit_identical(), "the workspace path must not change a single bit");
        assert_eq!(lab.workspace_max_diff(), 0.0);
    }

    /// **The honesty gate.** A clean run is quotable and a jittery one is not, and the verdict has to flip on the
    /// spread rather than on the median.
    #[test]
    fn a_jittery_measurement_refuses_to_be_quoted() {
        let mut lab = RunningCostLab::new();
        lab.set_jitter(0.0);
        eprintln!("no jitter:  {}", lab.report());
        assert!(lab.trustworthy(), "a clean run should be quotable, spread {}", lab.relative_spread());

        lab.set_jitter(0.5);
        eprintln!("50% jitter: {}", lab.report());
        assert!(!lab.trustworthy(), "a jittery run must refuse, spread {}", lab.relative_spread());
        assert!(lab.report().contains("NOISY"), "and say so in the line itself");
    }

    /// The median must barely move under jitter while the spread explodes: that asymmetry is the reason the median and
    /// MAD are reported instead of a mean and a standard deviation.
    #[test]
    fn the_median_is_robust_where_the_spread_is_not() {
        let mut lab = RunningCostLab::new();
        lab.set_jitter(0.0);
        let (m0, s0) = (lab.median_ns(), lab.relative_spread());
        lab.set_jitter(0.5);
        let (m1, s1) = (lab.median_ns(), lab.relative_spread());
        eprintln!("median {m0:.3} -> {m1:.3} ns, relative spread {s0:.4} -> {s1:.4}");
        assert!((m1 - m0).abs() / m0 < 0.6, "the median should be comparatively steady");
        assert!(s1 > 10.0 * s0.max(1e-6), "the spread should blow up: {s0} -> {s1}");
    }
}
