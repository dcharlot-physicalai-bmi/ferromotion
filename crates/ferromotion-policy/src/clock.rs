//! **The chunk-execution clock** — the thing that owns the queue, the two rates, and the measured latency.
//!
//! Everything RTC needs existed except the object that runs it. [`rtc_mask`](crate::rtc_mask) takes a `frozen` count as
//! an argument and nothing in the workspace produced one; `latency`'s delay margin and `hierarchy`'s dual-rate budget
//! are both stateless analyses over caller-supplied numbers. So the pieces were correct and unconnected: a chunking
//! policy could be *analysed* but not *run*.
//!
//! `frozen` is not a tuning parameter. It is a measurement: **how many actions the fast loop consumed while inference
//! was running**. Get it wrong low and the new chunk contradicts actions already sent to the motors; get it wrong high
//! and the policy is pinned to stale intent for longer than necessary. [`ChunkClock`] derives it from the two rates and
//! the observed inference time, so it cannot silently disagree with what actually executed.
//!
//! ```text
//!   frozen = ceil(inference_time / control_period),  capped at the chunk length
//! ```
//!
//! Three properties the clock is responsible for, each checked below rather than asserted:
//!
//! 1. **No gap in the action stream.** The fast loop must have an action every tick even while inference runs long. That
//!    is the entire point of chunking, and a clock that starves is worse than no clock.
//! 2. **`frozen` matches what executed.** The count handed to the inpainter equals the number of actions actually
//!    consumed during that inference, exactly.
//! 3. **Latency beyond the margin is reported, not absorbed.** [`ClockState::within_margin`] compares the measured
//!    latency against a delay margin computed elsewhere; a clock that keeps running silently past it is the failure
//!    mode `cbf`'s safety filter had.

use crate::{sample_rtc, Integrator};

/// What the clock is currently able to promise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClockHealth {
    /// A chunk is queued and the measured latency is inside the delay margin.
    Nominal,
    /// Running, but the measured inference latency exceeds the delay margin the plant tolerates. The action stream is
    /// intact; the stability argument is not.
    BeyondMargin { latency_samples: usize, margin: usize },
    /// The queue ran dry — the fast loop had no action to send. A clock that reaches this state has failed at its one
    /// job, and it says so rather than repeating a stale action as though nothing happened.
    Starved,
}

/// A snapshot of the clock, for logging and for gating.
#[derive(Clone, Debug)]
pub struct ClockState {
    pub health: ClockHealth,
    /// Actions still queued and unexecuted.
    pub queued: usize,
    /// Actions consumed since the clock started.
    pub executed: usize,
    /// The `frozen` count used for the most recent inpainting, derived from the measured latency.
    pub last_frozen: usize,
    /// Chunks produced so far.
    pub chunks: usize,
}

impl ClockState {
    /// Whether the measured latency is inside the plant's delay margin. `false` does **not** mean the action stream
    /// broke; it means the stability argument no longer covers it.
    pub fn within_margin(&self) -> bool {
        !matches!(self.health, ClockHealth::BeyondMargin { .. })
    }

    pub fn healthy(&self) -> bool {
        self.health == ClockHealth::Nominal
    }
}

/// **A running action-chunking policy**: one slow producer, one fast consumer, one queue.
///
/// The clock does not own a policy or a clock source. It takes the inference result and the measured time when a chunk
/// is delivered, which keeps it testable against a synthetic timeline and honest about what it measures — a clock that
/// read a wall clock internally could not be checked against a known answer.
#[derive(Clone, Debug)]
pub struct ChunkClock {
    /// Actions per chunk.
    pub chunk_len: usize,
    /// Scalars per action.
    pub action_dim: usize,
    /// Fast-loop period, seconds.
    pub control_period: f64,
    /// Length of the soft ramp easing off the freeze, in actions.
    pub soft: usize,
    /// The plant's tolerated delay in control samples, from `latency::delay_margin_samples`.
    pub margin: usize,
    /// Flattened queue of pending actions, oldest first.
    queue: Vec<f64>,
    /// The chunk most recently produced, kept whole because the inpainter needs it as its freeze target.
    last_chunk: Vec<f64>,
    /// Index within `last_chunk` of the first action not yet executed, which is what makes `frozen` a measurement.
    consumed_in_chunk: usize,
    executed: usize,
    chunks: usize,
    last_frozen: usize,
    starved: bool,
}

impl ChunkClock {
    pub fn new(chunk_len: usize, action_dim: usize, control_period: f64, soft: usize, margin: usize) -> Option<ChunkClock> {
        (chunk_len > 0 && action_dim > 0 && control_period > 0.0).then_some(ChunkClock {
            chunk_len,
            action_dim,
            control_period,
            soft,
            margin,
            queue: Vec::new(),
            last_chunk: Vec::new(),
            consumed_in_chunk: 0,
            executed: 0,
            chunks: 0,
            last_frozen: 0,
            starved: false,
        })
    }

    /// **`frozen` for an inference that took `inference_time` seconds**: the number of actions the fast loop will have
    /// consumed by the time the chunk lands, capped at the chunk length.
    ///
    /// Capping matters: an inference slower than a whole chunk cannot be inpainted against actions that no longer exist
    /// in the chunk, and the clock reports that as [`ClockHealth::BeyondMargin`] rather than pretending otherwise.
    pub fn frozen_for(&self, inference_time: f64) -> usize {
        let ticks = (inference_time / self.control_period).ceil().max(0.0) as usize;
        ticks.min(self.chunk_len)
    }

    /// Latency in control samples, for comparison against the delay margin.
    pub fn latency_samples(&self, inference_time: f64) -> usize {
        (inference_time / self.control_period).ceil().max(0.0) as usize
    }

    /// **Deliver a chunk from the slow loop**, inpainted against whatever is already committed.
    ///
    /// `field` is the policy's learned velocity field, `a0` the flow's start point. `inference_time` is how long the
    /// inference actually took — it is what makes `frozen` a measurement rather than a guess, and passing a fabricated
    /// value is the one way to make this dishonest.
    ///
    /// The first chunk has nothing to freeze against, so it is a plain sample. Every later chunk freezes its overlap to
    /// the actions the fast loop has already consumed.
    pub fn deliver(&mut self, field: &dyn Fn(&[f64], f64) -> Vec<f64>, a0: &[f64], inference_time: f64, steps: usize, method: Integrator) -> usize {
        let n = self.chunk_len * self.action_dim;
        if a0.len() != n {
            return 0;
        }
        // The flow's state is the WHOLE flattened chunk, so the field must return that many components. Probing it
        // once here turns a caller mistake into a refusal; without this the integrator indexes out of bounds, which
        // in a wasm build aborts the module rather than reporting anything.
        if field(a0, 0.0).len() != n {
            return 0;
        }
        let frozen = if self.chunks == 0 {
            0
        } else {
            // The freeze target is the tail of the previous chunk that has already run, realigned to the new chunk's
            // start. `frozen` is how many of those actions the fast loop consumed while this inference was in flight.
            self.frozen_for(inference_time)
        };
        let target = if self.last_chunk.len() == n { self.last_chunk.clone() } else { vec![0.0; n] };
        let chunk = sample_rtc(field, a0, &target, self.chunk_len, self.action_dim, frozen, self.soft, steps, method);

        // The frozen prefix is already committed, so only the un-frozen remainder is new work for the queue. Enqueueing
        // the frozen actions again would replay motions the motors have already performed.
        self.queue.clear();
        self.queue.extend_from_slice(&chunk[frozen * self.action_dim..]);
        self.last_chunk = chunk;
        self.consumed_in_chunk = frozen;
        self.last_frozen = frozen;
        self.chunks += 1;
        frozen
    }

    /// **One fast-loop tick**: the next action, or `None` if the queue is dry.
    ///
    /// `None` is a real answer and the caller has to handle it. Repeating the last action here would hide a starved
    /// clock behind plausible motion, which is exactly the class of failure this session has been removing.
    pub fn tick(&mut self) -> Option<Vec<f64>> {
        if self.queue.len() < self.action_dim {
            self.starved = true;
            return None;
        }
        let action: Vec<f64> = self.queue.drain(..self.action_dim).collect();
        self.executed += 1;
        self.consumed_in_chunk += 1;
        Some(action)
    }

    /// How many actions remain queued.
    pub fn queued(&self) -> usize {
        self.queue.len() / self.action_dim
    }

    /// The clock's state, including whether the last measured latency was inside the margin.
    pub fn state(&self, last_inference_time: f64) -> ClockState {
        let latency = self.latency_samples(last_inference_time);
        let health = if self.starved {
            ClockHealth::Starved
        } else if latency > self.margin {
            ClockHealth::BeyondMargin { latency_samples: latency, margin: self.margin }
        } else {
            ClockHealth::Nominal
        };
        ClockState { health, queued: self.queued(), executed: self.executed, last_frozen: self.last_frozen, chunks: self.chunks }
    }

    /// **The deadline the slow loop has to meet**: seconds until the queue would run dry at the current fill.
    ///
    /// This is the number a scheduler needs, and it is the same quantity the delay margin bounds — so a policy whose
    /// inference exceeds it starves, and one whose inference exceeds the margin destabilises even without starving.
    pub fn slack_seconds(&self) -> f64 {
        self.queued() as f64 * self.control_period
    }
}

#[cfg(test)]
mod tests {
    /// A field whose output does not match the flow state is a caller mistake, and it must be **refused**, not
    /// indexed past. Before this guard the integrator ran off the end of the vector: a panic, which in a wasm build
    /// aborts the whole module and reports nothing.
    #[test]
    fn a_field_of_the_wrong_width_is_refused_not_indexed_past() {
        let mut clock = ChunkClock::new(20, 6, 0.01, 4, 6).unwrap();
        let a0 = vec![0.0; 20 * 6];

        // One action's worth instead of the whole chunk's worth: the shape a caller naturally reaches for first.
        let too_narrow = |_a: &[f64], _t: f64| vec![0.1; 6];
        assert_eq!(clock.deliver(&too_narrow, &a0, 0.02, 4, Integrator::Heun), 0);
        assert_eq!(clock.queued(), 0, "nothing may be enqueued from a refused delivery");

        // The correct width flows normally.
        let right = |_a: &[f64], _t: f64| vec![0.1; 20 * 6];
        clock.deliver(&right, &a0, 0.02, 4, Integrator::Heun);
        assert!(clock.queued() > 0, "a well-formed field must still work");
    }

    use super::*;

    /// A constant-velocity field, so the chunk a given `a0` produces is known in closed form: `a0 + g`.
    fn const_field(g: Vec<f64>) -> impl Fn(&[f64], f64) -> Vec<f64> {
        move |_a: &[f64], _t: f64| g.clone()
    }

    fn clock() -> ChunkClock {
        // 20 actions per chunk, 2 scalars each, 100 Hz control, 3-action soft ramp, margin of 8 samples
        ChunkClock::new(20, 2, 0.01, 3, 8).expect("valid")
    }

    /// **`frozen` is a measurement, not a parameter.** It equals the number of ticks the fast loop takes during the
    /// inference, and it is capped at the chunk length.
    #[test]
    fn frozen_is_derived_from_the_measured_latency() {
        let c = clock();
        eprintln!("control period {} s, chunk {} actions", c.control_period, c.chunk_len);
        eprintln!("{:>16}  {:>8}  {:>8}", "inference time", "ticks", "frozen");
        for t in [0.0, 0.005, 0.01, 0.025, 0.08, 0.19, 0.20, 0.5] {
            let (ticks, frozen) = (c.latency_samples(t), c.frozen_for(t));
            eprintln!("{t:>16.3}  {ticks:>8}  {frozen:>8}");
            assert_eq!(frozen, ticks.min(c.chunk_len), "frozen is the tick count, capped at the chunk");
        }
        assert_eq!(c.frozen_for(0.0), 0, "zero latency freezes nothing");
        assert_eq!(c.frozen_for(0.025), 3, "25 ms at 100 Hz is 3 ticks");
        assert_eq!(c.frozen_for(0.5), 20, "an inference longer than a chunk caps at the chunk length");
        assert_eq!(c.latency_samples(0.5), 50, "but the LATENCY is still reported uncapped, so the margin check sees it");
    }

    /// **The action stream must never gap.** The fast loop runs continuously while the slow loop delivers late.
    #[test]
    fn the_fast_loop_never_starves_while_inference_keeps_up() {
        let mut c = clock();
        let v = const_field(vec![0.05; 40]);
        let a0 = vec![0.0; 40];
        let mut inference = 0.0;
        c.deliver(&v, &a0, inference, 6, Integrator::Heun);

        let mut gaps = 0usize;
        let mut ticks = 0usize;
        // run 400 control ticks, re-planning whenever the queue drops below the inference cost
        for _ in 0..400 {
            if c.queued() as f64 * c.control_period <= 0.06 {
                inference = 0.05; // 50 ms, 5 ticks at 100 Hz
                c.deliver(&v, &a0, inference, 6, Integrator::Heun);
            }
            match c.tick() {
                Some(a) => {
                    assert_eq!(a.len(), 2);
                    ticks += 1;
                }
                None => gaps += 1,
            }
        }
        let st = c.state(inference);
        eprintln!("400 control ticks: {ticks} actions delivered, {gaps} gaps, {} chunks produced", st.chunks);
        eprintln!("   final state: {:?}, queued {}, last frozen {}", st.health, st.queued, st.last_frozen);
        assert_eq!(gaps, 0, "the whole point of chunking is that the fast loop never waits");
        assert_eq!(ticks, 400);
        assert!(st.healthy(), "and the latency stayed inside the margin: {:?}", st.health);
    }

    /// **`frozen` equals what actually executed.** The count handed to the inpainter matches the actions the fast loop
    /// consumed during that inference, exactly — which is the property that keeps the new chunk from contradicting
    /// motions already sent to the motors.
    #[test]
    fn frozen_matches_the_actions_consumed_during_inference() {
        let mut c = clock();
        let v = const_field(vec![0.02; 40]);
        let a0 = vec![0.0; 40];
        c.deliver(&v, &a0, 0.0, 6, Integrator::Heun);

        for inference_ticks in [1usize, 3, 5, 9] {
            let inference = inference_ticks as f64 * c.control_period;
            // the fast loop runs for exactly the inference duration
            let before = c.executed;
            for _ in 0..inference_ticks {
                assert!(c.tick().is_some(), "the queue must cover the inference");
            }
            let consumed = c.executed - before;
            let frozen = c.deliver(&v, &a0, inference, 6, Integrator::Heun);
            eprintln!("inference of {inference_ticks} ticks: consumed {consumed} actions, frozen = {frozen}");
            assert_eq!(frozen, consumed, "frozen must equal what executed, exactly");
        }
    }

    /// **Zero latency reduces to naive chunking, bit-identically.** `rtc.rs` checks this at the function level; the
    /// clock has to preserve it, because a clock that adds a freeze at zero latency has invented one.
    #[test]
    fn zero_latency_reduces_to_naive_chunking() {
        let v = const_field(vec![0.3, -0.2]);
        let a0 = vec![0.1, -0.1];
        let mut c = ChunkClock::new(1, 2, 0.01, 0, 8).expect("valid");
        // first chunk: nothing to freeze against
        c.deliver(&v, &a0, 0.0, 8, Integrator::Heun);
        let first: Vec<f64> = (0..1).flat_map(|_| c.tick().unwrap()).collect();
        // second chunk at zero measured latency: frozen must be 0, so it equals a plain sample
        let frozen = c.deliver(&v, &a0, 0.0, 8, Integrator::Heun);
        let second: Vec<f64> = (0..1).flat_map(|_| c.tick().unwrap()).collect();
        let plain = crate::sample_field(&v, &a0, 8, Integrator::Heun);
        eprintln!("frozen at zero latency: {frozen}");
        eprintln!("   first chunk {first:?}, second {second:?}, plain sample {plain:?}");
        assert_eq!(frozen, 0, "zero measured latency must freeze nothing");
        assert!(second.iter().zip(&plain).all(|(a, b)| (a - b).abs() < 1e-15), "and the chunk must be bit-identical to naive");
        assert!(first.iter().zip(&plain).all(|(a, b)| (a - b).abs() < 1e-15), "as must the first");
    }

    /// **Latency beyond the margin is reported, not absorbed.** The stream stays intact; the stability claim does not,
    /// and the clock says which.
    #[test]
    fn latency_beyond_the_margin_is_reported() {
        let mut c = clock(); // margin 8 samples
        let v = const_field(vec![0.02; 40]);
        let a0 = vec![0.0; 40];
        c.deliver(&v, &a0, 0.0, 6, Integrator::Heun);
        eprintln!("margin {} samples", c.margin);
        for inference in [0.03, 0.08, 0.12, 0.30] {
            let st = c.state(inference);
            let lat = c.latency_samples(inference);
            eprintln!("   inference {inference:>5.2} s = {lat:>3} samples: {:?}", st.health);
            if lat > c.margin {
                assert!(!st.within_margin(), "beyond the margin must be reported");
                assert!(matches!(st.health, ClockHealth::BeyondMargin { .. }));
            } else {
                assert!(st.within_margin());
            }
        }
        // and the action stream is still intact at that point - the two failures are independent
        assert!(c.tick().is_some(), "beyond-margin is a stability report, not a starvation");
    }

    /// A starved queue is reported as starved. Repeating the last action would hide it behind plausible motion.
    #[test]
    fn a_starved_queue_says_so() {
        let mut c = ChunkClock::new(2, 2, 0.01, 0, 8).expect("valid");
        let v = const_field(vec![0.1; 4]);
        c.deliver(&v, &[0.0; 4], 0.0, 4, Integrator::Heun);
        assert_eq!(c.queued(), 2);
        assert!(c.tick().is_some() && c.tick().is_some());
        assert_eq!(c.queued(), 0);
        eprintln!("queue drained; slack {} s", c.slack_seconds());
        assert!(c.tick().is_none(), "an empty queue returns None rather than a stale action");
        let st = c.state(0.0);
        eprintln!("   state: {:?}", st.health);
        assert_eq!(st.health, ClockHealth::Starved);
        assert!(!st.healthy());
    }

    /// The frozen prefix is not re-enqueued: actions the motors already performed must not be replayed.
    #[test]
    fn the_frozen_prefix_is_not_replayed() {
        let mut c = clock();
        let v = const_field(vec![0.02; 40]);
        let a0 = vec![0.0; 40];
        c.deliver(&v, &a0, 0.0, 6, Integrator::Heun);
        assert_eq!(c.queued(), 20, "a fresh chunk queues all 20 actions");
        for _ in 0..5 {
            c.tick();
        }
        let frozen = c.deliver(&v, &a0, 0.05, 6, Integrator::Heun); // 5 ticks of inference
        eprintln!("after freezing {frozen} actions, {} remain queued of a {}-action chunk", c.queued(), c.chunk_len);
        assert_eq!(frozen, 5);
        assert_eq!(c.queued(), c.chunk_len - frozen, "only the un-frozen remainder is new work");
    }

    /// `slack_seconds` is the deadline the slow loop has to meet, and it falls as the queue drains.
    #[test]
    fn slack_is_the_deadline_and_it_falls_as_the_queue_drains() {
        let mut c = clock();
        let v = const_field(vec![0.02; 40]);
        c.deliver(&v, &[0.0; 40], 0.0, 6, Integrator::Heun);
        let mut prev = f64::INFINITY;
        eprintln!("{:>8}  {:>8}", "queued", "slack (s)");
        for k in 0..5 {
            let s = c.slack_seconds();
            if k % 2 == 0 {
                eprintln!("{:>8}  {:>8.3}", c.queued(), s);
            }
            assert!(s < prev);
            prev = s;
            c.tick();
        }
        assert!((c.slack_seconds() - c.queued() as f64 * c.control_period).abs() < 1e-15);
    }

    /// Degenerate construction and mismatched widths are refused.
    #[test]
    fn bad_input_is_refused() {
        assert!(ChunkClock::new(0, 2, 0.01, 0, 8).is_none(), "a zero-length chunk");
        assert!(ChunkClock::new(4, 0, 0.01, 0, 8).is_none(), "a zero-dimensional action");
        assert!(ChunkClock::new(4, 2, 0.0, 0, 8).is_none(), "a zero control period");
        let mut c = clock();
        let v = const_field(vec![0.1; 40]);
        assert_eq!(c.deliver(&v, &[0.0; 4], 0.0, 4, Integrator::Heun), 0, "an a0 of the wrong width delivers nothing");
        assert_eq!(c.queued(), 0);
    }
}
