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
    ///
    /// **Returns `frozen`, and `0` is ambiguous.** It means either "refused" (bad `a0` or field width) or the perfectly
    /// healthy cases of a first chunk and of zero measured latency. A caller testing success must look at
    /// [`ChunkClock::queued`], not at this return value.
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
        // **`soft` has to be zeroed with `frozen`, not just `frozen` (2026-08-14).** The doc above says the first
        // chunk "is a plain sample". It was not: `frozen` went to 0 but `self.soft` was still passed, and with no
        // previous chunk `target` falls through to `vec![0.0; n]` — so the first `soft` actions were soft-guided
        // toward an all-zero target that does not correspond to any committed action. The guided prefix converges
        // on that fabricated target as the step count rises, so the error is unbounded in the integrator
        // parameter, not a small blend artifact. Measured on this crate's own fixture (`ChunkClock::new(20, 2,
        // 0.01, 3, 8)`, a constant field at 0.05): the first three commanded actions came out 0.00278, 0.00820,
        // 0.02055 instead of 0.05 — the very first action sent to the motors was 18x too small.
        //
        // A freeze target and a soft-guide target are the same object here, so "nothing to freeze against" and
        // "nothing to guide toward" are the same condition and must be handled by the same branch.
        let (frozen, soft) = if self.chunks == 0 {
            (0, 0)
        } else {
            // The freeze target is the tail of the previous chunk that has already run, realigned to the new chunk's
            // start. `frozen` is how many of those actions the fast loop consumed while this inference was in flight.
            // **Reconciled against what the fast loop ACTUALLY consumed (2026-08-15).** `frozen_for` is a
            // prediction from the measured inference time; `consumed_in_chunk` is the truth. You can only freeze
            // the new chunk against actions that really were committed, so the prediction is capped by reality.
            //
            // Unreconciled, a prediction larger than the truth broke the realignment below in two ways at once.
            // `offset = consumed_in_chunk.saturating_sub(frozen)` clamped to 0, which silently restored the
            // UNSHIFTED freeze target — the exact defect the shift was added to fix — and simultaneously
            // restarted `queue` at chunk index `frozen`, dropping the slots between where playback had reached
            // and there. Measured with `ChunkClock::new(20, 1, 0.01, 0, 6)`, two ticks executed and a 50 ms
            // inference (`frozen = 5` against `consumed_in_chunk = 2`): three trajectory slots were skipped and
            // the commanded stream jumped by 0.2699 against a 0.0675 median interior step, a 4x boundary
            // discontinuity in the one place RTC exists to keep continuous.
            //
            // This is reachable whenever the fast loop ticks slower than the nominal control period, which is
            // the normal condition under load rather than an exotic one.
            (self.frozen_for(inference_time).min(self.consumed_in_chunk), self.soft)
        };
        // **Realign the freeze target to the playback position.** The actions the fast loop actually executed while
        // this inference was in flight are the previous chunk's indices [consumed_in_chunk - frozen,
        // consumed_in_chunk) — NOT [0, frozen). `rtc_mask` freezes and soft-guides the new chunk's leading entries
        // against `target`'s leading entries, so `target` has to start where playback currently is.
        //
        // Without this shift the blend target is where the previous chunk STARTED, which the arm left several ticks
        // ago, and the commanded stream jumps at every chunk boundary — measured at 16.6x a normal intra-chunk step,
        // which is exactly the discontinuity RTC exists to remove.
        let target = if self.last_chunk.len() == n {
            let offset = self.consumed_in_chunk.saturating_sub(frozen).min(self.chunk_len);
            let mut shifted = vec![0.0; n];
            for j in 0..self.chunk_len {
                let src = offset + j;
                // Past the end of the previous chunk there is nothing committed to blend against, so hold its last
                // action rather than invent a zero.
                let take = src.min(self.chunk_len - 1);
                shifted[j * self.action_dim..(j + 1) * self.action_dim]
                    .copy_from_slice(&self.last_chunk[take * self.action_dim..(take + 1) * self.action_dim]);
            }
            shifted
        } else {
            vec![0.0; n]
        };
        let chunk = sample_rtc(field, a0, &target, self.chunk_len, self.action_dim, frozen, soft, steps, method);

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

    /// **The freeze target must be aligned to the playback position, not to the previous chunk's start.**
    ///
    /// This is the property RTC exists for: the commanded action stream is continuous across a chunk boundary. It was
    /// broken. `rtc_mask` blends the new chunk's leading entries against `target`'s leading entries, and `target` was
    /// the previous chunk unshifted — where playback STARTED, not where it had reached. Measured on a field with real
    /// intra-chunk variation, the boundary jump was **16.6x a normal interior step**; realigned it is **0.4x**.
    #[test]
    fn the_commanded_stream_is_continuous_across_a_chunk_boundary() {
        let (chunk, adim, period, soft) = (20usize, 1usize, 0.01f64, 4usize);
        let field = |a: &[f64], _t: f64| a.iter().map(|x| 0.5 + 0.3 * x).collect::<Vec<f64>>();
        // A ramped start, so consecutive actions within a chunk genuinely differ and "is the boundary worse than a
        // normal step?" is a question with an answer.
        let a0: Vec<f64> = (0..chunk * adim).map(|i| 0.05 * i as f64).collect();
        let inference = 0.03;

        let mut clock = ChunkClock::new(chunk, adim, period, soft, 6).unwrap();
        clock.deliver(&field, &a0, inference, 4, Integrator::Heun);
        let mut sent: Vec<f64> = Vec::new();
        let mut boundaries: Vec<usize> = Vec::new();
        for _ in 0..60 {
            if clock.queued() <= soft {
                boundaries.push(sent.len());
                clock.deliver(&field, &a0, inference, 4, Integrator::Heun);
            }
            if let Some(a) = clock.tick() {
                sent.push(a[0]);
            }
        }
        let steps: Vec<f64> = sent.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let bset: std::collections::BTreeSet<usize> = boundaries.iter().copied().collect();
        let mut interior: Vec<f64> = steps
            .iter()
            .enumerate()
            .filter(|(i, _)| !bset.contains(&(i + 1)) && !bset.contains(&(i + 2)))
            .map(|(_, s)| *s)
            .collect();
        interior.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = interior[interior.len() / 2];
        let worst = boundaries
            .iter()
            .filter(|b| **b > 0 && **b < steps.len())
            .map(|b| steps[*b - 1])
            .fold(0.0f64, f64::max);
        eprintln!("median interior step {median:.6}; worst boundary jump {worst:.6} = {:.1}x", worst / median);
        assert!(median > 1e-6, "the fixture must have real intra-chunk variation, got {median:e}");
        assert!(worst < 3.0 * median, "boundary jump {worst:.4} is {:.1}x the interior step", worst / median);
    }

    /// **Scope of this test, stated because an adversarial audit found it weaker than it looks.** The loop below
    /// delivers and ticks within the same iteration, which models a scheduler that hands over a chunk instantly. On
    /// that timeline it reports 0 gaps even at 0.19 s inference. A time-honest loop — one that advances control ticks
    /// *during* inference — starves at 0.10 s (130 gaps in 400 ticks). What this pins is the queue/freeze bookkeeping,
    /// not the scheduler; the headline 20 ms configuration survives either timeline.
    /// **The action stream must never gap.** The fast loop runs continuously while the slow loop delivers late.
    #[test]
    fn a_latency_prediction_larger_than_reality_does_not_skip_actions() {
        // `frozen_for` predicts from inference time; `consumed_in_chunk` is what the fast loop really ticked.
        // When the prediction is larger, the unreconciled realignment clamped `offset` to 0 — restoring the
        // unshifted freeze target AND restarting the queue past where playback had reached, dropping the slots
        // in between. This is the normal condition when the fast loop runs slower than its nominal period.
        let mut c = ChunkClock::new(20, 1, 0.01, 0, 6).expect("valid");
        let field = |a: &[f64], _t: f64| a.iter().map(|x| 0.5 + 0.3 * x).collect::<Vec<f64>>();
        let a0: Vec<f64> = (0..20).map(|i| 0.05 * i as f64).collect();

        c.deliver(&field, &a0, 0.0, 4, Integrator::Heun);
        let mut stream = vec![c.tick().expect("first")[0], c.tick().expect("second")[0]];

        // A 50 ms inference is 5 ticks at 100 Hz, so the prediction is 5 while only 2 actions ran.
        assert_eq!(c.frozen_for(0.05), 5, "the prediction should exceed what was consumed");
        let frozen = c.deliver(&field, &a0, 0.05, 4, Integrator::Heun);
        assert_eq!(frozen, 2, "frozen must be capped by the 2 actions actually consumed, got {frozen}");

        for _ in 0..3 {
            stream.push(c.tick().expect("post-delivery")[0]);
        }
        // Continuity across the boundary: the jump must not dwarf a typical interior step.
        let steps: Vec<f64> = stream.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let boundary = steps[1];
        let mut interior = steps.clone();
        interior.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = interior[interior.len() / 2];
        assert!(
            boundary < 6.0 * median.max(1e-9),
            "boundary jump {boundary} against median interior step {median} — the unreconciled version measured \
             0.2699 against 0.0675, a 4x discontinuity in the one place RTC exists to keep smooth"
        );
    }

    #[test]
    fn the_first_chunk_really_is_a_plain_sample() {
        // The doc on `deliver` says so, and it was not true. `frozen` was zeroed for the first chunk but `soft`
        // was not, and with no previous chunk the blend target falls through to all-zeros — so the first `soft`
        // actions were guided toward a target that corresponds to no committed action anywhere. The very first
        // action sent to the motors came out 18x too small.
        //
        // A constant field makes the correct answer exact: with `a0 = 0` and `g = 0.05`, every action of a plain
        // sample is 0.05 regardless of integrator or step count.
        let mut c = clock(); // soft = 3, so the first three actions were the corrupted ones
        let v = const_field(vec![0.05; 40]);
        let a0 = vec![0.0; 40];
        // `deliver` returns `frozen`, which is legitimately 0 for a first chunk — check the queue, not the
        // return value. See the note on `deliver` about that ambiguity.
        c.deliver(&v, &a0, 0.0, 6, Integrator::Heun);
        assert!(c.queued() > 0, "the first chunk should have been enqueued");

        for i in 0..5 {
            let a = c.tick().expect("the first chunk should be queued");
            for (j, &x) in a.iter().enumerate() {
                assert!(
                    (x - 0.05).abs() < 1e-9,
                    "first chunk action {i} component {j} = {x}, want 0.05 — a plain sample of a constant field. \
                     Guiding the first chunk toward a fabricated zero target gave 0.00278 here."
                );
            }
        }

        // And the guard is specific to the FIRST chunk: once there is a real committed chunk to blend against,
        // `soft` must be back in force, otherwise this fix would have silently disabled the soft ramp entirely.
        let mut d = clock();
        d.deliver(&v, &a0, 0.0, 6, Integrator::Heun);
        for _ in 0..4 {
            d.tick();
        }
        let w = const_field(vec![0.9; 40]);
        d.deliver(&w, &a0, 0.05, 6, Integrator::Heun);
        let first = d.tick().expect("second chunk queued")[0];
        assert!(
            (first - 0.9).abs() > 1e-6,
            "the second chunk's leading action should still be blended toward what already ran, got {first} \
             which is an unguided plain sample — the soft ramp was lost"
        );
    }

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
