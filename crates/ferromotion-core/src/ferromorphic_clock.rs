//! Integer-tick boundary for the ferromorphic clock contract.
//!
//! Read from `docs/contracts/ferromotion-clock.md` at ferromorphic
//! `a6722b49ff7dc770b1cb062adac21aa5c7155401`. That document is a draft exchange.
//! Nothing here runs `Sim::step`, `DynamicsLab::step`, or a joint loop.
//!
//! `DynamicsLab::time` stays an accumulated `f64` inside ferromotion-wasm. This
//! module does not write that field and does not change the semi-implicit Euler
//! in `DynamicsLab::step`. Shared seconds, when a tick exists, are one multiply
//! `step as f64 * dt`.
//!
//! Mechanical `T + U` is not a compute-energy measurement. The meter reading
//! from the peer contract is [`Meter::Unmetered`].

/// Clock-contract document pin. Not the older joint-acceptance pin
/// `8dabe3a45e589908f524b595da09bbadbc8aa103`.
pub const FERROMORPHIC_CLOCK_REVISION: &str = "a6722b49ff7dc770b1cb062adac21aa5c7155401";
pub const FERROMORPHIC_CLOCK_PATH: &str = "docs/contracts/ferromotion-clock.md";

/// The schedule in that document was not executed in either repository.
pub const JOINT_LOOP_EXECUTED: bool = false;

/// Ferromorphic has no bind. Receipt text can round-trip without this becoming true.
/// It stays false until their bind matches the procedure in the clock contract and that
/// implementation is verified.
pub const FERROMORPHIC_BIND_VERIFIED: bool = false;

/// `0.1_f64` as named by the peer receipt.
pub const DT_POINT_ONE_BITS: u64 = 0x3fb9_9999_9999_999a;

/// Largest scheduling period whose integer value is a distinct `f64`.
/// `period_ns > MAX_EXACT_PERIOD_NS` is [`TimebaseError::InexactPeriod`].
pub const MAX_EXACT_PERIOD_NS: u64 = 1 << 53;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimebaseError {
    BadTick,
    DtMismatch,
    Late,
    BeyondHorizon,
    Overflow,
    Unordered,
    StampMismatch,
    MissingEpoch,
    AmbiguousEpoch,
    OffGrid,
    /// `dt` bits are not `((period_ns as f64) * 1e-9).to_bits()`. Envelope v1 has no ratio exemption.
    DtPeriodMismatch,
    /// `period_ns > 2^53`. Above that bound two integers can share one `f64`, so
    /// `(period_ns as f64) * 1e-9` would collide. `2^53` itself is accepted.
    InexactPeriod,
    /// `episode_start` would advance the episode identity past `u64::MAX`.
    EpochOverflow,
    /// The clock's episode identity is not the envelope's or the event's identity.
    StaleEpoch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Meter {
    /// No wall, board, or SoC sample. Not mechanical `T + U`.
    Unmetered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickPhase {
    Deliveries,
    NeuronUpdates,
    Enqueue,
    Increment,
}

pub const TICK_PHASES: [TickPhase; 4] = [
    TickPhase::Deliveries,
    TickPhase::NeuronUpdates,
    TickPhase::Enqueue,
    TickPhase::Increment,
];

/// Foreman decision for this adapter: ticks are `u64` and every sum is checked.
/// Overflow is a hard error. Values are never wrapped. Peer `Sim::step` is not
/// changed by this crate, so this is not yet an interoperability claim.
pub const TICK_ARITHMETIC: &str = "checked_u64_fail_closed";

/// Still open with the peer: `DynamicsLab` keeps its accumulated `f64` time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnresolvedPolicy {
    /// `DynamicsLab` keeps its `f64` accumulator. Replacing that with
    /// `step as f64 * dt` inside the lab is not done here.
    AccumulatorRemainsInsideDynamicsLab,
}

pub const UNRESOLVED: [UnresolvedPolicy; 1] = [UnresolvedPolicy::AccumulatorRemainsInsideDynamicsLab];

/// Outer episode envelope. Version 1 is a proposal. It is not codec v2 and it
/// is not part of the v2 digest. Peers have not accepted it. Receipt v2 text
/// matching the pinned ferromorphic bytes is not this flag.
pub const EPISODE_ENVELOPE_VERSION: u32 = 1;
pub const EPISODE_ENVELOPE_INTEROPERABLE: bool = false;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spike {
    pub t: u64,
    pub source: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepReceipt {
    /// `2`. Version 1 has no epoch line and is refused.
    pub version: u32,
    /// Monotonic episode identity. Not the episode tick, which is [`EPOCH_TICK0`].
    pub epoch: u64,
    pub step: u64,
    pub dt_bits: u64,
    pub mode: String,
    pub spikes: Vec<Spike>,
}

/// Public fields are an untrusted snapshot. [`BoundaryClock::episode_start`] checks
/// the forward mapping only at that call and does not rewrite `dt_bits`. A later write
/// to `tick`, `dt_bits`, `period_ns`, `origin_ns`, or `epoch` is not trusted.
/// [`bind_v2_timestamps`] rechecks the epoch, origin, period, and `dt` bits against
/// the envelope on every call and keeps no validity bit. A private [`EpisodeEnvelope`]
/// does not make this state trusted, and a successful bind does not execute a joint step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryClock {
    pub tick: u64,
    pub dt_bits: u64,
    /// Integer nanosecond scheduling period. Unset until [`BoundaryClock::episode_start`].
    /// Once a period is paired with `dt_bits`, those bits must equal
    /// `((period_ns as f64) * 1e-9).to_bits()`. Bind checks that pair again.
    pub period_ns: Option<u64>,
    /// Nanosecond stamp of tick 0. Not a v2 `t0_ns` field.
    pub origin_ns: Option<u64>,
    /// Spike vector. [`BoundaryClock::episode_start`] and [`BoundaryClock::reset_state`]
    /// both leave it unchanged.
    pub ledger: Vec<Spike>,
    pub neurons_cleared: bool,
    pub ring_cleared: bool,
    /// Monotonic episode identity. [`BoundaryClock::new`] opens `0`.
    /// [`BoundaryClock::episode_start`] advances it by one. [`BoundaryClock::reset_state`]
    /// does not write it. This is not [`EPOCH_TICK0`].
    pub epoch: u64,
}

/// Episode tick 0. Not a caller-chosen parameter.
pub const EPOCH_TICK0: u64 = 0;

impl BoundaryClock {
    pub fn new(tick: u64, dt: f64) -> Result<Self, TimebaseError> {
        if !finite_positive(dt) {
            return Err(TimebaseError::BadTick);
        }
        Ok(BoundaryClock {
            tick,
            dt_bits: dt.to_bits(),
            period_ns: None,
            origin_ns: None,
            ledger: Vec::new(),
            neurons_cleared: false,
            ring_cleared: false,
            epoch: 0,
        })
    }

    /// Open the next episode at tick 0 and install a scheduling period.
    ///
    /// `dt` is not an argument and is not rewritten. The bits already stored must equal
    /// `((period_ns as f64) * 1e-9).to_bits()`. The episode identity advances by one.
    /// Neuron and ring flags are cleared. The spike ledger is kept. This is the only
    /// call that advances `epoch`. [`BoundaryClock::reset_state`] does not.
    /// A refusal leaves every field unchanged.
    ///
    /// There is no `tick0` argument. The episode tick origin stays [`EPOCH_TICK0`].
    pub fn episode_start(&mut self, origin_ns: u64, period_ns: u64) -> Result<(), TimebaseError> {
        require_default_mapping(self.dt_bits, period_ns)?;
        let next = self.epoch.checked_add(1).ok_or(TimebaseError::EpochOverflow)?;
        self.epoch = next;
        self.tick = EPOCH_TICK0;
        self.origin_ns = Some(origin_ns);
        self.period_ns = Some(period_ns);
        self.neurons_cleared = true;
        self.ring_cleared = true;
        Ok(())
    }

    /// Next tick. Checked add. On overflow the tick is left unchanged.
    pub fn advance(&mut self) -> Result<u64, TimebaseError> {
        self.tick = self.tick.checked_add(1).ok_or(TimebaseError::Overflow)?;
        Ok(self.tick)
    }

    /// State-only reset: neurons and the ring clear. Tick, ledger, epoch,
    /// `period_ns`, origin, and `dt` bits stay. An episode is not started here.
    pub fn reset_state(&mut self) {
        self.neurons_cleared = true;
        self.ring_cleared = true;
    }
}

/// Proposal carried beside a codec v2 chunk. The v2 bytes are not rewritten.
/// Fields are private so a struct literal cannot skip [`EpisodeEnvelope::try_new`].
/// Envelope v1 has no physical-time ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpisodeEnvelope {
    version: u32,
    epoch: u64,
    dt_bits: u64,
    period_ns: u64,
    origin_ns: u64,
}

impl EpisodeEnvelope {
    /// Accepts only `dt_bits == ((period_ns as f64) * 1e-9).to_bits()`.
    /// `epoch` is the monotonic episode identity carried beside the chunk. It is not written
    /// into the codec v2 bytes.
    pub fn try_new(version: u32, epoch: u64, dt_bits: u64, period_ns: u64, origin_ns: u64) -> Result<Self, TimebaseError> {
        if version != EPISODE_ENVELOPE_VERSION {
            return Err(TimebaseError::OffGrid);
        }
        require_default_mapping(dt_bits, period_ns)?;
        Ok(EpisodeEnvelope { version, epoch, dt_bits, period_ns, origin_ns })
    }

    pub fn version(self) -> u32 {
        self.version
    }
    pub fn epoch(self) -> u64 {
        self.epoch
    }
    pub fn dt_bits(self) -> u64 {
        self.dt_bits
    }
    pub fn period_ns(self) -> u64 {
        self.period_ns
    }
    pub fn origin_ns(self) -> u64 {
        self.origin_ns
    }
}

/// `period_ns > 2^53` is refused. Every integer through [`MAX_EXACT_PERIOD_NS`] has its own `f64`.
pub fn require_exact_period(period_ns: u64) -> Result<(), TimebaseError> {
    if period_ns > MAX_EXACT_PERIOD_NS {
        Err(TimebaseError::InexactPeriod)
    } else {
        Ok(())
    }
}

/// `((period_ns as f64) * 1e-9).to_bits()` must equal `dt_bits`. No other mapping is accepted.
/// A period above [`MAX_EXACT_PERIOD_NS`] is [`TimebaseError::InexactPeriod`] before that comparison.
pub fn require_default_mapping(dt_bits: u64, period_ns: u64) -> Result<(), TimebaseError> {
    let dt = f64::from_bits(dt_bits);
    if !finite_positive(dt) || period_ns == 0 {
        return Err(TimebaseError::BadTick);
    }
    require_exact_period(period_ns)?;
    let forward = (period_ns as f64) * 1e-9;
    if !forward.is_finite() || forward <= 0.0 {
        return Err(TimebaseError::BadTick);
    }
    if dt_bits != forward.to_bits() {
        return Err(TimebaseError::DtPeriodMismatch);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProposedJointStep {
    pub k: u64,
    pub sense_at: u64,
    pub control_uses_spikes_before: u64,
    pub torque_held_one_dt: bool,
    pub spikes_visible_at: Result<u64, TimebaseError>,
    pub executed: bool,
}

pub fn finite_positive(dt: f64) -> bool {
    dt.is_finite() && dt > 0.0
}

/// Both finite and positive, and the same `f64` bits.
pub fn same_dt(a: f64, b: f64) -> bool {
    finite_positive(a) && finite_positive(b) && a.to_bits() == b.to_bits()
}

pub fn bind_dt(plant: f64, peer: f64) -> Result<u64, TimebaseError> {
    if !finite_positive(plant) || !finite_positive(peer) {
        return Err(TimebaseError::BadTick);
    }
    if plant.to_bits() != peer.to_bits() {
        return Err(TimebaseError::DtMismatch);
    }
    Ok(plant.to_bits())
}

/// One multiply. Not the sum of `steps` additions of `dt`.
pub fn step_seconds(step: u64, dt: f64) -> Result<f64, TimebaseError> {
    if !finite_positive(dt) {
        return Err(TimebaseError::BadTick);
    }
    Ok(step as f64 * dt)
}

/// `step + 1 + delay`, checked. Zero delay is the next tick.
pub fn delivery_step(step: u64, delay: u64) -> Result<u64, TimebaseError> {
    step.checked_add(1)
        .and_then(|next| next.checked_add(delay))
        .ok_or(TimebaseError::Overflow)
}

pub fn accept_event_step(event: u64, clock: u64, max_delay: u64) -> Result<(), TimebaseError> {
    let horizon = clock.checked_add(max_delay).ok_or(TimebaseError::Overflow)?;
    if event < clock {
        Err(TimebaseError::Late)
    } else if event > horizon {
        Err(TimebaseError::BeyondHorizon)
    } else {
        Ok(())
    }
}

/// Clocked refractory length `ceil(t_ref / dt)` in ticks.
pub fn refractory_ticks(t_ref: f64, dt: f64) -> Result<u64, TimebaseError> {
    if !finite_positive(dt) || !t_ref.is_finite() || t_ref < 0.0 {
        return Err(TimebaseError::BadTick);
    }
    let ticks = (t_ref / dt).ceil();
    if !ticks.is_finite() || ticks < 0.0 || ticks > u64::MAX as f64 {
        return Err(TimebaseError::Overflow);
    }
    Ok(ticks as u64)
}

/// Stable order of `(t, source)`. Equal keys keep emission order.
pub fn order_spikes(spikes: &mut [Spike]) {
    spikes.sort_by(|a, b| a.t.cmp(&b.t).then(a.source.cmp(&b.source)));
}

pub fn meter_reading() -> Meter {
    Meter::Unmetered
}

/// Forward product `((period_ns as f64) * 1e-9)`. A zero period is refused.
/// [`require_default_mapping`] compares `dt` bits to this value. There is no ratio exemption.
pub fn nominal_seconds(period_ns: u64) -> Result<f64, TimebaseError> {
    if period_ns == 0 {
        return Err(TimebaseError::BadTick);
    }
    require_exact_period(period_ns)?;
    let seconds = (period_ns as f64) * 1e-9;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(TimebaseError::BadTick);
    }
    Ok(seconds)
}

/// Nanosecond stamp of `tick` on an episode whose tick 0 is [`EPOCH_TICK0`]:
/// `origin_ns + tick * period_ns`, checked.
pub fn tick_stamp_ns(origin_ns: u64, period_ns: u64, tick: u64) -> Result<u64, TimebaseError> {
    if period_ns == 0 {
        return Err(TimebaseError::BadTick);
    }
    let delta = tick.checked_mul(period_ns).ok_or(TimebaseError::Overflow)?;
    origin_ns.checked_add(delta).ok_or(TimebaseError::Overflow)
}

/// Bind a codec v2 chunk to an episode without reading or writing its canonical bytes.
///
/// The clock argument is an untrusted per-call snapshot. This function does not keep a
/// validity bit from an earlier call and does not write `clock`. On every call it:
///
/// 1. Refuses a missing envelope with [`TimebaseError::MissingEpoch`].
/// 2. Refuses an envelope version other than [`EPISODE_ENVELOPE_VERSION`], or a zero
///    period, with [`TimebaseError::OffGrid`].
/// 3. Rechecks `dt_bits == ((period_ns as f64) * 1e-9).to_bits()` on the envelope.
///    There is no ratio exemption. A period above [`MAX_EXACT_PERIOD_NS`] is
///    [`TimebaseError::InexactPeriod`] before that comparison.
/// 4. Refuses a clock `period_ns` or a `step_duration_ns` above [`MAX_EXACT_PERIOD_NS`]
///    with [`TimebaseError::InexactPeriod`].
/// 5. Refuses a clock whose `period_ns` or `origin_ns` is unset with
///    [`TimebaseError::MissingEpoch`]. Identity 0 alone is not a bindable episode.
/// 6. Refuses `clock.epoch != envelope.epoch()` with [`TimebaseError::StaleEpoch`].
/// 7. Refuses an origin or period that differs from the envelope with
///    [`TimebaseError::AmbiguousEpoch`].
/// 8. Refuses `dt` bits that differ with [`TimebaseError::DtMismatch`].
/// 9. Requires `step_duration_ns == period_ns` and `t0_ns` on
///    `origin_ns + n * period_ns`.
///
/// `n_action_steps == 0` still runs those checks and stamps no index. The call does
/// not read `tick` as authorization. A match does not run inference or a joint step
/// and does not set [`EPISODE_ENVELOPE_INTEROPERABLE`]. Ferromorphic does not implement
/// this function. [`FERROMORPHIC_BIND_VERIFIED`] stays false.
pub fn bind_v2_timestamps(t0_ns: u64, step_duration_ns: u64, n_action_steps: u32, envelope: Option<EpisodeEnvelope>, clock: &BoundaryClock) -> Result<(), TimebaseError> {
    let Some(envelope) = envelope else {
        return Err(TimebaseError::MissingEpoch);
    };
    if envelope.version() != EPISODE_ENVELOPE_VERSION || envelope.period_ns() == 0 {
        return Err(TimebaseError::OffGrid);
    }
    require_default_mapping(envelope.dt_bits(), envelope.period_ns())?;
    if clock.period_ns.is_some_and(|period| period > MAX_EXACT_PERIOD_NS) || step_duration_ns > MAX_EXACT_PERIOD_NS {
        return Err(TimebaseError::InexactPeriod);
    }
    if clock.period_ns.is_none() || clock.origin_ns.is_none() {
        return Err(TimebaseError::MissingEpoch);
    }
    if clock.epoch != envelope.epoch() {
        return Err(TimebaseError::StaleEpoch);
    }
    if clock.origin_ns != Some(envelope.origin_ns()) || clock.period_ns != Some(envelope.period_ns()) {
        return Err(TimebaseError::AmbiguousEpoch);
    }
    if clock.dt_bits != envelope.dt_bits() {
        return Err(TimebaseError::DtMismatch);
    }
    if step_duration_ns != envelope.period_ns() || t0_ns < envelope.origin_ns() {
        return Err(TimebaseError::OffGrid);
    }
    let delta = t0_ns - envelope.origin_ns();
    if !delta.is_multiple_of(envelope.period_ns()) {
        return Err(TimebaseError::OffGrid);
    }
    let start = delta / envelope.period_ns();
    for k in 0..n_action_steps {
        let from_v2 = t0_ns
            .checked_add((k as u64).checked_mul(step_duration_ns).ok_or(TimebaseError::Overflow)?)
            .ok_or(TimebaseError::Overflow)?;
        let from_grid = tick_stamp_ns(
            envelope.origin_ns(),
            envelope.period_ns(),
            start.checked_add(k as u64).ok_or(TimebaseError::Overflow)?,
        )?;
        if from_v2 != from_grid {
            return Err(TimebaseError::OffGrid);
        }
    }
    let _ = (EPISODE_ENVELOPE_INTEROPERABLE, FERROMORPHIC_BIND_VERIFIED, JOINT_LOOP_EXECUTED);
    Ok(())
}

/// Refuse an event whose episode identity is not the clock's, then apply [`accept_event_step`].
///
/// This adapter has no delivery ring, so `max_delay` stays a caller argument.
/// [`accept_event_step`] itself does not read `epoch`.
pub fn accept_event(event_epoch: u64, event_step: u64, clock: &BoundaryClock, max_delay: u64) -> Result<(), TimebaseError> {
    if event_epoch != clock.epoch {
        return Err(TimebaseError::StaleEpoch);
    }
    accept_event_step(event_step, clock.tick, max_delay)
}

/// Result of counting a chunk index from tick 0. `EpochUnresolved` means the caller
/// said no episode has started. It is not a second epoch and it does not authorize a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkTickMap {
    EpochUnresolved,
    Mapped { tick: u64, spikes_visible_at: u64 },
}

/// `index` is a tick counted from [`EPOCH_TICK0`]. No other epoch tick is accepted.
pub fn map_chunk_index_to_tick(index: u64, episode_started: bool, delay: u64) -> Result<ChunkTickMap, TimebaseError> {
    if !episode_started {
        return Ok(ChunkTickMap::EpochUnresolved);
    }
    let tick = EPOCH_TICK0.checked_add(index).ok_or(TimebaseError::Overflow)?;
    Ok(ChunkTickMap::Mapped { tick, spikes_visible_at: delivery_step(tick, delay)? })
}

pub fn propose_joint_step(k: u64, delay: u64) -> ProposedJointStep {
    ProposedJointStep {
        k,
        sense_at: k,
        control_uses_spikes_before: k,
        torque_held_one_dt: true,
        spikes_visible_at: delivery_step(k, delay),
        executed: false,
    }
}

/// Adding `dt` once per step is the lab accumulator. It is not [`step_seconds`].
pub fn accumulated_f64(steps: u64, dt: f64) -> f64 {
    let mut acc = 0.0;
    for _ in 0..steps {
        acc += dt;
    }
    acc
}

impl StepReceipt {
    /// Canonical receipt v2 text. The first line is `ferromorphic-step 2` and the second is
    /// `epoch`. `dt_bits` is 16 lowercase hex digits. Ferromorphic `StepReceipt::from_text` at
    /// the pinned revision accepts this shape.
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "ferromorphic-step 2\nepoch {}\nstep {}\ndt_bits {:016x}\nmode {}\n",
            self.epoch, self.step, self.dt_bits, self.mode
        );
        for spike in &self.spikes {
            out.push_str(&format!("spike {} {}\n", spike.t, spike.source));
        }
        out
    }
}

fn parse_u64_field(line: Option<&str>, key: &str) -> Result<u64, TimebaseError> {
    let line = line.ok_or(TimebaseError::BadTick)?;
    let rest = line.strip_prefix(key).and_then(|s| s.strip_prefix(' ')).ok_or(TimebaseError::BadTick)?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TimebaseError::BadTick);
    }
    rest.parse().map_err(|_| TimebaseError::BadTick)
}

/// Parse receipt v2. The header is exactly `ferromorphic-step 2`, then `epoch`, `step`,
/// `dt_bits`, and `mode`, in that order. Version 1 is refused: that record has no epoch line.
/// A decreasing `(t, source)` is [`TimebaseError::Unordered`]. A spike tick other than `step`
/// is [`TimebaseError::StampMismatch`]. Ferromorphic's parser reports that second case as
/// `Unordered`; the bytes of a valid record are the same either way.
pub fn parse_step_receipt(text: &str) -> Result<StepReceipt, TimebaseError> {
    let mut lines = text.lines();
    if lines.next() != Some("ferromorphic-step 2") {
        return Err(TimebaseError::BadTick);
    }
    let epoch = parse_u64_field(lines.next(), "epoch")?;
    let step = parse_u64_field(lines.next(), "step")?;
    let bits_line = lines.next().ok_or(TimebaseError::BadTick)?;
    let hex = bits_line.strip_prefix("dt_bits ").ok_or(TimebaseError::BadTick)?;
    if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(TimebaseError::BadTick);
    }
    let dt_bits = u64::from_str_radix(hex, 16).map_err(|_| TimebaseError::BadTick)?;
    let mode_line = lines.next().ok_or(TimebaseError::BadTick)?;
    let mode = match mode_line.strip_prefix("mode ") {
        Some(mode @ ("clocked" | "event")) => mode.to_string(),
        _ => return Err(TimebaseError::BadTick),
    };
    let mut spikes: Vec<Spike> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let rest = line.strip_prefix("spike ").ok_or(TimebaseError::BadTick)?;
        let (t_text, source_text) = rest.split_once(' ').ok_or(TimebaseError::BadTick)?;
        if t_text.is_empty()
            || source_text.is_empty()
            || !t_text.bytes().all(|b| b.is_ascii_digit())
            || !source_text.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(TimebaseError::BadTick);
        }
        let t = t_text.parse::<u64>().map_err(|_| TimebaseError::BadTick)?;
        let source = source_text.parse::<u32>().map_err(|_| TimebaseError::BadTick)?;
        if let Some(prev) = spikes.last()
            && (t, source) < (prev.t, prev.source)
        {
            return Err(TimebaseError::Unordered);
        }
        spikes.push(Spike { t, source });
    }
    if spikes.iter().any(|spike| spike.t != step) {
        return Err(TimebaseError::StampMismatch);
    }
    Ok(StepReceipt { version: 2, epoch, step, dt_bits, mode, spikes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dt_matches_only_on_bits_and_point_one_is_the_receipt_constant() {
        assert_eq!(0.1f64.to_bits(), DT_POINT_ONE_BITS);
        assert!(same_dt(0.1, 0.1));
        assert!(!same_dt(0.1, 0.1 + f64::EPSILON));
        assert_eq!(bind_dt(0.1, 0.1), Ok(DT_POINT_ONE_BITS));
        assert_eq!(bind_dt(0.1, 0.1 + f64::EPSILON), Err(TimebaseError::DtMismatch));
        assert_eq!(bind_dt(0.0, 0.0), Err(TimebaseError::BadTick));
        assert_eq!(bind_dt(f64::NAN, 0.1), Err(TimebaseError::BadTick));
        const { assert!(!JOINT_LOOP_EXECUTED); }
        const { assert!(!FERROMORPHIC_BIND_VERIFIED); }
        const { assert!(!EPISODE_ENVELOPE_INTEROPERABLE); }
        assert_eq!(FERROMORPHIC_CLOCK_REVISION, "a6722b49ff7dc770b1cb062adac21aa5c7155401");
        assert_ne!(FERROMORPHIC_CLOCK_REVISION, crate::joint_acceptance::FERROMORPHIC_REVISION);
    }

    #[test]
    fn shared_seconds_are_one_multiply_and_the_accumulator_can_differ() {
        assert_eq!(step_seconds(4, 0.1).unwrap().to_bits(), (4.0f64 * 0.1).to_bits());
        let acc = accumulated_f64(10, 0.1);
        let shared = step_seconds(10, 0.1).unwrap();
        assert_ne!(acc.to_bits(), shared.to_bits());
        assert_eq!(step_seconds(1, f64::INFINITY), Err(TimebaseError::BadTick));
    }

    #[test]
    fn delay_zero_is_the_next_tick_and_overflow_is_checked() {
        assert_eq!(delivery_step(4, 0), Ok(5));
        assert_eq!(delivery_step(4, 5), Ok(10));
        assert_eq!(delivery_step(u64::MAX, 0), Err(TimebaseError::Overflow));
        assert_eq!(delivery_step(u64::MAX - 1, 1), Err(TimebaseError::Overflow));
        assert_eq!(TICK_PHASES[0], TickPhase::Deliveries);
        assert_eq!(TICK_PHASES[3], TickPhase::Increment);
    }

    #[test]
    fn past_and_beyond_horizon_events_are_rejected() {
        assert_eq!(accept_event_step(4, 4, 5), Ok(()));
        assert_eq!(accept_event_step(9, 4, 5), Ok(()));
        assert_eq!(accept_event_step(3, 4, 5), Err(TimebaseError::Late));
        assert_eq!(accept_event_step(10, 4, 5), Err(TimebaseError::BeyondHorizon));
        assert_eq!(accept_event_step(u64::MAX, u64::MAX, 1), Err(TimebaseError::Overflow));
    }

    #[test]
    fn refractory_is_ceil_of_t_ref_over_dt() {
        assert_eq!(refractory_ticks(0.0, 0.1), Ok(0));
        assert_eq!(refractory_ticks(0.1, 0.1), Ok(1));
        assert_eq!(refractory_ticks(0.15, 0.1), Ok(2));
        assert_eq!(refractory_ticks(1.0, 0.3), Ok(4));
        assert_eq!(refractory_ticks(-0.1, 0.1), Err(TimebaseError::BadTick));
        assert_eq!(refractory_ticks(0.1, 0.0), Err(TimebaseError::BadTick));
    }

    #[test]
    fn spike_order_is_tick_then_source_and_ties_keep_emission_order() {
        let mut spikes = vec![
            Spike { t: 5, source: 1 },
            Spike { t: 4, source: 2 },
            Spike { t: 4, source: 0 },
            Spike { t: 4, source: 2 },
        ];
        order_spikes(&mut spikes);
        assert_eq!(spikes[0], Spike { t: 4, source: 0 });
        assert_eq!(spikes[1], Spike { t: 4, source: 2 });
        assert_eq!(spikes[2], Spike { t: 4, source: 2 });
        assert_eq!(spikes[3], Spike { t: 5, source: 1 });
    }

    #[test]
    fn reset_keeps_tick_ledger_and_the_opening_epoch() {
        let mut clock = BoundaryClock::new(7, 0.1).unwrap();
        clock.ledger.push(Spike { t: 6, source: 1 });
        clock.reset_state();
        assert!(clock.neurons_cleared && clock.ring_cleared);
        assert_eq!(clock.tick, 7);
        assert_eq!(clock.ledger, vec![Spike { t: 6, source: 1 }]);
        assert_eq!(clock.epoch, 0);
        assert_eq!(clock.period_ns, None);
        assert_eq!(clock.dt_bits, DT_POINT_ONE_BITS);
        assert_eq!(accept_event(0, 7, &clock, 0), Ok(()));
        assert_eq!(accept_event(1, 7, &clock, 0), Err(TimebaseError::StaleEpoch));
        assert!(UNRESOLVED.contains(&UnresolvedPolicy::AccumulatorRemainsInsideDynamicsLab));
        assert_eq!(meter_reading(), Meter::Unmetered);
    }

    #[test]
    fn proposed_joint_step_is_not_executed_and_spikes_wait_one_tick_plus_delay() {
        let step = propose_joint_step(4, 0);
        assert!(!step.executed);
        const { assert!(!JOINT_LOOP_EXECUTED); }
        assert_eq!(step.sense_at, 4);
        assert_eq!(step.control_uses_spikes_before, 4);
        assert!(step.torque_held_one_dt);
        assert_eq!(step.spikes_visible_at, Ok(5));
        assert_eq!(propose_joint_step(4, 5).spikes_visible_at, Ok(10));
    }

    #[test]
    fn the_pinned_receipt_v2_parses_and_version_1_is_refused() {
        let text = "\
ferromorphic-step 2
epoch 0
step 4
dt_bits 3fb999999999999a
mode clocked
spike 4 0
spike 4 1
";
        let receipt = parse_step_receipt(text).unwrap();
        assert_eq!(receipt.version, 2);
        assert_eq!(receipt.epoch, 0);
        assert_eq!(receipt.step, 4);
        assert_eq!(receipt.dt_bits, DT_POINT_ONE_BITS);
        assert_eq!(receipt.mode, "clocked");
        assert_eq!(receipt.spikes, vec![Spike { t: 4, source: 0 }, Spike { t: 4, source: 1 }]);
        assert_eq!(receipt.to_text(), text);
        let v1 = "\
ferromorphic-step 1
step 4
dt_bits 3fb999999999999a
mode clocked
spike 4 0
spike 4 1
";
        assert_eq!(parse_step_receipt(v1), Err(TimebaseError::BadTick));
        let missing_epoch = text.replacen("epoch 0\n", "", 1);
        assert_eq!(parse_step_receipt(&missing_epoch), Err(TimebaseError::BadTick));
        let reversed = text.replace("spike 4 0\nspike 4 1", "spike 4 1\nspike 4 0");
        assert_eq!(parse_step_receipt(&reversed), Err(TimebaseError::Unordered));
        let stamped = text.replace("spike 4 0", "spike 3 0");
        assert_eq!(parse_step_receipt(&stamped), Err(TimebaseError::StampMismatch));
        let later = text.replacen("epoch 0", "epoch 7", 1);
        let parsed = parse_step_receipt(&later).unwrap();
        assert_eq!(parsed.epoch, 7);
        assert_eq!(parsed.to_text(), later);
        assert_eq!(FERROMORPHIC_CLOCK_PATH, "docs/contracts/ferromotion-clock.md");
    }

    #[test]
    fn episode_start_sets_tick_zero_advances_epoch_and_keeps_the_ledger() {
        let mut clock = BoundaryClock::new(7, 0.1).unwrap();
        assert_eq!(clock.epoch, 0);
        clock.dt_bits = f64::NAN.to_bits();
        assert_eq!(clock.episode_start(1_000, 1_000), Err(TimebaseError::BadTick));
        clock.dt_bits = (-0.1f64).to_bits();
        assert_eq!(clock.episode_start(1_000, 1_000), Err(TimebaseError::BadTick));
        clock.dt_bits = 0.0f64.to_bits();
        assert_eq!(clock.episode_start(1_000, 1_000), Err(TimebaseError::BadTick));
        clock.dt_bits = f64::INFINITY.to_bits();
        assert_eq!(clock.episode_start(1_000, 1_000), Err(TimebaseError::BadTick));
        clock.dt_bits = DT_POINT_ONE_BITS;
        assert_eq!(clock.episode_start(1_000, 0), Err(TimebaseError::BadTick));
        assert_eq!(clock.episode_start(0, 1_000), Err(TimebaseError::DtPeriodMismatch));
        assert_eq!(clock.episode_start(0, 20_000_000), Err(TimebaseError::DtPeriodMismatch));
        assert_eq!(clock.tick, 7, "a refused start does not move the tick");
        assert_eq!(clock.epoch, 0, "a refused start does not advance the epoch");
        assert_eq!(clock.dt_bits, DT_POINT_ONE_BITS, "a refused start does not rewrite dt");
        let dt_1us = f64::from_bits(0x3eb0c6f7a0b5ed8e);
        assert_eq!(dt_1us.to_bits(), (1_000_f64 * 1e-9).to_bits());
        clock.dt_bits = dt_1us.to_bits();
        clock.ledger.push(Spike { t: 3, source: 1 });
        clock.episode_start(1_000, 1_000).unwrap();
        assert_eq!(clock.tick, 0);
        assert_eq!(clock.epoch, 1);
        assert_eq!(clock.dt_bits, 0x3eb0c6f7a0b5ed8e);
        assert_eq!(clock.ledger, vec![Spike { t: 3, source: 1 }]);
        assert!(clock.neurons_cleared && clock.ring_cleared);
        assert_eq!(clock.advance().unwrap(), 1);
        assert_eq!(clock.advance().unwrap(), 2);
        clock.reset_state();
        assert_eq!(clock.tick, 2);
        assert_eq!(clock.epoch, 1);
        assert_eq!(clock.period_ns, Some(1_000));
        assert_eq!(clock.origin_ns, Some(1_000));
        assert_eq!(clock.ledger.len(), 1);
        assert!(clock.neurons_cleared && clock.ring_cleared);
        clock.episode_start(1_000, 1_000).unwrap();
        assert_eq!(clock.tick, 0);
        assert_eq!(clock.epoch, 2);
        assert_eq!(clock.ledger.len(), 1, "episode_start keeps the spike ledger");
        assert_eq!(clock.dt_bits, 0x3eb0c6f7a0b5ed8e);
        clock.epoch = u64::MAX;
        assert_eq!(clock.episode_start(1_000, 1_000), Err(TimebaseError::EpochOverflow));
        assert_eq!(clock.epoch, u64::MAX);
        assert_eq!(clock.tick, 0);
        clock.epoch = 2;
        clock.tick = u64::MAX;
        assert_eq!(clock.advance(), Err(TimebaseError::Overflow));
        assert_eq!(clock.tick, u64::MAX);
        const { assert!(!EPISODE_ENVELOPE_INTEROPERABLE); }
        const { assert!(!FERROMORPHIC_BIND_VERIFIED); }
        assert_eq!(TICK_ARITHMETIC, "checked_u64_fail_closed");
        assert_eq!(EPOCH_TICK0, 0);
    }

    #[test]
    fn forward_product_mapping_rejects_mismatched_dt_and_accepts_1000ns() {
        const ONE_US: u64 = 0x3eb0c6f7a0b5ed8e;
        const TWENTY_MS: u64 = 0x3f947ae147ae147b;
        const TWENTY_MS_ONE_BIT_OFF: u64 = 0x3f947ae147ae147c;
        assert_eq!((1_000_f64 * 1e-9).to_bits(), ONE_US);
        assert_eq!((20_000_000_f64 * 1e-9).to_bits(), TWENTY_MS);
        assert_eq!(nominal_seconds(0), Err(TimebaseError::BadTick));
        assert_eq!(bind_dt(0.1, 0.1 + f64::EPSILON), Err(TimebaseError::DtMismatch));
        let mut clock = BoundaryClock::new(3, 1.0).unwrap();
        assert_eq!(clock.episode_start(0, 1_000), Err(TimebaseError::DtPeriodMismatch));
        assert_eq!(clock.episode_start(0, 20_000_000), Err(TimebaseError::DtPeriodMismatch));
        clock.dt_bits = TWENTY_MS_ONE_BIT_OFF;
        assert_eq!(clock.episode_start(0, 20_000_000), Err(TimebaseError::DtPeriodMismatch));
        assert_eq!(clock.tick, 3);
        assert_eq!(clock.epoch, 0);
        assert_eq!(clock.dt_bits, TWENTY_MS_ONE_BIT_OFF);
        assert_eq!(
            EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, 0.1f64.to_bits(), 1_000, 0),
            Err(TimebaseError::DtPeriodMismatch)
        );
        assert_eq!(
            EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, TWENTY_MS_ONE_BIT_OFF, 20_000_000, 0),
            Err(TimebaseError::DtPeriodMismatch)
        );
        clock.dt_bits = ONE_US;
        clock.episode_start(0, 1_000).unwrap();
        assert_eq!(clock.epoch, 1);
        assert_eq!(clock.dt_bits, ONE_US);
        let envelope = EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, ONE_US, 1_000, 0).unwrap();
        assert_eq!(bind_v2_timestamps(0, 1_000, 2, Some(envelope), &clock), Ok(()));
        assert_eq!(bind_v2_timestamps(0, 1_000, 0, Some(envelope), &clock), Ok(()));
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, None, &clock), Err(TimebaseError::MissingEpoch));
        let other_origin = EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, ONE_US, 1_000, 5).unwrap();
        assert_eq!(bind_v2_timestamps(5, 1_000, 1, Some(other_origin), &clock), Err(TimebaseError::AmbiguousEpoch));
        let stale = EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 0, ONE_US, 1_000, 0).unwrap();
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(stale), &clock), Err(TimebaseError::StaleEpoch));
        assert_eq!(bind_v2_timestamps(1, 1_000, 1, Some(envelope), &clock), Err(TimebaseError::OffGrid));
        assert_eq!(bind_v2_timestamps(0, 2_000, 1, Some(envelope), &clock), Err(TimebaseError::OffGrid));
        assert_eq!(tick_stamp_ns(0, 1_000, 3).unwrap(), 3_000);
        assert_eq!(tick_stamp_ns(u64::MAX, 1_000, 1), Err(TimebaseError::Overflow));
        let twenty = EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, TWENTY_MS, 20_000_000, 0).unwrap();
        assert_eq!(bind_v2_timestamps(0, 20_000_000, 1, Some(twenty), &clock), Err(TimebaseError::AmbiguousEpoch));
        let idle = BoundaryClock::new(3, f64::from_bits(ONE_US)).unwrap();
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &idle), Err(TimebaseError::MissingEpoch));
    }

    #[test]
    fn a_public_clock_mutation_is_rejected_on_the_next_bind() {
        const ONE_US: u64 = 0x3eb0c6f7a0b5ed8e;
        const TWENTY_MS: u64 = 0x3f947ae147ae147b;
        let mut clock = BoundaryClock::new(9, f64::from_bits(ONE_US)).unwrap();
        clock.episode_start(0, 1_000).unwrap();
        assert_eq!(clock.epoch, 1);
        let envelope = EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, ONE_US, 1_000, 0).unwrap();
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Ok(()));
        clock.dt_bits ^= 1;
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Err(TimebaseError::DtMismatch));
        assert_eq!(bind_v2_timestamps(0, 1_000, 0, Some(envelope), &clock), Err(TimebaseError::DtMismatch));
        clock.dt_bits = ONE_US;
        clock.period_ns = Some(2_000);
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Err(TimebaseError::AmbiguousEpoch));
        clock.period_ns = Some(1_000);
        clock.origin_ns = Some(1);
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Err(TimebaseError::AmbiguousEpoch));
        clock.origin_ns = Some(0);
        clock.epoch = 2;
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Err(TimebaseError::StaleEpoch));
        clock.epoch = 1;
        clock.period_ns = None;
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Err(TimebaseError::MissingEpoch));
        clock.period_ns = Some(1_000);
        clock.tick = 40;
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Ok(()));
        assert!(!propose_joint_step(0, 0).executed);
        clock.dt_bits = TWENTY_MS;
        clock.period_ns = Some(20_000_000);
        assert_eq!(bind_v2_timestamps(0, 1_000, 1, Some(envelope), &clock), Err(TimebaseError::AmbiguousEpoch));
        assert_eq!(
            EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, 0.1f64.to_bits(), 1_000, 0),
            Err(TimebaseError::DtPeriodMismatch)
        );
    }

    #[test]
    fn a_period_above_2_pow_53_is_inexact_at_construction_start_and_bind() {
        const POW: u64 = 1 << 53;
        const NEXT: u64 = POW + 1;
        assert_eq!(MAX_EXACT_PERIOD_NS, POW);
        assert_eq!((POW as f64) as u64, POW);
        assert_ne!((NEXT as f64) as u64, NEXT);
        assert_eq!((POW as f64).to_bits(), (NEXT as f64).to_bits());
        let dt = (POW as f64) * 1e-9;
        assert!(dt.is_finite() && dt > 0.0);
        let collided = (NEXT as f64) * 1e-9;
        assert_eq!(dt.to_bits(), collided.to_bits());

        assert_eq!(
            EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, dt.to_bits(), NEXT, 0),
            Err(TimebaseError::InexactPeriod)
        );
        assert_eq!(
            EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, 1, dt.to_bits(), u64::MAX, 0),
            Err(TimebaseError::InexactPeriod)
        );

        let mut clock = BoundaryClock::new(6, dt).unwrap();
        let untouched = clock.clone();
        assert_eq!(clock.episode_start(0, NEXT), Err(TimebaseError::InexactPeriod));
        assert_eq!(clock.episode_start(0, u64::MAX), Err(TimebaseError::InexactPeriod));
        assert_eq!(clock, untouched);
        clock.episode_start(0, POW).unwrap();
        assert_eq!(clock.epoch, 1);
        assert_eq!(clock.tick, 0);
        assert_eq!(clock.period_ns, Some(POW));
        assert_eq!(clock.dt_bits, dt.to_bits());
        let envelope = EpisodeEnvelope::try_new(EPISODE_ENVELOPE_VERSION, clock.epoch, dt.to_bits(), POW, 0).unwrap();
        assert_eq!(bind_v2_timestamps(0, POW, 1, Some(envelope), &clock), Ok(()));
        assert_eq!(bind_v2_timestamps(0, POW, 0, Some(envelope), &clock), Ok(()));

        clock.period_ns = Some(NEXT);
        assert_eq!(bind_v2_timestamps(0, POW, 1, Some(envelope), &clock), Err(TimebaseError::InexactPeriod));
        clock.period_ns = Some(u64::MAX);
        assert_eq!(bind_v2_timestamps(0, POW, 1, Some(envelope), &clock), Err(TimebaseError::InexactPeriod));
        clock.period_ns = Some(POW);
        assert_eq!(bind_v2_timestamps(0, NEXT, 1, Some(envelope), &clock), Err(TimebaseError::InexactPeriod));
        assert_eq!(bind_v2_timestamps(0, u64::MAX, 0, Some(envelope), &clock), Err(TimebaseError::InexactPeriod));
        assert_eq!(nominal_seconds(POW).unwrap().to_bits(), dt.to_bits());
        assert_eq!(nominal_seconds(NEXT), Err(TimebaseError::InexactPeriod));
        assert_eq!(nominal_seconds(u64::MAX), Err(TimebaseError::InexactPeriod));
        const { assert!(!EPISODE_ENVELOPE_INTEROPERABLE); }
        const { assert!(!FERROMORPHIC_BIND_VERIFIED); }
        const { assert!(!JOINT_LOOP_EXECUTED); }
    }
}
