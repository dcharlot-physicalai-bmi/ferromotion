# Proposal: a minimal closed loop

Not built. `JOINT_LOOP_EXECUTED` stays false. `EPISODE_ENVELOPE_INTEROPERABLE` stays false. `FERROMORPHIC_BIND_VERIFIED` stays false. This document names the calls a later loop would make. It does not run them.

The loop is N steps. Each step is one ferromorphic tick, the spikes that tick emitted, one ferromotion action chunk for that tick, one stamped receipt, and one bind of the next snapshot. Both sides hash the bytes they emitted. The hashes are the trace. A matching hash is not authorization to actuate.

## One step

1. Ferromorphic holds a `Sim<N>`. Before the first step, `episode_start(origin_ns, period_ns)` installs the shared period. `dt` is not rewritten. The call refuses unless `dt.to_bits()` equals `nominal_seconds(period_ns).to_bits()`.
2. Copy `clock_snapshot()`. That copy is the untrusted snapshot for this step. The loop does not keep a validity bit.
3. `try_step(external)` advances one tick and returns the neuron indexes that fired. `t` is the tick just completed only after the call returns; the receipt uses the tick from before the call.
4. `StepReceipt::from_step(epoch, tick_before, dt, mode, &fired)` then `to_text()`. Ferromotion `parse_step_receipt` reads those bytes. SHA-256 of the text is ferromorphic's trace line.
5. Ferromotion builds one `ChunkV2` whose `t0_ns` is `origin_ns + tick_before * period_ns`, `step_duration_ns` is `period_ns`, and `n_action_steps` is the chunk width for this step. `encode` produces the codec v2 body. SHA-256 of that body is ferromotion's trace line. The body is not written into the receipt text.
6. `EpisodeEnvelope::try_new(1, epoch, dt.to_bits(), period_ns, origin_ns)`.
7. `bind_v2_timestamps(t0_ns, period_ns, n_action_steps, Some(envelope), &snapshot)` on the snapshot from step 2, or on a fresh `clock_snapshot()` taken after the step if the comparison is the post-step clock. The proposal uses the post-step snapshot for the next bind, and the pre-step snapshot only to show that a mutation is visible. A refusal stops the loop. `Ok` does not authorize a rollout.
8. Repeat from step 2 until N steps, or until a refusal.

`external` for step k+1 may be a function of the decoded chunk from step k. That function is not specified here. Until it is, the proposal uses an empty current vector, which `try_step` already accepts.

## What ferromotion needs from ferromorphic

Pinned at `a6722b49ff7dc770b1cb062adac21aa5c7155401`, `src/sim.rs`. All of these are public today.

| item | signature |
|---|---|
| `Sim::new` | `fn new(net: Net, neurons: Vec<N>, dt: f64, mode: Mode) -> Result<Sim<N>, SimError>` |
| `Sim::episode_start` | `fn episode_start(&mut self, origin_ns: u64, period_ns: u64) -> Result<u64, TimebaseError>` |
| `Sim::clock_snapshot` | `fn clock_snapshot(&self) -> ClockSnapshot` |
| `Sim::try_step` | `fn try_step(&mut self, external: &[f64]) -> Result<Vec<u32>, TimebaseError>` |
| `ClockSnapshot` | `epoch: u64`, `tick: u64`, `dt_bits: u64`, `period_ns: Option<u64>`, `origin_ns: Option<u64>` |
| `EpisodeEnvelope::try_new` | `fn try_new(version: u32, epoch: u64, dt_bits: u64, period_ns: u64, origin_ns: u64) -> Result<EpisodeEnvelope, TimebaseError>` |
| `bind_v2_timestamps` | `fn bind_v2_timestamps(t0_ns: u64, step_duration_ns: u64, n_action_steps: u32, envelope: Option<EpisodeEnvelope>, clock: &ClockSnapshot) -> Result<(), TimebaseError>` |
| `nominal_seconds` | `fn nominal_seconds(period_ns: u64) -> Result<f64, TimebaseError>` |
| `StepReceipt::from_step` | `fn from_step(epoch: u64, step: u64, dt: f64, mode: Mode, fired: &[u32]) -> StepReceipt` |
| `StepReceipt::to_text` | `fn to_text(&self) -> String` |
| readable fields | `Sim.dt: f64`, `Sim.t: u64`, `Sim.epoch: u64`, `Sim.mode: Mode`, `Sim.origin_ns: Option<u64>`, `Sim.period_ns: Option<u64>` |

`TimebaseError` variants the loop has to match exactly, not by display string: `InexactPeriod`, `DtPeriodMismatch { dt_bits, period_ns, nominal_bits }`, `StaleEpoch { event, clock }`, `AmbiguousEpoch`, `Mismatch { left, right }`, `OffGrid`, `MissingEpoch`, `ScheduleOverflow { tick, period_ns }`, `BadExternal { got, want }`, `Overflow { step, delay }`, `EpochOverflow`.

Determinism required from ferromorphic, and not yet a joint measurement:

* `Mode::Clocked` and `Mode::EventDriven` on the same net, the same `dt`, and the same `external` produce the same `Vec<u32>` from `try_step`, including order. Event-driven sorts the touched indexes before firing.
* `StepReceipt::to_text` is a pure function of `from_step`'s arguments. SHA-256 of that string is the spike trace.
* `bind_v2_timestamps` does not write `ClockSnapshot` and does not store a validity bit. A later write to `dt`, `t`, `epoch`, `origin_ns`, or `period_ns` is visible on the next call.
* `n_action_steps == 0` still runs the checks and stamps no index. It returns `Ok(())` when the checks pass. It is not a refusal.
* `period_ns > 2^53` and `step_duration_ns > 2^53` are `InexactPeriod`. `2^53` is accepted. `u64::MAX` is `InexactPeriod`.

Missing for the loop, so it is not built:

* No function returns the SHA-256 of a `StepReceipt`. The loop would hash `to_text()` itself.
* No function applies a ferromotion `ChunkV2` body as `external`. That adapter is ferromotion's, and it is not written.
* `EpisodeEnvelope` cannot be built with a forward-product mismatch. `try_new` refuses with `DtPeriodMismatch` first. `bind_v2_timestamps` has that check, and no public value can reach it.

## What ferromotion provides

Already in this crate. None of it runs the loop.

| item | signature |
|---|---|
| `ChunkV2` | `t0_ns: u64`, `step_duration_ns: u64`, `n_action_steps: u32`, `action_dim: u32`, `values_time_major: Vec<f32>`, plus the provenance strings |
| `encode` | `fn encode(chunk: &ChunkV2) -> Result<Vec<u8>, CodecError>` |
| `check_horizon` | `fn check_horizon(n_action_steps: u32, chunk_size: u32) -> Result<(), CodecError>` |
| `action_timestamp_ns` | `fn action_timestamp_ns(t0_ns: u64, step_duration_ns: u64, index: u32, steps: u32) -> Result<u64, CodecError>` |
| `parse_step_receipt` | `fn parse_step_receipt(text: &str) -> Result<StepReceipt, TimebaseError>` |
| local `bind_v2_timestamps` | same argument order, on `ferromotion_core::BoundaryClock`, not on `ferromorphic::sim::ClockSnapshot` |

The local bind is a separate function. A test that calls it is not a call into ferromorphic. The cross-crate file `ferromorphic_bind_v2.rs` calls `ferromorphic::sim::bind_v2_timestamps`.

SHA-256 of `encode`'s bytes is ferromotion's trace line. `authorize_rollout` still returns `EntryProposed`. The loop must not treat a matching bind or a matching digest as permission to actuate.
