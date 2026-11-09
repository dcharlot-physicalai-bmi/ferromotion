# Ferromotion side of the ferromorphic step clock

Read-only source: [ferromorphic `docs/contracts/ferromotion-clock.md`](https://github.com/dcharlot-physicalai-bmi/ferromorphic/blob/15e219b8f7f1bc76029c359df676cbbcf9817ec0/docs/contracts/ferromotion-clock.md)
at `15e219b8f7f1bc76029c359df676cbbcf9817ec0`.

That pin is the clock document. It is not `FERROMORPHIC_REVISION` in
`joint_acceptance` (`8dabe3a45e589908f524b595da09bbadbc8aa103`). No peer repository
was modified.

The adapter is `crates/ferromotion-core/src/ferromorphic_clock.rs`. It does not
call `Sim::step` or `DynamicsLab::step`. `JOINT_LOOP_EXECUTED` is false.
`EPISODE_ENVELOPE_INTEROPERABLE` is false. `FERROMORPHIC_BIND_VERIFIED` is false.
A receipt-v2 text round trip is none of those three claims, and it does not authorize
inference or physical actuation.

## What is implemented

* `same_dt` / `bind_dt`: both values finite and positive, and `to_bits` equal. `0.1` and `0.1 + f64::EPSILON` do not match. `0.1_f64` bits are `3fb999999999999a`.
* `step_seconds(step, dt)` is one multiply `step as f64 * dt`.
* `accumulated_f64` is the repeated addition `DynamicsLab` uses. Ten additions of `0.1` are not the bits of `10 as f64 * 0.1`. The lab function is unchanged.
* Delivery is checked `step + 1 + delay`. Delay 0 is the next tick. Delay 5 from step 4 is step 10. Overflow is `TimebaseError::Overflow`, not a wrap.
* `accept_event_step` accepts `clock..=clock+max_delay`. Earlier is `Late`. Later is `BeyondHorizon`. A horizon that does not fit in `u64` is `Overflow`. It does not read an episode identity. `accept_event` refuses a different identity with `StaleEpoch`, then calls `accept_event_step`. This adapter has no ring, so `max_delay` stays a caller argument.
* Spike order is `(t, source)`, stable for equal keys.
* `reset_state` clears the neuron and ring flags and leaves tick, ledger, epoch, period, origin, and `dt` bits unchanged.
* Refractory length is `ceil(t_ref / dt)` ticks.
* Receipt v2 parses. The pinned sample below is the bytes of ferromorphic `StepReceipt::to_text` at this pin. A decreasing `(t, source)` is `Unordered`. A spike tick other than the receipt step is `StampMismatch` on this receiver. Their `from_text` reports that second case as `Unordered`. Valid records use the same bytes either way.
* `meter_reading()` is `Unmetered`. Mechanical `T + U` from `DynamicsLab::energy` is not that reading.

## Epoch identity

The episode identity is one monotonic `u64`. `BoundaryClock::new` opens identity `0` and installs no period. `episode_start` advances it by one with checked addition. `u64::MAX` is `EpochOverflow` and changes nothing. `reset_state` does not write it. Tick `0` (`EPOCH_TICK0`) is the origin of every episode and is not the identity. The earlier form, unset or `Some(0)` only, is withdrawn.

## Receipt v2

The only accepted header is `ferromorphic-step 2`. The following lines are required, in order: `epoch <u64>`, `step <u64>`, `dt_bits <hex>`, `mode clocked` or `mode event`, then zero or more `spike <t> <source>` lines. `dt_bits` is emitted as 16 lowercase hex digits. Version 1 is refused. That version had no epoch line. `crates/ferromotion-core/tests/ferromorphic_receipt_v2.rs` compiles ferromorphic at this pin and round-trips the bytes both ways.

```text
ferromorphic-step 2
epoch 0
step 4
dt_bits 3fb999999999999a
mode clocked
spike 4 0
spike 4 1
```

## episode_start

`episode_start(origin_ns, period_ns)` does not take `dt` and does not write `dt_bits`. The bits already stored must equal `((period_ns as f64) * 1e-9).to_bits()`. It sets `tick` to `0`, stores the origin and the period, clears the neuron and ring flags, and keeps the spike ledger. It is the only call that advances the episode identity. There is no `tick0` argument. A refusal leaves every field unchanged.

`BoundaryClock::new` stores a finite positive `dt` and no period, so it has no pair to compare. Forming the pair is `episode_start` and `EpisodeEnvelope::try_new`. Both require the forward product. That is the construction check. It is not a ratio exemption. `bind_v2_timestamps` repeats it on every call.

`period_ns > 2^53` is `InexactPeriod` at that construction, at `episode_start`, and on every bind. `2^53` is accepted. Every integer through `2^53` has its own `f64`, so two accepted periods cannot share `(period_ns as f64) * 1e-9`. `2^53 + 1` and `u64::MAX` are refused. The same refusal applies to a clock `period_ns` or a `step_duration_ns` above `2^53` on every bind, including a period written onto the public snapshot after `episode_start`.

`reset_state` only clears neuron and ring flags. It leaves `tick`, the epoch, `period_ns`, `origin_ns`, `dt` bits, and the ledger unchanged.

`dt` is the canonical dynamics step, stored as `f64` bits. Two engines match only when those bits are equal and both values are finite and positive. `period_ns` is the integer scheduling period and must be nonzero. Envelope v1 has no ratio exemption. A 1000 ns period is accepted when `dt` already has bits `0x3eb0c6f7a0b5ed8e`. `dt = 0.1` with 1000 ns, `dt = 0.1` with 20 ms, and the one-bit-off 20 ms pattern `0x3f947ae147ae147c` are `DtPeriodMismatch`. Fields of the envelope are private, so a struct literal cannot skip that check.

## Bind procedure ferromorphic still has to implement

Ferromotion implements `bind_v2_timestamps`. Ferromorphic does not. Their `require_forward_product` runs at `Schedule::try_new` and `Sim::episode_start` only. A later write to their public `dt`, `t`, `epoch`, `origin_ns`, or `period_ns` is not rechecked. `FERROMORPHIC_BIND_VERIFIED` stays false until they implement the procedure below and that implementation is verified against it.

The clock argument is an untrusted per-call snapshot. The function stores no validity bit. A previous `Ok` is not reused. The call does not write the clock and does not read `tick` as authorization. `EpisodeEnvelope` version 1 carries `epoch`, `dt_bits`, `period_ns`, and `origin_ns` beside a codec v2 chunk. The envelope is not written into the v2 canonical bytes and it is not part of the v2 digest. Codec v2 stays. Codec v1 stays withdrawn.

On every call, in this order:

1. No envelope: `MissingEpoch`.
2. Envelope version other than 1, or envelope `period_ns == 0`: `OffGrid`.
3. `period_ns > 2^53` on the envelope is `InexactPeriod`. Otherwise `dt_bits == ((period_ns as f64) * 1e-9).to_bits()`. No ratio exemption. Failure is `BadTick`, `InexactPeriod`, or `DtPeriodMismatch`.
4. Clock `period_ns`, when set, or `step_duration_ns` above `2^53`: `InexactPeriod`.
5. Clock `period_ns` or `origin_ns` unset: `MissingEpoch`. Identity 0 with no period is not a bindable episode.
6. `clock.epoch != envelope.epoch`: `StaleEpoch`.
7. Clock origin or period differs from the envelope: `AmbiguousEpoch`.
8. `clock.dt_bits != envelope.dt_bits`: `DtMismatch`.
9. `step_duration_ns != period_ns`, or `t0_ns < origin_ns`, or `(t0_ns - origin_ns)` is not a multiple of `period_ns`: `OffGrid`.
10. For each `k` in `0..n_action_steps`, checked `t0_ns + k * step_duration_ns` equals `origin_ns + ((t0_ns - origin_ns) / period_ns + k) * period_ns`. Overflow is `Overflow`. A mismatch is `OffGrid`.
11. `n_action_steps == 0` still runs steps 1–9 and stamps no index. It is not a horizon success and it does not authorize a rollout.
12. `Ok` does not set `EPISODE_ENVELOPE_INTEROPERABLE`, does not run inference, and does not run a joint step.

A later write to `epoch`, `dt_bits`, `period_ns`, `origin_ns`, or `tick` is seen on the next call. Changing `tick` alone does not fail a bind that otherwise matches. A matching digest or a matching bind does not authorize inference or physical actuation.

## Proposed joint step, not run

At step `k` the peer schedule is: sense the plant at `k`; control with spikes `t < k`; hold torque for one shared `dt`; run `Sim::step` for tick `k`. Spikes stamped `k` become visible at `k + 1 + delay`. `propose_joint_step` returns that schedule with `executed: false`.

Tick phases recorded, and not simulated, are deliveries, neuron updates, enqueue at `t + 1 + delay`, then increment.

## Still open

`DynamicsLab::time` remains the sum of the `dt` values passed to `DynamicsLab::step`. That accumulator is not the shared tick. Replacing it is not part of this decision.

Codec v2 accepts `n_action_steps == 0` as a body when the rest of the payload is complete. `check_horizon` matches the pinned Ferric helper: only `n > chunk_size` is a horizon error. Asking for action index 0 of that body is `TimestampIndex`. `authorize_rollout` still returns `EntryProposed`. That is the authorization refusal. An empty-horizon execution gate is not in this revision. `accept_event_step` is a different helper: a zero `max_delay` accepts only an event equal to the clock. It does not encode a chunk and it does not run a joint step.

A scheduling period above `2^53` is `InexactPeriod`. It is not installed and it is not bound. Integer stamps of an accepted period remain `origin_ns + tick * period_ns` in `u64`. The forward product of an accepted period is still `(period_ns as f64) * 1e-9`, with no ratio exemption. That product is not interoperability.

## Action-chunk index

Codec v2 timestamps are `t0_ns + k * step_duration_ns` with checked multiplication. That nanosecond index is not `Sim::t`. After `episode_start`, the episode tick is 0. `map_chunk_index_to_tick` counts from that 0 when the caller says an episode has started, and returns `EpochUnresolved` when it has not. Overflow of `tick + 1 + delay` is `Overflow`. `EPISODE_ENVELOPE_INTEROPERABLE` stays false until ferromorphic's bind exists and is verified and both Ferric and ferromorphic accept this version. A receipt-text round trip is not that accept.

## What remains on the ferromorphic side

* Implement the bind procedure in the section above, including per-call revalidation of the public snapshot, no cached validity bit, and `InexactPeriod` for `period_ns > 2^53`. At their current pin, `2^53` and `2^53 + 1` both pass `Schedule::try_new`. This repository does not claim that work is done.
* They do not parse Ferric codec v2. The outer envelope has to keep the inner v2 bytes unchanged. This adapter does not write the envelope into those bytes.
* Their `from_text` returns `Unordered` when a spike tick is not the receipt step. This receiver returns `StampMismatch`. Valid receipt bytes already match. The error name is not required for that match.
* No joint loop has been run in either repository. Inference stays unexecuted. `model_executed` stays false.

## Not claimed

No end-to-end loop ran. No spike train was produced by a network. No joule sample was taken. Receipt v2 text matching this pin is not interoperability of the two clocks.
