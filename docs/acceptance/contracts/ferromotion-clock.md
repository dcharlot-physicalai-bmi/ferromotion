# Ferromotion side of the ferromorphic step clock

Read-only source: [ferromorphic `docs/contracts/ferromotion-clock.md`](https://github.com/dcharlot-physicalai-bmi/ferromorphic/blob/549aacdc5f7f4bd947c26877f14b8edc42febd5a/docs/contracts/ferromotion-clock.md)
at `549aacdc5f7f4bd947c26877f14b8edc42febd5a`.

That pin is the clock document. It is not `FERROMORPHIC_REVISION` in
`joint_acceptance` (`8dabe3a45e589908f524b595da09bbadbc8aa103`). No peer repository
was modified.

The adapter is `crates/ferromotion-core/src/ferromorphic_clock.rs`. It does not
call `Sim::step` or `DynamicsLab::step`. `JOINT_LOOP_EXECUTED` is false.

## What is implemented

* `same_dt` / `bind_dt`: both values finite and positive, and `to_bits` equal. `0.1` and `0.1 + f64::EPSILON` do not match. `0.1_f64` bits are `3fb999999999999a`.
* `step_seconds(step, dt)` is one multiply `step as f64 * dt`.
* `accumulated_f64` is the repeated addition `DynamicsLab` uses. Ten additions of `0.1` are not the bits of `10 as f64 * 0.1`. The lab function is unchanged.
* Delivery is checked `step + 1 + delay`. Delay 0 is the next tick. Delay 5 from step 4 is step 10. Overflow is `TimebaseError::Overflow`, not a wrap.
* `accept_event_step` accepts `clock..=clock+max_delay`. Earlier is `Late`. Later is `BeyondHorizon`. A horizon that does not fit in `u64` is `Overflow`.
* Spike order is `(t, source)`, stable for equal keys.
* `reset_state` clears the neuron and ring flags and leaves tick, ledger, and epoch unchanged.
* Refractory length is `ceil(t_ref / dt)` ticks.
* The sample receipt text in the peer document parses. A decreasing `(t, source)` is `Unordered`. A spike tick other than the receipt step is `StampMismatch` (this receiver's check of their stated equality).
* `meter_reading()` is `Unmetered`. Mechanical `T + U` from `DynamicsLab::energy` is not that reading.

## Proposed joint step, not run

At step `k` the peer schedule is: sense the plant at `k`; control with spikes `t < k`; hold torque for one shared `dt`; run `Sim::step` for tick `k`. Spikes stamped `k` become visible at `k + 1 + delay`. `propose_joint_step` returns that schedule with `executed: false`.

Tick phases recorded, and not simulated, are deliveries, neuron updates, enqueue at `t + 1 + delay`, then increment.

## Foreman decision, not yet a peer interoperability claim

`episode_start(origin_ns, dt, period_ns)` is the only call that resets the epoch. It sets `tick` to 0. There is no `tick0` argument. `reset_state` only clears neuron and ring flags. It leaves `tick`, the epoch, `period_ns`, `origin_ns`, `dt` bits, and the ledger unchanged.

`dt` is the canonical dynamics step, stored as `f64` bits. Two engines match only when those bits are equal and both values are finite and positive. `period_ns` is the integer scheduling period and must be nonzero. Envelope v1 has no ratio exemption. `episode_start` and `EpisodeEnvelope::try_new` both require `dt.to_bits() == ((period_ns as f64) * 1e-9).to_bits()`. A 1000 ns period is accepted when `dt` is that forward product (`0x3eb0c6f7a0b5ed8e`). `dt = 0.1` with 1000 ns, `dt = 0.1` with 20 ms, and the one-bit-off 20 ms pattern `0x3f947ae147ae147c` are `DtPeriodMismatch`. Fields of the envelope are private, so a struct literal cannot skip that check. `bind_v2_timestamps` runs the same check again.

Tick arithmetic on this adapter is checked `u64`. `advance` and `delivery_step` return `Overflow` and do not wrap. Peer `Sim::step` is untouched, so a peer build that still wraps in release is not this adapter.

## Outer envelope, codec v2 unchanged

`EpisodeEnvelope` version 1 carries `dt_bits`, `period_ns`, and `origin_ns` beside a codec v2 chunk. The episode tick is 0 and is not a field of the chunk. The envelope is not written into the v2 canonical bytes and it is not part of the v2 digest. `EPISODE_ENVELOPE_INTEROPERABLE` stays false until both Ferric and ferromorphic accept the full scope in writing. A narrow accept of the `dt` mapping is not that accept.

`BoundaryClock`'s fields are public and are an untrusted snapshot. `episode_start` checks the forward mapping only when it is called. A later write to `tick`, `dt_bits`, `period_ns`, `origin_ns`, or `epoch_tick0` is not trusted. A private `EpisodeEnvelope` constructor does not make those public fields trusted. `bind_v2_timestamps` rechecks the envelope mapping and requires the clock's epoch, origin, period, and `dt` bits to match that envelope. It refuses a missing epoch, an origin or period that disagrees, `dt` bits that disagree, a `step_duration_ns` that is not `period_ns`, and a `t0_ns` that is not `origin_ns + n * period_ns`. Bind does not read `tick` as authorization. A matching digest or a matching bind does not authorize inference or physical actuation, and it is not evidence that a joint step ran.

## Still open

`DynamicsLab::time` remains the sum of the `dt` values passed to `DynamicsLab::step`. That accumulator is not the shared tick. Replacing it is not part of this decision.

Codec v2 still refuses `n_action_steps == 0` at encode and decode (`CodecError::Horizon`). That refusal is unchanged. An empty chunk is not executable. Whether the protocol later allows a separate non-executable empty fixture, or Ferric's `check_horizon` also rejects 0, is waiting on bmi-3. This tree does not adopt either choice yet. `accept_event_step` is a different helper: a zero `max_delay` accepts only an event equal to the clock. It is not permission to encode a zero-length action chunk.

Nanosecond stamps above `2^53` are the declared `u64` grid. The `f64` forward product `(period_ns as f64) * 1e-9` cannot represent every integer nanosecond past that magnitude. That rounding is the formula already required above. It is not a ratio exemption and it is not interoperability.

## Action-chunk index

Codec v2 timestamps are `t0_ns + k * step_duration_ns` with checked multiplication. That nanosecond index is not `Sim::t`. After `episode_start`, the episode tick is 0. `map_chunk_index_to_tick` counts from that 0 when the caller says an episode has started, and returns `EpochUnresolved` when it has not. Overflow of `tick + 1 + delay` is `Overflow`. `EPISODE_ENVELOPE_INTEROPERABLE` stays false until both Ferric and ferromorphic accept this version.

## Not claimed

No end-to-end loop ran. No spike train was produced by a network. No joule sample was taken.
