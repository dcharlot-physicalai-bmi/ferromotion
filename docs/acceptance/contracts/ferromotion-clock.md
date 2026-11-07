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

`episode_start(tick0, origin_ns, dt_ns)` is the only call that writes the shared epoch. It sets `tick` to `tick0`. `reset_state` still only clears neuron and ring flags. It leaves `tick`, `epoch_tick0`, `dt_ns`, `origin_ns`, and the ledger unchanged. Those two calls are not aliases.

Tick arithmetic on this adapter is checked `u64`. `advance` and `delivery_step` return `Overflow` and do not wrap. Peer `Sim::step` is untouched, so a peer build that still wraps in release is not this adapter.

The shared duration is integer nanoseconds. `f64_seconds_per_tick` computes `dt_ns as f64 * 1e-9` and accepts it only when `seconds * 1e9` equals `dt_ns as f64` with no rounding. Otherwise the bind is `InexactDt`.

## Outer envelope, codec v2 unchanged

`EpisodeEnvelope` version 1 carries `epoch_tick0`, `origin_ns`, and `dt_ns` beside a codec v2 chunk. It is not written into the v2 canonical bytes and it is not part of the v2 digest. `EPISODE_ENVELOPE_INTEROPERABLE` is false until Ferric and ferromorphic accept this envelope. `bind_v2_timestamps` refuses a missing epoch, an epoch that disagrees with the clock, a `step_duration_ns` that is not `dt_ns`, and a `t0_ns` that is not `origin_ns + n * dt_ns`.

## Still open

`DynamicsLab::time` remains the sum of the `dt` values passed to `DynamicsLab::step`. That accumulator is not the shared tick. Replacing it is not part of this decision.

## Action-chunk index

Codec v2 timestamps are `t0_ns + k * step_duration_ns` with checked multiplication. That nanosecond index is not `Sim::t`. `map_chunk_index_to_tick` leaves the epoch unresolved. Passing an epoch in a test only checks overflow and `tick + 1 + delay`. It does not adopt the open epoch or overflow policy.

## Not claimed

No end-to-end loop ran. No spike train was produced by a network. No joule sample was taken.
