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

## Unresolved, for peer agreement

These are not applied to either integrator.

1. **Overflow.** The checked helper refuses a sum that does not fit. The peer's `Sim::step` still uses ordinary `+` (panic in debug, wrap in release). Whether that method returns an error is negotiable. This tree does not change it.
2. **Epoch.** The peer receipt has no epoch. `DynamicsLab::set_state` zeros the lab's `f64` time, which is a different reset from `reset_state`. Until agreed, `BoundaryClock.epoch` stays `None` and no receipt claims an epoch.
3. **Accumulator.** `DynamicsLab::time` remains the sum of the `dt` values passed to `DynamicsLab::step`, including that function's eight internal substeps of the pendulum. The shared clock, if adopted, is the integer tick and one multiply. Whether the lab drops the accumulator is negotiable. It is not dropped here.

## Action-chunk index

Codec v2 timestamps are `t0_ns + k * step_duration_ns` with checked multiplication. That nanosecond index is not `Sim::t`. `map_chunk_index_to_tick` leaves the epoch unresolved. Passing an epoch in a test only checks overflow and `tick + 1 + delay`. It does not adopt the open epoch or overflow policy.

## Not claimed

No end-to-end loop ran. No spike train was produced by a network. No joule sample was taken.
