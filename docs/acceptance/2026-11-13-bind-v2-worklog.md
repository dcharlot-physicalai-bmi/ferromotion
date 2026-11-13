# Work log: bind calls at a6722b49

Not a sign-off receipt. `JOINT_LOOP_EXECUTED`, `EPISODE_ENVELOPE_INTEROPERABLE`, and `FERROMORPHIC_BIND_VERIFIED` stay false. The rerun on bmi-2's post-upgrade SHA, with both crates on the same toolchain, is still waiting. When that toolchain moves, the update is a merge commit. Pushed commits are not rewritten.

* ferromotion sources under test: `f3de21dba8198aaf4939662476a764bae1eaccd7` plus this work log's commit
* ferromorphic git pin: `a6722b49ff7dc770b1cb062adac21aa5c7155401`
* ferromorphic tree, from the GitHub commit object with no token: `97e1de379a7126565bc165b343d0db397d52092f`
* Cargo.lock sha256 `8ce34abf3d25de358bf3e3fef701c4428b293fda00cd9f49bffd028d59d54cb5`
* `rustc -Vv`: rustc 1.98.0 (88d9e12ae 2026-08-18), commit-hash `88d9e12ae178fab0fb5cc050a94da85685d449ea`, host `x86_64-unknown-linux-gnu`, LLVM 22.1.8
* cargo 1.98.0 (797e8a9bc 2026-08-05)
* command `cargo test --locked -p ferromotion-core --test ferromorphic_bind_v2`
* log `docs/acceptance/2026-11-13-bind-v2-worklog.log`
* log sha256 `b6589b20cbdc2bc88f8ca2b6810b19bd1f377009df97e698c410f9b0922bc9b3`
* result: 6 passed, 0 failed

| test | result | ferromorphic function |
|---|---|---|
| `a_matching_snapshot_binds` | pass | `sim::bind_v2_timestamps` `Ok(())` |
| `a_zero_step_count_checks_and_leaves_the_snapshot_unchanged` | pass | `sim::bind_v2_timestamps` with `n_action_steps == 0` returns `Ok(())` and does not write the snapshot |
| `periods_above_2_pow_53_are_inexact_and_the_bound_is_accepted` | pass | `sim::bind_v2_timestamps` returns `TimebaseError::InexactPeriod` for a snapshot period or a `step_duration_ns` of `2^53 + 1` and of `u64::MAX`; `2^53` returns `Ok(())` |
| `epoch_origin_period_dt_bits_and_the_t0_grid_are_refused` | pass | `StaleEpoch { event: 1, clock: 9 }`, `AmbiguousEpoch` for origin, `AmbiguousEpoch` for period, `Mismatch { left, right }` for `dt` bits, `OffGrid` for `t0` |
| `a_snapshot_field_changed_after_a_good_bind_is_refused_on_the_next_call` | pass | next `bind_v2_timestamps` returns `Mismatch` after `dt_bits` is flipped |
| `a_forward_product_mismatch_is_refused_before_an_envelope_exists` | pass | `EpisodeEnvelope::try_new` returns `DtPeriodMismatch { dt_bits, period_ns, nominal_bits }` |

`bind_v2_timestamps` is not called for the forward-product case. `EpisodeEnvelope`'s fields are private (`src/sim.rs:952`). `try_new` (`src/sim.rs:969`) is the only constructor and it calls `require_forward_product` before it returns. No public value with a mismatched product can be passed into `bind_v2_timestamps`. The variant exists. The bind entry for that input does not.

## Stamp check

`bind_v2_timestamps` is a pure validator. Stamping is `StepReceipt::from_step`. The zero-step test is now `a_zero_step_count_checks_and_stamps_nothing`. It builds a two-neuron `Sim`, calls `episode_start` and `try_step`, stamps a receipt with `from_step`, then calls `bind_v2_timestamps` with `n_action_steps == 0`. The call returns `Ok(())`. The receipt's `to_text` bytes, the public `Sim` byte string, and the `ClockSnapshot` are the same before and after.

The byte string is little-endian `dt`, `t`, `epoch`, `origin_ns`, `period_ns`, `mode`, the six `Ledger` counters, and each `Lif` field (`tau_m`, `v_rest`, `v_th`, `v_reset`, `r_m`, `t_ref`, `v`, `refractory`). `ring`, `inject`, `touched`, and `as_of` are private and have no public byte export. The call is not given `&mut Sim`.

* command `cargo test --locked -p ferromotion-core --test ferromorphic_bind_v2`
* log `docs/acceptance/2026-11-13-bind-v2-stamp-worklog.log`
* log sha256 `973148b95c735892c8679383fe1dac47902db49ebfdb9cc8f197b4b76e53248d`
* Cargo.lock sha256 unchanged: `8ce34abf3d25de358bf3e3fef701c4428b293fda00cd9f49bffd028d59d54cb5`
* rustc 1.98.0 (88d9e12ae 2026-08-18), commit-hash `88d9e12ae178fab0fb5cc050a94da85685d449ea`
* result: 6 passed, 0 failed

| test | result | ferromorphic function |
|---|---|---|
| `a_matching_snapshot_binds` | pass | `sim::bind_v2_timestamps` `Ok(())` |
| `a_zero_step_count_checks_and_stamps_nothing` | pass | `Ok(())`; `StepReceipt::to_text` bytes and the public `Sim` bytes unchanged; snapshot unchanged |
| `periods_above_2_pow_53_are_inexact_and_the_bound_is_accepted` | pass | `InexactPeriod` for `2^53 + 1` and `u64::MAX`; `2^53` is `Ok(())` |
| `epoch_origin_period_dt_bits_and_the_t0_grid_are_refused` | pass | `StaleEpoch`, `AmbiguousEpoch`, `Mismatch`, `OffGrid` |
| `a_snapshot_field_changed_after_a_good_bind_is_refused_on_the_next_call` | pass | next call returns `Mismatch` |
| `a_forward_product_mismatch_is_refused_before_an_envelope_exists` | pass | `EpisodeEnvelope::try_new` returns `DtPeriodMismatch` |
