# Joint clock run, 2026-11-13

The suite sources are ferromotion `2d38a37009ff42d962126e47c633b355947c17c8`. Ferromorphic is the git dependency `a6722b49ff7dc770b1cb062adac21aa5c7155401`, with no path override. `FERROMORPHIC_CLOCK_REVISION` is that same commit.

The GitHub commit object was fetched with no token. Its sha is `a6722b49ff7dc770b1cb062adac21aa5c7155401` and its tree is `97e1de379a7126565bc165b343d0db397d52092f`. The parent is `eb64cb6e6fd2d6993601f1a31af86ab0ffb3e5d1`.

`JOINT_LOOP_EXECUTED` is false. `EPISODE_ENVELOPE_INTEROPERABLE` is false. `FERROMORPHIC_BIND_VERIFIED` is false. This run does not change those three.

* Cargo.lock sha256 `8ce34abf3d25de358bf3e3fef701c4428b293fda00cd9f49bffd028d59d54cb5`
* toolchain `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `cargo 1.98.0 (797e8a9bc 2026-08-05)`
* log `docs/acceptance/receipts/2026-11-13-joint-clock.log`
* log sha256 `298722b5be5d76fd4d47685462ec6698cf1342b0968d8b21e3a97619b435fce8`
* started `2026-11-13T14:06:03Z`

Commands, all `--locked`:

```text
cargo test --locked -p ferromotion-core --lib ferromorphic_clock::
cargo test --locked -p ferromotion-core --lib smolvla_receiver::
cargo test --locked -p ferromotion-core --test ferromorphic_receipt_v2
cargo test --workspace --locked --no-fail-fast
```

## Clock, 13 passed, 0 failed

* pass `ferromorphic_clock::tests::a_period_above_2_pow_53_is_inexact_at_construction_start_and_bind`
* pass `ferromorphic_clock::tests::a_public_clock_mutation_is_rejected_on_the_next_bind`
* pass `ferromorphic_clock::tests::delay_zero_is_the_next_tick_and_overflow_is_checked`
* pass `ferromorphic_clock::tests::dt_matches_only_on_bits_and_point_one_is_the_receipt_constant`
* pass `ferromorphic_clock::tests::episode_start_sets_tick_zero_advances_epoch_and_keeps_the_ledger`
* pass `ferromorphic_clock::tests::forward_product_mapping_rejects_mismatched_dt_and_accepts_1000ns`
* pass `ferromorphic_clock::tests::past_and_beyond_horizon_events_are_rejected`
* pass `ferromorphic_clock::tests::proposed_joint_step_is_not_executed_and_spikes_wait_one_tick_plus_delay`
* pass `ferromorphic_clock::tests::refractory_is_ceil_of_t_ref_over_dt`
* pass `ferromorphic_clock::tests::reset_keeps_tick_ledger_and_the_opening_epoch`
* pass `ferromorphic_clock::tests::shared_seconds_are_one_multiply_and_the_accumulator_can_differ`
* pass `ferromorphic_clock::tests::spike_order_is_tick_then_source_and_ties_keep_emission_order`
* pass `ferromorphic_clock::tests::the_pinned_receipt_v2_parses_and_version_1_is_refused`

## Receiver, 10 passed, 0 failed

* pass `smolvla_receiver::tests::chunk_index_does_not_become_a_ferromorphic_tick_without_an_epoch`
* pass `smolvla_receiver::tests::empty_horizon_negative_cases_stay_on_their_own_layers`
* pass `smolvla_receiver::tests::empty_noise_is_missing_noise_provenance_and_incomplete_hashes_refuse`
* pass `smolvla_receiver::tests::mutated_shape_limits_and_time_are_refused`
* pass `smolvla_receiver::tests::nonempty_golden_is_unchanged_and_matches_the_empty_identity`
* pass `smolvla_receiver::tests::normalization_pad_and_bounds_follow_the_contract_and_are_not_inference`
* pass `smolvla_receiver::tests::pinned_fixture_bytes_match_the_golden_digest_and_do_not_authorize_a_rollout`
* pass `smolvla_receiver::tests::published_empty_horizon_matches_canonical_bytes_and_refuses_index_zero`
* pass `smolvla_receiver::tests::sha256_of_abc_matches_the_known_vector`
* pass `smolvla_receiver::tests::timestamps_are_checked_and_a_digest_match_still_refuses_actuation_and_select_action`

## Cross-crate, 1 passed, 0 failed

* pass `ferromorphic_receipt_v2_round_trips_both_ways`

## Workspace

`cargo test --workspace --locked --no-fail-fast` exited 101. Across every target: 2104 passed, 3 failed, 12 ignored. The only failed target is `ferromotion-core` lib, 996 passed and 3 failed. The other targets passed. The three failures were not edited.

* fail `mjcf_tree::tests::geom_frames_match_mujocos_kinematics_bit_for_bit` at `crates/ferromotion-core/src/mjcf_tree.rs:9014`. `left == right` failed on geom 1. The first differing entry is index 3: ours `0.0666397393551407`, MuJoCo `0.0666397393551405`.
* fail `mjcf_tree::tests::spatial_tendons_step_as_mujoco_does_bit_for_bit` at `crates/ferromotion-core/src/mjcf_tree.rs:7900`. `left == right` failed. `left` `[4609734117662855034, 13842436752815519228, 4622613446824682926]`, `right` `[4609734117662855034, 13842436752815519230, 4622613446824682927]`.
* fail `mjcf_tree::tests::the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit` at `crates/ferromotion-core/src/mjcf_tree.rs:7614`. `left == right` failed. The first differing entry is index 9: ours `4580560284212516227`, MuJoCo `4580560284212516225`.

Per-target counts from that run:

| target | result |
|---|---|
| ferromotion lib | ok, 0 passed |
| ferromotion-bench lib | ok, 5 passed |
| ferromotion-circuit lib | ok, 14 passed |
| ferromotion-cloth lib | ok, 17 passed |
| ferromotion-control lib | ok, 517 passed |
| ferromotion-core lib | fail, 996 passed, 3 failed |
| ferromotion-core `ferromorphic_receipt_v2` | ok, 1 passed |
| ferromotion-core `nalgebra_nan_behaviour` | ok, 1 passed |
| ferromotion-core `no_panicking_float_comparisons` | ok, 1 passed |
| ferromotion-core `no_unguarded_dynamic_svd` | ok, 1 passed |
| ferromotion-core `nonfinite_public_api` | ok, 2 passed |
| ferromotion-core `stability_bounds_are_tested_from_outside` | ok, 1 passed |
| ferromotion-core `urdf_geometry_end_to_end` | ok, 2 passed |
| ferromotion-coupled lib | ok, 6 passed |
| ferromotion-dem lib | ok, 3 passed |
| ferromotion-fem lib | ok, 33 passed, 1 ignored |
| ferromotion-fluid lib | ok, 70 passed, 1 ignored |
| ferromotion-learn lib | ok, 87 passed, 1 ignored |
| ferromotion-models lib | ok, 155 passed, 8 ignored |
| ferromotion-mpm lib | ok, 8 passed |
| ferromotion-policy lib | ok, 26 passed |
| ferromotion-rod lib | ok, 9 passed |
| ferromotion-ruckig lib | ok, 8 passed |
| ferromotion-tactile lib | ok, 20 passed, 1 ignored |
| ferromotion-viz lib | ok, 1 passed |
| ferromotion-wasm lib | ok, 119 passed |
| ferromotion-wasm later targets | ok: 16 targets with 0 tests, then 1 passed, then 1 target with 0 tests |

Those later wasm lines are the doc-test and extra target invocations cargo printed for that crate. They are in the log.
