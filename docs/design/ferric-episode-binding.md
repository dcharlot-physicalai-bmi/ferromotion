# Proposal: binding a Ferric v2 chunk to an episode

Proposal only. No envelope field and no bind call changes in this commit. `EPISODE_ENVELOPE_INTEROPERABLE` stays false until bmi-3 signs (a)–(d). `predict_action_chunk` still returns `ContractError::ProposedEntryInvoked` at `crates/ferric-accept/src/smolvla.rs:486-487` on `cursor/toolchain-1.100`. A live-output episode needs that unblocked. Until then the chunks are fixtures.

## (a) Chunk time and the plant period

Ferric fixtures use `t0_ns = 1_000_000_000` and `step_duration_ns = 20_000_000` (`smolvla_receiver.rs` `protocol_fixture`, and the same pair in `docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json`). The Panda plant period in `panda_closed_loop.rs` is `2_000_000` ns, which is the MJCF timestep `0.002` s. `bind_v2_timestamps` returns `OffGrid` when `step_duration_ns != envelope.period_ns`. A fixture chunk bound to an envelope whose period is the plant period is `OffGrid`.

Two contracts:

1. Control period follows the chunk, and the plant substeps. The envelope period is `20_000_000` ns. One bind covers one chunk index. The plant holds that command for `20_000_000 / 2_000_000 = 10` steps of `2_000_000` ns. `t0` of chunk index `k` is `1_000_000_000 + k * 20_000_000`, which is on that grid.
2. Explicit resampling. The chunk's 20 ms samples are rewritten onto the 2 ms plant grid, and each plant tick binds with `step_duration_ns = 2_000_000`. The chunk then has no action index for 9 of every 10 plant ticks. The rewrite is a hold or an interpolation, and either one is a value the fixture did not publish.

Recommendation: (1). The published `step_duration_ns` stays the bind period, the fixture's `t0` and duration bind, and the plant keeps the MJCF timestep by holding the command for 10 steps. A hold inside those 10 steps is the plant integrator, not a second action stream.

## (b) Cross-binding by hash

Version 1, on both crates, is `version`, `epoch`, `dt_bits`, `period_ns`, `origin_ns`. The text header is `ferromorphic-episode 1`. The codec v2 body is not rewritten.

Proposed version 2, same fields on both sides, needs a ferromorphic change and bmi-2's agreement before either side parses it:

* `chunk_sha256: [u8; 32]`, the SHA-256 of the Ferric v2 body
* text header `ferromorphic-episode 2`
* one added line, after `origin_ns`: `chunk_sha256` and 64 lowercase hex digits

The binding, kept beside the chunk and not inside its bytes, carries `epoch` and the SHA-256 of the envelope text. That pair can sit on the step receipt. Version 1 `from_text` stays. A version 2 header is refused by today's parser.

## (c) Joint names, units, and the command

Ferric `JointMeta` (`crates/ferric-accept/src/smolvla.rs` on `cursor/toolchain-1.100`) stores `name`, `unit`, and `frame` and does not convert radians, degrees, or meters. The embodiment string does not select a joint map. The ferromotion protocol fixture names `j0` and `j1` with units `dataset_label_a` and `dataset_label_b` (`smolvla_receiver.rs` `protocol_fixture`). Those are not Panda joint names.

The executed loop names the plant joints from `franka_emika_panda/panda_nohand.xml`: `joint1` through `joint7`, file order. State is `qpos` in radians then `qvel` in rad/s. The command is torque in N·m on `qfrc_applied`, limited to ±87 for joints 1–4 and ±12 for joints 5–7 (`panda_closed_loop.rs`, and the actuator `forcerange` in that XML). It is not a position target. The XML `<general>` actuators are position servos; the loop zeros their gain and bias so the applied vector is the torque.

A future Panda chunk has to say those seven names in that order, and a unit of `N.m` when the value is torque. A value in `rad` is a position target and is not written to `qfrc_applied`. No such chunk exists in the fixtures.

## (d) Stale Ferric pins, not moved

`smolvla_receiver.rs` line 15, `FERRIC_CONTRACT_REVISION`, is `3ee11c015e3478c7c4aa0409499346f79ab1c8ce`. Line 49, `FERRIC_EMPTY_FIXTURE_REVISION`, is `c05228929efc3bcb4210b475dd959370fa69893a`. This proposal does not re-pin them.

`cursor/toolchain-1.100` head `2706d1b9d6c4a15194429463057de2100841bb4a` is CI-red on purpose, waiting on the `det_tanh` re-receipt. bmi-3 will send the green SHA, plus the contract (`8a92d996…`) and the fixture hashes (chunk `7dc00166…`, empty `d84e270d…`). Re-pinning waits on that SHA.
