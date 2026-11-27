# Binding a Ferric chunk to an episode

revision 4. Revision 3 was agreed by bmi-2, bmi-3 and bmi-4 on 2026-11-20. Revision 4 is bmi-2's hold wording from 2026-11-27: hold is a receiver state, not an `n = 0` binding.

Proposal only. The envelope parser and `bind_v2_timestamps` are unchanged in this commit. `EPISODE_ENVELOPE_INTEROPERABLE` stays false. `predict_action_chunk` still returns `ContractError::ProposedEntryInvoked` at `crates/ferric-accept/src/smolvla.rs:486-487` on `cursor/toolchain-1.100`, so a live-output episode uses fixtures until that refusal is gone.

## (1) Plant

The plant is `trs_so_arm100/so_arm100.xml` at Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`. `robotstudio_so101/so101.xml` is in that tree and is not this plant. `ctrl` is the position target in radians, sent through the file's own `<position>` actuators. No gain is invented.

Joint order is the file order. Ranges are radians. Every actuator is `<position>` with `inheritrange="1"`, and the class `so_arm100` sets `kp="50"`, `dampratio="1"`, and `forcerange="-3.5 3.5"`.

| order | name | range (rad) |
|---|---|---|
| 1 | Rotation | −1.92 … 1.92 |
| 2 | Pitch | −3.32 … 0.174 |
| 3 | Elbow | −0.174 … 3.14 |
| 4 | Wrist_Pitch | −1.66 … 1.66 |
| 5 | Wrist_Roll | −2.79 … 2.79 |
| 6 | Jaw | −0.174 … 1.75 |

`lerobot/smolvla_base` at commit `5e8d12a6e297` has `output_features.action.shape` `[6]` and `normalization_mapping.ACTION` `MEAN_STD`. The stats file `policy_preprocessor_step_5_normalizer_processor.safetensors` hashes to `490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8`. The named key is `so100.buffer.action`. Its f32 mean is `[1.5959141254425049, 119.9442367553711, 109.769775390625, 56.70551300048828, -27.422693252563477, 12.002785682678223]` and its f32 std is `[26.39177894592285, 52.41140365600586, 49.85380554199219, 36.9983024597168, 59.35995101928711, 19.039955139160156]`. The preprocessor's `eps` is `1e-8`. De-normalization is `physical = normalized * (std + eps) + mean`.

`lerobot/svla_so100_pickplace` `meta/info.json` records `fps` 30 and `robot_type` `so100`. LeRobot `src/lerobot/robots/so_follower/so_follower.py` at `326484d77d2ac90379e84222dfad21fe4acf1f71` (file sha256 `56cd915ab5d180dc858bbbcf10eb13e4669252d4109b868036a352a41e4aa2ea`) sets the five arm motors to `MotorNormMode.DEGREES`, the gripper to `MotorNormMode.RANGE_0_100`, and writes `Goal_Position`. That file is the unit source. Arm degrees become radians by `deg * π / 180`. The gripper's 0–100 maps linearly onto the Jaw range: `jaw = -0.174 + (p / 100) * (1.75 - -0.174)`.

The chunk's unit token is `deg` on Rotation, Pitch, Elbow, Wrist_Pitch, and Wrist_Roll, and `pct` on Jaw. `pct` means the closed interval 0–100. A name, order, or unit mismatch is an error. Nothing is remapped. A Panda retarget is out of scope unless it has its own receipted contract.

A target outside the actuator `ctrlrange` (the inherited joint range) saturates to that range. Each saturated component is counted on that tick and recorded on the receipt.

## (2) Period

The rate is 30 Hz. The period is the reduced rational `100000000/3` nanoseconds. `1e9/30` is not an integer. The grid is checked in integer arithmetic.

Rationals are reduced form only, with checked `u128` arithmetic. Overflow or a non-reduced input is refused. Equality of two rationals is exact on that reduced pair. There is no rounding step in a refusal.

The plant step and the neural step are both `5000000/3` ns, which is `1/600` s. `period / plant = 20` exactly, so one action is 20 plant substeps and 20 neural steps. Those counts are the quotients. They are not free parameters.

The round-to-nearest-even `f64` of `1/600` has bits `3f5b4e81b4e81b4f`. The harness sets the MJCF timestep to that value and records the resulting model hash and the timestep bits. The Menagerie file is not edited. The receiver refuses unless those timestep bits are `3f5b4e81b4e81b4f`. A 2 ms plant step is not this path. `dt_bits` on the envelope must equal the round-to-nearest-even `f64` of `dt_num_ns/dt_den`, or the envelope is refused.

## (3) Envelope v2

Header `ferromorphic-episode 2`. Fields, in order: `version`, `epoch`, `dt_num_ns`, `dt_den`, `dt_bits`, `period_num_ns`, `period_den`, `origin_ns`, `episode_nonce`. `episode_nonce` is 32 bytes, written as 64 lowercase hex digits.

For 30 Hz, `dt_num_ns/dt_den` is `5000000/3` and `period_num_ns/period_den` is `100000000/3`. `dt_bits` is the `f64` bits of that neural step.

`episode_nonce` is an explicit constructor input. The harness generates it and records it on the receipt. The library does not draw it from the operating system. The same nonce makes the envelope text byte-identical on a rerun. A fresh nonce changes `envelope_sha256`. Envelope `170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a` is version 1 and has no nonce, so it is identical on every run. The nonce has to live on the envelope: it is part of the episode identity, and putting it only on the binding would leave that envelope hash reusable across episodes.

Version 1 stays parseable forever, under the versioning rules below.

## (4) Chunk binding

Header `ferromorphic-chunk-binding 1`. Fields, in order:

1. `envelope_sha256`, SHA-256 of the exact envelope text, 64 lowercase hex digits
2. `epoch`
3. `chunk_index`
4. `t0_num_ns`
5. `t0_den`
6. `period_num_ns`
7. `period_den`
8. `n_action_steps`
9. `chunk_sha256`, 64 lowercase hex digits

`chunk_sha256` is Ferric's `digest_sha256` over the full canonical chunk bytes, header included. The version-2 nonempty golden, on the `20000000/1` grid only, is `6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86`. The old empty-body digest `6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e` is not a binding. The version-3 goldens at `100000000/3` come later with Ferric chunk v3.

The chain runs one way, from the binding to the envelope. The envelope is the clock for the whole episode. A chunk hash inside it would force a new envelope per chunk, and the binding's `envelope_sha256` would then cycle. `bind_v2_timestamps` still checks `t0` against `origin_ns + chunk_index * period` on that clock, in the integer arithmetic of the reduced rationals.

The receiver decodes the chunk and hands `(t0, period, n)` to the ferromorphic verifier. The verifier refuses a mismatch, a `chunk_index` that is not strictly increasing, and an earlier `t0` (`Late`). The SHA-256 of the binding text is stamped on every `StepReceipt` that consumes the chunk.

The receiver also refuses unless all of these hold: the binding period equals both the envelope period and the chunk period, `t0` is on the grid, and the binding's `n` equals the chunk's `n`, with `1 ≤ n ≤ 50`. It refuses `n_action_steps == 0`. Envelope v2 itself accepts any reduced period and dt. The 30 Hz clock, period `100000000/3` and dt `5000000/3` (`dt_bits` `3f5b4e81b4e81b4f`), is the ferric SO-100 profile.

Ferric's signed-off chunk-v3 contract is `docs/acceptance/contracts/smolvla-ferromotion-v3.md`, sha256 `bc5f67f054b1a84226d0438f959750697fea7a9c9eeca417b134b39a3e5c19fd`, at `9b0a2ffab2dcf902e0bec726cf30c437c69a6e84` and `13ffe43a30ade6f514697286a2af00f0bbbc6c2a`. It carries `t0` and the period as a reduced `(num_ns, den)`, because version 2's integer `t0_ns` cannot express `100000000/3`. The version-3 codec is not in this tree. No codec verdict is claimed. Version 2 stays the codec this receiver parses.

## (5) Receiver policy

`hold` is a receiver state. It is not a binding, and it is not `n_action_steps == 0`. Binding 1 (`ferromorphic-chunk-binding 1`) and Ferric chunk v3 (contract `bc5f67f054b1a84226d0438f959750697fea7a9c9eeca417b134b39a3e5c19fd`, line 60) both refuse `n_action_steps == 0`, so a chunk with that `n` cannot be bound.

The state is one of these:

* No chunk has arrived yet. The target is the `home` keyframe `qpos` (`0 -1.57 1.57 1.57 -1.57 0` in `so_arm100.xml`). The receipt source is `home`.
* The current chunk has no steps left. The target stays the last target from that chunk. The receipt source is `hold`, and each such tick increments a held count.

A `hold` longer than one chunk duration, the bound chunk's `n_action_steps` times the period, without a new binding is `Stale` and ends the episode. That `n` is the `n` of a chunk that was bound, so it is at least 1.

A new chunk supersedes the rest of the current chunk from its `t0`.

Executed-step counts and superseded-step counts are stamped on the `StepReceipt`s.

## Versioning

The version lives only in the header. There are no optional fields, so any field change is a new version. A strict parser refuses an unknown version, an unknown field, a missing field, uppercase hex, a missing final newline, and trailing bytes. Old parsers are kept forever. Nothing silently upgrades.

bmi-2 implements `ChunkBinding` and envelope v2 in ferromorphic. ferromotion and ferric then pin to that revision.

## (d) Pins

`FERRIC_CONTRACT_REVISION` stays `3ee11c015e3478c7c4aa0409499346f79ab1c8ce`. `FERRIC_EMPTY_FIXTURE_REVISION` stays `c05228929efc3bcb4210b475dd959370fa69893a`. Those are the codec version 2 pins. They are superseded as the contract-document pin and are not deleted.

`FERRIC_V3_CONTRACT_REVISION` is `9b0a2ffab2dcf902e0bec726cf30c437c69a6e84`. `FERRIC_V3_CONTRACT_FLEET_REVISION` is `13ffe43a30ade6f514697286a2af00f0bbbc6c2a`. Both name the contract file whose sha256 is `bc5f67f054b1a84226d0438f959750697fea7a9c9eeca417b134b39a3e5c19fd` and whose git blob is `b178ba4dfa051412653a9694211d383e2fa049c6`. That pin is the contract text only.
