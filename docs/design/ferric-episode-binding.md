# Binding a Ferric v2 chunk to an episode

agreed by bmi-2 and bmi-4 2026-11-20; bmi-3 pending

Proposal only. The envelope and `bind_v2_timestamps` are unchanged in this commit. `EPISODE_ENVELOPE_INTEROPERABLE` stays false until bmi-3 signs (a)–(d). `predict_action_chunk` still returns `ContractError::ProposedEntryInvoked` at `crates/ferric-accept/src/smolvla.rs:486-487` on `cursor/toolchain-1.100`. A live-output episode needs that unblocked. Until then the chunks are fixtures.

## (a) Period follows the chunk

The envelope period is the chunk's `step_duration_ns`. The plant holds that command for its substeps.

Integer nanoseconds are the authority. `substeps = period_ns / plant_ns`. `period_ns` must divide by `plant_ns` with no remainder. Construction refuses a remainder the way `InexactPeriod` refuses a period above `2^53`. The count is never computed from `f64`.

The MJCF timestep's bits must equal the bits of the `plant_ns` conversion, `(plant_ns as f64) * 1e-9`. For `panda_nohand` that plant period is `2_000_000` ns and the fixture control period is `20_000_000` ns, so `substeps` is 10.

`period_ns` must be an exact multiple of the ferromorphic `dt` named by `dt_bits`. Neural steps per action are that quotient. They are derived, never a free parameter.

Resampling onto the plant grid is rejected. The fixture never published those intermediate values.

`bind_v2_timestamps` still checks `t0_ns` against `origin_ns + chunk_index * period_ns`, with `step_duration_ns == period_ns`. A fixture with `t0_ns = 1_000_000_000` and `step_duration_ns = 20_000_000` binds when the envelope period is `20_000_000` ns.

## (b) Chunk binding, not an envelope field

The envelope stays version 1. It does not gain `chunk_sha256`. Its fields stay `version`, `epoch`, `dt_bits`, `period_ns`, and `origin_ns`, and its header stays `ferromorphic-episode 1`.

A separate ferromorphic record, header `ferromorphic-chunk-binding 1`, has these fields in order:

1. `envelope_sha256`, SHA-256 of the exact version-1 envelope text, 64 lowercase hex digits
2. `epoch`
3. `chunk_index`
4. `t0_ns`
5. `step_duration_ns`, equal to the envelope `period_ns`
6. `n_action_steps`
7. `chunk_sha256`, SHA-256 of the Ferric v2 body bytes exactly as delivered, 64 lowercase hex digits

The chain runs one way, from the chunk binding to the envelope. The envelope is the clock for the whole episode. A chunk hash inside it would force a new envelope for every chunk, and the envelope hash inside the binding would then cycle. `bind_v2_timestamps` keeps checking `t0_ns` against `origin_ns + chunk_index * period_ns` on that one clock.

The SHA-256 of the binding text is stamped on every `StepReceipt` that consumes the chunk.

Versioning for every text format: the version lives only in the header. There are no optional fields, so any field change is a new version. A strict parser refuses an unknown version, an unknown field, a missing field, uppercase hex, a missing final newline, and trailing bytes. Old parsers are kept forever. Nothing silently upgrades.

bmi-2 implements `ChunkBinding` in ferromorphic. ferromotion and ferric then pin to that revision.

## (c) Panda command

The joints are `joint1` through `joint7` in file order from `franka_emika_panda/panda_nohand.xml`. The command is torque on `qfrc_applied`. The unit token is exactly `N.m`. `rad` is refused on this path. A name or order mismatch is an error, never a remap.

Torque outside the actuator `forcerange` saturates the way that range saturates: ±87 N·m for joints 1–4 and ±12 N·m for joints 5–7. Every saturated value is counted per tick and recorded in the receipt.

Ferric `JointMeta` stores `name`, `unit`, and `frame` and does not convert them (`crates/ferric-accept/src/smolvla.rs` on `cursor/toolchain-1.100`). The protocol fixture's `j0` and `j1` with `dataset_label_a` and `dataset_label_b` are not this path.

## (d) Pins stay until the green Ferric SHA

`smolvla_receiver.rs` line 15, `FERRIC_CONTRACT_REVISION`, is `3ee11c015e3478c7c4aa0409499346f79ab1c8ce`. Line 49, `FERRIC_EMPTY_FIXTURE_REVISION`, is `c05228929efc3bcb4210b475dd959370fa69893a`. Neither is re-pinned here.

Re-pinning waits until bmi-3 sends the green Ferric SHA and the hashes. `cursor/toolchain-1.100` head `2706d1b9d6c4a15194429463057de2100841bb4a` is CI-red on purpose, waiting on the `det_tanh` re-receipt. The contract (`8a92d996…`) and the fixture hashes (chunk `7dc00166…`, empty `d84e270d…`) arrive with that green SHA.
