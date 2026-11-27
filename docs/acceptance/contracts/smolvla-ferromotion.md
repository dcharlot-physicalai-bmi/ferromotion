# SmolVLA receiver, codec version 2

Codec version 1 is withdrawn. The codec in this tree is still version 2. Its
contract pin, superseded as the contract-document pin by the version-3 text
below, is `3ee11c015e3478c7c4aa0409499346f79ab1c8ce`:

* <https://github.com/dcharlot-physicalai-bmi/ferric/blob/3ee11c015e3478c7c4aa0409499346f79ab1c8ce/docs/acceptance/contracts/smolvla-ferromotion.md>
* fixture `docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json` in that commit

The empty-horizon fixture is a later Ferric publication,
`c05228929efc3bcb4210b475dd959370fa69893a`:

* <https://github.com/dcharlot-physicalai-bmi/ferric/blob/c05228929efc3bcb4210b475dd959370fa69893a/docs/acceptance/contracts/fixtures/smolvla-empty-chunk-v2.json>

It uses the same identity, `action_dim` 2, and both joints. Only `n_action_steps`
is 0 and `values_time_major` is empty. Its canonical digest is
`6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e`.

Ferric is Apache-2.0. The fixture in `fixtures/` is a verbatim copy. See
`fixtures/NOTICE`. The Rust encoder in `smolvla_receiver` was written here from
the byte layout. It does not include Ferric source.

The entry `ferric.smolvla.predict_action_chunk.v1` is still **proposed**.
`predict_action_chunk` is the call. `select_action` is refused.

## Codec

Little-endian. Domain `ferric.smolvla.action-chunk`. Version `u32` = 2.
Magic `FERRIC-SMOLVLA-CHUNK` and a NUL. Strings are a `u32` byte length then
UTF-8, with no extra NUL. Weight, stats, and noise are 64 lowercase hex ASCII
bytes. Then embodiment, `t0_ns`, `step_duration_ns`, `n_action_steps`,
`action_dim`, joint count (equal to `action_dim`), each joint's name, unit,
frame, `f32` min, `f32` max, dtype byte `1`, order byte `1`, and time-major
`f32` values. SHA-256 covers those bytes and is not appended.

Golden digest of the fixture:
`6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86`.

`timestamp[k] = t0_ns + k * step_duration_ns` uses checked arithmetic. A zero
duration is refused. No fps is assumed. Index `k` is refused with
`TimestampIndex` when `k >= n_action_steps`, so an empty chunk has no action
index 0 and no schedule for that index.

`check_horizon(n, chunk_size)` refuses `n == 0` and `n > chunk_size`.
`check_horizon(0, 50)` and `check_horizon(0, 0)` are `Horizon`.
`ActionChunk::try_new(0, 1, [])` is `Horizon`. `action_dim == 0` is still
`ActionDim`.

On a `ferromorphic-chunk-binding 1` record the same two refusals have the
binding names. `n == 0` is `EpisodeError::NoActionSteps`. `n > 50` is
`EpisodeError::TooManyActionSteps`. `CodecError::Horizon` and
`NoActionSteps` are one refusal: a chunk with no steps is not a binding.
`verify_chunk_binding` calls `check_horizon` and maps `Horizon` onto those
two names. Identifiers stay internal. End-of-chunk hold is a receiver state
(the current chunk has no steps left, or no chunk has arrived and the receipt
source is `home`). It is not an `n = 0` binding.

Empty noise is `MissingNoiseProvenance`. Empty or non-hex weight and stats are
refused. A value outside its joint limits, `n_action_steps` above 50, or a
width outside `1..=32` is refused.

## What a digest match is

The fixture's `kind` is `protocol_test`. `not_model_evidence`,
`not_rollout_evidence`, and `not_energy_evidence` are true. `authorize_rollout`
still fails after a matching digest, because the entry is proposed.
`model_executed` stays false and `measured_receipts` stays 0. That is protocol
conformance, not inference fidelity.

Forward normalization is `(x - mean) / (std + 1e-8)`. The inverse is
`y * std + mean`. State wider than 32 is refused; a shorter state is zero-padded
on the right. Images, if a length is presented, are bilinear into 512² with
`align_corners` false and pad on the left and the top. Pixels outside `[0, 1]`
are refused before the `pixel * 2 - 1` map. Aloha joint deltas are refused.

## Clock envelope

Version 2 timestamps stay `t0_ns` and `step_duration_ns` inside the v2 body.
The episode clock is envelope v2, described in `ferromotion-clock.md`: any
reduced rational `dt` and period, the period a positive multiple of `dt`, and
an `episode_nonce`. That parser does not require 30 Hz. The 30 Hz restriction
is the ferric SO-100 profile, lines 246–250 of the version-3 contract. Binding
1 maps `n = 0` from `Horizon` to `NoActionSteps`. The published empty-body file
is still the `c052289` fixture. The receiver refuses those bytes: `parse_chunk`,
`encode`, `verify_chunk`, and `authorize_rollout` return `Horizon`. That refusal
is not an end-of-chunk hold. Asking for action index 0 of a zero-step count
returns `TimestampIndex { index: 0, steps: 0 }`. A zero `step_duration_ns` is
`StepDuration` (Ferric's fixture spells that case `StepDurationAbsent`).
These tests are not inference and not a joint step.
No inference receipt exists. `DynamicsLab::step` is unchanged.

## Contract v3 pin

The signed-off contract text is
`docs/acceptance/contracts/fixtures/ferric-9b0a2ffa/smolvla-ferromotion-v3.md`,
sha256 `bc5f67f054b1a84226d0438f959750697fea7a9c9eeca417b134b39a3e5c19fd`,
21831 bytes, git blob `b178ba4dfa051412653a9694211d383e2fa049c6`. It is ferric
`cursor/toolchain-1.100` at `9b0a2ffab2dcf902e0bec726cf30c437c69a6e84` and
`cursor/fleet-energy-harness-65da` at `13ffe43a30ade6f514697286a2af00f0bbbc6c2a`.
The version-2 pins above are superseded as the contract-document pin and are
kept. The version-3 codec is not in this tree. This pin claims no codec verdict.

Real weight, stats, and noise sha256 values for `lerobot/smolvla_base` are not
in the fixture. The hashes there are synthetic.
