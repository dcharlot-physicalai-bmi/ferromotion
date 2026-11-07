# SmolVLA receiver, codec version 2

Codec version 1 is withdrawn. This note follows the Ferric contract at
`3ee11c015e3478c7c4aa0409499346f79ab1c8ce`:

* <https://github.com/dcharlot-physicalai-bmi/ferric/blob/3ee11c015e3478c7c4aa0409499346f79ab1c8ce/docs/acceptance/contracts/smolvla-ferromotion.md>
* fixture `docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json` in that commit

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
duration is refused. No fps is assumed.

Empty noise is `MissingNoiseProvenance`. Empty or non-hex weight and stats are
refused. A value outside its joint limits, a bad horizon (`n_action_steps` not
in `1..=50`), or a width outside `1..=32` is refused.

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
The episode epoch is an outer version-1 envelope (`dt_bits`, `period_ns`,
`origin_ns`) with tick 0 fixed, described in `ferromotion-clock.md`. It is a
proposal to Ferric and ferromorphic. It is not interoperable until both confirm
it, and it does not change the golden v2 digest. A missing or disagreeing
epoch, a `dt` bit mismatch, or a timestamp off the nanosecond grid is refused.
A 1000 ns period is valid. `EntryProposed` and physical actuation stay refused.
No inference receipt exists. `DynamicsLab::step` is unchanged.

Real weight, stats, and noise sha256 values for `lerobot/smolvla_base` are not
in the fixture. The hashes there are synthetic.
