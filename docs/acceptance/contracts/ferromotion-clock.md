# Ferromotion clock: envelope v2

The episode clock implemented in `crates/ferromotion-core/src/episode_v2.rs` is envelope v2. Durations are reduced integer rationals in nanoseconds. The grid test is checked `u128` arithmetic. `dt_bits` is one round-to-nearest-even of that rational, by binary long division, not an `f64` division.

Two documents are the sources for this text:

* ferromorphic `docs/contracts/ferromotion-clock.md` at `30812e26027cafa50e17c848e5afe57b1f246838`, the envelope v2 and binding 1 section.
* ferric `docs/acceptance/contracts/smolvla-ferromotion-v3.md`, sha256 `bc5f67f054b1a84226d0438f959750697fea7a9c9eeca417b134b39a3e5c19fd`, lines 246–250. That file is pinned at ferric `9b0a2ffab2dcf902e0bec726cf30c437c69a6e84` and `13ffe43a30ade6f514697286a2af00f0bbbc6c2a`. The 30 Hz restriction lives there, in the SO-100 profile, and not in this parser.

`EPISODE_ENVELOPE_INTEROPERABLE` stays false. `FERROMORPHIC_BIND_VERIFIED` stays false. `JOINT_LOOP_EXECUTED` stays false. Parsing an envelope or accepting a binding does not run inference and does not run a joint step.

## Envelope v2

`EpisodeEnvelopeV2::to_text` writes the header `ferromorphic-episode 2` and then nine `key value` lines, in this order: `version`, `epoch`, `dt_num_ns`, `dt_den`, `dt_bits`, `period_num_ns`, `period_den`, `origin_ns`, `episode_nonce`. Every line ends in `\n`. `dt_bits` is 16 lowercase hex digits. `episode_nonce` is 64 lowercase hex digits, 32 bytes.

```text
ferromorphic-episode 2
version 2
epoch 1
dt_num_ns 5000000
dt_den 3
dt_bits 3f5b4e81b4e81b4f
period_num_ns 100000000
period_den 3
origin_ns 0
episode_nonce 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
```

Rules:

1. `dt` and the period are nanoseconds as reduced rationals. `den > 0`, `num > 0`, and `gcd(num, den) = 1`. A non-reduced value is `NotReduced`. Any reduced pair is accepted. This parser does not restrict the clock to 30 Hz.
2. `dt_bits` is one round-to-nearest-even of `dt_num_ns / (dt_den * 10^9)` seconds, from `rne_f64_seconds`. A mismatch is `DtBitsMismatch`. `5000000/3` (1/600 s) is `3f5b4e81b4e81b4f`. `2000000/1` (2 ms) is `3f60624dd2f1a9fc`. `100000000/3` (1/30 s) is `3fa1111111111111`.
3. `period / dt` must be a positive integer. The check is `period_num_ns * dt_den` divisible by `period_den * dt_num_ns`, in checked `u128`. Otherwise `PeriodNotMultipleOfDt`. `steps_per_action` is that quotient: 30 Hz over 1/600 s is 20, and 20 ms over 2 ms is 10.
4. `episode_nonce` is a 32-byte constructor input. The library reads no OS randomness and no clock. The same inputs give the same text.
5. A product that does not fit in `u128` is `Overflow`. Nothing wraps.
6. The parser refuses an unknown header version, a `version` line other than 2, an unknown, missing, duplicated or reordered line, uppercase hex, a sign, a leading zero other than `0` itself, an empty value, a missing final newline, and trailing bytes. A decimal above `u64::MAX` is `Overflow` at parse time. `to_text(from_text(x)) == x` for every accepted `x`.

## Binding 1

`ChunkBinding::to_text` writes the header `ferromorphic-chunk-binding 1` and nine lines: `envelope_sha256`, `epoch`, `chunk_index`, `t0_num_ns`, `t0_den`, `period_num_ns`, `period_den`, `n_action_steps`, `chunk_sha256`. The text rules are the envelope's. `envelope_sha256` is the SHA-256 of the exact v2 envelope text. `t0` may be `0/1`. Every other rational in the record is positive.

`verify_chunk_binding` refuses, in this order:

1. `EnvelopeHashMismatch`
2. `EpochMismatch`
3. `PeriodNotEnvelope`
4. `PeriodNotChunk`
5. `T0NotChunk`
6. `StepsNotChunk`
7. `NoActionSteps` when `n = 0`, and `TooManyActionSteps` when `n > 50`
8. The grid check: `BeforeOrigin`, `OffGrid`, or `Overflow`
9. `ForeignPrevious` when the previous state names another envelope or epoch
10. `ChunkIndexNotIncreasing` when the index is not strictly greater
11. `RepeatedT0` when `t0` equals the previous `t0`
12. `Late` when `t0` is earlier than the previous `t0`

The grid identity is `t0_num_ns * period_den == (origin_ns * period_den + k * period_num_ns) * t0_den` for some integer `k >= 0`. The implementation uses the divisibility form: `d = t0_num_ns * period_den - origin_ns * period_den * t0_den` must be `>= 0`, and `period_num_ns * t0_den` must divide `d`. Every product is checked `u128`. `d < 0` is `BeforeOrigin`. A non-zero remainder is `OffGrid`. A product that does not fit is `Overflow`.

`n = 0` on a chunk body is `CodecError::Horizon`. On a binding, `verify_chunk_binding` calls `check_horizon` and maps that same refusal to `EpisodeError::NoActionSteps`. `n > 50` is `Horizon` on the chunk body and `TooManyActionSteps` on the binding. Identifiers are internal. Hold is a receiver state: no chunk yet, source `home`, or the current chunk has no steps left, source `hold`. It is not an `n = 0` binding. The proposed v3/SO-100 profile below retires the `home` source. That retirement is not implemented.

## Where 30 Hz is required

Envelope v2 accepts any reduced `dt` and period that satisfy the rules above. Ferric's SO-100 profile, lines 246–250 of the pinned v3 contract, is what refuses anything else. For that profile the only legal clock is 30 Hz with 20 substeps: period `100000000/3`, dt `5000000/3`. This crate does not apply that profile.

## Hold and staleness (proposed for ferric v3 text revision)

Proposed. Not implemented. `EPISODE_ENVELOPE_INTEROPERABLE`, `FERROMORPHIC_BIND_VERIFIED`, and `JOINT_LOOP_EXECUTED` stay false. This section waits for a revised ferric contract sha before any code change.

1. There are no wall-clock thresholds. Plant time is the envelope grid, in exact integer rationals, with products checked in `u128`. An accepted binding covers `[t0, t0 + n * period)`. Substep `s` at plant time `t` uses `action_index = floor((t - t0) / period)` of the newest accepted binding that covers `t`. The grid index `k` is bounded to `u64`. A product that does not fit in `u128` is `Overflow`.
2. `Late` means `t0` is strictly less than the last accepted binding's `t0` in the episode (refusal 12; an equal `t0` is `RepeatedT0`). It is ordering, not plant time. `Stale` means `t0 + n * period <=` plant time at acceptance, compared as exact rationals. A binding with `t0 < t < t0 + n * period` is accepted and starts at `floor((t - t0) / period)`. `Late` is checked before `Stale`.
3. Hold: when no accepted binding covers `t`, the receiver applies a zero-order hold of the last applied position target, with no extrapolation. The stamp repeats that binding's hash, `chunk_index`, and `action_index`. The substep counter keeps counting. The `hold` field is the text value in item 6.
4. A step before any accepted chunk is refused (`NoBinding`). For the v3/SO-100 profile the `home` source is retired. The sentence above, "Hold is a receiver state", still describes the code. The retirement is this proposal.
5. A hold never crosses `episode_nonce`.
6. The stamp is text. One `ferromorphic-step 3` line per substep, LF-terminated ASCII. The fields are `binding_sha256` (64 lowercase hex, the SHA-256 of the binding-1 text), `chunk_index` `u64`, `action_index` `u32`, `substep_index` `u32`, and `hold`. `substep_index` lies in `[0, S)`, where `S = period / dt` is an integer (20 for SO-100). `hold` is one of `none`, `exhausted`, `late`, `stale`. `none` means an accepted binding covers the substep. `late` or `stale` means none covers it and the most recent binding presented since the last accepted one was refused `Late` or `Stale`. `exhausted` is any other uncovered substep. `StepReceipt` pins the SHA-256 of the exact stamp lines.

## Full state v1

Ferromorphic `3213150fd863f67d7027d1e3bebc8ea052886a75` publishes the Lif golden that `30812e26` only described. The files are under `docs/acceptance/contracts/fixtures/ferromorphic-3213150f/`. The 970-byte record is sha256 `7d9bc9c1400bd2b14d4d5336bb713b91733353b9e0a17fa6b1852ecbb05617f1`. The embedded `state_bytes` are 280 bytes, sha256 `01e52c73cf8376050d5b95581a6a8e15389668ad2f067d6e517b7d94fc753f13`.

The layout in their clock contract is little-endian: magic `ferromorphic-sim-state\0`, version `1` as `u32`, the model tag (`u32` length and bytes), `state_bytes` (`u64` length and bytes), the net, the neuron records, and a SHA-256 of every byte before it. Field widths are the ones `Sim::full_state_bytes` and `Sim::state_bytes` document in that commit. `state_bytes` starts with `dt` bits, a mode byte, `t`, `epoch`, a presence byte and value for `origin_ns`, and the same for `period_ns`.

The parser checks that structure. On this golden, `t` is 50, `origin_ns` is 7000, and `period_ns` is 100000. It does not read neuron fields and it does not restore a simulation.

## Earlier adapter

`crates/ferromotion-core/src/ferromorphic_clock.rs` still parses receipt v2 and the version-1 envelope (`dt_bits`, integer `period_ns`, `origin_ns`) and still runs `bind_v2_timestamps`. That path is not envelope v2. Its flags stay false. It is not the grammar in `episode_v2.rs`.
