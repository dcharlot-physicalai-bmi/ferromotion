# SmolVLA chunk contract v3 for Ferromotion (bmi-4) and Ferromorphic (bmi-2)

Status: **proposed**. This document is the contract text. It does not change the codec. `crates/ferric-accept/src/smolvla.rs` stays on version 2 until this text is reviewed. No golden for version 3 is invented here.

## What stays

Version 2 stays parseable forever. Magic `FERRIC-SMOLVLA-CHUNK` and a NUL, version `u32` = 2, and the rest of the version 2 layout in `docs/acceptance/contracts/smolvla-ferromotion.md` are unchanged. The two protocol fixtures stay byte-identical, and their digests stay:

- `docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json` → `6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86`
- `docs/acceptance/contracts/fixtures/smolvla-empty-chunk-v2.json` → `6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e`

Those two digests are version 2 only. Both fixtures use `t0_ns` `1000000000` and `step_duration_ns` `20000000`, the rationals `1000000000/1` and `20000000/1`. That clock is not the SO-100 30 Hz period `100000000/3`. Version 3 does not accept either fixture. Version 3 goldens will be generated on the `100000000/3` grid when the codec is written. This text does not write that codec.

A reader branches on the version integer after the shared magic. Version 2 takes the version 2 layout. Version 3 takes the layout below. Any other version is refused. A version 2 golden is not a version 3 chunk, and a version 3 chunk is not a version 2 golden.

## Version 3 header

Little-endian, same string rule as version 2: a `u32` byte length, then UTF-8, no trailing NUL. Seven provenance hashes stay fixed 64-byte lowercase hex, in header order: weight, stats, action mean, action std, state mean, state std, noise.

| Order | Field |
|---|---|
| 21 | magic `FERRIC-SMOLVLA-CHUNK` and a NUL |
| u32 | version `3` |
| string | LeRobot commit |
| string | model id |
| 64 | weight sha256 |
| 64 | stats sha256 |
| string | stats key, one of `so100`, `so100-blue`, `so100-red` |
| 64 | action mean sha256 |
| 64 | action std sha256 |
| 64 | state mean sha256 |
| 64 | state std sha256 |
| 64 | noise sha256 |
| u64 | odd caller seed |
| string | embodiment name |
| u64 | `t0_num_ns` |
| u64 | `t0_den` |
| u64 | `period_num_ns` |
| u64 | `period_den` |
| u32 | `n_action_steps` |
| u32 | `action_dim` |
| u32 | joint count, equal to `action_dim` |
| repeat | joint name, unit, frame, then f32 min, f32 max |
| u8 | `1` for f32 |
| u8 | `1` for time-major, joint order |
| f32… | values, row `t` then joint `0..action_dim` |

`t0` and the period are reduced rationals of nanoseconds: numerator `u64`, denominator `u64`. The denominator is greater than 0. `gcd(num, den) = 1`, computed with the Euclidean algorithm on the two `u64`s. A zero denominator, or a fraction that is not already reduced, is refused. The encoder does not reduce on the caller's behalf.

Every product and sum used to place step `k` is checked `u128`. Overflow refuses the chunk. Step `k` (0-based, `k < n_action_steps`) is the rational

```
t(k) = t0_num_ns / t0_den + k * period_num_ns / period_den
     = (t0_num_ns * period_den + k * period_num_ns * t0_den) / (t0_den * period_den)
```

The digest stores the four reduced inputs, not this sum. A caller who asks for an integer nanosecond gets one only when the denominator divides the numerator exactly. Otherwise that conversion is refused. The rational itself is the time.

For 30 Hz the period is `100000000/3` nanoseconds. `10^9 / 30 = 1000000000/30 = 100000000/3`, and `gcd(100000000, 3) = 1`. An integer `step_duration_ns` cannot say this; that is why version 2's two `u64` clocks are not the version 3 clock.

`n_action_steps` is an integer in `1..=50`. `50` is the SmolVLA `chunk_size`, and it is the time extent in the noise byte string below. `0` is refused. A value above `50` is refused.

## Joints

`lerobot/smolvla_base` config at Hub revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` has `output_features.action.shape = [6]` and `normalization_mapping` `VISUAL: IDENTITY`, `STATE: MEAN_STD`, `ACTION: MEAN_STD`.

Both dataset cards record `fps` 30:

- `lerobot/svla_so100_pickplace` `meta/info.json`, `robot_type` `so100`
- `lerobot/svla_so101_pickplace` `meta/info.json`, `robot_type` `so100_follower`

The bus order below is `src/lerobot/robots/so_follower/so_follower.py` at `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a`, lines 50–59. `use_degrees` defaults true (`config_so_follower.py:42`), so that file's five arm motors are `MotorNormMode.DEGREES` and the gripper is `MotorNormMode.RANGE_0_100`. That is the contract's unit vocabulary. It is not the scale of the pinned dataset numbers. The dataset was not recorded by that revision: see Dataset meta.

The version 3 unit tokens are `deg` and `percent_range`. `deg` is the contract spelling of `DEGREES`. `percent_range` is the contract spelling of `RANGE_0_100`. Any other unit string is refused. `deg` on the gripper, or `percent_range` on an arm joint, is refused.

The normative SO-100 names are the `svla_so100_pickplace` action names, in that order, which is the same motor order as the bus:

| Index | Name | Motor id | Unit token | MotorNormMode |
|---|---|---|---|---|
| 0 | `main_shoulder_pan` | 1 `shoulder_pan` | `deg` | `DEGREES` |
| 1 | `main_shoulder_lift` | 2 `shoulder_lift` | `deg` | `DEGREES` |
| 2 | `main_elbow_flex` | 3 `elbow_flex` | `deg` | `DEGREES` |
| 3 | `main_wrist_flex` | 4 `wrist_flex` | `deg` | `DEGREES` |
| 4 | `main_wrist_roll` | 5 `wrist_roll` | `deg` | `DEGREES` |
| 5 | `main_gripper` | 6 `gripper` | `percent_range` | `RANGE_0_100` |

`svla_so101_pickplace` records a different action-name spelling of the same six motors: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`. That spelling is not this table. A mixed list, a reordering, or a `.pos` name in an SO-100 chunk is refused. `action_dim` for this embodiment is 6.

Embodiment token: `so100`. That is `robot_type` in the dataset info below, and it is the stats key `so100` (not `so100-blue` or `so100-red`).

Frame token, every joint: `so100_joint`. `meta/info.json` has no frame field. `so100_joint` is not an MJCF frame. The mapping from this token onto an MJCF joint (sign, offset, scale, clamp) lives in bmi-4's pinned receiver calibration map, not in this codec. Any other frame string on this embodiment is refused.

Limits are a closed interval, `min < max`, value inside, as in version 2. The five arm limits are the dataset `action` min and max, labeled **observed, sole sourced bound in contract units**. They are not a mechanical stop and they are not the full encoder domain. Widening them to that domain stays deferred: the formula revision in Dataset meta explains the scale of the stored numbers, and the hub object was rewritten on 2026-07-16. That rewrite is not a recording commit, and these limits are not replaced here. The gripper limit is the closed interval 0 to 100.

| Index | min | max |
|---|---|---|
| 0 `main_shoulder_pan` | -37.177734375 | 72.7734375 |
| 1 `main_shoulder_lift` | -89.47265625 | 41.1328125 |
| 2 `main_elbow_flex` | -49.04296875 | 74.619140625 |
| 3 `main_wrist_flex` | 9.66796875 | 96.591796875 |
| 4 `main_wrist_roll` | 56.25 | 123.57421875 |
| 5 `main_gripper` | 0 | 100 |

The dataset's observed gripper action max is 34.93077850341797. A command between that and 100 is still inside the contract interval and is accepted. A command outside 0–100 is refused.

`percent_range` is linear in calibrated ticks. At `a0eb860d`, `motors_bus.py:871-873` computes `((tick - min) / (max - min)) * 100` and `drive_mode` replaces that with `100 - norm`. At the recording revision below, `feetech.py:478` computes `(present - start) / (end - start) * 100`, and `feetech_calibration.py:192` passes `invert_drive_mode=True` for the gripper, which chooses which end is `start`. Either way the reversal is calibration, not a constant in this codec. The map from `percent_range` onto an MJCF Jaw is calibration-dependent and is not in this codec.

## Dataset meta

`lerobot/svla_so100_pickplace` `meta/info.json` is pinned at Hub revision `728583b5eaf9e739a7f119e2def466fa1d552402` (dataset `sha` on 2026-11-27). The file sha256 is `aec8d0c7bd3d2dc693c160bd58f9fb6aabcbb24d4b28e2f09950a3aa6b246eef`. It records `robot_type` `so100`, `fps` 30, and the six action names above. It does not state units.

`meta/stats.json` at the same revision has sha256 `5f53e159022a2ecf14bb01dd0623fcc498bffa719a776f1ef1858c6098169b89`. The arm min and max in the table above are `action.min` and `action.max` from that file. The dataset does not state its units.

`a0eb860d` does not reproduce them. `motors_bus.py:874-877` at that commit sets degrees to `(val - mid) * 360 / max_res` with `max_res = resolution - 1`. `src/lerobot/motors/feetech/tables.py:190` sets `sts3215` resolution to 4096, so the divisor is 4095. Every arm extremum in this `stats.json`, both `action` and `observation.state`, is an exact multiple of `360/4096` and not of `360/4095`. `56.25 = 640 * 360 / 4096`. The `observation.state` gripper minimum is `-0.20718231797218323`. `RANGE_0_100` at `motors_bus.py:871-873` maps a tick clamped into `[min, max]` onto `[0, 100]`, and `drive_mode` only swaps that to `100 - norm`. The unnormalize clamp at lines 900–902 is `min(100, max(0, val))`. A negative state is outside both. Those two checks are why this dataset was not recorded at `a0eb860d`.

The scale that matches the stored numbers is commit `974028bd28` (2025-03-13), `lerobot/common/robot_devices/motors/feetech.py:460`: `values[i] / (resolution // 2) * HALF_TURN_DEGREE`, with `HALF_TURN_DEGREE = 180` at line 49 and `sts3215` resolution 4096 at line 127. That is `steps * 360 / 4096`. The hub dataset was created 2025-05-12, while that file was still the current Feetech calibration (the next change of that path is the 2025-06-05 hardware redesign, `e23b41e79a`). `info.json` `codebase_version` is the string `v3.0`. That is a dataset-container label, not a git revision. The hub object was last modified 2026-07-16. This contract pins `974028bd28` as the formula the numbers match. It does not claim the 2026-07-16 rewrite recomputed them.

The gripper at `974028bd28` is `CalibrationMode.LINEAR` (`lerobot/common/robot_devices/robots/feetech_calibration.py:270-271`). The formula is `feetech.py:478`. Values are accepted down to `LOWER_BOUND_LINEAR = -10` (`feetech.py:46`) before that file raises. `-0.20718231797218323` is inside that margin and outside `a0eb860d`'s `[0, 100]` clamp.

## Stats key

`policy_postprocessor_step_0_unnormalizer_processor.safetensors` at Hub revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` is 640 bytes. Its sha256 is `490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8`. The same bytes are `policy_preprocessor_step_5_normalizer_processor.safetensors` at that revision. The version 3 stats hash is this file. The stats key names which tensors inside it were used.

Each tensor is six little-endian f32s. The sha256 is of those 24 bytes:

| Key | Tensor | sha256 |
|---|---|---|
| `so100` | `so100.buffer.action.mean` | `e22c3af117ece699b816d50f60df8a1e7e4728e1babf5297634b74b8483ad924` |
| `so100` | `so100.buffer.action.std` | `a09262820c5532e2fffb6da68c7ee6d9670d0da7080f70aadd0b34b7af8755b9` |
| `so100-blue` | `so100-blue.buffer.action.mean` | `8f178ffad9bca313223b2ebf56102779a017dbb0409c557259fa8888e02155fb` |
| `so100-blue` | `so100-blue.buffer.action.std` | `a4762d32295adf3bf7d652ea86cb3e7f2cc575b496e7e30a220a572e927a6820` |
| `so100-red` | `so100-red.buffer.action.mean` | `f85d5119f589c058249515c89ed780b4bba4fdaa30b81277bb738e513b7e6cc8` |
| `so100-red` | `so100-red.buffer.action.std` | `4deb2fcd2af0aa4667f6b09748b5e527e543ca36c13e204b75273e7cc62c8bf6` |

A chunk records one of those three keys. A missing key, a different key, or a tensor sha256 that does not match the row is refused.

Embodiment and dataset pick the key:

| Embodiment token | Dataset | Stats key |
|---|---|---|
| `so100` | `lerobot/svla_so100_pickplace` | `so100` |
| `so100-blue` | a blue SO-100 policy that names that prefix | `so100-blue` |
| `so100-red` | a red SO-100 policy that names that prefix | `so100-red` |

`so100-blue` and `so100-red` are keys in the same safetensors file. They are not `svla_so100_pickplace`. A `so100` chunk that cites `so100-blue` is refused.

The header carries each tensor sha256 in the four 64-hex fields after the stats key and before the noise sha256, in this order: action mean, action std, state mean, state std. Action mean and action std must equal the two rows of the chunk's stats key in the table above. State mean and state std must equal the two rows below. A mismatch is refused. The model safetensors has no `observation.state` tensors. STATE mean and std come from `meta/stats.json` `observation.state` at the revision above, cast to little-endian f32 in joint order. Their sha256 values, of those 24 bytes, are:

| Tensor | sha256 |
|---|---|
| `observation.state` mean | `c0087077ccbf5362ee8dab2ec5cae2ee915123b176ade6c0be1fad3feb7a7eca` |
| `observation.state` std | `09d52284cad4193fdaf1fe35c89dab6aa2711514b3043296b7811c13cb8f13ef` |

MEAN_STD is `src/lerobot/processor/normalize_processor.py` at `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a`, lines 349–362, `eps` default `1e-8` (line 97), and the saved preprocessor and postprocessor configs both set `"eps": 1e-08`. Lines 341–345 cast the stored stats onto the tensor's device and dtype before either formula runs.

Forward, STATE into the network: `(x - mean) / (std + 1e-8)`.

Inverse, ACTION out of the network: `y * std + mean`, with `std` and `mean` already cast to the tensor dtype. The inverse does not add epsilon. A missing mean or std for a MEAN_STD feature is refused. It is not IDENTITY.

## Noise

`noise_sha256` is SHA-256 of this byte string, and of nothing smaller. A missing hash is refused. The placeholders `"11" * 32` and `"22" * 32` are not values of this hash.

Items 1, 4, and 7 are strings under the version 3 string rule: a `u32` little-endian byte length, then the UTF-8 bytes, no trailing NUL.

1. `u32` length `3`, then UTF-8 `f32`.
2. `u32` little-endian rank `3`.
3. Three `u64` little-endian extents: `1`, `50`, `32`. That is `x_T` shaped `(batch, chunk_size, max_action_dim)`.
4. `u32` length `2`, then UTF-8 `le`.
5. The `50 * 32` little-endian f32 payload, time-major then joint.
6. The `u64` little-endian caller seed. This is the seed before `seed | 1`.
7. `u32` length `24`, then UTF-8 `xorshift64-box-muller-v1`.

The reference implementation is toolchain commit `914fcc76e7d86a79d657b69435481c5dcfdc3314`. On that commit, `noise_from_seed` is `crates/ferric-smolvla/src/lib.rs:59` and `next_unit` is `lib.rs:102`. Fleet commit `b855cbe8340dbc4d8c7f2b4de04d8b7d67a2579e` has the same `noise_from_seed` at `lib.rs:59` and `next_unit` at `lib.rs:178`. `ln`, `sin`, and `cos` are `crates/ferric-smolvla/src/noise_math.rs:70`, `:254`, and `:274` on both commits. `k_cos` is `noise_math.rs:146` and `rem_pio2` is `noise_math.rs:198`, also on both. For an odd caller seed the generator state starts at `seed | 1`, which is that seed. Each step is `x ^= x << 13; x ^= x >> 7; x ^= x << 17`. The unit is `(x >> 11) as f64 / 2^53`.

`seed | 1` would make seeds `2k` and `2k+1` the same payload. The version 3 encoder and the version 3 decoder both refuse an even seed, so two caller seeds cannot share one payload. The header stores that odd caller seed as the `u64` immediately after the noise sha256. A receiver uses it to enforce the refusal and to recompute the seven-item noise string. Item 6 of that string is this same odd caller seed.

Box–Muller, in this order, for each pair `(u1, u2)`:

```
r = sqrt(-2 * ln(u1))
theta = (2 * pi) * u2
sample0 = r * cos(theta)
sample1 = r * sin(theta)
```

`pi` is the f64 constant. `sqrt` is IEEE-754 square root. Each sample is cast f64 to f32, round ties to even, and `sample0` is emitted before `sample1`. The xorshift state can be nonzero and still have its top 53 bits clear, so `u1 = 0` is reachable. Then `ln(0)` is `-∞` and `r` is `+∞`. `+∞` times `0` is NaN. The encoder refuses any seed whose payload contains a non-finite value, from `u1 = 0`, from `+∞·0`, or from anything else. The sample is not replaced with a finite substitute. Seed `0x5a17` does not hit that case: every sample is finite. No NaN bit pattern is pinned.

`ln` follows fdlibm `e_log.c` and is bounded by less than 1 ulp. `k_sin`, `k_cos`, and `rem_pio2` follow the FreeBSD msun sources as revised in musl 1.2.5: musl `__cos.c` lines 67–70, `__rem_pio2.c` lines 38–39 and 119–132, and FreeBSD `k_cos.c` line 74. That is not the Sun 5.3 qx form. On `[-π/4, π/4]` the kernel remainder is at most `2^-58`. These functions do not call platform libm. Platform `ln`, `sin`, and `cos` are not correctly rounded and are not interchangeable across machines. The producer's emitted payload bytes are authoritative. A verifier holding them hashes them and does not regenerate them. The chunk itself does not carry the payload.

The noise hash has no nonce. That is intentional. The chunk is bound to the episode by the envelope's `episode_nonce` together with `chunk_sha256`.

Golden, seed `0x5a17` (23063), 1600 little-endian f32 samples:

- payload sha256, item 5 alone: `09adba264f855351fd2a0911ce66497cba8e54f9ca73d6b8d905c22918e0e4c2`
- `noise_sha256`, the seven-item string above: `8db2ebc813c83673d6b61b10b2c2a2352018371cbfe9c661f3b405ac6370b61c`

The payload sha256 is the value already stored as `noise_sha256` in `docs/fleet/smolvla-cpu-infer.json`. Replacing platform libm with the fdlibm functions did not change those bytes, and it did not change that receipt. The v3 `noise_sha256` is the longer string, not the payload alone.

## Interop with ferromorphic-chunk-binding 1

The binding's chunk fields, and where they sit in this canonical layout:

| Binding field | Ferric field |
|---|---|
| `chunk_sha256` | `digest_sha256`, SHA-256 of the full canonical bytes, magic and header included |
| `t0_num_ns` | `t0_num_ns` |
| `t0_den` | `t0_den` |
| `period_num_ns` | `period_num_ns` |
| `period_den` | `period_den` |
| `n_action_steps` | `n_action_steps` |
| `envelope_sha256` | not a byte of the chunk. The `ferromorphic-chunk-binding 1` record carries it. |
| `epoch` | not a byte of the chunk. The `ferromorphic-chunk-binding 1` record carries it, and it must equal the envelope's epoch. |
| `chunk_index` | not a byte of the chunk. The `ferromorphic-chunk-binding 1` record carries it. Envelope v2 has no `chunk_index`. |

Binding 1's decimal text for `t0_num_ns`, `t0_den`, `period_num_ns`, and `period_den` is exactly these `u64` values. A value above `u64::MAX` is refused.

The chunk period `period_num_ns / period_den` must equal the envelope's `period_num_ns / period_den` exactly, both rationals already reduced. A chunk whose period is a different reduced fraction is refused.

`t0` is on the grid if and only if some integer `k >= 0` satisfies

```
t0_num_ns * period_den == (origin_ns * period_den + k * period_num_ns) * t0_den
```

Every product and sum in that identity is checked `u128` arithmetic. `origin_ns` and `period` come from the episode below. Equivalently, `d = t0_num_ns * period_den - origin_ns * t0_den * period_den` is `>= 0` and `period_num_ns * t0_den` divides `d`. A `t0` before `origin` is refused. Overflow is refused.

Within one episode, a later `chunk_index` must have a strictly larger `k`. That is the same requirement as a strictly increasing `t0`. `k` is not required to equal `chunk_index * n_action_steps`.

Late/Stale hold policy, `chunk_index` ordering, and the binding-hash stamp on StepReceipts are ferromorphic verifier checks under binding 1. Version 3 does not reimplement them in ferric.

## ferromorphic-episode 2

Fields, in this order:

1. `version`
2. `epoch`
3. `dt_num_ns`
4. `dt_den`
5. `dt_bits`
6. `period_num_ns`
7. `period_den`
8. `origin_ns`
9. `episode_nonce`

## SO-100 profile clock

The envelope itself carries any reduced period and any reduced dt. `ferromorphic-episode 2` does not refuse a different pair.

The ferric SO-100 profile is what refuses anything else. For this profile the only legal clock is 30 Hz with 20 substeps. `period_num_ns / period_den` is `100000000/3`. `dt_num_ns / dt_den` is `5000000/3`, which is that period divided by 20. Both fractions are already reduced (`gcd` with 3 is 1). Any other period or dt is refused.

## Provenance

The weight pin is `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`, the check at `crates/ferric-smolvla/src/bin/ferric-smolvla.rs:85`. Version 2's sentence "No weight sha256 is in this tree" is stale and is removed there.

`crates/ferric-smolvla/src/bin/ferric-smolvla.rs:301-302` still defaults a missing stats hash to `"11" * 32` and a missing noise hash to `"22" * 32`. Version 3 removes those defaults. A missing stats hash or a missing noise hash is refused. It is not filled in. The weight default on line 300 is already the pin above; stats and noise have no such pin and get none here.

## Digest

`digest_sha256` is SHA-256 of the full canonical byte string as emitted, magic and header included. The hash is not appended to those bytes. Ferromorphic chunk binding 1 calls this value `chunk_sha256`.

## Pin label

The LeRobot sources for the version 2 pins are commit `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a` (2026-07-23). GitHub compare puts that commit on the ancestry of tag `v0.6.1` (`7e241bd630a3719a56157a497ce5d08f244784f1`, 112 commits later) and of tag `v0.7.0` (`410c095eacc0011d998a73383eb94abed2715af0`, 364 commits later). The label for the pin is `a0eb860d (v0.6.1 lineage)`. Tag `v0.7.0` remains the name of `410c095e` only.
