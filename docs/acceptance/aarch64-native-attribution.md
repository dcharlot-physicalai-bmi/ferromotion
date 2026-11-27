# aarch64 native-array residual: attribution

The Thor2 stage-5 receipt for `2aeb3611c9f3edca1961eb25ee374df56229d799` (`rust212-20261120T135357Z/RECEIPT.txt`, sha256 `82263c2b9ec4dd633612e931d9a78c997b9f319d814ac27851726ebbee9bf45e`) has `mj_step` bit-exact over 20 steps on 212 of 212 models and 11 native fields short of 636 of 636. Every one of those 11 fields is attributed below to this port, not to the aarch64 oracle. The commit that adds this file fixes all three sites. With the fix, every native field is 636 of 636 against both contract-off oracles, and `mj_step` stays 212 of 212.

Equality is bitwise throughout: an `f64` passes only when `to_bits` agrees. No golden, dump, receipt or oracle was changed or re-pinned.

## Identities

* aarch64 oracle `mujoco-3.13.0-aarch64-contract-off`, libmujoco sha256 `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9` (thor2 `build-20261109T213225Z`, clang 18.1.3, `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a`, `-rtlib=compiler-rt`).
* x86 oracle `mujoco-3.13.0-x86_64-contract-off`, libmujoco sha256 `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`.
* Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`.
* oracle dumps: native `a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7` (73420440 bytes), step `a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be`.

## The aarch64 oracle is the x86 oracle on every dumped array

* `run-20261109T215401Z/native.txt`, written by `oracle_dump native` linked to `9164828b`, has sha256 `a3168d76…`. That is the x86 dump's hash, byte for byte.
* On 2026-11-27 the same `oracle_dump` (sha256 `514d43bfbe9f3c126c6eb6f879b2eec111bb6b51dfb16a68b79cf4f67cb508cd`) was re-run against `9164828b` into a new directory (`attr-aarch64-20261127/native-rerun.txt`, with `MUJOCO_PLUGIN_DIR` set as `run-212.sh` sets it). It reproduced `a3168d76…` again. `mj_forward` is recomputed by the linked library for every state, so this is a fresh aarch64 measurement, not a copy.
* Disassembly of `9164828b` (`objdump -d`, 680037 lines) has 0 `fmadd`, `fmsub`, `fnmadd`, `fnmsub`, `fmla` or `fmls`. All 219 `compile_commands.json` entries carry `-ffp-contract=off`. `-ffp-contract=off` was applied on aarch64.
* The libm calls (`sincos`, `pow`, `exp`, `atan2`, `sin`, `asin`, `tan`, `cos`, `log`, `acos`) did not change any dumped field, since the two dumps are identical. The port calls the host libm through `std` on both architectures. It does not use the `libm` crate and has no `target_arch` branch in `ferromotion-core`.

So there is no aarch64 oracle property to record for these arrays. Both oracles give the same bits.

## The port gave the same wrong bits on both architectures

The x86 receipt for the same commit (`docs/acceptance/receipts/2aeb3611c9f3edca1961eb25ee374df56229d799.supersedes-c590d3a2.md`) has the same 11 tallies, the same worst values and the same first-difference notes. Its native log sha256 is `43cd53254275a28bdc4c5e0c2bea9c481dd4e6e99243f3e1948891d8f91cd841`. From its `first_stage` lines on, the thor2 native log hashes to the same `43cd5325…` with rustc 1.99.1 (`rust212-20261127T132613Z`) and with rustc 1.100.0 (`rust212-20261120T135357Z`, `rust212-20261127T131407Z`).

## Per field, before the fix

Rows are (field, model, state): elements that differ, the largest absolute difference, and the largest distance in `f64` bit patterns. Near-zero elements give large bit distances for small absolute differences (`anymal_c/scene_mjx` state 0).

| field | model and state | elements | max abs | max bit distance |
|---|---|---|---|---|
| qfrc_spring | tetheria_aero_hand_open/left_hand.xml state 1 | 1 | 4.34e-19 | 1 |
| qfrc_spring | tetheria_aero_hand_open/left_hand.xml state 2 | 1 | 5.42e-20 | 1 |
| qfrc_spring | tetheria_aero_hand_open/right_hand.xml state 1 | 2 | 2.17e-19 | 1 |
| qfrc_spring | tetheria_aero_hand_open/right_hand.xml state 2 | 1 | 8.67e-19 | 1 |
| qfrc_spring | tetheria_aero_hand_open/scene_right.xml state 1 | 2 | 1.08e-19 | 1 |
| qfrc_spring | tetheria_aero_hand_open/scene_right.xml state 2 | 1 | 8.67e-19 | 1 |
| efc_J | anybotics_anymal_b/scene.xml state 2 | 5 | 2.78e-17 | 4 |
| efc_J | anybotics_anymal_c/scene.xml state 2 | 8 | 1.11e-16 | 16 |
| efc_J | berkeley_humanoid/scene.xml state 2 | 22 | 1.67e-16 | 38 |
| efc_J | pndbotics_adam_lite/scene.xml state 1 | 24 | 1.11e-16 | 64 |
| efc_vel | anybotics_anymal_c/scene.xml state 2 | 2 | 1.11e-16 | 1 |
| efc_vel | berkeley_humanoid/scene.xml state 2 | 3 | 1.11e-16 | 1 |
| efc_vel | pndbotics_adam_lite/scene.xml state 1 | 4 | 3.33e-16 | 10 |
| efc_aref | anybotics_anymal_c/scene.xml state 2 | 1 | 1.42e-14 | 1 |
| efc_aref | berkeley_humanoid/scene.xml state 2 | 2 | 5.68e-14 | 1 |
| efc_aref | pndbotics_adam_lite/scene.xml state 1 | 1 | 2.27e-13 | 1 |
| efc_b | anybotics_anymal_b/scene.xml state 2 | 1 | 1.42e-14 | 1 |
| efc_b | anybotics_anymal_c/scene.xml state 2 | 2 | 2.13e-14 | 32 |
| efc_b | berkeley_humanoid/scene.xml state 2 | 2 | 1.14e-13 | 1 |
| efc_b | pndbotics_adam_lite/scene.xml state 1 | 3 | 1.82e-12 | 16 |
| efc_diagA | anybotics_anymal_c/scene_mjx.xml state 2 | 4 | 1.73e-18 | 1 |
| efc_R | anybotics_anymal_c/scene_mjx.xml state 0 | 4 | 3.47e-18 | 1 |
| efc_R | anybotics_anymal_c/scene_mjx.xml state 2 | 8 | 1.06e-22 | 1 |
| efc_D | anybotics_anymal_c/scene_mjx.xml state 0 | 4 | 7.11e-15 | 1 |
| efc_force | anybotics_anymal_b/scene.xml state 2 | 33 | 2.85e-10 | 908 |
| efc_force | anybotics_anymal_c/scene.xml state 2 | 21 | 3.64e-11 | 482 |
| efc_force | anybotics_anymal_c/scene_mjx.xml state 0 | 16 | 8.1e-13 | 302 |
| efc_force | berkeley_humanoid/scene.xml state 2 | 29 | 9.01e-12 | 701 |
| efc_force | pndbotics_adam_lite/scene.xml state 1 | 59 | 1.91e-11 | 650 |
| qfrc_constraint | anybotics_anymal_b/scene.xml state 2 | 12 | 2.76e-10 | 8968 |
| qfrc_constraint | anybotics_anymal_c/scene.xml state 2 | 8 | 1.27e-11 | 384 |
| qfrc_constraint | anybotics_anymal_c/scene_mjx.xml state 0 | 18 | 2.5e-12 | 49807360 |
| qfrc_constraint | berkeley_humanoid/scene.xml state 2 | 18 | 1.08e-11 | 820 |
| qfrc_constraint | pndbotics_adam_lite/scene.xml state 1 | 25 | 1.46e-11 | 65 |
| qacc | anybotics_anymal_b/scene.xml state 2 | 16 | 6.37e-12 | 14 |
| qacc | anybotics_anymal_c/scene.xml state 2 | 18 | 3.64e-12 | 72 |
| qacc | anybotics_anymal_c/scene_mjx.xml state 0 | 18 | 2.1e-12 | 101855263 |
| qacc | berkeley_humanoid/scene.xml state 2 | 18 | 1.41e-11 | 248 |
| qacc | pndbotics_adam_lite/scene.xml state 1 | 26 | 7.28e-12 | 239 |

## The three sites

Each site had its fused form on the contract-off path, where the C rounds the product and then the sum.

1. `mj_springdamper`, tendon springs and dampers: `qfrc_spring[k] += J * frc_spring` and `qfrc_damper[k] += J * frc_damper`. The port had `jv.mul_add(frc_spring, qfrc_spring[k])` unconditionally (`mujoco_smooth.rs`). This is the first and only differing field on `tetheria_aero_hand_open` (`qfrc_spring`, 1 bit). `mj_springdamper` in `9164828b` (`0x1baa0c`) has 475 `fmul`, 289 `fadd` and 188 `fsub`, with no fused operation.
2. `mjc_PlaneCylinder`, triangle points A and B: `mji_cross(vec1, vec, axis)` (`res[0] = a[1]*b[2] - a[2]*b[1]`, …) and `mji_addToScl3` (`res[i] += vec[i]*scl`). The port fused both (`mujoco_collision.rs`). This is the first divergence on `anymal_b`, `anymal_c/scene`, `berkeley_humanoid` and `pndbotics_adam_lite`. A probe linked to `9164828b` (`contact_probe`, sha256 `bbafee82bec7c89b73cf60c0ab6ea1b21c978a37f381a40047acb95b1d2820f4`, reads the oracle library and writes nothing to it) printed `d->contact` after `mj_forward` at the dumped states. Before the fix the port's contacts differed from the oracle's in one coordinate of `pos` only: `berkeley_humanoid/scene` state 2 contact 30 (floor, geom 19) `pos.x` `3fe30ae42ddd7fc4` against `3fe30ae42ddd7fc3`; `anymal_b/scene` state 2 contact 63 `pos.y`; `anymal_c/scene` state 2 contacts 12, 13, 16 and 17 `pos.x`; `pndbotics_adam_lite/scene` state 1 contact 36 `pos.x` and `pos.y`. `dist` and `frame` already matched, so `efc_pos` matched and `efc_J` did not. After the fix the port's contact lists for those four states hash to the same values as the oracle's (`54520f2f…`, `d772ba07…`, `7f0a6312…`, `04834f35…`). In `mjc_PlaneCylinder` (`0x154038`) the cross product is six `fmul` then three `fsub` (`0x1542e0`–`0x154308`). The function has 26 `fmul`, 29 `fadd` and 11 `fsub` and no fused operation.
3. `mj_diagApprox`, pyramidal rows: `tran + fri*fri*(j < 2 ? tran : rot)`. The port had `(fri * fri).mul_add(w, tran)` (`mujoco_efc.rs`). `efc_J` and `M` already matched on `anymal_c/scene_mjx` (impratio 100, pyramidal). The first differing fields were `efc_R`, `efc_D` and the adjusted `efc_diagA`, then the solve. In `mj_diagApprox` (`0x15eaac`) the pyramidal pair is `fmul v1, v1, v1` (`fri*fri`), `fmul v1, v1, v5.d[1]` (`* w`), `fadd v3, v0, v1` (`tran + …`) at `0x15f474`–`0x15f488`. The function has 12 `fmul` and 11 `fadd` and no fused operation.

`efc_vel`, `efc_aref`, `efc_b`, `efc_force`, `qfrc_constraint` and `qacc` are downstream of those three. They became exact with no other change.

The published default keeps every fused form. Each change is behind `fp_contract::contract_off()`.

## After the fix

* x86, `64156abe` dumps, on the box: rustc 1.99.1 and rustc 1.100.0 each give every native field 636 of 636 and 212 of 212 for `mjModel`, with `first_stage` `no_listed_stage_mismatch` on all 212. The native log is `47956e7bdd28324a64f2afc81679c46094c7d8ceecdaf3a8cb6bc77b5c457db0` with both toolchains. `menagerie_native_step` is 212 of 212 bit-exact over 20 steps, and every ledger line is `bit_exact`.
* aarch64, thor2, `9164828b` dumps, through `run-rust-212.sh` on a copy of the 2aeb3611 codeload tree with this fix applied (`fix.diff` sha256 `d90b11c6798a81eba3554a699c2006ec9c6d08ceaaa47702342312a433bcabb4`, PROVENANCE says so):
  * rustc 1.99.1 (`58cf8f9c3`): `parity_accept models=212 steps=20 tallies=76`, receipt `rust212-20261127T131736Z/RECEIPT.txt` sha256 `f3e5c56bfcb3dba25d8e0b20ee875d13d8752b06ac8ef2f1ea18e85b1d287009`.
  * rustc 1.100.0 (`0ef1bc386`): `parity_accept models=212 steps=20 tallies=76`, receipt `rust212-20261127T132120Z/RECEIPT.txt` sha256 `4044aa73e08e6631e1ca8a4d853d9910957068eaa3b10b84e4f7c8f31789736f`.
  * From `first_stage` on, both aarch64 native logs hash to `47956e7b…`, the x86 log.
* `cargo test --release -p ferromotion-core --lib` (rustc 1.100.0, published default): 1010 passed, 4 failed. The same 4 fail without the fix: `a_pid_plugin_steps_as_mujoco_does_bit_for_bit`, `geom_frames_match_mujocos_kinematics_bit_for_bit`, `spatial_tendons_step_as_mujoco_does_bit_for_bit` and `the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit`.

These aarch64 receipts are experiment receipts on a patched copy, not the stage-5 codeload receipt. That receipt needs `05-rust-rereun.sh` on a pushed commit that is reachable on `cursor/ferromotion-fidelity-conformance-bb9d`.

## Baseline reruns, rustc 1.99.1 and 1.100.0

`run-rust-212.sh` on the unpatched 2aeb3611 tree:

* rustc 1.100.0 (`0ef1bc386`): `rust212-20261127T131407Z/RECEIPT.txt` sha256 `7492eadfe00d0a9fa941cef92525f91f121b8f2b59cbb9be950ea329d24161e8`, `parity_mismatch models=212 steps=20 unequal_tallies=11 stepped=212 of 212`.
* rustc 1.99.1 (`58cf8f9c3`): `rust212-20261127T132613Z/RECEIPT.txt` sha256 `83919cffae933edb5938696437c1f219d79c2b996995e4168f6c2bc5ecd6d36a`, the same result.

The receipts' `cargo_vv` lines are copied from stage 2's `provenance/cargo-vv.txt`. They are not the cargo the runner used. All four rust212 receipts above say `cargo 1.99.1`. The compiler was read from the binaries (`rustc/<commit-hash>` in `menagerie_native`) and from `cargo-target/.rustc_info.json`. `rust212-20261120T135357Z` (`82263c2b…`) records cargo 1.99.1, but its binaries were built by rustc 1.100.0 (`0ef1bc386`), the thor2 stable toolchain.
