# Panda reach contact, revision 3, 2026-11-27

This file supersedes `docs/acceptance/receipts/2026-11-27-panda-contact-reach-2.md`.

* superseded file sha256 `aae81216b943a5cb6e6c446c7f39edcc0c122efe526d27035c9267b6f499fb4a`
* that file supersedes `docs/acceptance/receipts/2026-11-27-panda-contact-reach.md`, sha256 `77b4cac62fa2bfb15c9b6aba5faef2fb89789dba88f2884f7495ec4fae7e6936`
* neither of those files is modified

`docs/acceptance/receipts/2026-11-20-panda-contact.md` stays. This episode does not set a flag. `JOINT_LOOP_EXECUTED` stays the signed contact-free scope. `EPISODE_ENVELOPE_INTEROPERABLE` stays false.

The episode hashes below are the same values receipt -2 recorded. The code that produced them is commit `daef5faf088c5c508843215ff5a1690f6854e9bf`. The per-tick control assertions and the non-zero `qfrc_applied` comparison are commit `1a5a6b34ef18f6126ec4309b232a0eaa5637664a`. This file does not change that code.

## What the torque does

This is a gravity-loaded settling contact. The arm starts 1 mm into the floor (`f64::from_bits(0xbf50624dd2eef860)` = `-9.9999999996171901e-4` m). Gravity and the contact force carry the motion. The neural torque only slightly perturbs that settling. This is not a competence claim.

Measured on this host, plant versus the zero-torque run, same pose and same stepper:

* Tick 0 has zero readout torque. The `qfrc_constraint` bits match. Both norms are `4.7127916425884052e1`. Both minimum-distance bits are `bf50624dd2eef860`.
* Ticks 1–15: the `qfrc_constraint` bit vectors differ. The test asserts that difference. The largest per-joint absolute difference is `1.8151861339062059e-1` N·m, on tick 13. The largest `|tau|` on a tick is `2.0000000000000001e-1` N·m.
* The minimum distance differs by at most `7.5336878527232365e-8` m (tick 15). Ticks 0 and 1 have the same minimum-distance bits.
* The zero-torque run also separates monotonically. Its minimum distance stays negative and increases every tick. The test asserts those bits.

| tick | plant norm | zero-torque norm | max \|joint diff\| N·m | max \|tau\| N·m | \|min-distance diff\| m | plant min bits | zero-torque min bits |
|---|---|---|---|---|---|---|---|
| 0 | `4.7127916425884052e1` | `4.7127916425884052e1` | `0` | `0` | `0` | `bf50624dd2eef860` | `bf50624dd2eef860` |
| 1 | `4.4953910237920759e1` | `4.5007106521191957e1` | `5.6017259448381651e-2` | `5.0000000000000003e-2` | `0` | `bf5037999123d0c0` | `bf5037999123d0c0` |
| 2 | `4.3248283844774768e1` | `4.3252633750480683e1` | `5.2469542616984555e-2` | `5.0000000000000003e-2` | `9.6025695034418490e-10` | `bf4fd71e770fd700` | `bf4fd71c6727b040` |
| 3 | `4.1781382460220712e1` | `4.1805481276421361e1` | `6.0101369202028021e-2` | `5.0000000000000003e-2` | `2.9737484003033998e-9` | `bf4f0be1726a3600` | `bf4f0bdb0f945480` |
| 4 | `4.0532608289274862e1` | `4.0615868896204390e1` | `9.8975433454416617e-2` | `1.0000000000000001e-1` | `6.0617481242863924e-9` | `bf4e1a4d8ae0d100` | `bf4e1a4086659c40` |
| 5 | `3.9588322506985712e1` | `3.9641998664945447e1` | `6.7425228220721056e-2` | `1.0000000000000001e-1` | `1.0847925321910701e-8` | `bf4d0cf2f8578400` | `bf4d0cdbaca1c080` |
| 6 | `3.8738249091279243e1` | `3.8848847912386574e1` | `1.3718816203527684e-1` | `1.5000000000000002e-1` | `1.6126158089435627e-8` | `bf4bec792a1e0b80` | `bf4bec5688ab0f00` |
| 7 | `3.8133205419000291e1` | `3.8207076730092155e1` | `1.3307987304627389e-1` | `1.5000000000000002e-1` | `2.2451420285396484e-8` | `bf4abff4f35cbdc0` | `bf4abfc4bc903d80` |
| 8 | `3.7695476815920344e1` | `3.7692077672045677e1` | `1.0613430674685365e-1` | `1.5000000000000002e-1` | `2.9836414096318098e-8` | `bf498d254e178480` | `bf498ce53b598940` |
| 9 | `3.7259219268465174e1` | `3.7283171174950816e1` | `1.2671347460712212e-1` | `1.5000000000000002e-1` | `3.7373079318780977e-8` | `bf4858a9a331cec0` | `bf48585961207e00` |
| 10 | `3.6934268237121906e1` | `3.6962936699381345e1` | `1.4740919766310512e-1` | `2.0000000000000001e-1` | `4.5185404057290768e-8` | `bf47263445ddc080` | `bf4725d33ced7980` |
| 11 | `3.6619687811703102e1` | `3.6716663205228521e1` | `1.5270529510249453e-1` | `1.5000000000000002e-1` | `5.2903807779003387e-8` | `bf45f8ae290b1b40` | `bf45f83c8cde1340` |
| 12 | `3.6441255757939601e1` | `3.6531900609775313e1` | `1.4878850336561378e-1` | `1.5000000000000002e-1` | `5.9945160217622639e-8` | `bf44d2586386d280` | `bf44d1d7a85388c0` |
| 13 | `3.6296919379889857e1` | `3.6398094453183482e1` | `1.8151861339062059e-1` | `2.0000000000000001e-1` | `6.5931133441665679e-8` | `bf43b4e8efd6e380` | `bf43b45b59d0c380` |
| 14 | `3.6239870719157068e1` | `3.6306287868102288e1` | `1.3315581678665467e-1` | `1.5000000000000002e-1` | `7.1879767547800011e-8` | `bf42a1a4d6acadc0` | `bf42a10a7a5abb00` |
| 15 | `3.6176722062276376e1` | `3.6248877333887343e1` | `1.1510118135171155e-1` | `2.0000000000000001e-1` | `7.5336878527232365e-8` | `bf419969c2648200` | `bf4198c7f9817180` |

Norms are the Euclidean norm of the seven `qfrc_constraint` components. The plant column is the contacting loop. The zero-torque column is the same plant with a zero `qfrc_applied`.

## Non-zero qfrc_applied

The menagerie 20-step comparison leaves `qfrc_applied` at zero. A separate step in the same job compares a non-zero value to libmujoco `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`.

`scripts/qfrc_applied_oracle.py` builds a two-hinge arm (`implicitfast`, timestep 0.002) and sets `qfrc_applied` to `1.25` and `-0.4`. It runs four `mj_step`s on the contract-off library. `examples/qfrc_applied_oracle` repeats those steps with `MjcfTree::mujoco_native_step_witness_applied`, `ctrl` empty, and checks `qpos`, `qvel`, `act`, `qacc_warmstart`, and `qfrc_constraint` with `f64::to_bits`.

oracle-profile run `46021395105`, job `contract-off conformance`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395105/job/140581645364>, step `Non-zero qfrc_applied`:

* `qfrc_applied non-zero, 4 steps, nv=2`
* `qfrc_applied non-zero, bit-exact for 4 steps (qpos qvel act qacc_warmstart qfrc_constraint)`

The same job's `20-step comparison` step still reports `step exact 212 of 212; ledger 212`. That trajectory's `qfrc_applied` is zero. The non-zero coverage is the four-step arm above, not those 212 models.

## Profile

The plant is `MjcfTree::mujoco_native_step_witness_applied` after `install_policy("mujoco-3.13.0-x86_64-contract-off", "64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc", "")`. `ctrl` is seven zeros. The readout torque is `qfrc_applied`. The published default is not installed and is not changed.

* profile id `mujoco-3.13.0-x86_64-contract-off`
* libmujoco sha256 `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`

## Scene

Menagerie `franka_emika_panda/panda_nohand.xml` at `4d038b3feae26ec82b46a4d586379114012a8ac7`, sha256 `e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071`. The wrapper is `crates/ferromotion-core/tests/fixtures/franka_emika_panda/contact_reach.xml`, sha256 `473d6d43acb9a67e051677149e21b5bbb4df8ee2c452070233ffb6a7680472ad`. It adds a floor plane at `z = 0`. The Menagerie XML and meshes are not edited. Mesh sha256s are the list in `docs/acceptance/receipts/2026-11-20-panda-contact-assets.sha256`.

Joint4 is `f64::from_bits(0xbfff5c4214179e64)` = `-1.9600239548543863`, the shallow end of forty bisections of joint4 on `[-2.2, -1.9]` with the other joints held at `0 0.8 0 _ 0 2.2 0`. The start pose is `0 0.8 0 -1.9600239548543863 0 2.2 0`.

Every tick 0–15 of the contacting loop has three solver contacts, all `world/floor` with `link7/geom59`. link7 rides a joint, so `ncon_moving` is 3 and `nefc` is 12. There is no link5 contact and no link6 contact. The minimum distance stays negative and increases every tick.

The zero-torque run asserts, on every tick: `ncon` 3, `ncon_moving` 3, the same three `world/floor`–`link7/geom59` pairs, the minimum-distance bits in the table, clamp counts 0, and monotonic separation. The contacts-disabled run asserts, on every tick: `ncon` 0, no minimum distance, no pairs, and clamp counts 0. Clamp counts are the observation joint-limit clamp, the encoder ±1e-6 A clamp, the readout ±`TAU_LIM` clamp, and actuator `forcerange` binds.

`qfrc_constraint` is non-zero on joints 1–7 at every tick 0–15 of the contacting loop. The non-zero readout-torque ticks are 1–15. Tick 0's torque is zero. On ticks 1–15 the contacting `qfrc_constraint` bits differ from the zero-torque run.

`N` is 16. The window is 4. Period `2_000_000` ns, matching the file timestep bits `3f60624dd2f1a9fc`. Ferromorphic pin `82211dc861af4ecc1afa9603c2caf20d52712b6d`.

The closed loop was run twice. Tick hashes and the plant hash matched. The zero-torque control was run twice and matched itself. Its final `qpos`/`qvel` hash differs. The contacts-disabled loop's final state differs from the contacting loop.

* spikes: 54
* first spike tick: 1
* non-zero-tau ticks, by index: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
* envelope sha256 `170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a`
* final `Sim::state_bytes` sha256 `4e49f36ed5148e355c13dd8a58c7d6b48d0062fb4cd56cce059117178ea40e50`
* contacting loop final `qpos`/`qvel` sha256 `c3b70e92db29d9f112626d68d2d5c387895fb6f72dc95bf98d70262c34d300b2`
* zero-torque final `qpos`/`qvel` sha256 `b3dc651ba290426b3e81a70db2e5aa59dbcc9fa787498934c07b5882925e04e8`
* contacts-disabled loop final `qpos`/`qvel` sha256 `a3a1c16ef5627785074a3c219490e1296a8233e62495430b517272f815a23852`

* Cargo.lock sha256 `fb1c9967807c1121e23e6ed2575eb023f08a067cd944202e438092b06ef55ae4`
* `rustc -Vv`: rustc 1.100.0 (0ef1bc386 2026-11-09), commit-hash `0ef1bc386059596dada8fb65b50425aab2ea637a`, host `x86_64-unknown-linux-gnu`, LLVM 23.1.5
* cargo 1.100.0 (3acc79c0c 2026-11-05)

| tick | sha256 | ncon | nefc |
|---|---|---|---|
| 0 | `59e1f6b5238ea02b4f521bfb804f430a403e6885e7545df393200bc21d0f343e` | 3 | 12 |
| 1 | `edeaba3a38f36e7d5623be2dd17a815814c9bad11dae64dd055800776df24a11` | 3 | 12 |
| 2 | `79d7fb2c1b08465a658616c672ad720005f535c8a9ccadda08369b4fef7d30ea` | 3 | 12 |
| 3 | `6bcb14857a948b97d4bb1d6b9b7f8b47ca6df0d294a4e452461845cb904b5fcb` | 3 | 12 |
| 4 | `5f2e3460b0b0013be644d17c907c74512814fe44d19bcd9279e971135dfa2f16` | 3 | 12 |
| 5 | `b0838e7e582dd97a02d9b41b3df4bfad062ebe46ad0db01c19d8a69f9d39ac3f` | 3 | 12 |
| 6 | `2536a08458cba8cd9d09110bf86d71cfe89f9467bb08b465b659d43428735b18` | 3 | 12 |
| 7 | `fa816e4ddb81770d6f854527fa5c6f0e862a25347cff9ade8f9570821ae95705` | 3 | 12 |
| 8 | `0c98a2b6f5d653bc93eeb9ad30e7974e2ac635ebd1047dce8b67add333bc6966` | 3 | 12 |
| 9 | `fcec610f7c47c3cb6e69d16a1097ef9d4b56e774566692c25b6c6d63ededb2b7` | 3 | 12 |
| 10 | `337a1024d6715899539a80a4851d5849d474e3b40a8236632e02b894690a0c56` | 3 | 12 |
| 11 | `b212a05de02aad23550cd4395b71d3653d25278edb5957380e03c7df8eb121e8` | 3 | 12 |
| 12 | `c849c9347a37532deb40a815734d8b65c3e84123a4d5ceb6bc37da883d3ee5db` | 3 | 12 |
| 13 | `905fd8c71e000c874a67854c2347a2312ca2a552fb60b8ab4c8d2f0d8294916c` | 3 | 12 |
| 14 | `15d05a124f8b09827aacbc58773d6d4c3859d8b51921be2bd9d74e1866179f4a` | 3 | 12 |
| 15 | `b8f43856d97bfe62b2175b0bf36f4a01e989ebc1055b0a3bb2d6146c16943f62` | 3 | 12 |

## CI on `1a5a6b3`

Every claim below is a job on commit `1a5a6b34ef18f6126ec4309b232a0eaa5637664a`.

Workflow CI, run <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395196>, is red because the `test` job fails.

* `test`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395196/job/140581645620>, failure. Log line: `test result: FAILED. 1014 passed; 3 failed`. The three failures are `geom_frames_match_mujocos_kinematics_bit_for_bit`, `spatial_tendons_step_as_mujoco_does_bit_for_bit`, and `the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit`.
* `integration tests`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395196/job/140581645544>, success. This job runs `panda_contact_reach`.
* `clippy`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395196/job/140581645196>, success.
* `wasm32`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395196/job/140581645329>, success.
* `docs`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395196/job/140581645790>, success. The job name on that page is `docs`.

Workflow oracle-profile, run <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395105>, success.

* `contract-off conformance`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395105/job/140581645364>, success. `step exact 212 of 212; ledger 212`. The non-zero `qfrc_applied` step is in this job and is bit-exact for 4 steps.
* `contract-off native`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395105/job/140581645078>, success.

Workflow menagerie-conformance, run <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395165>, failure.

* `bit-exact`, <https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/runs/46021395165/job/140581645307>, failure. Published default, `1 of 212 models stepped (0 the loader refuses)`.

## Rerun

x86_64-unknown-linux-gnu only, rustc commit `0ef1bc386059596dada8fb65b50425aab2ea637a`, from the repository root:

```
cargo test --locked -p ferromotion-core --test panda_contact_reach
```

The three loop binaries:

```
cargo test --locked -p ferromotion-core --test panda_closed_loop --test panda_contact_loop --test panda_contact_reach
```

`scripts/aarch64_loop_tests.sh` is the Thor2 command. It has not been run here. An aarch64 result is not claimed.

## Scope

* measured on `x86_64-unknown-linux-gnu` only
* Thor2 `aarch64-unknown-linux-gnu` pending
* no competence claim
* no cross-arch claim
