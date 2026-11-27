# Panda reach contact, revision 2, 2026-11-27

This file supersedes `docs/acceptance/receipts/2026-11-27-panda-contact-reach.md`.

* superseded file sha256 `77b4cac62fa2bfb15c9b6aba5faef2fb89789dba88f2884f7495ec4fae7e6936`
* that file was reviewed by bmi-2 at `330e631be8fcfebbe3cb46fbd216239933a8a692`
* that file is not modified

`docs/acceptance/receipts/2026-11-20-panda-contact.md` (`98ce3db15693078bc826b3b2c8eaecaae8fa3bc5`) stays. This episode does not set a flag. `JOINT_LOOP_EXECUTED` stays the signed contact-free scope. `EPISODE_ENVELOPE_INTEROPERABLE` stays false.

## Profile

The plant is `MjcfTree::mujoco_native_step_witness_applied` after `install_policy("mujoco-3.13.0-x86_64-contract-off", "64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc", "")`. `ctrl` is seven zeros. The readout torque is `qfrc_applied`. The published default is not installed and is not changed.

The measured code is commit `daef5faf088c5c508843215ff5a1690f6854e9bf`. oracle-profile run `46014593264` on that commit:

* contract-off conformance: `mj_step against MuJoCo 3.13.0, BIT FOR BIT at every one of 20 steps: 212 of 212 models stepped (0 the loader refuses)`
* contract-off conformance: `step exact 212 of 212; ledger 212`
* contract-off native: `mjData qacc 631 of 636`, `mjData qfrc_constraint 631 of 636`, `mjData efc_force 631 of 636`
* contract-off native: `mjModel body_inertia 212 of 212` and the other `mjModel` rows in that log `212 of 212`

* profile id `mujoco-3.13.0-x86_64-contract-off`
* libmujoco sha256 `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`
* policy line: `policy id=mujoco-3.13.0-x86_64-contract-off sha256=64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc mul_quat=Unfused rot_vec_quat=Unfused mul_mat_vec3=Contracted`

`FERROMOTION_CONTRACT_SITES` is empty.

## Scene

Menagerie `franka_emika_panda/panda_nohand.xml` at `4d038b3feae26ec82b46a4d586379114012a8ac7`, sha256 `e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071`. The wrapper is `crates/ferromotion-core/tests/fixtures/franka_emika_panda/contact_reach.xml`, sha256 `473d6d43acb9a67e051677149e21b5bbb4df8ee2c452070233ffb6a7680472ad`. It adds a floor plane at `z = 0`. The Menagerie XML and meshes are not edited. Mesh sha256s are the list in `docs/acceptance/receipts/2026-11-20-panda-contact-assets.sha256`.

The reviewed pose `0 0.8 0 -2.2 0 2.2 0` put link7 about 60.5 mm into the floor. Joint4 was bisected forty times on `[-2.2, -1.9]`, with the other joints held. A midpoint deeper than −1 mm moved the deep end. No contact, or a contact no deeper than −1 mm, moved the shallow end. The accepted value is that shallow end, `f64::from_bits(0xbfff5c4214179e64)` = `-1.9600239548543863`. The start pose is `0 0.8 0 -1.9600239548543863 0 2.2 0`.

Tick 0's minimum solver-contact distance is `f64::from_bits(0xbf50624dd2eef860)` = `-9.9999999996171901e-4` (−1.000000 mm). link7 meets the floor. link0, link5, and link6 do not.

## Torque path

The command is `qfrc_applied`, the same path as `panda_closed_loop.rs`. `gainprm[0]`, `biasprm[0]`, `biasprm[1]`, and `biasprm[2]` are set to 0 before the first step, so the position servo contributes no force. `ctrl` stays zero. `forcerange` stays the file's ±87 / ±12 and is not the path the torque takes.

`SpikeReadout::apply` still clamps to those limits. The test records `ReadoutTrace.clamp` and does not compare `output` with a second clamp of `output`.

## Clamps

Counts are per tick, for the contacting loop. Every tick is 0.

| tick | joint-limit clamps | encoder ±1e-6 A clamps | readout ±TAU_LIM clamps | actuator forcerange binds |
|---|---|---|---|---|
| 0–15 | 0 | 0 | 0 | 0 |

## Contact

`ncon` is the number of contacts passed to the solver with `exclude` false. The pairs are those contacts, in that order. Every tick 0–15 has three contacts, all `world/floor` with `link7/geom59`. link7 rides a joint, so `ncon_moving` is 3 and `nefc` is 12. There is no link5 contact and no link6 contact.

The minimum distance stays negative and increases at every tick. The contacts are separating. They have not separated.

| tick | minimum distance (m) | mm |
|---|---|---|
| 0 | `-9.9999999996171901e-4` | −1.000000 |
| 1 | `-9.8981854222039700e-4` | −0.989819 |
| 2 | `-9.7168910553932508e-4` | −0.971689 |
| 3 | `-9.4746120596916894e-4` | −0.947461 |
| 4 | `-9.1866289381539734e-4` | −0.918663 |
| 5 | `-8.8655341886234584e-4` | −0.886553 |
| 6 | `-8.5216442333578557e-4` | −0.852164 |
| 7 | `-8.1634006958169331e-4` | −0.816340 |
| 8 | `-7.7976533446756646e-4` | −0.779765 |
| 9 | `-7.4299128608720094e-4` | −0.742991 |
| 10 | `-7.0645859135078093e-4` | −0.706459 |
| 11 | `-6.7051414415506688e-4` | −0.670514 |
| 12 | `-6.3542667253686791e-4` | −0.635427 |
| 13 | `-6.0140012617813932e-4` | −0.601400 |
| 14 | `-5.6858586080948265e-4` | −0.568586 |
| 15 | `-5.3708709745953209e-4` | −0.537087 |

`qfrc_constraint` is non-zero on joints 1–7 at every tick 0–15. The non-zero readout-torque ticks are 1–15. Tick 0's torque is zero.

`N` is 16. The window is 4. Period `2_000_000` ns, matching the file timestep bits `3f60624dd2f1a9fc`. Ferromorphic pin `82211dc861af4ecc1afa9603c2caf20d52712b6d`.

The closed loop was run twice. Tick hashes and the plant hash matched. The zero-torque control, from the same pose, was run twice and matched itself. Its final `qpos`/`qvel` hash differs. A third run uses the same torque loop with every geom `contype` and `conaffinity` set to 0 before the collision cache is built. Its `ncon` is 0 on every tick and its final state differs from the contacting loop.

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

| tick | sha256 | ncon | nefc | qfrc_constraint bits, joints 1–7 |
|---|---|---|---|---|
| 0 | `59e1f6b5238ea02b4f521bfb804f430a403e6885e7545df393200bc21d0f343e` | 3 | 12 | `4016b88c610fe5e1 c0467a6ba75568df 40113a62595a8fc2 402791e3e09c6a9c bfd40d2c037c4152 4003c8f36a8f64d9 40001801706ac019` |
| 1 | `edeaba3a38f36e7d5623be2dd17a815814c9bad11dae64dd055800776df24a11` | 3 | 12 | `401574fb6a9a0a6f c0456ff86b6fe7d9 4010320f1ce3026b 4026a01c6f695a62 bfd35e8f7001d91d 400316bb6499f24a 3ffeae5e4b8a8fb9` |
| 2 | `79d7fb2c1b08465a658616c672ad720005f535c8a9ccadda08369b4fef7d30ea` | 3 | 12 | `4014692f06b38fbe c0449e145e0b582b 400ea6503c550a83 4025f373d6e13d99 bfd2fa38b1c23f7d 4002a57b8dfcde8a 3ffd76e23b7f8cef` |
| 3 | `6bcb14857a948b97d4bb1d6b9b7f8b47ca6df0d294a4e452461845cb904b5fcb` | 3 | 12 | `4013b0cba960c4a9 c043e7a1585dd242 400d8c609ff6ee08 402567c2e2eab676 bfd2d75f7db0f85e 40025337be8d9b32 3ffc46f1bd64ac0e` |
| 4 | `5f2e3460b0b0013be644d17c907c74512814fe44d19bcd9279e971135dfa2f16` | 3 | 12 | `4012bb7179cf4426 c04350516404fbf1 400bd627742bed0f 4024da312ba187ef bfd24a74a3189551 4001e9ea3a8bfca6 3ffb941362cd2085` |
| 5 | `b0838e7e582dd97a02d9b41b3df4bfad062ebe46ad0db01c19d8a69f9d39ac3f` | 3 | 12 | `40124f51026e0c0a c042d8967b367ce5 400b376e99ab827f 40249b3dfa82cdc4 bfd282a78a0b2200 4001e04e8b6087ef 3ffaaf2dcc9d7869` |
| 6 | `2536a08458cba8cd9d09110bf86d71cfe89f9467bb08b465b659d43428735b18` | 3 | 12 | `401202d85387481f c0426eb9c872ddbf 400ad4335da89b8b 40243ffe2811ff60 bfd268c462a51574 4001a426ad55dafc 3ff9fc953be94fdf` |
| 7 | `fa816e4ddb81770d6f854527fa5c6f0e862a25347cff9ade8f9570821ae95705` | 3 | 12 | `4011381198fa6821 c042290f6bc7eb40 40094160c7ea0d87 4023e77b140c1c06 bfd1c0c8c049cc72 400154786ba632c3 3ff9f3c1a728192c` |
| 8 | `0c98a2b6f5d653bc93eeb9ad30e7974e2ac635ebd1047dce8b67add333bc6966` | 3 | 12 | `401111ebf4e7211e c041f01ceab88f31 40090f5f0db4a669 4023d946c3d2a02c bfd20cc9d1466627 40016abe6eb89a5d 3ff9777348355d39` |
| 9 | `fcec610f7c47c3cb6e69d16a1097ef9d4b56e774566692c25b6c6d63ededb2b7` | 3 | 12 | `401083b229d630a0 c041bd9dc03bd34b 4007f34d68aa57e0 40239aac47020fa8 bfd19b4747e4d0ea 400135c9816c0b04 3ff97101196d90ff` |
| 10 | `337a1024d6715899539a80a4851d5849d474e3b40a8236632e02b894690a0c56` | 3 | 12 | `400fa87110ccb7b6 c04198e836f3cf06 400683ea287a0fd2 402376675fc55282 bfd1397cbb3ac182 40011f2b901652c2 3ff98d056eb652a5` |
| 11 | `b212a05de02aad23550cd4395b71d3653d25278edb5957380e03c7df8eb121e8` | 3 | 12 | `400ffc85bc1f4531 c0416f0d6d6290cf 4006f631bdfd4f97 4023616d79544e3e bfd180efd9aa817e 4001232d7f536268 3ff91b43d9615a61` |
| 12 | `c849c9347a37532deb40a815734d8b65c3e84123a4d5ceb6bc37da883d3ee5db` | 3 | 12 | `400f765df0277932 c041596d4c6ca666 400669a5832c41f5 40235744b16cabaa bfd17a17e237e30a 40012a5e5c0e3f7e 3ff9148be8ce1e08` |
| 13 | `905fd8c71e000c874a67854c2347a2312ca2a552fb60b8ab4c8d2f0d8294916c` | 3 | 12 | `400fa1ad485f661c c041473784504989 40069d5bd1705dcd 40233f5727d8a10b bfd17b5366a0565e 40011bade0126670 3ff8f6ea22c81de2` |
| 14 | `15d05a124f8b09827aacbc58773d6d4c3859d8b51921be2bd9d74e1866179f4a` | 3 | 12 | `400f4e061109ae28 c0413ef0e7b5d9c8 40063dc5222534f9 402354fe1696836d bfd1b402cbf121ed 40014a4c9b56f702 3ff8ee9eaac81505` |
| 15 | `b8f43856d97bfe62b2175b0bf36f4a01e989ebc1055b0a3bb2d6146c16943f62` | 3 | 12 | `400ef8f3e840d7a2 c04136c15a097570 4005dbea38a083dc 40235e65c56d643c bfd1ce4b91001cae 4001676a770c2098 3ff8f2df031d1aab` |

## CI on `daef5fa`

These are the workflows that ran on `daef5faf088c5c508843215ff5a1690f6854e9bf`. The CI workflow is red because its `test` job fails.

Workflow CI, run `46014593255`, conclusion failure:

* `test`: failure. `1013 passed; 3 failed`. The three failures are `geom_frames_match_mujocos_kinematics_bit_for_bit`, `spatial_tendons_step_as_mujoco_does_bit_for_bit`, and `the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit`.
* `integration tests`: success. 18 passed and 0 failed across the per-crate `--test '*'` binaries.
* `clippy`: success
* `docs`: success
* `wasm32`: success

Workflow oracle-profile, run `46014593264`, conclusion success:

* `contract-off conformance`: success. `step exact 212 of 212; ledger 212`
* `contract-off native`: success. `qacc` / `qfrc_constraint` / `efc_force` 631 of 636

Workflow menagerie-conformance, run `46014593265`, conclusion failure:

* `bit-exact`: failure. Published default, `1 of 212 models stepped (0 the loader refuses)`

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
