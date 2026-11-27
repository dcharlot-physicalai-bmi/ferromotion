# Panda reach contact, 2026-11-27

This episode corrects `docs/acceptance/receipts/2026-11-20-panda-contact.md` (`98ce3db15693078bc826b3b2c8eaecaae8fa3bc5`). That file stays. This one does not replace its hashes and it does not set a flag. Pending bmi-2 review. `JOINT_LOOP_EXECUTED` stays the signed contact-free scope. `EPISODE_ENVELOPE_INTEROPERABLE` stays false.

## Profile

The plant is `MjcfTree::mujoco_native_step` after `install_policy("mujoco-3.13.0-x86_64-contract-off", "64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc", "")`. That is the function `examples/menagerie_native_step` calls once `install_policy_from_env` has installed the same id. The published default is not installed and is not changed.

* profile id `mujoco-3.13.0-x86_64-contract-off`
* libmujoco sha256 `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`
* code path `2aeb3611c9f3edca1961eb25ee374df56229d799` (`MjcfTree::mujoco_native_step`)
* receipt commit `54999d65ee234f9486aca4d386568fdbe07b2fae`
* receipt file `docs/acceptance/receipts/2aeb3611c9f3edca1961eb25ee374df56229d799.supersedes-c590d3a2.md`, sha256 `d6b8600825d4ed0634df581d269216f22930aa263bd401c7a9a5d43efbbef6d7`
* that receipt: native `efc_force` / `qacc` / `qfrc_constraint` 631 of 636, step 212 of 212
* `c590d3a219638c4f247ee2eeab3085b758e56176` is superseded by `54999d65`
* exact set sha256 `caa71f582e61cf5eda1adcba7ff572da4ac4fb76596f8fd3a0f2232a1d5018bc`
* step oracle dump sha256 `a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be`
* policy line: `policy id=mujoco-3.13.0-x86_64-contract-off sha256=64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc mul_quat=Unfused rot_vec_quat=Unfused mul_mat_vec3=Contracted`

`FERROMOTION_CONTRACT_SITES` is empty. Installing the id selects unfused `mul_quat` and `rot_vec_quat`. `mul_mat_vec3` stays contracted.

## Scene

Menagerie `franka_emika_panda/panda_nohand.xml` at `4d038b3feae26ec82b46a4d586379114012a8ac7`, sha256 `e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071`. The wrapper is `crates/ferromotion-core/tests/fixtures/franka_emika_panda/contact_reach.xml`, sha256 `473d6d43acb9a67e051677149e21b5bbb4df8ee2c452070233ffb6a7680472ad`. It adds a floor plane at `z = 0`. The Menagerie XML and meshes are not edited. Mesh sha256s are the list in `docs/acceptance/receipts/2026-11-20-panda-contact-assets.sha256`.

The start pose is `qpos = 0 0.8 0 -2.2 0 2.2 0`. link6 and link7 meet the floor. link0 does not. The position actuators are fixed gain 1 and bias none, so `ctrl` is the readout torque in N·m. `forcerange` stays the file's ±87 / ±12. There is no clamp after `SpikeReadout::apply`. The test asserts each readout component's bits already equal the bits of that component clamped to those limits.

`N` is 16. The window is 4. Period `2_000_000` ns, matching the file timestep bits `3f60624dd2f1a9fc`. Ferromorphic pin `82211dc861af4ecc1afa9603c2caf20d52712b6d`.

`ncon` counts solver-facing contacts. On this pose every one of them touches link6 or link7, so `ncon_moving` equals `ncon`. `nefc` counts the `efc` rows of those contacts. `qfrc_constraint` is the native solve's joint-space constraint force.

The closed loop was run twice. Tick hashes, `ncon`, `nefc`, the plant hash, and `Sim::state_bytes` matched. The zero-torque control, from the same pose, was run twice and matched itself. Its final `qpos`/`qvel` hash differs. A third run uses the same torque loop with every geom `contype` and `conaffinity` set to 0 before the collision cache is built. Its `ncon` is 0 on every tick and its final state differs from the contacting loop.

* spikes: 45
* first spike tick: 1
* non-zero-tau ticks: 15
* envelope sha256 `170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a`
* final `Sim::state_bytes` sha256 `01e990011ad4d1e59d902d3fc1f480fca708b1cc574970de96f69fbe7a22fe82`
* contacting loop final `qpos`/`qvel` sha256 `5c4c4514c93d3c667deecb6b4f834562ff7b2ad14dc61b6c88943c2f3b443fe7`
* zero-torque final `qpos`/`qvel` sha256 `5fd57262dd51c8a559f470e7e5bc0b8a5fc2122bceb001696af4662d77fbe13c`
* contacts-disabled loop final `qpos`/`qvel` sha256 `2e3a8ce12ebd612b665b4f2d3516e1c31dae9b94f202f7bbd63c6bd50544227c`

* Cargo.lock sha256 `fb1c9967807c1121e23e6ed2575eb023f08a067cd944202e438092b06ef55ae4`
* `rustc -Vv`: rustc 1.100.0 (0ef1bc386 2026-11-09), commit-hash `0ef1bc386059596dada8fb65b50425aab2ea637a`, host `x86_64-unknown-linux-gnu`, LLVM 23.1.5
* cargo 1.100.0 (3acc79c0c 2026-11-05)

| tick | sha256 | ncon | nefc | qfrc_constraint bits, joints 1–7 |
|---|---|---|---|---|
| 0 | `52094552e834ab5371d8dba92c3c91e77245671045d8369d4877c5082f3f9572` | 5 | 20 | `4053569c20721d8c c082eeb8dbbcb652 404f8076cf1942f5 4052fef8930b658f 3fe9ed5341a9ae38 4027c076212beb68 40444736dcbb5564` |
| 1 | `fbfbc3bf6dc5043f5a756ed646a784f0b93764aa2818146ce75678733b83f8fd` | 5 | 20 | `404e761d2c3953f3 c07e0884f549f853 4048a539f5ef9456 404df56a1b9f4734 3fe270cda05fd938 402268dec7365fd0 404025bdb928c950` |
| 2 | `48ab9ded3c3388aa6a52a807d7107fa1672e02de1624a6b249c503a4e7d07002` | 5 | 20 | `4047b55349baac0e c07785edebfeff0a 40430c5d6ee5a33c 40473a2ece6a9a44 3fdb2885297ea0d0 401bbb57ec6ca714 403977987fbc98f3` |
| 3 | `570e32c982417047cd47f882eb13b9c431991529a4fce799ac7f3a1fa478fc6a` | 5 | 20 | `40422d3fc1b42395 c072236041a8bb54 403d007946df8915 4041aa342ae917c0 3fd6581a375de330 401434e8561cb89a 4033d8efdec6e9b6` |
| 4 | `72c297b61ef520d56a6b9caa6b71f1447043d296a67675c6a5123102d4b53fc6` | 4 | 16 | `403b40b3dd69df14 c06b6b5e6d08b52b 40358b600a3dc9bc 403a2fc1d524962e 3fd488b7007cdc48 400c3c51c36271f4 402e7924ade4e482` |
| 5 | `f6b60fcdd7228aa89a2245016a46ea046106b66f4a14a243af7bebef198dca28` | 3 | 12 | `4033d7a3cc4a2440 c0642fe8e5657d4e 402f05eae64e96a2 4032f23c3fe9567e 3fd22e35eb1826c0 400300d22b659916 4026d104d9ef5dde` |
| 6 | `2e2b5e30402e9601024be406cd18b9894ae565c6ccef801dd457cc1ec036dd25` | 3 | 12 | `402b73a415013777 c05c9ff3ef66ab26 4025057dda8b2a0d 402a38fca52761cf 3fcfdf572a465fd0 3ff7c882d114cd30 40208782b8c593a8` |
| 7 | `273eb522ea6643d44a8940387857f5e781483b8e25af3e8348e98c130ad6a4c6` | 3 | 12 | `40219bad879abf14 c05321f0f4085d93 401a105a141b07fc 4020c1ea69aa2036 3fcbe450f9c043c4 3fe9a4f98ac7f9b8 4016c123c635cf9b` |
| 8 | `8ea2048ef87fd16dc32c3c714151c199b02b178edfae082f5fdd017dfb855a0c` | 3 | 12 | `40133b175a1d23ac c04727937f49c146 400a1d3d8a65daf0 40136270e00d78ee 3fc717f3d2e78aaf 3fd782a9059ea2d4 400cb1dca9e58d6a` |
| 9 | `9de978190bf62a38341a26268d01306ba146b3822c1c9c0ae4d9698f0ee4c25a` | 3 | 12 | `3ffa8b6b746c164c c0365ad5ef484aba 3fe7b970ec57243b 40004cd938f50336 3fc33416b3008130 3fac032b6d78a280 3ffdfe6d12aa4274` |
| 10 | `dd3324a4db3c9696a5fc9b10cce2a8c755fbdcc5d0b972f297a26f53f27050c2` | 3 | 12 | `bfebf29fc12e54ed c00c3c795b1842e7 bff4c884f790098e bfca7faccc65b620 3fc0050d605f6809 bfc5bb14e3c7ea2e 3fe04952533207c6` |
| 11 | `7e3a005b526836e1532cacb18716ac1a83e2a3b625602ddba6ce6609ba93e441` | 3 | 12 | `0000000000000000` on all seven joints |
| 12 | `7d985ab1ecfa907a23b8497b47a68c198f049ad1072c838a2d66d48cc29c5f0a` | 3 | 12 | `0000000000000000` on all seven joints |
| 13 | `05baeb7e0e71fb37ffc0a1aababe2d067c9bdbc25a5b60e6ce0c2d4dfa917eb9` | 3 | 12 | `0000000000000000` on all seven joints |
| 14 | `0e60bc1ae887a59541a480a0886ecd5ce7fd30d9d61c184c0b370252fb3f86b4` | 3 | 12 | `0000000000000000` on all seven joints |
| 15 | `b9410c326d52ca25f70e6435a827723ef38ee3dd1ec9bc575ab3cf20689b34ac` | 3 | 12 | `0000000000000000` on all seven joints |

Ticks 0–10 have a non-zero `qfrc_constraint` on every actuated joint. Ticks 11–15 have separated far enough that the constraint force is zero while three contacts and twelve `efc` rows remain.
