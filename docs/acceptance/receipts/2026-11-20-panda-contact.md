# Panda contact episode, 2026-11-20

This is a second episode. It does not replace `docs/acceptance/receipts/2026-11-20-panda-closed-loop-2.md` and it does not change that receipt. No new flag is set. `JOINT_LOOP_EXECUTED` stays the signed contact-free scope. `EPISODE_ENVELOPE_INTEROPERABLE` stays false.

Scope: contacts. The plant is `MjcfTree::step` on a scene that includes the untouched Menagerie `franka_emika_panda/panda_nohand.xml` and a floor plane. Menagerie pin `4d038b3feae26ec82b46a4d586379114012a8ac7`. The vendored tree is `crates/ferromotion-core/tests/fixtures/franka_emika_panda/`. `panda_nohand.xml` sha256 `e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071`, the same bytes as `crates/ferromotion-core/tests/fixtures/panda_nohand.xml`. The wrapper is `contact_scene.xml`, sha256 `1666a7d0c6083aa52ac92aa48043a93abc081acc99a6e3ff5999239431bd3b6d`. It adds `<geom name="floor" type="plane" pos="0 0 0.2" size="0 0 0.05"/>`. That plane intersects the real `link0.stl` collision mesh. The Menagerie XML and meshes are not edited.

The plant compiles every mesh `panda_nohand.xml` names: `link0.stl`, `link1.stl`, `link2.stl`, `link3.stl`, `link4.stl`, `link5_collision_0.obj`, `link5_collision_1.obj`, `link5_collision_2.obj`, `link6.stl`, `link7.stl`, and the visual OBJ files named in that XML. `hand.xml`, `panda.xml`, `scene.xml`, and the hand meshes are the rest of the Menagerie directory at the same pin. They are hashed below and are not compiled by this scene.

`ncon` counts contacts whose `record.exclude` is false. It is 4 on every tick of the closed loop and of the zero-torque control.

The position servos stay general actuators with gain and the first three bias parameters set to zero after their `forcerange` is checked, same as the contact-free episode. Torque is `qfrc_applied`. There is no clamp after `SpikeReadout::apply`. The test asserts that each readout component's bits already equal the bits of that component clamped to ±87 N·m for joints 1–4 and ±12 N·m for joints 5–7.

Scale stays `5e-9` A/rad, offset stays `3e-9` A, weight stays `0.05` N·m per count. `N` is 16. The window is 4 ticks. The episode starts at the `home` keyframe.

Ferromorphic pin `82211dc861af4ecc1afa9603c2caf20d52712b6d`, no path override. The closed loop was run twice. Tick hashes, `ncon`, the plant hash, and `Sim::state_bytes` matched. The zero-torque control, from the same home keyframe and the same 16 steps, was also run twice and matched itself. Its final `qpos`/`qvel` hash differs from the closed-loop plant hash.

* spikes over the episode: 52
* first tick with a spike: 1
* ticks with non-zero tau: 15
* `ncon` on ticks 0–15, closed loop and control: 4
* max `|tau|` by joint, N·m, and bits: joint1 `0.2` `3fc999999999999a`, joint2 `0.2` `3fc999999999999a`, joint3 `0.2` `3fc999999999999a`, joint4 `0.1` `3fb999999999999a`, joint5 `0.1` `3fb999999999999a`, joint6 `0.15000000000000002` `3fc3333333333334`, joint7 `0.2` `3fc999999999999a`

Oracle profile the plant runs: `MjcfTree::step`, the stepper `scripts/menagerie_conformance.py` compares to MuJoCo 3.13.0, the Linux x86_64 wheel `libmujoco.so.3.13.0`. Ledger pins for Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`: step oracle sha256 `6acb55b5ca91040397afb4c29733571a21323876a140aff8731fb8cee49428a4`, native oracle sha256 `b1126e028df2bd4d0690f43fd67c5ced9147f8179f4b6ab70c5ba81a5cf04b11`. The solver is each model's own MuJoCo option block. 212 is the number of `*/*.xml` files that wheel loads at this pin. The checked-in ledger records `passed: false` and 1 of 212 models step-exact. The sentence "212 of 212" is the unverified author claim at ferromotion `5daca8aac1b513a18c54792383d021cf1d27af56`. This episode does not re-run that oracle and does not claim 212 of 212.

* Cargo.lock sha256 `7a0ddbf92a1233fcc17f57ca5e5ae2f060096c33a60a4b94082779219b2653e9`
* `rustc -Vv`: rustc 1.100.0 (0ef1bc386 2026-11-09), commit-hash `0ef1bc386059596dada8fb65b50425aab2ea637a`, host `x86_64-unknown-linux-gnu`, LLVM 23.1.5
* cargo 1.100.0 (3acc79c0c 2026-11-05)
* log `docs/acceptance/receipts/2026-11-20-panda-contact.log`
* log sha256 `93b022b417cfb157525d137246d5f31086a5ce49ba92ba346a2ad30661368e25`
* asset list `docs/acceptance/receipts/2026-11-20-panda-contact-assets.sha256`
* asset list sha256 `6c3346a6747c5dc6c12492f216c0744f585fff5298e23954191404702dcd7f6e`
* envelope text sha256 `170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a`
* final `Sim::state_bytes` sha256 `b2bfb4081bcd7e2de05fcd249c65b2c5a5cfe7817850eac854467df32ea3a126`
* closed-loop final `qpos`/`qvel` sha256 `9807515212ffc7c673100e9aa586f66d789504b052178b93b628a5883b131535`
* zero-torque control final `qpos`/`qvel` sha256 `455d169e610ce94c850b3609a020bf3bc1d7a0c01abf28ff3789637fb15794c8`

| tick | sha256 | ncon |
|---|---|---|
| 0 | `0aa21599ee8eeefd21101508ce69279a1fa994f04662d264decdc16bcd373574` | 4 |
| 1 | `968d22bd5f41d0d9804f179c71d94b8d4861069812e9d2867982f07ce2868f1e` | 4 |
| 2 | `3c99a3a300e5db8d4d7277d3055fb2be9c9b611ec96d05422d08c3345e55d19e` | 4 |
| 3 | `fd5e15f960c8ba945bfcfe0943b5e9363d53bb4c4b1498b970d538e77c126fd6` | 4 |
| 4 | `ba24a66fa952ee5c8e210b4c4b6c3851f83c03bc5a08cdfb9282cb10ac5ea630` | 4 |
| 5 | `db5beec23e53e50b4738a4200de9d11b9a66a7996a53351812415230453ca478` | 4 |
| 6 | `13354c71f3d9763a582557c373fb37ec56f4c2d473e717ac5f908c7ace7dc726` | 4 |
| 7 | `5a532ec8bc72a70cecefec348c5a3f40cf171858c6b46848e3b2699a0c50eed5` | 4 |
| 8 | `38a21f3a5afa90844cbd478ac339aae73ce332ea14f69316b5d60ad9b8a7067f` | 4 |
| 9 | `a4b573fc3916b708ab21e04462b62226f4a138d8f70beeee2f0a2e637addcddc` | 4 |
| 10 | `372d1ad4cf26e11a16618fee7560bf0a3605967151a18a16eee2ffca9067bcfb` | 4 |
| 11 | `da38b138a4973228b45a8a8492b6341ec0bfb2fc4ead5b006f7418a48c9cc620` | 4 |
| 12 | `4577dcd998b2c5a574af4caadc9b51a65d36157fde48e956410bd79c1aed2768` | 4 |
| 13 | `e6c6a445a3d7f90fcb154f6138667553b1a00fe39261449f49dda00f23a3d360` | 4 |
| 14 | `7e22d7e476354cbac6bb21a1abad974097f82906425ec30ba4963352780521a0` | 4 |
| 15 | `11e43eeca393545d0cc2e9bd27bf774c1966124a53fc4ce47c79ef5fbc6d43fd` | 4 |

Asset sha256, path relative to `crates/ferromotion-core/tests/fixtures/franka_emika_panda/`. The asset-list file is the same digests in `sha256sum` form, each path prefixed with `./`.

* `215098c0fcdf7192f6c503a37cf49e3efecf8a9e1320f02b17c5584d89cc5b2c` `assets/finger_0.obj`
* `cf239e9aca26bba09dbbf1c544c47d87cfc9a2f0637c8bb786af750b6d4eca44` `assets/finger_1.obj`
* `b94aab901086f85df04c019b30f43bf9aa23a7ad0443ec80a8c14662a41cabd2` `assets/hand_0.obj`
* `cd76f7494bb314406a1819c46eb826ee254894f87235714f604a470653d9badb` `assets/hand_1.obj`
* `e0f44a71712afa5b98eb20355eff17763d3e2e19f459b2bef19abed3727e64b3` `assets/hand_2.obj`
* `a431a06d99e88a1b322b3122a01424ee7704daeaad0bc343b90b9604f014ad0c` `assets/hand_3.obj`
* `21f96ef6a8de082e41b1ea9834f1043de359e5d60305541bcb63539581f53bff` `assets/hand_4.obj`
* `94493e94f30fe940f2c8ca2f155c3bbe67bbff406d3edf5e261670d2f0f6e2ed` `assets/hand.stl`
* `6b9f41a4c203540daaeb7a33cc3d0d6c6a32dc8bba187074eeaf97b3e36b3f9c` `assets/link0_0.obj`
* `c9198b9d8534d628d097b0f2b239bb30343353182df098ce0d9db6bc13326848` `assets/link0_10.obj`
* `f0adec78d6677b7799d25121981270e8bfe119b5c18d08a91ee37a54ad75f65c` `assets/link0_11.obj`
* `57673d2ee2a9db2e10c8986ae57c56a7c39c012b60fd26d7d8549553b664a5ed` `assets/link0_1.obj`
* `1ca435a2a87a9608a146a47955724d43dd7ce10fc3dd3018800aa98f8ba86e05` `assets/link0_2.obj`
* `3e9c6334d2b3b47944af3458ee5f20519abc20fe7ee23063bb53405ccab28046` `assets/link0_3.obj`
* `215d25789704936120f119c0bc1cc7f68516dd3f8f1df61457b71acac3873f26` `assets/link0_4.obj`
* `9361b104e93743045107aa5b15fb1a0835ba9c1f355aae71f56403cf6a9c33f6` `assets/link0_5.obj`
* `2dc0b8ebbbc80e4c065d71e195adbc384f673438641fac82b07b801b4d2837cb` `assets/link0_7.obj`
* `3b8c436999db5586b8eb47bed2b0aa7de6993868b2681b4c04246eeac42fdb7a` `assets/link0_8.obj`
* `45178f97d2b5da6bce11f610212c4ef7d08c73db33a4fc38e1b7f66b93d023f5` `assets/link0_9.obj`
* `dfc6d94330de8ddb005b311bfdba9f3b8e1aa7c256b71592ee7ff32cb9a9a5aa` `assets/link0.stl`
* `51091e776b16babdbe2f3849b5c041f921d1f11dd94ac19a0f1c81abf496fbae` `assets/link1.obj`
* `e41a39a94108fcf56aacff603fc91ec80541f4c1af17b51a0de5617f5566e6d2` `assets/link1.stl`
* `5c9f8e715de52c3eb71b475de35993b3b96081e02abea9f28c414768877fc59b` `assets/link2.obj`
* `370f7605a0fae3529db169ded50f52f171024aa792d4d773bc84197301f6a039` `assets/link2.stl`
* `ce2da87f51708e11b189249d41401065dabe89a06e251185a06cef4d5ff27178` `assets/link3_0.obj`
* `2880b7753a7173f38fa87ee21da4dfc7fa7be46fbe07bac5aef9e128513cb118` `assets/link3_1.obj`
* `670cff81ae84693769ca4ae658c154cf909d81167e7c8d22b70ea977f2504efa` `assets/link3_2.obj`
* `30ede0c812efd0e0b767d134a4eae0b84e0fa407bb6325a5a555e0569b3a3ffd` `assets/link3_3.obj`
* `0a8d638b9349c6c0eefc4e888636ac4838c4b27170f18a51699321118af709c1` `assets/link3.stl`
* `443bf6d178177cf347951ecf107ccd10ca444e605e375b67304579e5f948806f` `assets/link4_0.obj`
* `ca935ec7cc8b3f0e86aa079c76b0b3bf593b1e9830bab0fa7131ac51577c74f0` `assets/link4_1.obj`
* `c89caf26d379c35bb9e191ee2d91d09ab29ecc792b48d8bc10b39e5b38222175` `assets/link4_2.obj`
* `1495d17f45bdb230cf58b0beef9da4e8c9cfe76a3d2d38da89017efe5e093136` `assets/link4_3.obj`
* `0180ebb5772ec9840cb049750cffb29a9ddc90311752a16ea34757782ef9e48d` `assets/link4.stl`
* `a3238b8f2af0f2eb66d9fe63dc5d41ac52d5d38507241b5ed10e3cb019ab3e71` `assets/link5_0.obj`
* `9d994bbf83ec241e301b5197a6846da13f2a4a0582edb2c0f76962c5549649d6` `assets/link5_1.obj`
* `83b4ffe9f772cd7fe182e98640650eebc899aaf01842656574b0416c90dd40a4` `assets/link5_2.obj`
* `3e5aa75fb13f193555c3dfb999df91d81df25006a5c3af5338297d78137c129b` `assets/link5_collision_0.obj`
* `87cd8f621071df010f774cbde789282b8463840dfe704f8bdfa2e29be0142e15` `assets/link5_collision_1.obj`
* `823ab6c8f7b1d79e0844fa071d82a03499ff68e34fddb1ba3451032c22736040` `assets/link5_collision_2.obj`
* `0504c70ebf1b284ee8faae7dcb896b8830bc672a6552015dd5dbd121fa031fb9` `assets/link6_0.obj`
* `365f26c21d10332cab8d8704645cb46930c30b3085956f2d097498c793f24db4` `assets/link6_10.obj`
* `8ae61e08b3ad064f25bbf03b22e59cc15d65883b5a7ce92711aff29df2493901` `assets/link6_11.obj`
* `f28d1bfd62b0244ffd19fb35fe9edefdf583d3c9167db944e87185d919e1d021` `assets/link6_12.obj`
* `0a55fa92dccb261d07200e6565fca4c850fd17f0c1ecf38821c0b97d51796c7d` `assets/link6_13.obj`
* `750ae3ee4781e62e3a045c662195247b03f6642b05aa93104f08258e5f6eba26` `assets/link6_14.obj`
* `5999c1c2497a004a7c4bbe034cdf13744f839d85ae7a11b1bc386c48a3e642e6` `assets/link6_15.obj`
* `551d7a8b37c53e7a152d008b4e36992e5b5abe74a4be18b57ee6319c27def09f` `assets/link6_16.obj`
* `45f7ca77695198b6d86ec2417ad4e42e6275044f0f96266699ee8e1057757ac1` `assets/link6_1.obj`
* `e54eee4d8232947ee8af65afd5a067e09dac2da6e7e585cb1dccb65496457da0` `assets/link6_2.obj`
* `d4f6a25ef32fa1fa4734146981994ee760232cb077c80fea6487d78a077c7f27` `assets/link6_3.obj`
* `be353480fe71d46f7a96f0a3cbb7e112caaf8d88368826f321a36a253c8749b2` `assets/link6_4.obj`
* `7c87664623c3e1a6cb330db273227a69626dc5e53b4caff1d4ee7911eb3340c6` `assets/link6_5.obj`
* `8fbc031e0d07b795ebf99115c75223141caaba317cab569159f3d477a7821681` `assets/link6_6.obj`
* `7dc5ebb799fe281aab27d6e84e7a4efa0a8e42da2369ec83de3ea3f206f62d73` `assets/link6_7.obj`
* `e3f77381817a341030f9dcc4c4aa7c559f80411b453fd87eb0fefecca82151bf` `assets/link6_8.obj`
* `25a56ea9b02a03b6cb5a81f31c54f54d0185e1a0b3937c37e4289d896e50215a` `assets/link6_9.obj`
* `20b768e99a0e0440b5754dcca108016434e57937cc356acd9c352ccd3cb27f77` `assets/link6.stl`
* `04586b0bde011715cfc22072ac72fe0717eb4d4817bcf665987ce6cd88eb2b93` `assets/link7_0.obj`
* `c3f72289f394c9d6749a4a327fb11f05b203c37c95032c92f28019b40ebe7e00` `assets/link7_1.obj`
* `d2faecedb304d0989040d082410378dc949dbbda23bd18c05134f08aae984db5` `assets/link7_2.obj`
* `b62d87c4d4bdb7b281ed341d347c39dad0e50bf36aed8ac11d050073701d5f70` `assets/link7_3.obj`
* `cb78ccb61fd4ad6c7bff4c4d2e6cb41e1964a5a3e1a28e67c0c6202371cfc018` `assets/link7_4.obj`
* `099071d3090b51471ba52f4f90814fd3089f4975d196e4d488117073d0bfe491` `assets/link7_5.obj`
* `99469e2d95ab74ce0292f2f16954c87a8a7dc16652e62e970fe37c94330a81bb` `assets/link7_6.obj`
* `426c3db8a19373f3dc73d4bec672e948121b2914c10e2e7b9c7ea80643a19e54` `assets/link7_7.obj`
* `92ac6afcf7574c034d3170d8a68e95ac9048ab9d0dd5bbd8311b86e551b9ab1c` `assets/link7.stl`
* `1666a7d0c6083aa52ac92aa48043a93abc081acc99a6e3ff5999239431bd3b6d` `contact_scene.xml`
* `d2a6539d0fd7c55c75c14f11c78f04505ee33f31d93e6112e3ad7120e5ee786a` `hand.xml`
* `e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071` `panda_nohand.xml`
* `96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273` `panda.xml`
* `4b08c2e5108f0ed67fa689e096e58850210130c3e6ef889b1e22822d369fd8c3` `scene.xml`
