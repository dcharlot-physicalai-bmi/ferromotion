# Versioned MuJoCo oracles

The 3.13.0 contract-off oracle is unchanged. These builds are additional named oracles. None of them is selected by `install_policy`. Ferromotion arithmetic on the 3.13.0 identity was not edited.

Tags and commits were read from `github.com/google-deepmind/mujoco` and `doc/changelog.rst` at those tags.

| tag | commit | committer date | tree |
| --- | --- | --- | --- |
| 3.13.0 | `123347c0eeab7e13c8da0828ab593bbd95bcf335` | 2026-09-09 | `c0408a44f164299fb200258a9226de55f508c59d` |
| 3.14.0 | `9ecbb9d7b5ee623f54745638d36799ff90e6f7cd` | 2026-09-22 | `cece271bb60f5b3b07273f98ce71a6675b76444b` |
| 3.15.0 | `9ea3cdfcae93bf2cc4dc0e1a1627c5a39a1e06e5` | 2026-10-05 | `9dc96300a8b94fccb96f94c3c79aca6edcb0d80a` |
| 3.16.0 | `a892750516336d8358949f4518fcc4fbbd85beae` | 2026-10-14 | `ad5b62f51a8ad9ed87439de2a87bbc51ba5f9159` |
| 3.17.0 | `800a109a4bdd02cb60404d565db4ae16fb04a53b` | 2026-10-22 | `dea445731888479258ad1c34dcfbcab9ef75a832` |
| 4.0.0 | `ac60eae30fa7e0d31315c38d9476752fb81be67e` | 2026-11-06 | `6008e06cb48f7836818364cb808fe033e72d88e6` |

4.0.0 is the changelog commit dated 2026-11-06. The claimed prefix `ac60eae3` is this commit.

## x86_64 contract-off libraries

Recipe, same as `scripts/build_mujoco_3130_x86_contract_off.sh`: `/usr/lib/llvm-18/bin/clang` 18.1.3, `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx`, `LDFLAGS=-fuse-ld=gold`, AVX and LTO off. Source layout is `/tmp/mujoco-<version>-src` and the build directory recorded in `docs/acceptance/oracles/manifest.json`. The 3.13.0 library hash changes with the source path (`64156abe…` at `/tmp/mujoco-3.13.0-src`, the other recorded value `64c5451f…`). This host reproduced `64156abe…`.

| id | libmujoco sha256 | actuator | elasticity | sensor | sdf |
| --- | --- | --- | --- | --- | --- |
| `mujoco-3.13.0-x86_64-contract-off` | `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc` | `d4e8a8c9f09cc48ea0d6c93e173f5ac84df6010108a9d1c05b14c2f9e3b19e6b` | `f66481f7eb39e41d693acfa592329d08f04d931cf03a77cc24ed4690032cc504` | `1c95a7186e02d390e38308fa836ba8bf7a7c0b916d428e01926d89f95ba4b0fc` | `f04e5207c5f7080ef4564f7425b8b583d5aaa24da5181fcf8ff843a1beca70b9` |
| `mujoco-3.14.0-x86_64-contract-off` | `8e9e455408b43e76929bfc199ce0be98ee38daaab8b048ed824ce47f559cc9e3` | `6fff3a3d838d21596ba3ff56cab3845b4f737ff7565d49b2b078bf25ed084f87` | `c83798351c475a49b442a6d34bc98061cbee5e3f695cfdab5f54381a276cafb1` | `723fb75f3fb0bff2c49f10eaf2da48eaa57612fb48cb16f963ebab8d5d7b0a7a` | `ad43f08655257c67749d5dfb2d917bc8ce624701a0203ea76eddf47f2ef98161` |
| `mujoco-3.15.0-x86_64-contract-off` | `bdafe2e11ea0c3044a5d4db9b40b4f1ab21aa7bf9e8484b2955a4f9a22480ca4` | `9bd90583de7781a33d15b1be3177c0be172f30e0da5024496ef754a2e2c2720e` | `e874f1fa1c8489c91e79c7804d920b00d4a91b0f3c196fa64302eb0d23c64170` | `c95b308f27358146c6a9dd880f98ae379a3cdabfbb9aad7a29d56b8e40ccbcb0` | `e5b7b467d78e0c8e7b6db1d61b0bea36546c59aa1d2daeb020fa9c1b70542dbb` |
| `mujoco-3.16.0-x86_64-contract-off` | `14511e7c311e7af755f67586450faafda04c0a57809d4df58bb36790fae5e1c6` | `1b2b1a3a74f3f8102493994d81ac66f9fa503bfd6ecbcf0d129c7cadcceb75b7` | `0975f62a301db83266866c353b9fd15590c1661bde1d9847b7d6e3e59bb50554` | `318bdb16c3c05333f84f713c843557399528f1d7613b84ceec2e550eb6f04b70` | `1777cde99c57ef4af5c4a42b9995df789071fc4ada89e07df3594c77d71f28ac` |
| `mujoco-3.17.0-x86_64-contract-off` | `5932d34c5c8cca5b862e4ae7234d915b6f99ad546fdf0011010b860a67db874a` | `a058052445f00bd7ec2d20f2dac50ef997b75ff971fcca5553af9bcfc4ae1198` | `a0627566098e2debec22b84adfdd11d2ee8957869dae596ad840755113c44b3e` | `c972f36f7038bcf6f5b6b8c60012678d38026f930e0dc97797396859ac405fa7` | `f0d155e80e03d8b6542f6add853fbcba15156ecff18cf50e178fc9b671860306` |
| `mujoco-4.0.0-x86_64-contract-off` | `4849b2dd05ce3c57d75098ca47a10373866e048fbdf9b57cb7073c0be2210a86` | `b28d650cac564d472122f28a1276210cc02ddaea217fb6d3ebfd4223e1f5dd3e` | removed | `5153cfe3a20015977850c93d10f2a0e23568c1ffc297ee56582c470f74c2b54f` | `3af541e2317bc882e5e4743371b3e23bd64e610adf66fa0dd0e164314fc20e31` |

4.0.0 plugins are in `lib/mujoco_plugin/`. The elasticity plugin is not in that tree. A second 3.15.0 compile from `/tmp/mujoco-full` produced a different library hash and the same `dynamixel_2r` inertia, `xpos`, and `qacc` bits as the table row. The hash difference is the source path.

## Aarch64

`scripts/arm_oracle/direct-pull/` mirrors the Thor2 stage scripts for 3.14.0 through 4.0.0. Each stage refuses unless the machine is Linux aarch64, the hostname is `thor2`, and `FERROMOTION_THOR2_APPROVED=1`. They were not run. x86-versus-aarch64 bitwise parity is not proved, so there is no aarch64 oracle id that counts.

`scripts/arm_oracle/4.0.0-aarch64/run.sh` is the unexecuted 4.0.0 recipe for `mujoco-4.0.0-aarch64-contract-off`. It accepts `thor2` and `thor.local`, fetches the pinned public tarballs, and writes a new run under `$HOME/ferromotion-oracle-4.0.0`. It does not read the 3.13.0 tree, `build-20261109T213225Z`, or the library `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`. On `thor.local` it holds `flock` `LOCK_EX` on `/tmp/bmi-energy.lock` for the whole run. The commands are in `COMMANDS.txt`. The script has not been run, so that id still does not count.

## Dump schema

`scripts/oracle_schema_dump.c` emits schema `ferromotion.mujoco-dump.v1` when `mjNEQDATA` is 11. That is the line layout of `scripts/mujoco_native_oracle.py` and `scripts/mujoco_native_step_oracle.py`. v2 (`mjNEQDATA` 12, 3.17.0) appends `m eq_data`. v3 (header `4000000`) also appends `m tendon_type`, `d xiquat`, `d geom_xquat`, `d site_xquat`, and `d contact_surfacevel`.

The 3.13.0 native dump is sha256 `a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7` (73420440 bytes). The step dump is sha256 `a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be` (7582957 bytes). A second native run produced the same digest. Those two digests are the previously pinned x86 contract-off schedule files. The v1 dump contains none of the v2 or v3 field names.

## Menagerie classification

Pin `4d038b3feae26ec82b46a4d586379114012a8ac7`, 217 one-level XML files. Counts compared: `nbody`, `njnt`, `ngeom`, `nq`, `nv`, `nu`, `neq`, `ntendon`, `nsensor`.

| compiler | compiles, same topology as 3.13.0 | rejected, new | topology changed | already rejected by 3.13.0 |
| --- | --- | --- | --- | --- |
| 3.14.0 | 212 | 0 | 0 | 5 |
| 3.15.0 | 212 | 0 | 0 | 5 |
| 3.16.0 | 212 | 0 | 0 | 5 |
| 3.17.0 | 212 | 0 | 0 | 5 |
| 4.0.0 | 212 | 0 | 0 | 5 |

The five failures are the same files, and the 4.0.0 compiler returns the same error bytes as 3.13.0:

* `aloha/filtered_cartesian_actuators.xml`: `Error: transmission target 'left/gripper' not found in actuator 0`
* `aloha/joint_position_actuators.xml`: `XML Error: unknown default class name 'waist'`
* `aloha/keyframe_ctrl.xml`: `Error: keyframe 'neutral_pose': invalid qpos size, expected 0, got 16`
* `aloha/keyframe_no_act.xml`: the same keyframe error
* `shadow_hand/keyframes.xml`: `Error: keyframe 'scissors': invalid qpos size, expected 0, got 24`

## PID

`plugin/actuator/pid.cc` is byte-identical from 3.13.0 through 3.17.0 (sha256 `9cb0c2d89cf2e1e30a0fa05e012e2e4888ad5c734cbfee9eca2f6b16738015bf`). It does not call `mj_readCtrl`. The 4.0.0 file indexes controls and outputs through `actuator_ctrladr` and `actuator_outadr`.

The ferromotion contract-off fixture (explicit `kp`/`ki`/`kd` on `mujoco.pid`, ten steps) produces the same bits on 3.13.0, 3.14.0, 3.16.0, 3.17.0, and 4.0.0:

* `qpos` `3fc0eb6ec5b1ea25 bfcdc7309e13383c`
* `qvel` `bfdd5ed456588f98 4014f1cceb4c77f3`
* `act` `3fbd71ae90c1ab78 3feb851eb851eb85`

Those are the bits pinned by `a_pid_plugin_steps_as_the_contract_off_oracle_does_bit_for_bit`. `ab7f005c9` does not modify `pid.cc`. It is in the 3.16.0 changelog: a builtin `<pid>` shortcut with no `kp` gets gain 1, stored as `biasprm[1] = -kp`. Measured `biasprm[1]` is 0 on 3.13.0, 3.14.0, and 3.15.0, and `bff0000000000000` (−1) on 3.16.0 and 4.0.0. The 3.14.0 `mj_readCtrl` change does not move this plugin trajectory. Neither upstream change alters the ferromotion pid tests, including on 4.0.0 for this fixture.

## 3.13.0 non-regression

`cargo` was rustc 1.99.1, the CI toolchain. Policy `mujoco-3.13.0-x86_64-contract-off` / `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`. No physics source was edited.

* 20-step: 6 of 212
* `M` 636 of 636, `qLD` 636 of 636, `xpos` 636 of 636, `qfrc_bias` 636 of 636
* `qacc` 148 of 636
* `qacc_smooth` 575 of 636
* `efc_J` 382 of 636
* 212 models compared, 0 refused

`.github/workflows/oracle-313-profile.yml` rebuilds that oracle on this branch and fails unless those counts are unchanged. It does not select a later oracle.

## Incremental rebuild

Clang 18.1.3 on this image has no `llvm-ar` beside it. An incremental CMake reconfigure recorded `CMAKE_C_COMPILER_AR-NOTFOUND`, and `cmake_link_script` then failed with `Error running link command: No such file or directory` while archiving `liblodepng.a` (the same failure hits `libccd.a`, `libqhullstatic_r.a`, and `libtinyxml2.a`). The first 3.13.0 configure had used `/usr/bin/ar`. `scripts/build_mujoco_x86_contract_off.sh` and `scripts/oracle_commit_build.py` pass `-DCMAKE_C_COMPILER_AR:FILEPATH=/usr/bin/ar` and the C++ and ranlib counterparts. Commit searches use a clean worktree and build directory per commit, so a later configure cannot rewrite an earlier link script. A link script that still contains `NOTFOUND` is rejected.

## Release diffs and commits

States are the 3.13.0 native schedule (seed 20260927) and step schedule (seed 20260928), replayed on each later library. Search commits are the first-parent commits that touch `src/`, `include/`, or `plugin/`. A full rev-list pulls side-branch trees from months earlier; one of those, between 3.14 and 3.15, still has no `m->nout`, and the dumper does not compile there. Counts below are parent-versus-commit dumps of the whole 212-model corpus. A model count is the number of models whose field text differs in any native state, or at that step. The union of those model sets equals the tag-versus-tag set for every numeric field in the range. Overlap is expected: one model can move at more than one commit. The 3.14.0 tag's native dump and step dump are byte-identical to the tree at `4c33a8355`.

### 3.13.0 → 3.14.0

| commit | field | models | mechanism |
| --- | --- | --- | --- |
| `73e96b9dfaa76ea5faa130ef6a2286e159f21fa8` | contact rows (`efc_*`, `efc_J`, `qacc`, `solver_niter`, `efc_AR`) | 34 for `qacc`/`efc_aref`/`efc_J`; 17 for `efc_type` | Pick the best faces in multicontact. |
| `066092b4b8e9cbe88a57812187071e01a9367ed4` | the same contact rows | the same counts, back toward 3.13.0 | Revert picking best normals in `alignedFaces`. Net of these two commits against 3.13.0 is `qfrc_damper` on `tetheria_aero_hand_open` left, right, and `scene_right` only. |
| `3ae1553ec18c5a058e223fe284fa05e921f792b5` | `body_inertia`, `body_iquat` | 104 | `mju_eig3` / `mjuu_eig3` converge to machine precision; broadphase stops using the transposed eigenvectors. |
| `3ae1553ec18c5a058e223fe284fa05e921f792b5` | `dof_M0`, `body_invweight0`, `dof_invweight0`, `M`, `qLD`, `qLDiagInv`, `qacc_smooth`, `ximat`, `cinert`, `crb` | 81 to 104 | Same eigendecomposition, downstream of the inertia frame. |
| `3ae1553ec18c5a058e223fe284fa05e921f792b5` | `qacc` | 140 | Same commit. 16 of the state-0 `qacc` models are not in the inertia set; their contacts move with the broadphase frame fix. |
| `3ae1553ec18c5a058e223fe284fa05e921f792b5` | `body_mass` | 2 (`arx_l5/arx_l5.xml`, `arx_l5/scene.xml`) | Mass shifts with the recomputed mesh inertia. |
| `3ae1553ec18c5a058e223fe284fa05e921f792b5` | `tendon_invweight0` | 24 | Tendon inverse weight is recomputed from the new inertia. |
| `4c33a83552c8af40d322074f9fae302f8cb87fc4` | `body_inertia`, `body_iquat` | 44, a further change after `3ae1553ec` | Mesh inertia eigendecomposition uses a float32 relative tolerance of 1e-7. |
| `4c33a83552c8af40d322074f9fae302f8cb87fc4` | `qacc` | 102, further change | Same mesh-inertia tolerance, through `M` and the contact solve. |

Step `qpos` / `qvel` / `qacc_warmstart` at the 3.14.0 tag match `4c33a8355` on all 212 models and all 20 steps. Against 3.13.0, step 1 `qpos` differs on 106 models and `qvel` on 111. Those trajectories are the integration of the `qacc` changes above.

#### 3.14.0 → 3.15.0

Tag versus tag: `qacc` 55, `efc_J` 42, `efc_pos` 42, `actuator_force` 22, `qacc_smooth` 22, `solver_niter` 16. Step 1 `qpos` and `qvel` differ on 33 models. No compiled `body_inertia` change. Four commits, and every native field and every step field is inside their union.

| commit | fields | models | mechanism |
| --- | --- | --- | --- |
| `7419720426d1037c5e984b3dc22a9aa464a051af` | `qacc`, `efc_J`, `efc_aref`, `efc_pos`, `efc_force` | 34 | Multicontact keeps the best face instead of the first face inside `mjFACE_TOL`. Step 1 `qpos` 9. |
| `b47e70849e9ca488dc0f8a80dc4310dbea472daf` | `qacc`, `efc_J`, `efc_pos` | 8 | Polygon winding follows a negative mesh scale. Step 1 `qpos` 2. |
| `6224e95f6264b99790341f505f8528ece2b866ac` | `efc_J`, `efc_pos`, `efc_aref` | 8; `qacc` 7 | Nativeccd multicontact for capsules. Step 1 `qpos` 3. |
| `b3dd9e6172a4d5c2bebe523efb43921939e29eb0` | `actuator_force`, `qfrc_actuator`, `qacc_smooth`, `qacc` | 22 | `dampratio` mass is `1 / (J inv(M) J')`, not `dof_M0 / transmission²`. Step 1 `qpos` 23. |

### 3.15.0 → 3.16.0

Tag versus tag: `qacc` 32, `actuator_force` 19, `qacc_smooth` 19, `ten_length` 9, `tendon_invweight0` 6, `efc_aref` 16. Step 1 `qpos` differs on 20 models and `qvel` on 22. Six commits. `qacc` on `boston_dynamics_spot/spot_arm.xml` and `scene_arm.xml` differs only at native state 2, so a search that watches states 0 and 1 does not see it.

| commit | fields | models | mechanism |
| --- | --- | --- | --- |
| `3bc4b769af3f1cbfea1b3bc36cca4f19991aca13` | `ten_length`, `actuator_length` | 9; `tendon_invweight0` 6; `qacc` 9; `actuator_force` 9 | Sphere and cylinder wrap: `atan2` arc length, intersection test from the tangent separation, Newton angle `phi = asin(z)`. Step 1 `qpos` 6. |
| `9fd6ee025fede513a93007722dd22faa4a814557` | `qacc`, `efc_J` | 2 | An element quaternion replaces the orientation it inherited from its default class. No step change. |
| `ed357ff8c887dd92caeb46829ea2b1c4d8092f68` | `qacc`, `efc_J` | 8 | Distance queries below `ccd_tolerance` keep the GJK witnesses. Step 1 `qpos` 2. |
| `ab7f005c91fc8798a05d18a9a0a6664f37ebbef1` | `actuator_force`, `qacc_smooth`, `qacc` | 10 | A shortcut inherits gain and bias from a default written with the same shortcut or with `general`, not from another shortcut. The ten models are the Rainbow RB-Y1 position files. Step 1 `qpos` 10. The ferromotion pid fixture is unchanged. |
| `efffd3a5fb96659c320224ee20db3e17e3650218` | `qacc`, `efc_J` | 5 | Capsule-capsule `det` is the squared cross of the axes, with a relative parallel test. Step 1 `qpos` 3. |
| `dcdfe84a48ed14394df36ad17828b85a45a55eaf` | `qacc`, `efc_J` | 2 | When `polygonClip` writes no witness, the edge-contact path does not swap EPA's pair. State 2 only, on the Spot arm. Step 1 `qpos` is `robotis_omy_3m/scene.xml`. Step 20 also moves `pal_talos/talos_position.xml`. |

### 3.16.0 → 3.17.0

`m eq_data` is absent in the 3.16.0 schema and present, 12 wide, on all 212 models in 3.17.0. Numeric tag-versus-tag counts: `qacc` 72, `efc_aref` 74, `efc_J` 74, `efc_pos` 71, `solver_niter` 13. Step 1 `qpos` differs on 35 models and `qvel` on 37.

`abddec5b9193a56fe6a41833e30206b6a0c4ce61` is the schema line and nothing else on this corpus. Parent versus commit adds `m eq_data` on all 212 models and changes no numeric native field and no step field. The 72-model `qacc` change is the other five commits.

| commit | fields | models | mechanism |
| --- | --- | --- | --- |
| `b9243297550a5ead15249e8d4f99d6b07dc377fb` | `qacc`, `efc_J`, `efc_aref` | 16; `efc_pos` 13 | Segment-segment weights moved into `mjc_segWeights`. The dots are reassociated. |
| `f640345f72b773e7ec52cd9426a57335cf9c6c28` | `qacc`, `efc_J`, `efc_force` | 54; `efc_pos` 56; `solver_niter` 11 | Witness recovery on a non-parallel face projects onto the plane, then along the search direction. |
| `e15ecd842a086fbddacba3f75a2efb34bef13068` | `qacc`, `efc_J`, `efc_aref` | 4 | Multicontact keeps a clipped vertex out to `margin`, not only on the negative side of the face. |
| `9f035384db1a0f023c68fb8388e736f8ff811f73` | `qacc`, `efc_force` | 57; `efc_J` 58; `efc_pos` 57 | Polygon normals use Newell's method instead of the first three vertices. |
| `c8362756d06408a772f42f893234de1bf138d228` | `qacc`, `efc_force`, `solver_niter` | 1 (`hello_robot_stretch_3/scene.xml`) | Solver termination scale is `1 / trace(M)` of the current inertia, not `1 / (stat.meaninertia · nv)`. |

`7394a8672c3e050dbd032e406fef98ac5e59b37d` (boxQP Cholesky, single-precision accuracy) changes 0 native fields and 0 step fields on the 212 models. Refuted for this corpus.

### 3.17.0 → 4.0.0

Schema lines added on all 212 models: `tendon_type`, `xiquat`, `geom_xquat`, `site_xquat`, `contact_surfacevel`. Values that differ inside the earlier field names: `xpos` 193, `xmat` 206, `ximat` 210, `M` 196, `qacc` 203, `qacc_smooth` 197, `dof_M0` 54, `ten_length` 11. Step 1 `qpos` differs on 155 models and `qvel` on 190. `dynamixel_2r/dynamixel_2r.xml` keeps the same `body_pos`, `body_quat`, `xpos`, `xmat`, and `xquat` at state 0.

| commit | fields | models | mechanism |
| --- | --- | --- | --- |
| `3ab3f0f0c9f3472d982de29402efdd1b74a62b50` | `qacc` | 50; `efc_force` 51 | Elliptic-cone linesearch stores the tangential norm and cancels in the difference, not in `T²`. Step 1 `qpos` 23. |
| `6f3db612a7304128fe9946b66a401a077f6c4928` | `qacc` | 166; `efc_force` 153 | Dense `mju_cholSolve` backward substitution updates a contiguous row with `mju_addToScl` after the diagonal divide. Step 1 `qpos` 77. |
| `e161611dd09021337350c025a6c69d375ddcfc16` | `qacc`, `efc_aref`, `efc_J`, `efc_pos` | 20 | Flex stiffness cache key hashes every node, vertex, and element, not a stride sample. Step 1 `qpos` 12. |
| `a650501666a708920d4e2eec7cf43e7c5925203e` | `ten_length`, `actuator_length` | 9; `qacc` 8; `tendon_invweight0` 6 | `sidesite` picks the wrap by the winding of `end → side → end`, not by a dot product with the tangent midpoint. Step 1 `qpos` 6. |
| `9a7f240f0336bca1a61720f83eaf1874c92d2b14` | step `qpos`, `qvel`, `qacc_warmstart` | 2 (`bitcraze_crazyflie_2/cf2.xml`, `scene.xml`), all 20 steps | RK4 integrates ball and free orientations through `dexp⁻¹`. No native field changes. |
| `a63f59172407da77376edb6ab9583f159b8ae73c` | `qacc`, `efc_force` | 54 | Elliptic-cone Hessian and linesearch divide the tangent by `T` before the cubic term. Step 1 `qpos` 28. |
| `d9972b337c9846b4924ad32c2dae85cbbe08dd3c` | `xpos` 193, `xmat` 206, `ximat` 210, `M` 196, `qacc_smooth` 197, `qacc` 200, `dof_M0` 54, `ten_length` 11 | see counts | Scalar 4-vector sums match a 256-bit reduction: `(a+b)+(c+d)` in normalize and `mulQuat`, `(q00−q22)+(q11−q33)` in `quat2Mat`. Step 1 `qpos` 140. |
| `3bbf510d477a8ea10c67c8d6fcca808c9f5b4082` | `efc_J` | 71; `qacc` 69; `efc_aref` 65 | Nativeccd drops the legacy `gjkIntersect` fallback and hardens EPA barycentric coordinates and multicontact. Step 1 `qpos` 35. `contact_surfacevel` values differ on 18 models. |
| `aeb871ba174200a76d61e72487b8f84c788239e2` | step `qpos`, `qvel` | 36 from step 1 | Implicit actuator derivatives are skipped when the joint force is outside `actfrcrange`. No native field changes. |

`a206daa42ea982dd00214255578415459e196450` (`mju_mat2Rot` at 180 degrees) changes 0 native fields and 0 step fields. Refuted for this corpus.

Schema, separate from those numeric commits:

* `64af6504f9af66958e4147c83796b2014b2bd0d8` adds `mjModel.tendon_type`.
* `629feb74b5a1325d99bcb61aacc79a73bf408817` adds `xiquat`, `geom_xquat`, and `site_xquat`. Parent versus commit, on the v2 line layout, changes 0 native fields and 0 step fields.
* `9916cc5f7cd89c12193582f3b938153444df5f47` moves contact surface velocity onto `mjContact`.
* `3190fbbdf4938a9fa70a7805cd7f42f184d999c8` sets the version header to 4.0.0. That header is what makes the dumper emit the five new lines. The engine edit is the version string in `engine_support.c`.

## What ferromotion would have to match

`d9972b33` and `6f3db612` are implemented, and only when
`FERROMOTION_BUILD_POLICY=mujoco-4.0.0-x86_64-contract-off` (sha
`4849b2dd05ce3c57d75098ca47a10373866e048fbdf9b57cb7073c0be2210a86`).
The 3.13.0 profile and the published default keep the previous formulas.
The other rows are not implemented. Counts in the table are the
parent-versus-commit corpus counts above. The profile measurement is
the next section.

| release | commit | ferromotion change | fields and counts |
| --- | --- | --- | --- |
| 3.15 | `7419720426d1` | In `mujoco_ccd` multicontact, keep the best face inside the face tolerance, not the first. | `qacc`/`efc_J` 34; step 1 `qpos` 9 |
| 3.15 | `b47e70849e9c` | In the mesh compiler, reverse polygon winding when the scale determinant is negative. | `qacc` 8; step 1 `qpos` 2 |
| 3.15 | `6224e95f6264` | Run capsule pairs through the nativeccd multicontact path. | `efc_J` 8, `qacc` 7; step 1 `qpos` 3 |
| 3.15 | `b3dd9e6172a4` | `resolve_dampratio` in `mujoco_actuator.rs` divides by the operational-space inertia `J inv(M) J'`, not by `dof_M0 / transmission²`. | `actuator_force`/`qacc`/`qacc_smooth` 22; step 1 `qpos` 23 |
| 3.16 | `3bc4b769af3f` | `wrap_circle`, `length_circle`, and `wrap_inside` in `mjcf_tree.rs`: `atan2` arc, separation-vector intersection, Newton in the incidence angle. | `ten_length` 9, `qacc` 9, `tendon_invweight0` 6 |
| 3.16 | `9fd6ee025fed` | A quaternion on an element replaces the class orientation it inherited. | `qacc`/`efc_J` 2 |
| 3.16 | `ed357ff8c887` | A distance query below `ccd_tolerance` returns the GJK witnesses. | `qacc`/`efc_J` 8; step 1 `qpos` 2 |
| 3.16 | `ab7f005c91fc` | `ActRecord` inherits gain and bias only from the same shortcut or from `general`. The pid fixture's explicit `kp`/`ki`/`kd` stay. | `actuator_force`/`qacc` 10; step 1 `qpos` 10 |
| 3.16 | `efffd3a5fb96` | `capsule_capsule` in `mujoco_collision.rs` builds `det` from the axis cross, with a relative parallel threshold and a sequential solve. | `qacc`/`efc_J` 5; step 1 `qpos` 3 |
| 3.16 | `dcdfe84a48ed` | In `mujoco_ccd.rs`, do not swap the witness pair when `polygon_clip` writes nothing. The current port swaps EPA's pair, which is the 3.13 behavior. | `qacc` 2 at state 2; step 1 `qpos` 1; step 20 `qpos` 2 |
| 3.17 | `b9243297550a` | Segment-segment closest-point weights use the factored `mjc_segWeights` association. | `qacc`/`efc_J` 16 |
| 3.17 | `f640345f72b7` | `witness_on_face` projects onto the face plane, then along the direction. | `qacc`/`efc_J` 54, `efc_pos` 56 |
| 3.17 | `e15ecd842a08` | `polygon_clip` keeps vertices out to the margin. | `qacc`/`efc_J` 4 |
| 3.17 | `9f035384db1a` | `make_normal` in `mujoco_hull.rs` sums Newell's cross around the polygon. | `qacc` 57, `efc_J` 58 |
| 3.17 | `c8362756d064` | The Newton and noslip scales in `mujoco_solver.rs` and `mujoco_contact.rs` use `1 / trace(M)`. | `qacc` 1, `hello_robot_stretch_3/scene.xml` |
| 3.17 | `abddec5b9193a` | Emit `m eq_data` at width 12, with the weld relative pose in the last column. No other field on this corpus. | `eq_data` added on 212; numeric `qacc` 0 |
| 4.0 | `3ab3f0f0c9f3` | `elliptic_cost_dif` stores the tangential norm and differences it. | `qacc` 50; step 1 `qpos` 23 |
| 4.0 | `6f3db612a730` | Implemented for the 4.0 profile. `chol_solve` divides the diagonal, then updates the row with `add_to_scl`. The profile pin is the receipt, not this cell. | `qacc` 166, `efc_force` 153; step 1 `qpos` 77 |
| 4.0 | `e161611dd090` | Flex stiffness cache identity covers every node and element. Ferromotion refuses `<flexcomp>` today, so this waits on a flex compiler. | `qacc`/`efc_aref` 20 |
| 4.0 | `a650501666a7` | `wrap_circle` chooses the side-site homotopy by winding angle. | `ten_length` 9, `qacc` 8 |
| 4.0 | `9a7f240f0336` | RK4 maps ball and free rates through `dexp⁻¹` before the stage sum. | step only, 2 Crazyflie models |
| 4.0 | `a63f59172407` | Elliptic-cone tangent is `U/T` before the Hessian's cubic term. | `qacc` 54; step 1 `qpos` 28 |
| 4.0 | `d9972b337c98` | Implemented for the 4.0 profile. `normalize4`, `mul_quat`, and `quat2mat` pair the sums the way the 256-bit reduction does. The profile pin is the receipt, not this cell. | `xpos` 193, `xmat` 206, `M` 196, `qacc` 200, `dof_M0` 54; step 1 `qpos` 140 |
| 4.0 | `3bbf510d477a` | Nativeccd precision pass: no legacy GJK-intersect fallback, hardened EPA barycentrics and multicontact. | `efc_J` 71, `qacc` 69; `contact_surfacevel` 18 |
| 4.0 | `aeb871ba1742` | The implicit step skips an actuator derivative when `qfrc_actuator` is outside the joint `actfrcrange`. | step `qpos` 36; native fields 0 |
| 4.0 | `64af6504f`, `629feb74b`, `9916cc5f7` | Emit `tendon_type`, `xiquat`, `geom_xquat`, `site_xquat`, and `contact_surfacevel` under the 4.0 dump schema. | added on 212 |

`7394a8672c3e` and `a206daa42ea9` are not in that list. Each one left the 212-model native dump and the 20-step dump unchanged.

## 4.0 profile measurement

The 4.0.0 x86 profile pin is the receipt
`docs/acceptance/receipts/f8fe0779ca35549d590f005f2a57768b522459a6.md`.
The 4.0.1 identity is a separate oracle. Its pin is
`docs/acceptance/receipts/d9e1bc38d6e4cc9606f6b506291f7dd220b4f3d8.md`.
The library-versus-library dump diff is `docs/acceptance/oracles/4.0.1-dump-diff.md`.
`scripts/oracle_profile_gate.py` reads the latest receipt for the oracle id.
Counts and the exact model set are in that file, hashed, and are not copied here.

The 3.13 profile pin on this branch is the fidelity receipt
`docs/acceptance/receipts/b6a3a885f8c405b3f2c098ece0cc1bdc8ad5ce03.md`,
the same bytes as on `cursor/ferromotion-fidelity-conformance-bb9d`.
The later fidelity receipt `701413e` records commits that are not in this branch.
