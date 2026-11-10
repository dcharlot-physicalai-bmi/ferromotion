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

## Release diffs and commits

States are the 3.13.0 native schedule (seed 20260927) and step schedule (seed 20260928), replayed on each later library. Search commits are those that touch `src/`, `include/`, or `plugin/`. A full-corpus dump at each surviving 3.13→3.14 boundary is what the model counts below use. The 3.14.0 tag's native dump and step dump are byte-identical to the tree at `4c33a8355`.

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

Native fields that differ, state-aggregated: `efc_type` 28, `efc_pos` 42, `efc_aref` 42, `efc_J` 42, `qacc` 55, `actuator_force` 22, `qacc_smooth` 22, `solver_niter` 16. Step 1 `qpos` and `qvel` differ on 33 models. No compiled `body_inertia` change. The commit was not isolated. The incremental rebuild died while archiving `lodepng` (`Error running link command: No such file or directory`) after the 3.15.0 endpoint had already matched the release dump on the sentinels. Changelog candidates were not promoted to a result.

### 3.15.0 → 3.16.0

`ten_length` 9, `actuator_length` 9, `actuator_force` 19, `qacc` 32, `qacc_smooth` 19, `tendon_invweight0` 6, `efc_aref` 16. Step 1 `qpos` differs on 20 models and `qvel` on 22. Commit not isolated, same rebuild failure.

### 3.16.0 → 3.17.0

`m eq_data` is absent in the 3.16.0 schema and present, 12 wide, on all 212 models in 3.17.0. `include/mujoco/mjmodel.h` changes `mjNEQDATA` from 11 to 12 in `abddec5b9` (2026-10-21, "Compute the relative pose of welds in mj_setConst, as marked in eq_data[11]"). That is the schema line. Numeric fields that also differ: `qacc` 72, `efc_aref` 74, `efc_J` 74, `efc_pos` 71, `solver_niter` 13. Those numeric fields were not isolated to `abddec5b9` or to `c8362756d`. Step 1 `qpos` differs on 35 models and `qvel` on 37.

### 3.17.0 → 4.0.0

Schema lines added on all 212 models: `tendon_type`, `xiquat`, `geom_xquat`, `site_xquat`, `contact_surfacevel`. Values that differ inside the v1 field names: `xpos` 193, `xmat` 206, `ximat` 210, `M` 196, `qacc` 203, `qacc_smooth` 197, `dof_M0` 54, `ten_length` 11. Step 1 `qpos` differs on 155 models and `qvel` on 190. `dynamixel_2r/dynamixel_2r.xml` keeps the same `body_pos`, `body_quat`, `xpos`, `xmat`, and `xquat` at state 0, so this is not a dumper shift. The commit was not isolated. `a206daa42` (`mju_mat2Rot`), `a65050166` (tendon side site), `9a7f240f0` (RK4 orientation), and `3bbf510d4` (NativeCCD) are the changelog entries in this range. They were not confirmed by a dump at the parent and the commit.
