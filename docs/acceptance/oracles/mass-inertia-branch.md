# Compiled body mass and inertia, contract-off oracle

The published default stays contracted. Baseline ledger and golden arrays are unchanged. The three kinematics sites are unchanged. The measured `qacc` regression, 24/636 with `mul_quat`+`rot_vec_quat` unfused and 23/636 after `mul_mat_vec3` is also unfused, was not re-measured. Mass and contact do not yet both have a verdict against that regression.

## Two artifacts, two hashes

Uninstrumented oracle `mujoco-3.13.0-x86_64-contract-off`:

* source tag `123347c0eeab7e13c8da0828ab593bbd95bcf335`
* flags `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx`, AVX and LTO off, gold link
* artifact `/tmp/oracle-x86-contract-off/build/lib/libmujoco.so.3.13.0`
* sha256 `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`

Diagnostic sibling, fprintf only, same flags, patch `docs/acceptance/oracles/mesh-trace-3.13.0.patch`, script `scripts/build_mujoco_3130_x86_contract_off_meshtrace.sh`:

* artifact `/tmp/oracle-x86-contract-off-meshtrace/build/lib/libmujoco.so.3.13.0`
* sha256 `fda4efc90705c350bfcd9db1adfa2288a5980ca82fcc6ee9b70fc0388ee4cd35`
* this hash is not the uninstrumented oracle

Before any trace was read, `body_mass` and `body_inertia` were bit-identical between the two libraries on `hello_robot_stretch_3/stretch.xml` (38 bodies), `agilex_piper/piper.xml` (10), `google_robot/robot.xml` (13), `realsense_d435i/d435i.xml` (2) and `unitree_go2/go2.xml` (14).

PyPI wheel `libmujoco.so.3.13.0` sha256 `dc9599faeffba8e6ea9f53928de08301ee8c2ba780c4afceadeebe63573c3a1a` is a third binary (clang 20.1.8 / LLD, libc++ tinyobj, no `tryParseDouble`). It is not this source build.

## What the trace showed

`MUJOCO_MESH_TRACE` on the diagnostic library, `link_SG3_gripper_body`:

* `inertia_mode 2` (legacy), scale 1, refpos 0, 14083 vertices, **23776 faces**
* volume `0.00054308312614509977` exactly
* `mjuu_eig3` returned 3, stopped on off-diagonal `4.3841380564471616e-13` at iteration 3, one sort swap (`j 1`), quaternion `0.703007795615885 0.71081095506330805 -0.016477849623581765 -0.016009557678653287`
* box `0.024618919926663811 0.0338018982697093 0.057986441467871318`, the same half-sizes the uninstrumented inertia inverts to

The wheel's `mesh_facenum` for that mesh is **23575**. Its 23575 triangles are an exact subset of the source build's 23776, same winding and same order. The extra 201 triangles are the builtin ear clip of one 215-gon. A regular n-gon confirms the fork is not a one-ulp ear test: the wheel fails to compile a regular 101-gon (volume too small, the polygon is absent) while the source build emits n−2 triangles for n = 101, 215 and 250. Only one Menagerie OBJ has a face with more than 100 vertices.

`mjuu_eig3` from the uninstrumented `user_util.o`, on the pre-fix Rust tensor, returns the wheel body inertia (`0.00043552413190669603` and companions). On the traced source tensor it returns the source body inertia and the traced quaternion. The eigenroutine is not the site.

## Fix

Rust had been dropping every OBJ face with more than 100 vertices so it would match the wheel. That drop is not what tinyobj `2945a967` does. Removing it and running the existing ear clip makes `link_SG3_gripper_body` triangle-for-triangle identical to the source build, including order: 14083 vertices, 23776 faces.

Stretch3, the only body whose mass/inertia bits changed:

| | mass | inertia |
| --- | --- | --- |
| before (matched the wheel) | 0.29 | 0.00043552413190669576, 0.0003834907121708115, 0.0001693904410921506 |
| after | 0.29 | 0.0004354829196717889, 0.00038362346587136084, 0.0001690370893491552 |
| contract-off | 0.29 | 0.00043548291967178881, 0.00038362346587136057, 0.00016903708934915525 |

Absolute inertia error went from `3.533517429953547e-7` to about `2.7e-19`. Volume of the mesh is bit-identical to the trace. The inertia tensor and CoM remain a few ulps from the trace; `mjuu_eig3` on that tensor reproduces the residual. No ulp nudge was applied. The face integral stays on the published contracted `mul_add` path. Unfusing that integral was already measured, against the old face set, to move volume by `1.084e-19` and leave the `3.53e-7` error in place. It was not retried as a default change.

## Full 212, published default, same contract-off dumps

Trajectory (`menagerie_native_step` against `/tmp/oracle-x86-contract-off-steps.txt`): **1 of 212**, unchanged. The step example prints at most 25 first misses; the summary is the count.

Native (`menagerie_native` against `/tmp/oracle-x86-native.txt`), before this fix and after:

| array | before | after |
| --- | --- | --- |
| `body_inertia` | 115/212, worst `3.53e-7` on `hello_robot_stretch_3/stretch.xml` | 115/212, worst `2.27e-13` on `google_barkour_v0/scene_barkour.xml` |
| `body_mass` | 192/212, worst `4.44e-16` on `google_robot/scene.xml` | 192/212, same worst |
| `efc_J` | 359/636 | 359/636 |
| `qacc` | 20/636 | 20/636 |
| `M` | 18/636, worst `5.63e-5` on stretch state 1 | 18/636, worst `4.26e-14` on `pal_tiago/scene_motor.xml` state 1 |
| `qacc_smooth` | 23/636, worst `3.42e0` on stretch state 2 | 23/636, worst `1.33e-6` on stretch state 2 |
| `body_invweight0` | 19/212, worst `5.28e-1` on stretch | 19/212, worst `7.50e-8` on stretch |

Stretch is still a `body_inertia` first-stage miss. The residual is the few ulps above, so the match count did not move. The `3.53e-7` error is gone. `qacc` on the published default is still 20/636. The opt-in matvec regression, 24/636 with `mul_quat`+`rot_vec_quat` unfused versus 23/636 after `mul_mat_vec3`, was not re-run.

## Contact rows

Not changed. On the published default, `agilex_piper/piper.xml` state 0 is `efc_J` **80 vs 80** both before and after this fix. The first mismatch is a value at index 70 (`-7.393718467647721e-7` versus `-7.393718468757946e-7`), not a missing row. The 56-versus-80 length (three rows, one condim-3 contact) is in the `mul_quat` and `mul_quat`+`rot_vec_quat` unfused sweeps. The all-three unfused sweep is 80 vs 80 again. That length gap is not a detection-margin change to make on the published default, and it was not touched here.

`normalize4` is still later.
