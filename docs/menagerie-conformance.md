# Menagerie conformance

This is the record of what is compared, what is pinned, and what is not redistributed.
It is not a claim that a checked-in file passed.

## What is executed

`scripts/menagerie_conformance.py` re-runs both comparisons against a checkout of
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
at the sha in `conformance/menagerie_pin.json`:

1. Twenty `mj_step` steps from the initialization in
   `scripts/mujoco_native_step_oracle.py` (NumPy seed `20260928`). Each hinge and
   slide is drawn inside its limit, every velocity in `[-0.5, 0.5]`, every control
   inside its range, every activation in `[0, 1]`. The model's own solver options
   are left as written. `qpos`, `qvel`, `act`, and `qacc_warmstart` must match
   `f64` bit for bit after every step.
2. Three `mj_forward` states per model from `scripts/mujoco_native_oracle.py`
   (seed `20260927`): `qpos0` at rest, then two states that move every joint.
   `efc_J` and `qacc` must match bit for bit on every state.

Reference engine: MuJoCo **3.13.0**, double precision, the Linux wheel's
`libmujoco.so.3.13.0` when the job runs on `ubuntu-latest`. Equality is
`f64::to_bits`, not an absolute tolerance.

At pin `4d038b3feae26ec82b46a4d586379114012a8ac7` (2026-09-29) the tree contains
**217** `*/*.xml` files. MuJoCo 3.13.0 loads **212** of them. That 212 is the count
this script measured, and it is the same count as the earlier author claim of
"212 models" and "636 states" (212 × 3). The claim itself, made at ferromotion
`5daca8aac1b513a18c54792383d021cf1d27af56`, was not re-used as evidence. The five
files that do not load stay in the ledger as `mujoco_load: error`:

* `aloha/filtered_cartesian_actuators.xml`
* `aloha/joint_position_actuators.xml`
* `aloha/keyframe_ctrl.xml`
* `aloha/keyframe_no_act.xml`
* `shadow_hand/keyframes.xml`

They are not omitted. A sixth load failure, a missing file, or a bit mismatch
fails the process.

## Two floating-point contracts

These are not the same arithmetic, and a match to one is not a match to the other.

* **Unit-test goldens** in `ferromotion-core` are bits from the MuJoCo 3.13.0 arm64 dylib (`libmujoco.3.13.0.dylib`): Apple clang contraction and `__sincos_stret`.
* **This conformance oracle** is the Linux x86_64 wheel `libmujoco.so.3.13.0` linked to glibc 2.39 libm, with that wheel's own contraction. Non-Apple `sincos` in this crate is that host `sin`/`cos` pair. It is not `__sincos_stret`.

`copysign(sqrt(1 − sin²), cos)` when `sin² + cos² ≠ 1` reproduces one arm64 `fromto` cosine (`0x3fe6d214bbbba928` → cosine `0x3fe833ad8be0fb0b`, host cosine `0x3fe833ad8be0fb0a`) and moves `0.3°/2`, whose host pair is already Pythagorean. That is not a general port. It is not used.

## Polygon order is an oracle-identity property

`mjCMesh::MakePolygons` stores merged coplanar faces in a `std::unordered_map` keyed by the normal's angles after rounding to 0.01 rad. `alignedFaceEdge` keeps the first incident face in that map's iteration order. The order is the C++ standard library's, not a sort.

The published path follows libc++: `std::hash<double>` is the value's bits (±0 hashes to 0) and the bucket walk is power-of-two. That is the Apple dylib.

The contract-off oracle `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc` is built by `clang++` (`scripts/build_mujoco_3130_x86_contract_off.sh`), which links libstdc++. libstdc++ 14 hashes a double with `_Hash_bytes` of the IEEE bits and seed `0xc70f6907` (±0 still hashes to 0), and its `unordered_map` grows by prime bucket counts. The same `(θ, φ)` keys therefore come out in a different order. On the oracle identity the hull walks that libstdc++ order. The published identity keeps libc++.

The Thor2 aarch64 contract-off oracle links libstdc++ as well. Its polygon walk matches this x86 oracle, which is why those two dumps agree with each other and both disagree with the Apple libc++ order.

## Independent execution on this host

Ran on Ubuntu 24.04, x86_64, glibc 2.39, MuJoCo 3.13.0 (`libmujoco.so.3.13.0`), rustc 1.99.1, against Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`, after non-Apple `sincos` was host libm again. This is not the prior claim. The command was:

```
python3 scripts/menagerie_conformance.py /tmp/mujoco_menagerie --out conformance/menagerie_ledger.json
```

It exited 1. The ledger records `passed: false`.

* 217 `*/*.xml` files. MuJoCo loaded 212 and rejected 5 (listed above). The rust loader refused none of the 212.
* 20-step `mj_step`, bit-exact: **1 of 212** (`realsense_d435i/d435i.xml`). The other 211 disagree at step 1.
* Native `mj_forward`, 636 states: `efc_J` **359 of 636** (worst non-finite on `franka_emika_panda/scene.xml` state 0, a length mismatch), `qacc` **20 of 636** (worst `1.08e4` on `hello_robot_stretch_3/stretch.xml` state 1). `M` is 18 of 636. `solver_niter` is 636 of 636 on this run; matching the iteration count does not make `qacc` exact.
* `mjModel body_invweight0` is 19 of 212, worst absolute `1.31e-7` on `hello_robot_stretch_3/stretch.xml`. That array is compiled from the body's orientation, so the stretch miss starts before `mj_step`.

The conformance job fails closed unless all 212 loadable models are bit-exact for 20 steps and `efc_J` and `qacc` are bit-exact on all 636 states. `git HEAD` stored in the ledger is the parent of the commit that contains this rerun, because the sha cannot name the commit that adds the file.

## First divergent stage

Remeasured after non-Apple `sincos` was returned to host libm. The mass-matrix and tendon angles in these fixtures were already Pythagorean, so these bits are the same as under the adjustment. `geom_frames_match_mujocos_kinematics_bit_for_bit` is the case that adjustment had been covering: geom 0 (the Euler box) still matches the arm64 golden. Geom 1 is the capsule `fromto`. Its position matches. The first mismatch is `geom_xmat[0]` (`m00`): ferromotion `0x3fb10f4d4d24f06a`, golden `0x3fb10f4d4d24f05c`, +14 ulps. The other eight matrix entries differ by −1, −1, −1, +3, −9, −1, −1, and −2 ulps. Commands and raw bits:

Mass-matrix fixture (`the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit`), same `qpos`/`qvel` in both programs. `cargo +stable test -p ferromotion-core --lib the_native_mass_matrix_and_bias` and `mujoco.mj_forward` on the same XML.

* `qfrc_bias` matches the arm64 golden on all 11 entries. The Linux wheel's bias matches that golden on 3 of 11 (first mismatch index 1: wheel `0xbfe3fd29118b3571`, golden and ferromotion `0xbfe3fd29118b356e`).
* Sparse `M` versus the arm64 golden differs at 6 of 66 entries, ferromotion minus golden: index 9 by +2 ulps, 13 by +8, 14 by +1, 24 by +16, 25 by +1, 26 by +2. Index 9 is ferromotion `0x3f916b4f44bfd583`, golden `0x3f916b4f44bfd581`, Linux wheel `0x3f916b4f44bfd584`.
* Versus the Linux wheel the first differing stage is body `xpos` index 7 (arm, y) by −1 ulp (`0xbfb771f2b86a019f` vs `0xbfb771f2b86a01a0`). Then `xipos` 5/15, `cinert` 23/50, `cdof` 25/66, `crb` 29/50, `M` 52/66. Fusing `M += mji_dot6` into the last product, which would not touch the bias statement, missed the golden and broke `ten_native_steps_are_mujocos_trajectory_bit_for_bit`. The separate `dot6` then add is the contraction those passing tests use. No permutation of that dot on these `crb`/`cdof` values was found that hits the six golden entries. The arm64 `crb`/`cdof` are not available on this host, so the six ulps are not isolated past "final `M` differs, final bias does not."

Tendon fixture (`spatial_tendons_step_as_mujoco_does_bit_for_bit`), initial state `[0.3, -0.6, 0.4]`, `[0.5, -1, 0.8]`.

* Body `xpos` 0/12 differ from the Linux wheel. Geom `geom_xpos` and row-major `geom_xmat` differ in 0 of 5 geoms, including the sphere and the cylinder.
* Native site `s2` x is −1 ulp (`0x3fdaf24032cb21e2` vs wheel `0x3fdaf24032cb21e3`). `s1`, `s3`, and the sphere frame match. That site is an endpoint of both wraps.
* `ten_length[0]` (cylinder) matches the wheel (`0x3fe1ac896f321600`). `ten_length[1]` (sphere) is +2 ulps (`0x3fd8e6e424911d82` vs `0x3fd8e6e424911d80`). `cdof` and `cvel` match; `qfrc_bias[2]` does not (`0xbfdb1be7dd2e22f4` vs `0xbfdb1be7dd2e22f8`).
* After one `mj_step`, `qpos` matches the wheel and `qvel` does not (first component −3 ulps: `0x3fe66327cffb07ff` vs `0x3fe66327cffb0802`). After ten steps, `qvel` versus the arm64 golden is exact, −2 ulps, −1 ulp: `0x3ff910c1b04b0f7a` (match), `0xc01a36e2e56175fc` vs `0xc01a36e2e56175fe`, `0x4026d2702fede5ae` vs `0x4026d2702fede5af`.

`franka_emika_panda/panda.xml` at `qpos0`. Ferromotion reports 2 contacts and 10 `efc` rows (types 0, 3, and eight pyramidal rows). The wheel reports `ncon = 10`, `nefc = 42` (1 equality, 1 limit, 40 pyramidal rows from 10 contacts). The two deep pairs are the same geoms, `(29, 64)` and `(30, 64)`. The eight missing rows are fingertip pads at distance about `−6.9e-18`. Feeding the wheel's `geom_xpos`/`geom_xmat` and `xipos`/`ximat` into the same collider returns 10 contacts on those pairs. The first body `xpos` difference is `link5`, absolute error about `1e-17`, larger than the pad gap. The row-count gap is that pose difference, not a separate midphase reject.

`shadow_dexee/shadow_dexee.xml` at `qpos0`: contact count 43 and `efc` length 230 match the wheel. `solver_niter` is 9 versus 8. `qacc` differs on all 12 dofs, max abs about `1.8e-8`. `agilex_piper/piper.xml` at `qpos0`: 1 contact, 10 `efc` rows, `solver_niter` 0, and `qacc` all zero, matching the wheel. Its step-1 miss is not a `qpos0` row-count bug.

## What was not changed

Golden arrays, equality, the default solver, initialization, the corpus sha, and the 212-model goal were not changed. Warp, Newton, Ferric, and ferromorphic event contracts were not executed. No measured energy and no GPU result is claimed.

## PID trajectory tests

`a_pid_plugin_steps_as_mujoco_does_bit_for_bit` pins the ten-step PID plus servo trajectory
that commit `2eb1e3c` captured against `libactuator.dylib`'s contracted force
(`fma(kp, e, kd*edot)`). That is not `libmujoco` `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`.

`a_pid_plugin_steps_as_the_contract_off_oracle_does_bit_for_bit` pins the same model and controls
against that contract-off library (`mujoco-3.13.0-x86_64-contract-off`), captured by
`mj_loadPluginLibrary` of its `libactuator.so` and ten `mj_step`s. The dylib golden is unchanged.
Retiring it is not decided here.

## CI

Workflow `menagerie-conformance`, job `bit-exact`. GitHub reports the check as
`menagerie-conformance / bit-exact`. The job clones the pinned sha, installs
MuJoCo 3.13.0, and runs the script. It does not treat a previously uploaded
ledger as success.

Branch protection was not modified. Making the check required is a repository
administration setting, not a file in this repo: require the status check
`menagerie-conformance / bit-exact` on `main`. Until that setting exists, the
workflow runs but does not block a merge by itself.

## Model × feature × solver

The native tally is the MuJoCo 3.13.0 CPU column of that matrix: trajectories (`qpos`/`qvel` in the step ledger), mass matrix (`M`), constraints and contact (`efc_*`, `qfrc_constraint`), actuator outputs (`actuator_force`, `qfrc_actuator`), and solver iteration counts. It is not a pass. Derivatives were not run.

MuJoCo Warp `4d1c64bc077d725247e9bf5e2ac3f357749ff804` (Apache-2.0, Google DeepMind and NVIDIA) and Newton `1a497c00ed255db343e39eec4d614864c9aa9b95` (Apache-2.0 code, CC-BY-4.0 documentation) were checked as external oracles only. This host had no NVIDIA device and neither package imported. That cell is **blocked**. Warp's own README at that revision says differentiability is not yet available; a later GPU forward would still not count as a gradient test. Menagerie assets stay in the upstream checkout under each model's license.

SmolVLA (`lerobot/smolvla_base`, Apache-2.0, paper arXiv:2506.01844) stays an unexecuted model. Codec version 2 is pinned to Ferric `3ee11c015e3478c7c4aa0409499346f79ab1c8ce` in `docs/acceptance/contracts/smolvla-ferromotion.md`. The fixture digest is a protocol test. It does not authorize a rollout, and physical actuation is still refused. The entry remains proposed. Real weight sha256 values were not published.

The ferromorphic clock document `549aacdc5f7f4bd947c26877f14b8edc42febd5a` is recorded in `docs/acceptance/contracts/ferromotion-clock.md`. The integer-tick adapter does not replace `DynamicsLab`'s accumulated `f64` time, and the joint loop was not executed.

Counterpart inputs still required, not invented here:

* bmi-3 Ferric: a released sha for `ferric.smolvla.predict_action_chunk.v1` (the contract pin above is still marked proposed), the real weight, stats, and noise sha256 values, and a joule trace only if an instrument recorded one. The codec v2 golden is a protocol vector, not that release.
* bmi-2 ferromorphic: agreement on the three open clock policies (checked overflow versus wrapping `Sim::step`, whether a joint epoch exists, and whether `DynamicsLab` keeps its `f64` accumulator). The clock document pin above is not a completed joint run.

## Licenses and redistribution

Menagerie model XML and meshes are not vendored here and must not be copied into
this repository. Each model directory has its own `LICENSE`. Those terms differ
and some are not a general redistribution grant. The Menagerie README states that
model files follow the license in that directory, and that the remaining Menagerie
content is Copyright 2022 DeepMind Technologies Limited, Apache-2.0. MuJoCo itself
is Apache-2.0. The conformance job checks the models out at the pinned sha and
discards the checkout with the runner. Publishing a ledger of pass/fail and
hashes does not republish the models.
