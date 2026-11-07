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

## Independent execution on this host

Ran on Ubuntu 24.04, x86_64, glibc 2.39, MuJoCo 3.13.0 (`libmujoco.so.3.13.0`), rustc 1.99.1, against Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`. This is not the prior claim.

* 217 `*/*.xml` files. MuJoCo loaded 212 and rejected 5 (listed above). The rust loader refused none of the 212.
* 20-step `mj_step`, bit-exact: **1 of 212** (`realsense_d435i/d435i.xml`). The other 211 disagree at step 1. The disagreements are small on many models (warm-start accelerations from about `1e-15` to `1e-7` on the first models inspected) and are still not bit-exact.
* Native `mj_forward`, 636 states: `efc_J` **359 of 636** (worst non-finite on `franka_emika_panda/scene.xml` state 0), `qacc` **20 of 636** (worst `3.09` on `shadow_dexee/scene.xml` state 2).

The arm64 unit-test oracles in this repo and this Linux wheel are different floating-point results. Matching one does not match the other. The conformance job fails closed on these mismatches.

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

SmolVLA (`lerobot/smolvla_base` at `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`, Apache-2.0, paper arXiv:2506.01844) is scoped as an observation → action-chunk → simulated rollout → recovery contract at the Ferric boundary. Weights were not downloaded. There is no action-output agreement, no rollout, and no claim that the model fits a browser or a device. Physical actuation is rejected.

Counterpart inputs still required, not invented here:

* bmi-3 Ferric: a pinned inference entry point that accepts the SmolVLA checkpoint, the preprocessing and normalization definition, and a sha256 of the action chunk in joint units, plus the resource bound and a joule trace only if an instrument actually recorded one.
* bmi-2 ferromorphic: the event and time contract (ordering, timestep, timestamp units) that a policy receipt must carry so a rollout can be aligned with the ferromotion step clock.

## Licenses and redistribution

Menagerie model XML and meshes are not vendored here and must not be copied into
this repository. Each model directory has its own `LICENSE`. Those terms differ
and some are not a general redistribution grant. The Menagerie README states that
model files follow the license in that directory, and that the remaining Menagerie
content is Copyright 2022 DeepMind Technologies Limited, Apache-2.0. MuJoCo itself
is Apache-2.0. The conformance job checks the models out at the pinned sha and
discards the checkout with the runner. Publishing a ledger of pass/fail and
hashes does not republish the models.
