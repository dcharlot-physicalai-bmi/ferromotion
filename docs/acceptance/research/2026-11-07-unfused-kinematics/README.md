# Archived experiment: blanket unfused kinematics

Not adopted. The published default remains contracted `mul_quat`. `conformance/menagerie_ledger.json` was not overwritten.

The patch is the uncommitted edit to `mujoco_kinematics.rs` and `mjcf_tree.rs` that replaced several `mul_add` chains with separate products. It was discarded from the working tree after this archive. No golden array was edited.

## Files

| File | sha256 |
| --- | --- |
| `diff.patch` | `024d27566a1a98a948421a00b0c0061ca8df8720bd8b5b019b8d9b73b39b9684` |
| `menagerie_ledger_linux_unfused.json` | `b19f12d251115fa5ceb98f7247c47decb393288a626fd856edd146f79fd31abf` |
| baseline `conformance/menagerie_ledger.json` | `ca4761e6a3c6cfa022d283e5888acf8e6e196a5fa3946f9fde3b691fa159a776` |

The unfused ledger records `ferromotion_sha` `aed4a76a38b67bf3b1f5b63adb2c51073e53d3bd` because that was HEAD when the run started. The binary also contained this uncommitted patch. The oracle was the Linux x86_64 wheel `libmujoco.so.3.13.0`, sha256 `dc9599faeffba8e6ea9f53928de08301ee8c2ba780c4afceadeebe63573c3a1a`.

## x86_64 counts

| | Published baseline | Blanket unfused |
| --- | --- | --- |
| 20-step bit-exact | 1/212 | 1/212 |
| mismatch | 211 | 211 |
| exact model | `realsense_d435i/d435i.xml` only | same, none gained or lost |
| `efc_J` | 359/636 | 375/636 |
| `qacc` | 20/636 | 23/636 |
| `M` | 18/636 | 22/636 |
| `qfrc_bias` | 88/636 | 176/636 |

First recorded nonzero array in the step detail (not an opcode):

| First nonzero array | Baseline | Unfused |
| --- | --- | --- |
| `qpos` | 172 | 168 |
| `qvel` | 37 | 41 |
| `qacc_warmstart` | 2 | 2 |

Failed step was 1 for 210 models and 2 for 1 model, in both ledgers.

## Arm64

Not executed. The unit-test goldens on the published tree still fail three tests on this x86 host (`geom_frames_match_mujocos_kinematics_bit_for_bit`, `spatial_tendons_step_as_mujoco_does_bit_for_bit`, `the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit`). With this patch applied, 16 of those arm64 golden tests failed. That is a regression against unproven arm64 arrays, not a fix.

## Wheel micro-check that remains true

On this artifact, `objdump` of `mju_mulQuat` shows `vmulpd` and `vaddpd` and the sampled lines have no `vfmadd`. One exported product, `mju_mulQuat([0.9, 0.1, -0.2, 0.3], [0.8, -0.1, 0.4, 0.2])`, is bits `3fe8000000000000 bfc5c28f5c28f5c2 3fc3333333333333 3fdc28f5c28f5c29`. The unfused formula matches that vector. The published contracted formula does not. That is why the unfused formula exists as a non-default policy, not why the whole kinematic tree was switched.
