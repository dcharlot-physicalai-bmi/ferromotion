# Oracle provenance

Machine-readable record: `docs/acceptance/oracles/manifest.json`.

`mul_quat` reads `fp_contract::MUL_QUAT`. That constant is `contracted`, the published `mul_add` chain. The unfused formula is available as `mul_quat_with(Contraction::Unfused)` and is checked against one vector from the pinned Linux wheel. It is not the default. Other call sites are unchanged. Equality checks are still `f64::to_bits`. No golden array was edited.

Clock interoperability stays false.

## What is pinned

The Linux comparison artifact is PyPI `mujoco==3.13.0`, file `libmujoco.so.3.13.0`, sha256 `dc9599faeffba8e6ea9f53928de08301ee8c2ba780c4afceadeebe63573c3a1a`, ELF x86-64. Its `.comment` section lists GCC 8.5.0, GCC 14.2.1, clang 20.1.8, and LLD 20.1.8. Which of those compiled `mju_mulQuat` is unproven provenance. `-ffp-contract` and `-mfma` were not recovered from the file. `objdump` of `mju_mulQuat` shows `vmulpd` and `vaddpd`. That observation is not a flag string.

The Menagerie pin is `4d038b3feae26ec82b46a4d586379114012a8ac7`. The preserved baseline ledger is `conformance/menagerie_ledger.json`, sha256 `ca4761e6a3c6cfa022d283e5888acf8e6e196a5fa3946f9fde3b691fa159a776`: 1 of 212 models bit-exact for 20 steps, `efc_J` 359/636, `qacc` 20/636. The first recorded nonzero array is `qpos` for 172 models, `qvel` for 37, and `qacc_warmstart` for 2. That is the ledger's array detail, not an opcode bisect.

## What is unproven

The arm64 unit-test goldens claim MuJoCo 3.13.0. The source SHA, dylib hash, compiler, version, `-ffp-contract`, `-mfma`, ISA, and libm are unproven provenance. Comments in the sources and `scripts/mujoco_ccd_harness_build.sh` are not that build record. Those arrays were not edited. They have not been reproduced bitwise. No ARM worker was used. An unexecuted ARM count is not zero passes and is not a pass.

## Fresh build, not adopted

If those arm64 flags stay irrecoverable, the replacement would be a new oracle named `mujoco-3.13.0-arm64-pinned-build`, with a recorded source SHA, compiler command, artifact sha256, and a verified aarch64 host. It would not be labeled as the old goldens. Adopting it needs a foreman decision, and only after a run reproduces the existing arrays bitwise may those arrays be treated as proven. This note does not make that decision.

The blanket unfuse experiment is archived under `docs/acceptance/research/2026-11-07-unfused-kinematics/` and is not the default.

## Named source build `mujoco-3.13.0-x86_64-contract-off`

Built on this x86_64 host from MuJoCo tag `3.13.0`, commit `123347c0eeab7e13c8da0828ab593bbd95bcf335`. Compiler `/usr/lib/llvm-18/bin/clang` 18.1.3. Flags include `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx`. ARM flags are N/A for this build. Linker is `clang++ -fuse-ld=gold` (GNU gold 1.16). Artifact `libmujoco.so.3.13.0` sha256 `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`. The reproducible command is `scripts/build_mujoco_3130_x86_contract_off.sh`. Host CPU is an Intel Xeon family 6 model 207. The compile does not use the host's FMA or AVX. libm is glibc 2.39. This oracle does not replace the PyPI wheel, the arm64 golden arrays, or `conformance/menagerie_ledger.json`.

`mju_mulQuat` in this artifact is SSE `mulpd`/`addpd`. The probe vector used by `fp_contract` is the same bits as that export. `MUL_QUAT` stays `contracted`. The contracted formula does not match that export. Switching the default to unfused is not selected: the arm64 golden still does not match this oracle, and the op bisect is not a single site.

On the unit-test model `geom_frames_match_mujocos_kinematics_bit_for_bit`, this oracle and the PyPI wheel produced the same 4 geom frames. 38 of 48 components differ from the embedded arm64 golden, 10 match. There is no bitwise link. The arm64 arrays stay unproven provenance.

At `qpos0`, default ferromotion geom frames were bit-exact for 0 of 14842 geoms against this oracle. The largest absolute error is far above an ulp (`wonik_allegro/right_hand.xml`), so that composite is not only contraction.

Instrumented call order at `qpos0`, 212 loaded models and 5 refusals, is in `docs/acceptance/oracles/x86_64-contract-off-op-bisect.json`. Exports compared to this `.so`: `mul_quat` 103 first, `rot_vec_quat` 56 first, no covered-export divergence 53. Including a same-flags recompile of `mji_mulMatVec3` (not an export of the `.so`): `mul_quat` 83, `rot_vec_quat` 52, `mul_mat_vec3` 24, and 53 with none of those. `sincos`, the mass matrix, constraints, and tendons are not in that bisect. An uninstrumented op is not a pass. The 53 models are not 212/212.

The 20-step trajectory against this same `.so`, with the published step seed, is 1 of 212 bit-exact (`realsense_d435i/d435i.xml`) under the published default and stays 1 of 212 after each opt-in site. Native counts, same oracle, 212 models and 3 states:

| Policy sites unfused | M | efc_J | qacc | qfrc_bias | ten_length | xpos | body_inertia |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (published default) | 18/636 | 359/636 | 20/636 | 88/636 | 597/636 | 150/636 | 115/212 |
| `mul_quat` | 21/636 | 365/636 | 23/636 | 144/636 | 598/636 | 293/636 | 115/212 |
| `mul_quat`, `rot_vec_quat` | 22/636 | 367/636 | 24/636 | 151/636 | 599/636 | 296/636 | 115/212 |
| those plus `mul_mat_vec3` | 22/636 | 369/636 | 23/636 | 167/636 | 601/636 | 563/636 | 115/212 |

`qacc` lost one match when `mul_mat_vec3` was added (24/636 back to 23/636). That row is the pre-earclip receipt. After the ear clip the same loss was measured again (24/636 to 23/636) and the site was removed from `install_policy`. See `docs/acceptance/oracles/mass-inertia-branch.md`. `body_inertia` in this table did not move. Its worst absolute error here is 3.53e-7 on `hello_robot_stretch_3/stretch.xml`. `body_mass` stays 192/212; its worst error is 4.44e-16.

With `quat2mat` and `sincos` added to the qpos0 op comparison, the published default still has 53 models with no covered-op mismatch (`mul_quat` 103, `rot_vec_quat` 56). Those 53 next fail a listed native stage: `body_inertia` 21, `xpos` 27, `body_mass` 2, `qfrc_bias` 1, `xquat` 1, `qacc` 1. After all three sites are unfused, covered-op mismatches fall to `normalize4` on 6 models and 206 with no covered-op mismatch. Those 206 are not trajectory passes. `efc_J` still has length mismatches (a different contact row count), which are not ulp errors.

The published default is what the arm64 golden tests run. That suite still fails the same three tests and passes the other 13 `bit_for_bit` tests. The contract-off sites are not selected by architecture. A wrong sha refuses the process.

The table above is the pre-earclip receipt against this source oracle. It is not rewritten. `conformance/menagerie_ledger.json` (sha256 `ca4761e6a3c6cfa022d283e5888acf8e6e196a5fa3946f9fde3b691fa159a776`) remains the PyPI-wheel baseline and was not edited.

## OBJ faces: source ear clip versus the PyPI drop

MuJoCo 3.13.0 tag `123347c0` pins tinyobj `2945a967` and builds it without `TINYOBJLOADER_USE_MAPBOX_EARCUT`. That builtin ear clip triangulates polygons with more than 100 vertices. The PyPI wheel's older tinyobj drops them: a regular 101-gon does not compile (volume too small), and `link_SG3_gripper_body` has 23575 faces. The source build has 23776. Ferromotion follows the source build. The Rust ear clip is an adaptation of that tinyobjloader algorithm (MIT, copyright 2012-Present Syoyo Fujita and contributors), not a claim that the algorithm is original. The wheel baseline ledger is a different artifact and is not silently updated to this policy.

`docs/acceptance/oracles/mass-inertia-branch.md` records the diagnostic build (its own sha256, not `64156abe…`) and the post-earclip 212 against the same source-oracle dumps.

## Named source build `mujoco-3.13.0-aarch64-contract-off`

Measured on thor2.local. The record is `docs/acceptance/oracles/arm64-thor2-parity.md`. Artifact `libmujoco.so.3.13.0` sha256 `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`, build `build-20261109T213225Z`, MuJoCo `123347c0eeab7e13c8da0828ab593bbd95bcf335`, clang 18.1.3, shared link `-rtlib=compiler-rt -unwindlib=libgcc`.

On that Menagerie corpus the ARM `native.txt` and `step.txt` were byte-identical to the pinned x86 contract-off schedules. That is the equivalence. It is this corpus only. The run's ferromotion `e0557dd109151a02702dfb262ea0a8c2b9dd07db` matched 20-step on 1 of 212 models against those dumps. The Darwin golden arrays were not reproduced, so `mujoco-3.13.0-arm64-pinned-build` stays unadopted and `MUL_QUAT` stays contracted.
