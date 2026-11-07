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
