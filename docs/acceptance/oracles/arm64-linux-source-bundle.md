# Linux aarch64 source oracle bundle

The Thor2 run is recorded in `docs/acceptance/oracles/arm64-thor2-parity.md` as `mujoco-3.13.0-aarch64-contract-off`, library sha256 `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`. This file is the earlier bundle note. It is not a Darwin build and it does not replace the embedded arm64 golden arrays.

A packed source bundle for a later thor2 run, not executed here, is `ferromotion-arm64-linux-oracle-thor2.tar.gz` in the cloud artifact store. Its README is the runbook. thor stays unused until that thor2 run is reviewed.

The script `scripts/build_mujoco_3130_arm64_linux_contract_off.sh` exits unless `uname` is Linux and `aarch64`. On this x86_64 cloud host it refuses. It must not be copied onto a Mac and run there. Apple `__sincos_stret` and the unproven dylib goldens are a different oracle.

When a verified Linux aarch64 worker is approved, the script clones nothing by itself. The caller supplies a checkout of MuJoCo `3.13.0` at `123347c0eeab7e13c8da0828ab593bbd95bcf335`. Flags are `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a`. x86 `-mno-fma` and `-mno-avx` are N/A and are not passed. The id reserved in this bundle note was `mujoco-3.13.0-arm64-linux-contract-off`. The measured oracle is `mujoco-3.13.0-aarch64-contract-off`. No policy selects it. `MUL_QUAT` stays contracted.

A one-line probe after the build, still on that worker:

```text
sha256sum libmujoco.so.3.13.0
```

and a call of `mju_mulQuat` on `[0.9, 0.1, -0.2, 0.3]` and `[0.8, -0.1, 0.4, 0.2]`, compared with the x86 contract-off bits `3fe8000000000000 bfc5c28f5c28f5c2 3fc3333333333333 3fdc28f5c28f5c29`. A difference does not edit any golden.

The packed V2 archive is sha256 `0c98f3bc63175eb56af0c40164ae78aac43efef231875042986fe1df1bd0c0ad` (159 parts, 35371 manifest entries). Script, receipt, and hexfloat fixes after that archive are a small overlay applied onto an unpacked V2 tree. They do not replace the archive. The Thor2 measurement is the parity record above, not a claim that this archive's manifest was rebuilt.

`scripts/arm_oracle/direct-pull/` is a later public-fetch procedure for thor2. It checks out ferromotion `e0557dd109151a02702dfb262ea0a8c2b9dd07db` and the pinned MuJoCo, dependency, and Menagerie commits. It does not recreate the archive manifest. The two x86 schedule files are not in that commit, so the corpus and Rust stages stop until those exact bytes are supplied.
