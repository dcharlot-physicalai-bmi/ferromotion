# Linux aarch64 source oracle bundle

Not built. Not run. This is not a Darwin build and it does not replace the embedded arm64 golden arrays.

The script `scripts/build_mujoco_3130_arm64_linux_contract_off.sh` exits unless `uname` is Linux and `aarch64`. On this x86_64 cloud host it refuses. It must not be copied onto a Mac and run there. Apple `__sincos_stret` and the unproven dylib goldens are a different oracle.

When a verified Linux aarch64 worker is approved, the script clones nothing by itself. The caller supplies a checkout of MuJoCo `3.13.0` at `123347c0eeab7e13c8da0828ab593bbd95bcf335`. Flags are `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a`. x86 `-mno-fma` and `-mno-avx` are N/A and are not passed. The oracle id to record after the build, once its sha256 is known, is `mujoco-3.13.0-arm64-linux-contract-off`. Until that sha exists, no policy may select it.

A one-line probe after the build, still on that worker:

```text
sha256sum libmujoco.so.3.13.0
```

and a call of `mju_mulQuat` on `[0.9, 0.1, -0.2, 0.3]` and `[0.8, -0.1, 0.4, 0.2]`, compared with the x86 contract-off bits `3fe8000000000000 bfc5c28f5c28f5c2 3fc3333333333333 3fdc28f5c28f5c29`. A difference does not edit any golden.
