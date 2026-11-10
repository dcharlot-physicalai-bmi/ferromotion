# Aarch64 direct-pull for MuJoCo 3.14.0–3.17.0 and 4.0.0

These scripts mirror `scripts/arm_oracle/direct-pull` on `cursor/mesh-earclip-inertia-bb9d` (the 3.13.0 Thor2 procedure). They are the commands for a later approved run. They have not been executed. This tree does not contain an aarch64 library, and no x86-versus-aarch64 bitwise comparison has been made.

An oracle id `mujoco-<version>-aarch64-contract-off` does not count until a run on Thor2 reproduces the x86 contract-off dumps for that version bitwise. David's approval is required before any stage runs on Thor2. Running a stage on any other host is refused.

## Pins

`pins.sh` records the tag commits checked against `github.com/google-deepmind/mujoco` and each commit's tree. Dependency commits are the ones CMake fetched for the x86 contract-off build of that tag. Menagerie stays at `4d038b3feae26ec82b46a4d586379114012a8ac7`.

## Flags

The aarch64 compile is the 3.13.0 Thor2 set, not the x86 `-march` set:

`-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a`

x86 `-mno-fma` and `-mno-avx` are not passed. The x86 oracles use `LDFLAGS=-fuse-ld=gold`. This aarch64 script does not select a linker until the Thor2 linker is recorded. Clang must be 18.1.3.

## Stages

```text
bash 01-materialize.sh <version>
bash 02-verify.sh <run>
bash 03-cbuild.sh <run>
bash 04-corpus.sh <run>
bash 05-rust.sh <run>
```

`<version>` is one of `3.14.0`, `3.15.0`, `3.16.0`, `3.17.0`, `4.0.0`. The 3.13.0 Thor2 procedure is the one already on the mesh-earclip branch. It is not re-labeled here.

Stage 1 is the only network stage. It refuses unless all of these hold: Linux, `aarch64`, hostname `thor2`, and `FERROMOTION_THOR2_APPROVED=1`. It does not use sudo, apt, or a copied credential. Stage 4 stops until the x86 native and step dumps for that version are placed in the run directory and match the sha256 recorded from the x86 build. It does not download them. A finished stage 4 writes dumps. It does not mean they match x86. Stage 5 is not a parity claim: parity is a later comparison against those x86 files, and it has not been run.
