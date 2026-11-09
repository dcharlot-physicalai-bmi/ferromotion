# Thor2 measurement: `mujoco-3.13.0-aarch64-contract-off`

Recorded from the Thor2 run of stages 04 and 05 at ferromotion `e9e20184c954de478fe6bdbd57b503253d7e020b`. This file does not extend the measurement past that corpus. `conformance/menagerie_ledger.json` was not rewritten. The embedded Darwin arm64 golden arrays were not edited. The published default stays contracted.

## Host and build

* host `thor2.local`, Ubuntu 24.04, aarch64, clang 18.1.3
* build directory `build-20261109T213225Z`
* MuJoCo `123347c0eeab7e13c8da0828ab593bbd95bcf335`
* `libmujoco.so.3.13.0` sha256 `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`
* compiler-rt sha256 `a38c82691dba41f2fc96e98786f0d9c7c4f1c42f7569b94ce3e9e822cd52427d`
* shared link `-rtlib=compiler-rt -unwindlib=libgcc`
* plugin sha256 prefixes reported from that build: actuator `6858228a`, elasticity `6f9f5dda`, sensor `f37f9182`, sdf `5c860556`

## Named oracle

`mujoco-3.13.0-aarch64-contract-off` is this library: sha256 `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`.

On this corpus it is equivalent to `mujoco-3.13.0-x86_64-contract-off`. The ARM `native.txt` and `step.txt` were byte-identical to the pinned x86 schedules, sha256 `a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7` and `a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be`. That equivalence is this corpus only.

## Stage 04

Exit 0.

The mulquat probe matched the x86 contract-off export: `3fe8000000000000 bfc5c28f5c28f5c2 3fc3333333333333 3fdc28f5c28f5c29`.

Classify was 217 xml: 212 ok, 5 pinned fails. `classify.txt` sha256 prefix `0e5cf64c`.

`stage4-x86-verdict` was `full_parity`: every model field 212/212, every data field 636/636, `efc_AR` 30/30 on 10 models, step fields 4240/4240.

The stage 4 log showed 12 MuJoCo QACC-instability warnings. Both sides have those warnings identically.

Receipt sha256 prefixes: probe `3214b952`, corpus `758a0a59`.

## Stage 05

Exit 1. That is a finished mismatch.

The ferromotion tree that ran was the run directory's `e0557dd109151a02702dfb262ea0a8c2b9dd07db`. It was not `7998624df8a4ac2591187540a59ddc9b13d51640`.

Against the ARM dumps, that tree matched 20-step on 1/212 models, with `body_inertia` 115/212, `body_mass` 192/212, and `qacc` 20/636.

The three Darwin goldens failed on Linux. The stage 5 script expects that failure.

Receipt sha256 prefixes: goldens `df0a1a98`, rust212 `5cfc11a4`.

The rust212 receipt is the directory `rust212-20261109T215708Z` under the MuJoCo build home. It is a dated record of the e0557dd comparison. A later stage 05 re-run does not overwrite it.
