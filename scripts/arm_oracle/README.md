# Linux aarch64 contract-off bundle

Not executed on ARM. The x86 proof below used the pinned contract-off library
`64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc` and its plugin
libraries. Darwin arm64 goldens are a different oracle and are not relabeled.

Unpack into `$HOME/ferromotion-oracle`. The top directory is MuJoCo commit
`123347c0eeab7e13c8da0828ab593bbd95bcf335`.

## Provenance

`provenance/REVISIONS` lists upstream commits. `provenance/*.patch` is `git diff HEAD`
from the checkout that was configured on x86 (ccd and qhull cmake patches, lodepng's
added CMakeLists.txt). `deps/SHA256SUMS` hashes those patch files and `provenance/REVISIONS`.
`provenance/mujoco-ls-tree.txt` is `git ls-tree -r HEAD` of the MuJoCo commit.
`provenance/menagerie-ls-files.txt` is `git ls-files -s` at Menagerie
`4d038b3feae26ec82b46a4d586379114012a8ac7`.
Ferromotion source is the commit in `provenance/ferromotion-commit.txt` on `cursor/mesh-earclip-inertia-bb9d`.

## x86 proof, already run, ARM not run

`oracle_dump` linked to that x86 library, with `MUJOCO_PLUGIN_DIR` pointing at
`libactuator.so`, `libelasticity.so`, `libsensor.so`, and `libsdf_plugin.so`:

* native replay of the existing 212-model oracle: `cmp` identical
* 20-step replay of the existing step oracle: `cmp` identical
* classify: 217 one-level XML files, 212 `ok`, 5 `fail`, names match
  `probes/expected-load-failures.txt`, bytewise sorted (`LC_ALL=C`)

## Thor2 order

Each script refuses any host whose `hostname -s` is not `thor2`, refuses a reused
output directory, uses 2 compile jobs, `OMP_NUM_THREADS=1`, and `ulimit -t 14400`.
No sudo, no apt, no pip, no rustup.

```text
sha256sum -c MANIFEST.sha256
./scripts/build-thor2.sh
./scripts/run-probes.sh <build-dir>
./scripts/run-212.sh <menagerie-default> <build-dir>
```

`run-probes.sh` exits 1 when the mulquat bits differ from the x86 contract-off vector.
The bits are still written. They do not edit a golden.

`run-212.sh` checks Menagerie HEAD and a clean `git status --porcelain`, then requires
217 XML, 212 loads, and the five pinned failures. It writes a fresh `native.txt` and
`step.txt` in the same field layout as `scripts/mujoco_native_oracle.py` and
`scripts/mujoco_native_step_oracle.py` (3 states, 20 steps).

Rust, only after David approves the isolated toolchain in `rust/OPTIONAL-RUSTUP.md`:

```text
./scripts/run-goldens.sh
./scripts/run-rust-212.sh <run-dir>
```

`run-goldens.sh` runs the three embedded Darwin tests and exits with cargo's status.
It does not change the tests. On x86 those three fail. That result is not a Linux oracle.

`run-rust-212.sh` runs `menagerie_native` and `menagerie_native_step` against the ARM
dumps. Exit 0 means the runners finished and printed summaries, not that every array matched.
