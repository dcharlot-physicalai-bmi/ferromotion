# Linux aarch64 contract-off bundle

Not executed on ARM. The x86 proof below used the pinned contract-off library
`64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc` and its plugin
libraries. Darwin arm64 goldens are a different oracle and are not relabeled.

Unpack into `$HOME/ferromotion-oracle`. The top directory is MuJoCo commit
`123347c0eeab7e13c8da0828ab593bbd95bcf335`.

V2 archive sha256 `0c98f3bc63175eb56af0c40164ae78aac43efef231875042986fe1df1bd0c0ad`
is the base. Script fixes after that archive are an overlay. Do not replace the
archive. `provenance/OVERLAY.txt` names the overlay. Apply it with the overlay's
`apply-overlay.sh` before these commands. The apply script refuses a tree whose
replaced files are not the V2 bytes.

## Provenance

`provenance/REVISIONS` lists upstream commits. `provenance/*.patch` is `git diff HEAD`
from the checkout that was configured on x86 (ccd and qhull cmake patches, lodepng's
added CMakeLists.txt). `deps/SHA256SUMS` hashes those patch files and `provenance/REVISIONS`.
`provenance/mujoco-ls-tree.txt` is `git ls-tree -r HEAD` of the MuJoCo commit.
`provenance/menagerie-ls-files.txt` is `git ls-files -s` at Menagerie
`4d038b3feae26ec82b46a4d586379114012a8ac7`.
Ferromotion source archived in the V2 tree is the commit in `provenance/ferromotion-commit.txt`
(`7e500d4530a460303ec20e3e3a05294dd31b1700`) on `cursor/mesh-earclip-inertia-bb9d`.
The overlay changes scripts under that tree. It does not replace the crate sources.
`RECEIPT.txt` from each run hashes the main library, the four plugins, both schedules,
the executables, the outputs, the compiler, and this source. `RECEIPT.sha256` is the
hash of that file. A libmujoco line by itself is not the receipt.

## Resource limits

`ulimit -t 14400` is RLIMIT_CPU: 14400 CPU seconds for each process. It is not a
sum across the build, not wall-clock time, and not memory. `ulimit -v 50331648`
is RLIMIT_AS: 48 GiB of virtual address space for each process, in KiB. It is not
resident set size and not a cgroup `memory.max`. `timeout -k 30 14400` (600 for
the mulquat probe, after `ulimit -t 600`) is the wall-clock cap and kills the
process group. `--foreground` is not used, because that mode does not time out
children. Compile jobs are `CMAKE_BUILD_PARALLEL_LEVEL=2`, `cmake --build -j 2`,
`CARGO_BUILD_JOBS=2`, and `cargo --jobs 2`. `OMP_NUM_THREADS=1`.
`--test-threads=2` limits test execution only.

Cargo also requires `CARGO_NET_OFFLINE=true`, `--locked`, a `vendor/` directory,
and `.cargo/config.toml` with `vendored-sources` and `net.offline = true`.
The config file does not pass `--locked` and does not choose the target directory.
The scripts do. `CARGO_TARGET_DIR` is a fresh directory under the run output,
outside the unpacked tree. A `ferromotion/target` directory is a refusal.

No sudo, no apt, no pip, no rustup from these scripts.

## x86 proof, already run, ARM not run

`oracle_dump` linked to that x86 library, with `MUJOCO_PLUGIN_DIR` pointing at
`libactuator.so`, `libelasticity.so`, `libsensor.so`, and `libsdf_plugin.so`.
After the subnormal hexfloat fix (`0x0.<fraction>p-1022`, the same spelling as
Python `float.hex`):

* hexfloat selftest: exact bits for +/-0, min/max subnormal, min normal, +/-1, max finite; NaN and Inf refused
* native replay of the existing 212-model oracle: `cmp` identical
* 20-step replay of the existing step oracle: `cmp` identical
* classify, from the earlier plugin-enabled run: 217 one-level XML files, 212 `ok`, 5 `fail`, names match
  `probes/expected-load-failures.txt`, bytewise sorted (`LC_ALL=C`)

The selftest and both replays were run again on x86 after the formatter change.
They were not run on ARM.

## Thor2 order

Each script refuses any host whose `hostname -s` is not `thor2`, and refuses a
reused output directory. `FERROMOTION_ORACLE_SMOKE=x86` exists so the x86 host
can check the Rust runners. A smoke receipt is marked `x86_smoke_not_arm`.
Do not set that variable on thor2.

```text
sha256sum -c MANIFEST.sha256
./scripts/build-thor2.sh
./scripts/run-probes.sh <build-dir>
./scripts/run-212.sh <menagerie-default> <build-dir>
```

`build-thor2.sh` runs `probes/hexfloat_selftest` before configure. A selftest
failure exits 3 and does not start cmake.

`run-probes.sh` exits 1 when the mulquat bits differ from the x86 contract-off vector.
The bits are still written. They do not edit a golden. The receipt is written
before that exit.

`run-212.sh` checks Menagerie HEAD and a clean `git status --porcelain`, then requires
217 XML, 212 loads, and the five pinned failures. It writes a fresh `native.txt` and
`step.txt` in the same field layout as the x86 schedules (3 states, 20 steps).
Exit 0 means those dumps were written and the load set matched. It does not mean
the floats equal the x86 files. NaN or Inf in a field exits 3.

## Exit codes

* 2 refusal: wrong host, missing tool, reused directory, bad corpus, vendor/config, missing plugin on a receipt
* 3 execution or incomplete: compiler, cargo, timeout, missing summary, counts other than 212 models and 20 steps, loader refusal, hexfloat selftest
* 1 comparison finished and is not bit-exact. The logs stay.
* 0 the check that script owns passed. For `run-rust-212.sh`, 0 is full parity only.

Rust, only after David approves the isolated toolchain in `rust/OPTIONAL-RUSTUP.md`:

```text
./scripts/run-goldens.sh
./scripts/run-rust-212.sh <run-dir>
```

`run-goldens.sh` runs the three embedded Darwin tests. Exit 1 means those tests
failed. Exit 0 means they passed on that host. Neither result rewrites the tests,
and neither result is the Linux oracle. On x86 those three fail.

`run-rust-212.sh` runs `menagerie_native` and `menagerie_native_step` against the
ARM dumps. Those examples return 0 after printing mismatches. The wrapper's
exit 0 requires the step line `212 of 212` at 20 steps, the native line
`212 models compared`, zero loader refusals, and every `N of M` tally equal.
`1 of 212` or `0 of 212`, with the rest of the summary complete, is exit 1.
A missing summary or a count other than 212 and 20 is exit 3.
Cargo failing is exit 3, not exit 1.
