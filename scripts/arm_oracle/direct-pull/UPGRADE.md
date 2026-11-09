# Direct-pull receipt for the linked Thor2 build

## Stage 5 re-run

Stage 04 and the e0557dd stage 05 already ran. Do not run them again. The rust212 receipt `rust212-20261109T215708Z` stays. This re-run checks out the fidelity branch tip `515a7dec4b4ed2fa7257f328c7c3f9b169b8d7ba` (tree `67e20a00014302498e9b350862549f9d705911a6`), which contains `7998624df8a4ac2591187540a59ddc9b13d51640`, into a sibling directory. Cargo uses the existing vendor and `--offline`. A missing crate is listed and cargo is not started.

Replace `REPLACE_WITH_FULL_SHA` with the full SHA of the commit that contains these scripts. Fetch with tokens unset. `sha256sum -c` is the check.

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=REPLACE_WITH_FULL_SHA
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}"
D="$HOME/ferromotion-oracle/stage5-rereun-${C}"
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
mkdir -p "$D/runners"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/05-rust-rereun.sh" "$BASE/scripts/arm_oracle/direct-pull/05-rust-rereun.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/direct-pull-receipt.sh" "$BASE/scripts/arm_oracle/direct-pull/direct-pull-receipt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/lib.sh" "$BASE/scripts/arm_oracle/direct-pull/lib.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/pins.sh" "$BASE/scripts/arm_oracle/direct-pull/pins.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/compiler-rt.sh" "$BASE/scripts/arm_oracle/compiler-rt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/limits.sh" "$BASE/scripts/arm_oracle/limits.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-goldens.sh" "$BASE/scripts/arm_oracle/run-goldens.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-rust-212.sh" "$BASE/scripts/arm_oracle/run-rust-212.sh"
cd "$D"
sha256sum -c - <<'EOF'
972220a41726986e6909f0b6fcb0ec26510bee55076ce36f34549686a1a22e1e  05-rust-rereun.sh
6f6cbf65af7d13edbac0814510438e47f0af783b4b3c5ac62b3f891521f7f484  direct-pull-receipt.sh
baedafed35e277629cef12370b15762ac9a41b3361ee912b6f4ad8c916e1b011  lib.sh
d59c887207dc2f46023d8fd634ec684d50f895beb7bc7ab6b13c057b12742028  pins.sh
61972b22b66be5b5de9b9b887d80505b2d8b3a2b9e77f0845b424b0cba34ba08  runners/compiler-rt.sh
b0c08e9569be14e624caffe17563b4ade2b34c1e33cbe71b601401ef22e49f38  runners/limits.sh
2f45544068625850ee7d9b15ebfb94d7fd524594d25d63d0ecef931e1f69f41e  runners/run-goldens.sh
8f6a361a4db8487b95ab9b55fd3c8a9ff4e051b89908a3f392d0bccc49020b1d  runners/run-rust-212.sh
EOF
chmod +x 05-rust-rereun.sh direct-pull-receipt.sh runners/*.sh
# CPU only. The runners force 2 jobs, --offline, and OMP_NUM_THREADS=1.
setsid nohup bash -c '
  set -euo pipefail
  export LC_ALL=C
  export OMP_NUM_THREADS=1
  export CARGO_NET_OFFLINE=true
  unset CUDA_VISIBLE_DEVICES MUJOCO_GL || true
  bash "$1/05-rust-rereun.sh" "$2" > "$1/stage5-rereun.log" 2>&1
' bash "$D" "$RUN" > "$D/driver.log" 2>&1 < /dev/null &
echo $! > "$D/driver.pid"
echo "driver pid $(cat "$D/driver.pid")"
echo "tail -f $D/stage5-rereun.log"
```

Exit 0 is Rust-versus-ARM-dump parity for `515a7dec4b4ed2fa7257f328c7c3f9b169b8d7ba`. Exit 1 is a finished mismatch. Exit 2 is a refusal, including a vendored-crate gap. The new receipt names that commit and tree `67e20a00014302498e9b350862549f9d705911a6`.

## Stages 4 and 5 only

Stage 3 already passed. Do not run it again. `provenance/build-dir.txt` and `stage3-ok` point at `build-20261109T213225Z`. libmujoco is `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`. Leave `build-20261107T131346Z` untouched.

The mulquat fixture's `#` lines are comments. The expected value is the one data line. The probe, corpus, and Rust output directories get their own `identity.txt` (role, stage-3 path, compiler-rt path and sha256). The stage-3 `identity.txt` check stays on the build receipt. The Rust examples do not link `libmujoco`. Jobs stay at 2. Nothing here enables a GPU.

Replace `REPLACE_WITH_FULL_SHA` with the full SHA of the commit that contains these scripts. The sha256 lines below are the fetched file bytes. They do not cover this markdown file. Fetch with tokens unset. `sha256sum -c` is the check.

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=REPLACE_WITH_FULL_SHA
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}"
D="$HOME/ferromotion-oracle/stage45-${C}"
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
mkdir -p "$D/runners"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/04-corpus.sh" "$BASE/scripts/arm_oracle/direct-pull/04-corpus.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/05-rust.sh" "$BASE/scripts/arm_oracle/direct-pull/05-rust.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/compare-arm-x86.sh" "$BASE/scripts/arm_oracle/direct-pull/compare-arm-x86.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/direct-pull-receipt.sh" "$BASE/scripts/arm_oracle/direct-pull/direct-pull-receipt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/lib.sh" "$BASE/scripts/arm_oracle/direct-pull/lib.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/pins.sh" "$BASE/scripts/arm_oracle/direct-pull/pins.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/compiler-rt.sh" "$BASE/scripts/arm_oracle/compiler-rt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/limits.sh" "$BASE/scripts/arm_oracle/limits.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/read-mulquat-fixture.sh" "$BASE/scripts/arm_oracle/read-mulquat-fixture.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-probes.sh" "$BASE/scripts/arm_oracle/run-probes.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-212.sh" "$BASE/scripts/arm_oracle/run-212.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-goldens.sh" "$BASE/scripts/arm_oracle/run-goldens.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-rust-212.sh" "$BASE/scripts/arm_oracle/run-rust-212.sh"
cd "$D"
sha256sum -c - <<'EOF'
7e98a4d9ab69c79251663e8ff58c961ea91891e09a5d7c8f3f948cd174194345  04-corpus.sh
1ea51dd026e7493e7debcaf4862b54cc982072630bf400c65d0b6809e5ceac58  05-rust.sh
5ea7fee85bf55cbe239df15abbb59123a027d275566c5903986acbb32b3a59ff  compare-arm-x86.sh
6f6cbf65af7d13edbac0814510438e47f0af783b4b3c5ac62b3f891521f7f484  direct-pull-receipt.sh
baedafed35e277629cef12370b15762ac9a41b3361ee912b6f4ad8c916e1b011  lib.sh
d59c887207dc2f46023d8fd634ec684d50f895beb7bc7ab6b13c057b12742028  pins.sh
61972b22b66be5b5de9b9b887d80505b2d8b3a2b9e77f0845b424b0cba34ba08  runners/compiler-rt.sh
b0c08e9569be14e624caffe17563b4ade2b34c1e33cbe71b601401ef22e49f38  runners/limits.sh
64c139aec23de404820a41a3a8ce5f842d6c48a1432e446ab373a721b8c8b1d8  runners/read-mulquat-fixture.sh
e9c5cf157fff0009c2530bf3da9885aa061e668cd458b4b26fb45fffda6bf8c9  runners/run-probes.sh
e2cdf6e2532b599ce77b4fb4337bf602e32fa0ec77765b3b853044043786cacc  runners/run-212.sh
2f45544068625850ee7d9b15ebfb94d7fd524594d25d63d0ecef931e1f69f41e  runners/run-goldens.sh
8f6a361a4db8487b95ab9b55fd3c8a9ff4e051b89908a3f392d0bccc49020b1d  runners/run-rust-212.sh
EOF
chmod +x 04-corpus.sh 05-rust.sh compare-arm-x86.sh direct-pull-receipt.sh runners/*.sh
# CPU only. The scripts force 2 jobs, OMP_NUM_THREADS=1, and no GPU cmake flag.
# Exit 1 from stage 4 is a finished x86 comparison. Stage 5 still runs.
# Exit 2 or 3 stops the driver.
setsid nohup bash -c '
  set -euo pipefail
  export LC_ALL=C
  export OMP_NUM_THREADS=1
  unset CUDA_VISIBLE_DEVICES MUJOCO_GL || true
  bash "$1/04-corpus.sh" "$2" > "$1/stage4.log" 2>&1 || {
    rc=$?
    if [[ "$rc" -ne 1 ]]; then exit "$rc"; fi
  }
  bash "$1/05-rust.sh" "$2" > "$1/stage5.log" 2>&1
' bash "$D" "$RUN" > "$D/driver.log" 2>&1 < /dev/null &
echo $! > "$D/driver.pid"
echo "driver pid $(cat "$D/driver.pid")"
echo "tail -f $D/stage4.log"
```

Stage 4 exit 0 is ARM-versus-x86 full parity. Exit 1 is a finished divergence: `provenance/stage4-x86-verdict.txt` names the first field, and `stage4-ok` is already written. Exit 3 is still a compile or dump failure. Stage 5 exit 0 is Rust-versus-ARM-dump parity. It is not ARM-versus-x86.

## Stage 3 that already passed

`build-20261107T131346Z` (libmujoco `cefada9b5165a787e2c2695c9db172993c922f5533e19ed8108a181c53f85845`) imports `__muloti4`. That symbol is GLOBAL HIDDEN in `libclang_rt.builtins-aarch64.a` (sha `a38c82691dba41f2fc96e98786f0d9c7c4f1c42f7569b94ce3e9e822cd52427d` on Thor2), so appending the archive to the probe link cannot satisfy it. Leave that directory untouched.

On aarch64, clang lowers `__builtin_mul_overflow` of `__int128` to `__muloti4`. On x86_64 the same builtin is inlined. The shared-library link passes `-rtlib=compiler-rt -unwindlib=libgcc`. Compile flags stay `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a`. That recipe produced `build-20261109T213225Z`. Do not start `03-cbuild.sh` again.

## Earlier receipt recovery

Obsolete for the 2026-11-07 library. Do not run `recover-stage3.sh` against `build-20261107T131346Z`. That script now refuses libmujoco `cefada9b5165a787e2c2695c9db172993c922f5533e19ed8108a181c53f85845`. The historical steps below are how that directory was recorded. They are not the next build.

`41ed70b` reached the five library hashes, then the archive `write-receipt.sh` exited 2 because `scripts/README.md` is not part of the stage-1 tree. That file is not created. The direct-pull receipt records the checkout commit and tree SHAs and sets `archive_manifest_rebuilt no`. It hashes `ferromotion/scripts/arm_oracle/README.md`, which the checkout already contains.

On thor2, from an empty directory, with no `GH_TOKEN`:

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=93117aa508b99de85fb550364a0aa8cc64b563b4
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}/scripts/arm_oracle/direct-pull"
D="$HOME/ferromotion-oracle/receipt-${C}"
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
mkdir -p "$D"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/recover-stage3.sh" "$BASE/recover-stage3.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/direct-pull-receipt.sh" "$BASE/direct-pull-receipt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/mulquat.c" "$BASE/probes/mulquat.c"
cd "$D"
sha256sum -c - <<'EOF'
4d78326f0c309d5f4d87e13fe526a0a26bd2578547db4ff02fdeef23f4fc4bbc  recover-stage3.sh
02ce38e7a7013539fee944c1ce3ba7473580d7bc8de673dcf04ebc674165e3a9  direct-pull-receipt.sh
1bb5bd6fb656623496075e1e8a89470731c7e44fbc6ad9cf7de82361e4b4bb80  mulquat.c
EOF
chmod +x recover-stage3.sh direct-pull-receipt.sh
# Replace the archive receipt only when it is still the e0557dd copy.
echo "bc7805f6768fedd3336dcc9e1ef8e7dd6647cc199bf839ea285fd74b59316a70  $RUN/scripts/write-receipt.sh" | sha256sum -c
cp -f direct-pull-receipt.sh "$RUN/scripts/write-receipt.sh"
cp -f mulquat.c "$RUN/probes/mulquat.c"
bash recover-stage3.sh
```

`recover-stage3.sh` still requires Linux aarch64, hostname `thor2`, and the five library hashes from the linked build. It does not run cmake. `stage3-ok` is written only after `direct-pull-receipt.sh` succeeds. Then `ecfccc6`'s `04-corpus.sh` can run. Do not start `03-cbuild.sh` again.
