# Direct-pull receipt for the linked Thor2 build

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
50541eeae5424cbbcf42d48dca5cc4112f74fb08d3f8a32a519183b0f8024774  04-corpus.sh
8b2408ee41f950dd329af8ad04a361fb449d186a1b10127f2f40599bfd2bbdd4  05-rust.sh
5ea7fee85bf55cbe239df15abbb59123a027d275566c5903986acbb32b3a59ff  compare-arm-x86.sh
a6aa4ab9b14b36bfa2f7aac0e49b3cc1ce799bb5d1ff1507dca5b56160e2cd45  direct-pull-receipt.sh
2276697c6c8e2474b6bc42e7052c6f3ae31677d8fde8ef65f41cefc0de06ca32  lib.sh
1bbe650fac082b8c01ba00ab2c2e1965fea7a5fb3ff75285d1086cb75b7e0f66  pins.sh
61972b22b66be5b5de9b9b887d80505b2d8b3a2b9e77f0845b424b0cba34ba08  runners/compiler-rt.sh
394990f2ca39eba1be353136fdb810bb0b269e892cc32f3b1ddf53b19735c049  runners/limits.sh
64c139aec23de404820a41a3a8ce5f842d6c48a1432e446ab373a721b8c8b1d8  runners/read-mulquat-fixture.sh
e9c5cf157fff0009c2530bf3da9885aa061e668cd458b4b26fb45fffda6bf8c9  runners/run-probes.sh
e2cdf6e2532b599ce77b4fb4337bf602e32fa0ec77765b3b853044043786cacc  runners/run-212.sh
414530c0c9f76377a610a754e20fd208fc1892e304aa56070d53dd5f8637e21a  runners/run-goldens.sh
da3307c9025b5783e422207714d6b0ffed1587a80a78efd1ffa8f5ae6ba711ee  runners/run-rust-212.sh
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
