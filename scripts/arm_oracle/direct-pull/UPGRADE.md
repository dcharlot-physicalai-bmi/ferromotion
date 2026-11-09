# Direct-pull receipt for the linked Thor2 build

## Fresh stage 3, then 4 and 5

`build-20261107T131346Z` (libmujoco `cefada9b5165a787e2c2695c9db172993c922f5533e19ed8108a181c53f85845`) imports `__muloti4`. That symbol is GLOBAL HIDDEN in `libclang_rt.builtins-aarch64.a` (sha `a38c82691dba41f2fc96e98786f0d9c7c4f1c42f7569b94ce3e9e822cd52427d` on Thor2), so appending the archive to the probe link cannot satisfy it. Leave that directory untouched.

On aarch64, clang lowers `__builtin_mul_overflow` of `__int128` to `__muloti4`. On x86_64 the same builtin is inlined. The shared-library link now passes `-rtlib=compiler-rt -unwindlib=libgcc`. Compile flags stay `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a`. Stage 3 fails if `nm -D --undefined-only` on a built `.so` still lists `__muloti4` or another `__*ti*` compiler-rt helper. The new build directory is a new `build-$STAMP`. `build-dir.txt` and `stage3-ok` point at it. Probes do not append the archive.

The Rust examples do not link `libmujoco`. Jobs stay at 2. Nothing here enables a GPU.

Replace `REPLACE_WITH_FULL_SHA` with the full SHA of the commit that contains these scripts. The sha256 lines below are the fetched file bytes. They do not cover this markdown file. Fetch with tokens unset. `sha256sum -c` is the check.

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=REPLACE_WITH_FULL_SHA
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}"
D="$HOME/ferromotion-oracle/stage3-${C}"
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
mkdir -p "$D/runners"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/03-cbuild.sh" "$BASE/scripts/arm_oracle/direct-pull/03-cbuild.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/04-corpus.sh" "$BASE/scripts/arm_oracle/direct-pull/04-corpus.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/05-rust.sh" "$BASE/scripts/arm_oracle/direct-pull/05-rust.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/compare-arm-x86.sh" "$BASE/scripts/arm_oracle/direct-pull/compare-arm-x86.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/direct-pull-receipt.sh" "$BASE/scripts/arm_oracle/direct-pull/direct-pull-receipt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/lib.sh" "$BASE/scripts/arm_oracle/direct-pull/lib.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/pins.sh" "$BASE/scripts/arm_oracle/direct-pull/pins.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/build-thor2.sh" "$BASE/scripts/arm_oracle/build-thor2.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/compiler-rt.sh" "$BASE/scripts/arm_oracle/compiler-rt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/limits.sh" "$BASE/scripts/arm_oracle/limits.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-probes.sh" "$BASE/scripts/arm_oracle/run-probes.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-212.sh" "$BASE/scripts/arm_oracle/run-212.sh"
cd "$D"
sha256sum -c - <<'EOF'
f39728791ff3db2dd2b5a454364f67d65e061b5ae9c56661d2260fd142b0944f  03-cbuild.sh
d8ceef20ca1b201efc9321c843aed9b633457a2d7a732e65215709c9e871d136  04-corpus.sh
a3d4d5930ed83b044ca10942a882d5c04748536c20f299c24a52c0cc3875a289  05-rust.sh
12363c3ff02ebb3dd7d85195682d390a4ee79acf4c6311bc3e6eeea6c99a4353  compare-arm-x86.sh
4f5e974bf2fd1bbdb462ffc0e5f6c1c1368d61ddf498283f27143a6251124cd3  direct-pull-receipt.sh
bcd48f1e7f777992d60bdde1a2a6793b9ee00f93462bbcc28f655feb9ef1cfb1  lib.sh
397d3abc74fab2c6ec28776231829caac9052292d60588dabe5272350957d277  pins.sh
6232f8d1221b92d6bfbedb66712c8f0d24be7ae87a56241f98314497454771c5  runners/build-thor2.sh
9dc55efdf673f9a8496f6c3c72a311905e30dcb45a9107b59f9930236d80b835  runners/compiler-rt.sh
394990f2ca39eba1be353136fdb810bb0b269e892cc32f3b1ddf53b19735c049  runners/limits.sh
c09fd221be6bc55c07566ee6d9c84d9d521381e6a44b04e5532680296fcaee83  runners/run-probes.sh
eb2c88e5a0ce32e6a44d1a71586b76c868bc73fed31d5073cbdf2fe8dc9821bc  runners/run-212.sh
EOF
chmod +x 03-cbuild.sh 04-corpus.sh 05-rust.sh compare-arm-x86.sh direct-pull-receipt.sh runners/*.sh
# CPU only. The scripts force 2 jobs, OMP_NUM_THREADS=1, and no GPU cmake flag.
setsid nohup bash -c '
  set -euo pipefail
  export LC_ALL=C
  export OMP_NUM_THREADS=1
  unset CUDA_VISIBLE_DEVICES MUJOCO_GL || true
  bash "$1/03-cbuild.sh" "$2" > "$1/stage3.log" 2>&1
  bash "$1/04-corpus.sh" "$2" > "$1/stage4.log" 2>&1
  bash "$1/05-rust.sh" "$2" > "$1/stage5.log" 2>&1
' bash "$D" "$RUN" > "$D/driver.log" 2>&1 < /dev/null &
echo $! > "$D/driver.pid"
echo "driver pid $(cat "$D/driver.pid")"
echo "tail -f $D/stage3.log"
```

Stage 3 writes a new `build-$STAMP` and `provenance/stage3-ok`. Its receipt has `compiler_rt_path` and `compiler_rt_sha256`. Stage 4 exit 0 is ARM-versus-x86 full parity. Exit 1 is a finished divergence: `provenance/stage4-x86-verdict.txt` names the first field, and `stage4-ok` is already written. Exit 3 is still a compile or dump failure.

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
