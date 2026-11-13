# Direct-pull receipt for the linked Thor2 build

## Oracle-profile re-run

Two ferromotion commits, each in its own sibling directory. Nothing already on disk is overwritten. Cargo stays offline. A missing vendored crate is listed and cargo is not started. The commit and the tree are checked from the codeload tarball. The GitHub API is not used.

The build policy argument is `mujoco-3.13.0-aarch64-contract-off`, bound to libmujoco `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`. `6092a8e537d9eb1264fce1775daf809e15616b03` (tree `99b771db50be4e3f377592212b589aba172d572a`) and `d14e65fac3bcc44fa603dc9c8d87e4ea149a4200` (tree `be573c7badb5195e49e5435038f63acce112de7d`) do not select that id. `install_policy` accepts only `mujoco-3.13.0-x86_64-contract-off` bound to `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`. The script does not substitute that x86 id. It writes a receipt with the commit, the tree, the requested policy, the measured libmujoco sha256, and `counts not_run`, then exits 2 before cargo.

To run the unfused profile against the 9164828b dumps, `install_policy` in the tested commit has to gain a match arm for `mujoco-3.13.0-aarch64-contract-off` that requires artifact sha256 `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9` and sets `mul_quat` Unfused, `rot_vec_quat` Unfused, and `mul_mat_vec3` Contracted. Unknown sites and `mul_mat_vec3` are refused, the same way the x86 arm refuses them.

### thor2

Stage 04 already ran. Do not run stages 01–04 again. thor2 does not take `/tmp/bmi-energy.lock`. CPU only. The runners force 2 jobs.

Replace `REPLACE_WITH_FULL_SHA` with the full SHA of the commit that contains these scripts. Fetch with tokens unset. `sha256sum -c` is the check.

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN FERROMOTION_ORACLE_HOME || true
C=REPLACE_WITH_FULL_SHA
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}"
D="$HOME/ferromotion-oracle/stage5-profile-${C}"
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
fc7b516db873a31beacb8640932063729615a1d8ca9e533097ea51d4d1ce70dc  05-rust-rereun.sh
15cd7716ab4d3821706b6ef4e589309cec7af64810b64d15f96e02e6f6a92f47  direct-pull-receipt.sh
3a0f9b1256317c310ea7e802aa058305d15ce16077036e1a485ae5470b50203d  lib.sh
607ed62911bdc090190d37a7cf7782c74daf996d085b173527ba4d9892836513  pins.sh
61972b22b66be5b5de9b9b887d80505b2d8b3a2b9e77f0845b424b0cba34ba08  runners/compiler-rt.sh
b0c08e9569be14e624caffe17563b4ade2b34c1e33cbe71b601401ef22e49f38  runners/limits.sh
a1e77ec49a52fe82863922a49863e0c66def4c0fcb30621fe4b3b4b60587a11d  runners/run-goldens.sh
3d493666ab92252e8a743492bab6ed71db76469d6b6345f65042d33723a55350  runners/run-rust-212.sh
EOF
chmod +x 05-rust-rereun.sh direct-pull-receipt.sh runners/*.sh
# CPU only. thor2 does not take the energy lock. The runners force 2 jobs.
setsid nohup bash -c '
  set -euo pipefail
  export LC_ALL=C
  export OMP_NUM_THREADS=1
  export CARGO_NET_OFFLINE=true
  unset CUDA_VISIBLE_DEVICES MUJOCO_GL GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
  set +e
  bash "$1/05-rust-rereun.sh" "$2" 6092a8e537d9eb1264fce1775daf809e15616b03 mujoco-3.13.0-aarch64-contract-off > "$1/stage5-6092a8e.log" 2>&1
  echo "6092a8e_exit $?" >> "$1/driver-status.txt"
  bash "$1/05-rust-rereun.sh" "$2" d14e65fac3bcc44fa603dc9c8d87e4ea149a4200 mujoco-3.13.0-aarch64-contract-off > "$1/stage5-d14e65f.log" 2>&1
  echo "d14e65f_exit $?" >> "$1/driver-status.txt"
' bash "$D" "$RUN" > "$D/driver.log" 2>&1 < /dev/null &
echo $! > "$D/driver.pid"
echo "driver pid $(cat "$D/driver.pid")"
echo "tail -f $D/stage5-6092a8e.log"
```

### thor.local

A fresh run. Stages 01 through 04 reproduce the 3.13 aarch64 contract-off library and the 212-model dumps, then stage 05 runs the same two commits. Do not point `FERROMOTION_ORACLE_HOME` at an existing tree. Do not copy credentials. Do not apt, sudo, or otherwise change the system.

Each stage takes `flock -x` (`LOCK_EX`) on `/tmp/bmi-energy.lock`, waits at most 120 seconds, and writes the lock window. Do not flock the driver. A driver-held lock would make the stage wait on itself.

After stage 3, compare `libmujoco.so.3.13.0` with `9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9`. A mismatch is recorded and is not relabeled as the thor2 pin. Stage 4 exit 1 is a finished x86 divergence; stage 05 still runs. CPU only. The build and the runners force 2 jobs.

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN FERROMOTION_ORACLE_HOME || true
C=REPLACE_WITH_FULL_SHA
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}"
D="$HOME/ferromotion-oracle/thor-local-profile-${C}"
mkdir -p "$D/runners" "$D/patches" "$D/fixtures" "$D/probes" "$D/schedules"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/01-materialize.sh" "$BASE/scripts/arm_oracle/direct-pull/01-materialize.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/02-verify.sh" "$BASE/scripts/arm_oracle/direct-pull/02-verify.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/03-cbuild.sh" "$BASE/scripts/arm_oracle/direct-pull/03-cbuild.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/04-corpus.sh" "$BASE/scripts/arm_oracle/direct-pull/04-corpus.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/05-rust-rereun.sh" "$BASE/scripts/arm_oracle/direct-pull/05-rust-rereun.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/compare-arm-x86.sh" "$BASE/scripts/arm_oracle/direct-pull/compare-arm-x86.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/direct-pull-receipt.sh" "$BASE/scripts/arm_oracle/direct-pull/direct-pull-receipt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/lib.sh" "$BASE/scripts/arm_oracle/direct-pull/lib.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/pins.sh" "$BASE/scripts/arm_oracle/direct-pull/pins.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/materialize-schedules.sh" "$BASE/scripts/arm_oracle/direct-pull/materialize-schedules.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/patches/ccd.patch" "$BASE/scripts/arm_oracle/direct-pull/patches/ccd.patch"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/patches/qhull.patch" "$BASE/scripts/arm_oracle/direct-pull/patches/qhull.patch"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/fixtures/x86-contract-off-mulquat.txt" "$BASE/scripts/arm_oracle/direct-pull/fixtures/x86-contract-off-mulquat.txt"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/probes/mulquat.c" "$BASE/scripts/arm_oracle/direct-pull/probes/mulquat.c"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/schedules/x86-native.txt.xz" "$BASE/scripts/arm_oracle/direct-pull/schedules/x86-native.txt.xz"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/schedules/x86-step.txt.xz" "$BASE/scripts/arm_oracle/direct-pull/schedules/x86-step.txt.xz"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/build-thor2.sh" "$BASE/scripts/arm_oracle/build-thor2.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/compiler-rt.sh" "$BASE/scripts/arm_oracle/compiler-rt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/limits.sh" "$BASE/scripts/arm_oracle/limits.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/read-mulquat-fixture.sh" "$BASE/scripts/arm_oracle/read-mulquat-fixture.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-probes.sh" "$BASE/scripts/arm_oracle/run-probes.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-212.sh" "$BASE/scripts/arm_oracle/run-212.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-goldens.sh" "$BASE/scripts/arm_oracle/run-goldens.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-rust-212.sh" "$BASE/scripts/arm_oracle/run-rust-212.sh"
cd "$D"
sha256sum -c - <<'EOF'
07a27a28fa2cc7fd1e4b2c394884b6d52b1999af5cfec30d5d121aa898729054  01-materialize.sh
90ead490a43c51c76d107a2f8e505f3f5e5aaaf2ed6afb1e4217564871fcd09a  02-verify.sh
7e763bc28a5400d27c8c4eea618c7e9bda0f6a50d7f755531a9ba0bc5da8e5c3  03-cbuild.sh
8d6471620522e48c8301e8db7d5f2b6030920d2df87a76df1724237855af15fb  04-corpus.sh
fc7b516db873a31beacb8640932063729615a1d8ca9e533097ea51d4d1ce70dc  05-rust-rereun.sh
5ea7fee85bf55cbe239df15abbb59123a027d275566c5903986acbb32b3a59ff  compare-arm-x86.sh
15cd7716ab4d3821706b6ef4e589309cec7af64810b64d15f96e02e6f6a92f47  direct-pull-receipt.sh
3a0f9b1256317c310ea7e802aa058305d15ce16077036e1a485ae5470b50203d  lib.sh
607ed62911bdc090190d37a7cf7782c74daf996d085b173527ba4d9892836513  pins.sh
ed3a5247ea4817fd9f83842618dac753867b89d580f90d50014abdb55802865f  materialize-schedules.sh
599e5b91ed6603effd57e74e3b008fe24fe777b6a1917360a931afee5e6a91fa  patches/ccd.patch
c15466df72091ce8210bf745eae3f6906930465e00d192fad7caca77477d5939  patches/qhull.patch
358875076e2be19ff4ee24d154a1189dfe5f3928cabf6da5da45f35930ce070f  fixtures/x86-contract-off-mulquat.txt
1bb5bd6fb656623496075e1e8a89470731c7e44fbc6ad9cf7de82361e4b4bb80  probes/mulquat.c
1ed398c90a318fd9a514c7f6c8e14b966427c577f31410c872e90f84df537c4e  schedules/x86-native.txt.xz
237fdfb8ea17eafe3a01f73c3418ecdc03af65b86e711f80d137353a1ccd55ed  schedules/x86-step.txt.xz
6232f8d1221b92d6bfbedb66712c8f0d24be7ae87a56241f98314497454771c5  runners/build-thor2.sh
61972b22b66be5b5de9b9b887d80505b2d8b3a2b9e77f0845b424b0cba34ba08  runners/compiler-rt.sh
b0c08e9569be14e624caffe17563b4ade2b34c1e33cbe71b601401ef22e49f38  runners/limits.sh
64c139aec23de404820a41a3a8ce5f842d6c48a1432e446ab373a721b8c8b1d8  runners/read-mulquat-fixture.sh
e9c5cf157fff0009c2530bf3da9885aa061e668cd458b4b26fb45fffda6bf8c9  runners/run-probes.sh
e2cdf6e2532b599ce77b4fb4337bf602e32fa0ec77765b3b853044043786cacc  runners/run-212.sh
a1e77ec49a52fe82863922a49863e0c66def4c0fcb30621fe4b3b4b60587a11d  runners/run-goldens.sh
3d493666ab92252e8a743492bab6ed71db76469d6b6345f65042d33723a55350  runners/run-rust-212.sh
EOF
chmod +x 01-materialize.sh 02-verify.sh 03-cbuild.sh 04-corpus.sh 05-rust-rereun.sh compare-arm-x86.sh direct-pull-receipt.sh materialize-schedules.sh runners/*.sh
# CPU only. Each stage holds LOCK_EX itself. Do not flock this driver.
setsid nohup bash -c '
  set -euo pipefail
  export LC_ALL=C
  export OMP_NUM_THREADS=1
  unset CUDA_VISIBLE_DEVICES MUJOCO_GL GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN FERROMOTION_ORACLE_HOME || true
  bash "$1/01-materialize.sh" > "$1/stage1.log" 2>&1
  RUN=$(awk "/^run /{print \$2; exit}" "$1/stage1.log")
  echo "$RUN" > "$1/run-directory.txt"
  bash "$1/02-verify.sh" "$RUN" > "$1/stage2.log" 2>&1
  bash "$1/03-cbuild.sh" "$RUN" > "$1/stage3.log" 2>&1
  BUILD=$(tr -d "[:space:]" < "$RUN/provenance/build-dir.txt")
  LIB="$BUILD/cmake/lib/libmujoco.so.3.13.0"
  sha256sum "$LIB" | tee "$1/libmujoco-measured.txt"
  set +e
  echo "9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9  $LIB" | sha256sum -c - > "$1/libmujoco-vs-9164828b.txt" 2>&1
  echo "libmujoco_matches_9164828b_exit $?" | tee -a "$1/libmujoco-vs-9164828b.txt"
  bash "$1/04-corpus.sh" "$RUN" > "$1/stage4.log" 2>&1
  stage4_rc=$?
  if [[ "$stage4_rc" -ne 0 && "$stage4_rc" -ne 1 ]]; then
    echo "stage4_exit $stage4_rc" >> "$1/driver-status.txt"
    exit "$stage4_rc"
  fi
  echo "stage4_exit $stage4_rc" >> "$1/driver-status.txt"
  export CARGO_NET_OFFLINE=true
  bash "$1/05-rust-rereun.sh" "$RUN" 6092a8e537d9eb1264fce1775daf809e15616b03 mujoco-3.13.0-aarch64-contract-off > "$1/stage5-6092a8e.log" 2>&1
  echo "6092a8e_exit $?" >> "$1/driver-status.txt"
  bash "$1/05-rust-rereun.sh" "$RUN" d14e65fac3bcc44fa603dc9c8d87e4ea149a4200 mujoco-3.13.0-aarch64-contract-off > "$1/stage5-d14e65f.log" 2>&1
  echo "d14e65f_exit $?" >> "$1/driver-status.txt"
' bash "$D" > "$D/driver.log" 2>&1 < /dev/null &
echo $! > "$D/driver.pid"
echo "driver pid $(cat "$D/driver.pid")"
echo "tail -f $D/stage1.log"
```

## Stage 5 re-run

The block below is the published-default re-run that already finished. It is not the oracle-profile command.

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
fc7b516db873a31beacb8640932063729615a1d8ca9e533097ea51d4d1ce70dc  05-rust-rereun.sh
15cd7716ab4d3821706b6ef4e589309cec7af64810b64d15f96e02e6f6a92f47  direct-pull-receipt.sh
3a0f9b1256317c310ea7e802aa058305d15ce16077036e1a485ae5470b50203d  lib.sh
607ed62911bdc090190d37a7cf7782c74daf996d085b173527ba4d9892836513  pins.sh
61972b22b66be5b5de9b9b887d80505b2d8b3a2b9e77f0845b424b0cba34ba08  runners/compiler-rt.sh
b0c08e9569be14e624caffe17563b4ade2b34c1e33cbe71b601401ef22e49f38  runners/limits.sh
a1e77ec49a52fe82863922a49863e0c66def4c0fcb30621fe4b3b4b60587a11d  runners/run-goldens.sh
3d493666ab92252e8a743492bab6ed71db76469d6b6345f65042d33723a55350  runners/run-rust-212.sh
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
8d6471620522e48c8301e8db7d5f2b6030920d2df87a76df1724237855af15fb  04-corpus.sh
f01994cc777b45cb7bbb3dbf8fb9086ae07b005d2c6f39d5e2d2b659927d0dca  05-rust.sh
5ea7fee85bf55cbe239df15abbb59123a027d275566c5903986acbb32b3a59ff  compare-arm-x86.sh
15cd7716ab4d3821706b6ef4e589309cec7af64810b64d15f96e02e6f6a92f47  direct-pull-receipt.sh
3a0f9b1256317c310ea7e802aa058305d15ce16077036e1a485ae5470b50203d  lib.sh
607ed62911bdc090190d37a7cf7782c74daf996d085b173527ba4d9892836513  pins.sh
61972b22b66be5b5de9b9b887d80505b2d8b3a2b9e77f0845b424b0cba34ba08  runners/compiler-rt.sh
b0c08e9569be14e624caffe17563b4ade2b34c1e33cbe71b601401ef22e49f38  runners/limits.sh
64c139aec23de404820a41a3a8ce5f842d6c48a1432e446ab373a721b8c8b1d8  runners/read-mulquat-fixture.sh
e9c5cf157fff0009c2530bf3da9885aa061e668cd458b4b26fb45fffda6bf8c9  runners/run-probes.sh
e2cdf6e2532b599ce77b4fb4337bf602e32fa0ec77765b3b853044043786cacc  runners/run-212.sh
a1e77ec49a52fe82863922a49863e0c66def4c0fcb30621fe4b3b4b60587a11d  runners/run-goldens.sh
3d493666ab92252e8a743492bab6ed71db76469d6b6345f65042d33723a55350  runners/run-rust-212.sh
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
