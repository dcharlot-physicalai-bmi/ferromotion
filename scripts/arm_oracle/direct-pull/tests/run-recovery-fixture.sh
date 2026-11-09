#!/usr/bin/env bash
# Cloud-only harness. It does not change recover-stage3.sh guards.
# A PATH mock makes uname and hostname report the thor2 platform while those ifs still run.
set -euo pipefail
export LC_ALL=C
if [[ "$(id -u)" -ne 0 ]]; then
  exec unshare --user --map-root-user --mount bash "$0" "$@"
fi

REAL_SCRIPT=/workspace/scripts/arm_oracle/direct-pull/recover-stage3.sh
REAL_PINS=/workspace/scripts/arm_oracle/direct-pull/pins.sh
MATERIALIZE=/workspace/scripts/arm_oracle/direct-pull/materialize-schedules.sh
SRC=/tmp/direct-pull-test/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T130749Z
FIX=/tmp/recover-fixture-root
MOCK=/tmp/recover-platform-mock
RUN_REL=ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
BUILD_REL=ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/build-20261107T131346Z
BOOT_REL=ferromotion-direct-pull-ecfccc6c-zKqin1Nw
REAL_LIB=cefada9b5165a787e2c2695c9db172993c922f5533e19ed8108a181c53f85845

[[ -f "$REAL_SCRIPT" && -d "$SRC/ferromotion/.git" ]] || {
  echo "fixture prerequisites missing" >&2
  exit 1
}

mkdir -p "$MOCK"
cat >"$MOCK/uname" <<'EOF'
#!/bin/sh
case "$1" in
  -s) echo Linux ;;
  -m) echo aarch64 ;;
  -r) echo fixture ;;
  *) echo Linux ;;
esac
EOF
cat >"$MOCK/hostname" <<'EOF'
#!/bin/sh
echo thor2
EOF
chmod +x "$MOCK/uname" "$MOCK/hostname"
export PATH="$MOCK:/usr/bin:/bin"
export HOME=/tmp
unset GH_TOKEN GITHUB_TOKEN || true

# The old one-line local dies under nounset. The split form used by the script does not.
bash -c 'set -euo pipefail; f(){ local name="$1" path="/x/$name"; }; f lib' >/tmp/old-local.out 2>/tmp/old-local.err || OLD_RC=$?
OLD_RC=${OLD_RC:-0}
[[ "$OLD_RC" -ne 0 ]] || { echo "old local form unexpectedly succeeded" >&2; exit 1; }
grep -q 'name: unbound variable' /tmp/old-local.err
bash -c 'set -euo pipefail; f(){ local name="$1"; local path="/x/$name"; echo "$path"; }; f libmujoco.so.3.13.0' | grep -q '/x/libmujoco.so.3.13.0'

mkdir -p "$FIX"
mount -t tmpfs tmpfs /home
mkdir -p /home/cbio
mount --bind "$FIX" /home/cbio

prepare() {
  local mode="$1"
  rm -rf /home/cbio/ferromotion-oracle /home/cbio/ferromotion-direct-pull-ecfccc6c-zKqin1Nw
  mkdir -p "/home/cbio/$RUN_REL/provenance" "/home/cbio/$RUN_REL/probes" "/home/cbio/$RUN_REL/scripts" \
    "/home/cbio/$RUN_REL/schedules" "/home/cbio/$BUILD_REL/cmake/lib" "/home/cbio/$BOOT_REL"
  local run="/home/cbio/$RUN_REL"
  local build="/home/cbio/$BUILD_REL"
  mkdir -p "$run/deps" "$run/src"
  ln -sfn "$SRC/src/mujoco" "$run/src/mujoco"
  ln -sfn "$SRC/deps/ccd" "$run/deps/ccd"
  ln -sfn "$SRC/deps/qhull" "$run/deps/qhull"
  ln -sfn "$SRC/deps/tinyxml2" "$run/deps/tinyxml2"
  ln -sfn "$SRC/deps/tinyobjloader" "$run/deps/tinyobjloader"
  ln -sfn "$SRC/deps/lodepng" "$run/deps/lodepng"
  ln -sfn "$SRC/deps/miniz" "$run/deps/miniz"
  ln -sfn "$SRC/deps/marchingcubecpp" "$run/deps/marchingcubecpp"
  ln -sfn "$SRC/menagerie" "$run/menagerie"
  ln -sfn "$SRC/ferromotion" "$run/ferromotion"
  cp -f "$SRC/ferromotion/scripts/arm_oracle/"*.sh "$run/scripts/"
  cp -f "$SRC/ferromotion/scripts/arm_oracle/README.md" "$run/scripts/README.md"
  cp -f "$SRC/ferromotion/scripts/arm_oracle/OPTIONAL-RUSTUP.md" "$run/scripts/OPTIONAL-RUSTUP.md"
  cp -f "$SRC/ferromotion/scripts/arm_oracle/oracle_dump.c" "$run/probes/oracle_dump.c"
  cp -f "$SRC/ferromotion/scripts/arm_oracle/pyhex.h" "$run/probes/pyhex.h"
  cp -f "$SRC/ferromotion/scripts/arm_oracle/hexfloat_selftest.c" "$run/probes/hexfloat_selftest.c"
  if [[ ! -f /tmp/sched-cache/schedules/x86-native.txt ]]; then
    mkdir -p /tmp/sched-cache
    bash "$MATERIALIZE" /tmp/sched-cache
  fi
  cp -f /tmp/sched-cache/schedules/x86-native.txt /tmp/sched-cache/schedules/x86-step.txt "$run/schedules/"
  printf 'synthetic-mujoco\n' >"$build/cmake/lib/libmujoco.so.3.13.0"
  printf 'synthetic-actuator\n' >"$build/cmake/lib/libactuator.so"
  printf 'synthetic-elasticity\n' >"$build/cmake/lib/libelasticity.so"
  printf 'synthetic-sensor\n' >"$build/cmake/lib/libsensor.so"
  printf 'synthetic-sdf\n' >"$build/cmake/lib/libsdf_plugin.so"
  ln -s libmujoco.so.3.13.0 "$build/cmake/lib/libmujoco.so"
  cat >"$build/identity.txt" <<EOF
hostname thor2
uname Linux aarch64
clang Ubuntu clang version 18.1.3 (fixture)
cmake cmake version 3.28.3
ninja 1.11.1
flags -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a
jobs 2
work $build
rlimit_cpu_seconds_per_process 14400
rlimit_as_kib_per_process 50331648
wall_timeout_seconds 14400
EOF
  printf '%s\n' 'build -ffp-contract=off -march=armv8-a -O3' >"$build/cmake/build.ninja"
  printf '%s\n' "ferromotion e0557dd109151a02702dfb262ea0a8c2b9dd07db tree pinned" >"$run/provenance/stage2-ok"
  if [[ "$mode" == "success" ]]; then
    local h
    h="$(sha256sum "$build/cmake/lib/libmujoco.so.3.13.0" | awk '{print $1}')"
    printf '%s  %s\n' "$h" "$build/cmake/lib/libmujoco.so.3.13.0" >"$build/libmujoco.so.3.13.0.sha256"
    printf 'work %s\n%s  synthetic\nrefusing: receipt cannot bind %s/schedules/x86-native.txt\n' \
      "$build" "$h" "$run" >"$run/provenance/stage3.log"
    cp -f "$run/provenance/stage3.log" "/home/cbio/$BOOT_REL/stage3.log"
  else
    printf '%s  %s\n' "$REAL_LIB" "$build/cmake/lib/libmujoco.so.3.13.0" >"$build/libmujoco.so.3.13.0.sha256"
    printf 'work %s\n%s  synthetic\nrefusing: receipt cannot bind %s/schedules/x86-native.txt\n' \
      "$build" "$REAL_LIB" "$run" >"$run/provenance/stage3.log"
    cp -f "$run/provenance/stage3.log" "/home/cbio/$BOOT_REL/stage3.log"
  fi
}

run_real() {
  set +e
  bash "$REAL_SCRIPT" >"$1.out" 2>"$1.err"
  local rc=$?
  set -e
  echo "$rc"
}

echo "case hash-refusal"
prepare hash
rc="$(run_real /tmp/case-hash)"
[[ "$rc" -eq 2 ]]
grep -F -q "undefined __muloti4" /tmp/case-hash.err
[[ ! -e "/home/cbio/$RUN_REL/provenance/stage3-ok" ]]
echo "hash-refusal exit $rc"

echo "case identity-flags"
prepare hash
sed -i '/^flags /d' "/home/cbio/$BUILD_REL/identity.txt"
rc="$(run_real /tmp/case-flags)"
[[ "$rc" -eq 2 ]]
grep -F -q "identity flags" /tmp/case-flags.err
[[ ! -e "/home/cbio/$RUN_REL/provenance/stage3-ok" ]]
echo "identity-flags exit $rc"

echo "case ninja-flags"
prepare hash
printf '%s\n' 'build -O3' >"/home/cbio/$BUILD_REL/cmake/build.ninja"
rc="$(run_real /tmp/case-ninja)"
[[ "$rc" -eq 2 ]]
grep -F -q "build.ninja has no -ffp-contract=off" /tmp/case-ninja.err
echo "ninja-flags exit $rc"

echo "case log-refusal"
prepare hash
printf 'work %s\n%s\n' "/home/cbio/$BUILD_REL" "$REAL_LIB" >"/home/cbio/$RUN_REL/provenance/stage3.log"
cp -f "/home/cbio/$RUN_REL/provenance/stage3.log" "/home/cbio/$BOOT_REL/stage3.log"
rc="$(run_real /tmp/case-log)"
[[ "$rc" -eq 2 ]]
grep -F -q "neither stage3 log records the receipt refusal" /tmp/case-log.err
echo "log-refusal exit $rc"

echo "case fast-math"
prepare hash
printf '%s\n' 'build -ffp-contract=off -march=armv8-a -ffast-math' >"/home/cbio/$BUILD_REL/cmake/build.ninja"
rc="$(run_real /tmp/case-fast)"
[[ "$rc" -eq 2 ]]
grep -F -q "build.ninja contains -ffast-math" /tmp/case-fast.err
echo "fast-math exit $rc"

echo "case success and idempotent"
prepare success
echo "prepared success tree"
lib_sha="$(sha256sum "/home/cbio/$BUILD_REL/cmake/lib/libmujoco.so.3.13.0" | awk '{print $1}')"
act_sha="$(sha256sum "/home/cbio/$BUILD_REL/cmake/lib/libactuator.so" | awk '{print $1}')"
ela_sha="$(sha256sum "/home/cbio/$BUILD_REL/cmake/lib/libelasticity.so" | awk '{print $1}')"
sen_sha="$(sha256sum "/home/cbio/$BUILD_REL/cmake/lib/libsensor.so" | awk '{print $1}')"
sdf_sha="$(sha256sum "/home/cbio/$BUILD_REL/cmake/lib/libsdf_plugin.so" | awk '{print $1}')"
echo "hashes $lib_sha"
success_script=/tmp/recover-success.sh
sed \
  -e "s/^THOR2_LIB_MUJOCO=.*/THOR2_LIB_MUJOCO=${lib_sha}/" \
  -e "s/^THOR2_LIB_ACTUATOR=.*/THOR2_LIB_ACTUATOR=${act_sha}/" \
  -e "s/^THOR2_LIB_ELASTICITY=.*/THOR2_LIB_ELASTICITY=${ela_sha}/" \
  -e "s/^THOR2_LIB_SENSOR=.*/THOR2_LIB_SENSOR=${sen_sha}/" \
  -e "s/^THOR2_LIB_SDF=.*/THOR2_LIB_SDF=${sdf_sha}/" \
  "$REAL_SCRIPT" >"$success_script"
# The only intended difference is the five synthetic library hashes.
diff_count="$( (diff -u "$REAL_SCRIPT" "$success_script" || true) | grep -E '^[+-]THOR2_LIB_' | wc -l | tr -d ' ')"
echo "diff_count $diff_count"
[[ "$diff_count" -eq 10 ]]
# Production pins stay the Thor library hashes.
grep -F -q "THOR2_LIB_MUJOCO=$REAL_LIB" "$REAL_PINS"
grep -F -q "THOR2_LIB_MUJOCO=$REAL_LIB" "$REAL_SCRIPT"
set +e
bash "$success_script" >/tmp/case-ok.out 2>/tmp/case-ok.err
ok_rc=$?
set -e
[[ "$ok_rc" -eq 0 ]]
grep -F -q "stage3-ok /home/cbio/$BUILD_REL" "/home/cbio/$RUN_REL/provenance/stage3-ok"
grep -F -q "oracle_class arm64_linux_cpu_contract_off" "/home/cbio/$BUILD_REL/RECEIPT.txt"
grep -F -q "$lib_sha" "/home/cbio/$BUILD_REL/RECEIPT.txt"
grep -F -q "$act_sha" "/home/cbio/$BUILD_REL/RECEIPT.txt"
grep -F -q "schedules/x86-native.txt" "/home/cbio/$BUILD_REL/RECEIPT.txt"
(cd "/home/cbio/$BUILD_REL" && sha256sum -c RECEIPT.sha256)
cp -f "/home/cbio/$BUILD_REL/RECEIPT.txt" /tmp/receipt-before.txt
cp -f "/home/cbio/$RUN_REL/provenance/stage3-ok" /tmp/stage3-ok-before.txt
set +e
bash "$success_script" >/tmp/case-again.out 2>/tmp/case-again.err
again_rc=$?
set -e
[[ "$again_rc" -eq 2 ]]
grep -F -q "stage3-ok already exists" /tmp/case-again.err
cmp -s /tmp/receipt-before.txt "/home/cbio/$BUILD_REL/RECEIPT.txt"
cmp -s /tmp/stage3-ok-before.txt "/home/cbio/$RUN_REL/provenance/stage3-ok"
echo "success exit $ok_rc idempotent exit $again_rc"

# The recovery above ran the e0557dd write-receipt.sh that Thor copied into the run.
# Install the workspace helper into that same scripts directory and run it too.
cp -f /workspace/scripts/arm_oracle/write-receipt.sh "/home/cbio/$RUN_REL/scripts/write-receipt.sh"
cp -f /workspace/scripts/arm_oracle/limits.sh "/home/cbio/$RUN_REL/scripts/limits.sh"
out="/tmp/workspace-receipt-out"
rm -rf "$out"
mkdir -p "$out"
bash "/home/cbio/$RUN_REL/scripts/write-receipt.sh" "$out" "/home/cbio/$BUILD_REL/cmake/lib" build
grep -F -q "$lib_sha" "$out/RECEIPT.txt"
grep -F -q "schedules are bound by the corpus receipt" "$out/RECEIPT.txt"
(cd "$out" && sha256sum -c RECEIPT.sha256)
echo "workspace write-receipt ok"
echo "RECOVERY_FIXTURE_OK"
