#!/usr/bin/env bash
# Uses a stage-1/2 tree produced by 01-materialize.sh and 02-verify.sh.
# Does not copy README.md into scripts/. Platform mocks apply only to the recovery process.
set -euo pipefail
export LC_ALL=C
if [[ "${1:-}" == "--inside" ]]; then
  INSIDE=1
  REAL_RUN="${2:?stage1 run directory}"
else
  INSIDE=0
  REAL_RUN="${1:?stage1 run directory}"
fi
RECEIPT=/workspace/scripts/arm_oracle/direct-pull/direct-pull-receipt.sh
RECOVER=/workspace/scripts/arm_oracle/direct-pull/recover-stage3.sh
MATERIALIZE=/workspace/scripts/arm_oracle/direct-pull/materialize-schedules.sh
MULQUAT=/workspace/scripts/arm_oracle/direct-pull/probes/mulquat.c
if [[ ! -d /workspace/scripts/arm_oracle/direct-pull ]]; then
  RECEIPT=/agent/repos/ferromotion/scripts/arm_oracle/direct-pull/direct-pull-receipt.sh
  RECOVER=/agent/repos/ferromotion/scripts/arm_oracle/direct-pull/recover-stage3.sh
  MATERIALIZE=/agent/repos/ferromotion/scripts/arm_oracle/direct-pull/materialize-schedules.sh
  MULQUAT=/agent/repos/ferromotion/scripts/arm_oracle/direct-pull/probes/mulquat.c
fi
[[ -f "$REAL_RUN/provenance/stage2-ok" ]] || { echo "need a verified stage1 tree" >&2; exit 1; }
[[ ! -f "$REAL_RUN/scripts/README.md" ]] || { echo "stage1 tree unexpectedly has scripts/README.md" >&2; exit 1; }

if [[ "$INSIDE" -eq 0 ]]; then
rm -f "$REAL_RUN/schedules/x86-native.txt" "$REAL_RUN/schedules/x86-step.txt"
echo "=== preflight before schedules ==="
set +e
bash "$RECEIPT" --preflight "$REAL_RUN" > /tmp/preflight-before.txt 2>/tmp/preflight-before.err
echo "exit:$?" | tee -a /tmp/preflight-before.txt
set -e
grep -F -q "schedules/x86-native.txt" /tmp/preflight-before.txt
grep -F -q "probes/mulquat.c" /tmp/preflight-before.txt
grep -F -q "scripts/README.md" /tmp/preflight-before.txt && { echo "preflight required the archive README" >&2; exit 1; }
echo "preflight listed schedules and mulquat, not scripts/README.md"

bash "$MATERIALIZE" "$REAL_RUN"
echo "=== reproduce archive receipt README refusal ==="
# These three files are what 41ed70b wrote before calling the archive receipt.
# They are removed again. scripts/README.md is never created.
note="$REAL_RUN/ferromotion/scripts/arm_oracle/OPTIONAL-RUSTUP.md"
mkdir -p "$REAL_RUN/rust"
cp -f "$note" "$REAL_RUN/rust/OPTIONAL-RUSTUP.md"
printf '%s\n' e0557dd109151a02702dfb262ea0a8c2b9dd07db > "$REAL_RUN/provenance/ferromotion-commit.txt"
printf '%s\n' "mujoco 123347c0eeab7e13c8da0828ab593bbd95bcf335" > "$REAL_RUN/provenance/REVISIONS"
libdir=/tmp/archive-receipt-libs
rm -rf "$libdir" /tmp/archive-receipt-out
mkdir -p "$libdir" /tmp/archive-receipt-out
for n in libmujoco.so.3.13.0 libactuator.so libelasticity.so libsensor.so libsdf_plugin.so; do
  printf 'x\n' > "$libdir/$n"
done
set +e
bash "$REAL_RUN/scripts/write-receipt.sh" /tmp/archive-receipt-out "$libdir" build > /tmp/archive-receipt.out 2>/tmp/archive-receipt.err
arc=$?
set -e
echo "archive_receipt_exit $arc"
cat /tmp/archive-receipt.err
[[ "$arc" -eq 2 ]]
grep -F -q "scripts/README.md" /tmp/archive-receipt.err
[[ ! -f /tmp/archive-receipt-out/RECEIPT.txt ]]
rm -f "$REAL_RUN/rust/OPTIONAL-RUSTUP.md" "$REAL_RUN/provenance/ferromotion-commit.txt" "$REAL_RUN/provenance/REVISIONS"
rmdir "$REAL_RUN/rust" 2>/dev/null || true
echo "reproduced README refusal and removed the stand-in files"

echo "=== preflight after real schedules, still no README copy ==="
bash "$RECEIPT" --preflight "$REAL_RUN" | tee /tmp/preflight-after.txt
[[ ! -f "$REAL_RUN/scripts/README.md" ]]
echo "direct-pull preflight ok"

echo "=== drop-in receipt on the real tree, no scripts/README.md ==="
saved="$(mktemp)"
cp -f "$REAL_RUN/scripts/write-receipt.sh" "$saved"
cp -f "$RECEIPT" "$REAL_RUN/scripts/write-receipt.sh"
drop_out=/tmp/dp-receipt-out
drop_lib=/tmp/dp-receipt-libs
rm -rf "$drop_out" "$drop_lib"
mkdir -p "$drop_out" "$drop_lib"
for n in libmujoco.so.3.13.0 libactuator.so libelasticity.so libsensor.so libsdf_plugin.so; do
  printf 'drop-%s\n' "$n" > "$drop_lib/$n"
done
printf '%s\n' 'flags -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a' > "$drop_out/identity.txt"
bash "$REAL_RUN/scripts/write-receipt.sh" "$drop_out" "$drop_lib" direct-pull-build > /tmp/drop-receipt.out 2>/tmp/drop-receipt.err
grep -F -q "receipt_schema direct-pull-v1" "$drop_out/RECEIPT.txt"
grep -F -q "archive_manifest_rebuilt no" "$drop_out/RECEIPT.txt"
grep -F -q "ferromotion/scripts/arm_oracle/README.md" "$drop_out/RECEIPT.txt"
[[ ! -f "$REAL_RUN/scripts/README.md" ]]
(cd "$drop_out" && sha256sum -c RECEIPT.sha256)
cp -f "$saved" "$REAL_RUN/scripts/write-receipt.sh"
rm -f "$saved"
echo "drop-in receipt bound the checkout README, not a scripts/README.md placeholder"
fi

if [[ "$INSIDE" -eq 0 ]]; then
  exec unshare --user --map-root-user --mount bash "$0" --inside "$REAL_RUN"
fi

MOCK=/tmp/recover-platform-mock
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
mount -t tmpfs tmpfs /home
mkdir -p /home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335
mkdir -p /home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
mkdir -p /home/cbio/ferromotion-direct-pull-ecfccc6c-zKqin1Nw
mount --bind "$REAL_RUN" /home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
BUILD=/home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/build-20261107T131346Z
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
rm -f "$RUN/provenance/stage3-ok" "$RUN/provenance/stage3-recovery.txt" "$RUN/provenance/build-dir.txt" "$RUN/provenance/stage3.log" "$RUN/provenance/stage4-probes.log"
mkdir -p "$BUILD/cmake/lib"
printf 'not-the-arm-library\n' >"$BUILD/cmake/lib/libmujoco.so.3.13.0"
printf 'synthetic-actuator\n' >"$BUILD/cmake/lib/libactuator.so"
printf 'synthetic-elasticity\n' >"$BUILD/cmake/lib/libelasticity.so"
printf 'synthetic-sensor\n' >"$BUILD/cmake/lib/libsensor.so"
printf 'synthetic-sdf\n' >"$BUILD/cmake/lib/libsdf_plugin.so"
ln -s libmujoco.so.3.13.0 "$BUILD/cmake/lib/libmujoco.so"
cat >"$BUILD/identity.txt" <<EOF
hostname thor2
uname Linux aarch64
clang Ubuntu clang version 18.1.3 (fixture)
cmake cmake version 3.28.3
ninja 1.11.1
flags -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a
jobs 2
work $BUILD
rlimit_cpu_seconds_per_process 14400
rlimit_as_kib_per_process 50331648
wall_timeout_seconds 14400
EOF
printf '%s\n' 'build -ffp-contract=off -march=armv8-a -O3' >"$BUILD/cmake/build.ninja"
REAL_LIB=cefada9b5165a787e2c2695c9db172993c922f5533e19ed8108a181c53f85845
printf '%s  %s\n' "$REAL_LIB" "$BUILD/cmake/lib/libmujoco.so.3.13.0" >"$BUILD/libmujoco.so.3.13.0.sha256"
printf 'work %s\n%s  log\nrefusing: receipt cannot bind %s/schedules/x86-native.txt\n' \
  "$BUILD" "$REAL_LIB" "$RUN" | tee "$RUN/provenance/stage3.log" > /home/cbio/ferromotion-direct-pull-ecfccc6c-zKqin1Nw/stage3.log
export PATH="$MOCK:/usr/bin:/bin"
export HOME=/tmp
set +e
bash "$RECOVER" >/tmp/case-hash.out 2>/tmp/case-hash.err
echo "hash_refusal:$?"
set -e
grep -F -q "undefined __muloti4" /tmp/case-hash.err
[[ ! -f "$RUN/provenance/stage3-ok" ]]
[[ ! -f "$RUN/scripts/README.md" ]]
echo "hash refusal on the real tree, no marker, no README invented"

# Success uses the same script with only the five expected library hashes replaced.
lib_sha="$(sha256sum "$BUILD/cmake/lib/libmujoco.so.3.13.0" | awk '{print $1}')"
act_sha="$(sha256sum "$BUILD/cmake/lib/libactuator.so" | awk '{print $1}')"
ela_sha="$(sha256sum "$BUILD/cmake/lib/libelasticity.so" | awk '{print $1}')"
sen_sha="$(sha256sum "$BUILD/cmake/lib/libsensor.so" | awk '{print $1}')"
sdf_sha="$(sha256sum "$BUILD/cmake/lib/libsdf_plugin.so" | awk '{print $1}')"
mkdir -p /tmp/recover-pair
cp -f "$(dirname "$RECOVER")/direct-pull-receipt.sh" /tmp/recover-pair/direct-pull-receipt.sh
sed \
  -e "s/^THOR2_LIB_MUJOCO=.*/THOR2_LIB_MUJOCO=${lib_sha}/" \
  -e "s/^THOR2_LIB_ACTUATOR=.*/THOR2_LIB_ACTUATOR=${act_sha}/" \
  -e "s/^THOR2_LIB_ELASTICITY=.*/THOR2_LIB_ELASTICITY=${ela_sha}/" \
  -e "s/^THOR2_LIB_SENSOR=.*/THOR2_LIB_SENSOR=${sen_sha}/" \
  -e "s/^THOR2_LIB_SDF=.*/THOR2_LIB_SDF=${sdf_sha}/" \
  "$RECOVER" > /tmp/recover-pair/recover-stage3.sh
printf '%s  %s\n' "$lib_sha" "$BUILD/cmake/lib/libmujoco.so.3.13.0" >"$BUILD/libmujoco.so.3.13.0.sha256"
printf 'work %s\n%s  log\nrefusing: receipt cannot bind %s/schedules/x86-native.txt\n' \
  "$BUILD" "$lib_sha" "$RUN" | tee "$RUN/provenance/stage3.log" > /home/cbio/ferromotion-direct-pull-ecfccc6c-zKqin1Nw/stage3.log
bash /tmp/recover-pair/recover-stage3.sh
grep -F -q "receipt_schema direct-pull-v1" "$BUILD/RECEIPT.txt"
grep -F -q "archive_manifest_rebuilt no" "$BUILD/RECEIPT.txt"
grep -F -q "source_head ferromotion e0557dd109151a02702dfb262ea0a8c2b9dd07db" "$BUILD/RECEIPT.txt"
grep -F -q "ferromotion/scripts/arm_oracle/README.md" "$BUILD/RECEIPT.txt"
[[ ! -f "$RUN/scripts/README.md" ]]
(cd "$BUILD" && sha256sum -c RECEIPT.sha256)
cp -f "$BUILD/RECEIPT.txt" /tmp/receipt-before.txt
set +e
bash /tmp/recover-pair/recover-stage3.sh >/tmp/case-again.out 2>/tmp/case-again.err
again=$?
set -e
[[ "$again" -eq 2 ]]
grep -F -q "stage3-ok already exists" /tmp/case-again.err
cmp -s /tmp/receipt-before.txt "$BUILD/RECEIPT.txt"
echo "recovery success and idempotent refusal"

export PATH="/usr/bin:/bin"
set +e
bash "$(dirname "$RECOVER")/04-corpus.sh" "$RUN" >/tmp/stage4.out 2>/tmp/stage4.err
s4=$?
set -e
echo "stage4_exit $s4"
cat /tmp/stage4.err
[[ "$s4" -eq 2 ]]
grep -F -q "Linux aarch64 only" /tmp/stage4.err
echo "REAL_TREE_RECEIPT_OK"
