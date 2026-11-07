#!/usr/bin/env bash
# Record the Thor2 build that already linked, after the schedules exist.
# Does not run cmake or ninja. Does not write stage3-ok unless every check passes
# and the on-host write-receipt.sh then succeeds.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
RUN="$THOR2_RUN"
BUILD="$THOR2_BUILD"
BOOT="$THOR2_BOOTSTRAP"
LIBDIR="$BUILD/cmake/lib"
fail() {
  echo "refusing: $*" >&2
  exit 2
}
[[ -d "$RUN" ]] || fail "run directory $RUN is not on this machine"
[[ -d "$BUILD" ]] || fail "build directory $BUILD is not on this machine"
[[ -d "$BOOT" ]] || fail "bootstrap directory $BOOT is not on this machine"
[[ ! -e "$RUN/provenance/stage3-ok" ]] || fail "stage3-ok already exists; not overwriting it"
[[ -f "$RUN/provenance/stage2-ok" ]] || fail "stage2-ok is missing"
grep -F -q "ferromotion $FERROMOTION_COMMIT" "$RUN/provenance/stage2-ok" || fail "stage2-ok is not $FERROMOTION_COMMIT"
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  fail "recovery of this ARM build is Linux aarch64 only (got $(uname -s) $(uname -m))"
fi
if [[ "$(hostname -s)" != "thor2" ]]; then
  fail "recovery of this build is thor2 only (hostname -s is $(hostname -s))"
fi

identity="$BUILD/identity.txt"
[[ -f "$identity" ]] || fail "identity.txt is missing"
grep -F -x -q "uname Linux aarch64" "$identity" || fail "identity uname is not Linux aarch64"
grep -F -q "18.1.3" "$identity" || fail "identity clang is not 18.1.3"
grep -F -q "cmake version 3.28." "$identity" || fail "identity cmake is not 3.28"
grep -F -x -q "flags -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a" "$identity" || fail "identity flags are not contract-off armv8-a"
grep -F -x -q "jobs 2" "$identity" || fail "identity jobs are not 2"
grep -F -x -q "work $BUILD" "$identity" || fail "identity work path is not $BUILD"
grep -F -x -q "rlimit_cpu_seconds_per_process 14400" "$identity" || fail "identity CPU cap is not 14400"
grep -F -x -q "rlimit_as_kib_per_process 50331648" "$identity" || fail "identity address-space cap is not 50331648"
grep -F -q -- "-ffp-contract=off" "$BUILD/cmake/build.ninja" || fail "build.ninja has no -ffp-contract=off"
grep -F -q -- "-march=armv8-a" "$BUILD/cmake/build.ninja" || fail "build.ninja has no -march=armv8-a"
if grep -F -q -- "-ffast-math" "$BUILD/cmake/build.ninja"; then
  fail "build.ninja contains -ffast-math"
fi

run_log="$RUN/provenance/stage3.log"
boot_log="$BOOT/stage3.log"
[[ -f "$run_log" ]] || fail "run stage3.log is missing"
[[ -f "$boot_log" ]] || fail "bootstrap stage3.log is missing"
if grep -F -q "execution: cmake configure failed" "$run_log" "$boot_log"; then
  fail "a stage3 log records cmake configure failure"
fi
if grep -F -q "execution: cmake build failed" "$run_log" "$boot_log"; then
  fail "a stage3 log records cmake build failure"
fi
grep -F -q "$THOR2_LIB_MUJOCO" "$run_log" || fail "run stage3.log does not contain the libmujoco sha256"
grep -F -q "work $BUILD" "$run_log" || fail "run stage3.log does not name $BUILD"
if ! grep -F -q "refusing: receipt cannot bind" "$run_log" "$boot_log"; then
  fail "neither stage3 log records the receipt refusal"
fi
if ! grep -F -q "schedules/x86-native.txt" "$run_log" "$boot_log"; then
  fail "neither stage3 log names the missing native schedule"
fi

sha_file="$BUILD/libmujoco.so.3.13.0.sha256"
[[ -f "$sha_file" ]] || fail "libmujoco sha256 file is missing"
got_file="$(awk 'NR==1 {print $1}' "$sha_file")"
[[ "$got_file" == "$THOR2_LIB_MUJOCO" ]] || fail "sha256 file says $got_file"

check_lib() {
  local name="$1" want="$2" path="$LIBDIR/$name" got
  [[ -f "$path" ]] || fail "$path is missing"
  got="$(direct_pull_sha "$path")"
  [[ "$got" == "$want" ]] || fail "$name sha256 $got is not $want"
  echo "library $name $got"
}
check_lib libmujoco.so.3.13.0 "$THOR2_LIB_MUJOCO"
check_lib libactuator.so "$THOR2_LIB_ACTUATOR"
check_lib libelasticity.so "$THOR2_LIB_ELASTICITY"
check_lib libsensor.so "$THOR2_LIB_SENSOR"
check_lib libsdf_plugin.so "$THOR2_LIB_SDF"

bash "$DIRECT_PULL_DIR/materialize-schedules.sh" "$RUN"

expect_head() {
  local dir="$1" want="$2" got
  [[ -d "$dir/.git" ]] || fail "$dir has no .git"
  got="$(git -C "$dir" rev-parse HEAD)"
  [[ "$got" == "$want" ]] || fail "$dir HEAD $got is not $want"
}
expect_head "$RUN/src/mujoco" "$MUJOCO_COMMIT"
expect_head "$RUN/deps/ccd" "$CCD_COMMIT"
expect_head "$RUN/deps/qhull" "$QHULL_COMMIT"
expect_head "$RUN/deps/tinyxml2" "$TINYXML2_COMMIT"
expect_head "$RUN/deps/tinyobjloader" "$TINYOBJ_COMMIT"
expect_head "$RUN/deps/lodepng" "$LODEPNG_COMMIT"
expect_head "$RUN/deps/miniz" "$MINIZ_COMMIT"
expect_head "$RUN/deps/marchingcubecpp" "$MARCHING_COMMIT"
expect_head "$RUN/menagerie" "$MENAGERIE_COMMIT"
expect_head "$RUN/ferromotion" "$FERROMOTION_COMMIT"

revisions="$RUN/provenance/REVISIONS"
rev_body="$(cat <<EOF
mujoco $MUJOCO_COMMIT
ccd $CCD_COMMIT
qhull $QHULL_COMMIT
tinyxml2 $TINYXML2_COMMIT
tinyobjloader $TINYOBJ_COMMIT
lodepng $LODEPNG_COMMIT
miniz $MINIZ_COMMIT
marchingcubecpp $MARCHING_COMMIT
menagerie $MENAGERIE_COMMIT
ferromotion $FERROMOTION_COMMIT
EOF
)"
if [[ -f "$revisions" ]]; then
  got="$(cat "$revisions")"
  [[ "$got" == "$rev_body" ]] || fail "existing REVISIONS does not match the checked-out commits"
else
  printf '%s\n' "$rev_body" >"$revisions"
fi
commit_file="$RUN/provenance/ferromotion-commit.txt"
if [[ -f "$commit_file" ]]; then
  got="$(tr -d '[:space:]' <"$commit_file")"
  [[ "$got" == "$FERROMOTION_COMMIT" ]] || fail "existing ferromotion-commit.txt is $got"
else
  printf '%s\n' "$FERROMOTION_COMMIT" >"$commit_file"
fi
rust_note="$RUN/rust/OPTIONAL-RUSTUP.md"
src_note="$RUN/ferromotion/scripts/arm_oracle/OPTIONAL-RUSTUP.md"
[[ -f "$src_note" ]] || fail "pinned OPTIONAL-RUSTUP.md is not in the ferromotion checkout"
direct_pull_require_sha "$src_note" "$OPTIONAL_RUSTUP_SHA256"
if [[ -f "$rust_note" ]]; then
  direct_pull_require_sha "$rust_note" "$OPTIONAL_RUSTUP_SHA256"
else
  mkdir -p "$RUN/rust"
  cp -f "$src_note" "$rust_note"
fi

# shellcheck source=../limits.sh
source "$RUN/scripts/limits.sh"
arm_oracle_apply_limits
export ARM_ORACLE_ROOT="$RUN"
"$RUN/scripts/write-receipt.sh" "$BUILD" "$LIBDIR" build
grep -F -q "$THOR2_LIB_MUJOCO" "$BUILD/RECEIPT.txt" || fail "receipt does not contain the libmujoco sha256"
grep -F -q "$THOR2_LIB_ACTUATOR" "$BUILD/RECEIPT.txt" || fail "receipt does not contain libactuator"
grep -F -q "$THOR2_LIB_ELASTICITY" "$BUILD/RECEIPT.txt" || fail "receipt does not contain libelasticity"
grep -F -q "$THOR2_LIB_SENSOR" "$BUILD/RECEIPT.txt" || fail "receipt does not contain libsensor"
grep -F -q "$THOR2_LIB_SDF" "$BUILD/RECEIPT.txt" || fail "receipt does not contain libsdf_plugin"
grep -F -q "oracle_class arm64_linux_cpu_contract_off" "$BUILD/RECEIPT.txt" || fail "receipt class is not the ARM build"
(cd "$BUILD" && sha256sum -c RECEIPT.sha256)
printf '%s\n' "$BUILD" >"$RUN/provenance/build-dir.txt"
{
  echo "stage3-ok $BUILD"
} | tee "$RUN/provenance/stage3-ok"
{
  echo "recovered_existing_binaries yes"
  echo "cmake_rerun no"
  echo "build $BUILD"
  echo "run $RUN"
} | tee "$RUN/provenance/stage3-recovery.txt"
echo "recovery wrote stage3-ok from the existing libraries"
