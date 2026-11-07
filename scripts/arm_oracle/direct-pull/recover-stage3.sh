#!/usr/bin/env bash
# Record the Thor2 build that already linked, after the schedules exist.
# Does not run cmake or ninja. Does not write stage3-ok unless every check passes
# and the on-host write-receipt.sh then succeeds.
set -euo pipefail
export LC_ALL=C
# Constants are in this file so one download is enough. They match pins.sh.
FERROMOTION_COMMIT=e0557dd109151a02702dfb262ea0a8c2b9dd07db
MUJOCO_COMMIT=123347c0eeab7e13c8da0828ab593bbd95bcf335
MENAGERIE_COMMIT=4d038b3feae26ec82b46a4d586379114012a8ac7
CCD_COMMIT=7931e764a19ef6b21b443376c699bbc9c6d4fba8
QHULL_COMMIT=d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6
TINYXML2_COMMIT=e6caeae85799003f4ca74ff26ee16a789bc2af48
TINYOBJ_COMMIT=2945a967c5303b2c8c14174117c45f3302591150
LODEPNG_COMMIT=17d08dd26cac4d63f43af217ebd70318bfb8189c
MINIZ_COMMIT=d10b03cc73475af673df40f06e5cefd1d5f940d9
MARCHING_COMMIT=f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d
SCHEDULE_NATIVE_SHA256=a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7
SCHEDULE_STEP_SHA256=a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be
OPTIONAL_RUSTUP_SHA256=f12be4b1317237cfd74c4486558717adec0b92ff57dd3074b2e96e600cc1890d
THOR2_RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
THOR2_BUILD=/home/cbio/ferromotion-oracle/123347c0eeab7e13c8da0828ab593bbd95bcf335/build-20261107T131346Z
THOR2_BOOTSTRAP=/home/cbio/ferromotion-direct-pull-ecfccc6c-zKqin1Nw
THOR2_LIB_MUJOCO=cefada9b5165a787e2c2695c9db172993c922f5533e19ed8108a181c53f85845
THOR2_LIB_ACTUATOR=34050ae42ebb3b4f3dabf9e7f68c855ac7cad47cd14fe0a1f0d19822050ed8ba
THOR2_LIB_ELASTICITY=41f7c82839c38b656c9e42cc64979a8043686487a69b80f7689f07d5171390a7
THOR2_LIB_SENSOR=4500baca85f2b3f9b6c2ba7862cd59b8ac2a78c38a24e013f24c2c43cdedd8c5
THOR2_LIB_SDF=f3c586acf687a3a0bd2097b5f5f38671e16051665803d0c75ac4da3aabc070b3
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

sha_of() {
  local file="$1"
  sha256sum "$file" | awk '{print $1}'
}
check_lib() {
  local name="$1"
  local want="$2"
  local path="$LIBDIR/$name"
  local got
  [[ -f "$path" ]] || fail "$path is missing"
  got="$(sha_of "$path")"
  [[ "$got" == "$want" ]] || fail "$name sha256 $got is not $want"
  echo "library $name $got"
}
check_lib libmujoco.so.3.13.0 "$THOR2_LIB_MUJOCO"
check_lib libactuator.so "$THOR2_LIB_ACTUATOR"
check_lib libelasticity.so "$THOR2_LIB_ELASTICITY"
check_lib libsensor.so "$THOR2_LIB_SENSOR"
check_lib libsdf_plugin.so "$THOR2_LIB_SDF"

native="$RUN/schedules/x86-native.txt"
step="$RUN/schedules/x86-step.txt"
[[ -f "$native" && -f "$step" ]] || fail "pinned schedules are not in $RUN/schedules"
[[ "$(sha_of "$native")" == "$SCHEDULE_NATIVE_SHA256" ]] || fail "x86-native.txt sha256 is not the pin"
[[ "$(sha_of "$step")" == "$SCHEDULE_STEP_SHA256" ]] || fail "x86-step.txt sha256 is not the pin"

expect_head() {
  local dir="$1"
  local want="$2"
  local got
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
[[ "$(sha_of "$src_note")" == "$OPTIONAL_RUSTUP_SHA256" ]] || fail "OPTIONAL-RUSTUP.md sha256 is not the pin"
if [[ -f "$rust_note" ]]; then
  [[ "$(sha_of "$rust_note")" == "$OPTIONAL_RUSTUP_SHA256" ]] || fail "run OPTIONAL-RUSTUP.md sha256 is not the pin"
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
