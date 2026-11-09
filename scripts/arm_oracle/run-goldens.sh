#!/usr/bin/env bash
# Compare the embedded Darwin arm64 goldens. Does not edit them.
# Exit 0: exactly those three tests ran and passed. That is still not the Linux oracle.
# Exit 1: exactly those three tests ran and failed. A finished Darwin mismatch.
# Exit 2: refused, or the cargo result is not those three tests.
# goldens-20261109T223345Z is left in place. A later run uses a new directory.
set -Eeuo pipefail
export LC_ALL=C
refuse_unexpected() {
  local rc=$?
  echo "refusing: unexpected abort at line ${BASH_LINENO[0]} status $rc" >&2
  exit 2
}
trap refuse_unexpected ERR
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=limits.sh
source "$SCRIPT_DIR/limits.sh"
arm_oracle_require_thor2 "golden comparison is thor2 first"
arm_oracle_apply_limits
if ! ROOT="$(arm_oracle_root "$0")"; then
  exit 2
fi
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
arm_oracle_select_cargo
FM="${ARM_ORACLE_FERROMOTION:-$ROOT/ferromotion}"
arm_oracle_require_vendor "$FM"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
kept="$HOME/ferromotion-oracle/$REV/goldens-20261109T223345Z"
if [[ -d "$kept" ]]; then
  echo "left_untouched $kept"
fi
OUT="$HOME/ferromotion-oracle/$REV/goldens-$STAMP"
if [[ "$(basename "$OUT")" == "goldens-20261109T223345Z" || -e "$OUT" ]]; then
  echo "refusing: $OUT already exists and is not overwritten" >&2
  exit 2
fi
mkdir -p "$OUT"
arm_oracle_write_ferromotion_tested "$FM" "$OUT/ferromotion-tested.txt"
arm_oracle_cargo_target "$OUT"
cd "$FM"
# set +e does not disarm an ERR trap. A classified cargo failure must not abort here.
trap - ERR
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" \
  "${ARM_ORACLE_CARGO[@]}" test --offline --locked --jobs 2 -p ferromotion-core --lib -- --test-threads=2 \
  geom_frames_match_mujocos_kinematics_bit_for_bit \
  spatial_tendons_step_as_mujoco_does_bit_for_bit \
  the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit \
  >"$OUT/cargo-test.txt" 2>&1
rc=$?
set -e
trap refuse_unexpected ERR
{
  echo "cargo_exit $rc"
  echo "cargo_target_dir $CARGO_TARGET_DIR"
  echo "These tests embed the unproven Darwin arrays. A failure is not relabeled as the Linux oracle."
} >"$OUT/result.txt"
if [[ -e "$FM/target" || -e "$ROOT/ferromotion/target" ]]; then
  echo "refusing: cargo wrote a target directory inside a ferromotion source tree" >&2
  exit 2
fi
names=(
  geom_frames_match_mujocos_kinematics_bit_for_bit
  spatial_tendons_step_as_mujoco_does_bit_for_bit
  the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit
)
ran=0
failed=0
passed=0
for name in "${names[@]}"; do
  if grep -E -q "^test [^ ]*${name} \\.\\.\\. FAILED$" "$OUT/cargo-test.txt"; then
    ran=$((ran + 1))
    failed=$((failed + 1))
  elif grep -E -q "^test [^ ]*${name} \\.\\.\\. ok$" "$OUT/cargo-test.txt"; then
    ran=$((ran + 1))
    passed=$((passed + 1))
  fi
done
if [[ "$ran" -ne 3 ]]; then
  echo "refusing: $ran of the 3 Darwin golden tests ran (cargo exit $rc)" | tee -a "$OUT/result.txt" >&2
  exit 2
fi
{
  echo "identity_role goldens"
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname $(hostname)"
  echo "uname $(uname -s) $(uname -m)"
  echo "rust_links_libmujoco no"
} >"$OUT/identity.txt"
if [[ "$rc" -eq 0 && "$passed" -eq 3 ]]; then
  echo "darwin_golden_tests_exit 0" | tee -a "$OUT/result.txt"
  echo "The three Darwin tests passed on this host. That result is not the Linux aarch64 oracle." | tee -a "$OUT/result.txt"
  export ARM_ORACLE_ROOT="$ROOT"
  "$SCRIPT_DIR/write-receipt.sh" "$OUT" "${ARM_ORACLE_LIBDIR:-/nonexistent}" goldens
  exit 0
fi
if [[ "$rc" -ne 0 && "$failed" -eq 3 ]] && grep -q '^test result: FAILED' "$OUT/cargo-test.txt"; then
  echo "darwin_golden_tests_exit 1" | tee -a "$OUT/result.txt"
  export ARM_ORACLE_ROOT="$ROOT"
  "$SCRIPT_DIR/write-receipt.sh" "$OUT" "${ARM_ORACLE_LIBDIR:-/nonexistent}" goldens
  exit 1
fi
echo "refusing: Darwin golden tests were not the 3 known failures (cargo exit $rc, failed $failed, passed $passed)" | tee -a "$OUT/result.txt" >&2
exit 2
