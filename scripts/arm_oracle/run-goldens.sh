#!/usr/bin/env bash
# Compare the embedded Darwin arm64 goldens. Does not edit them.
# Exit 0: those three tests passed on this host. That is still not the Linux oracle.
# Exit 1: the tests ran and failed. A failure is not relabeled as a Linux pass.
# Exit 3: cargo did not finish the tests (compile, timeout, or missing summary).
# Exit 2: refused before cargo (host, toolchain, vendor, target directory).
set -euo pipefail
export LC_ALL=C
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
arm_oracle_require_vendor "$ROOT/ferromotion"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/ferromotion-oracle/$REV/goldens-$STAMP"
if [[ -e "$OUT" ]]; then
  echo "refusing: $OUT already exists" >&2
  exit 2
fi
mkdir "$OUT"
arm_oracle_cargo_target "$OUT"
cd "$ROOT/ferromotion"
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" \
  "${ARM_ORACLE_CARGO[@]}" test --locked --jobs 2 -p ferromotion-core --lib -- --test-threads=2 \
  geom_frames_match_mujocos_kinematics_bit_for_bit \
  spatial_tendons_step_as_mujoco_does_bit_for_bit \
  the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit \
  >"$OUT/cargo-test.txt" 2>&1
rc=$?
set -e
{
  echo "cargo_exit $rc"
  echo "cargo_target_dir $CARGO_TARGET_DIR"
  echo "These tests embed the unproven Darwin arrays. A failure is not relabeled as the Linux oracle."
} >"$OUT/result.txt"
if [[ -e "$ROOT/ferromotion/target" ]]; then
  echo "execution: cargo wrote $ROOT/ferromotion/target inside the source tree" >&2
  exit 3
fi
ran="$(grep -Ec 'test [^ ]*(geom_frames_match_mujocos_kinematics_bit_for_bit|spatial_tendons_step_as_mujoco_does_bit_for_bit|the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit) \.\.\. ' "$OUT/cargo-test.txt" || true)"
ran="${ran:-0}"
if [[ "$ran" -ne 3 ]]; then
  echo "execution: $ran of the 3 Darwin golden tests ran (cargo exit $rc)" | tee -a "$OUT/result.txt" >&2
  exit 3
fi
if [[ "$rc" -eq 0 ]]; then
  echo "darwin_golden_tests_exit 0" | tee -a "$OUT/result.txt"
  echo "The three Darwin tests passed on this host. That result is not the Linux aarch64 oracle." | tee -a "$OUT/result.txt"
  export ARM_ORACLE_ROOT="$ROOT"
  "$SCRIPT_DIR/write-receipt.sh" "$OUT" "${ARM_ORACLE_LIBDIR:-/nonexistent}" goldens
  exit 0
fi
if grep -q '^test result: FAILED' "$OUT/cargo-test.txt"; then
  echo "darwin_golden_tests_exit 1" | tee -a "$OUT/result.txt"
  export ARM_ORACLE_ROOT="$ROOT"
  "$SCRIPT_DIR/write-receipt.sh" "$OUT" "${ARM_ORACLE_LIBDIR:-/nonexistent}" goldens
  exit 1
fi
echo "execution: cargo did not finish the Darwin golden tests (cargo exit $rc)" | tee -a "$OUT/result.txt" >&2
exit 3
