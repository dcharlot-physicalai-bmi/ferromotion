#!/usr/bin/env bash
# Compare ferromotion to the dumps in a run-212 directory.
# menagerie_native and menagerie_native_step do not link libmujoco, so they
# do not need compiler-rt. A build script that linked the .so would.
# The example programs return 0 when they finish printing mismatches.
# This wrapper does not. See scripts/accept-rust-212.sh.
# Exit 2: refused before cargo.
# Exit 3: cargo/timeout failed, or the summary is not 212 models and 20 steps.
# Exit 1: all 212 models and all 20 steps were compared, and something mismatched.
# Exit 0: every tally matched. That is the only acceptance.
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
arm_oracle_require_thor2 "rust comparison is thor2 first"
arm_oracle_apply_limits
if ! ROOT="$(arm_oracle_root "$0")"; then
  exit 2
fi
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
RUN="${1:?run directory from run-212.sh}"
if [[ ! -f "$RUN/native.txt" || ! -f "$RUN/step.txt" ]]; then
  echo "refusing: $RUN is missing native.txt or step.txt" >&2
  exit 2
fi
if [[ ! -f "$RUN/libdir.txt" ]]; then
  echo "refusing: $RUN/libdir.txt is missing, so the receipt cannot bind the five libraries" >&2
  exit 2
fi
LIBDIR="$(tr -d '[:space:]' <"$RUN/libdir.txt")"
if [[ ! -d "$LIBDIR" ]]; then
  echo "refusing: library directory $LIBDIR does not exist" >&2
  exit 2
fi
arm_oracle_select_cargo
FM="${ARM_ORACLE_FERROMOTION:-$ROOT/ferromotion}"
arm_oracle_require_vendor "$FM"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/ferromotion-oracle/$REV/rust212-$STAMP"
if [[ "$(basename "$OUT")" == "rust212-20261109T215708Z" ]]; then
  echo "refusing: $OUT is the dated e0557dd rust212 receipt and is not overwritten" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "refusing: $OUT already exists" >&2
  exit 2
fi
mkdir -p "$OUT"
arm_oracle_write_ferromotion_tested "$FM" "$OUT/ferromotion-tested.txt"
arm_oracle_cargo_target "$OUT"
cd "$FM"
# set +e does not disarm an ERR trap. These cargo statuses are classified below.
trap - ERR
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" \
  "${ARM_ORACLE_CARGO[@]}" run --offline --locked --jobs 2 --release -p ferromotion-core --example menagerie_native -- \
  "$ROOT/menagerie" "$RUN/native.txt" >"$OUT/native.txt" 2>&1
native_rc=$?
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" \
  "${ARM_ORACLE_CARGO[@]}" run --offline --locked --jobs 2 --release -p ferromotion-core --example menagerie_native_step -- \
  "$ROOT/menagerie" "$RUN/step.txt" >"$OUT/step.txt" 2>&1
step_rc=$?
set -e
trap refuse_unexpected ERR
{
  echo "menagerie_native_exit $native_rc"
  echo "menagerie_native_step_exit $step_rc"
  echo "cargo_target_dir $CARGO_TARGET_DIR"
} >"$OUT/result.txt"
if [[ -e "$FM/target" || -e "$ROOT/ferromotion/target" ]]; then
  echo "execution: cargo wrote a target directory inside a ferromotion source tree" >&2
  exit 3
fi
export ARM_ORACLE_ROOT="$ROOT"
stage3="$(cd "$LIBDIR/../.." && pwd)"
arm_oracle_write_role_identity "$OUT/identity.txt" rust "$stage3"
if [[ "$native_rc" -ne 0 || "$step_rc" -ne 0 ]]; then
  echo "not_run cargo_failed native=$native_rc step=$step_rc" | tee "$OUT/accept.txt" | tee -a "$OUT/result.txt" >&2
  "$SCRIPT_DIR/write-receipt.sh" "$OUT" "$LIBDIR" rust
  exit 3
fi
trap - ERR
set +e
"$SCRIPT_DIR/accept-rust-212.sh" "$OUT/native.txt" "$OUT/step.txt" >"$OUT/accept.txt"
accept_rc=$?
set -e
trap refuse_unexpected ERR
tee -a "$OUT/result.txt" <"$OUT/accept.txt"
"$SCRIPT_DIR/write-receipt.sh" "$OUT" "$LIBDIR" rust
if [[ "$accept_rc" -eq 0 || "$accept_rc" -eq 1 ]]; then
  exit "$accept_rc"
fi
echo "refusing: rust comparison exit $accept_rc" >&2
exit 2
