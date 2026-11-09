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
set -euo pipefail
export LC_ALL=C
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
arm_oracle_require_vendor "$ROOT/ferromotion"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/ferromotion-oracle/$REV/rust212-$STAMP"
if [[ -e "$OUT" ]]; then
  echo "refusing: $OUT already exists" >&2
  exit 2
fi
mkdir "$OUT"
arm_oracle_cargo_target "$OUT"
cd "$ROOT/ferromotion"
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" \
  "${ARM_ORACLE_CARGO[@]}" run --locked --jobs 2 --release -p ferromotion-core --example menagerie_native -- \
  "$ROOT/menagerie" "$RUN/native.txt" >"$OUT/native.txt" 2>&1
native_rc=$?
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" \
  "${ARM_ORACLE_CARGO[@]}" run --locked --jobs 2 --release -p ferromotion-core --example menagerie_native_step -- \
  "$ROOT/menagerie" "$RUN/step.txt" >"$OUT/step.txt" 2>&1
step_rc=$?
set -e
{
  echo "menagerie_native_exit $native_rc"
  echo "menagerie_native_step_exit $step_rc"
  echo "cargo_target_dir $CARGO_TARGET_DIR"
} >"$OUT/result.txt"
if [[ -e "$ROOT/ferromotion/target" ]]; then
  echo "execution: cargo wrote $ROOT/ferromotion/target inside the source tree" >&2
  exit 3
fi
export ARM_ORACLE_ROOT="$ROOT"
if [[ "$native_rc" -ne 0 || "$step_rc" -ne 0 ]]; then
  echo "not_run cargo_failed native=$native_rc step=$step_rc" | tee "$OUT/accept.txt" | tee -a "$OUT/result.txt" >&2
  "$SCRIPT_DIR/write-receipt.sh" "$OUT" "$LIBDIR" rust
  exit 3
fi
set +e
"$SCRIPT_DIR/accept-rust-212.sh" "$OUT/native.txt" "$OUT/step.txt" >"$OUT/accept.txt"
accept_rc=$?
set -e
cat "$OUT/accept.txt" | tee -a "$OUT/result.txt"
"$SCRIPT_DIR/write-receipt.sh" "$OUT" "$LIBDIR" rust
exit "$accept_rc"
