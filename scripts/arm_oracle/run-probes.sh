#!/usr/bin/env bash
# Mulquat probe. A mismatch is written and then exits 1. It does not edit a golden.
set -euo pipefail
export LC_ALL=C
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=limits.sh
source "$SCRIPT_DIR/limits.sh"
arm_oracle_require_thor2 "probes run on thor2 first"
arm_oracle_apply_limits
ulimit -t 600
export ARM_ORACLE_WALL_TIMEOUT=600
export ARM_ORACLE_CPU_SECONDS_PER_PROCESS=600
if ! ROOT="$(arm_oracle_root "$0")"; then
  exit 2
fi
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
BUILD="${1:?build directory from build-thor2.sh}"
SO="$BUILD/cmake/lib/libmujoco.so.3.13.0"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/ferromotion-oracle/$REV/probe-$STAMP"
if [[ -e "$OUT" ]]; then
  echo "refusing: $OUT already exists" >&2
  exit 2
fi
mkdir "$OUT"
set +e
# The shared library already contains __muloti4. Appending the builtins archive
# here fails: the symbol is GLOBAL HIDDEN, and ld will not use it to satisfy a DSO.
arm_oracle_timeout 600 clang -O2 -I"$ROOT/src/mujoco/include" "$ROOT/probes/mulquat.c" "$SO" \
  -o "$OUT/mulquat" -Wl,-rpath,"$BUILD/cmake/lib" -lm
clang_rc=$?
set -e
if [[ "$clang_rc" -ne 0 ]]; then
  echo "execution: mulquat failed to compile ($clang_rc)" >&2
  exit 3
fi
set +e
got="$(arm_oracle_timeout 600 "$OUT/mulquat")"
run_rc=$?
set -e
if [[ "$run_rc" -ne 0 ]]; then
  echo "execution: mulquat exited $run_rc" >&2
  exit 3
fi
want="$("$SCRIPT_DIR/read-mulquat-fixture.sh" "$ROOT/fixtures/x86-contract-off-mulquat.txt")"
printf '%s\n' "$got" >"$OUT/mulquat-arm64.txt"
printf '%s\n' "$want" >"$OUT/mulquat-x86.txt"
echo "arm64_mulquat $got"
echo "x86_contract_off $want"
arm_oracle_write_role_identity "$OUT/identity.txt" probe "$BUILD"
if [[ "$got" != "$want" ]]; then
  echo "mulquat_bits differ_from_x86_contract_off" | tee "$OUT/result.txt"
  export ARM_ORACLE_ROOT="$ROOT"
  "$SCRIPT_DIR/write-receipt.sh" "$OUT" "$BUILD/cmake/lib" probe
  echo "recorded in $OUT. This does not edit a golden." >&2
  exit 1
fi
echo "mulquat_bits match_x86_contract_off" | tee "$OUT/result.txt"
export ARM_ORACLE_ROOT="$ROOT"
"$SCRIPT_DIR/write-receipt.sh" "$OUT" "$BUILD/cmake/lib" probe
