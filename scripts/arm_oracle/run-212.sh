#!/usr/bin/env bash
# CPU-only. Fresh output directory. Enforces 217 XML, 212 loads, and the five named failures.
# Corpus must be a clean checkout of the pin. Probe or count mismatches exit non-zero.
# Exit 0 means the ARM dumps were written and the load set matched. It does not mean they match x86.
set -euo pipefail
export LC_ALL=C
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=limits.sh
source "$SCRIPT_DIR/limits.sh"
arm_oracle_require_thor2 "corpus run is thor2 first"
arm_oracle_apply_limits
if ! ROOT="$(arm_oracle_root "$0")"; then
  exit 2
fi
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
PIN="4d038b3feae26ec82b46a4d586379114012a8ac7"
MENAGERIE="${1:-$ROOT/menagerie}"
BUILD="${2:?build directory from build-thor2.sh}"
if [[ ! -d "$BUILD/cmake/lib" ]]; then
  echo "refusing: $BUILD is not a finished build" >&2
  exit 2
fi
if [[ ! -d "$MENAGERIE/.git" ]]; then
  echo "refusing: $MENAGERIE has no .git. This script does not clone." >&2
  exit 2
fi
got="$(git -C "$MENAGERIE" rev-parse HEAD)"
if [[ "$got" != "$PIN" ]]; then
  echo "refusing: menagerie HEAD $got is not $PIN" >&2
  exit 2
fi
dirty="$(git -C "$MENAGERIE" status --porcelain)"
if [[ -n "$dirty" ]]; then
  echo "refusing: menagerie worktree is not clean" >&2
  printf '%s\n' "$dirty" >&2
  exit 2
fi
mapfile -t xmls < <(find "$MENAGERIE" -mindepth 2 -maxdepth 2 -name '*.xml' -printf '%P\n' | sort)
if [[ "${#xmls[@]}" -ne 217 ]]; then
  echo "refusing: ${#xmls[@]} one-level XML files, pin is 217" >&2
  exit 2
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/ferromotion-oracle/$REV/run-$STAMP"
if [[ -e "$OUT" ]]; then
  echo "refusing: $OUT already exists" >&2
  exit 2
fi
mkdir "$OUT"
SO="$BUILD/cmake/lib/libmujoco.so.3.13.0"
export MUJOCO_PLUGIN_DIR="$BUILD/cmake/lib"
printf '%s\n' "$BUILD/cmake/lib" >"$OUT/libdir.txt"
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" clang -O2 -I"$ROOT/probes" -I"$ROOT/src/mujoco/include" \
  "$ROOT/probes/oracle_dump.c" "$SO" -o "$OUT/oracle_dump" -Wl,-rpath,"$BUILD/cmake/lib" -lm
clang_rc=$?
set -e
if [[ "$clang_rc" -ne 0 ]]; then
  echo "execution: oracle_dump failed to compile ($clang_rc)" >&2
  exit 3
fi
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" "$OUT/oracle_dump" classify "$MENAGERIE" "$OUT/classify.txt"
class_rc=$?
set -e
if [[ "$class_rc" -ne 0 ]]; then
  echo "execution: classify exited $class_rc" >&2
  exit 3
fi
ok=$(grep -c '^ok	' "$OUT/classify.txt" || true)
bad=$(grep -c '^fail	' "$OUT/classify.txt" || true)
if [[ "$ok" -ne 212 || "$bad" -ne 5 ]]; then
  echo "refusing: loadable $ok fail $bad, pin is 212 and 5" >&2
  exit 2
fi
awk -F'\t' '$1=="fail"{print $2}' "$OUT/classify.txt" | sort >"$OUT/failures.txt"
if ! cmp -s "$OUT/failures.txt" "$ROOT/probes/expected-load-failures.txt"; then
  echo "refusing: failure set is not the pinned five fragments" >&2
  diff -u "$ROOT/probes/expected-load-failures.txt" "$OUT/failures.txt" >&2 || true
  exit 2
fi
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" "$OUT/oracle_dump" native "$MENAGERIE" "$ROOT/schedules/x86-native.txt" "$OUT/native.txt"
native_rc=$?
set -e
if [[ "$native_rc" -ne 0 ]]; then
  echo "execution: native replay exited $native_rc" >&2
  exit 3
fi
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" "$OUT/oracle_dump" step "$MENAGERIE" "$ROOT/schedules/x86-step.txt" "$OUT/step.txt" 20
step_rc=$?
set -e
if [[ "$step_rc" -ne 0 ]]; then
  echo "execution: step replay exited $step_rc" >&2
  exit 3
fi
export ARM_ORACLE_ROOT="$ROOT"
"$SCRIPT_DIR/write-receipt.sh" "$OUT" "$BUILD/cmake/lib" corpus
echo "wrote $OUT/native.txt and $OUT/step.txt"
echo "These are the Linux aarch64 CPU oracle. They are not the x86 files and not Darwin goldens."
echo "Exit 0 means the load set matched and both dumps were written. It does not mean the floats match x86."
