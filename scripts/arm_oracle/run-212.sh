#!/usr/bin/env bash
# CPU-only. Fresh output directory. Enforces 217 XML, 212 loads, and the five named failures.
# Corpus must be a clean checkout of the pin. Probe or count mismatches exit non-zero.
set -euo pipefail
export LC_ALL=C
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "refusing: Linux aarch64 only" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "thor2" ]]; then
  echo "refusing: corpus run is thor2 first" >&2
  exit 2
fi
ulimit -t 14400
export OMP_NUM_THREADS=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
clang -O2 -I"$ROOT/src/mujoco/include" "$ROOT/probes/oracle_dump.c" "$SO" -o "$OUT/oracle_dump" -Wl,-rpath,"$BUILD/cmake/lib" -lm
set +e
"$OUT/oracle_dump" classify "$MENAGERIE" "$OUT/classify.txt"
class_rc=$?
set -e
if [[ "$class_rc" -ne 0 ]]; then
  echo "classify exited $class_rc" >&2
  exit "$class_rc"
fi
ok=$(grep -c '^ok	' "$OUT/classify.txt" || true)
bad=$(grep -c '^fail	' "$OUT/classify.txt" || true)
if [[ "$ok" -ne 212 || "$bad" -ne 5 ]]; then
  echo "refusing: loadable $ok fail $bad, pin is 212 and 5" >&2
  exit 2
fi
awk -F'\t' '$1=="fail"{print $2}' "$OUT/classify.txt" | sort > "$OUT/failures.txt"
if ! cmp -s "$OUT/failures.txt" "$ROOT/probes/expected-load-failures.txt"; then
  echo "refusing: failure set is not the pinned five fragments" >&2
  diff -u "$ROOT/probes/expected-load-failures.txt" "$OUT/failures.txt" >&2 || true
  exit 2
fi
"$OUT/oracle_dump" native "$MENAGERIE" "$ROOT/schedules/x86-native.txt" "$OUT/native.txt"
"$OUT/oracle_dump" step "$MENAGERIE" "$ROOT/schedules/x86-step.txt" "$OUT/step.txt" 20
echo "wrote $OUT/native.txt and $OUT/step.txt"
echo "These are the Linux aarch64 CPU oracle. They are not the x86 files and not Darwin goldens."
