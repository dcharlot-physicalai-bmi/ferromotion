#!/usr/bin/env bash
# Mulquat probe. A mismatch is written and then exits non-zero. It does not edit a golden.
set -euo pipefail
export LC_ALL=C
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "refusing: Linux aarch64 only" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "thor2" ]]; then
  echo "refusing: probes run on thor2 first" >&2
  exit 2
fi
ulimit -t 600
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
clang -O2 -I"$ROOT/src/mujoco/include" "$ROOT/probes/mulquat.c" "$SO" -o "$OUT/mulquat" -Wl,-rpath,"$BUILD/cmake/lib" -lm
got="$("$OUT/mulquat")"
want="$(awk 'NF==4 {print; exit}' "$ROOT/fixtures/x86-contract-off-mulquat.txt")"
printf '%s\n' "$got" > "$OUT/mulquat-arm64.txt"
printf '%s\n' "$want" > "$OUT/mulquat-x86.txt"
echo "arm64_mulquat $got"
echo "x86_contract_off $want"
if [[ "$got" != "$want" ]]; then
  echo "mulquat_bits differ_from_x86_contract_off" | tee "$OUT/result.txt"
  echo "recorded in $OUT. This does not edit a golden." >&2
  exit 1
fi
echo "mulquat_bits match_x86_contract_off" | tee "$OUT/result.txt"
