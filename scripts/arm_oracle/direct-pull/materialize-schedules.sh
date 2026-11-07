#!/usr/bin/env bash
# Expand the lossless xz fixtures to the pinned schedule bytes.
# Does not recompute physics. A hash mismatch deletes the partial file.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
RUN="${1:?run directory}"
if ! command -v xz >/dev/null 2>&1; then
  echo "refusing: xz is not on PATH. This script does not install it." >&2
  exit 2
fi
mkdir -p "$RUN/schedules"
native="$RUN/schedules/x86-native.txt"
step="$RUN/schedules/x86-step.txt"
if [[ -f "$native" && -f "$step" ]]; then
  direct_pull_require_sha "$native" "$SCHEDULE_NATIVE_SHA256"
  direct_pull_require_sha "$step" "$SCHEDULE_STEP_SHA256"
  echo "schedules already match the pinned sha256"
  exit 0
fi
src_n="$DIRECT_PULL_DIR/schedules/x86-native.txt.xz"
src_s="$DIRECT_PULL_DIR/schedules/x86-step.txt.xz"
if [[ ! -f "$src_n" || ! -f "$src_s" ]]; then
  echo "refusing: $src_n or $src_s is not in this checkout." >&2
  echo "ecfccc6 does not contain these fixtures. Use UPGRADE.md." >&2
  exit 2
fi
direct_pull_require_sha "$src_n" "$SCHEDULE_NATIVE_XZ_SHA256"
direct_pull_require_sha "$src_s" "$SCHEDULE_STEP_XZ_SHA256"
expand_one() {
  local src="$1" dest="$2" want="$3" bytes="$4" partial
  partial="$dest.partial"
  rm -f "$partial"
  xz -dc "$src" >"$partial"
  local got n
  got="$(direct_pull_sha "$partial")"
  n="$(wc -c <"$partial" | tr -d ' ')"
  if [[ "$got" != "$want" || "$n" != "$bytes" ]]; then
    rm -f "$partial"
    echo "refusing: $dest sha256 $got bytes $n is not $want $bytes" >&2
    exit 2
  fi
  mv "$partial" "$dest"
}
expand_one "$src_n" "$native" "$SCHEDULE_NATIVE_SHA256" "$SCHEDULE_NATIVE_BYTES"
expand_one "$src_s" "$step" "$SCHEDULE_STEP_SHA256" "$SCHEDULE_STEP_BYTES"
echo "expanded schedules into $RUN/schedules"
