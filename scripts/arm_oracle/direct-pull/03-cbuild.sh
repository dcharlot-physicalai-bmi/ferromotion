#!/usr/bin/env bash
# Stage 3. One isolated contract-off CPU build. Calls the pinned thor2 script.
# Refuses unless stage 2 passed. The thor2 script itself refuses any other host.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
RUN="${1:?run directory that passed 02-verify.sh}"
if [[ ! -f "$RUN/provenance/stage2-ok" ]]; then
  echo "refusing: $RUN has no stage2-ok" >&2
  exit 2
fi
export RUSTUP_AUTO_INSTALL=0
# The build script enforces thor2, clang 18.1.3, cmake 3.28, ninja, 2 jobs,
# per-process CPU and virtual-memory caps, and a wall-clock timeout.
set +e
"$RUN/scripts/build-thor2.sh" | tee "$RUN/provenance/stage3.log"
build_rc="${PIPESTATUS[0]}"
set -e
so_line="$(grep -E '^built ' "$RUN/provenance/stage3.log" | tail -n 1 || true)"
if [[ -n "$so_line" ]]; then
  lib="${so_line#built }"
  build_dir="$(cd "$(dirname "$lib")/../.." && pwd)"
  printf '%s\n' "$build_dir" >"$RUN/provenance/build-dir.txt"
fi
if [[ "$build_rc" -ne 0 ]]; then
  echo "stage 3 command exit $build_rc" >&2
  exit "$build_rc"
fi
if [[ -z "$so_line" ]]; then
  echo "refusing: build log has no library path" >&2
  exit 3
fi
echo "stage3-ok $build_dir" | tee "$RUN/provenance/stage3-ok"
