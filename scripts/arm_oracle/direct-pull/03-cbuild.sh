#!/usr/bin/env bash
# Stage 3. One isolated contract-off CPU build. Calls the pinned thor2 script.
# Refuses unless stage 2 passed. The thor2 script itself refuses any other host.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
RUN="${1:?run directory that passed 02-verify.sh}"
direct_pull_hold_energy_lock "$RUN" stage3
if [[ ! -f "$RUN/provenance/stage2-ok" ]]; then
  echo "refusing: $RUN has no stage2-ok" >&2
  exit 2
fi
export RUSTUP_AUTO_INSTALL=0
# Install the shared-link recipe over the stage-1 copy from e0557dd.
# That copy does not pass -rtlib=compiler-rt, so the .so imports __muloti4.
runner_src=""
if [[ -f "$DIRECT_PULL_DIR/runners/build-thor2.sh" ]]; then
  runner_src="$DIRECT_PULL_DIR/runners"
elif [[ -f "$DIRECT_PULL_DIR/../build-thor2.sh" ]]; then
  runner_src="$DIRECT_PULL_DIR/.."
else
  echo "refusing: build-thor2.sh is not beside this script or in scripts/arm_oracle" >&2
  exit 2
fi
for f in build-thor2.sh compiler-rt.sh limits.sh; do
  [[ -f "$runner_src/$f" ]] || {
    echo "refusing: $runner_src/$f is missing" >&2
    exit 2
  }
  cp -f "$runner_src/$f" "$RUN/scripts/$f"
done
if [[ -f "$DIRECT_PULL_DIR/direct-pull-receipt.sh" ]]; then
  cp -f "$DIRECT_PULL_DIR/direct-pull-receipt.sh" "$RUN/scripts/write-receipt.sh"
fi
chmod +x "$RUN/scripts/"*.sh
grep -q 'CMAKE_SHARED_LINKER_FLAGS' "$RUN/scripts/build-thor2.sh" || {
  echo "refusing: installed build-thor2.sh does not set the shared-library link flags" >&2
  exit 2
}
grep -q -- '-rtlib=compiler-rt -unwindlib=libgcc' "$RUN/scripts/compiler-rt.sh" || {
  echo "refusing: installed compiler-rt.sh does not name -rtlib=compiler-rt -unwindlib=libgcc" >&2
  exit 2
}
previous_build=""
if [[ -f "$RUN/provenance/build-dir.txt" ]]; then
  previous_build="$(tr -d '[:space:]' <"$RUN/provenance/build-dir.txt")"
fi
# The build script enforces thor2, clang 18.1.3, cmake 3.28, ninja, 2 jobs,
# per-process CPU and virtual-memory caps, and a wall-clock timeout.
# It creates a new build-$STAMP directory. It does not delete an older one.
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
{
  echo "stage3-ok $build_dir"
  if [[ "$previous_build" == *build-20261107T131346Z* && "$build_dir" != "$previous_build" ]]; then
    echo "previous_build_left_untouched $previous_build"
  fi
} | tee "$RUN/provenance/stage3-ok"
