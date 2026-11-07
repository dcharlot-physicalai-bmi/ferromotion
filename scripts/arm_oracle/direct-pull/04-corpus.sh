#!/usr/bin/env bash
# Stage 4. Probes, then the 212-model native dump and 20-step dump.
# Stops if the x86 replay schedules are absent or are not the pinned bytes.
# It does not download them and does not invent a replacement.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
RUN="${1:?run directory from 01-materialize.sh}"
native="$RUN/schedules/x86-native.txt"
step="$RUN/schedules/x86-step.txt"
if [[ ! -f "$native" || ! -f "$step" ]]; then
  cat >&2 <<EOF
STOP before the corpus comparison.
x86-native.txt and x86-step.txt are not in ferromotion ${FERROMOTION_COMMIT} and they have no public URL.
This script will not fetch them, synthesize them, or treat a different file as those inputs.
If a later decision supplies the original bytes, put them at:
  $native
  $step
Required sha256:
  ${SCHEDULE_NATIVE_SHA256}  x86-native.txt
  ${SCHEDULE_STEP_SHA256}  x86-step.txt
EOF
  exit 2
fi
direct_pull_require_sha "$native" "$SCHEDULE_NATIVE_SHA256"
direct_pull_require_sha "$step" "$SCHEDULE_STEP_SHA256"
if [[ ! -f "$RUN/provenance/stage3-ok" ]]; then
  echo "refusing: $RUN has no stage3-ok. The C library has not been built by stage 3." >&2
  exit 2
fi
BUILD="$(tr -d '[:space:]' <"$RUN/provenance/build-dir.txt")"
export RUSTUP_AUTO_INSTALL=0
set +e
"$RUN/scripts/run-probes.sh" "$BUILD" | tee "$RUN/provenance/stage4-probes.log"
probe_rc="${PIPESTATUS[0]}"
set -e
if [[ "$probe_rc" -ne 0 && "$probe_rc" -ne 1 ]]; then
  echo "execution: probe wrapper exit $probe_rc" >&2
  exit "$probe_rc"
fi
"$RUN/scripts/run-212.sh" "$RUN/menagerie" "$BUILD" | tee "$RUN/provenance/stage4-212.log"
{
  echo "stage4-ok"
  echo "probe_wrapper_exit $probe_rc"
} | tee "$RUN/provenance/stage4-ok"
