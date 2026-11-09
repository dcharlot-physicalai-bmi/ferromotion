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
direct_pull_hold_energy_lock "$RUN" stage4
native="$RUN/schedules/x86-native.txt"
step="$RUN/schedules/x86-step.txt"
if [[ ! -f "$native" || ! -f "$step" ]]; then
  if [[ -f "$DIRECT_PULL_DIR/schedules/x86-native.txt.xz" ]]; then
    bash "$DIRECT_PULL_DIR/materialize-schedules.sh" "$RUN"
  fi
fi
if [[ ! -f "$native" || ! -f "$step" ]]; then
  cat >&2 <<EOF
STOP before the corpus comparison.
The xz fixtures are not beside this script, so the pinned schedule bytes were not expanded.
This script does not synthesize them.
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
# Install the runners that link the self-contained library. They do not append
# the builtins archive: __muloti4 is GLOBAL HIDDEN, so that append cannot
# satisfy an undefined reference in the DSO.
runner_src=""
if [[ -f "$DIRECT_PULL_DIR/runners/compiler-rt.sh" ]]; then
  runner_src="$DIRECT_PULL_DIR/runners"
elif [[ -f "$DIRECT_PULL_DIR/../compiler-rt.sh" ]]; then
  runner_src="$DIRECT_PULL_DIR/.."
else
  echo "refusing: compiler-rt.sh is not beside this script or in scripts/arm_oracle" >&2
  exit 2
fi
for f in compiler-rt.sh limits.sh read-mulquat-fixture.sh run-probes.sh run-212.sh; do
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
grep -q 'rtlib=compiler-rt' "$RUN/scripts/compiler-rt.sh" || {
  echo "refusing: installed compiler-rt.sh does not query rtlib=compiler-rt" >&2
  exit 2
}
BUILD="$(tr -d '[:space:]' <"$RUN/provenance/build-dir.txt")"
direct_pull_accept_stage3_build "$BUILD"
export RUSTUP_AUTO_INSTALL=0
# set +e does not disarm an ERR trap. Exit 1 is a finished probe mismatch.
trap - ERR
set +e
"$RUN/scripts/run-probes.sh" "$BUILD" | tee "$RUN/provenance/stage4-probes.log"
probe_rc="${PIPESTATUS[0]}"
set -e
if [[ "$probe_rc" -ne 0 && "$probe_rc" -ne 1 ]]; then
  echo "execution: probe wrapper exit $probe_rc" >&2
  exit "$probe_rc"
fi
"$RUN/scripts/run-212.sh" "$RUN/menagerie" "$BUILD" | tee "$RUN/provenance/stage4-212.log"
wrote="$(grep -E '^wrote .*/native\.txt and .*/step\.txt$' "$RUN/provenance/stage4-212.log" | tail -n 1)"
arm_native="${wrote#wrote }"
arm_native="${arm_native%% and *}"
arm_step="${wrote##* and }"
if [[ ! -f "$arm_native" || ! -f "$arm_step" ]]; then
  echo "refusing: stage 4 log did not name native.txt and step.txt" >&2
  exit 3
fi
trap - ERR
set +e
bash "$DIRECT_PULL_DIR/compare-arm-x86.sh" "$arm_native" "$native" "$arm_step" "$step" \
  "$RUN/provenance/stage4-x86-verdict.txt"
x86_rc=$?
set -e
{
  echo "stage4-ok"
  echo "probe_wrapper_exit $probe_rc"
  echo "x86_parity_exit $x86_rc"
  echo "x86_verdict_file $RUN/provenance/stage4-x86-verdict.txt"
} | tee "$RUN/provenance/stage4-ok"
# Dumps exist either way. Exit 1 is a finished x86 mismatch, not a skipped probe.
if [[ "$x86_rc" -ne 0 ]]; then
  exit "$x86_rc"
fi
