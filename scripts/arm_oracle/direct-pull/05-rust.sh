#!/usr/bin/env bash
# Stage 5. Not a parity result. Refuses until stage 4 has an aarch64 dump.
set -euo pipefail
export LC_ALL=C
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIR/lib.sh"
arm_oracle_refuse_unless_approved
RUN="${1:?run directory}"
if [[ ! -f "$RUN/aarch64-native.txt" || ! -f "$RUN/aarch64-step.txt" ]]; then
  echo "refusing: aarch64 dumps are absent. Stage 5 does not claim parity." >&2
  exit 2
fi
echo "refusing: no approved comparison of these dumps to x86 has been recorded." >&2
exit 2
