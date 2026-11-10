#!/usr/bin/env bash
# Stage 4. Stops until the x86 dumps for this version are supplied.
# Exit 0 would mean dumps were written. It would not mean they match x86.
set -euo pipefail
export LC_ALL=C
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIR/lib.sh"
arm_oracle_refuse_unless_approved
RUN="${1:?run directory}"
VER="$(awk '$1=="version"{print $2}' "$RUN/pins.txt")"
for f in native.txt step.txt; do
  if [[ ! -f "$RUN/x86-$f" ]]; then
    echo "refusing: $RUN/x86-$f is absent. Stage 4 does not download or synthesize the x86 dump." >&2
    echo "x86-versus-aarch64 parity for $VER is not proved." >&2
    exit 2
  fi
done
if [[ ! -f "$RUN/x86-sha256.txt" ]]; then
  echo "refusing: $RUN/x86-sha256.txt is absent, so the supplied dumps are not pinned." >&2
  exit 2
fi
(cd "$RUN" && sha256sum -c x86-sha256.txt)
echo "stage 4 would compile oracle_schema_dump against the aarch64 library and replay x86-$VER."
echo "That comparison has no result in this repository."
exit 2
