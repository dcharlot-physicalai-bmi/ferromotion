#!/usr/bin/env bash
# Diagnostic sibling of mujoco-3.13.0-x86_64-contract-off.
# Same tag, same flags, fprintf-only mesh tensor and mjuu_eig3 traces.
# This artifact is not the uninstrumented oracle. Do not copy its sha256
# onto mujoco-3.13.0-x86_64-contract-off.
set -euo pipefail
SRC="${1:?clean mujoco 3.13.0 source tree (tag 123347c0)}"
WORK="${2:?output directory, must not be the uninstrumented oracle}"
if [[ "$(realpath -m "$WORK")" == "$(realpath -m /tmp/oracle-x86-contract-off)" ]]; then
  echo "refusing: $WORK is the uninstrumented oracle directory" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/docs/acceptance/oracles/mesh-trace-3.13.0.patch"
STAGE="$(mktemp -d /tmp/mujoco-3.13.0-meshtrace.XXXXXX)"
cp -a "$SRC"/. "$STAGE"/
patch -d "$STAGE" -p0 < "$PATCH"
"$ROOT/scripts/build_mujoco_3130_x86_contract_off.sh" "$STAGE" "$WORK"
echo "diagnostic source left at $STAGE"
echo "diagnostic build at $WORK (not sha 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc)"
