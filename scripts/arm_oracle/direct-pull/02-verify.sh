#!/usr/bin/env bash
# Stage 2. Offline check of the commits materialized in stage 1.
set -euo pipefail
export LC_ALL=C
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIR/lib.sh"
# shellcheck source=pins.sh
source "$DIR/pins.sh"
arm_oracle_refuse_unless_approved
RUN="${1:?run directory from 01-materialize.sh}"
VER="$(awk '$1=="version"{print $2}' "$RUN/pins.txt")"
[[ "$(git -C "$RUN/mujoco" rev-parse HEAD)" == "$(mujoco_commit "$VER")" ]]
[[ "$(git -C "$RUN/mujoco" rev-parse 'HEAD^{tree}')" == "$(mujoco_tree "$VER")" ]]
[[ "$(git -C "$RUN/menagerie" rev-parse HEAD)" == "$(menagerie_commit)" ]]
echo "stage 2 matched $VER $(mujoco_commit "$VER")"
