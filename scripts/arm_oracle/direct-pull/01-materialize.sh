#!/usr/bin/env bash
# Stage 1, the only network stage. Not run in this workspace.
set -euo pipefail
export LC_ALL=C
export GIT_TERMINAL_PROMPT=0
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIR/lib.sh"
# shellcheck source=pins.sh
source "$DIR/pins.sh"
arm_oracle_refuse_unless_approved
VER="${1:?version: 3.14.0, 3.15.0, 3.16.0, 3.17.0, or 4.0.0}"
COMMIT="$(mujoco_commit "$VER")"
TREE="$(mujoco_tree "$VER")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="${FERROMOTION_ORACLE_HOME:-$HOME/ferromotion-oracle}/$VER-$STAMP"
if [[ -e "$RUN" ]]; then
  echo "refusing: $RUN already exists" >&2
  exit 2
fi
mkdir -p "$RUN"
{
  echo "version $VER"
  echo "mujoco_commit $COMMIT"
  echo "mujoco_tree $TREE"
  echo "menagerie_commit $(menagerie_commit)"
} | tee "$RUN/pins.txt"
git -C "$RUN" clone --depth 1 --branch "$VER" https://github.com/google-deepmind/mujoco.git mujoco
got="$(git -C "$RUN/mujoco" rev-parse HEAD)"
got_tree="$(git -C "$RUN/mujoco" rev-parse 'HEAD^{tree}')"
[[ "$got" == "$COMMIT" && "$got_tree" == "$TREE" ]] || {
  echo "refusing: mujoco HEAD $got tree $got_tree is not $COMMIT $TREE" >&2
  exit 2
}
git -C "$RUN" clone https://github.com/google-deepmind/mujoco_menagerie.git menagerie
git -C "$RUN/menagerie" checkout --detach "$(menagerie_commit)"
echo "$RUN" | tee "$RUN/run-dir.txt"
echo "stage 1 wrote $RUN. It did not compile."
