#!/usr/bin/env bash
# Rebuild everything the Menagerie parity sweeps read: a MuJoCo venv, the Menagerie checkout, the contact
# oracle and its flat form, and the convex-hull oracle. Idempotent — each step is skipped if already present.
#
#     scripts/menagerie_oracle_setup.sh <work dir> [mujoco version] [samples per model]
#
# Then:
#     cargo run --release --example menagerie_contacts -- <work dir>/menagerie <work dir>/oracle_flat
#     cargo run --release --example menagerie_hulls    -- <work dir>/menagerie <work dir>/hull_oracle.txt
set -euo pipefail
WORK="${1:?usage: menagerie_oracle_setup.sh <work dir> [mujoco version] [samples]}"
MJ="${2:-3.13.0}"
K="${3:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"
cd "$WORK"

# a native-architecture interpreter: `uv venv -p 3.12` can pick up a cross-architecture build that cannot run
if [ ! -x oracle-venv/bin/python ]; then
  PY=""
  for cand in /opt/homebrew/opt/python@3.12/bin/python3.12 /usr/local/opt/python@3.12/bin/python3.12 python3.12; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0)' 2>/dev/null; then PY="$cand"; break; fi
  done
  [ -n "$PY" ] || { echo "no runnable python3.12 found" >&2; exit 1; }
  rm -rf oracle-venv
  uv venv -p "$PY" oracle-venv
  VIRTUAL_ENV="$WORK/oracle-venv" uv pip install -q "mujoco==$MJ" numpy
fi
PYBIN="$WORK/oracle-venv/bin/python"
"$PYBIN" -c "import platform, mujoco; print('python', platform.machine(), 'mujoco', mujoco.__version__)"

[ -d menagerie ] || git clone --depth 1 -q https://github.com/google-deepmind/mujoco_menagerie menagerie
echo "menagerie: $(ls menagerie | wc -l | tr -d ' ') entries"

[ -d oracle_contacts ] || "$PYBIN" "$HERE/menagerie_contacts_oracle.py" menagerie oracle_contacts "$K"
[ -d oracle_flat ]     || python3 "$HERE/menagerie_contacts_flat.py" oracle_contacts oracle_flat
[ -f hull_oracle.txt ] || "$PYBIN" "$HERE/mujoco_mesh_hull_oracle.py" menagerie hull_oracle.txt
echo "ready: $(ls oracle_flat | wc -l | tr -d ' ') flat models, $(grep -c '^mesh' hull_oracle.txt) hull meshes"
