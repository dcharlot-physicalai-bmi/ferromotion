#!/usr/bin/env bash
# Compile mju_mulQuat, mju_rotVecQuat and mju_mulMatVec3 from a MuJoCo 3.13.0
# source tree with that arch's contract-off flags, then evaluate every row of
# a vectors file. The functions are taken from the tree passed on the command
# line. This script does not vendor them.
#
#   scripts/arm_oracle/op-vectors.sh /path/to/mujoco-3.13.0-src \
#     docs/acceptance/oracles/x86_64-contract-off-op-vectors.txt \
#     docs/acceptance/oracles/aarch64-contract-off-op-vectors.txt
#
# On Thor2, that third path is the aarch64 oracle file. x86 must reproduce
# docs/acceptance/oracles/x86_64-contract-off-op-vectors.txt byte for byte
# (pass the same path as the input and as --check).
#
#   scripts/arm_oracle/op-vectors.sh /path/to/mujoco-3.13.0-src \
#     docs/acceptance/oracles/x86_64-contract-off-op-vectors.txt --check
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 3 ]]; then
  echo "usage: op-vectors.sh <mujoco-3.13.0-src> <vectors.txt> <out.txt|--check>" >&2
  exit 2
fi

SRC="$1"
IN="$2"
DEST="$3"
SPATIAL="$SRC/src/engine/engine_util_spatial.c"
BLAS="$SRC/src/engine/engine_util_blas.c"
if [[ ! -f "$SPATIAL" || ! -f "$BLAS" ]]; then
  echo "refusing: $SRC is not a MuJoCo source tree (missing engine_util_spatial.c)" >&2
  exit 2
fi
if [[ ! -f "$IN" ]]; then
  echo "refusing: vectors file is missing: $IN" >&2
  exit 2
fi

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    # scripts/build_mujoco_3130_x86_contract_off.sh
    FLAGS=(-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx)
    ;;
  Linux-aarch64)
    # scripts/build_mujoco_3130_arm64_linux_contract_off.sh
    FLAGS=(-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a)
    ;;
  *)
    echo "refusing: contract-off op vectors are built on Linux x86_64 or Linux aarch64 (got $(uname -s) $(uname -m))" >&2
    exit 2
    ;;
esac

if ! command -v clang >/dev/null 2>&1; then
  echo "refusing: clang is not on PATH" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/direct-pull/probes/op-vectors.c"
if [[ ! -f "$DRIVER" ]]; then
  echo "refusing: $DRIVER is missing" >&2
  exit 2
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

python3 - "$SPATIAL" "$BLAS" "$work/ops.c" <<'PY'
import sys
spatial, blas, dest = sys.argv[1:]

def extract(path, name):
    lines = open(path, encoding="utf-8").read().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"void {name}("):
            start = i
            break
    if start is None:
        sys.stderr.write(f"refusing: {path} has no void {name}\n")
        sys.exit(2)
    depth = 0
    began = False
    out = []
    for line in lines[start:]:
        out.append(line)
        depth += line.count("{") - line.count("}")
        if "{" in line:
            began = True
        if began and depth == 0:
            return "\n".join(out) + "\n"
    sys.stderr.write(f"refusing: {name} in {path} has no closing brace\n")
    sys.exit(2)

rot = extract(spatial, "mju_rotVecQuat")
quat = extract(spatial, "mju_mulQuat")
mat = extract(blas, "mju_mulMatVec3")
need = "quat[0]*vec[0] + quat[2]*vec[2] - quat[3]*vec[1]"
if need not in rot:
    sys.stderr.write("refusing: extracted mju_rotVecQuat is not the 3.13.0 body\n")
    sys.exit(2)
if "qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3]" not in quat:
    sys.stderr.write("refusing: extracted mju_mulQuat is not the 3.13.0 body\n")
    sys.exit(2)
if "mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2]" not in mat:
    sys.stderr.write("refusing: extracted mju_mulMatVec3 is not the 3.13.0 body\n")
    sys.exit(2)
open(dest, "w", encoding="utf-8").write(
    "typedef double mjtNum;\n"
    "static void mju_zero3(mjtNum res[3]) { res[0] = 0; res[1] = 0; res[2] = 0; }\n"
    "static void mji_copy3(mjtNum *res, const mjtNum *vec) {\n"
    "  res[0] = vec[0]; res[1] = vec[1]; res[2] = vec[2];\n"
    "}\n"
    + quat + rot + mat
)
PY

clang "${FLAGS[@]}" -std=c11 -o "$work/op-vectors" "$DRIVER" "$work/ops.c" -lm
if [[ "$DEST" == "--check" ]]; then
  "$work/op-vectors" "$IN" >"$work/out.txt"
  if ! cmp -s "$IN" "$work/out.txt"; then
    echo "refusing: compiled oracle ops do not reproduce $IN" >&2
    diff -u "$IN" "$work/out.txt" | head -40 >&2 || true
    exit 1
  fi
  echo "op-vectors check ok $(uname -m) $(sha256sum "$IN" | awk '{print $1}')"
  exit 0
fi
"$work/op-vectors" "$IN" >"$DEST"
echo "wrote $DEST"
