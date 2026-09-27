#!/usr/bin/env bash
# Build scripts/qhull_reference_harness.c against qhull d1c2fc0 (the commit MuJoCo 3.13.0 pins), twice: with
# floating-point contraction on (as the Apple-silicon MuJoCo wheel is built) and off.
#
#     scripts/qhull_reference_build.sh <work dir>        # then <work dir>/qg_on and <work dir>/qg_off
#     cargo run --release --example menagerie_qhull -- <menagerie> <graph oracle> --dump verts.txt
#     <work dir>/qg_on < verts.txt > reference.txt
set -euo pipefail
WORK="${1:?usage: qhull_reference_build.sh <work dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$WORK/qhull"
mkdir -p "$SRC"
REV=d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6
FILES="libqhull_r.c geom_r.c geom2_r.c poly_r.c poly2_r.c merge_r.c qset_r.c mem_r.c global_r.c io_r.c stat_r.c random_r.c user_r.c usermem_r.c userprintf_r.c"
for f in $FILES libqhull_r.h geom_r.h io_r.h mem_r.h merge_r.h poly_r.h qhull_ra.h qset_r.h random_r.h stat_r.h user_r.h; do
  [ -f "$SRC/$f" ] || curl -sfL -o "$SRC/$f" "https://raw.githubusercontent.com/qhull/qhull/$REV/src/libqhull_r/$f"
done
for c in on off; do
  cc -O2 -ffp-contract=$c -w -I"$SRC" "$HERE/qhull_reference_harness.c" $(for f in $FILES; do echo "$SRC/$f"; done) -o "$WORK/qg_$c" -lm
done
echo "built $WORK/qg_on and $WORK/qg_off"
