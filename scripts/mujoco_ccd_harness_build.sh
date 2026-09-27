#!/usr/bin/env bash
# Build scripts/mujoco_ccd_harness.c (fixed probes) and scripts/mujoco_ccd_pairs.c (Menagerie pairs from
# MENAGERIE_CCD_DUMP) against MuJoCo's OWN engine_collision_gjk.c, twice: once with
# floating-point contraction off and once on. Where the geometry ties, the two builds disagree — which is how
# we know a difference is MuJoCo's own non-determinism and not a port bug.
#
#     scripts/mujoco_ccd_harness_build.sh <work dir with oracle-venv>   # then ccd_{off,on}, pairs_{off,on}
set -euo pipefail
WORK="${1:?usage: mujoco_ccd_harness_build.sh <work dir>}"
MJ="${2:-3.13.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$WORK/mjsrc"
DY=$(ls "$WORK"/oracle-venv/lib/python*/site-packages/mujoco/libmujoco.*.dylib | head -1)
INC=$(dirname "$DY")/include
mkdir -p "$SRC/engine" "$SRC/ccd"
cd "$SRC"
for f in engine_collision_gjk.c engine_collision_gjk.h engine_util_blas.c engine_util_blas.h \
         engine_util_spatial.c engine_util_spatial.h engine_util_errmem.h engine_util_misc.h \
         engine_macro.h engine_collision_convex.h engine_collision_primitive.h engine_inline.h \
         engine_util_blas_avx.h; do
  [ -f "engine/$f" ] || curl -sfL -o "engine/$f" "https://raw.githubusercontent.com/google-deepmind/mujoco/$MJ/src/engine/$f"
done
# libccd is only needed for a type in a header the CCD sources include; the CCD itself does not use it
printf 'typedef double ccd_real_t;\ntypedef struct { ccd_real_t v[3]; } ccd_vec3_t;\n' > ccd/vec3.h
# the dylib announces itself as a framework; give the linker that path
mkdir -p mujoco.framework/Versions/A
ln -sf "$(cd "$(dirname "$DY")" && pwd)/$(basename "$DY")" mujoco.framework/Versions/A/"$(basename "$DY")"
for c in off on; do
  for f in engine_collision_gjk engine_util_blas engine_util_spatial; do
    cc -O2 -ffp-contract=$c -I. -I"$INC" -c "engine/$f.c" -o "${f}_$c.o"
  done
  cc -O2 -ffp-contract=$c -I. -I"$INC" -o "ccd_$c" "$HERE/mujoco_ccd_harness.c" \
     engine_collision_gjk_$c.o engine_util_blas_$c.o engine_util_spatial_$c.o "$DY" -Wl,-rpath,"$SRC"
  cc -O2 -ffp-contract=$c -I. -I"$INC" -o "pairs_$c" "$HERE/mujoco_ccd_pairs.c" \
     engine_collision_gjk_$c.o engine_util_blas_$c.o engine_util_spatial_$c.o "$DY" -Wl,-rpath,"$SRC"
done
echo "built $SRC/ccd_off, $SRC/ccd_on, $SRC/pairs_off and $SRC/pairs_on"
