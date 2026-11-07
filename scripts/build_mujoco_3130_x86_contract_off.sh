#!/usr/bin/env bash
# Named oracle: mujoco-3.13.0-x86_64-contract-off
# Does not replace the PyPI wheel or the arm64 golden arrays.
set -euo pipefail
SRC="${1:-/tmp/mujoco-3.13.0-src}"
WORK="${2:-/tmp/oracle-x86-contract-off}"
CC_BIN="$(command -v clang)"
CXX_BIN="$(command -v clang++)"
# Explicit, reproducible. -march=x86-64-v2 has no AVX and no FMA.
# -mno-fma and -mno-avx repeat that ISA choice. -ffp-contract=off is the contraction policy.
# ARM flags are not used: this is an x86_64 build.
FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx"
# clang++ links libstdc++. The unversioned library lives with the gcc 13 toolchain.
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/14${LIBRARY_PATH:+:$LIBRARY_PATH}"
mkdir -p "$WORK"
rm -rf "$WORK/build"
cmake -S "$SRC" -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_C_FLAGS_RELEASE="$FLAGS" \
  -DCMAKE_CXX_FLAGS_RELEASE="$FLAGS" \
  -DCMAKE_C_FLAGS_DEBUG="$FLAGS" \
  -DCMAKE_CXX_FLAGS_DEBUG="$FLAGS" \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_ENABLE_AVX=OFF \
  -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF \
  -DMUJOCO_ENABLE_LTO=OFF \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
cmake --build "$WORK/build" --target mujoco -j"$(nproc)"
echo "built $WORK/build"
