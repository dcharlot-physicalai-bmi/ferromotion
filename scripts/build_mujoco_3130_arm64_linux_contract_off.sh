#!/usr/bin/env bash
# Named oracle to build later: mujoco-3.13.0-arm64-linux-contract-off
# Linux aarch64 only. This is not the Darwin arm64 golden dylib and not an x86 build.
# Do not run it on macOS. Do not run it until a verified aarch64 Linux worker is approved.
set -euo pipefail
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "refusing: this bundle runs only on Linux aarch64 (got $(uname -s) $(uname -m))" >&2
  echo "Darwin arm64 goldens are a separate unproven oracle and are not built here." >&2
  exit 2
fi
echo "aarch64 Linux guard passed. A foreman-approved worker may continue the build." >&2
SRC="${1:-/tmp/mujoco-3.13.0-src}"
WORK="${2:-/tmp/oracle-arm64-linux-contract-off}"
# Explicit contraction and ISA. x86 -mno-fma/-mavx are N/A and are not passed.
# -ffp-contract=off is the contraction policy. No -mcpu is selected until the worker's CPU is recorded.
FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a"
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
cmake -S "$SRC" -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v "$CC")" \
  -DCMAKE_CXX_COMPILER="$(command -v "$CXX")" \
  -DCMAKE_C_FLAGS_RELEASE="$FLAGS" \
  -DCMAKE_CXX_FLAGS_RELEASE="$FLAGS" \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_ENABLE_AVX=OFF \
  -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF \
  -DMUJOCO_ENABLE_LTO=OFF \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
cmake --build "$WORK/build" --target mujoco -j"$(nproc)"
echo "built $WORK/build/lib/libmujoco.so.3.13.0"
echo "Record sha256sum of that file, clang --version, and /proc/cpuinfo before any comparison."
