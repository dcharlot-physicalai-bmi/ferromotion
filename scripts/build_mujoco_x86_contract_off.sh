#!/usr/bin/env bash
# Named x86_64 contract-off oracle for one MuJoCo release.
# Same compiler, flags, linker, and path layout as
# scripts/build_mujoco_3130_x86_contract_off.sh, which reproduces
# libmujoco.so.3.13.0 sha256
# 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc
# from source /tmp/mujoco-3.13.0-src and work /tmp/oracle-x86-contract-off.
#
# A different source path changes that hash (the other recorded value is
# 64c5451f). Do not retarget the 3.13.0 oracle at another directory.
# This script does not build on aarch64 and does not replace the 3.13.0 id.
set -euo pipefail
VER="${1:?mujoco version, for example 4.0.0}"
SRC="${2:-/tmp/mujoco-${VER}-src}"
if [[ "$VER" == "3.13.0" ]]; then
  WORK="${3:-/tmp/oracle-x86-contract-off}"
else
  WORK="${3:-/tmp/oracle-x86-contract-off-${VER}}"
fi
CC_BIN="${CC:-/usr/lib/llvm-18/bin/clang}"
CXX_BIN="${CXX:-/usr/lib/llvm-18/bin/clang++}"
FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=x86-64-v2 -mno-fma -mno-avx"
# gold is the effective linker. CMake also probes lld; this flag is last.
export LDFLAGS="-fuse-ld=gold${LDFLAGS:+ $LDFLAGS}"
# clang++ looks up libstdc++ next to the selected GCC installation.
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/14${LIBRARY_PATH:+:$LIBRARY_PATH}"
if [[ ! -x "$CC_BIN" || ! -x "$CXX_BIN" ]]; then
  echo "refusing: clang 18 is not at $CC_BIN" >&2
  exit 2
fi
if [[ ! -d "$SRC" ]]; then
  echo "refusing: source tree $SRC does not exist" >&2
  exit 2
fi
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
# elasticity lives in plugin/elasticity; sdf_plugin lives in plugin/sdf.
# A release that removed a plugin (4.0.0 removed elasticity) skips that target.
targets=(mujoco)
[[ -f "$SRC/plugin/actuator/CMakeLists.txt" ]] && targets+=(actuator)
[[ -f "$SRC/plugin/elasticity/CMakeLists.txt" ]] && targets+=(elasticity)
[[ -f "$SRC/plugin/sensor/CMakeLists.txt" ]] && targets+=(sensor)
[[ -f "$SRC/plugin/sdf/CMakeLists.txt" ]] && targets+=(sdf_plugin)
cmake --build "$WORK/build" --target "${targets[@]}" -j"${JOBS:-$(nproc)}"
echo "built $WORK/build targets: ${targets[*]}"
sha256sum "$WORK"/build/lib/libmujoco.so* "$WORK"/build/lib/libactuator.so \
  "$WORK"/build/lib/libelasticity.so "$WORK"/build/lib/libsensor.so \
  "$WORK"/build/lib/libsdf_plugin.so 2>/dev/null || \
  sha256sum "$WORK"/build/lib/*.so "$WORK"/build/lib/*.so.* 2>/dev/null || true
