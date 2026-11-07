#!/usr/bin/env bash
# Linux aarch64, hostname thor2 only. CPU compile, 2 jobs. No sudo, no apt, no rustup, no pip.
# Each run gets a fresh directory. An existing one is refused, not overwritten.
set -euo pipefail
export LC_ALL=C
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "refusing: Linux aarch64 only (got $(uname -s) $(uname -m)). Darwin goldens are a different oracle." >&2
  exit 2
fi
if [[ "$(hostname -s)" != "thor2" ]]; then
  echo "refusing: first build is thor2 only (hostname -s is '$(hostname -s)'). Do not run this on thor." >&2
  exit 2
fi
clang_line="$(clang --version | head -n 1)"
[[ "$clang_line" == *"18.1.3"* ]] || { echo "refusing: need clang 18.1.3, got: $clang_line" >&2; exit 2; }
cmake_line="$(cmake --version | head -n 1)"
[[ "$cmake_line" == "cmake version 3.28."* ]] || { echo "refusing: need cmake 3.28.x, got: $cmake_line" >&2; exit 2; }
command -v ninja >/dev/null || { echo "refusing: ninja is not on PATH. This script does not install it." >&2; exit 2; }
ulimit -t 14400
export OMP_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$HOME/ferromotion-oracle/$REV/build-$STAMP"
if [[ -e "$WORK" ]]; then
  echo "refusing: $WORK already exists. Receipts are not overwritten." >&2
  exit 2
fi
mkdir -p "$HOME/ferromotion-oracle/$REV"
mkdir "$WORK"
if [[ -d /usr/lib/gcc/aarch64-linux-gnu/13 ]]; then
  export LIBRARY_PATH="/usr/lib/gcc/aarch64-linux-gnu/13${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a"
{
  echo "hostname $(hostname)"
  echo "uname $(uname -s) $(uname -m)"
  echo "clang $clang_line"
  echo "cmake $cmake_line"
  echo "ninja $(ninja --version)"
  echo "flags $FLAGS"
  echo "jobs 2"
  echo "work $WORK"
} | tee "$WORK/identity.txt"
cmake -S "$ROOT/src/mujoco" -B "$WORK/cmake" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v clang)" \
  -DCMAKE_CXX_COMPILER="$(command -v clang++)" \
  -DCMAKE_C_FLAGS_RELEASE="$FLAGS" \
  -DCMAKE_CXX_FLAGS_RELEASE="$FLAGS" \
  -DCMAKE_C_FLAGS_DEBUG="$FLAGS" \
  -DCMAKE_CXX_FLAGS_DEBUG="$FLAGS" \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_TEST_PYTHON_UTIL=OFF \
  -DMUJOCO_ENABLE_AVX=OFF \
  -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF \
  -DMUJOCO_ENABLE_LTO=OFF \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DFETCHCONTENT_SOURCE_DIR_CCD="$ROOT/deps/ccd" \
  -DFETCHCONTENT_SOURCE_DIR_QHULL="$ROOT/deps/qhull" \
  -DFETCHCONTENT_SOURCE_DIR_TINYXML2="$ROOT/deps/tinyxml2" \
  -DFETCHCONTENT_SOURCE_DIR_TINYOBJLOADER="$ROOT/deps/tinyobjloader" \
  -DFETCHCONTENT_SOURCE_DIR_LODEPNG="$ROOT/deps/lodepng" \
  -DFETCHCONTENT_SOURCE_DIR_MINIZ="$ROOT/deps/miniz" \
  -DFETCHCONTENT_SOURCE_DIR_MARCHINGCUBECPP="$ROOT/deps/marchingcubecpp"
cmake --build "$WORK/cmake" --target mujoco actuator elasticity sensor sdf_plugin -j 2
SO="$(echo "$WORK/cmake/lib/libmujoco.so.3.13.0")"
sha256sum "$SO" | tee "$WORK/libmujoco.so.3.13.0.sha256"
echo "built $SO"
echo "record $WORK/libmujoco.so.3.13.0.sha256 as mujoco-3.13.0-arm64-linux-contract-off"
echo "It is not 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc and not a Darwin dylib."
