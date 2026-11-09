#!/usr/bin/env bash
# Linux aarch64, hostname thor2 only. CPU compile, 2 jobs. No sudo, no apt, no rustup, no pip.
# Each run gets a fresh directory. An existing one is refused, not overwritten.
set -euo pipefail
export LC_ALL=C
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=limits.sh
source "$SCRIPT_DIR/limits.sh"
arm_oracle_require_thor2 "first build is thor2 only"
arm_oracle_apply_limits
if ! ROOT="$(arm_oracle_root "$0")"; then
  exit 2
fi
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$HOME/ferromotion-oracle/$REV/build-$STAMP"
if [[ -e "$WORK" ]]; then
  echo "refusing: $WORK already exists. Receipts are not overwritten." >&2
  exit 2
fi
mkdir -p "$HOME/ferromotion-oracle/$REV"
mkdir "$WORK"
clang -O2 -I"$ROOT/probes" "$ROOT/probes/hexfloat_selftest.c" -o "$WORK/hexfloat_selftest"
set +e
"$WORK/hexfloat_selftest" >"$WORK/hexfloat_selftest.txt" 2>&1
self_rc=$?
set -e
if [[ "$self_rc" -ne 0 ]]; then
  echo "execution: hexfloat selftest failed" >&2
  cat "$WORK/hexfloat_selftest.txt" >&2 || true
  exit 3
fi
if [[ "${FERROMOTION_ORACLE_SMOKE:-}" == "x86" ]]; then
  echo "smoke: hexfloat selftest passed on $(uname -m). Refusing to configure or label an aarch64 library here." | tee "$WORK/result.txt"
  exit 2
fi
clang_line="$(clang --version | head -n 1)"
[[ "$clang_line" == *"18.1.3"* ]] || {
  echo "refusing: need clang 18.1.3, got: $clang_line" >&2
  exit 2
}
cmake_line="$(cmake --version | head -n 1)"
[[ "$cmake_line" == "cmake version 3.28."* ]] || {
  echo "refusing: need cmake 3.28.x, got: $cmake_line" >&2
  exit 2
}
if ! command -v ninja >/dev/null 2>&1; then
  echo "refusing: ninja is not on PATH. This script does not install it." >&2
  exit 2
fi
if [[ -d /usr/lib/gcc/aarch64-linux-gnu/13 ]]; then
  export LIBRARY_PATH="/usr/lib/gcc/aarch64-linux-gnu/13${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a"
rt="$(arm_oracle_compiler_rt_path)" || exit 2
arm_oracle_compiler_rt_record "$WORK/compiler-rt.txt" || exit 2
SHARED_LINK="$(arm_oracle_shared_link_flags)"
{
  echo "hostname $(hostname)"
  echo "uname $(uname -s) $(uname -m)"
  echo "clang $clang_line"
  echo "cmake $cmake_line"
  echo "ninja $(ninja --version)"
  echo "flags $FLAGS"
  echo "shared_link $SHARED_LINK"
  echo "compiler_rt_path $rt"
  echo "jobs 2"
  echo "work $WORK"
  echo "rlimit_cpu_seconds_per_process $(ulimit -t)"
  echo "rlimit_as_kib_per_process $(ulimit -v)"
  echo "wall_timeout_seconds $ARM_ORACLE_WALL_TIMEOUT"
} | tee "$WORK/identity.txt"
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" cmake -S "$ROOT/src/mujoco" -B "$WORK/cmake" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v clang)" \
  -DCMAKE_CXX_COMPILER="$(command -v clang++)" \
  -DCMAKE_C_FLAGS_RELEASE="$FLAGS" \
  -DCMAKE_CXX_FLAGS_RELEASE="$FLAGS" \
  -DCMAKE_C_FLAGS_DEBUG="$FLAGS" \
  -DCMAKE_CXX_FLAGS_DEBUG="$FLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$SHARED_LINK" \
  -DCMAKE_MODULE_LINKER_FLAGS="$SHARED_LINK" \
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
cfg_rc=$?
set -e
if [[ "$cfg_rc" -ne 0 ]]; then
  echo "execution: cmake configure failed ($cfg_rc)" >&2
  exit 3
fi
set +e
arm_oracle_timeout "$ARM_ORACLE_WALL_TIMEOUT" cmake --build "$WORK/cmake" --target mujoco actuator elasticity sensor sdf_plugin -j 2
build_rc=$?
set -e
if [[ "$build_rc" -ne 0 ]]; then
  echo "execution: cmake build failed ($build_rc)" >&2
  exit 3
fi
SO="$WORK/cmake/lib/libmujoco.so.3.13.0"
if ! arm_oracle_reject_undefined_builtins "$WORK/cmake/lib"; then
  echo "execution: a built shared library still imports a compiler-rt builtin" >&2
  exit 3
fi
sha256sum "$SO" | tee "$WORK/libmujoco.so.3.13.0.sha256"
echo "built $SO"
export ARM_ORACLE_ROOT="$ROOT"
"$SCRIPT_DIR/write-receipt.sh" "$WORK" "$WORK/cmake/lib" build
echo "record $WORK/libmujoco.so.3.13.0.sha256 together with $WORK/RECEIPT.txt"
echo "The receipt binds libmujoco.so.3.13.0, libactuator.so, libelasticity.so, libsensor.so, and libsdf_plugin.so."
echo "It is not 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc and not a Darwin dylib."
