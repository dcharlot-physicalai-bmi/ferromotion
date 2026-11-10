#!/usr/bin/env bash
# Stage 3. Linux aarch64, hostname thor2, approval required. CPU compile, 2 jobs.
# Does not install packages. Does not select the x86 linker or the x86 -march flags.
set -euo pipefail
export LC_ALL=C
DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIR/lib.sh"
# shellcheck source=pins.sh
source "$DIR/pins.sh"
arm_oracle_refuse_unless_approved
RUN="${1:?run directory from 01-materialize.sh}"
VER="$(awk '$1=="version"{print $2}' "$RUN/pins.txt")"
clang_line="$(clang --version | head -n 1)"
[[ "$clang_line" == *"18.1.3"* ]] || {
  echo "refusing: need clang 18.1.3, got: $clang_line" >&2
  exit 2
}
command -v cmake >/dev/null
command -v ninja >/dev/null
FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a"
WORK="$RUN/build"
if [[ -e "$WORK" ]]; then
  echo "refusing: $WORK already exists" >&2
  exit 2
fi
{
  echo "hostname $(hostname)"
  echo "uname $(uname -s) $(uname -m)"
  echo "clang $clang_line"
  echo "flags $FLAGS"
  echo "jobs 2"
  echo "version $VER"
} | tee "$WORK.identity.txt"
mkdir "$WORK"
cmake -S "$RUN/mujoco" -B "$WORK/cmake" -G Ninja \
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
  -DMUJOCO_ENABLE_AVX=OFF \
  -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF \
  -DMUJOCO_ENABLE_LTO=OFF \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
targets=(mujoco)
[[ -f "$RUN/mujoco/plugin/actuator/CMakeLists.txt" ]] && targets+=(actuator)
[[ -f "$RUN/mujoco/plugin/elasticity/CMakeLists.txt" ]] && targets+=(elasticity)
[[ -f "$RUN/mujoco/plugin/sensor/CMakeLists.txt" ]] && targets+=(sensor)
[[ -f "$RUN/mujoco/plugin/sdf/CMakeLists.txt" ]] && targets+=(sdf_plugin)
cmake --build "$WORK/cmake" --target "${targets[@]}" -j 2
sha256sum "$WORK"/cmake/lib/libmujoco.so* "$WORK"/cmake/lib/lib*.so "$WORK"/cmake/lib/mujoco_plugin/lib*.so 2>/dev/null | tee "$WORK/sha256.txt" || true
echo "stage 3 built $VER. The sha256 is not an oracle id until x86 parity is proved."
