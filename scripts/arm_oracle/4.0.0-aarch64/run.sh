#!/usr/bin/env bash
# mujoco-4.0.0-aarch64-contract-off
#
# One run on thor2 or thor.local. Public tarballs only. No sudo, no apt, no pip,
# no credential. A new directory under $HOME/ferromotion-oracle-4.0.0.
# It does not read or write the 3.13.0 sources, build-20261109T213225Z,
# or libmujoco sha256 9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9.
#
# Commands for each host are in COMMANDS.txt.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

ORACLE_ID="mujoco-4.0.0-aarch64-contract-off"
MUJOCO_COMMIT="ac60eae30fa7e0d31315c38d9476752fb81be67e"
MENAGERIE_COMMIT="4d038b3feae26ec82b46a4d586379114012a8ac7"
FORBIDDEN_REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
FORBIDDEN_BUILD="build-20261109T213225Z"
FORBIDDEN_LIB="9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9"
FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a"
SHARED_LINK="-rtlib=compiler-rt -unwindlib=libgcc"
LOCK_WAIT="${BMI_ENERGY_LOCK_WAIT:-120}"

unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GH_HOST GIT_ASKPASS SSH_ASKPASS || true
export GIT_TERMINAL_PROMPT=0
export RUSTUP_AUTO_INSTALL=0

arm40_refuse() {
  echo "refusing: $*" >&2
  exit 2
}

arm40_protected() {
  local path="$1"
  case "$path" in
    *"$FORBIDDEN_REV"*|*"$FORBIDDEN_BUILD"*|/tmp/mujoco-3.13.0-src|/*/mujoco-3.13.0-src|/*/mujoco-3.13.0-src/*)
      arm40_refuse "$path is a 3.13.0 source or the build-20261109T213225Z tree"
      ;;
  esac
}

arm40_host_kind() {
  local short full name
  short="$(hostname -s 2>/dev/null || true)"
  full="$(hostname -f 2>/dev/null || true)"
  name="$(hostname 2>/dev/null || true)"
  if [[ "$short" == "thor2" || "$name" == "thor2" ]]; then
    printf '%s\n' thor2
    return 0
  fi
  if [[ "$short" == "thor.local" || "$full" == "thor.local" || "$name" == "thor.local" ]]; then
    printf '%s\n' thor.local
    return 0
  fi
  if [[ "$short" == "thor" && "$full" == "thor.local" ]]; then
    printf '%s\n' thor.local
    return 0
  fi
  return 1
}

arm40_need() {
  command -v "$1" >/dev/null 2>&1 || arm40_refuse "$1 is not on PATH. This script does not install it."
}

arm40_compiler_rt_path() {
  local path base
  path="$(clang -print-libgcc-file-name -rtlib=compiler-rt)" || arm40_refuse "clang -print-libgcc-file-name -rtlib=compiler-rt failed"
  base="$(basename "$path")"
  [[ -f "$path" && "$base" == libclang_rt.builtins*.a ]] || arm40_refuse "compiler-rt builtins archive is not a file ($path)"
  printf '%s\n' "$path"
}

arm40_reject_undefined_builtins() {
  local libdir="$1" so sym bad=0 path
  shopt -s nullglob
  for so in "$libdir"/*.so "$libdir"/*.so.* "$libdir"/mujoco_plugin/*.so; do
    [[ -f "$so" ]] || continue
    path="$so"
    while IFS= read -r sym; do
      [[ -n "$sym" ]] || continue
      if [[ "$sym" == "__muloti4" ]] || [[ "$sym" =~ ^__[A-Za-z0-9_]*ti[0-9]*$ ]] || [[ "$sym" =~ ^__[A-Za-z0-9_]*o[dt]i[0-9]+$ ]]; then
        echo "refusing: $path has undefined compiler-rt symbol $sym" >&2
        bad=1
      fi
    done < <(nm -D --undefined-only "$path" 2>/dev/null | awk '{print $NF}')
  done
  shopt -u nullglob
  [[ "$bad" -eq 0 ]] || return 3
}

arm40_finish_lock() {
  [[ "${BMI_ENERGY_LOCK_HELD:-}" == "1" ]] || return 0
  if [[ -n "${ENERGY_LOCK_FILE:-}" ]]; then
    echo "energy_lock_released_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$ENERGY_LOCK_FILE"
  fi
}
trap arm40_finish_lock EXIT

arm40_hold_lock() {
  local requested acquired
  mkdir -p "$RUN/provenance"
  if [[ "$HOST_KIND" != "thor.local" ]]; then
    {
      echo "energy_lock_taken 0"
      echo "energy_lock_host $HOST_KIND"
      echo "energy_lock_reason host_is_not_thor.local"
    } >"$RUN/provenance/energy-lock.txt"
    return 0
  fi
  arm40_need flock
  exec 9>/tmp/bmi-energy.lock
  requested="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! flock -x -w "$LOCK_WAIT" 9; then
    arm40_refuse "/tmp/bmi-energy.lock not acquired within ${LOCK_WAIT}s (LOCK_EX)"
  fi
  acquired="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  export BMI_ENERGY_LOCK_HELD=1
  ENERGY_LOCK_FILE="$RUN/provenance/energy-lock.txt"
  {
    echo "energy_lock_path /tmp/bmi-energy.lock"
    echo "energy_lock_mode LOCK_EX"
    echo "energy_lock_host thor.local"
    echo "energy_lock_wait_bound_s $LOCK_WAIT"
    echo "energy_lock_requested_utc $requested"
    echo "energy_lock_acquired_utc $acquired"
    echo "energy_lock_pid $$"
  } >"$ENERGY_LOCK_FILE"
}

# One row is "<file> <url>". The dry-run and the download loop both read this.
arm40_fetch_table() {
  cat <<EOF
mujoco-4.0.0.tar.gz https://codeload.github.com/google-deepmind/mujoco/tar.gz/refs/tags/4.0.0
ccd.tar.gz https://codeload.github.com/danfis/libccd/tar.gz/7931e764a19ef6b21b443376c699bbc9c6d4fba8
qhull.tar.gz https://codeload.github.com/qhull/qhull/tar.gz/d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6
tinyxml2.tar.gz https://codeload.github.com/leethomason/tinyxml2/tar.gz/e6caeae85799003f4ca74ff26ee16a789bc2af48
tinyobjloader.tar.gz https://codeload.github.com/tinyobjloader/tinyobjloader/tar.gz/2945a967c5303b2c8c14174117c45f3302591150
lodepng.tar.gz https://codeload.github.com/lvandeve/lodepng/tar.gz/22561883dd63fd1850f18e1f6adac321e4f609b0
miniz.tar.gz https://codeload.github.com/richgel999/miniz/tar.gz/77d0dce8627735138c51770d1799a1ef48f2117d
marchingcubecpp.tar.gz https://codeload.github.com/aparis69/MarchingCubeCpp/tar.gz/f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d
menagerie.tar.gz https://codeload.github.com/google-deepmind/mujoco_menagerie/tar.gz/${MENAGERIE_COMMIT}
EOF
}

arm40_each_fetch() {
  local name url
  while read -r name url; do
    [[ -n "${name:-}" ]] || continue
    [[ "${name}" != \#* ]] || continue
    [[ -n "${url:-}" ]] || arm40_refuse "fetch entry ${name} has no URL"
    "$@" "$name" "$url"
  done < <(arm40_fetch_table)
}

# $1 and $2 are expanded before local runs. Referencing $name in that same
# local command sees the unset global and set -u aborts the fetch.
arm40_fetch() {
  local name url dest
  [[ $# -eq 2 ]] || arm40_refuse "arm40_fetch expects a name and a URL"
  name="$1"
  url="$2"
  dest="$RUN/tarballs/$name"
  arm40_need curl
  [[ ! -e "$dest" ]] || arm40_refuse "$dest already exists"
  # -q ignores ~/.curlrc. No netrc. HTTPS only.
  curl -q --fail --location --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 \
    --max-time 14400 -A 'ferromotion-4.0.0-aarch64' -o "$dest" "$url"
}

arm40_print_fetch() {
  printf 'fetch %s %s\n' "$1" "$2"
}

arm40_dry_run() {
  local name url listed=0 sha_names
  arm40_each_fetch arm40_print_fetch
  sha_names="$(awk '{print $2}' "$SCRIPT_DIR/tarballs.sha256" | sort)"
  while read -r name url; do
    [[ -n "${name:-}" ]] || continue
    listed=$((listed + 1))
    grep -qx "$name" <<<"$sha_names" || arm40_refuse "tarballs.sha256 has no line for $name"
    [[ "$url" == https://codeload.github.com/* ]] || arm40_refuse "$name URL is not a public codeload URL"
  done < <(arm40_fetch_table)
  [[ "$listed" -eq 9 ]] || arm40_refuse "expected 9 fetch entries, got $listed"
  local sha_count
  sha_count="$(grep -cve '^[[:space:]]*$' "$SCRIPT_DIR/tarballs.sha256")"
  [[ "$sha_count" -eq 9 ]] || arm40_refuse "tarballs.sha256 has $sha_count lines"
  echo "dry-run ok: 9 fetch entries, no download"
}

arm40_unpack() {
  local archive="$1" dest="$2"
  mkdir -p "$dest"
  tar -xzf "$archive" -C "$dest" --strip-components=1
}

# x86 can run this. It prints every name/URL pair and does not download.
if [[ "${1:-}" == "--dry-run" ]]; then
  arm40_dry_run
  exit 0
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  arm40_refuse "Linux aarch64 only (got $(uname -s) $(uname -m))"
fi
HOST_KIND="$(arm40_host_kind)" || arm40_refuse "hostname is $(hostname), not thor2 or thor.local"
if [[ "${FERROMOTION_THOR2_APPROVED:-}" != "1" ]]; then
  arm40_refuse "FERROMOTION_THOR2_APPROVED is not 1"
fi

for tool in curl sha256sum tar patch clang clang++ cmake ninja python3 timeout nm; do
  arm40_need "$tool"
done
clang_line="$(clang --version | head -n 1)"
[[ "$clang_line" == *"18.1.3"* ]] || arm40_refuse "need clang 18.1.3, got: $clang_line"
clangxx_line="$(clang++ --version | head -n 1)"
[[ "$clangxx_line" == *"18.1.3"* ]] || arm40_refuse "need clang++ 18.1.3, got: $clangxx_line"
cmake_line="$(cmake --version | head -n 1)"
[[ "$cmake_line" == "cmake version 3.28."* ]] || arm40_refuse "need cmake 3.28.x, got: $cmake_line"
python3 -c 'import numpy' >/dev/null 2>&1 || arm40_refuse "python3 numpy is not importable. This script does not install it."

BASE="${FERROMOTION_ORACLE_4_HOME:-$HOME/ferromotion-oracle-4.0.0}"
# A new second-resolution stamp every invocation. Existing build-* directories,
# including the failed build-20261113T131521Z, are left in place.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/$MUJOCO_COMMIT/build-$STAMP"
arm40_protected "$BASE"
arm40_protected "$RUN"
[[ ! -e "$RUN" ]] || arm40_refuse "$RUN already exists. This script does not reuse or delete a build directory."
mkdir -p "$RUN/tarballs" "$RUN/src" "$RUN/deps" "$RUN/dumps" "$RUN/provenance"

arm40_hold_lock

{
  echo "oracle_id $ORACLE_ID"
  echo "host_kind $HOST_KIND"
  echo "hostname $(hostname)"
  echo "run $RUN"
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$RUN/provenance/start.txt"

cd "$RUN/tarballs"
arm40_each_fetch arm40_fetch
sha256sum -c "$SCRIPT_DIR/tarballs.sha256"

arm40_unpack "$RUN/tarballs/mujoco-4.0.0.tar.gz" "$RUN/src/mujoco"
arm40_unpack "$RUN/tarballs/ccd.tar.gz" "$RUN/deps/ccd"
arm40_unpack "$RUN/tarballs/qhull.tar.gz" "$RUN/deps/qhull"
arm40_unpack "$RUN/tarballs/tinyxml2.tar.gz" "$RUN/deps/tinyxml2"
arm40_unpack "$RUN/tarballs/tinyobjloader.tar.gz" "$RUN/deps/tinyobjloader"
arm40_unpack "$RUN/tarballs/lodepng.tar.gz" "$RUN/deps/lodepng"
arm40_unpack "$RUN/tarballs/miniz.tar.gz" "$RUN/deps/miniz"
arm40_unpack "$RUN/tarballs/marchingcubecpp.tar.gz" "$RUN/deps/marchingcubecpp"
arm40_unpack "$RUN/tarballs/menagerie.tar.gz" "$RUN/src/menagerie"

grep -q 'mjVERSION_HEADER 4000000' "$RUN/src/mujoco/include/mujoco/mujoco.h" || arm40_refuse "unpacked MuJoCo header is not 4.0.0"
# The emscripten patches are the ones CMake would apply. Apply them here so the
# disconnected configure does not call git.
patch -p1 --forward --batch -d "$RUN/deps/ccd" <"$RUN/src/mujoco/cmake/ccd-support-emscripten.patch"
patch -p1 --forward --batch -d "$RUN/deps/qhull" <"$RUN/src/mujoco/cmake/qhull-support-emscripten.patch"
cp -f "$RUN/src/mujoco/cmake/third_party_deps/lodepng/CMakeLists.txt" "$RUN/deps/lodepng/CMakeLists.txt"

if [[ -d /usr/lib/gcc/aarch64-linux-gnu/13 ]]; then
  export LIBRARY_PATH="/usr/lib/gcc/aarch64-linux-gnu/13${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
ulimit -t 14400 || arm40_refuse "cannot set the per-process CPU limit"
ulimit -v 50331648 || arm40_refuse "cannot set the per-process virtual address space"
export OMP_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=2

RT="$(arm40_compiler_rt_path)"
RT_SHA="$(sha256sum "$RT" | awk '{print $1}')"
{
  echo "compiler_rt_query clang -print-libgcc-file-name -rtlib=compiler-rt"
  echo "compiler_rt_path $RT"
  echo "compiler_rt_sha256 $RT_SHA"
  echo "shared_link $SHARED_LINK"
} >"$RUN/provenance/compiler-rt.txt"

WORK="$RUN/cmake"
{
  echo "hostname $(hostname)"
  echo "host_kind $HOST_KIND"
  echo "uname $(uname -s) $(uname -m)"
  echo "clang $clang_line"
  echo "cmake $cmake_line"
  echo "ninja $(ninja --version)"
  echo "flags $FLAGS"
  echo "shared_link $SHARED_LINK"
  echo "jobs 2"
  echo "work $WORK"
} | tee "$RUN/provenance/identity.txt"

timeout -k 30 14400 cmake -S "$RUN/src/mujoco" -B "$WORK" -G Ninja \
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
  -DFETCHCONTENT_SOURCE_DIR_CCD="$RUN/deps/ccd" \
  -DFETCHCONTENT_SOURCE_DIR_QHULL="$RUN/deps/qhull" \
  -DFETCHCONTENT_SOURCE_DIR_TINYXML2="$RUN/deps/tinyxml2" \
  -DFETCHCONTENT_SOURCE_DIR_TINYOBJLOADER="$RUN/deps/tinyobjloader" \
  -DFETCHCONTENT_SOURCE_DIR_LODEPNG="$RUN/deps/lodepng" \
  -DFETCHCONTENT_SOURCE_DIR_MINIZ="$RUN/deps/miniz" \
  -DFETCHCONTENT_SOURCE_DIR_MARCHINGCUBECPP="$RUN/deps/marchingcubecpp"

timeout -k 30 14400 cmake --build "$WORK" --target mujoco actuator sensor sdf_plugin -j 2

SO="$WORK/lib/libmujoco.so.4.0.0"
[[ -f "$SO" ]] || arm40_refuse "build did not write $SO"
arm40_reject_undefined_builtins "$WORK/lib"
LIB_SHA="$(sha256sum "$SO" | awk '{print $1}')"
if [[ "$LIB_SHA" == "$FORBIDDEN_LIB" ]]; then
  arm40_refuse "built library matches the 3.13.0 ARM oracle $FORBIDDEN_LIB"
fi
sha256sum "$SO" | tee "$RUN/libmujoco.so.4.0.0.sha256"
echo "built $SO"

PLUG="$WORK/lib/mujoco_plugin"
[[ -d "$PLUG" ]] || PLUG="$WORK/lib"
DUMP_SRC="$REPO_ROOT/scripts/oracle_schema_dump.c"
SCHED_PY="$REPO_ROOT/scripts/oracle_schema_schedule.py"
[[ -f "$DUMP_SRC" && -f "$SCHED_PY" ]] || arm40_refuse "dumper sources are missing from $REPO_ROOT"
clang -O2 -I"$RUN/src/mujoco/include" "$DUMP_SRC" "$SO" -o "$RUN/dumps/oracle_schema_dump" \
  -Wl,-rpath,"$WORK/lib" -lm
"$RUN/dumps/oracle_schema_dump" schema | tee "$RUN/dumps/schema.txt"
"$RUN/dumps/oracle_schema_dump" props "$RUN/src/menagerie" "$PLUG" "$RUN/dumps/props.txt"
python3 "$SCHED_PY" "$RUN/dumps/props.txt" native "$RUN/dumps/native-schedule.txt"
python3 "$SCHED_PY" "$RUN/dumps/props.txt" step "$RUN/dumps/step-schedule.txt"
"$RUN/dumps/oracle_schema_dump" native "$RUN/src/menagerie" "$PLUG" "$RUN/dumps/native-schedule.txt" "$RUN/dumps/native.txt"
"$RUN/dumps/oracle_schema_dump" step "$RUN/src/menagerie" "$PLUG" "$RUN/dumps/step-schedule.txt" "$RUN/dumps/step.txt"

NATIVE_SHA="$(sha256sum "$RUN/dumps/native.txt" | awk '{print $1}')"
STEP_SHA="$(sha256sum "$RUN/dumps/step.txt" | awk '{print $1}')"
{
  echo "oracle_id $ORACLE_ID"
  echo "oracle_class arm64_linux_cpu_contract_off"
  echo "counts 0"
  echo "aarch64_counts false"
  echo "mujoco_version 4.0.0"
  echo "mujoco_commit $MUJOCO_COMMIT"
  echo "menagerie_commit $MENAGERIE_COMMIT"
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname $(hostname)"
  echo "host_kind $HOST_KIND"
  echo "uname $(uname -s) $(uname -m)"
  echo "clang $clang_line"
  echo "cmake $cmake_line"
  echo "ninja $(ninja --version)"
  echo "flags $FLAGS"
  echo "shared_link $SHARED_LINK"
  echo "jobs 2"
  echo "compiler_rt_path $RT"
  echo "compiler_rt_sha256 $RT_SHA"
  echo "run $RUN"
  echo "libmujoco_path $SO"
  echo "libmujoco_sha256 $LIB_SHA"
  echo "native_txt $RUN/dumps/native.txt"
  echo "native_txt_sha256 $NATIVE_SHA"
  echo "step_txt $RUN/dumps/step.txt"
  echo "step_txt_sha256 $STEP_SHA"
  echo "not_313_arm_oracle $FORBIDDEN_LIB"
  echo "not_313_build $FORBIDDEN_BUILD"
  echo "not_313_revision $FORBIDDEN_REV"
  if [[ -f "$PLUG/libactuator.so" ]]; then
    echo "libactuator_sha256 $(sha256sum "$PLUG/libactuator.so" | awk '{print $1}')"
  fi
  if [[ -f "$PLUG/libsensor.so" ]]; then
    echo "libsensor_sha256 $(sha256sum "$PLUG/libsensor.so" | awk '{print $1}')"
  fi
  if [[ -f "$PLUG/libsdf_plugin.so" ]]; then
    echo "libsdf_plugin_sha256 $(sha256sum "$PLUG/libsdf_plugin.so" | awk '{print $1}')"
  fi
  echo "schema $(tr '\n' ' ' <"$RUN/dumps/schema.txt")"
  cat "$RUN/provenance/energy-lock.txt"
} >"$RUN/RECEIPT.txt"
sha256sum "$RUN/RECEIPT.txt" | tee "$RUN/RECEIPT.sha256"
echo "receipt $RUN/RECEIPT.txt"
echo "native $RUN/dumps/native.txt"
echo "step $RUN/dumps/step.txt"
