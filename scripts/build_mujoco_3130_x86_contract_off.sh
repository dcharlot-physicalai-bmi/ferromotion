#!/usr/bin/env bash
# Named oracle: mujoco-3.13.0-x86_64-contract-off
# Does not replace the PyPI wheel or the arm64 golden arrays.
set -euo pipefail
SRC="${1:-/tmp/mujoco-3.13.0-src}"
WORK="${2:-/tmp/oracle-x86-contract-off}"
# MuJoCo's cmake (`MujocoLinkOptions.cmake`) passes `-fuse-ld=lld` when that
# probe succeeds, and `-fuse-ld=gold` only when it does not. The pin
# `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc`
# (4981136 bytes, GNU gold 1.16, build-id 150451907a8f62aad0262fee541db1d8b8126779)
# is the gold link. The same objects linked with lld are
# `8fb6d7e099e4d2e06c4c1c901ebf3829bc93a5ec89543f903e5fbd6e72a3f103`
# (4965992 bytes). That is the library the ubuntu-24.04 runner produced for
# ab3c6f6, because the image has lld and the probe succeeded. Relinking those
# objects with gold reproduces the pin. The lld library's 20-step dump still
# hashed to `a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be`.
# The hash check is unchanged and the pin is not the lld library.
# The wrappers rewrite the probe and the link onto gold.
REAL_CC="$(command -v clang)"
REAL_CXX="$(command -v clang++)"
if ! command -v ld.gold >/dev/null 2>&1; then
  echo "ld.gold is required to reproduce the 64156abe pin" >&2
  exit 1
fi
wrap="$WORK/compiler"
mkdir -p "$wrap"
write_wrapper() {
  local dest="$1" real="$2"
  cat > "$dest" << EOF
#!/bin/bash
args=()
for a in "\$@"; do
  if [[ "\$a" == "-fuse-ld=lld" ]]; then
    args+=("-fuse-ld=gold")
  else
    args+=("\$a")
  fi
done
exec $(printf '%q' "$real") "\${args[@]}"
EOF
  chmod +x "$dest"
}
write_wrapper "$wrap/clang" "$REAL_CC"
write_wrapper "$wrap/clang++" "$REAL_CXX"
CC_BIN="$wrap/clang"
CXX_BIN="$wrap/clang++"
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
