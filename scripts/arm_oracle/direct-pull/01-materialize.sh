#!/usr/bin/env bash
# Stage 1. Public HTTPS only. No credential helper, no prompt, no rustup install.
# Creates one new directory and stops. It does not configure or compile MuJoCo.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
direct_pull_need git
direct_pull_need curl
direct_pull_need sha256sum
direct_pull_need timeout
RUN="$(direct_pull_run_root)"
mkdir -p "$RUN/provenance" "$RUN/deps" "$RUN/probes" "$RUN/scripts" "$RUN/fixtures" "$RUN/schedules"
direct_pull_askpass "$RUN/provenance"
{
  echo "run $RUN"
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname $(uname -s) $(uname -m)"
} | tee "$RUN/provenance/stage1-start.txt"

echo "preflight unauthenticated API"
direct_pull_http_200 https://api.github.com/repos/dcharlot-physicalai-bmi/ferromotion
direct_pull_http_200 https://api.github.com/repos/google-deepmind/mujoco
direct_pull_http_200 https://api.github.com/repos/google-deepmind/mujoco_menagerie
direct_pull_http_200 https://api.github.com/repos/danfis/libccd
direct_pull_http_200 https://api.github.com/repos/qhull/qhull
direct_pull_http_200 https://api.github.com/repos/leethomason/tinyxml2
direct_pull_http_200 https://api.github.com/repos/tinyobjloader/tinyobjloader
direct_pull_http_200 https://api.github.com/repos/lvandeve/lodepng
direct_pull_http_200 https://api.github.com/repos/richgel999/miniz
direct_pull_http_200 https://api.github.com/repos/aparis69/MarchingCubeCpp

direct_pull_fetch "$RUN/src/mujoco" "$MUJOCO_URL" "$MUJOCO_COMMIT" "$MUJOCO_TREE"
direct_pull_fetch "$RUN/deps/ccd" "$CCD_URL" "$CCD_COMMIT" "$CCD_TREE"
direct_pull_fetch "$RUN/deps/qhull" "$QHULL_URL" "$QHULL_COMMIT" "$QHULL_TREE"
direct_pull_fetch "$RUN/deps/tinyxml2" "$TINYXML2_URL" "$TINYXML2_COMMIT" "$TINYXML2_TREE"
direct_pull_fetch "$RUN/deps/tinyobjloader" "$TINYOBJ_URL" "$TINYOBJ_COMMIT" "$TINYOBJ_TREE"
direct_pull_fetch "$RUN/deps/lodepng" "$LODEPNG_URL" "$LODEPNG_COMMIT" "$LODEPNG_TREE"
direct_pull_fetch "$RUN/deps/miniz" "$MINIZ_URL" "$MINIZ_COMMIT" "$MINIZ_TREE"
direct_pull_fetch "$RUN/deps/marchingcubecpp" "$MARCHING_URL" "$MARCHING_COMMIT" "$MARCHING_TREE"
direct_pull_fetch "$RUN/ferromotion" "$FERROMOTION_URL" "$FERROMOTION_COMMIT" "$FERROMOTION_TREE"
direct_pull_fetch "$RUN/menagerie" "$MENAGERIE_URL" "$MENAGERIE_COMMIT" "$MENAGERIE_TREE"

direct_pull_require_sha "$DIRECT_PULL_DIR/patches/ccd.patch" "$CCD_PATCH_SHA256"
direct_pull_require_sha "$DIRECT_PULL_DIR/patches/qhull.patch" "$QHULL_PATCH_SHA256"
gitc -C "$RUN/deps/ccd" apply "$DIRECT_PULL_DIR/patches/ccd.patch"
gitc -C "$RUN/deps/qhull" apply "$DIRECT_PULL_DIR/patches/qhull.patch"
direct_pull_require_sha "$RUN/deps/ccd/src/CMakeLists.txt" "$CCD_CMAKE_SHA256"
direct_pull_require_sha "$RUN/deps/qhull/CMakeLists.txt" "$QHULL_CMAKE_SHA256"
cp -f "$RUN/src/mujoco/cmake/third_party_deps/lodepng/CMakeLists.txt" "$RUN/deps/lodepng/CMakeLists.txt"
direct_pull_require_sha "$RUN/deps/lodepng/CMakeLists.txt" "$LODEPNG_CMAKE_SHA256"

direct_pull_require_sha "$DIRECT_PULL_DIR/fixtures/x86-contract-off-mulquat.txt" "$MULQUAT_FIXTURE_SHA256"
cp -f "$DIRECT_PULL_DIR/fixtures/x86-contract-off-mulquat.txt" "$RUN/fixtures/x86-contract-off-mulquat.txt"
direct_pull_require_sha "$RUN/fixtures/x86-contract-off-mulquat.txt" "$MULQUAT_FIXTURE_SHA256"

# Runners and probes are the files inside the pinned ferromotion commit.
cp -f "$RUN/ferromotion/scripts/arm_oracle/"*.sh "$RUN/scripts/"
cp -f "$RUN/ferromotion/scripts/arm_oracle/oracle_dump.c" "$RUN/probes/oracle_dump.c"
cp -f "$RUN/ferromotion/scripts/arm_oracle/pyhex.h" "$RUN/probes/pyhex.h"
cp -f "$RUN/ferromotion/scripts/arm_oracle/hexfloat_selftest.c" "$RUN/probes/hexfloat_selftest.c"
cp -f "$RUN/ferromotion/scripts/arm_oracle/expected-load-failures.txt" "$RUN/probes/expected-load-failures.txt"
if [[ -f "$DIRECT_PULL_DIR/probes/mulquat.c" ]]; then
  cp -f "$DIRECT_PULL_DIR/probes/mulquat.c" "$RUN/probes/mulquat.c"
fi
chmod +x "$RUN/scripts/"*.sh

direct_pull_need cargo
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_HOME="$RUN/cargo-home"
unset CARGO_TARGET_DIR || true
export CARGO_NET_OFFLINE=false
if ! cargo +1.99.1 -vV >"$RUN/provenance/cargo-vv.txt"; then
  echo "refusing: cargo +1.99.1 is not installed. RUSTUP_AUTO_INSTALL=0, so this script did not install it." >&2
  exit 2
fi
if ! grep -q "^release: ${RUSTC_RELEASE}$" "$RUN/provenance/cargo-vv.txt"; then
  echo "refusing: cargo +1.99.1 is not release ${RUSTC_RELEASE}" >&2
  cat "$RUN/provenance/cargo-vv.txt" >&2
  exit 2
fi
(
  cd "$RUN/ferromotion"
  cargo +1.99.1 vendor --locked vendor
)
mkdir -p "$RUN/ferromotion/.cargo"
cat >"$RUN/ferromotion/.cargo/config.toml" <<'EOF'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"

[net]
offline = true
EOF
direct_pull_require_sha "$RUN/ferromotion/Cargo.lock" "$FERROMOTION_CARGO_LOCK_SHA256"

echo "stage1-ok $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN/provenance/stage1-ok"
echo "RUN $RUN"
echo "Stage 1 finished. It did not compile. Run 02-verify.sh on this directory before any build."
