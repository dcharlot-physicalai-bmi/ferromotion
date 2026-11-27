#!/usr/bin/env bash
# Stage 2. No network. Re-checks commit, tree, patch, fixture, and vendored lock.
# A passing check is not a claim that the V2 archive manifest was reconstructed.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
RUN="${1:?run directory from 01-materialize.sh}"
direct_pull_hold_energy_lock "$RUN" stage2
if [[ ! -f "$RUN/provenance/stage1-ok" ]]; then
  echo "refusing: $RUN has no stage1-ok" >&2
  exit 2
fi
direct_pull_askpass "$RUN/provenance"
export CARGO_NET_OFFLINE=true
export CARGO_HOME="$RUN/cargo-home"
unset CARGO_TARGET_DIR || true
export PATH="${HOME}/.cargo/bin:${PATH}"
export RUSTUP_AUTO_INSTALL=0

check_clean_pin() {
  local dest="$1" commit="$2" tree="$3" got tree_got dirty
  got="$(gitc -C "$dest" rev-parse HEAD)"
  tree_got="$(gitc -C "$dest" rev-parse 'HEAD^{tree}')"
  if [[ "$got" != "$commit" || "$tree_got" != "$tree" ]]; then
    echo "refusing: $dest HEAD $got tree $tree_got" >&2
    exit 2
  fi
  dirty="$(gitc -C "$dest" status --porcelain)"
  if [[ -n "$dirty" ]]; then
    echo "refusing: $dest worktree is not the pinned commit" >&2
    printf '%s\n' "$dirty" >&2
    exit 2
  fi
  echo "verified $commit tree $tree_got $dest"
}

check_clean_pin "$RUN/src/mujoco" "$MUJOCO_COMMIT" "$MUJOCO_TREE"
check_clean_pin "$RUN/deps/tinyxml2" "$TINYXML2_COMMIT" "$TINYXML2_TREE"
check_clean_pin "$RUN/deps/tinyobjloader" "$TINYOBJ_COMMIT" "$TINYOBJ_TREE"
check_clean_pin "$RUN/deps/miniz" "$MINIZ_COMMIT" "$MINIZ_TREE"
check_clean_pin "$RUN/deps/marchingcubecpp" "$MARCHING_COMMIT" "$MARCHING_TREE"
check_clean_pin "$RUN/menagerie" "$MENAGERIE_COMMIT" "$MENAGERIE_TREE"

for dest in "$RUN/deps/ccd" "$RUN/deps/qhull" "$RUN/deps/lodepng" "$RUN/ferromotion"; do
  got="$(gitc -C "$dest" rev-parse HEAD)"
  tree_got="$(gitc -C "$dest" rev-parse 'HEAD^{tree}')"
  echo "verified_head $got tree $tree_got $dest"
done
if [[ "$(gitc -C "$RUN/deps/ccd" rev-parse HEAD)" != "$CCD_COMMIT" || "$(gitc -C "$RUN/deps/ccd" rev-parse 'HEAD^{tree}')" != "$CCD_TREE" ]]; then
  echo "refusing: ccd commit or tree moved" >&2
  exit 2
fi
if [[ "$(gitc -C "$RUN/deps/qhull" rev-parse HEAD)" != "$QHULL_COMMIT" || "$(gitc -C "$RUN/deps/qhull" rev-parse 'HEAD^{tree}')" != "$QHULL_TREE" ]]; then
  echo "refusing: qhull commit or tree moved" >&2
  exit 2
fi
if [[ "$(gitc -C "$RUN/deps/lodepng" rev-parse HEAD)" != "$LODEPNG_COMMIT" || "$(gitc -C "$RUN/deps/lodepng" rev-parse 'HEAD^{tree}')" != "$LODEPNG_TREE" ]]; then
  echo "refusing: lodepng commit or tree moved" >&2
  exit 2
fi
if [[ "$(gitc -C "$RUN/ferromotion" rev-parse HEAD)" != "$FERROMOTION_COMMIT" || "$(gitc -C "$RUN/ferromotion" rev-parse 'HEAD^{tree}')" != "$FERROMOTION_TREE" ]]; then
  echo "refusing: ferromotion commit or tree moved" >&2
  exit 2
fi
direct_pull_require_sha "$RUN/deps/ccd/src/CMakeLists.txt" "$CCD_CMAKE_SHA256"
direct_pull_require_sha "$RUN/deps/qhull/CMakeLists.txt" "$QHULL_CMAKE_SHA256"
direct_pull_require_sha "$RUN/deps/lodepng/CMakeLists.txt" "$LODEPNG_CMAKE_SHA256"
direct_pull_require_sha "$RUN/ferromotion/Cargo.lock" "$FERROMOTION_CARGO_LOCK_SHA256"
direct_pull_require_sha "$RUN/fixtures/x86-contract-off-mulquat.txt" "$MULQUAT_FIXTURE_SHA256"

fm_dirty="$(gitc -C "$RUN/ferromotion" status --porcelain | grep -v -E '^\?\? vendor/|^\?\? \.cargo/' || true)"
if [[ -n "$fm_dirty" ]]; then
  echo "refusing: ferromotion has changes outside vendor/ and .cargo/" >&2
  printf '%s\n' "$fm_dirty" >&2
  exit 2
fi
if ! grep -q 'offline = true' "$RUN/ferromotion/.cargo/config.toml"; then
  echo "refusing: vendored config does not set net.offline" >&2
  exit 2
fi
(
  cd "$RUN/ferromotion"
  cargo +1.99.1 metadata --offline --locked --format-version 1 >"$RUN/provenance/cargo-metadata.json"
)
echo "cargo_metadata_offline_locked_ok" | tee "$RUN/provenance/stage2-cargo.txt"
{
  echo "stage2-ok $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ferromotion $FERROMOTION_COMMIT tree $FERROMOTION_TREE"
  echo "mujoco $MUJOCO_COMMIT tree $MUJOCO_TREE"
  echo "menagerie $MENAGERIE_COMMIT tree $MENAGERIE_TREE"
  echo "archive_manifest_rebuilt no"
  echo "schedules_present no_unless_later_supplied"
} | tee "$RUN/provenance/stage2-ok"
echo "Stage 2 finished. Commit trees match the pins. The V2 archive manifest was not rebuilt."
