#!/usr/bin/env bash
# receipt_schema direct-pull-v1
# Binds the files a direct-pull run actually has. archive_manifest_rebuilt is no.
# Archive paths such as scripts/README.md, rust/OPTIONAL-RUSTUP.md, and
# provenance/REVISIONS are not required and are not created by this script.
# Every non-preflight role requires identity.txt. build and direct-pull-build
# use the stage-3 file. probe, corpus, and rust use the identity the runner
# writes in the output directory: role, stage-3 path, and compiler-rt path
# and sha256. goldens writes identity_role goldens and does not bind libmujoco.
set -euo pipefail
export LC_ALL=C
if [[ "${1:-}" == "--preflight" ]]; then
  PREFLIGHT=1
  RUN="${2:?run directory}"
  BUILD="${3:-}"
  LIBDIR="${4:-}"
  ROLE="${5:-direct-pull-build}"
elif [[ "$(basename "$0")" == "write-receipt.sh" ]]; then
  # Drop-in for the runners, which call write-receipt.sh OUT LIBDIR ROLE.
  PREFLIGHT=0
  RUN="$(cd "$(dirname "$0")/.." && pwd)"
  BUILD="${1:?output directory}"
  LIBDIR="${2:?library directory}"
  ROLE="${3:-build}"
else
  PREFLIGHT=0
  RUN="${1:?run directory}"
  BUILD="${2:?build directory}"
  LIBDIR="${3:?library directory}"
  ROLE="${4:-direct-pull-build}"
fi

fail() { echo "refusing: $*" >&2; exit 2; }

sha_of() {
  local file="$1"
  sha256sum "$file" | awk '{print $1}'
}

# Paths relative to the run directory. stage4 marks files the corpus runners open
# that stage 1 of ecfccc6 did not copy.
stage1_files=(
  probes/oracle_dump.c
  probes/pyhex.h
  probes/hexfloat_selftest.c
  probes/expected-load-failures.txt
  scripts/limits.sh
  scripts/apply-overlay.sh
  scripts/accept-rust-212.sh
  scripts/write-receipt.sh
  scripts/build-thor2.sh
  scripts/run-212.sh
  scripts/run-goldens.sh
  scripts/run-probes.sh
  scripts/run-rust-212.sh
  fixtures/x86-contract-off-mulquat.txt
  ferromotion/scripts/arm_oracle/README.md
  ferromotion/scripts/arm_oracle/OPTIONAL-RUSTUP.md
  ferromotion/Cargo.lock
  ferromotion/.cargo/config.toml
  provenance/stage2-ok
  provenance/cargo-vv.txt
)
schedule_files=(
  schedules/x86-native.txt
  schedules/x86-step.txt
)
stage4_files=(
  probes/mulquat.c
  scripts/compiler-rt.sh
  scripts/read-mulquat-fixture.sh
)
repos=(
  "mujoco|src/mujoco|123347c0eeab7e13c8da0828ab593bbd95bcf335|c0408a44f164299fb200258a9226de55f508c59d"
  "ccd|deps/ccd|7931e764a19ef6b21b443376c699bbc9c6d4fba8|24104c36f6f32113dffaddd9e339f3cca64f9939"
  "qhull|deps/qhull|d1c2fc0caa5f644f3a0f220290d4a868c68ed4f6|d628b21c2a676f71cf13208dc49d7336fe698104"
  "tinyxml2|deps/tinyxml2|e6caeae85799003f4ca74ff26ee16a789bc2af48|b0095c6d2d899222047e22b4b164fc3979393c7e"
  "tinyobjloader|deps/tinyobjloader|2945a967c5303b2c8c14174117c45f3302591150|bade1d1afeefc9c96bef58af4445a06485d2e1b4"
  "lodepng|deps/lodepng|17d08dd26cac4d63f43af217ebd70318bfb8189c|bfe03b87a19df25bf9c39df84f95f7b90551c87e"
  "miniz|deps/miniz|d10b03cc73475af673df40f06e5cefd1d5f940d9|33cd1ab4b9966f08dab259d63929fc001b69aab5"
  "marchingcubecpp|deps/marchingcubecpp|f03a1b3ec29b1d7d865691ca8aea4f1eb2c2873d|a367ee1b40674af441f3984159dc17ba84c45a29"
  "menagerie|menagerie|4d038b3feae26ec82b46a4d586379114012a8ac7|1fb749524d966e3ab6c52d94c7864e2c19a01de7"
  "ferromotion|ferromotion|e0557dd109151a02702dfb262ea0a8c2b9dd07db|b30b3d5a5600110e1b871d931aceecfb284b5b67"
)

missing=()
note_missing() { missing+=("$1"); echo "missing $1"; }

for rel in "${stage1_files[@]}" "${schedule_files[@]}"; do
  [[ -f "$RUN/$rel" ]] || note_missing "$RUN/$rel"
done
for rel in "${stage4_files[@]}"; do
  if [[ ! -f "$RUN/$rel" ]]; then
    echo "missing_for_stage4 $RUN/$rel"
    if [[ "$ROLE" == "corpus" || "$ROLE" == "probe" ]]; then
      missing+=("$RUN/$rel")
    fi
  fi
done
if [[ "$ROLE" != "goldens" && ( "$PREFLIGHT" -eq 0 || -n "$LIBDIR" ) ]]; then
  for lib in libmujoco.so.3.13.0 libactuator.so libelasticity.so libsensor.so libsdf_plugin.so; do
    [[ -n "$LIBDIR" && -f "$LIBDIR/$lib" ]] || note_missing "${LIBDIR:-LIBDIR}/$lib"
  done
fi
if [[ "$PREFLIGHT" -eq 0 ]]; then
  [[ -f "$BUILD/identity.txt" ]] || note_missing "$BUILD/identity.txt"
fi
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "refusing: ${#missing[@]} required direct-pull paths are absent" >&2
  exit 2
fi
if [[ "$PREFLIGHT" -eq 1 && -z "$LIBDIR" ]]; then
  echo "preflight_source_paths_ok"
  exit 0
fi
if [[ "$PREFLIGHT" -eq 1 ]]; then
  echo "preflight_ok"
  exit 0
fi

# Layout tests set this on a machine that is not thor2. Thor2 always checks the pins.
skip_pins=0
if [[ "${FERROMOTION_RECEIPT_LAYOUT_TEST:-}" == "1" && "$(hostname -s)" != "thor2" ]]; then
  skip_pins=1
fi
rt_path=""
rt_sha=""
bind_role_identity() {
  local file="$BUILD/identity.txt"
  case "$ROLE" in
    probe|corpus|rust)
      grep -F -x -q "identity_role $ROLE" "$file" || fail "$file is not a $ROLE identity"
      rt_path="$(awk '/^compiler_rt_path /{print $2}' "$file")"
      rt_sha="$(awk '/^compiler_rt_sha256 /{print $2}' "$file")"
      [[ -n "$rt_path" && -f "$rt_path" ]] || fail "compiler-rt path in $file is not a file"
      rt_got="$(sha_of "$rt_path")"
      [[ "$rt_got" == "$rt_sha" ]] || fail "compiler-rt sha256 does not match $rt_path"
      ;;
    goldens)
      grep -F -x -q "identity_role goldens" "$file" || fail "$file is not a goldens identity"
      ;;
    build|direct-pull-build) ;;
    *) fail "unknown role $ROLE" ;;
  esac
}
bind_role_identity

for spec in "${repos[@]}"; do
  IFS='|' read -r name rel commit tree <<<"$spec"
  [[ -d "$RUN/$rel/.git" ]] || fail "$RUN/$rel has no .git"
  got="$(git -C "$RUN/$rel" rev-parse HEAD)"
  got_tree="$(git -C "$RUN/$rel" rev-parse 'HEAD^{tree}')"
  if [[ "$skip_pins" -eq 0 ]]; then
    [[ "$got" == "$commit" ]] || fail "$name HEAD $got is not $commit"
    [[ "$got_tree" == "$tree" ]] || fail "$name tree $got_tree is not $tree"
  else
    [[ -n "$got" && -n "$got_tree" ]] || fail "$name has no HEAD"
  fi
done

case "$ROLE" in
  build)
    extras=("$BUILD/compiler-rt.txt")
    ;;
  direct-pull-build) extras=() ;;
  probe)
    extras=("$BUILD/mulquat" "$BUILD/mulquat-arm64.txt" "$BUILD/mulquat-x86.txt" "$BUILD/result.txt")
    ;;
  corpus)
    extras=("$BUILD/oracle_dump" "$BUILD/classify.txt" "$BUILD/failures.txt" "$BUILD/native.txt" "$BUILD/step.txt" "$BUILD/libdir.txt")
    ;;
  rust)
    extras=("$BUILD/native.txt" "$BUILD/step.txt" "$BUILD/result.txt" "$BUILD/accept.txt")
    ;;
  goldens)
    extras=("$BUILD/cargo-test.txt" "$BUILD/result.txt")
    ;;
  *) fail "unknown role $ROLE" ;;
esac
if [[ "${#extras[@]}" -gt 0 ]]; then
  for path in "${extras[@]}"; do
    [[ -f "$path" ]] || note_missing "$path"
  done
fi
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "refusing: ${#missing[@]} role outputs are absent" >&2
  exit 2
fi

receipt="$BUILD/RECEIPT.txt"
tmp="$(mktemp)"
{
  echo "receipt_schema direct-pull-v1"
  echo "archive_manifest_rebuilt no"
  echo "receipt_role $ROLE"
  if [[ "$(uname -m)" == "aarch64" && "$(hostname -s)" == "thor2" ]]; then
    echo "oracle_class arm64_linux_cpu_contract_off"
  else
    echo "oracle_class not_thor2_aarch64"
  fi
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname $(hostname)"
  echo "uname_s $(uname -s)"
  echo "uname_m $(uname -m)"
  echo "uname_r $(uname -r)"
  echo "clang $(clang --version 2>/dev/null | head -n 1 || echo absent)"
  if [[ "$ROLE" == "build" ]]; then
    [[ -f "$BUILD/compiler-rt.txt" ]] || fail "$BUILD/compiler-rt.txt is missing"
    rt_path="$(awk '/^compiler_rt_path /{print $2}' "$BUILD/compiler-rt.txt")"
    rt_sha="$(awk '/^compiler_rt_sha256 /{print $2}' "$BUILD/compiler-rt.txt")"
    [[ -n "$rt_path" && -f "$rt_path" ]] || fail "compiler-rt path is not a file"
    rt_got="$(sha_of "$rt_path")"
    [[ "$rt_got" == "$rt_sha" ]] || fail "compiler-rt sha256 does not match $rt_path"
    echo "shared_link -rtlib=compiler-rt -unwindlib=libgcc"
    cat "$BUILD/compiler-rt.txt"
  elif [[ "$ROLE" == "probe" || "$ROLE" == "corpus" || "$ROLE" == "rust" ]]; then
    echo "identity_role $ROLE"
    echo "compiler_rt_path $rt_path"
    echo "compiler_rt_sha256 $rt_sha"
    if [[ "$ROLE" == "rust" ]]; then
      echo "rust_links_libmujoco no"
    fi
  fi
  echo "cmake $(cmake --version 2>/dev/null | head -n 1 || echo absent)"
  if command -v ninja >/dev/null 2>&1; then
    echo "ninja $(ninja --version)"
  else
    echo "ninja absent"
  fi
  if [[ -f "$RUN/provenance/cargo-vv.txt" ]]; then
    awk 'NR==1 || /^release:/ || /^host:/ {print "cargo_vv "$0}' "$RUN/provenance/cargo-vv.txt"
  fi
  echo "identity_file $BUILD/identity.txt"
  echo
  echo "# source commits and trees"
  for spec in "${repos[@]}"; do
    IFS='|' read -r name rel commit tree <<<"$spec"
    echo "source_head $name $(git -C "$RUN/$rel" rev-parse HEAD)"
    echo "source_tree $name $(git -C "$RUN/$rel" rev-parse 'HEAD^{tree}')"
  done
  echo
  echo "# sha256  path"
  if [[ "$ROLE" == "goldens" ]]; then
    echo "# goldens receipt does not bind libmujoco. The corpus receipt does."
  else
    for lib in libmujoco.so.3.13.0 libactuator.so libelasticity.so libsensor.so libsdf_plugin.so; do
      sha256sum "$LIBDIR/$lib"
    done
    if [[ -f "$LIBDIR/libmujoco.so" ]]; then
      sha256sum "$LIBDIR/libmujoco.so"
    fi
  fi
  if [[ -n "${rt_path:-}" ]]; then
    sha256sum "$rt_path"
  fi
  sha256sum "$BUILD/identity.txt"
  for rel in "${stage1_files[@]}" "${schedule_files[@]}"; do
    sha256sum "$RUN/$rel"
  done
  for rel in "${stage4_files[@]}"; do
    if [[ -f "$RUN/$rel" ]]; then
      sha256sum "$RUN/$rel"
    fi
  done
  if [[ "${#extras[@]}" -gt 0 ]]; then
    for path in "${extras[@]}"; do
      sha256sum "$path"
    done
  fi
} >"$tmp"
mv "$tmp" "$receipt"
sha256sum "$receipt" >"$BUILD/RECEIPT.sha256"
echo "wrote $receipt"
