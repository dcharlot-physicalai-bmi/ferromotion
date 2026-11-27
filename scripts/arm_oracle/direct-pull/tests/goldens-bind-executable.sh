#!/usr/bin/env bash
# The lib test executable is cargo's JSON `executable` field.
# A binary that exists only under debug/deps is not accepted, and a binary that
# exists only under debug/build/ferromotion-core/<hash>/out is.
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
oracle="$(cd "$here/../.." && pwd)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
REV=123347c0eeab7e13c8da0828ab593bbd95bcf335
bundle="$tmp/bundle"
home="$tmp/home"
mkdir -p "$bundle/scripts" "$bundle/probes" "$bundle/schedules" \
  "$bundle/ferromotion/vendor" "$bundle/ferromotion/.cargo" \
  "$home/ferromotion-oracle/$REV" "$tmp/bin"
cat >"$bundle/ferromotion/.cargo/config.toml" <<'EOF'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "vendor"

[net]
offline = true
EOF
cat >"$bundle/ferromotion/PROVENANCE.txt" <<'EOF'
ferromotion_tested_commit 515a7dec4b4ed2fa7257f328c7c3f9b169b8d7ba
ferromotion_tested_tree 67e20a00014302498e9b350862549f9d705911a6
EOF
cp -f "$oracle/run-goldens.sh" "$oracle/limits.sh" "$oracle/compiler-rt.sh" "$oracle/toolchain.sh" "$bundle/scripts/"
cat >"$bundle/scripts/write-receipt.sh" <<'EOF'
#!/bin/sh
printf 'receipt\n' >"$1/RECEIPT.txt"
exit 0
EOF
cat >"$tmp/bin/cargo" <<'EOF'
#!/bin/sh
if [ "$1" = "-vV" ]; then
  printf '%s\n' 'release: 1.99.1'
  exit 0
fi
for a in "$@"; do
  if [ "$a" = "--message-format=json" ]; then
    case "${FAKE_CARGO_BIND:-buildout}" in
      buildout)
        dest="${CARGO_TARGET_DIR}/debug/build/ferromotion-core/abchash/out/ferromotion_core-abchash"
        mkdir -p "$(dirname "$dest")"
        printf '#!/bin/sh\nexit 0\n' >"$dest"
        chmod +x "$dest"
        python3 -c 'import json,sys; json.dump({"reason":"compiler-artifact","package_id":"path+file:///src/crates/ferromotion-core#0.73.0","target":{"kind":["lib"],"name":"ferromotion_core","test":True},"profile":{"test":True},"executable":sys.argv[1]}, sys.stdout); print()' "$dest"
        printf '%s\n' '{"reason":"compiler-artifact","package_id":"path+file:///src/crates/build-script-build#0.0.0","target":{"kind":["custom-build"],"name":"build-script-build"},"profile":{"test":false},"executable":null}'
        ;;
      deps-only)
        decoy="${CARGO_TARGET_DIR}/debug/deps/ferromotion_core-decoy"
        mkdir -p "$(dirname "$decoy")"
        printf '#!/bin/sh\nexit 0\n' >"$decoy"
        chmod +x "$decoy"
        printf '%s\n' '{"reason":"compiler-artifact","package_id":"path+file:///src/crates/ferromotion-core#0.73.0","target":{"kind":["lib"],"name":"ferromotion_core"},"profile":{"test":true},"executable":null}'
        ;;
      *)
        echo "fixture cargo: unknown FAKE_CARGO_BIND" >&2
        exit 2
        ;;
    esac
    exit 0
  fi
done
printf '%s\n' 'test geom_frames_match_mujocos_kinematics_bit_for_bit ... FAILED'
printf '%s\n' 'test spatial_tendons_step_as_mujoco_does_bit_for_bit ... FAILED'
printf '%s\n' 'test the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit ... FAILED'
printf '%s\n' 'test result: FAILED. 0 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'
exit 101
EOF
chmod +x "$tmp/bin/cargo" "$bundle/scripts/write-receipt.sh" "$bundle/scripts/"*.sh

run_one() {
  local mode="$1"
  FAKE_CARGO_BIND="$mode" \
  FERROMOTION_ORACLE_SMOKE=x86 \
  HOME="$home" \
  PATH="$tmp/bin:$PATH" \
    "$bundle/scripts/run-goldens.sh" >"$tmp/${mode}.out" 2>"$tmp/${mode}.err"
}

set +e
run_one buildout
build_rc=$?
set -e
[[ "$build_rc" -eq 1 ]]
out_dir="$(find "$home/ferromotion-oracle/$REV" -type d -name 'goldens-*' ! -name 'goldens-20261109T223345Z')"
[[ -n "$out_dir" ]]
grep -q 'debug/build/ferromotion-core/abchash/out/ferromotion_core-abchash' "$out_dir/toolchain-bins.txt"
if grep -q 'debug/deps/' "$out_dir/toolchain-bins.txt"; then
  echo "build-out bind also recorded a deps binary" >&2
  exit 1
fi

# The output directory name is the UTC second. The second run needs its own.
sleep 1
set +e
run_one deps-only
deps_rc=$?
set -e
[[ "$deps_rc" -eq 2 ]]
if ! grep -q 'cargo JSON recorded no lib test executable' "$tmp/deps-only.err"; then
  echo "deps-only stderr was:" >&2
  cat "$tmp/deps-only.err" >&2
  exit 1
fi
if grep -R -q 'ferromotion_core-decoy' "$home/ferromotion-oracle/$REV" --include toolchain-bins.txt; then
  echo "a deps-only binary was bound" >&2
  exit 1
fi
echo "goldens-bind-executable ok"
