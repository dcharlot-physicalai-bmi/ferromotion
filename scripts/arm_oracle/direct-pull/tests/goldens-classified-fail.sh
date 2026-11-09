#!/usr/bin/env bash
# A cargo exit 101 for the three Darwin tests is exit 1, and the rust stage runs.
# Any other cargo result is exit 2 and does not start rust.
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
oracle="$(cd "$here/../.." && pwd)"
pull="$(cd "$here/.." && pwd)"

if ! command -v shellcheck >/dev/null 2>&1; then
  echo "refusing: shellcheck is not on PATH" >&2
  exit 1
fi
shellcheck -s bash --severity=warning \
  "$oracle/run-goldens.sh" \
  "$oracle/run-rust-212.sh" \
  "$pull/05-rust-rereun.sh" \
  "$pull/05-rust.sh" \
  "$pull/04-corpus.sh" \
  "$pull/03-cbuild.sh"

# The re-run disarms ERR around the classified goldens and rust commands.
awk '
  /trap - ERR/ { disarmed = 1 }
  disarmed && /run-goldens\.sh/ { goldens = 1 }
  goldens && /trap refuse_unexpected ERR/ { restored = 1 }
  END { exit !(disarmed && goldens && restored) }
' "$pull/05-rust-rereun.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
REV=123347c0eeab7e13c8da0828ab593bbd95bcf335
bundle="$tmp/bundle"
home="$tmp/home"
mkdir -p "$bundle/scripts" "$bundle/probes" "$bundle/schedules" \
  "$bundle/ferromotion/vendor" "$bundle/ferromotion/.cargo" \
  "$home/ferromotion-oracle/$REV/goldens-20261109T223345Z" \
  "$tmp/bin"
printf 'frozen\n' >"$home/ferromotion-oracle/$REV/goldens-20261109T223345Z/cargo-test.txt"
kept_sha="$(sha256sum "$home/ferromotion-oracle/$REV/goldens-20261109T223345Z/cargo-test.txt")"
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
cp -f "$oracle/run-goldens.sh" "$oracle/limits.sh" "$oracle/compiler-rt.sh" "$bundle/scripts/"
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
if [ "${FAKE_CARGO_RESULT:-fail3}" = "partial" ]; then
  printf '%s\n' 'test geom_frames_match_mujocos_kinematics_bit_for_bit ... FAILED'
  printf '%s\n' 'test spatial_tendons_step_as_mujoco_does_bit_for_bit ... FAILED'
  printf '%s\n' 'test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'
  exit 101
fi
printf '%s\n' 'test geom_frames_match_mujocos_kinematics_bit_for_bit ... FAILED'
printf '%s\n' 'test spatial_tendons_step_as_mujoco_does_bit_for_bit ... FAILED'
printf '%s\n' 'test the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit ... FAILED'
printf '%s\n' 'test result: FAILED. 0 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'
exit 101
EOF
cat >"$bundle/scripts/run-rust-212.sh" <<'EOF'
#!/bin/sh
printf 'rust-stage-ran\n' >"$1"
exit 0
EOF
chmod +x "$tmp/bin/cargo" "$bundle/scripts/write-receipt.sh" "$bundle/scripts/"*.sh

run_handoff() {
  local mode="$1"
  local marker="$tmp/rust-marker-${mode}"
  rm -f "$marker"
  FAKE_CARGO_RESULT="$mode" \
  FERROMOTION_ORACLE_SMOKE=x86 \
  HOME="$home" \
  PATH="$tmp/bin:$PATH" \
  bash -c '
    set -Eeuo pipefail
    refuse_unexpected() { echo "refusing: unexpected abort status $?" >&2; exit 2; }
    trap refuse_unexpected ERR
    trap - ERR
    set +e
    "$1/scripts/run-goldens.sh"
    goldens_rc=$?
    set -e
    trap refuse_unexpected ERR
    if [[ "$goldens_rc" -eq 1 ]]; then
      "$1/scripts/run-rust-212.sh" "$2"
      exit 0
    fi
    if [[ "$goldens_rc" -eq 0 ]]; then
      exit 0
    fi
    echo "refusing: golden wrapper exit $goldens_rc" >&2
    exit 2
  ' bash "$bundle" "$marker" >"$tmp/hand-${mode}.out" 2>"$tmp/hand-${mode}.err"
}

set +e
run_handoff fail3
fail3_rc=$?
run_handoff partial
partial_rc=$?
set -e
[[ "$fail3_rc" -eq 0 ]]
[[ -f "$tmp/rust-marker-fail3" ]]
grep -q 'rust-stage-ran' "$tmp/rust-marker-fail3"
grep -R -q 'darwin_golden_tests_exit 1' "$home/ferromotion-oracle/$REV" --include result.txt
[[ "$partial_rc" -eq 2 ]]
[[ ! -f "$tmp/rust-marker-partial" ]]
now_sha="$(sha256sum "$home/ferromotion-oracle/$REV/goldens-20261109T223345Z/cargo-test.txt")"
[[ "$now_sha" == "$kept_sha" ]]
grep -q 'left_untouched' "$tmp/hand-fail3.out"
echo "goldens-classified-fail ok"
