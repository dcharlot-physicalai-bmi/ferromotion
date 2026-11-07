#!/usr/bin/env bash
# Compare the embedded Darwin arm64 goldens. Does not edit them and does not call a mismatch a Linux pass.
set -euo pipefail
export LC_ALL=C
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "refusing: Linux aarch64 only" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "thor2" ]]; then
  echo "refusing: golden comparison is thor2 first" >&2
  exit 2
fi
ulimit -t 14400
export OMP_NUM_THREADS=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
if [[ ! -x "${CARGO_HOME:-}/bin/cargo" && ! -x "$HOME/.cargo/bin/cargo" && ! command -v cargo >/dev/null ]]; then
  echo "refusing: cargo is not available. See rust/OPTIONAL-RUSTUP.md. This script does not install it." >&2
  exit 2
fi
if [[ -x "${CARGO_HOME:-}/bin/cargo" ]]; then
  export PATH="$CARGO_HOME/bin:$PATH"
elif [[ -x "$HOME/.cargo/bin/cargo" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/ferromotion-oracle/$REV/goldens-$STAMP"
if [[ -e "$OUT" ]]; then
  echo "refusing: $OUT already exists" >&2
  exit 2
fi
mkdir "$OUT"
cd "$ROOT/ferromotion"
set +e
cargo test -p ferromotion-core --lib -- --test-threads=2 \
  geom_frames_match_mujocos_kinematics_bit_for_bit \
  spatial_tendons_step_as_mujoco_does_bit_for_bit \
  the_native_mass_matrix_and_bias_are_mujocos_bit_for_bit \
  > "$OUT/cargo-test.txt" 2>&1
rc=$?
set -e
echo "darwin_golden_tests_exit $rc" | tee "$OUT/result.txt"
echo "These tests embed the unproven Darwin arrays. A failure is not relabeled as the Linux oracle." >&2
exit "$rc"
