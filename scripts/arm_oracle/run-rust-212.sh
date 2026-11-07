#!/usr/bin/env bash
# Compare ferromotion to the ARM oracle dumps. Completing with mismatches is not a bit-exact pass.
set -euo pipefail
export LC_ALL=C
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "refusing: Linux aarch64 only" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "thor2" ]]; then
  echo "refusing: rust comparison is thor2 first" >&2
  exit 2
fi
ulimit -t 14400
export OMP_NUM_THREADS=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV="123347c0eeab7e13c8da0828ab593bbd95bcf335"
RUN="${1:?run directory from run-212.sh}"
if [[ ! -f "$RUN/native.txt" || ! -f "$RUN/step.txt" ]]; then
  echo "refusing: $RUN is missing native.txt or step.txt" >&2
  exit 2
fi
if [[ -x "${CARGO_HOME:-}/bin/cargo" ]]; then
  export PATH="$CARGO_HOME/bin:$PATH"
elif [[ -x "$HOME/.cargo/bin/cargo" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
elif ! command -v cargo >/dev/null; then
  echo "refusing: cargo is not available. See rust/OPTIONAL-RUSTUP.md. This script does not install it." >&2
  exit 2
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$HOME/ferromotion-oracle/$REV/rust212-$STAMP"
if [[ -e "$OUT" ]]; then
  echo "refusing: $OUT already exists" >&2
  exit 2
fi
mkdir "$OUT"
cd "$ROOT/ferromotion"
set +e
cargo run --release -p ferromotion-core --example menagerie_native -- \
  "$ROOT/menagerie" "$RUN/native.txt" > "$OUT/native.txt" 2>&1
native_rc=$?
cargo run --release -p ferromotion-core --example menagerie_native_step -- \
  "$ROOT/menagerie" "$RUN/step.txt" > "$OUT/step.txt" 2>&1
step_rc=$?
set -e
echo "menagerie_native_exit $native_rc" | tee "$OUT/result.txt"
echo "menagerie_native_step_exit $step_rc" | tee -a "$OUT/result.txt"
if [[ "$native_rc" -ne 0 || "$step_rc" -ne 0 ]]; then
  echo "rust comparison failed to run" >&2
  exit 1
fi
if ! grep -q "BIT FOR BIT" "$OUT/native.txt" || ! grep -q "of 212 models stepped" "$OUT/step.txt"; then
  echo "refusing: comparison summary is missing" >&2
  exit 1
fi
echo "comparison finished. Exit 0 means the runners completed, not that every array matched." | tee -a "$OUT/result.txt"
grep -E "body_inertia|body_mass|efc_J| qacc |of 212 models stepped" "$OUT/native.txt" "$OUT/step.txt" | tee -a "$OUT/result.txt" || true
