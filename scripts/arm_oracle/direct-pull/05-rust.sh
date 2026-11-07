#!/usr/bin/env bash
# Stage 5. Darwin golden tests, then the 212-model Rust comparison.
# Uses the already installed cargo +1.99.1. Does not install a toolchain.
# Exit 0 from run-rust-212.sh is full parity only.
set -euo pipefail
export LC_ALL=C
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
RUN="${1:?run directory that passed 04-corpus.sh}"
if [[ ! -f "$RUN/provenance/stage4-ok" ]]; then
  echo "refusing: $RUN has no stage4-ok. The ARM dumps do not exist yet." >&2
  exit 2
fi
export RUSTUP_AUTO_INSTALL=0
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_HOME="$RUN/cargo-home"
unset CARGO_TARGET_DIR || true
# The runners set CARGO_NET_OFFLINE, --locked, --jobs 2, and an external target dir.
# Exit 1 is a finished Darwin mismatch and is kept. Exit 2 or 3 stops this stage.
set +e
"$RUN/scripts/run-goldens.sh" | tee "$RUN/provenance/stage5-goldens.log"
goldens_rc="${PIPESTATUS[0]}"
set -e
if [[ "$goldens_rc" -ne 0 && "$goldens_rc" -ne 1 ]]; then
  echo "execution: golden wrapper exit $goldens_rc" >&2
  exit "$goldens_rc"
fi
corpus_run="$(awk '/^wrote /{print $2}' "$RUN/provenance/stage4-212.log" | tail -n 1)"
corpus_run="$(dirname "$corpus_run")"
if [[ ! -f "$corpus_run/native.txt" || ! -f "$corpus_run/libdir.txt" ]]; then
  echo "refusing: stage 4 log did not name a finished corpus directory" >&2
  exit 3
fi
set +e
"$RUN/scripts/run-rust-212.sh" "$corpus_run" | tee "$RUN/provenance/stage5-rust.log"
rust_rc="${PIPESTATUS[0]}"
set -e
{
  echo "darwin_golden_wrapper_exit $goldens_rc"
  echo "rust_212_wrapper_exit $rust_rc"
} | tee "$RUN/provenance/stage5-ok"
# Preserve the rust wrapper's exit. 0 is full parity. 1 is a finished mismatch.
exit "$rust_rc"
