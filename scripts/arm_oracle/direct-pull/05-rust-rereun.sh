#!/usr/bin/env bash
# Re-run stage 5 against the fidelity-branch tip.
# That tip contains 7998624df8a4ac2591187540a59ddc9b13d51640 and is one commit later.
# The e0557dd checkout and rust212-20261109T215708Z stay where they are.
# Cargo stays offline. A missing vendored crate is listed and cargo is not started.
# Exit 1 is only the rust comparison's finished mismatch. Any other failure here is exit 2.
# A runner's exit 3 still means cargo did not finish.
set -Eeuo pipefail
export LC_ALL=C
refuse_unexpected() {
  local rc=$?
  echo "refusing: unexpected abort at line ${BASH_LINENO[0]} status $rc" >&2
  exit 2
}
trap refuse_unexpected ERR
DIRECT_PULL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$DIRECT_PULL_DIR/lib.sh"
if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "refusing: run directory that passed 04-corpus.sh" >&2
  exit 2
fi
RUN="$1"
direct_pull_hold_energy_lock "$RUN" stage5-rereun
if [[ ! -f "$RUN/provenance/stage4-ok" ]]; then
  echo "refusing: $RUN has no stage4-ok. The ARM dumps do not exist yet." >&2
  exit 2
fi
if [[ ! -d "$RUN/ferromotion/.git" || ! -d "$RUN/ferromotion/vendor" ]]; then
  echo "refusing: $RUN/ferromotion is not the e0557dd checkout with its vendor directory" >&2
  exit 2
fi
old_head="$(git -C "$RUN/ferromotion" rev-parse HEAD)"
old_tree="$(git -C "$RUN/ferromotion" rev-parse 'HEAD^{tree}')"
if [[ "$old_head" != "$FERROMOTION_COMMIT" || "$old_tree" != "$FERROMOTION_TREE" ]]; then
  echo "refusing: $RUN/ferromotion is $old_head tree $old_tree, not the e0557dd pin" >&2
  exit 2
fi
dest="$RUN/ferromotion-${FERROMOTION_RERUN_COMMIT}"
case "$dest" in
  "$RUN"/ferromotion-*) ;;
  *)
    echo "refusing: $dest is not a sibling of the e0557dd tree" >&2
    exit 2
    ;;
esac
if [[ "$dest" -ef "$RUN/ferromotion" ]]; then
  echo "refusing: the re-run directory is the e0557dd tree" >&2
  exit 2
fi
superseded="$THOR2_RUST212_SUPERSEDED"
superseded_sha=""
if [[ -f "$superseded/RECEIPT.txt" ]]; then
  superseded_sha="$(sha256sum "$superseded/RECEIPT.txt")"
fi

keep_superseded() {
  if [[ -n "$superseded_sha" ]]; then
    local now
    now="$(sha256sum "$superseded/RECEIPT.txt")"
    if [[ "$now" != "$superseded_sha" ]]; then
      echo "refusing: $superseded/RECEIPT.txt changed" >&2
      exit 2
    fi
  fi
  local now_head now_tree
  now_head="$(git -C "$RUN/ferromotion" rev-parse HEAD)"
  now_tree="$(git -C "$RUN/ferromotion" rev-parse 'HEAD^{tree}')"
  if [[ "$now_head" != "$old_head" || "$now_tree" != "$old_tree" ]]; then
    echo "refusing: $RUN/ferromotion changed" >&2
    exit 2
  fi
}

fetch_codeload() {
  local tar api_json top sha tree got
  command -v python3 >/dev/null 2>&1 || {
    echo "refusing: python3 is not on PATH. This script does not install it." >&2
    exit 2
  }
  tar="$(mktemp)"
  api_json="$(mktemp)"
  curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' \
    -o "$tar" \
    "https://codeload.github.com/dcharlot-physicalai-bmi/ferromotion/tar.gz/${FERROMOTION_RERUN_COMMIT}"
  curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' \
    -H 'Accept: application/vnd.github+json' \
    -o "$api_json" \
    "https://api.github.com/repos/dcharlot-physicalai-bmi/ferromotion/commits/${FERROMOTION_RERUN_COMMIT}"
  sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha"])' "$api_json")"
  tree="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"]["tree"]["sha"])' "$api_json")"
  rm -f "$api_json"
  if [[ "$sha" != "$FERROMOTION_RERUN_COMMIT" || "$tree" != "$FERROMOTION_RERUN_TREE" ]]; then
    rm -f "$tar"
    echo "refusing: GitHub commit $sha tree $tree is not ${FERROMOTION_RERUN_COMMIT} ${FERROMOTION_RERUN_TREE}" >&2
    exit 2
  fi
  mkdir "$dest"
  tar -xzf "$tar" -C "$dest" --strip-components=1
  rm -f "$tar"
  git -C "$dest" init -q
  git -C "$dest" config core.autocrlf false
  git -C "$dest" config core.filemode true
  git -C "$dest" add -A
  got="$(git -C "$dest" write-tree)"
  if [[ "$got" != "$FERROMOTION_RERUN_TREE" ]]; then
    echo "refusing: codeload tree $got is not ${FERROMOTION_RERUN_TREE}" >&2
    rm -rf "$dest"
    exit 2
  fi
  {
    echo "ferromotion_tested_commit $FERROMOTION_RERUN_COMMIT"
    echo "ferromotion_tested_tree $FERROMOTION_RERUN_TREE"
    echo "ferromotion_tested_contains $FERROMOTION_RERUN_CONTAINS"
    echo "ferromotion_tested_dir $dest"
    echo "ferromotion_source codeload"
    echo "ferromotion_supersedes $superseded"
  } >"$dest/PROVENANCE.txt"
}

if [[ -d "$dest" ]]; then
  if [[ ! -f "$dest/PROVENANCE.txt" ]] \
    || ! grep -F -x -q "ferromotion_tested_commit $FERROMOTION_RERUN_COMMIT" "$dest/PROVENANCE.txt" \
    || ! grep -F -x -q "ferromotion_tested_tree $FERROMOTION_RERUN_TREE" "$dest/PROVENANCE.txt"; then
    echo "refusing: $dest exists and is not the pinned re-run tree" >&2
    exit 2
  fi
  echo "reusing $dest"
else
  fetch_codeload
fi

if [[ -e "$dest/vendor" && ! -L "$dest/vendor" ]]; then
  echo "refusing: $dest/vendor is not a link to the e0557dd vendor" >&2
  exit 2
fi
if [[ ! -L "$dest/vendor" ]]; then
  ln -s "$RUN/ferromotion/vendor" "$dest/vendor"
fi
# Fresh and reused trees both land here. PROVENANCE is left as fetched.
direct_pull_install_offline_cargo_config "$dest"

export CARGO_NET_OFFLINE=true
export RUSTUP_AUTO_INSTALL=0
missing_list="$RUN/provenance/stage5-rereun-missing.txt"
set +e
direct_pull_list_missing_vendor "$dest/Cargo.lock" "$RUN/ferromotion/vendor" | tee "$missing_list"
vendor_rc="${PIPESTATUS[0]}"
set -e
if [[ "$vendor_rc" -ne 0 ]]; then
  echo "refusing: vendored deps do not cover ${FERROMOTION_RERUN_COMMIT}. Cargo was not started." >&2
  echo "missing list: $missing_list" >&2
  keep_superseded
  exit 2
fi
rm -f "$missing_list"

runner_src=""
if [[ -f "$DIRECT_PULL_DIR/runners/run-rust-212.sh" ]]; then
  runner_src="$DIRECT_PULL_DIR/runners"
elif [[ -f "$DIRECT_PULL_DIR/../run-rust-212.sh" ]]; then
  runner_src="$DIRECT_PULL_DIR/.."
else
  echo "refusing: run-rust-212.sh is not beside this script or in scripts/arm_oracle" >&2
  exit 2
fi
mkdir -p "$RUN/scripts" "$RUN/provenance"
for f in compiler-rt.sh limits.sh run-goldens.sh run-rust-212.sh; do
  cp -f "$runner_src/$f" "$RUN/scripts/$f"
done
if [[ -f "$DIRECT_PULL_DIR/direct-pull-receipt.sh" ]]; then
  cp -f "$DIRECT_PULL_DIR/direct-pull-receipt.sh" "$RUN/scripts/write-receipt.sh"
fi
chmod +x "$RUN/scripts/"*.sh

export ARM_ORACLE_FERROMOTION="$dest"
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_HOME="$RUN/cargo-home"
unset CARGO_TARGET_DIR || true
set +e
"$RUN/scripts/run-goldens.sh" | tee "$RUN/provenance/stage5-rereun-goldens.log"
goldens_rc="${PIPESTATUS[0]}"
set -e
if [[ "$goldens_rc" -ne 0 && "$goldens_rc" -ne 1 ]]; then
  keep_superseded
  echo "execution: golden wrapper exit $goldens_rc" >&2
  if [[ "$goldens_rc" -eq 3 ]]; then
    exit 3
  fi
  exit 2
fi
corpus_run="$(grep -E '^wrote .+\/native\.txt and .+\/step\.txt$' "$RUN/provenance/stage4-212.log" | tail -n 1)"
corpus_run="${corpus_run#wrote }"
corpus_run="${corpus_run%% and *}"
corpus_run="$(dirname "$corpus_run")"
if [[ ! -f "$corpus_run/native.txt" || ! -f "$corpus_run/libdir.txt" ]]; then
  echo "refusing: stage 4 log did not name a finished corpus directory" >&2
  exit 2
fi
set +e
"$RUN/scripts/run-rust-212.sh" "$corpus_run" | tee "$RUN/provenance/stage5-rereun-rust.log"
rust_rc="${PIPESTATUS[0]}"
set -e
{
  echo "ferromotion_tested_commit $FERROMOTION_RERUN_COMMIT"
  echo "ferromotion_tested_tree $FERROMOTION_RERUN_TREE"
  echo "ferromotion_tested_contains $FERROMOTION_RERUN_CONTAINS"
  echo "darwin_golden_wrapper_exit $goldens_rc"
  echo "rust_212_wrapper_exit $rust_rc"
  echo "supersedes $superseded"
} | tee "$RUN/provenance/stage5-rereun-ok"
keep_superseded
# 0 is parity. 1 is a finished mismatch. 3 is cargo that did not finish.
case "$rust_rc" in
  0|1|3) exit "$rust_rc" ;;
  *)
    echo "refusing: rust wrapper exit $rust_rc" >&2
    exit 2
    ;;
esac
