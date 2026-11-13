#!/usr/bin/env bash
# Re-run stage 5 against a ferromotion commit and an explicit build policy.
# Any full 40-hex commit is accepted. The tree comes from the GitHub commit
# object, fetched with no token. The codeload tarball must reproduce that tree.
# The commit must be reachable on cursor/ferromotion-fidelity-conformance-bb9d.
# Cargo stays offline. A missing vendored crate is listed and cargo is not started.
# Each invocation uses a new sibling directory. An existing one is not overwritten.
# rust212-20261109T215708Z is the dated e0557dd receipt. It is not overwritten.
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
if [[ $# -ne 3 || -z "${2:-}" || -z "${3:-}" ]]; then
  echo "refusing: usage 05-rust-rereun.sh <run> <ferromotion commit> <build policy>" >&2
  exit 2
fi
COMMIT="$2"
POLICY="$3"
if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "refusing: $COMMIT is not a full 40-hex commit" >&2
  exit 2
fi
trap - ERR
set +e
TREE="$(direct_pull_fidelity_tree "$COMMIT")"
tree_rc=$?
set -e
trap refuse_unexpected ERR
if [[ "$tree_rc" -ne 0 || ! "$TREE" =~ ^[0-9a-f]{40}$ ]]; then
  exit 2
fi
case "$POLICY" in
  "$ORACLE_AARCH64_CONTRACT_OFF_ID") POLICY_SHA="$THOR2_LIB_MUJOCO_STAGE3" ;;
  "$ORACLE_X86_CONTRACT_OFF_ID") POLICY_SHA="$ORACLE_X86_CONTRACT_OFF_SHA256" ;;
  published-default) POLICY_SHA="published-default" ;;
  *)
    echo "refusing: build policy $POLICY is not published-default, $ORACLE_X86_CONTRACT_OFF_ID, or $ORACLE_AARCH64_CONTRACT_OFF_ID" >&2
    exit 2
    ;;
esac
dest="$RUN/ferromotion-${COMMIT}-${POLICY}"
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
if [[ -e "$dest" ]]; then
  echo "refusing: $dest exists and is not overwritten" >&2
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
  local tar got
  command -v python3 >/dev/null 2>&1 || {
    echo "refusing: python3 is not on PATH. This script does not install it." >&2
    exit 2
  }
  tar="$(mktemp)"
  curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' \
    -o "$tar" \
    "https://codeload.github.com/dcharlot-physicalai-bmi/ferromotion/tar.gz/${COMMIT}"
  if ! direct_pull_codeload_commit "$tar" "$COMMIT" >/dev/null; then
    rm -f "$tar"
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
  if [[ "$got" != "$TREE" ]]; then
    echo "refusing: codeload tree $got is not ${TREE}" >&2
    rm -rf "$dest"
    exit 2
  fi
  {
    echo "ferromotion_tested_commit $COMMIT"
    echo "ferromotion_tested_tree $TREE"
    echo "ferromotion_source codeload"
    echo "ferromotion_tree_source github-commit-object"
    echo "ferromotion_tested_dir $dest"
    echo "ferromotion_build_policy $POLICY"
    echo "ferromotion_oracle_sha256 $POLICY_SHA"
    echo "ferromotion_supersedes $superseded"
  } >"$dest/PROVENANCE.txt"
}

write_profile_receipt() {
  local status="$1" counts="$2" binding="$3"
  local stamp receipt
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN/provenance"
  receipt="$RUN/provenance/stage5-profile-${COMMIT}-${stamp}.txt"
  if [[ -e "$receipt" || -e "${receipt}.sha256" ]]; then
    echo "refusing: $receipt exists and is not overwritten" >&2
    exit 2
  fi
  {
    echo "ferromotion_tested_commit $COMMIT"
    echo "ferromotion_tested_tree $TREE"
    echo "ferromotion_build_policy $POLICY"
    echo "ferromotion_oracle_sha256 $POLICY_SHA"
    echo "dump_libmujoco_sha256 ${DUMP_LIB_SHA:-absent}"
    echo "libmujoco_thor2_pin $THOR2_LIB_MUJOCO_STAGE3"
    echo "libmujoco_matches_thor2_pin ${LIB_MATCH:-absent}"
    echo "policy_selected_sha256 ${binding:-not_selected}"
    echo "counts $counts"
    echo "status $status"
    if [[ "$status" == "policy_not_selected" ]]; then
      echo "refusing: ${COMMIT} install_policy does not select ${POLICY} bound to ${POLICY_SHA}."
      echo "The only oracle identity these commits accept is ${ORACLE_X86_CONTRACT_OFF_ID} bound to ${ORACLE_X86_CONTRACT_OFF_SHA256}."
      echo "Selecting that x86 id while the dumps were produced by libmujoco ${THOR2_LIB_MUJOCO_STAGE3} would relabel the oracle. This script does not do that."
      echo "To run the unfused profile against those dumps, install_policy in the tested commit must gain a match arm for ${ORACLE_AARCH64_CONTRACT_OFF_ID} that requires artifact sha256 ${THOR2_LIB_MUJOCO_STAGE3} and sets mul_quat Unfused, rot_vec_quat Unfused, and mul_mat_vec3 Contracted. Unknown sites and mul_mat_vec3 must be refused, the same way the x86 arm refuses them. That arm is not in this commit. Cargo was not started."
    fi
  } >"$receipt"
  sha256sum "$receipt" >"${receipt}.sha256"
  echo "wrote $receipt"
}

fetch_codeload

if [[ -e "$dest/vendor" && ! -L "$dest/vendor" ]]; then
  echo "refusing: $dest/vendor is not a link to the e0557dd vendor" >&2
  exit 2
fi
if [[ ! -L "$dest/vendor" ]]; then
  ln -s "$RUN/ferromotion/vendor" "$dest/vendor"
fi
direct_pull_install_offline_cargo_config "$dest"

export CARGO_NET_OFFLINE=true
export RUSTUP_AUTO_INSTALL=0
missing_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
missing_list="$RUN/provenance/stage5-rereun-missing-${COMMIT}-${missing_stamp}.txt"
if [[ -e "$missing_list" ]]; then
  echo "refusing: $missing_list exists and is not overwritten" >&2
  exit 2
fi
trap - ERR
set +e
direct_pull_list_missing_vendor "$dest/Cargo.lock" "$RUN/ferromotion/vendor" | tee "$missing_list"
vendor_rc="${PIPESTATUS[0]}"
set -e
trap refuse_unexpected ERR
if [[ "$vendor_rc" -ne 0 ]]; then
  echo "refusing: vendored deps do not cover ${COMMIT}. Cargo was not started." >&2
  echo "missing list: $missing_list" >&2
  keep_superseded
  exit 2
fi
rm -f "$missing_list"

corpus_run="$(grep -E '^wrote .+\/native\.txt and .+\/step\.txt$' "$RUN/provenance/stage4-212.log" | tail -n 1)"
corpus_run="${corpus_run#wrote }"
corpus_run="${corpus_run%% and *}"
corpus_run="$(dirname "$corpus_run")"
if [[ ! -f "$corpus_run/native.txt" || ! -f "$corpus_run/libdir.txt" ]]; then
  echo "refusing: stage 4 log did not name a finished corpus directory" >&2
  exit 2
fi
LIBDIR="$(tr -d '[:space:]' <"$corpus_run/libdir.txt")"
if [[ ! -f "$LIBDIR/libmujoco.so.3.13.0" ]]; then
  echo "refusing: $LIBDIR/libmujoco.so.3.13.0 is missing" >&2
  exit 2
fi
DUMP_LIB_SHA="$(sha256sum "$LIBDIR/libmujoco.so.3.13.0" | awk '{print $1}')"
if [[ "$DUMP_LIB_SHA" == "$THOR2_LIB_MUJOCO_STAGE3" ]]; then
  LIB_MATCH=yes
else
  LIB_MATCH=no
fi
echo "libmujoco_sha256 $DUMP_LIB_SHA"
echo "libmujoco_thor2_pin $THOR2_LIB_MUJOCO_STAGE3"
echo "libmujoco_matches_thor2_pin $LIB_MATCH"
# thor2's dumps are the pinned library. A different file is not that oracle.
# thor.local's fresh build is compared and recorded. The profile run still
# names the sha it measured, and it does not relabel a mismatch as 9164828b.
if [[ "$LIB_MATCH" != yes ]] && ! direct_pull_host_is_thor_local; then
  echo "refusing: dump libmujoco sha256 $DUMP_LIB_SHA is not $THOR2_LIB_MUJOCO_STAGE3" >&2
  exit 2
fi

binding=""
trap - ERR
set +e
binding="$(direct_pull_policy_binding "$dest/crates/ferromotion-core/src/fp_contract.rs" "$POLICY")"
bind_rc=$?
set -e
trap refuse_unexpected ERR
if [[ "$bind_rc" -ne 0 || "$binding" != "$POLICY_SHA" ]]; then
  echo "refusing: ${COMMIT} install_policy does not select ${POLICY} bound to ${POLICY_SHA}." >&2
  echo "The only oracle identity these commits accept is ${ORACLE_X86_CONTRACT_OFF_ID} bound to ${ORACLE_X86_CONTRACT_OFF_SHA256}." >&2
  echo "Selecting that x86 id while the dumps were produced by libmujoco ${THOR2_LIB_MUJOCO_STAGE3} would relabel the oracle. This script does not do that." >&2
  echo "To run the unfused profile against those dumps, install_policy in the tested commit must gain a match arm for ${ORACLE_AARCH64_CONTRACT_OFF_ID} that requires artifact sha256 ${THOR2_LIB_MUJOCO_STAGE3} and sets mul_quat Unfused, rot_vec_quat Unfused, and mul_mat_vec3 Contracted. Unknown sites and mul_mat_vec3 must be refused, the same way the x86 arm refuses them. That arm is not in this commit. Cargo was not started." >&2
  write_profile_receipt policy_not_selected not_run "${binding:-not_selected}"
  keep_superseded
  exit 2
fi

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
export FERROMOTION_BUILD_POLICY="$POLICY"
export FERROMOTION_ORACLE_SHA256="$POLICY_SHA"
unset FERROMOTION_CONTRACT_SITES || true
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_HOME="$RUN/cargo-home"
unset CARGO_TARGET_DIR || true
# set +e does not disarm an ERR trap. Exit 1 here is the finished Darwin mismatch.
goldens_log="$RUN/provenance/stage5-rereun-goldens-${COMMIT}-${missing_stamp}.log"
rust_log="$RUN/provenance/stage5-rereun-rust-${COMMIT}-${missing_stamp}.log"
summary="$RUN/provenance/stage5-rereun-${COMMIT}-${missing_stamp}.txt"
if [[ -e "$goldens_log" || -e "$rust_log" || -e "$summary" ]]; then
  echo "refusing: a stage5 log for ${COMMIT} at ${missing_stamp} exists and is not overwritten" >&2
  exit 2
fi
trap - ERR
set +e
"$RUN/scripts/run-goldens.sh" | tee "$goldens_log"
goldens_rc="${PIPESTATUS[0]}"
set -e
trap refuse_unexpected ERR
if [[ "$goldens_rc" -ne 0 && "$goldens_rc" -ne 1 ]]; then
  keep_superseded
  echo "refusing: golden wrapper exit $goldens_rc" >&2
  exit 2
fi
trap - ERR
set +e
"$RUN/scripts/run-rust-212.sh" "$corpus_run" | tee "$rust_log"
rust_rc="${PIPESTATUS[0]}"
set -e
trap refuse_unexpected ERR
{
  echo "ferromotion_tested_commit $COMMIT"
  echo "ferromotion_tested_tree $TREE"
  echo "ferromotion_build_policy $POLICY"
  echo "ferromotion_oracle_sha256 $POLICY_SHA"
  echo "dump_libmujoco_sha256 $DUMP_LIB_SHA"
  echo "libmujoco_thor2_pin $THOR2_LIB_MUJOCO_STAGE3"
  echo "libmujoco_matches_thor2_pin $LIB_MATCH"
  echo "darwin_golden_wrapper_exit $goldens_rc"
  echo "rust_212_wrapper_exit $rust_rc"
  echo "supersedes $superseded"
} | tee "$summary"
keep_superseded
# 0 is parity. 1 is a finished mismatch. 3 is cargo that did not finish.
case "$rust_rc" in
  0|1|3) exit "$rust_rc" ;;
  *)
    echo "refusing: rust wrapper exit $rust_rc" >&2
    exit 2
    ;;
esac
