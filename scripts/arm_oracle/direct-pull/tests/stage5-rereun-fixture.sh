#!/usr/bin/env bash
# Vendor gaps are listed and cargo is not started. thor.local holds the energy lock.
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
pull="$(cd "$here/.." && pwd)"
root="$(cd "$here/../.." && pwd)"
# shellcheck source=../lib.sh
source "$pull/lib.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/vendor/present"
printf 'version = "1.0.0"\n' >"$tmp/vendor/present/Cargo.toml"
printf '{}\n' >"$tmp/vendor/present/.cargo-checksum.json"
cat >"$tmp/Cargo.lock" <<'EOF'
[[package]]
name = "present"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "absent"
version = "2.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "fromgit"
version = "0.1.0"
source = "git+https://github.com/example/fromgit"
EOF
direct_pull_list_missing_vendor "$tmp/Cargo.lock" "$tmp/vendor" >"$tmp/missing.txt" && {
  echo "missing crates were accepted" >&2
  exit 1
}
grep -F -x -q 'missing registry absent 2.0.0' "$tmp/missing.txt"
grep -F -q 'missing git fromgit 0.1.0 git+https://github.com/example/fromgit' "$tmp/missing.txt"
grep -q 'present' "$tmp/missing.txt" && {
  echo "vendored crate was listed missing" >&2
  exit 1
}
cat >"$tmp/Cargo.lock" <<'EOF'
[[package]]
name = "present"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
EOF
direct_pull_list_missing_vendor "$tmp/Cargo.lock" "$tmp/vendor" >"$tmp/ok.txt"

# thor2 does not take the energy lock.
mkdir -p "$tmp/bin" "$tmp/run/provenance"
cat >"$tmp/bin/hostname" <<'EOF'
#!/bin/sh
echo thor2
EOF
chmod +x "$tmp/bin/hostname"
PATH="$tmp/bin:$PATH" direct_pull_hold_energy_lock "$tmp/run" stage-test
[[ ! -f "$tmp/run/provenance/energy-lock-stage-test.txt" ]]

# thor.local takes the lock and records the window when the process exits.
cat >"$tmp/bin/hostname" <<'EOF'
#!/bin/sh
case "$1" in
  -s) echo thor ;;
  -f) echo thor.local ;;
  *) echo thor.local ;;
esac
EOF
PATH="$tmp/bin:$PATH" BMI_ENERGY_LOCK_WAIT=2 bash -c '
  source "$1"
  direct_pull_hold_energy_lock "$2" stage-held
  grep -q "^energy_lock_acquired_utc " "$2/provenance/energy-lock-stage-held.txt"
  grep -q "^energy_lock_wait_bound_s 2$" "$2/provenance/energy-lock-stage-held.txt"
' bash "$pull/lib.sh" "$tmp/run"
grep -q '^energy_lock_released_utc ' "$tmp/run/provenance/energy-lock-stage-held.txt"

flock /tmp/bmi-energy.lock sleep 30 &
holder=$!
sleep 0.2
set +e
PATH="$tmp/bin:$PATH" BMI_ENERGY_LOCK_WAIT=1 bash -c '
  source "$1"
  direct_pull_hold_energy_lock "$2" stage-wait
' bash "$pull/lib.sh" "$tmp/run" >"$tmp/wait.out" 2>"$tmp/wait.err"
wait_rc=$?
set -e
kill "$holder" 2>/dev/null || true
wait "$holder" 2>/dev/null || true
[[ "$wait_rc" -eq 2 ]]
grep -q 'not acquired within 1s' "$tmp/wait.err"

cat >"$tmp/bin/uname" <<'EOF'
#!/bin/sh
if [ "$1" = "-m" ]; then echo aarch64; exit 0; fi
echo Linux
EOF
chmod +x "$tmp/bin/uname"
set +e
PATH="$tmp/bin:$PATH" bash -c '
  source "$1"
  arm_oracle_require_thor2 "fixture"
' bash "$root/limits.sh" >"$tmp/host.out" 2>"$tmp/host.err"
host_rc=$?
set -e
[[ "$host_rc" -eq 2 ]]
grep -q 'without /tmp/bmi-energy.lock' "$tmp/host.err"
set +e
BMI_ENERGY_LOCK_HELD=1 PATH="$tmp/bin:$PATH" bash -c '
  source "$1"
  arm_oracle_require_thor2 "fixture"
' bash "$root/limits.sh" >"$tmp/host-ok.out" 2>"$tmp/host-ok.err"
host_ok=$?
set -e
[[ "$host_ok" -eq 0 ]]

# The fetched tree has no .cargo. The reuse path still writes the offline config.
reuse="$tmp/ferromotion-${FERROMOTION_RERUN_COMMIT}"
mkdir -p "$reuse"
cat >"$reuse/PROVENANCE.txt" <<EOF
ferromotion_tested_commit $FERROMOTION_RERUN_COMMIT
ferromotion_tested_tree $FERROMOTION_RERUN_TREE
EOF
[[ ! -d "$reuse/.cargo" ]]
direct_pull_install_offline_cargo_config "$reuse"
[[ -f "$reuse/.cargo/config.toml" ]]
grep -F -x -q 'offline = true' "$reuse/.cargo/config.toml"
grep -F -x -q "ferromotion_tested_commit $FERROMOTION_RERUN_COMMIT" "$reuse/PROVENANCE.txt"
awk '
  /exists and is not overwritten/ && !refuse { refuse = NR }
  /direct_pull_install_offline_cargo_config/ { install = NR }
  END { exit !(refuse && install && refuse < install) }
' "$pull/05-rust-rereun.sh"

set +e
bash "$pull/05-rust-rereun.sh" >"$tmp/noarg.out" 2>"$tmp/noarg.err"
noarg_rc=$?
bash "$pull/05-rust-rereun.sh" "$tmp/run" >"$tmp/nostage.out" 2>"$tmp/nostage.err"
nostage_rc=$?
set -e
[[ "$noarg_rc" -eq 2 ]]
[[ "$nostage_rc" -eq 2 ]]
grep -q 'refusing:' "$tmp/noarg.err"
grep -q 'stage4-ok' "$tmp/nostage.err"

# A missing directory on a redirect is status 1. The script trap turns that into 2.
set +e
bash -c '
  set -Eeuo pipefail
  refuse_unexpected() { exit 2; }
  trap refuse_unexpected ERR
  cat >"$1/missing-dir/config.toml" <<EOF
x
EOF
' bash "$tmp"
redir_rc=$?
set -e
[[ "$redir_rc" -eq 2 ]]

grep -q 'FERROMOTION_RERUN_COMMIT=515a7dec4b4ed2fa7257f328c7c3f9b169b8d7ba' "$pull/pins.sh"
grep -q 'FERROMOTION_RERUN_CONTAINS=7998624df8a4ac2591187540a59ddc9b13d51640' "$pull/pins.sh"
grep -q 'FERROMOTION_PROFILE_6092_COMMIT=6092a8e537d9eb1264fce1775daf809e15616b03' "$pull/pins.sh"
grep -q 'FERROMOTION_PROFILE_D14_COMMIT=d14e65fac3bcc44fa603dc9c8d87e4ea149a4200' "$pull/pins.sh"
grep -q 'rust212-20261109T215708Z' "$pull/05-rust-rereun.sh"
grep -q 'codeload.github.com' "$pull/05-rust-rereun.sh"
grep -q 'CARGO_NET_OFFLINE=true' "$pull/05-rust-rereun.sh"
grep -q 'Cargo was not started' "$pull/05-rust-rereun.sh"
grep -q 'api.github.com' "$pull/lib.sh"
grep -q 'codeload.github.com' "$pull/05-rust-rereun.sh"
if grep -q 'Authorization:' "$pull/05-rust-rereun.sh" "$pull/lib.sh"; then
  echo "commit fetch sends an Authorization header" >&2
  exit 1
fi
direct_pull_require_full_sha b6a3a885 && {
  echo "a short sha was accepted" >&2
  exit 1
}
direct_pull_require_full_sha b6a3a885f8c405b3f2c098ece0cc1bdc8ad5ce03
direct_pull_fidelity_status_ok identical
direct_pull_fidelity_status_ok behind
direct_pull_fidelity_status_ok diverged && {
  echo "a diverged commit was treated as reachable" >&2
  exit 1
}
if grep -q 'not a pinned oracle-profile commit' "$pull/lib.sh" "$pull/05-rust-rereun.sh"; then
  echo "the commit allowlist is still in place" >&2
  exit 1
fi
grep -q -- '--offline' "$root/run-rust-212.sh"
repo="$(git -C "$pull" rev-parse --show-toplevel)"
git -C "$repo" show "${FERROMOTION_PROFILE_6092_COMMIT}:crates/ferromotion-core/src/fp_contract.rs" >"$tmp/fp6092.rs"
git -C "$repo" show "${FERROMOTION_PROFILE_D14_COMMIT}:crates/ferromotion-core/src/fp_contract.rs" >"$tmp/fpd14.rs"
set +e
direct_pull_policy_binding "$tmp/fp6092.rs" "$ORACLE_AARCH64_CONTRACT_OFF_ID" >"$tmp/arm6092.out" 2>"$tmp/arm6092.err"
arm6092_rc=$?
direct_pull_policy_binding "$tmp/fpd14.rs" "$ORACLE_AARCH64_CONTRACT_OFF_ID" >"$tmp/armd14.out" 2>"$tmp/armd14.err"
armd14_rc=$?
direct_pull_policy_binding "$repo/crates/ferromotion-core/src/fp_contract.rs" "$ORACLE_AARCH64_CONTRACT_OFF_ID" >"$tmp/armhead.out" 2>"$tmp/armhead.err"
armhead_rc=$?
set -e
[[ "$arm6092_rc" -eq 3 && "$armd14_rc" -eq 3 && "$armhead_rc" -eq 3 ]]
[[ "$(direct_pull_policy_binding "$tmp/fp6092.rs" "$ORACLE_X86_CONTRACT_OFF_ID")" == "$ORACLE_X86_CONTRACT_OFF_SHA256" ]]
[[ "$(direct_pull_policy_binding "$tmp/fpd14.rs" "$ORACLE_X86_CONTRACT_OFF_ID")" == "$ORACLE_X86_CONTRACT_OFF_SHA256" ]]
[[ "$(direct_pull_policy_binding "$repo/crates/ferromotion-core/src/fp_contract.rs" published-default)" == "published-default" ]]
echo "stage5-rereun-fixture ok"
