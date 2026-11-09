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

grep -q 'FERROMOTION_RERUN_COMMIT=515a7dec4b4ed2fa7257f328c7c3f9b169b8d7ba' "$pull/pins.sh"
grep -q 'FERROMOTION_RERUN_CONTAINS=7998624df8a4ac2591187540a59ddc9b13d51640' "$pull/pins.sh"
grep -q 'rust212-20261109T215708Z' "$pull/05-rust-rereun.sh"
grep -q 'codeload.github.com' "$pull/05-rust-rereun.sh"
grep -q 'CARGO_NET_OFFLINE=true' "$pull/05-rust-rereun.sh"
grep -q -- '--offline' "$root/run-rust-212.sh"
echo "stage5-rereun-fixture ok"
