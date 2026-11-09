# Shared helpers. Source this file. Do not execute it.
# shellcheck shell=bash

DIRECT_PULL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pins.sh
source "$DIRECT_PULL_DIR/pins.sh"

export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export RUSTUP_AUTO_INSTALL=0
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN GH_HOST GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS || true

direct_pull_askpass() {
  local ask="$1/askpass-fail"
  printf '%s\n' '#!/bin/sh' 'echo direct-pull askpass refused >&2' 'exit 1' >"$ask"
  chmod +x "$ask"
  export GIT_ASKPASS="$ask"
  export SSH_ASKPASS="$ask"
  export ASK="$ask"
}

gitc() {
  git -c credential.helper= -c core.askPass="${ASK:?}" -c advice.detachedHead=false "$@"
}

# build-20261109T213225Z (libmujoco 9164828bf04729b20b640bd09105baea98137a8749e262fc06fddc44ca869eb9)
# already passed stage 3. Stage 4 accepts that directory and any later build-$STAMP.
# It refuses only build-20261107T131346Z, whose libmujoco imports __muloti4.
direct_pull_accept_stage3_build() {
  local build="$1"
  case "$build" in
    *build-20261107T131346Z*)
      echo "refusing: $build is the library with undefined __muloti4. Leave it untouched. Run 03-cbuild.sh for a new build directory." >&2
      return 2
      ;;
  esac
  return 0
}

direct_pull_need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "refusing: $1 is not on PATH. This script does not install it." >&2
    exit 2
  fi
}

direct_pull_http_200() {
  local url="$1" code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 -A 'ferromotion-direct-pull' "$url")"
  if [[ "$code" != "200" ]]; then
    echo "refusing: unauthenticated GET $url returned $code" >&2
    exit 2
  fi
  echo "public 200 $url"
}

direct_pull_fetch() {
  local dest="$1" url="$2" commit="$3" tree="$4"
  if [[ -e "$dest" ]]; then
    echo "refusing: $dest already exists" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$dest")"
  gitc init -q "$dest"
  gitc -C "$dest" remote add origin "$url"
  if ! command -v timeout >/dev/null 2>&1; then
    echo "refusing: timeout(1) is not on PATH" >&2
    exit 2
  fi
  # timeout(1) executes a binary. gitc is a shell function, so the flags are repeated here.
  timeout -k 30 14400 git -c credential.helper= -c core.askPass="${ASK:?}" -c advice.detachedHead=false \
    -C "$dest" fetch --depth 1 origin "$commit"
  gitc -C "$dest" checkout --detach FETCH_HEAD
  local got tree_got dirty
  got="$(gitc -C "$dest" rev-parse HEAD)"
  tree_got="$(gitc -C "$dest" rev-parse 'HEAD^{tree}')"
  if [[ "$got" != "$commit" ]]; then
    echo "refusing: $dest HEAD $got is not $commit" >&2
    exit 2
  fi
  if [[ "$tree_got" != "$tree" ]]; then
    echo "refusing: $dest tree $tree_got is not $tree" >&2
    exit 2
  fi
  dirty="$(gitc -C "$dest" status --porcelain)"
  if [[ -n "$dirty" ]]; then
    echo "refusing: $dest is dirty immediately after checkout" >&2
    printf '%s\n' "$dirty" >&2
    exit 2
  fi
  echo "fetched $commit tree $tree_got $dest"
}

direct_pull_sha() {
  sha256sum "$1" | awk '{print $1}'
}

direct_pull_require_sha() {
  local path="$1" want="$2" got
  got="$(direct_pull_sha "$path")"
  if [[ "$got" != "$want" ]]; then
    echo "refusing: $path sha256 $got is not $want" >&2
    exit 2
  fi
}

direct_pull_host_is_thor_local() {
  local short full name
  short="$(hostname -s 2>/dev/null || true)"
  full="$(hostname -f 2>/dev/null || true)"
  name="$(hostname 2>/dev/null || true)"
  if [[ "$short" == "thor.local" || "$full" == "thor.local" || "$name" == "thor.local" ]]; then
    return 0
  fi
  if [[ "$short" == "thor" && "$full" == "thor.local" ]]; then
    return 0
  fi
  return 1
}

# On thor.local, hold bmi-1's energy-measurement lock until this process exits.
# Other hosts, including thor2, do not take it. The wait is bounded.
direct_pull_hold_energy_lock() {
  local run="$1" stage="$2"
  local wait_s="${BMI_ENERGY_LOCK_WAIT:-120}"
  local requested acquired
  if ! direct_pull_host_is_thor_local; then
    return 0
  fi
  if ! command -v flock >/dev/null 2>&1; then
    echo "refusing: flock is not on PATH, so thor.local cannot hold /tmp/bmi-energy.lock" >&2
    exit 2
  fi
  mkdir -p "$run/provenance"
  exec 9>/tmp/bmi-energy.lock
  requested="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! flock -w "$wait_s" 9; then
    echo "refusing: /tmp/bmi-energy.lock not acquired within ${wait_s}s on thor.local" >&2
    exit 2
  fi
  acquired="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  export BMI_ENERGY_LOCK_HELD=1
  export ENERGY_LOCK_FILE="$run/provenance/energy-lock-${stage}.txt"
  {
    echo "energy_lock_path /tmp/bmi-energy.lock"
    echo "energy_lock_host thor.local"
    echo "energy_lock_stage $stage"
    echo "energy_lock_wait_bound_s $wait_s"
    echo "energy_lock_requested_utc $requested"
    echo "energy_lock_acquired_utc $acquired"
    echo "energy_lock_pid $$"
  } >"$ENERGY_LOCK_FILE"
  trap 'direct_pull_finish_energy_lock' EXIT
}

direct_pull_finish_energy_lock() {
  local rel file list receipt
  [[ "${BMI_ENERGY_LOCK_HELD:-}" == "1" ]] || return 0
  rel="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  file="${ENERGY_LOCK_FILE:-}"
  if [[ -n "$file" ]]; then
    echo "energy_lock_released_utc $rel" >>"$file"
    list="$(dirname "$file")/energy-lock-receipts.txt"
    if [[ -f "$list" ]]; then
      while IFS= read -r receipt; do
        [[ -f "$receipt" ]] || continue
        if ! grep -q '^energy_lock_released_utc ' "$receipt"; then
          echo "energy_lock_released_utc $rel" >>"$receipt"
          sha256sum "$receipt" >"$(dirname "$receipt")/RECEIPT.sha256"
        fi
      done <"$list"
    fi
  fi
}

# List Cargo.lock crates the vendored directory cannot satisfy offline.
# Prints one missing line per crate and returns 2 when any are missing.
direct_pull_list_missing_vendor() {
  local lockfile="$1" vendor="$2"
  local missing=0 name ver src dir found cargo_ver
  [[ -f "$lockfile" ]] || {
    echo "missing lock $lockfile" >&2
    return 2
  }
  [[ -d "$vendor" ]] || {
    echo "missing vendor $vendor" >&2
    return 2
  }
  while IFS=$'\t' read -r name ver src; do
    [[ -n "$name" ]] || continue
    if [[ "$src" == git+* ]]; then
      echo "missing git ${name} ${ver} ${src}"
      missing=1
      continue
    fi
    if [[ "$src" != registry+* ]]; then
      continue
    fi
    found=0
    for dir in "$vendor/$name" "$vendor/${name}-${ver}"; do
      [[ -f "$dir/Cargo.toml" && -f "$dir/.cargo-checksum.json" ]] || continue
      cargo_ver="$(awk -F'"' '/^version = /{print $2; exit}' "$dir/Cargo.toml")"
      if [[ "$cargo_ver" == "$ver" ]]; then
        found=1
        break
      fi
    done
    if [[ "$found" -ne 1 ]]; then
      echo "missing registry ${name} ${ver}"
      missing=1
    fi
  done < <(awk '
    function emit() {
      if (name != "" && src != "") printf "%s\t%s\t%s\n", name, ver, src
    }
    /^\[\[package\]\]/ { emit(); name=""; ver=""; src=""; next }
    /^name = / { name=$0; sub(/^name = "/, "", name); sub(/"$/, "", name); next }
    /^version = / { ver=$0; sub(/^version = "/, "", ver); sub(/"$/, "", ver); next }
    /^source = / { src=$0; sub(/^source = "/, "", src); sub(/"$/, "", src); next }
    END { emit() }
  ' "$lockfile")
  [[ "$missing" -eq 0 ]]
}

direct_pull_run_root() {
  local base="${FERROMOTION_ORACLE_HOME:-$HOME/ferromotion-oracle}"
  local stamp run
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  run="$base/${FERROMOTION_COMMIT}-${stamp}"
  if [[ -e "$run" ]]; then
    echo "refusing: $run already exists" >&2
    exit 2
  fi
  printf '%s\n' "$run"
}
