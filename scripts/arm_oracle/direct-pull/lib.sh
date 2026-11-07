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
