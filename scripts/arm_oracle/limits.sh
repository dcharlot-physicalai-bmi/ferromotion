# Shared bounds for the thor2 oracle scripts. Source this file. Do not execute it.
# shellcheck shell=bash
# shellcheck source=compiler-rt.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compiler-rt.sh"

# ulimit -t sets RLIMIT_CPU. That is CPU seconds for one process.
# It is not added up across cmake, ninja, cargo, or rustc.
# It is not wall-clock time and it is not a memory cap.
# ulimit -v sets RLIMIT_AS, in KiB, for one process. That is virtual
# address space, not resident set size, and not a cgroup memory.max.
# Two jobs can each map up to that cap. This host is not placed in a cgroup.
# timeout(1) is the wall-clock cap. It is applied to the heavy command
# and, without --foreground, kills that command's process group.
# --foreground is not used: in that mode, children are not timed out.
# CARGO_BUILD_JOBS and cmake -j limit compilation.
# --test-threads limits how many tests run at once. It does not limit rustc.

arm_oracle_apply_limits() {
  ulimit -t 14400 || {
    echo "refusing: cannot set the per-process CPU limit (ulimit -t)" >&2
    exit 2
  }
  ulimit -v 50331648 || {
    echo "refusing: cannot set the per-process virtual address space (ulimit -v)" >&2
    exit 2
  }
  export OMP_NUM_THREADS=1
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  export CARGO_BUILD_JOBS=2
  export CARGO_NET_OFFLINE=true
  export ARM_ORACLE_LIMITS=applied
  export ARM_ORACLE_WALL_TIMEOUT="${ARM_ORACLE_WALL_TIMEOUT:-14400}"
  export ARM_ORACLE_CPU_SECONDS_PER_PROCESS=14400
  export ARM_ORACLE_AS_KIB_PER_PROCESS=50331648
}

arm_oracle_timeout() {
  local seconds="$1"
  shift
  if ! command -v timeout >/dev/null 2>&1; then
    echo "refusing: timeout(1) is not on PATH. Not starting an unbounded command." >&2
    exit 2
  fi
  timeout -k 30 "$seconds" "$@"
}

arm_oracle_root() {
  local script_dir parent
  script_dir="$(cd "$(dirname "$1")" && pwd)"
  parent="$(cd "$script_dir/.." && pwd)"
  if [[ -d "$parent/ferromotion" && -d "$parent/probes" && -d "$parent/schedules" ]]; then
    printf '%s\n' "$parent"
    return 0
  fi
  echo "refusing: $script_dir is not scripts/ inside the unpacked oracle bundle" >&2
  return 2
}

arm_oracle_require_thor2() {
  local why="$1"
  if [[ "${FERROMOTION_ORACLE_SMOKE:-}" == "x86" ]]; then
    if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
      echo "refusing: FERROMOTION_ORACLE_SMOKE=x86 is Linux x86_64 only (got $(uname -s) $(uname -m))" >&2
      exit 2
    fi
    echo "smoke: Linux x86_64. This run is not a thor2 receipt and is not an aarch64 oracle." >&2
    return 0
  fi
  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
    echo "refusing: Linux aarch64 only (got $(uname -s) $(uname -m)). $why" >&2
    exit 2
  fi
  if [[ "$(hostname -s)" != "thor2" ]]; then
    echo "refusing: $why (hostname -s is '$(hostname -s)'). Do not run this on thor." >&2
    exit 2
  fi
}

arm_oracle_version_at_least() {
  local IFS=.
  local -a got need
  read -r -a got <<<"${1:-0.0.0}"
  read -r -a need <<<"$2"
  local i av bv
  for i in 0 1 2; do
    av="${got[$i]:-0}"
    bv="${need[$i]:-0}"
    av="${av%%[^0-9]*}"
    bv="${bv%%[^0-9]*}"
    av="${av:-0}"
    bv="${bv:-0}"
    if ((10#$av > 10#$bv)); then
      return 0
    fi
    if ((10#$av < 10#$bv)); then
      return 1
    fi
  done
  return 0
}

arm_oracle_select_cargo() {
  local rel
  rel="$(cargo -vV 2>/dev/null | awk '/^release:/{print $2; exit}' || true)"
  if arm_oracle_version_at_least "$rel" 1.89.0; then
    ARM_ORACLE_CARGO=(cargo)
    return 0
  fi
  if cargo +stable -vV >/dev/null 2>&1; then
    rel="$(cargo +stable -vV | awk '/^release:/{print $2; exit}')"
    if arm_oracle_version_at_least "$rel" 1.89.0; then
      ARM_ORACLE_CARGO=(cargo +stable)
      echo "cargo: default rustc is older than 1.89; using cargo +stable ($rel). This script did not install it." >&2
      return 0
    fi
  fi
  echo "refusing: rustc >= 1.89 is required (project rust-version). This script does not install a toolchain. See rust/OPTIONAL-RUSTUP.md." >&2
  exit 2
}

arm_oracle_cargo() {
  "${ARM_ORACLE_CARGO[@]}" "$@"
}

arm_oracle_require_vendor() {
  local fm="$1"
  local cfg="$fm/.cargo/config.toml"
  if [[ ! -d "$fm/vendor" ]]; then
    echo "refusing: $fm/vendor is missing. This script does not fetch crates." >&2
    exit 2
  fi
  if [[ ! -f "$cfg" ]]; then
    echo "refusing: $cfg is missing" >&2
    exit 2
  fi
  if ! grep -q 'replace-with = "vendored-sources"' "$cfg"; then
    echo "refusing: $cfg does not replace crates.io with vendored-sources" >&2
    exit 2
  fi
  if ! grep -q 'directory = "vendor"' "$cfg"; then
    echo "refusing: $cfg does not point vendored-sources at vendor/" >&2
    exit 2
  fi
  if ! grep -Eq '^[[:space:]]*offline[[:space:]]*=[[:space:]]*true[[:space:]]*$' "$cfg"; then
    echo "refusing: $cfg does not set net.offline = true. Pointing crates.io at vendor/ does not by itself stop network access." >&2
    exit 2
  fi
}

arm_oracle_cargo_target() {
  local out="$1"
  local fm out_abs parent
  fm="$(cd "$ROOT/ferromotion" && pwd)"
  parent="$(cd "$(dirname "$out")" && pwd)"
  out_abs="$parent/$(basename "$out")"
  case "$out_abs" in
    "$fm"|"$fm"/*|"$ROOT"|"$ROOT"/*)
      echo "refusing: cargo target $out_abs is inside the bundle or the ferromotion source tree" >&2
      exit 2
      ;;
  esac
  export CARGO_TARGET_DIR="$out_abs/cargo-target"
  mkdir -p "$CARGO_TARGET_DIR"
  if [[ -e "$fm/target" ]]; then
    echo "refusing: $fm/target already exists. Cargo must not write its target directory into the source tree." >&2
    exit 2
  fi
}
