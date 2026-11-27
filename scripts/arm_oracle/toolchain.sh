#!/usr/bin/env bash
# Toolchain of the binary this build just produced.
# rustc -vV is executed here. A cargo_vv line from an older provenance file is not read.
# Sourced. It does not change the caller's shell options.

arm_oracle_toolchain_fail() {
  echo "refusing: $*" >&2
  return 2
}

# The rustc cargo will invoke for ARM_ORACLE_CARGO, executed now.
arm_oracle_rustc_vv() {
  if [[ -n "${RUSTC:-}" ]]; then
    "$RUSTC" -vV
    return
  fi
  if [[ "${ARM_ORACLE_CARGO[1]:-}" == "+stable" ]]; then
    rustup run stable rustc -vV
    return
  fi
  rustc -vV
}

# Write a fresh rustc -vV. dest's directory must be this run's output, not an older provenance dir.
arm_oracle_capture_rustc_vv() {
  local dest="$1"
  local tmp
  mkdir -p "$(dirname "$dest")"
  tmp="$(mktemp)"
  if ! arm_oracle_rustc_vv >"$tmp"; then
    rm -f "$tmp"
    arm_oracle_toolchain_fail "rustc -vV failed"
    return 2
  fi
  if ! head -n 1 "$tmp" | grep -q '^rustc '; then
    rm -f "$tmp"
    arm_oracle_toolchain_fail "rustc -vV did not start with a rustc line"
    return 2
  fi
  if ! grep -q '^release: ' "$tmp" || ! grep -q '^commit-hash: ' "$tmp"; then
    rm -f "$tmp"
    arm_oracle_toolchain_fail "rustc -vV is missing release or commit-hash"
    return 2
  fi
  mv "$tmp" "$dest"
  : >"$(dirname "$dest")/toolchain-bins.txt"
}

arm_oracle_bind_built_binary() {
  local out="$1" bin="$2"
  if [[ ! -f "$bin" ]]; then
    arm_oracle_toolchain_fail "built binary is missing: $bin"
    return 2
  fi
  printf '%s\n' "$bin" >>"$out/toolchain-bins.txt"
}

# The lib test executable is the JSON `executable` of the ferromotion-core lib
# artifact from `cargo test --lib --no-run --message-format=json`.
# Thor2's cargo writes that binary under debug/build/ferromotion-core/<hash>/out/,
# not debug/deps/. A deps/ glob is not a second guess.
arm_oracle_bind_lib_test_executable() {
  local out="$1" json="$2"
  local line exe n=0
  if [[ ! -s "$json" ]]; then
    arm_oracle_toolchain_fail "cargo JSON for the lib test executable is missing"
    return 2
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *'"reason":"compiler-artifact"'* ]] || continue
    [[ "$line" == *'"kind":["lib"]'* && "$line" == *'"name":"ferromotion-core"'* ]] || continue
    [[ "$line" == *'"executable":"'* ]] || continue
    [[ "$line" == *'"executable":null'* ]] && continue
    exe="${line#*\"executable\":\"}"
    exe="${exe%%\"*}"
    if [[ "$exe" == *\\* || -z "$exe" ]]; then
      arm_oracle_toolchain_fail "cargo JSON executable is not a plain path"
      return 2
    fi
    if [[ ! -x "$exe" ]]; then
      arm_oracle_toolchain_fail "cargo JSON executable is missing: $exe"
      return 2
    fi
    if [[ -s "$out/toolchain-bins.txt" ]] && grep -F -x -q "$exe" "$out/toolchain-bins.txt"; then
      n=$((n + 1))
      continue
    fi
    arm_oracle_bind_built_binary "$out" "$exe"
    n=$((n + 1))
  done <"$json"
  if [[ "$n" -eq 0 ]]; then
    arm_oracle_toolchain_fail "cargo JSON recorded no lib test executable"
    return 2
  fi
}

# Print the rustc version string from .comment. Empty if the section or the string is absent.
arm_oracle_elf_comment_rustc() {
  local bin="$1"
  local text="" decoded=""
  if command -v readelf >/dev/null 2>&1; then
    text="$(readelf -p .comment "$bin" 2>/dev/null || true)"
  fi
  if [[ "$text" == *"rustc version "* ]]; then
    printf '%s\n' "$text" | grep -o 'rustc version [^[:cntrl:]]*' | sed 's/[[:space:]]*$//'
    return 0
  fi
  if ! command -v objdump >/dev/null 2>&1; then
    return 0
  fi
  local hex
  hex="$(objdump -s -j .comment "$bin" 2>/dev/null | awk '
    /^[[:space:]]*[0-9a-fA-F]+[[:space:]]/ {
      for (i = 2; i <= NF; i++) if ($i ~ /^[0-9a-fA-F]+$/ && length($i) % 2 == 0) printf "%s", $i
    }
  ' || true)"
  if [[ -n "$hex" ]]; then
    # NUL bytes separate .comment strings. Translate them before the capture.
    decoded="$(printf '%b' "$(printf '%s\n' "$hex" | sed 's/../\\x&/g')" | tr '\0' '\n')"
  fi
  printf '%s\n' "$decoded" | grep -o 'rustc version [^[:cntrl:]]*' | sed 's/[[:space:]]*$//' || true
}

# Receipt lines from this run's rustc -vV and the built binary. Missing or disagreeing evidence refuses.
arm_oracle_toolchain_receipt_lines() {
  local vv="$1"
  shift
  local bin line first expect comment comments="" distinct
  if [[ ! -s "$vv" ]]; then
    arm_oracle_toolchain_fail "rustc -vV from this build is missing"
    return 2
  fi
  if [[ "$#" -lt 1 ]]; then
    arm_oracle_toolchain_fail "no built binary was recorded for this receipt"
    return 2
  fi
  first="$(head -n 1 "$vv")"
  if [[ "$first" != rustc\ * ]]; then
    arm_oracle_toolchain_fail "rustc -vV from this build does not start with rustc"
    return 2
  fi
  expect="rustc version ${first#rustc }"
  for bin in "$@"; do
    if [[ ! -f "$bin" ]]; then
      arm_oracle_toolchain_fail "built binary is missing: $bin"
      return 2
    fi
    comments="$(arm_oracle_elf_comment_rustc "$bin")"
    if [[ -z "$comments" ]]; then
      arm_oracle_toolchain_fail "ELF .comment has no rustc version in $bin"
      return 2
    fi
    distinct="$(printf '%s\n' "$comments" | sort -u)"
    if [[ "$(printf '%s\n' "$distinct" | wc -l | tr -d ' ')" -ne 1 || "$distinct" != "$expect" ]]; then
      arm_oracle_toolchain_fail "rustc -vV and ELF .comment disagree for $bin"
      printf 'rustc -vV: %s\n.comment: %s\n' "$first" "$distinct" >&2
      return 2
    fi
  done
  echo "toolchain_rustc $first"
  echo "toolchain_comment $expect"
  echo "toolchain_release $(awk '/^release:/{print $2; exit}' "$vv")"
  echo "toolchain_commit $(awk '/^commit-hash:/{print $2; exit}' "$vv")"
  echo "toolchain_host $(awk '/^host:/{print $2; exit}' "$vv")"
  echo "toolchain_source rustc-vv-at-build-and-elf-comment"
}

# Used by receipt writers. Reads only this output directory.
arm_oracle_emit_toolchain() {
  local out="$1"
  local bins=()
  if [[ ! -s "$out/toolchain-bins.txt" ]]; then
    arm_oracle_toolchain_fail "this build recorded no binary"
    return 2
  fi
  mapfile -t bins <"$out/toolchain-bins.txt"
  arm_oracle_toolchain_receipt_lines "$out/rustc-vv.txt" "${bins[@]}"
}
