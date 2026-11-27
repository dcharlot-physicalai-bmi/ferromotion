#!/usr/bin/env bash
# The lib-test binder reads cargo's JSON. The path-form line is a recorded
# `cargo test --lib --no-run --message-format=json` artifact
# (package_id path+file://…/crates/ferromotion-core#0.73.0, target.name ferromotion_core).
# The registry-form line is that artifact with a crates.io package_id.
# Only the path form is the workspace lib test. Exactly one executable is bound.
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
oracle="$(cd "$here/../.." && pwd)"
# shellcheck source=../../toolchain.sh
source "$oracle/toolchain.sh"

fix="$here/fixtures"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

make_bin() {
  local dest="$1"
  mkdir -p "$(dirname "$dest")"
  printf '#!/bin/sh\nexit 0\n' >"$dest"
  chmod +x "$dest"
}

# Keep the recorded object. Only the executable path is local to this run.
rewrite_exe() {
  local src="$1" dest="$2" exe="$3"
  python3 - "$src" "$dest" "$exe" <<'PY'
import json
import sys

src, dest, exe = sys.argv[1:]
out = []
for raw in open(src, encoding="utf-8"):
    line = raw.strip()
    if not line.startswith("{"):
        out.append(line + "\n")
        continue
    obj = json.loads(line)
    if obj.get("reason") == "compiler-artifact" and isinstance(obj.get("executable"), str):
        obj["executable"] = exe
        if isinstance(obj.get("filenames"), list):
            obj["filenames"] = [exe]
    out.append(json.dumps(obj, separators=(",", ":")) + "\n")
open(dest, "w", encoding="utf-8").writelines(out)
PY
}

expect_bind() {
  local json="$1" want="$2" out="$tmp/bound"
  rm -rf "$out"
  mkdir -p "$out"
  arm_oracle_bind_lib_test_executable "$out" "$json" >/dev/null
  [[ "$(cat "$out/toolchain-bins.txt")" == "$want" ]]
}

expect_refuse() {
  local json="$1" needle="$2" out="$tmp/refused"
  rm -rf "$out"
  mkdir -p "$out"
  set +e
  arm_oracle_bind_lib_test_executable "$out" "$json" >/dev/null 2>"$tmp/stderr"
  local rc=$?
  set -e
  [[ "$rc" -eq 2 ]]
  grep -q "$needle" "$tmp/stderr"
  if [[ -s "$out/toolchain-bins.txt" ]]; then
    echo "a refused bind recorded a binary" >&2
    exit 1
  fi
}

path_bin="$tmp/cargo-target/debug/build/ferromotion-core/abchash/out/ferromotion_core-abchash"
make_bin "$path_bin"
rewrite_exe "$fix/cargo-lib-test-path-id.jsonl" "$tmp/path.jsonl" "$path_bin"
{
  printf '%s\n' '{"reason":"compiler-message","message":{"rendered":"ok"}}'
  cat "$tmp/path.jsonl"
} >"$tmp/path-with-noise.jsonl"
expect_bind "$tmp/path-with-noise.jsonl" "$path_bin"

reg_bin="$tmp/cargo-target/debug/build/ferromotion-core/reghash/out/ferromotion_core-reghash"
make_bin "$reg_bin"
rewrite_exe "$fix/cargo-lib-test-registry-id.jsonl" "$tmp/registry.jsonl" "$reg_bin"
expect_refuse "$tmp/registry.jsonl" "cargo JSON recorded no lib test executable"

python3 - "$tmp/path.jsonl" "$tmp/hyphen.jsonl" <<'PY'
import json
import sys

src, dest = sys.argv[1:]
obj = json.loads(open(src, encoding="utf-8").read().strip())
obj["target"]["name"] = "ferromotion-core"
open(dest, "w", encoding="utf-8").write(json.dumps(obj) + "\n")
PY
expect_refuse "$tmp/hyphen.jsonl" "cargo JSON recorded no lib test executable"

other="$tmp/cargo-target/debug/build/ferromotion-core/other/out/ferromotion_core-other"
make_bin "$other"
rewrite_exe "$fix/cargo-lib-test-path-id.jsonl" "$tmp/other.jsonl" "$other"
cat "$tmp/path.jsonl" "$tmp/other.jsonl" >"$tmp/two.jsonl"
expect_refuse "$tmp/two.jsonl" "cargo JSON recorded more than one lib test executable"

echo "goldens-bind-package-id ok"
