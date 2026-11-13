#!/usr/bin/env bash
# Build the pinned contract-off library, point Python at it, and write one dump.
# Usage: oracle_profile_prepare.sh native|step
set -euo pipefail
phase="${1:?native or step}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
work="${RUNNER_TEMP:-/tmp}"
# `__FILE__` strings in the pinned library are `/tmp/mujoco-3.13.0-src/...`.
# A checkout anywhere else changes the sha256 even with the same clang.
src=/tmp/mujoco-3.13.0-src
build=/tmp/oracle-x86-contract-off
# The pin is Ubuntu clang 18.1.3. A newer clang, including the Ubuntu 26.04
# clang-18 package that is a wrapper onto clang 21, does not reproduce it.
if [[ -x /usr/bin/clang-18 && -x /usr/bin/clang++-18 ]]; then
  mkdir -p "$work/clang-bin"
  ln -sfn /usr/bin/clang-18 "$work/clang-bin/clang"
  ln -sfn /usr/bin/clang++-18 "$work/clang-bin/clang++"
  export PATH="$work/clang-bin:$PATH"
fi
clang --version
clang++ --version
out="$work/oracle"
menagerie="$work/menagerie"

if [[ ! -d "$src/src" ]]; then
  curl -fsSL -o "$work/mujoco-3.13.0.tar.gz" https://github.com/google-deepmind/mujoco/archive/refs/tags/3.13.0.tar.gz
  mkdir -p "$src"
  tar -xzf "$work/mujoco-3.13.0.tar.gz" -C "$src" --strip-components=1
fi
bash "$ROOT/scripts/build_mujoco_3130_x86_contract_off.sh" "$src" "$build"
mkdir -p "$out/lib"
cp -a "$build/build/lib/libmujoco.so.3.13.0" "$out/lib/libmujoco.so.3.13.0"
sha256sum "$out/lib/libmujoco.so.3.13.0"

if [[ ! -d "$menagerie/.git" ]]; then
  git init "$menagerie"
  git -C "$menagerie" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
fi
sha="$(python3 -c 'import json; print(json.load(open("'"$ROOT"'/conformance/menagerie_pin.json"))["menagerie_sha"])')"
git -C "$menagerie" fetch --depth 1 origin "$sha"
git -C "$menagerie" checkout --detach FETCH_HEAD
test "$(git -C "$menagerie" rev-parse HEAD)" = "$sha"

venv="$work/oracle-venv"
if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install -q "mujoco==3.13.0" numpy
fi
pkg="$venv/lib/python3.12/site-packages/mujoco"
rm -f "$pkg/libmujoco.so.3.13.0"
ln -s "$out/lib/libmujoco.so.3.13.0" "$pkg/libmujoco.so.3.13.0"

if [[ "$phase" == "native" ]]; then
  "$venv/bin/python" "$ROOT/scripts/mujoco_native_oracle.py" "$menagerie" "$out/x86-native.txt" 3
else
  "$venv/bin/python" "$ROOT/scripts/mujoco_native_step_oracle.py" "$menagerie" "$out/x86-step.txt" 20
fi
sha256sum "$out"/x86-*.txt
