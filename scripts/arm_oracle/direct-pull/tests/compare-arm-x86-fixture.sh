#!/usr/bin/env bash
# The pinned x86 schedules are the contract-off dumps. Comparing a file to
# itself is full parity. One changed qacc line is the first divergence.
set -euo pipefail
export LC_ALL=C
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cmp="$DIR/compare-arm-x86.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
xz -dc "$DIR/schedules/x86-native.txt.xz" >"$work/native.txt"
xz -dc "$DIR/schedules/x86-step.txt.xz" >"$work/step.txt"
cp -f "$work/native.txt" "$work/native-arm.txt"
cp -f "$work/step.txt" "$work/step-arm.txt"
echo "a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7  $work/native.txt" | sha256sum -c -
echo "a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be  $work/step.txt" | sha256sum -c -
bash "$cmp" "$work/native-arm.txt" "$work/native.txt" "$work/step-arm.txt" "$work/step.txt" "$work/same.txt"
grep -F -q "verdict full_parity" "$work/same.txt"
grep -F -q "native_field qpos models 212/212 states 636/636" "$work/same.txt"
grep -F -q "step_field qpos models 212/212 steps 4240/4240" "$work/same.txt"
grep -F -q "x86_libmujoco_sha256 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc" "$work/same.txt"

awk 'BEGIN{changed=0} { if (!changed && $0 ~ /^d\tqacc\t/) { sub(/0x/, "0y"); changed=1 } print }' \
  "$work/native.txt" >"$work/native-bad.txt"
set +e
bash "$cmp" "$work/native-bad.txt" "$work/native.txt" "$work/step.txt" "$work/step.txt" "$work/bad.txt"
bad_rc=$?
set -e
[[ "$bad_rc" -eq 1 ]]
grep -F -q "verdict first_divergence" "$work/bad.txt"
grep -F -q "field d qacc" "$work/bad.txt"
# The tally is still complete. The changed state is one of 636.
grep -E -q "native_field d qacc models 211/212 states 635/636" "$work/bad.txt"

rm -f "$work/step.txt"
set +e
bash "$cmp" "$work/native.txt" "$work/native.txt" "$work/missing-step.txt" "$work/native.txt" "$work/missing.txt"
miss_rc=$?
set -e
[[ "$miss_rc" -eq 2 ]]
grep -F -q "verdict not_run" "$work/missing.txt"
echo "arm versus x86 fixture passed"
