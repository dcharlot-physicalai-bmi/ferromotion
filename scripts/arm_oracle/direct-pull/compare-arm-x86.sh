#!/usr/bin/env bash
# Bitwise ARM dump versus the pinned x86 contract-off dumps.
# The x86 reference is schedules/x86-native.txt and schedules/x86-step.txt,
# the recorded output of mujoco-3.13.0-x86_64-contract-off
# (libmujoco 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc).
# Exit 0: full parity. Exit 1: first divergence, after the full tally.
# Exit 2: verdict not_run (a dump is missing or is not the pinned x86 bytes).
# Dump lines are records. A line that begins with '#' is still a record.
# This script does not read the mulquat fixture, whose '#' lines are comments.
set -euo pipefail
export LC_ALL=C
ARM_NATIVE="${1:?arm native.txt}"
X86_NATIVE="${2:?x86 native.txt}"
ARM_STEP="${3:?arm step.txt}"
X86_STEP="${4:?x86 step.txt}"
OUT="${5:?verdict file}"

X86_LIB=64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc
X86_NATIVE_SHA=a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7
X86_STEP_SHA=a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be

write_not_run() {
  local why="$1"
  {
    echo "verdict not_run"
    echo "reason $why"
    echo "x86_oracle mujoco-3.13.0-x86_64-contract-off"
    echo "x86_libmujoco_sha256 $X86_LIB"
  } | tee "$OUT"
  exit 2
}

[[ -f "$ARM_NATIVE" && -f "$ARM_STEP" && -f "$X86_NATIVE" && -f "$X86_STEP" ]] || write_not_run "a dump is missing"
if [[ "${FERROMOTION_X86_DUMP_SHA_CHECK:-1}" != "0" ]]; then
  got_n="$(sha256sum "$X86_NATIVE" | awk '{print $1}')"
  got_s="$(sha256sum "$X86_STEP" | awk '{print $1}')"
  [[ "$got_n" == "$X86_NATIVE_SHA" ]] || write_not_run "x86-native sha256 is not the pinned oracle dump"
  [[ "$got_s" == "$X86_STEP_SHA" ]] || write_not_run "x86-step sha256 is not the pinned oracle dump"
fi

tmp="$(mktemp)"
awk -F '\t' -v native_arm="$ARM_NATIVE" -v native_x86="$X86_NATIVE" -v step_arm="$ARM_STEP" -v step_x86="$X86_STEP" '
function field_of(k1, k2) {
  if (k1 == "model" || k1 == "s" || k1 == "") return ""
  if (k1 == "m" || k1 == "d") return k1 " " k2
  return k1
}
function flush_model(prefix,    f) {
  if (model == "") return
  for (f in seen) {
    model_total[prefix SUBSEP f]++
    if (!miss[f]) model_exact[prefix SUBSEP f]++
  }
  delete seen
  delete miss
}
function note(prefix, f, state, equal, arm_line) {
  if (f == "") return
  seen[f] = 1
  if (prefix == "native" && substr(f, 1, 2) == "m ") {
    # Model fields are once per model. The 636 count is the state fields.
  } else if (prefix == "step" && state < 1) {
    # The block before s 1 is the initial state, not one of the 4240 steps.
  } else {
    state_total[prefix SUBSEP f]++
    if (equal) state_exact[prefix SUBSEP f]++
  }
  if (!equal) {
    miss[f] = 1
    if (first == "") {
      first = prefix " model " model " state " state " field " f
      first_arm = arm_line
    }
  }
}
function compare_pair(prefix, arm_path, x86_path,    arm, x86, na, nx, k1, k2, f, equal) {
  while (1) {
    na = (getline arm < arm_path)
    nx = (getline x86 < x86_path)
    if (na <= 0 && nx <= 0) break
    if (na <= 0 || nx <= 0) {
      if (first == "") first = prefix " model " model " state " state " field truncated"
      break
    }
    if (arm == "" && x86 == "") continue
    split(arm, aa, "\t")
    k1 = aa[1]
    k2 = aa[2]
    if (k1 == "model") {
      flush_model(prefix)
      model = k2
      state = 0
      if (arm != x86 && first == "") first = prefix " model " model " state 0 field model"
      continue
    }
    if (prefix == "native" && k1 == "qpos") state++
    if (prefix == "step" && k1 == "s") state = k2 + 0
    f = field_of(k1, k2)
    equal = (arm == x86)
    note(prefix, f, state, equal, arm)
  }
  flush_model(prefix)
  close(arm_path)
  close(x86_path)
}
BEGIN {
  PROCINFO["sorted_in"] = "@ind_str_asc"
  model = ""
  state = 0
  first = ""
  compare_pair("native", native_arm, native_x86)
  model = ""
  state = 0
  delete seen
  delete miss
  compare_pair("step", step_arm, step_x86)
  for (key in model_total) {
    split(key, parts, SUBSEP)
    prefix = parts[1]
    f = parts[2]
    mt = model_total[key] + 0
    me = model_exact[key] + 0
    st = state_total[key] + 0
    se = state_exact[key] + 0
    if (prefix == "native" && st == 0) {
      printf "native_field %s models %d/%d states -\n", f, me, mt
    } else if (prefix == "native") {
      printf "native_field %s models %d/%d states %d/%d\n", f, me, mt, se, st
      if (f == "qpos" && (mt != 212 || st != 636 || me != mt || se != st)) bad_count = 1
    } else if (st == 0) {
      printf "step_field %s models %d/%d steps -\n", f, me, mt
    } else if (prefix == "step") {
      printf "step_field %s models %d/%d steps %d/%d\n", f, me, mt, se, st
      if (f == "qpos" && (mt != 212 || st != 4240 || me != mt || se != st)) bad_count = 1
    }
  }
  if (first != "") {
    print "verdict first_divergence"
    print "first " first
  } else if (bad_count) {
    print "verdict first_divergence"
    print "first count is not 212 models, 636 native states, and 4240 steps"
  } else {
    print "verdict full_parity"
  }
}
' >"$tmp"

{
  echo "x86_oracle mujoco-3.13.0-x86_64-contract-off"
  echo "x86_libmujoco_sha256 $X86_LIB"
  echo "x86_native_sha256 $X86_NATIVE_SHA"
  echo "x86_step_sha256 $X86_STEP_SHA"
  cat "$tmp"
} | tee "$OUT"
rm -f "$tmp"
if grep -q '^verdict full_parity$' "$OUT"; then
  exit 0
fi
exit 1
