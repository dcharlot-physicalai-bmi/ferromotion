#!/usr/bin/env bash
# Classify menagerie_native and menagerie_native_step logs.
# The example programs return 0 after printing mismatches. This script does not.
# 0  every one of 212 models was compared, all 20 steps are present,
#    every model stepped, and every "N of M" tally has N == M.
# 1  those counts are complete and at least one tally does not match.
# 3  the summary is missing, a denominator is 0, a model was refused,
#    or the counts are not 212 models and 20 steps.
set -euo pipefail
export LC_ALL=C
NATIVE="${1:?native log}"
STEP="${2:?step log}"
if [[ ! -f "$NATIVE" || ! -f "$STEP" ]]; then
  echo "incomplete: summary file missing" >&2
  exit 3
fi

mapfile -t native_hits < <(grep -E "^MuJoCo 3\\.13\\.0's own arrays, BIT FOR BIT: [0-9]+ models compared \\([0-9]+ refused\\)\$" "$NATIVE" || true)
mapfile -t step_hits < <(grep -E "^mj_step against MuJoCo 3\\.13\\.0, BIT FOR BIT at every one of [0-9]+ steps: [0-9]+ of [0-9]+ models stepped \\([0-9]+ the loader refuses\\)\$" "$STEP" || true)
if [[ "${#native_hits[@]}" -ne 1 || "${#step_hits[@]}" -ne 1 ]]; then
  echo "incomplete: native_summaries=${#native_hits[@]} step_summaries=${#step_hits[@]} (need one of each)" >&2
  exit 3
fi
native_line="${native_hits[0]}"
step_line="${step_hits[0]}"
if [[ "$native_line" =~ BIT\ FOR\ BIT:\ ([0-9]+)\ models\ compared\ \(([0-9]+)\ refused\) ]]; then
  nmodels="${BASH_REMATCH[1]}"
  native_refused="${BASH_REMATCH[2]}"
else
  echo "incomplete: cannot parse native summary" >&2
  exit 3
fi
if [[ "$step_line" =~ at\ every\ one\ of\ ([0-9]+)\ steps:\ ([0-9]+)\ of\ ([0-9]+)\ models\ stepped\ \(([0-9]+)\ the\ loader\ refuses\) ]]; then
  nsteps="${BASH_REMATCH[1]}"
  exact="${BASH_REMATCH[2]}"
  compared="${BASH_REMATCH[3]}"
  step_refused="${BASH_REMATCH[4]}"
else
  echo "incomplete: cannot parse step summary" >&2
  exit 3
fi
native_tallies="$(grep -Ec '^  .*[0-9]+ of [0-9]+' "$NATIVE" || true)"
native_tallies="${native_tallies:-0}"
if [[ "$nmodels" -ne 212 || "$compared" -ne 212 || "$nsteps" -ne 20 || "$native_refused" -ne 0 || "$step_refused" -ne 0 || "$native_tallies" -lt 1 ]]; then
  echo "incomplete: native_models=$nmodels native_refused=$native_refused step_models=$compared step_refused=$step_refused steps=$nsteps native_tallies=$native_tallies (need 212 models, 0 refused, 20 steps, and at least one native tally)" >&2
  exit 3
fi

unequal=0
seen=0
while IFS= read -r line; do
  [[ "$line" =~ ([0-9]+)\ of\ ([0-9]+) ]] || continue
  seen=$((seen + 1))
  n="${BASH_REMATCH[1]}"
  m="${BASH_REMATCH[2]}"
  if [[ "$m" -eq 0 ]]; then
    echo "incomplete: tally denominator is 0: $line" >&2
    exit 3
  fi
  if [[ "$n" -ne "$m" ]]; then
    unequal=$((unequal + 1))
  fi
done < <(cat "$NATIVE" "$STEP")
if [[ "$seen" -lt 1 ]]; then
  echo "incomplete: no N-of-M tallies" >&2
  exit 3
fi
if [[ "$unequal" -ne 0 || "$exact" -ne 212 ]]; then
  echo "parity_mismatch models=212 steps=20 unequal_tallies=$unequal stepped=$exact of $compared"
  exit 1
fi
echo "parity_accept models=212 steps=20 tallies=$seen"
exit 0
