#!/usr/bin/env bash
# Print the one data line of the x86 mulquat fixture.
# Comment lines and blank lines are not data. Zero data lines, or more than one, is a refusal.
set -euo pipefail
export LC_ALL=C
file="${1:?mulquat fixture}"
[[ -f "$file" ]] || {
  echo "refusing: mulquat fixture $file is missing" >&2
  exit 2
}
n=0
data=""
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[[:space:]]*$ ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  if [[ ! "$line" =~ ^[0-9a-f]{16}[[:space:]]+[0-9a-f]{16}[[:space:]]+[0-9a-f]{16}[[:space:]]+[0-9a-f]{16}$ ]]; then
    echo "refusing: mulquat fixture data line is not four 16-digit hex fields: $line" >&2
    exit 2
  fi
  n=$((n + 1))
  data="$line"
done <"$file"
if [[ "$n" -ne 1 ]]; then
  echo "refusing: mulquat fixture has $n data lines, need exactly one" >&2
  exit 2
fi
printf '%s\n' "$data"
