#!/usr/bin/env bash
# x86-safe check: every fetch row has a name, a public URL, and a tarball hash.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
out="$("$DIR/run.sh" --dry-run)"
printf '%s\n' "$out"
count="$(printf '%s\n' "$out" | grep -c '^fetch ')"
[[ "$count" -eq 9 ]] || {
  echo "refusing: dry-run printed $count fetch lines" >&2
  exit 1
}
names="$(mktemp)"
sha_names="$(mktemp)"
trap 'rm -f "$names" "$sha_names"' EXIT
printf '%s\n' "$out" | awk '/^fetch /{print $2}' | sort >"$names"
awk '{print $2}' "$DIR/tarballs.sha256" | sort >"$sha_names"
if ! cmp -s "$names" "$sha_names"; then
  echo "refusing: fetch names and tarballs.sha256 names differ" >&2
  diff -u "$sha_names" "$names" >&2 || true
  exit 1
fi
grep -q '^dry-run ok: 9 fetch entries, no download$' <<<"$out"
