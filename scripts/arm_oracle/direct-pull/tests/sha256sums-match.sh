#!/usr/bin/env bash
# SHA256SUMS is the tree. A hashed file that drifted, a listed path that is
# gone, or a file under direct-pull that was never listed, is a failure.
set -euo pipefail
export LC_ALL=C
pull="$(cd "$(dirname "$0")/.." && pwd)"
cd "$pull"
sha256sum -c SHA256SUMS >/dev/null
tree="$(mktemp)"
sums="$(mktemp)"
trap 'rm -f "$tree" "$sums"' EXIT
find . -type f ! -name SHA256SUMS | sed 's#^\./##' | sort >"$tree"
awk '{print $2}' SHA256SUMS | sed 's#^\./##' | sort >"$sums"
missing="$(comm -23 "$tree" "$sums" | grep -v '^\.\./' || true)"
if [[ -n "$missing" ]]; then
  echo "refusing: direct-pull files are not in SHA256SUMS" >&2
  printf '%s\n' "$missing" >&2
  exit 1
fi
while IFS= read -r path; do
  if [[ ! -f "$path" ]]; then
    echo "refusing: SHA256SUMS lists a missing file: $path" >&2
    exit 1
  fi
done <"$sums"
echo "sha256sums-match ok"
