#!/usr/bin/env bash
# Apply the small V2 script overlay onto an unpacked tree.
# Does not read the 1.27GB archive unless --archive is passed.
# Refuses unless --base-sha is the V2 archive digest, the manifest file is
# still the V2 manifest, and every replaced file still has its V2 hash.
set -euo pipefail
export LC_ALL=C
BASE_SHA=0c98f3bc63175eb56af0c40164ae78aac43efef231875042986fe1df1bd0c0ad
EXPECTED_LINES=35371
OVERLAY_DIR="$(cd "$(dirname "$0")" && pwd)"
TREE=""
ARCHIVE=""
BASE_ARG=""
usage() {
  echo "usage: $0 --tree <unpacked 123347c0... directory> --base-sha $BASE_SHA [--archive <v2.tar.gz>]" >&2
  exit 2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree) TREE="${2:?}"; shift 2 ;;
    --base-sha) BASE_ARG="${2:?}"; shift 2 ;;
    --archive) ARCHIVE="${2:?}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$TREE" && -n "$BASE_ARG" ]] || usage
if [[ "$BASE_ARG" != "$BASE_SHA" ]]; then
  echo "refusing: --base-sha is not the V2 archive $BASE_SHA" >&2
  exit 2
fi
if [[ ! -f "$OVERLAY_DIR/manifest-delta.txt" || ! -f "$OVERLAY_DIR/BASE-MANIFEST.sha256" || ! -d "$OVERLAY_DIR/files" ]]; then
  echo "refusing: $OVERLAY_DIR is missing manifest-delta.txt, BASE-MANIFEST.sha256, or files/" >&2
  exit 2
fi
TREE="$(cd "$TREE" && pwd)"
if [[ ! -f "$TREE/MANIFEST.sha256" ]]; then
  echo "refusing: $TREE/MANIFEST.sha256 is missing" >&2
  exit 2
fi
if [[ -n "$ARCHIVE" ]]; then
  got="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  if [[ "$got" != "$BASE_SHA" ]]; then
    echo "refusing: archive sha256 $got is not $BASE_SHA" >&2
    exit 2
  fi
  echo "archive sha256 matches $BASE_SHA"
fi
manifest_got="$(sha256sum "$TREE/MANIFEST.sha256" | awk '{print $1}')"
manifest_want="$(awk '{print $1}' "$OVERLAY_DIR/BASE-MANIFEST.sha256")"
if [[ "$manifest_got" != "$manifest_want" ]]; then
  echo "refusing: MANIFEST.sha256 sha256 $manifest_got is not the V2 manifest $manifest_want" >&2
  exit 2
fi
lines="$(grep -c . "$TREE/MANIFEST.sha256")"
if [[ "$lines" -ne "$EXPECTED_LINES" ]]; then
  echo "refusing: MANIFEST.sha256 has $lines entries, V2 has $EXPECTED_LINES" >&2
  exit 2
fi

allow_path() {
  case "$1" in
    scripts/*|probes/*|ferromotion/scripts/arm_oracle/*|provenance/OVERLAY.txt|ferromotion/.cargo/config.toml|rust/OPTIONAL-RUSTUP.md) ;;
    *)
      echo "refusing: $1 is outside the overlay allowlist" >&2
      exit 2
      ;;
  esac
  case "$1" in
    *..*|/*|./*|*//*)
      echo "refusing: unsafe path $1" >&2
      exit 2
      ;;
  esac
}

declare -A old_of new_of
declare -a order kind_of
n=0
while read -r kind a b c; do
  [[ -z "${kind:-}" || "$kind" == \#* ]] && continue
  if [[ "$kind" == "REPLACE" ]]; then
    [[ -n "$c" ]] || { echo "refusing: REPLACE line is short" >&2; exit 2; }
    allow_path "$c"
    old_of["$c"]="$a"
    new_of["$c"]="$b"
    order[$n]="$c"
    kind_of[$n]="REPLACE"
  elif [[ "$kind" == "ADD" ]]; then
    [[ -n "$b" && -z "${c:-}" ]] || { echo "refusing: ADD line is malformed" >&2; exit 2; }
    allow_path "$b"
    new_of["$b"]="$a"
    order[$n]="$b"
    kind_of[$n]="ADD"
  else
    echo "refusing: bad delta kind $kind" >&2
    exit 2
  fi
  n=$((n + 1))
done <"$OVERLAY_DIR/manifest-delta.txt"
[[ "$n" -gt 0 ]] || { echo "refusing: empty delta" >&2; exit 2; }

work="$(mktemp -d)"
cp -f "$TREE/MANIFEST.sha256" "$work/MANIFEST.sha256.v2"
i=0
n_replace=0
n_add=0
while [[ "$i" -lt "$n" ]]; do
  path="${order[$i]}"
  src="$OVERLAY_DIR/files/$path"
  [[ -f "$src" ]] || { echo "refusing: overlay payload missing $path" >&2; exit 2; }
  got="$(sha256sum "$src" | awk '{print $1}')"
  [[ "$got" == "${new_of[$path]}" ]] || { echo "refusing: payload $path sha256 $got is not ${new_of[$path]}" >&2; exit 2; }
  if [[ "${kind_of[$i]}" == "REPLACE" ]]; then
    n_replace=$((n_replace + 1))
    [[ -f "$TREE/$path" ]] || { echo "refusing: $TREE/$path is missing" >&2; exit 2; }
    got="$(sha256sum "$TREE/$path" | awk '{print $1}')"
    [[ "$got" == "${old_of[$path]}" ]] || { echo "refusing: $path sha256 $got is not V2 hash ${old_of[$path]}" >&2; exit 2; }
    if ! grep -Fxq "${old_of[$path]}  ./${path}" "$TREE/MANIFEST.sha256"; then
      echo "refusing: MANIFEST.sha256 has no V2 line for $path" >&2
      exit 2
    fi
    mkdir -p "$work/backup/$(dirname "$path")"
    cp -f "$TREE/$path" "$work/backup/$path"
  else
    n_add=$((n_add + 1))
    if [[ -e "$TREE/$path" ]]; then
      echo "refusing: ADD destination already exists: $path" >&2
      exit 2
    fi
    if awk -v p="./$path" '$2 == p { found = 1 } END { exit found ? 0 : 1 }' "$TREE/MANIFEST.sha256"; then
      echo "refusing: MANIFEST.sha256 already lists $path" >&2
      exit 2
    fi
  fi
  i=$((i + 1))
done

restore() {
  local j=0
  while [[ "$j" -lt "$n" ]]; do
    local path="${order[$j]}"
    if [[ "${kind_of[$j]}" == "REPLACE" ]]; then
      cp -f "$work/backup/$path" "$TREE/$path"
    else
      rm -f "$TREE/$path"
    fi
    j=$((j + 1))
  done
  cp -f "$work/MANIFEST.sha256.v2" "$TREE/MANIFEST.sha256"
  echo "restored the V2 bytes for the overlay paths" >&2
}

i=0
while [[ "$i" -lt "$n" ]]; do
  path="${order[$i]}"
  mkdir -p "$TREE/$(dirname "$path")"
  cp -f "$OVERLAY_DIR/files/$path" "$TREE/$path"
  if [[ "$path" == *.sh ]]; then
    chmod +x "$TREE/$path"
  fi
  got="$(sha256sum "$TREE/$path" | awk '{print $1}')"
  if [[ "$got" != "${new_of[$path]}" ]]; then
    restore
    echo "refusing: installed $path sha256 $got is not ${new_of[$path]}" >&2
    exit 3
  fi
  i=$((i + 1))
done

: >"$work/MANIFEST.sha256"
seen=0
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  path="${line#*  ./}"
  hash="${line%%  .*}"
  if [[ -n "${old_of[$path]:-}" ]]; then
    if [[ "$hash" != "${old_of[$path]}" ]]; then
      restore
      echo "refusing: manifest hash for $path changed during splice" >&2
      exit 3
    fi
    printf '%s  ./%s\n' "${new_of[$path]}" "$path" >>"$work/MANIFEST.sha256"
    seen=$((seen + 1))
  else
    printf '%s\n' "$line" >>"$work/MANIFEST.sha256"
  fi
done <"$work/MANIFEST.sha256.v2"
if [[ "$seen" -ne "$n_replace" ]]; then
  restore
  echo "refusing: spliced $seen manifest lines, delta replaces $n_replace" >&2
  exit 3
fi
i=0
while [[ "$i" -lt "$n" ]]; do
  if [[ "${kind_of[$i]}" == "ADD" ]]; then
    printf '%s  ./%s\n' "${new_of[${order[$i]}]}" "${order[$i]}" >>"$work/MANIFEST.sha256"
  fi
  i=$((i + 1))
done
new_lines="$(grep -c . "$work/MANIFEST.sha256")"
if [[ "$new_lines" -ne $((EXPECTED_LINES + n_add)) ]]; then
  restore
  echo "refusing: new manifest has $new_lines lines, expected $((EXPECTED_LINES + n_add))" >&2
  exit 3
fi
cp -f "$work/MANIFEST.sha256" "$TREE/MANIFEST.sha256"
echo "checking all $new_lines manifest entries"
if ! (cd "$TREE" && sha256sum -c MANIFEST.sha256); then
  restore
  echo "refusing: manifest check failed after the overlay. V2 bytes for the overlay paths were restored." >&2
  exit 3
fi
echo "overlay applied: $n_replace replaced, $n_add added, manifest entries $new_lines, base $BASE_SHA"
