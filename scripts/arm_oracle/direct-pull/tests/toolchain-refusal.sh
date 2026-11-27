#!/usr/bin/env bash
# A receipt may record only the rustc that built the binary.
# A mismatch, a missing rustc -vV, or a missing .comment rustc string is refused.
# An older cargo-vv.txt is not copied into the toolchain lines.
set -euo pipefail
export LC_ALL=C
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../../toolchain.sh
source "$here/../../toolchain.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/probe.rs" <<'EOF'
fn main() {}
EOF
echo 'int main(void) { return 0; }' >"$tmp/c.c"
rustc +1.83.0 "$tmp/probe.rs" -o "$tmp/rustc183"
rustc +1.100.0 "$tmp/probe.rs" -o "$tmp/rustc100"
gcc -o "$tmp/cbin" "$tmp/c.c"

rustc +1.100.0 -vV >"$tmp/vv100"
rustc +1.83.0 -vV >"$tmp/vv183"
printf 'COPIED_FROM_OLD_PROVENANCE cargo 1.99.1\nrelease: 1.99.1\n' >"$tmp/cargo-vv.txt"

lines="$tmp/lines"
if ! arm_oracle_toolchain_receipt_lines "$tmp/vv100" "$tmp/rustc100" >"$lines"; then
  echo "a matching rustc -vV and .comment was refused" >&2
  exit 1
fi
grep -q '^toolchain_rustc rustc 1.100.0 ' "$lines"
grep -q '^toolchain_comment rustc version 1.100.0 ' "$lines"
grep -q '^toolchain_source rustc-vv-at-build-and-elf-comment' "$lines"
if grep -q 'COPIED_FROM_OLD_PROVENANCE\|cargo-vv.txt' "$lines"; then
  echo "toolchain lines copied an older provenance file" >&2
  exit 1
fi

if ! arm_oracle_elf_comment_rustc "$tmp/rustc100" | grep -q '^rustc version 1.100.0 '; then
  echo "readelf did not read the rustc comment" >&2
  exit 1
fi
objdump_only="$(mktemp -d)"
ln -s "$(command -v objdump)" "$objdump_only/objdump"
ln -s "$(command -v grep)" "$objdump_only/grep"
ln -s "$(command -v sed)" "$objdump_only/sed"
ln -s "$(command -v awk)" "$objdump_only/awk"
ln -s "$(command -v tr)" "$objdump_only/tr"
ln -s /bin/bash "$objdump_only/bash"
if ! PATH="$objdump_only" arm_oracle_elf_comment_rustc "$tmp/rustc100" | grep -q '^rustc version 1.100.0 '; then
  echo "objdump -s -j .comment did not yield the rustc version" >&2
  exit 1
fi

refuse() {
  local name="$1"
  shift
  if "$@" >"$tmp/out" 2>"$tmp/err"; then
    echo "$name was accepted" >&2
    exit 1
  fi
  if ! grep -q '^refusing:' "$tmp/err"; then
    echo "$name did not say refusing" >&2
    cat "$tmp/err" >&2
    exit 1
  fi
}

refuse "disagree" arm_oracle_toolchain_receipt_lines "$tmp/vv100" "$tmp/rustc183"
refuse "missing rustc -vV" arm_oracle_toolchain_receipt_lines "$tmp/no-such-vv" "$tmp/rustc100"
refuse "missing binary" arm_oracle_toolchain_receipt_lines "$tmp/vv100" "$tmp/no-such-bin"
refuse "comment without rustc" arm_oracle_toolchain_receipt_lines "$tmp/vv100" "$tmp/cbin"
refuse "no binary recorded" arm_oracle_toolchain_receipt_lines "$tmp/vv100"

# The receipt helper reads the output directory and still ignores cargo-vv.txt.
mkdir -p "$tmp/out-dir/provenance"
cp "$tmp/vv100" "$tmp/out-dir/rustc-vv.txt"
printf '%s\n' "$tmp/rustc100" >"$tmp/out-dir/toolchain-bins.txt"
cp "$tmp/cargo-vv.txt" "$tmp/out-dir/provenance/cargo-vv.txt"
arm_oracle_emit_toolchain "$tmp/out-dir" >"$tmp/emitted"
if grep -q 'COPIED_FROM_OLD_PROVENANCE\|1\.99\.1' "$tmp/emitted"; then
  echo "emit copied cargo-vv.txt" >&2
  cat "$tmp/emitted" >&2
  exit 1
fi
grep -q '^toolchain_comment rustc version 1.100.0 ' "$tmp/emitted"

echo "toolchain refusal ok"
