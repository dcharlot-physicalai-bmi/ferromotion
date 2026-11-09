#!/usr/bin/env bash
# The probe link appends the archive named by
# `clang -print-libgcc-file-name -rtlib=compiler-rt`.
# This host may not have the distro archive installed. A stub archive with the
# required basename still has to satisfy the helper, and a real clang link of
# an object that references __muloti4 has to fail without that archive and
# succeed with it.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../../compiler-rt.sh
source "$ROOT/compiler-rt.sh"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat >"$work/stub.s" <<'EOF'
.globl __muloti4
__muloti4:
  ret
EOF
clang -c "$work/stub.s" -o "$work/stub.o"
ar rcs "$work/libclang_rt.builtins-x86_64.a" "$work/stub.o"

cat >"$work/clang" <<EOF
#!/bin/sh
if [ "\$1" = "-print-libgcc-file-name" ]; then
  echo "$work/libclang_rt.builtins-x86_64.a"
  exit 0
fi
echo "\$*" >> "$work/clang-args"
exit 0
EOF
chmod +x "$work/clang"
PATH="$work:$PATH"
got="$(arm_oracle_compiler_rt_path)"
[[ "$got" == "$work/libclang_rt.builtins-x86_64.a" ]]
arm_oracle_compiler_rt_record "$work/compiler-rt.txt"
grep -F -q "compiler_rt_query clang -print-libgcc-file-name -rtlib=compiler-rt" "$work/compiler-rt.txt"
grep -F -q "compiler_rt_path $work/libclang_rt.builtins-x86_64.a" "$work/compiler-rt.txt"
sha="$(sha256sum "$work/libclang_rt.builtins-x86_64.a" | awk '{print $1}')"
grep -F -q "compiler_rt_sha256 $sha" "$work/compiler-rt.txt"

# A path that is not the builtins archive is refused.
cat >"$work/clang" <<EOF
#!/bin/sh
echo "$work/libgcc.a"
exit 0
EOF
chmod +x "$work/clang"
printf 'not builtins\n' >"$work/libgcc.a"
if arm_oracle_compiler_rt_path >"$work/bad.out" 2>"$work/bad.err"; then
  echo "libgcc.a was accepted" >&2
  exit 1
fi
grep -F -q "not libclang_rt.builtins" "$work/bad.err"

# Real link: undefined __muloti4 fails closed, and the archive resolves it.
PATH="/usr/bin:/bin"
printf '.globl main\nmain:\ncall __muloti4\nret\n' >"$work/need.s"
clang -c "$work/need.s" -o "$work/need.o"
set +e
clang "$work/need.o" -o "$work/without" >"$work/without.err" 2>&1
without_rc=$?
set -e
[[ "$without_rc" -ne 0 ]]
grep -F -q "__muloti4" "$work/without.err"
clang "$work/need.o" -o "$work/with" -lm "$work/libclang_rt.builtins-x86_64.a"
[[ -x "$work/with" ]]

# aarch64 object still names __muloti4. A relocatable link needs an aarch64 ld.
printf '.text\n.globl main\nmain:\nbl __muloti4\nret\n' >"$work/needa.s"
printf '.text\n.globl __muloti4\n__muloti4:\nret\n' >"$work/stuba.s"
clang --target=aarch64-linux-gnu -c "$work/needa.s" -o "$work/needa.o"
clang --target=aarch64-linux-gnu -c "$work/stuba.s" -o "$work/stuba.o"
nm "$work/needa.o" | grep -q ' U __muloti4'
ar rcs "$work/libclang_rt.builtins-aarch64.a" "$work/stuba.o"
set +e
clang --target=aarch64-linux-gnu -nostdlib -r "$work/needa.o" "$work/libclang_rt.builtins-aarch64.a" -o "$work/a-with.o" >"$work/a-with.err" 2>&1
a_with=$?
set -e
if [[ "$a_with" -eq 0 ]]; then
  if nm "$work/a-with.o" | grep -q ' U __muloti4'; then
    echo "aarch64 relocatable link left __muloti4 undefined" >&2
    exit 1
  fi
  echo "aarch64 relocatable link resolved __muloti4"
else
  grep -F -q "unrecognised emulation mode: aarch64linux" "$work/a-with.err"
  echo "aarch64 link not available on this ld; the object still references __muloti4"
fi
echo "compiler-rt link check passed"
