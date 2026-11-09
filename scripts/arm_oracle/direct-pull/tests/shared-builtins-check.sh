#!/usr/bin/env bash
# x86 stand-in for the Thor2 failure. A hidden __muloti4 in a static archive
# cannot satisfy an undefined reference in an already-linked DSO. Defining the
# symbol inside the shared library does. aarch64 clang emits the libcall for
# __builtin_mul_overflow of __int128; x86_64 inlines it.
set -euo pipefail
export LC_ALL=C
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=../../compiler-rt.sh
source "$ROOT/compiler-rt.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat >"$work/ov.c" <<'EOF'
int mul128(__int128 a, __int128 b, __int128 *r) { return __builtin_mul_overflow(a, b, r); }
int mul64(long long a, long long b, long long *r) { return __builtin_mul_overflow(a, b, r); }
EOF
clang -O2 -c "$work/ov.c" -o "$work/x.o"
if nm "$work/x.o" | grep -q ' U __muloti4'; then
  echo "x86_64 unexpectedly references __muloti4" >&2
  exit 1
fi
clang --target=aarch64-linux-gnu -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a -c "$work/ov.c" -o "$work/a.o"
nm "$work/a.o" | grep -q ' U __muloti4'

printf '.hidden __muloti4\n.globl __muloti4\n__muloti4:\nret\n' >"$work/hid.s"
printf '.text\n.globl use\nuse:\ncall __muloti4\nret\n' >"$work/use.s"
clang -c -fPIC "$work/hid.s" -o "$work/hid.o"
clang -c -fPIC "$work/use.s" -o "$work/use.o"
ar rcs "$work/libclang_rt.builtins-x86_64.a" "$work/hid.o"
mkdir -p "$work/badlib" "$work/goodlib"
clang -shared -fPIC "$work/use.o" -o "$work/badlib/libmujoco.so.3.13.0"
if arm_oracle_reject_undefined_builtins "$work/badlib" >"$work/bad.out" 2>"$work/bad.err"; then
  echo "undefined __muloti4 was accepted" >&2
  exit 1
fi
grep -F -q "undefined compiler-rt symbol __muloti4" "$work/bad.err"

# CMake puts SHARED_LINKER_FLAGS before the .o. A bare archive there does not bind.
clang -shared -fPIC "$work/libclang_rt.builtins-x86_64.a" -o "$work/badlib/libearly.so" "$work/use.o"
if arm_oracle_reject_undefined_builtins "$work/badlib" >"$work/early.out" 2>"$work/early.err"; then
  echo "archive-before-object cleared __muloti4" >&2
  exit 1
fi

clang -shared -fPIC -Wl,--whole-archive "$work/libclang_rt.builtins-x86_64.a" -Wl,--no-whole-archive \
  -o "$work/goodlib/libmujoco.so.3.13.0" "$work/use.o"
arm_oracle_reject_undefined_builtins "$work/goodlib"
printf 'int main(void){return 0;}\n' >"$work/main.c"
clang "$work/main.c" "$work/goodlib/libmujoco.so.3.13.0" -o "$work/exe-good"
set +e
clang "$work/main.c" "$work/badlib/libmujoco.so.3.13.0" "$work/libclang_rt.builtins-x86_64.a" -o "$work/exe-bad" >"$work/exe-bad.err" 2>&1
exe_bad=$?
set -e
[[ "$exe_bad" -ne 0 ]]
grep -F -q "hidden symbol" "$work/exe-bad.err"
grep -F -q "referenced by DSO" "$work/exe-bad.err"

grep -F -q 'FLAGS="-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -fno-math-errno -march=armv8-a"' "$ROOT/build-thor2.sh"
grep -F -q 'arm_oracle_shared_link_flags' "$ROOT/build-thor2.sh"
grep -F -q 'CMAKE_SHARED_LINKER_FLAGS' "$ROOT/build-thor2.sh"
grep -F -q -- '-rtlib=compiler-rt -unwindlib=libgcc' "$ROOT/compiler-rt.sh"
if grep -q 'arm_oracle_compiler_rt_path' "$ROOT/run-probes.sh" || grep -q -- '-rtlib=compiler-rt' "$ROOT/run-probes.sh"; then
  echo "run-probes.sh still appends compiler-rt" >&2
  exit 1
fi
grep -F -q 'undefined __muloti4' "$ROOT/direct-pull/recover-stage3.sh"
echo "shared builtins check passed"
