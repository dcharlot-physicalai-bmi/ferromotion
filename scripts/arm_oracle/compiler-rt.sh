# clang's compiler-rt builtins. Source this file. Do not execute it.
# shellcheck shell=bash
#
# On aarch64, clang lowers __builtin_mul_overflow of __int128 to a call to
# __muloti4. On x86_64 the same builtin is inlined (mulq writes the 128-bit
# product in rdx:rax), so the x86 library does not reference __muloti4.
# The symbol in libclang_rt.builtins-*.a is GLOBAL HIDDEN. ld will not use a
# hidden archive symbol to satisfy an undefined reference in a DSO, which is
# why appending the archive to the probe link fails. The shared-library link
# has to absorb the builtin. CMAKE_SHARED_LINKER_FLAGS are emitted before the
# .o files, so a bare archive there is skipped. -rtlib=compiler-rt is a driver
# option and is not a compile flag.

# Driver options for the shared-library link only. Compile flags stay contract-off.
arm_oracle_shared_link_flags() {
  printf '%s\n' "-rtlib=compiler-rt -unwindlib=libgcc"
}

# Fail if a built shared library still imports a compiler-rt integer helper.
# Those names are __muloti4 and the other __*ti* / overflow helpers.
arm_oracle_reject_undefined_builtins() {
  local libdir="$1"
  local so sym bad path
  bad=0
  if ! command -v nm >/dev/null 2>&1; then
    echo "refusing: nm is not on PATH" >&2
    return 3
  fi
  shopt -s nullglob
  for so in "$libdir"/*.so "$libdir"/*.so.*; do
    [[ -f "$so" ]] || continue
    path="$so"
    while IFS= read -r sym; do
      [[ -n "$sym" ]] || continue
      if [[ "$sym" == "__muloti4" ]] || [[ "$sym" =~ ^__[A-Za-z0-9_]*ti[0-9]*$ ]] || [[ "$sym" =~ ^__[A-Za-z0-9_]*o[dt]i[0-9]+$ ]]; then
        echo "refusing: $path has undefined compiler-rt symbol $sym" >&2
        bad=1
      fi
    done < <(nm -D --undefined-only "$path" 2>/dev/null | awk '{print $NF}')
  done
  shopt -u nullglob
  [[ "$bad" -eq 0 ]] || return 3
  return 0
}

arm_oracle_compiler_rt_path() {
  local path base
  if ! command -v clang >/dev/null 2>&1; then
    echo "refusing: clang is not on PATH" >&2
    return 2
  fi
  if ! path="$(clang -print-libgcc-file-name -rtlib=compiler-rt)"; then
    echo "refusing: clang -print-libgcc-file-name -rtlib=compiler-rt failed" >&2
    return 2
  fi
  base="$(basename "$path")"
  if [[ -z "$path" || "$path" == "libgcc.a" || ! -f "$path" ]]; then
    echo "refusing: compiler-rt builtins archive is not a file (${path:-empty})" >&2
    return 2
  fi
  case "$base" in
    libclang_rt.builtins*.a) ;;
    *)
      echo "refusing: $path is not libclang_rt.builtins" >&2
      return 2
      ;;
  esac
  printf '%s\n' "$path"
}

# Write identity.txt for a probe, corpus, or rust output directory.
# The stage-3 build directory keeps its own identity.txt. This file records
# that path, the role, and the compiler-rt archive the libraries were linked with.
arm_oracle_write_role_identity() {
  local dest="$1" role="$2" stage3="$3"
  local rt="$stage3/compiler-rt.txt" ident="$stage3/identity.txt"
  local path sha got
  [[ -f "$ident" ]] || {
    echo "refusing: $ident is missing" >&2
    return 2
  }
  [[ -f "$rt" ]] || {
    echo "refusing: $rt is missing" >&2
    return 2
  }
  path="$(awk '/^compiler_rt_path /{print $2}' "$rt")"
  sha="$(awk '/^compiler_rt_sha256 /{print $2}' "$rt")"
  [[ -n "$path" && -f "$path" ]] || {
    echo "refusing: compiler-rt path in $rt is not a file" >&2
    return 2
  }
  got="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$got" == "$sha" ]] || {
    echo "refusing: compiler-rt sha256 does not match $path" >&2
    return 2
  }
  {
    echo "identity_role $role"
    echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname $(hostname)"
    echo "uname $(uname -s) $(uname -m)"
    echo "clang $(clang --version 2>/dev/null | head -n 1 || echo absent)"
    echo "stage3_build $stage3"
    echo "stage3_identity_sha256 $(sha256sum "$ident" | awk '{print $1}')"
    cat "$rt"
  } >"$dest"
}

arm_oracle_compiler_rt_record() {
  local dest="$1"
  local path sha
  path="$(arm_oracle_compiler_rt_path)" || return $?
  sha="$(sha256sum "$path" | awk '{print $1}')"
  {
    echo "compiler_rt_query clang -print-libgcc-file-name -rtlib=compiler-rt"
    echo "compiler_rt_path $path"
    echo "compiler_rt_sha256 $sha"
  } >"$dest"
}
