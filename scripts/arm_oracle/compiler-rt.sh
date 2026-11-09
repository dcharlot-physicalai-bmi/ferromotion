# clang's compiler-rt builtins. Source this file. Do not execute it.
# shellcheck shell=bash
#
# libmujoco built by clang on aarch64 references __muloti4. That symbol is
# provided by libclang_rt.builtins, not by libgcc_s. The shared library's
# NEEDED list does not change. Callers pass the archive on the link line.

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
