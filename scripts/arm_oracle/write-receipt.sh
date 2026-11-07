#!/usr/bin/env bash
# Bind the libraries, schedules, executables, oracle outputs, compiler,
# source, toolchain, and platform. A libmujoco hash alone is refused.
# Missing plugin, schedule, or required output => exit 2.
set -euo pipefail
export LC_ALL=C
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=limits.sh
source "$SCRIPT_DIR/limits.sh"
OUT="${1:?output directory}"
LIBDIR="${2:?library directory}"
ROLE="${3:?role build|probe|corpus|rust|goldens}"
if ! ROOT="$(arm_oracle_root "$0")"; then
  exit 2
fi
if [[ ! -d "$OUT" ]]; then
  echo "refusing: receipt directory $OUT does not exist" >&2
  exit 2
fi

libs=(
  libmujoco.so.3.13.0
  libactuator.so
  libelasticity.so
  libsensor.so
  libsdf_plugin.so
)
need_libs=1
if [[ "$ROLE" == "goldens" ]]; then
  need_libs=0
fi
if [[ "$need_libs" -eq 1 ]]; then
  for lib in "${libs[@]}"; do
    if [[ ! -f "$LIBDIR/$lib" ]]; then
      echo "refusing: receipt cannot bind $LIBDIR/$lib" >&2
      exit 2
    fi
  done
fi

bound_inputs=(
  "$ROOT/schedules/x86-native.txt"
  "$ROOT/schedules/x86-step.txt"
  "$ROOT/provenance/REVISIONS"
  "$ROOT/provenance/ferromotion-commit.txt"
  "$ROOT/probes/oracle_dump.c"
  "$ROOT/probes/pyhex.h"
  "$ROOT/probes/hexfloat_selftest.c"
  "$ROOT/scripts/limits.sh"
  "$ROOT/scripts/apply-overlay.sh"
  "$ROOT/scripts/accept-rust-212.sh"
  "$ROOT/scripts/write-receipt.sh"
  "$ROOT/scripts/build-thor2.sh"
  "$ROOT/scripts/run-212.sh"
  "$ROOT/scripts/run-goldens.sh"
  "$ROOT/scripts/run-probes.sh"
  "$ROOT/scripts/run-rust-212.sh"
  "$ROOT/scripts/README.md"
  "$ROOT/rust/OPTIONAL-RUSTUP.md"
)
for path in "${bound_inputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "refusing: receipt cannot bind $path" >&2
    exit 2
  fi
done

extras=()
case "$ROLE" in
  build) ;;
  probe)
    extras=("$OUT/mulquat" "$OUT/mulquat-arm64.txt" "$OUT/mulquat-x86.txt" "$OUT/result.txt")
    ;;
  corpus)
    extras=("$OUT/oracle_dump" "$OUT/classify.txt" "$OUT/failures.txt" "$OUT/native.txt" "$OUT/step.txt" "$OUT/libdir.txt")
    ;;
  rust)
    extras=("$OUT/native.txt" "$OUT/step.txt" "$OUT/result.txt" "$OUT/accept.txt")
    ;;
  goldens)
    extras=("$OUT/cargo-test.txt" "$OUT/result.txt")
    ;;
  *)
    echo "refusing: unknown receipt role $ROLE" >&2
    exit 2
    ;;
esac
for path in "${extras[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "refusing: receipt cannot bind output $path" >&2
    exit 2
  fi
done
if [[ "$ROLE" == "rust" || "$ROLE" == "goldens" ]]; then
  arm_oracle_select_cargo
  if ! arm_oracle_cargo -vV >/dev/null 2>&1; then
    echo "refusing: receipt cannot bind the cargo toolchain" >&2
    exit 2
  fi
fi

receipt="$OUT/RECEIPT.txt"
tmp="$(mktemp)"
{
  echo "receipt_role $ROLE"
  if [[ "${FERROMOTION_ORACLE_SMOKE:-}" == "x86" ]]; then
    echo "oracle_class x86_smoke_not_arm"
  else
    echo "oracle_class arm64_linux_cpu_contract_off"
  fi
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname $(hostname)"
  echo "uname_s $(uname -s)"
  echo "uname_m $(uname -m)"
  echo "uname_r $(uname -r)"
  echo "clang $(clang --version 2>/dev/null | head -n 1 || echo absent)"
  echo "cmake $(cmake --version 2>/dev/null | head -n 1 || echo absent)"
  if command -v ninja >/dev/null 2>&1; then
    echo "ninja $(ninja --version)"
  else
    echo "ninja absent"
  fi
  if [[ "$ROLE" == "rust" || "$ROLE" == "goldens" ]]; then
    arm_oracle_cargo -vV | awk 'NR==1 || /^release:/ || /^host:/ {print "cargo_vv "$0}'
    if [[ "${ARM_ORACLE_CARGO[1]:-}" == "+stable" ]]; then
      echo "rustc_bin $(rustup which --toolchain stable rustc)"
      rustup run stable rustc --version | awk '{print "rustc_version "$0}'
    else
      echo "rustc_bin $(command -v rustc)"
      rustc --version | awk '{print "rustc_version "$0}'
    fi
  elif command -v cargo >/dev/null 2>&1; then
    cargo -vV | awk 'NR==1 || /^release:/ || /^host:/ {print "cargo_vv "$0}'
  else
    echo "cargo_vv absent"
  fi
  echo "mujoco_commit 123347c0eeab7e13c8da0828ab593bbd95bcf335"
  echo "menagerie_pin 4d038b3feae26ec82b46a4d586379114012a8ac7"
  echo "ferromotion_commit_file $(tr -d '[:space:]' <"$ROOT/provenance/ferromotion-commit.txt")"
  echo "limits_marker ${ARM_ORACLE_LIMITS:-unset}"
  echo "wall_timeout_seconds ${ARM_ORACLE_WALL_TIMEOUT:-unset}"
  echo "rlimit_cpu_seconds_per_process $(ulimit -t)"
  echo "rlimit_as_kib_per_process $(ulimit -v)"
  echo "rlimit_cpu_means per-process CPU seconds, not a sum across the process tree, not wall clock, and not memory"
  echo "rlimit_as_means per-process virtual address space in KiB, not resident set size, and not a cgroup memory.max"
  echo "wall_clock_means timeout(1) -k 30 on the heavy command process group; --foreground is not used"
  echo "job_cap 2"
  echo "omp_num_threads ${OMP_NUM_THREADS:-unset}"
  if [[ -f "$ROOT/provenance/OVERLAY.txt" ]]; then
    echo "overlay_file present"
  else
    echo "overlay_file absent"
  fi
  if [[ -L "$LIBDIR/libmujoco.so" ]]; then
    echo "libmujoco.so symlink $(readlink "$LIBDIR/libmujoco.so")"
  elif [[ -f "$LIBDIR/libmujoco.so" ]]; then
    echo "libmujoco.so regular_file"
  else
    echo "libmujoco.so absent"
  fi
  echo
  echo "# sha256  path"
  if [[ "$need_libs" -eq 1 ]]; then
    for lib in "${libs[@]}"; do
      sha256sum "$LIBDIR/$lib"
    done
    if [[ -f "$LIBDIR/libmujoco.so" ]]; then
      sha256sum "$LIBDIR/libmujoco.so"
    fi
  else
    echo "# goldens receipt does not bind libmujoco. The corpus receipt does."
  fi
  for path in "${bound_inputs[@]}"; do
    sha256sum "$path"
  done
  if [[ -f "$ROOT/provenance/OVERLAY.txt" ]]; then
    sha256sum "$ROOT/provenance/OVERLAY.txt"
  fi
  for path in "${extras[@]}"; do
    sha256sum "$path"
  done
} >"$tmp"
mv "$tmp" "$receipt"
sha256sum "$receipt" >"$OUT/RECEIPT.sha256"
echo "wrote $receipt"
