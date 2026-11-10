# Shared guards for the unexecuted aarch64 direct-pull.
arm_oracle_refuse_unless_approved() {
  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
    echo "refusing: aarch64 direct-pull runs only on Linux aarch64 (got $(uname -s) $(uname -m))" >&2
    exit 2
  fi
  if [[ "$(hostname)" != "thor2" ]]; then
    echo "refusing: hostname is $(hostname), not thor2" >&2
    exit 2
  fi
  if [[ "${FERROMOTION_THOR2_APPROVED:-}" != "1" ]]; then
    echo "refusing: FERROMOTION_THOR2_APPROVED is not 1. Thor2 needs David's approval." >&2
    exit 2
  fi
}
