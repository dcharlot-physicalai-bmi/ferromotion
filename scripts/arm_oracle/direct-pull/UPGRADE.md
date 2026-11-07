# Corrected recovery for the linked Thor2 build

Do not run `03-cbuild.sh` again. The libraries are already linked. The schedules are already materialized.

`recover-stage3.sh` is the only file to download. sha256:

```text
6316c325563b10dfc7d19c8738f863db77ce5dcf5794bd5a0627ad5c5498ff97  recover-stage3.sh
```

The immutable URL is the raw file at the commit that contains this exact blob. On thor2:

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=41ed70b2c14aee8f028018df14c60c05faf668be
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' \
  -o recover-stage3.sh \
  "https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}/scripts/arm_oracle/direct-pull/recover-stage3.sh"
sha256sum -c - <<'EOF'
6316c325563b10dfc7d19c8738f863db77ce5dcf5794bd5a0627ad5c5498ff97  recover-stage3.sh
EOF
bash recover-stage3.sh
```

The script still requires Linux aarch64, hostname `thor2`, the original five library hashes, the contract-off flags, and both stage3 logs. It does not run cmake and it does not write `stage3-ok` unless `write-receipt.sh` succeeds.
