# Upgrade from ecfccc6 for this Thor2 run

Stages 1 and 2 of `ecfccc6c37edf54b0b51c3649789732b3c60dc89` stay as they ran.
Do not run stage 3 again. The libraries are already at the build directory below.
`ecfccc6` stopped after the link because `write-receipt.sh` could not bind the missing schedules, so it did not write `stage3-ok`.

Schedule fixtures and the recovery script are in `0238786ef740de57b5a766c07f3de98c4cfb7d55`.
Inspect these URLs, then run the block. It is not `curl | sh`. The downloads are the two xz files only. The shell below checks their sha256 before decompressing.

```text
https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/0238786ef740de57b5a766c07f3de98c4cfb7d55/scripts/arm_oracle/direct-pull/schedules/x86-native.txt.xz
https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/0238786ef740de57b5a766c07f3de98c4cfb7d55/scripts/arm_oracle/direct-pull/schedules/x86-step.txt.xz
https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/0238786ef740de57b5a766c07f3de98c4cfb7d55/scripts/arm_oracle/direct-pull/recover-stage3.sh
https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/0238786ef740de57b5a766c07f3de98c4cfb7d55/scripts/arm_oracle/direct-pull/materialize-schedules.sh
```

xz sha256:

```text
1ed398c90a318fd9a514c7f6c8e14b966427c577f31410c872e90f84df537c4e  x86-native.txt.xz
237fdfb8ea17eafe3a01f73c3418ecdc03af65b86e711f80d137353a1ccd55ed  x86-step.txt.xz
```

Plaintext sha256, after `xz -dc`:

```text
a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7  x86-native.txt
a039f949d938cefc0cf3633edec1415497f8400a776b88809adf9bb06f5678be  x86-step.txt
```

On thor2, from a new empty directory, with no `GH_TOKEN` and no credential helper:

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=0238786ef740de57b5a766c07f3de98c4cfb7d55
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}/scripts/arm_oracle/direct-pull"
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
D="$HOME/ferromotion-oracle/schedule-fetch-${C}"
mkdir -p "$D"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/x86-native.txt.xz" "$BASE/schedules/x86-native.txt.xz"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/x86-step.txt.xz" "$BASE/schedules/x86-step.txt.xz"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/materialize-schedules.sh" "$BASE/materialize-schedules.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/recover-stage3.sh" "$BASE/recover-stage3.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/pins.sh" "$BASE/pins.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/lib.sh" "$BASE/lib.sh"
cd "$D"
sha256sum -c - <<'EOF'
1ed398c90a318fd9a514c7f6c8e14b966427c577f31410c872e90f84df537c4e  x86-native.txt.xz
237fdfb8ea17eafe3a01f73c3418ecdc03af65b86e711f80d137353a1ccd55ed  x86-step.txt.xz
EOF
# materialize-schedules.sh expects the xz files beside itself and sources pins.sh and lib.sh
mkdir -p "$D/schedules"
mv x86-native.txt.xz x86-step.txt.xz schedules/
bash materialize-schedules.sh "$RUN"
bash recover-stage3.sh
```

`recover-stage3.sh` refuses unless the five libraries, `identity.txt`, `build.ninja`, and both stage3 logs match this build. It does not run cmake. After it writes `stage3-ok`, `ecfccc6`'s `04-corpus.sh` can run against the same run directory. Do not start `03-cbuild.sh` again.
