# Direct-pull receipt for the linked Thor2 build

Do not run `03-cbuild.sh` again. The libraries are already linked. The schedules are already checked.

`41ed70b` reached the five library hashes, then the archive `write-receipt.sh` exited 2 because `scripts/README.md` is not part of the stage-1 tree. That file is not created. The direct-pull receipt records the checkout commit and tree SHAs and sets `archive_manifest_rebuilt no`. It hashes `ferromotion/scripts/arm_oracle/README.md`, which the checkout already contains.

On thor2, from an empty directory, with no `GH_TOKEN`:

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=93117aa508b99de85fb550364a0aa8cc64b563b4
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}/scripts/arm_oracle/direct-pull"
D="$HOME/ferromotion-oracle/receipt-${C}"
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
mkdir -p "$D"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/recover-stage3.sh" "$BASE/recover-stage3.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/direct-pull-receipt.sh" "$BASE/direct-pull-receipt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/mulquat.c" "$BASE/probes/mulquat.c"
cd "$D"
sha256sum -c - <<'EOF'
4d78326f0c309d5f4d87e13fe526a0a26bd2578547db4ff02fdeef23f4fc4bbc  recover-stage3.sh
02ce38e7a7013539fee944c1ce3ba7473580d7bc8de673dcf04ebc674165e3a9  direct-pull-receipt.sh
1bb5bd6fb656623496075e1e8a89470731c7e44fbc6ad9cf7de82361e4b4bb80  mulquat.c
EOF
chmod +x recover-stage3.sh direct-pull-receipt.sh
# Replace the archive receipt only when it is still the e0557dd copy.
echo "bc7805f6768fedd3336dcc9e1ef8e7dd6647cc199bf839ea285fd74b59316a70  $RUN/scripts/write-receipt.sh" | sha256sum -c
cp -f direct-pull-receipt.sh "$RUN/scripts/write-receipt.sh"
cp -f mulquat.c "$RUN/probes/mulquat.c"
bash recover-stage3.sh
```

`recover-stage3.sh` still requires Linux aarch64, hostname `thor2`, and the five library hashes from the linked build. It does not run cmake. `stage3-ok` is written only after `direct-pull-receipt.sh` succeeds. Then `ecfccc6`'s `04-corpus.sh` can run. Do not start `03-cbuild.sh` again.
