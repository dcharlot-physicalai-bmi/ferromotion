# Direct-pull receipt for the linked Thor2 build

## Stage 4 and 5 after the `__muloti4` link failure

Do not rerun stage 3. The library bytes stay `cefada9b5165a787e2c2695c9db172993c922f5533e19ed8108a181c53f85845`. The probe and `oracle_dump` link that library with `libclang_rt.builtins-aarch64.a` from `clang -print-libgcc-file-name -rtlib=compiler-rt`. The Rust examples do not link `libmujoco`.

Replace `REPLACE_WITH_FULL_SHA` with the full SHA of the commit that contains these scripts. The sha256 lines below are the fetched file bytes. They do not cover this markdown file. Fetch with tokens unset. `sha256sum -c` is the check. Then run stage 4 and stage 5 on the existing run directory.

```bash
set -euo pipefail
export LC_ALL=C
unset GH_TOKEN GITHUB_TOKEN GH_ENTERPRISE_TOKEN || true
C=REPLACE_WITH_FULL_SHA
BASE="https://raw.githubusercontent.com/dcharlot-physicalai-bmi/ferromotion/${C}"
D="$HOME/ferromotion-oracle/stage4-${C}"
RUN=/home/cbio/ferromotion-oracle/e0557dd109151a02702dfb262ea0a8c2b9dd07db-20261107T131232Z
mkdir -p "$D/runners"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/04-corpus.sh" "$BASE/scripts/arm_oracle/direct-pull/04-corpus.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/05-rust.sh" "$BASE/scripts/arm_oracle/direct-pull/05-rust.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/compare-arm-x86.sh" "$BASE/scripts/arm_oracle/direct-pull/compare-arm-x86.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/direct-pull-receipt.sh" "$BASE/scripts/arm_oracle/direct-pull/direct-pull-receipt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/lib.sh" "$BASE/scripts/arm_oracle/direct-pull/lib.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/pins.sh" "$BASE/scripts/arm_oracle/direct-pull/pins.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/compiler-rt.sh" "$BASE/scripts/arm_oracle/compiler-rt.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/limits.sh" "$BASE/scripts/arm_oracle/limits.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-probes.sh" "$BASE/scripts/arm_oracle/run-probes.sh"
curl -fsSL --proto '=https' -A 'ferromotion-direct-pull' -o "$D/runners/run-212.sh" "$BASE/scripts/arm_oracle/run-212.sh"
cd "$D"
sha256sum -c - <<'EOF'
b5400144299f607595629ee8e65d285cf713dfe457e2f791786a1d553885e61c  04-corpus.sh
a3d4d5930ed83b044ca10942a882d5c04748536c20f299c24a52c0cc3875a289  05-rust.sh
12363c3ff02ebb3dd7d85195682d390a4ee79acf4c6311bc3e6eeea6c99a4353  compare-arm-x86.sh
8ff7a83c73ddb8ab37760057b5da1663a684270e6522d2d6ff86acc9361b08e7  direct-pull-receipt.sh
bcd48f1e7f777992d60bdde1a2a6793b9ee00f93462bbcc28f655feb9ef1cfb1  lib.sh
5332e33e520f3b7bd66427174d6fada90aa1d16f5b1c812b4a53df099a7cc3a2  pins.sh
371b5f93051c9e3d2f303650705aa6b5e1de950b0fcb1f8c0b8785af0ce05045  runners/compiler-rt.sh
394990f2ca39eba1be353136fdb810bb0b269e892cc32f3b1ddf53b19735c049  runners/limits.sh
98106bbd38677fd3cf783d2c95b718d7472dc4c4d1ec2e62a0cc58e3b7c6854c  runners/run-probes.sh
244c719093d3bc9d3bc0df129f0b2d14f3a7edaaed14604c4c37f83f7cceb43f  runners/run-212.sh
EOF
chmod +x 04-corpus.sh 05-rust.sh compare-arm-x86.sh direct-pull-receipt.sh runners/*.sh
bash 04-corpus.sh "$RUN"
bash 05-rust.sh "$RUN"
```

Stage 4 exit 0 is ARM-versus-x86 full parity. Exit 1 is a finished divergence: `provenance/stage4-x86-verdict.txt` names the first field, and `stage4-ok` is already written. Exit 3 is still a compile or dump failure. `compiler_rt_path` and `compiler_rt_sha256` are lines in the probe and corpus `RECEIPT.txt` files.

## Earlier receipt recovery

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
