#!/usr/bin/env bash
# Every published crate must carry the license text it claims.
#
# The workspace declares `license = "MIT OR Apache-2.0"`, and LICENSE-MIT / LICENSE-APACHE live at the
# repository root. But `cargo package` only includes files UNDER each crate's own directory and will not
# reach outside it — so before this, every published .crate contained src/, examples/ and README.md with no
# license text at all, while asserting a dual license. MIT requires its notice to ship "in all copies or
# substantial portions of the Software", and a published crate is a copy.
#
# Symlinks are not a reliable fix (cargo's handling of them inside packages has varied across versions), and
# `license-file` replaces the SPDX expression rather than complementing it. Copying is what works.
#
# Run after adding a crate:  bash scripts/sync-crate-licenses.sh
set -euo pipefail
cd "$(dirname "$0")/.."
n=0
for d in crates/*/; do
  cp -f LICENSE-MIT "$d/LICENSE-MIT"
  cp -f LICENSE-APACHE "$d/LICENSE-APACHE"
  n=$((n + 1))
done
echo "synced LICENSE-MIT + LICENSE-APACHE into $n crate directories"
