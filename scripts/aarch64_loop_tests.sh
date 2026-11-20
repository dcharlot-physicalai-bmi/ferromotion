#!/usr/bin/env bash
# One command for the contact-free loop and the contact loop.
#
# rustc 1.100.0 (0ef1bc386 2026-11-09)
# commit-hash 0ef1bc386059596dada8fb65b50425aab2ea637a
# Jetson Thor host: aarch64-unknown-linux-gnu
#
# The locked receipt hashes were measured on x86_64-unknown-linux-gnu.
# This script does not skip either test on aarch64. The rerun is the comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
identity="$(rustc -Vv)"
printf '%s\n' "$identity" | grep -q '0ef1bc386059596dada8fb65b50425aab2ea637a'
exec cargo test --locked -p ferromotion-core --test panda_closed_loop --test panda_contact_loop "$@"
