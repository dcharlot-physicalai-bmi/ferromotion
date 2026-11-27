#!/usr/bin/env bash
# One command for the contact-free loop, the first contact loop, and the corrected reach.
#
# rustc 1.100.0 (0ef1bc386 2026-11-09)
# commit-hash 0ef1bc386059596dada8fb65b50425aab2ea637a
# Jetson Thor host: aarch64-unknown-linux-gnu
#
# The locked receipt hashes were measured on x86_64-unknown-linux-gnu.
# This script does not skip a test on aarch64. The rerun is the comparison.
#
# panda_contact_reach, profile mujoco-3.13.0-x86_64-contract-off,
# libmujoco sha256 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc.
# The x86_64 receipt is docs/acceptance/receipts/2026-11-27-panda-contact-reach-2.md.
# These hashes were measured on x86_64. An aarch64 match is not claimed.
#   plant    c3b70e92db29d9f112626d68d2d5c387895fb6f72dc95bf98d70262c34d300b2
#   zero     b3dc651ba290426b3e81a70db2e5aa59dbcc9fa787498934c07b5882925e04e8
#   open     a3a1c16ef5627785074a3c219490e1296a8233e62495430b517272f815a23852
#   state    4e49f36ed5148e355c13dd8a58c7d6b48d0062fb4cd56cce059117178ea40e50
#   envelope 170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a
set -euo pipefail
cd "$(dirname "$0")/.."
identity="$(rustc -Vv)"
printf '%s\n' "$identity" | grep -q '0ef1bc386059596dada8fb65b50425aab2ea637a'
exec cargo test --locked -p ferromotion-core --test panda_closed_loop --test panda_contact_loop --test panda_contact_reach "$@"
