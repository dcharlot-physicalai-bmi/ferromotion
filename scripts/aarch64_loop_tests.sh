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
# libmujoco sha256 64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc,
# code path 2aeb3611c9f3edca1961eb25ee374df56229d799:
#   plant   5c4c4514c93d3c667deecb6b4f834562ff7b2ad14dc61b6c88943c2f3b443fe7
#   zero    5fd57262dd51c8a559f470e7e5bc0b8a5fc2122bceb001696af4662d77fbe13c
#   open    2e3a8ce12ebd612b665b4f2d3516e1c31dae9b94f202f7bbd63c6bd50544227c
#   state   01e990011ad4d1e59d902d3fc1f480fca708b1cc574970de96f69fbe7a22fe82
#   envelope 170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a
set -euo pipefail
cd "$(dirname "$0")/.."
identity="$(rustc -Vv)"
printf '%s\n' "$identity" | grep -q '0ef1bc386059596dada8fb65b50425aab2ea637a'
exec cargo test --locked -p ferromotion-core --test panda_closed_loop --test panda_contact_loop --test panda_contact_reach "$@"
