# Sign-off, panda closed loop, 2026-11-20

Signed off by physical-ai@bmi-2 (ferromorphic owner), 2026-11-20.

The signed head is ferromotion `31ad06a202af982af2564b1497d7ce4d779c9c00`. The receipt is `docs/acceptance/receipts/2026-11-20-panda-closed-loop-2.md`, sha256 `853bb5ecb56a3425f2553224e6ab18d8e318a0f0f0c03731885effefc13bc837`. Ferromorphic pin `82211dc861af4ecc1afa9603c2caf20d52712b6d`.

## What bmi-2 verified independently

Receipt `853bb5ecb56a3425f2553224e6ab18d8e318a0f0f0c03731885effefc13bc837` and log `e67a826f1af8349df5fe983da0102c7ae10c94a3ee07180d539f51d15cd48cd6`. The commits under review have 0 `Co-authored-by` trailers.

bmi-2 reran `cargo test -p ferromotion-core --test panda_closed_loop` on rustc 1.100.0, x86_64, Cargo.lock sha256 `7a0ddbf92a1233fcc17f57ca5e5ae2f060096c33a60a4b94082779219b2653e9`. That rerun gave 52 spikes, the first on tick 1, non-zero tau on 15 of 16 ticks, and the same max `|tau|` bits: joint1 `3fc999999999999a`, joint2 `3fc999999999999a`, joint3 `3fc999999999999a`, joint4 `3fb999999999999a`, joint5 `3fb999999999999a`, joint6 `3fc3333333333334`, joint7 `3fc999999999999a`.

ENVELOPE `170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a`, STATE `b2bfb4081bcd7e2de05fcd249c65b2c5a5cfe7817850eac854467df32ea3a126`, PLANT `b6ed1ec1ca5d2086865d82a056ed16871198f2ebdeb233d6115670b028a3b6e0`, and CONTROL `0041db4cb786ce8a7ad90624804aede87b00eda1da873e6aef9bc10d520432a5` matched. All 16 per-tick hashes matched:

| tick | sha256 |
|---|---|
| 0 | `42a48d5af5c79d62600e463e22e4b395dc387eb1a827cac416f7046c1aef6863` |
| 1 | `07d7e42a0055bb9c121d330a82981dad36060dfcc205dc020daf76681ae8aa83` |
| 2 | `066bee7c8664e92db77590a09e6c07d9ff3a21f9a1e5d7b4e3a12050086537c2` |
| 3 | `b2ab3175268745b8caee23e38d1e6fa58376a6c5ccebc4b10103aa0448193925` |
| 4 | `1a398f6ffe542d075a507ce2f66466854ed0befa4f6749cb4806e999c16e2298` |
| 5 | `eb8130fc8c878fbdae01aa8a8771a8ca7aeccea7cd2f188890bb14acc20b570a` |
| 6 | `823c7aeaade6d0776231e9d4603a7ad012babb3b2c9132f81f98c0a2cc6339e5` |
| 7 | `18a2de87ef8aa2a1fea39e4a25ddfcabb7330fff86fda38011c9dc57a533d4cb` |
| 8 | `23e7bdcae03b8aa038337ee30c7020fb0a18fa62116f29bf76b6487566ae20c6` |
| 9 | `49efabe6b7aa84f6b4f6d34726dd4a0617198dfc7de6e805ba0e571f7953cccd` |
| 10 | `87df866cb055b1e01e0a2e995356c6f1c462adddf5c5bb7603f329613eaafd3c` |
| 11 | `51628f2ee596f0522f5647b1a0ccdada9db3b2091a1ced557fb03da6a13ed696` |
| 12 | `35742d9c469b468c254705813e2d9c809e5baeb7fdc95c91bd49e3df65692250` |
| 13 | `7ce54b5b94d224d6317a44a98af9d19c465a04b16c157bab6f8b70335231450a` |
| 14 | `d2e919dfa2eb325456d5d6f3b8e3c5923fb1a5a815c6da252b4d48218ce68a31` |
| 15 | `9518d4d6641cdf55f0aee23405b8be704273e9e85f9cb72d336d1bab6bd24a6d` |

Bind tests passed 6/6.

## Scope

a 16-tick executed loop on panda_nohand, contact-free dynamics only (ncon=0 on every tick, mesh-free collision geometry), x86_64, determinism within one host. It is not a task-competence claim and not a cross-arch claim.

## CI caveat

the 3 published-default mjcf_tree failures are in `--lib`, so cargo stops before integration tests and skips clippy, docs and wasm. CI did not run panda_closed_loop or bind_v2 on 31ad06a2, so the sign-off rests on bmi-2's independent rerun.

## Envelope

bmi-2's half of EPISODE_ENVELOPE_INTEROPERABLE, dated 2026-11-20: both directions round-trip byte for byte through ferromorphic::sim::EpisodeEnvelope::from_text and through ours, and envelope 170e9795 reproduced on bmi-2's machine. The flag stays false until bmi-3 signs off.
