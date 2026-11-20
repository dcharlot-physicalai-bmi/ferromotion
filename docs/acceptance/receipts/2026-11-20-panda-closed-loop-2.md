# Panda closed loop, second receipt, 2026-11-20

This receipt supersedes `docs/acceptance/receipts/2026-11-20-panda-closed-loop.md`. That file is kept and marked superseded. Its hashes are the episode before the spike and torque gates, the zero-torque control, the readout-limit assert, and the per-tick contact count.

Scope: contact-free dynamics only. `JOINT_LOOP_EXECUTED` stays pending bmi-2 review. `EPISODE_ENVELOPE_INTEROPERABLE` stays false. bmi-2 will sign off the envelope together with the loop.

The plant is `MjcfTree::step` on `franka_emika_panda/panda_nohand.xml` from Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`. The vendored file is `crates/ferromotion-core/tests/fixtures/panda_nohand.xml`, sha256 `e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071`. `fullinertia` on every body covers mass and inertia. The 1 mm cube stand-ins replace the Menagerie collision geometry. `ncon` is 0 on every tick of the closed loop and of the zero-torque control, so that stand-in geometry does not enter the step.

The position servos stay general actuators with gain and the first three bias parameters set to zero after their `forcerange` is checked. Torque is `qfrc_applied`. There is no clamp after `SpikeReadout::apply`. The test asserts that each readout component's bits already equal the bits of that component clamped to ±87 N·m for joints 1–4 and ±12 N·m for joints 5–7.

Scale stays `5e-9` A/rad and offset stays `3e-9` A. At the home pose the joints whose position is 0 draw 3 nA, and the default LIF threshold is below the steady potential of that current. The weight stays `0.05` N·m per count. Those parameters already fire, so they were not changed.

Ferromorphic pin `82211dc861af4ecc1afa9603c2caf20d52712b6d`, tree `7ba6c96268ced276556b85bed24173c56d3ec3d6`, no path override. `N` is 16. The window is 4 ticks. The episode starts at the `home` keyframe. The envelope text still crosses the crate boundary in both directions, byte for byte. The second run of the episode matches the first. The zero-torque control, from the same home keyframe and the same 16 steps, was also run twice and matched itself. Its final `qpos`/`qvel` hash differs from the closed-loop plant hash.

* spikes over the episode: 52
* first tick with a spike: 1
* ticks with non-zero tau: 15
* `ncon` on ticks 0–15, closed loop and control: 0
* max `|tau|` by joint, N·m, and bits: joint1 `0.2` `3fc999999999999a`, joint2 `0.2` `3fc999999999999a`, joint3 `0.2` `3fc999999999999a`, joint4 `0.1` `3fb999999999999a`, joint5 `0.1` `3fb999999999999a`, joint6 `0.15000000000000002` `3fc3333333333334`, joint7 `0.2` `3fc999999999999a`

* Cargo.lock sha256 `7a0ddbf92a1233fcc17f57ca5e5ae2f060096c33a60a4b94082779219b2653e9`
* `rustc -Vv`: rustc 1.100.0 (0ef1bc386 2026-11-09), commit-hash `0ef1bc386059596dada8fb65b50425aab2ea637a`, host `x86_64-unknown-linux-gnu`, LLVM 23.1.5
* cargo 1.100.0 (3acc79c0c 2026-11-05)
* log `docs/acceptance/receipts/2026-11-20-panda-closed-loop-2.log`
* log sha256 `e67a826f1af8349df5fe983da0102c7ae10c94a3ee07180d539f51d15cd48cd6`
* envelope text sha256 `170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a`
* final `Sim::state_bytes` sha256 `b2bfb4081bcd7e2de05fcd249c65b2c5a5cfe7817850eac854467df32ea3a126`
* closed-loop final `qpos`/`qvel` sha256 `b6ed1ec1ca5d2086865d82a056ed16871198f2ebdeb233d6115670b028a3b6e0`
* zero-torque control final `qpos`/`qvel` sha256 `0041db4cb786ce8a7ad90624804aede87b00eda1da873e6aef9bc10d520432a5`

Menagerie pin `4d038b3feae26ec82b46a4d586379114012a8ac7`. Oracle pins in `conformance/menagerie_ledger.json` for that commit: MuJoCo 3.13.0, step oracle sha256 `6acb55b5ca91040397afb4c29733571a21323876a140aff8731fb8cee49428a4`, native oracle sha256 `b1126e028df2bd4d0690f43fd67c5ced9147f8179f4b6ab70c5ba81a5cf04b11`. This episode does not re-run that oracle.

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
