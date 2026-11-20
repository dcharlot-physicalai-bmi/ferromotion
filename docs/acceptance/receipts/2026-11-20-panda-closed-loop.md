# Superseded

Superseded by `docs/acceptance/receipts/2026-11-20-panda-closed-loop-2.md`. The hashes below are the episode before the spike and torque gates, the zero-torque control, the readout-limit assert, and the per-tick contact count. Do not use them as the current episode.

# Panda closed loop, 2026-11-20

Pending the commit that sets `JOINT_LOOP_EXECUTED`. This receipt is the episode. `EPISODE_ENVELOPE_INTEROPERABLE` stays false until bmi-2 and bmi-3 (Ferric) sign off.

The plant is `MjcfTree::step` on `franka_emika_panda/panda_nohand.xml` from Menagerie `4d038b3feae26ec82b46a4d586379114012a8ac7`. The vendored file is `crates/ferromotion-core/tests/fixtures/panda_nohand.xml`, sha256 `e7090a5e2384b18a223ef532d98bf914f7f9a6b74dc8bc9f74c8291201074071`. Every body in that file states `fullinertia`. The Menagerie STL and OBJ assets are not in this tree. The loader is given a 1 mm cube for each mesh path so the mesh compiler accepts the file. That cube is not the published collision shape. The step uses the compiled inertia, joint damping, armature, gravity, and `implicitfast` integrator.

The position servos are general actuators. The agreed command is joint torque. After the loaded `forcerange` values are checked, `gainprm[0]` and the first three `biasprm` entries are set to zero, and the saturated torque is `qfrc_applied`. `ctrl` is zero.

Ferromorphic is git rev `82211dc861af4ecc1afa9603c2caf20d52712b6d`, tree `7ba6c96268ced276556b85bed24173c56d3ec3d6`, no path override. Verify run `44755415928` concluded success on that head, including jobs `nine release gates` and `manifest, mutations, references`. `git diff 1ceec2fc884668db16905bed190a8c05873423e7 82211dc861af4ecc1afa9603c2caf20d52712b6d` changes comments in `src/sim.rs`, `docs/contracts/ferromotion-clock.md`, and `tools/fetch_heldout_inputs.py`. The bind sign-off stays `docs/acceptance/receipts/2026-11-20-bind-v2-signoff.md` for receipt `cfc22d8e` at pin `1ceec2fc`.

`N` is 16. The readout window is 4 ticks. The default LIF membrane constant is 20 ms and the plant timestep is 2 ms, so 16 steps is 32 ms: long enough for a suprathreshold current to fire and for four full windows to reach the readout.

Each tick reads `qpos` then `qvel`, length 14. `qpos` is clamped to joint1 ±2.8973, joint2 ±1.7628, joint3 ±2.8973, joint4 −3.0718…−0.0698, joint5 ±2.8973, joint6 −0.0175…3.7525, joint7 ±2.8973, in radians. Velocity is unclamped, in rad/s. `CurrentEncoder::apply` maps that observation to 14 ampere channels. `Sim::try_step` consumes them. Spike counts over the last 4 ticks, zero-padded at the start, go to `SpikeReadout::apply`. `tau` has length 7, in N·m, and is saturated at ±87 for joints 1–4 and ±12 for joints 5–7. `StepReceipt::from_step` stamps the spikes. `bind_v2_timestamps` is called with `n_action_steps == 1` on the snapshot after the neural step. The episode starts at the `home` keyframe.

`EpisodeEnvelope::to_text` on the ferromotion value is parsed by `ferromorphic::sim::EpisodeEnvelope::from_text`, and the reverse text is parsed by ferromotion `EpisodeEnvelope::from_text`. Both directions match byte for byte, including the final newline. The envelope `bind_v2_timestamps` receives is the ferromorphic value parsed from the ferromotion text.

The episode function was run twice. The per-tick hashes, the `state_bytes` hash, the envelope hash, and the plant hash matched.

* Cargo.lock sha256 `7a0ddbf92a1233fcc17f57ca5e5ae2f060096c33a60a4b94082779219b2653e9`
* `rustc -Vv`: rustc 1.100.0 (0ef1bc386 2026-11-09), commit-hash `0ef1bc386059596dada8fb65b50425aab2ea637a`, host `x86_64-unknown-linux-gnu`, LLVM 23.1.5
* cargo 1.100.0 (3acc79c0c 2026-11-05)
* log `docs/acceptance/receipts/2026-11-20-panda-closed-loop.log`
* log sha256 `e90db679dd90d59383368be5ee638433158819b88093065457b8e9b4d549fde2`

Menagerie pin `4d038b3feae26ec82b46a4d586379114012a8ac7`. The oracle pins recorded for that commit in `conformance/menagerie_ledger.json` are MuJoCo 3.13.0, step oracle sha256 `6acb55b5ca91040397afb4c29733571a21323876a140aff8731fb8cee49428a4`, native oracle sha256 `b1126e028df2bd4d0690f43fd67c5ced9147f8179f4b6ab70c5ba81a5cf04b11`. This episode does not re-run that oracle.

`Sim::state_bytes` sha256 `b2bfb4081bcd7e2de05fcd249c65b2c5a5cfe7817850eac854467df32ea3a126`

Envelope text sha256 `170e9795fc32875de5a6f6a531e5efd51760b73013e2b1b4fc5e9ab7d4f8091a`

Final plant `qpos`/`qvel` bits sha256 `b6ed1ec1ca5d2086865d82a056ed16871198f2ebdeb233d6115670b028a3b6e0`

| tick | sha256 |
|---|---|
| 0 | `f0b6b4fb27384ea9a4e0542655028a603843f1f59af19c50fe81980cd849a283` |
| 1 | `83f9efd472fbf0bd43d08904c11580962ea431ebac6f4e183640c3c997c0d0b2` |
| 2 | `d11a0f8df3e32895691c2a66b76df76cae9d26e70444f5a66fce5197067eee91` |
| 3 | `0209e8e7a701d680e84b47ca35483ec3aeadd52c56a7c44a187db1bc6899e967` |
| 4 | `26208d19502d98025ee87b190f2ce46353a1fa7d330b0e08257905d40d076fde` |
| 5 | `8fdb5964ddd56242e6610cf655413809d23ae93f2d2e09d4008134563cf9f684` |
| 6 | `e5e4f23c7d2f314f842f7ae07a8041468d5f19140170fc5bc4ff56847b8e53a2` |
| 7 | `49cf979a2df2daca9e6d421447fcda78ea14b7e2970fe5c091025dd95a90622a` |
| 8 | `0d5320f40dc92c1d90876f0b05dd65294e702a942c05c82c02a5c9fb40f9a01a` |
| 9 | `4f7625d7fc8f30d7173af1b66cc442dcf36b8297a538b621284deae5962278db` |
| 10 | `acfa06a0f20580e9bbcde7bf422613b34b920beb7d3152845fb987ced8958186` |
| 11 | `4d2a4ef4b517315322450382fce0690a74cf6b7bebaa11b33af64ab88c639c9a` |
| 12 | `c9fa92c1ae74ddf07ff8cc32baf5a8b127b2e85e676f7b426c0b320f0c288163` |
| 13 | `96636dc808673c50532a4cb985afd64102073a33e62dadbdfb7f3089e79067c0` |
| 14 | `ae78d91d1b6f11f69c1d272b760538949de8e32c97c1e9b0bb0e38baacd0434e` |
| 15 | `ef87fb1942a177b548a0d6c4b83a4c6bd49bb07a7db7a19af2dce17b8968b0ae` |
