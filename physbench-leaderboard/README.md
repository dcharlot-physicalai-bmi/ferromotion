# Physics-Fidelity Benchmark — Leaderboard

An open, CI-scored leaderboard for the [physics-fidelity benchmark](https://physicalai-bmi.org/research/physics-fidelity):
does a dynamics model **obey physics** (conservation laws), or only look accurate?

A submission is any model that can step a state forward. The **harness** scores it — never the submission — so
the board cannot be gamed. Scores are computed by CI on every push; the committed [`standings.json`](./standings.json)
and [`STANDINGS.md`](./STANDINGS.md) are the persistent board, and the website reads the same `standings.json`.

## Current board

See [STANDINGS.md](./STANDINGS.md). The tell: a model can be accurate step to step and still violate the
invariant over a rollout — the symplectic and explicit-Euler references have the *same* one-step accuracy and
opposite verdicts. Structure, not per-step accuracy, is what earns a PASS.

## Submit your model

1. **Fork** this repository.
2. **Add one file** `physbench-leaderboard/submissions/<your-model>.rs` that defines a `Model` and its metadata:

   ```rust
   use crate::bench::{Meta, Model};

   pub const META: Meta = Meta { name: "my-model", author: "your-handle", kind: "learned" };

   pub struct M;
   impl Model for M {
       // Advance the frictionless-pendulum state (θ, ω) forward by dt. This is where your model goes —
       // an integrator, a Hamiltonian net, a learned world model with its weights embedded, anything.
       fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64) {
           // ... your dynamics ...
           (th + dt * w, w)
       }
   }
   ```

   That is the whole interface. Submissions are auto-discovered — no shared file to edit, no registration.
   A learned model embeds its (small) weights and runs its forward pass inside `step`; see
   [`crates/ferromotion-core/examples/wm_on_the_stand.rs`](../crates/ferromotion-core/examples/wm_on_the_stand.rs)
   for a trained black-box (86% energy drift, FAIL) vs a structured net (4.2%, PASS) on this exact system.

3. **Open a pull request.** CI builds every submission, runs the harness, and posts the updated standings to
   the job summary. On merge to `main`, CI regenerates and commits `standings.json` and `STANDINGS.md`.

Run it yourself: `cd physbench-leaderboard && cargo run --release --bin score`.

## What is scored

The system is the frictionless pendulum (θ̈ = −(g/L)sinθ, L=1), whose invariants have a closed form:

| metric | meaning | pass |
|--------|---------|------|
| energy drift | max relative change of total energy over an 8 s rollout | < 5% |
| reversibility | forward-then-reverse return error | < 5% |
| one-step RMSE | per-step accuracy vs an RK4 reference (reported, **not** a pass gate) | — |

Verdict = energy drift < 5% **and** reversibility < 5%. New probes (with analytic or engine-verified ground
truth) and new systems are welcome — open an issue or PR.
