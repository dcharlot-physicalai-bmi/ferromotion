# The Physics-Fidelity Benchmark

*Open, pure-Rust, reproducible. Reference implementation: `phys_fidelity.rs` and `wm_on_the_stand.rs` in this directory.*

## Why this exists

Learned world models and sim engines are judged on how **real** they look — and the whole field is scaling **data** to get there (video-hours, the largest-ever robot datasets, trillion-token world models). But a model can look perfect and still hallucinate energy, break momentum, penetrate contacts, or slide where friction says stick, because pixel/latent prediction carries no physics **structure**.

This benchmark scores the axis nobody is competing on: **does the model obey real physics?** It is not about appearance or per-step accuracy — it is about conservation-law fidelity, with **analytic ground truth** and a verified rigid-body engine ([ferromotion](../../..)) as the calibrated reference. A world model earns trust here by *obeying* physics, not by looking real.

## The model interface

A candidate **model** is a next-state predictor: given a state, predict the next state at a fixed timestep `dt`. That is all a learned world model, a neural simulator, or a numerical integrator has in common — so that is the interface the benchmark scores. Every probe rolls the model forward and checks an invariant with a known answer.

## Probes (v1) — `cargo run --release --example phys_fidelity`

| # | System | Invariant (ground truth) | Pass | Reference (correct physics) |
|---|--------|--------------------------|------|-----------------------------|
| A | simple pendulum | total mechanical energy is constant; period = 2π√(L/g) | energy drift < 1%, period err < 1% | symplectic: 0.15% / 0.06% |
| B | **the real SO-101** (ferromotion) | frictionless zero-torque energy is constant | drift < 2% | symplectic step: 0.13% |
| C | ball on a plane | non-penetration (z ≥ r); no energy created at contact | pen 0, gain 0 | restitution model: 0 / 0 |
| D | two masses + spring, isolated | linear & angular momentum conserved (Newton's 3rd law) | |ΔP|,|ΔL| ≈ 0 | physics: 1e-15 (machine precision) |
| E | block on an incline | static friction holds below the cone angle; slides at a = g(sinθ − μcosθ) above | rest-drift 0, accel-err < 2% | Coulomb: 0 / 0.03% |
| F | damped pendulum, unknown L, c | recover the true parameters from a few samples and extrapolate | — | physics-first: 0.0%, extrapolates 10,911× better than a structure-free fit |

Each probe is scored against three model classes shipped as controls: **PHYSICS** (structure-correct — passes), **explicit-Euler** (numerically wrong), and **learned/no-structure** (per-step accurate but structure-free). The reference column is what a correct engine produces; a candidate model prints alongside.

## Diagnosis → cure — `cargo run --release --example wm_on_the_stand`

A genuinely trained neural world model (2→16→16→2 MLP, Adam, pure Rust, **gradient-checked** so the training is verifiably real), same data and capacity throughout:

| Model | one-step accuracy | rollout energy drift | verdict |
|-------|-------------------|----------------------|---------|
| vanilla black-box next-state map | MSE 1.9e-4 | **86%** | FAIL — accurate yet non-conservative |
| **structured** (learn the force → integrate symplectically) | MSE 2.1e-6 | **4.2%** | PASS — obeys physics by construction |

Same data, same network capacity, same step size. The difference is not data or accuracy; it is **structure**. This is the constructive result behind the whole benchmark: you *can* have a learned model that obeys physics — by building the structure in (the Hamiltonian / Lagrangian / symplectic-net family).

### On the real 5-DOF arm — `cargo run --release --example wm_so101_cure`

The toy cure does not transfer as-is to a multi-body robot, and *why* is the deeper lesson. On the real SO-101, two learned models fit the same gravity data with the same true inertia M(q) and Coriolis and the same symplectic step; the only difference is how the configuration force is parameterized. A force field conserves energy only if it is a **gradient** — equivalently, its Jacobian ∂g/∂q is symmetric (‖J−Jᵀ‖ = 0). Both are gradient-checked.

| Model | force-Jacobian ‖J−Jᵀ‖ | true-energy drift @ 1 s / 3 s / 5 s |
|-------|----------------------|-------------------------------------|
| generic learned gravity field | 0.041 | 26% → 2,497% → **12,637%** — runaway |
| **potential gradient** g = ∇V | **0.000 (by construction)** | 49% → 100% → **100%** — bounded |
| exact gravity (fit → perfect) | 0.000 | 0.8% → 1.8% → 1.8% |

The finding is sharper than "it conserves." A *tiny* non-conservativeness (0.041) drives a **runaway**: the invented energy accelerates the arm into larger errors, which invent more energy — the drift compounds without bound. Parameterizing the force as ∇V makes it curl-free **by construction**, at any fit error, and the runaway disappears. What remained for the cure was a **bounded** offset, and chasing it down turned into the most useful result here. We first assumed it was ordinary fit error that training would remove. It was not: training the potential net 12× longer improved its force fit **95×** (MSE 1.31e-5 → 1.38e-7) and moved the 5-second drift only 162% → 127%, still 70× above the exact-force reference. So we instrumented the rollout and found the real cause — **the arm swings far outside the box the net was trained on, and outside its own physical joint limits**: joint 2 reached **+27 rad**, more than four revolutions, against a hardware limit of ±1.69 and a training range of ±1.01. The learned force was being asked for values in a regime it had never seen and the robot cannot reach, so no amount of in-distribution accuracy could help.

The fix is structural, not statistical. Gravity torque is **2π-periodic** in each joint angle, but a net fed raw angles cannot know that q = 27 rad is the same configuration as q = 1.87. Feed it **[sin q, cos q]** instead and that identity is built in, making the force valid at *any* angle the rollout wanders to:

| potential net input | force fit MSE | ‖J−Jᵀ‖ | energy drift @ 1 s / 3 s / 5 s |
|---|---|---|---|
| raw angles q | 1.31e-5 | 0.000 | 59% / 122% / **162%** |
| raw angles, 12× training | 1.38e-7 | 0.000 | 18% / 93% / **127%** |
| **[sin q, cos q]** | 8.12e-6 | 0.000 | 1.0% / 2.4% / **2.4%** |
| exact force (reference) | 0 | 0.000 | 0.8% / 1.8% / **1.8%** |

Encoding the periodicity brings the learned model to **2.4% against the exact-force reference's 1.8%** — a 65× improvement that 95× more fit accuracy could not buy. Checked across four initial conditions rather than trusting one run, and the stronger claim is that the learned model *tracks* the reference wherever it goes: 2.3% vs 2.5%, 2.4% vs 1.8%, 4.9% vs 4.3%, 1.6% vs 1.5% — including one start where the learned force conserves slightly better than the exact one. The reference's own figure moves between 1.5% and 4.3% as the trajectory changes, because that is the integrator's error, not the model's; a learned force that follows those movements is doing as well as the physics it replaced. Note the shape of the answer: twice now the fix was to build a known physical fact into the model (the force is a gradient; the torque is periodic), and both times more data or more training was the wrong lever. Remaining generalization: learn M(q) itself as a positive-definite net (a full Hamiltonian net); here M(q) is the known robot inertia — the same prior a free-form learned force was given, and still failed with, so the fix is the force parameterization.

### Learn the metric too — `cargo run --release --example hamiltonian_net`

The deepest form of the cure: let the model learn the arm's **inertia metric** itself and still conserve energy by construction. A mechanical system's Coriolis force is not free — it is the Christoffel term of the mass matrix M(q). So learn M̂(q) = L(q)L(q)ᵀ + εI (symmetric positive-definite by construction, so kinetic energy is never negative) and compute the Coriolis as the exact Christoffel of that **same** M̂. Then the Coriolis does no net work on the model's own energy, as an identity. Verified three ways (Christoffel routine vs analytic Coriolis 1.3e-8; metric-net training gradient-checked; analytic ∂M̂/∂q vs finite-difference 2.5e-8).

| Coriolis on a learned metric M̂ | energy-injection rate \|dÊ/dt\| | rollout energy drift @ 5 s |
|--------------------------------|-------------------------------|----------------------------|
| **Christoffel of M̂ (built in)** | **9.8e-16** (machine zero) | **0.000%** (= true-system reference) |
| free-form field, same data | 3.96 (~137% of the dynamics' own power scale) | 5.2% and climbing |

The energy-injection rate is a **pointwise identity**, independent of fit quality, conditioning, or trajectory — the decisive result. A free-form Coriolis, however accurately fit, does real work and drifts. Conservation is a property you build into the model's structure, not a number you fit toward. (Debug notes worth stating: plain semi-implicit Euler is not symplectic for a q-dependent mass matrix, so we integrate the corroborating rollout with RK4 at small step; and a value-fit potential net has an uncontrolled gradient, so the potential is kept exact here to isolate the metric.)

## Scoring a generative video model (the two rules that decide whether a number means anything)

A video world model outputs pixels, so scoring it means perceive-then-test: recover state from the frames, then check an invariant. Two disciplines are not optional — each caught a confident, wrong result in our own runs on a frontier 4B video model.

**1. Look at the frames.** Trajectory math on video that lacks the thing you are measuring will manufacture a clean answer. Ours did, four times: three on incoherent output (a centroid drifting through smears reported a tidy "fall"), and once on coherent output that simply contained no collision (apex jitter reported restitution e≈0.24, "physical"). Always render a montage and confirm the event exists before believing any number. Gate the measurement on the event too — a bounce only counts as a bounce if the reversal happens *at* the floor, not at the apex.

**2. Never trust n=1.** A single generated sample is an anecdote. Our first run showed a ball falling with textbook acceleration; re-running the identical scene across seeds, it fell in **3 of 9** (95% CI 12–65%) and hovered motionless in the rest. The single-sample verdict ("the model obeys gravity") was true of that sample and wrong as a claim about the model.

Stated correctly, with both halves measured: *when this model generates motion, the motion obeys gravity — acceleration, landing, permanence, and continuity hold in every falling sample; but it generates that motion in about a third of samples, otherwise the object hovers in mid-air.* The failure mode is not wrong physics, it is **absent dynamics** — and an unsupported object that hangs in place is its own violation.

**3. Screen position is not physical position.** This one invalidates more work than the other two. A generated scene is 3D under perspective: an object receding along the floor climbs toward the horizon and shrinks, which in screen coordinates is *identical* to rising into the air. We watched a ball rest on the floor and then apparently ascend — energy from nothing, seemingly a flagrant violation. It was receding: its area fell 80% as it "climbed" (correlation +0.79 between screen height and size), and a monocular depth model over the same frames showed it moving steadily away. No violation, and no gravity reading either — because screen height conflates height with distance. Two cheap fixes we tried both failed, each caught by a sanity check worth copying: a ground-plane fit put the object *below the floor* in 51 of 72 frames (impossible), and a shadow-based estimator locked onto the floor edge instead of the shadow (its "shadow" sat at the same row every frame). A relative-depth model settles *whether* something violated physics; a metric number additionally needs scale — a reference object of known size, or a calibrated camera.

The pattern behind all three: **make the scorer self-falsifying.** Every error above was caught by a check that could fail loudly — impossible positions, a shadow that never moves, an event that never occurs, a rate that shifts when the sample grows. A scorer that cannot embarrass itself will hand you a confident number for video that contains nothing you are measuring.

Also measure your perception floor before scoring: run the tracker on frames whose physics you know, and treat only violations beyond that floor as the model's. Ours was 0.003 m position RMS; passing the frames through the model's own VAE added ~1 mm, and a wrong-gravity scene was still caught through the codec (9.6 vs 3.0 m/s²), which is what makes a verdict about generated video trustworthy at all. Note the scope: those invariants hold on a *known 2D* scene; on a *generated 3D* one, rule 3 applies first.

## Add your model — `cargo run --release --example physbench`

`physbench` is the scoreboard harness. A submission is any type that implements one small trait:

```rust
trait Model { fn name(&self) -> &'static str; fn step(&self, th: f64, w: f64, dt: f64) -> (f64, f64); }
```

Add your model to `entries()` and re-run; the harness scores it on the conservation invariants and prints a ranked standings table next to the references. Current board (frictionless pendulum):

| # | model | energy drift | one-step accuracy | verdict |
|---|-------|--------------|-------------------|---------|
| 1 | velocity-verlet (structure) | 0.07% | 4.6e-5 | PASS |
| 2 | symplectic (structure) | 2.64% | 3.4e-3 | PASS |
| 3 | lossy (wrong invariant) | 30.8% | 4.0e-3 | FAIL |
| 4 | explicit-euler (no structure) | 59.8% | 3.4e-3 | FAIL |

The tell: symplectic and explicit-euler have the **same one-step accuracy** (3.4e-3) and opposite verdicts — accuracy per step does not predict the invariant over a rollout. A learned world model plugs in identically; `wm_on_the_stand.rs` scores a trained black-box (86% drift, FAIL) against a structured net (4.2%, PASS) on this exact system. New probes (contact-impulse cone as a scored force ratio, angular momentum under external torque, restitution-coefficient fidelity) are welcome — each must carry analytic or engine-verified ground truth.

## Scope & honesty

v1 covers analytic systems plus the real SO-101, with ferromotion as the reference. Scoring a frontier **video** world model (a 4B-class model) requires a render → predict → perception pipeline to extract physical state from generated frames, which adds a perception confound (a violation could be the tracker's, not the model's); that is the flagship follow-on, and the trained-model result above already shows a real learned model fails these invariants without that confound. Everything here is pure Rust, free, and reproducible from source — no weights, no GPU, no data downloads required to run the reference.
