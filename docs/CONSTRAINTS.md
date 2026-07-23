# The constraint API — design (v0.31 → v0.32 arc)

Tier-2 item 4 of the mid-2026 roadmap: the Pinocchio-4-class capability — constraints as
first-class *models* over the dynamics, a Delassus-operator abstraction, PGS/ADMM solvers in the
core library, and mimic joints. Design first; staged implementation after.

## What exists, and what the API unifies

Ferromotion already solves constrained problems in four scattered places:

| module | formulation | scope |
|---|---|---|
| `closed_loop` | acceleration-level KKT + Baumgarte | equality loop closures |
| `contact` | velocity-impulse complementarity (QP, clarabel) | unilateral normal contact |
| `robot_contact` / `ipm` | interior-point frictional cone | differentiable contact steps |
| `constraints` | augmented Lagrangian | trajectory-optimization inequalities |

Each is right for its niche and **stays**. What is missing is what Pinocchio 4.0 made table
stakes: one API where a *user* declares constraint **models** — an anchor, a joint limit, dry
friction, a mimic coupling, a contact — and the dynamics layer assembles and solves them together,
with the solver a swappable choice. That unified layer is new; the niches become special cases or
oracles for it.

## Formulation: velocity-impulse time stepping

One step of size `h` from `(q, v)` under torque `τ`:

```text
M (v⁺ − v) = h (τ − bias(q, v)) + Jᵀ λ          (dynamics, impulses λ)
v_c = J v⁺ + b                                   (constraint-space velocity, bias b)
per row-group: a LAW couples λ and v_c            (below)
```

Constraint space: with `v_free = v + h M⁻¹(τ − bias)`,

```text
G λ = −(J v_free + b),   G = J M⁻¹ Jᵀ  (the Delassus operator)
```

solved by sweeping per-group **projections** — the same structure PGS and ADMM share, and the
reason limits, friction, mimic couplings, anchors and contacts all fit one solver.

Why velocity level (not acceleration): unilateral logic (limits, contact) is a complementarity on
*velocities after impulse* — the Anitescu/Stewart–Trinkle form `contact.rs` already uses — and it
is what PGS/ADMM solve robustly. Position drift is handled with Baumgarte terms folded into `b`.
The acceleration-level KKT path in `closed_loop` remains the exact-equality oracle.

## The laws

Each constraint model contributes rows to `J`, entries to `b`, and one law per row-group:

- **Equality** — `v_c = 0` (λ free). Anchors, loop closures, mimic couplings.
- **Unilateral** — `0 ≤ λ ⟂ v_c ≥ 0`. Joint limits (activated near the bound), normal contact.
- **Box** — `λ ∈ [−λmax, λmax]`, `v_c` driven to 0 inside the box, sliding at the bounds.
  Dry joint friction (λmax = h·τ_coulomb).
- **Cone** *(stage 2)* — `‖λ_t‖ ≤ μ λ_n` coupled to a unilateral normal row. Point contact.

## The models (user-facing)

```rust
let mut cs = ConstraintSet::new();
cs.anchor_point(link, local_point, world_target);   // weld a point (3 equality rows)
cs.joint_limits(&robot);                            // from Joint::limits, auto-activated
cs.joint_friction(j, tau_coulomb);                  // dry friction as a box law
cs.mimic(follower, leader, ratio, offset);          // v_f = ratio·v_l (+ Baumgarte on position)
// stage 2: cs.point_contact(link, point, normal, mu)
let (v_next, lambdas) = constrained_step(&robot, &inertia, &q, &v, &tau, h, gravity, &cs, Solver::Pgs);
```

Mimic as a constraint row (not model surgery): `v_f − ratio·v_l = 0` with position-error Baumgarte
— zero changes to `Robot`, works in every algorithm that goes through the constrained step, and
matches how MJCF equality couplings are declared. The reduced-coordinate alternative (eliminating
the DoF) is noted for a later pass; the constraint-row form is what Pinocchio 4 ships first too.

## The Delassus operator

Dense v1: `G = J M⁻¹ Jᵀ` built from the existing `mass_matrix` + Cholesky, with an explicit
regularization `G + εI` (reported, not hidden). Exposed as a type (`Delassus::build / apply /
solve`) because every solver — and downstream users like differentiable layers — consume it.
Sparse / O(n) `lcaba`-style factorizations are stage 3; at course-and-calibration scale dense is
the right first trade, and the API does not change when the backend does.

## Solvers

- **PGS** (stage 1): projected Gauss–Seidel over the row-groups — the robust workhorse; iterate
  until the projected residual stalls below tolerance.
- **ADMM** (stage 2): the Pinocchio-4/SAP-family choice for stiff problems and better
  conditioning, over the same projection interface — so the two cross-validate on every problem.

## Oracles (each stage lands with these)

1. **Equality vs KKT**: anchor-only problems must match `closed_loop`'s acceleration-level KKT
   solution (integrated over the step) — the in-house exact oracle.
2. **Analytic limits**: a single joint pushed into its bound → `v⁺ = 0` at the bound, `λ ≥ 0`;
   pulled away → inactive (`λ = 0`). Approach velocity kills exactly.
3. **Analytic friction**: below breakaway (`h·τ < λmax`) the joint stays stuck (`v⁺ = 0`); above,
   it slides with effective torque reduced by exactly `τ_coulomb`.
4. **Mimic**: follower tracks `ratio·leader + offset` through arbitrary motion; the coupling
   transmits torque (loading the follower loads the leader).
5. **Momentum/energy sanity**: impulses do no work in sticking contacts; equality anchors conserve
   energy up to Baumgarte damping.
6. **PGS ↔ ADMM agreement** (stage 2) on randomized mixed problems.
7. **Four-bar** (stage 3): cut-joint closed loop vs the `closed_loop` module end-to-end.

## Staging

- **Stage 1 (this arc):** laws + `Delassus` + PGS + `ConstraintSet` with anchor / joint-limit /
  dry-friction / mimic + `constrained_step` + oracles 1–5.
- **Stage 2:** friction-cone point contact (reusing `gjk`/`sdf` queries) + ADMM + oracle 6.
- **Stage 3:** four-bar cut-joint workflows, MJCF `<equality>` ingestion, sparse/O(n) Delassus
  backends, `gendyn` instantiation of the step (differentiable constrained dynamics).
