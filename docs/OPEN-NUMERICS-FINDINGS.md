# Open numerics findings — ferromotion, from the 2026-08-14 adversarial audit

Six lenses over ~95k lines of Rust produced 18 findings with concrete triggers. **Eight were fixed on
2026-08-14**, each with a test that fails against the old code (`log_so3` across four copies, `Rod::relax`,
`contact_ipm`'s two readouts, `MacFluid`'s three monitors, `Msnn::fit`, `ChunkClock::deliver`, `plan_vv_t`).

The rest are recorded here because they came with reproducible triggers and should not be lost. **They were
NOT adversarially verified** — a verify pass was budgeted for the six highest-confidence findings only, and in
that pass 6 of 6 were confirmed and 0 refuted, so the prior on these is high but not established. Treat each
as a hypothesis with a named experiment, and reproduce the trigger EXACTLY before concluding: two of the eight
fixes initially failed to reproduce because a parameter was misread.

**ALL TEN RESOLVED as of 2026-08-15** — seven fixed, three documented as genuine limitations where a fix was
impossible or could not be justified. Every one was a wrong number or a false claim reaching a caller, and not one
was caught by the existing suite. Details for each are below and in the Closed section. Two were verified and closed on 2026-08-14 and are recorded at the bottom of this file:
`Admittance::step`'s "unconditional stability" claim and `Vof`'s unconditional boundedness claim. Both were
confirmed by direct measurement, both are now documented with their real bounds, and both crates gained a way for
a caller to check the bound rather than discover it by divergence.

## 1. `force_closure_q1` — crates/ferromotion-core/src/grasp.rs:55

**CLOSED 2026-08-14 — confirmed via MLS Prop. 5.2/5.3 — rank gate added.**

*reported confidence 0.9*

**Claim.** The planar Ferrari-Canny Q1 has no wrench-rank gate, so for a rank-deficient grasp (true Q1 = 0, i.e. not force closure) it returns a strictly positive "robustness margin" that decays only as ~1/n_dirs and never reaches 0 — the doc's "`> 0` <=> force closure ... `<= 0` <=> not force closure" is false. Its own 6D sibling documents this exact failure and gates against it; the planar function was left un-gated.

**Trigger.** Any grasp whose primitive wrenches are rank-deficient in R^3. Cleanest: contacts on the unit disk with inward radial normals and mu = 0.0, so every line of action passes through the reference point and every torque is exactly 0, confining the wrench hull to the plane tau = 0. GraspLab's own geometry: 2 antipodal contacts, or 3 at 120 deg, or 4 at 90 deg, mu = 0.

**Reported wrong output.** 3 contacts at 120 deg, mu=0: force_closure_q1 returns +0.02040816 at n_dirs=1200 (+0.02499 at 800, the value the module's own tests use) where the true Q1 is exactly 0. 2 antipodal frictionless contacts (wrench rank 1): +0.00027104. 4 at 90 deg: +0.04039995. ESCAPES to ferromotion-wasm/src/grasp_lab.rs:93 `is_force_closure()` = `self.q1() > 1e-6`, which therefore returns **true** for two frictionless fingers pinching a disk — a grasp that can resist no torque and no shear at all — and grasp_lab.rs:9 tells the reader "`Q1 > 0` <=> force closure; larger is firmer". weakest_dir() (grasp_lab.rs:98) likewise returns a direction that is not the weakest.

## 2. `force_closure_q1_spatial (with primitive_wrenches_spatial, line 120)` — crates/ferromotion-core/src/grasp_spatial.rs:188

**CLOSED 2026-08-14 — resolved as a documented limitation: MLS App. A.3.2 proves no frame-invariant metric on se(3) exists.**

*reported confidence 0.9*

**Claim.** The 6D wrench is assembled as [f; p x f] with no characteristic length, so a `unit direction` on S^5 mixes newtons and newton-metres; the resulting anisotropy makes the 20000-direction sampled minimum exceed a rigorous upper bound on the true Q1 by 1.5x-3.8x at any object scale other than |p| ~ 1, and it makes the module's headline planar-vs-spatial ratio a unit-of-length artifact.

**Trigger.** The module's own fixture `three_coplanar(mu)` with contact positions scaled by R (the identical physical grasp expressed in a different length unit), facets = 24, n_dirs = 20_000, mu = 0.6. R = 0.1 is exactly the scale of hand_object.rs's `SphereObject::new(_, 0.10, 0.2)`; R = 10 is that same 10 cm object with positions in centimetres.

**Reported wrong output.** R = 10: returns 0.792015, but d = (0,0,1,0,0,0) (a pure +z force direction) gives max_i w_i.d = mu/sqrt(1+mu^2) = 0.514496, so true Q1 <= 0.514496 by definition — the returned value is 1.54x a provable ceiling (a 2e6-direction estimate gives 0.4936). R = 100: returns 1.976109 against the same 0.514496 ceiling, 3.84x. R = 0.1: returns 0.078750 while d = (0,0,0,1,0,0) gives 0.044557, so 1.77x a ceiling (dense estimate 0.03714, i.e. 2.1x). Separately, the doc's claim that the planar metric `overstates` the spatial one by `a consistent factor of about 1.5` (and the test `the_planar_metric_overstates_the_spatial_quality`, which asserts pl > sp and 1.3 < ratio < 1.9) inverts off the unit disk: ratio = 0.764 at R = 0.1 and 0.393 at R = 100, i.e. the planar metric UNDERSTATES the spatial one on a 10 cm object.

## 3. `module doc `Which way each approximation errs` / the facet assertion in the_two_approximations_err_in_known_directions (line 397)` — crates/ferromotion-core/src/grasp_spatial.rs:31

**CLOSED 2026-08-15 — the claim was FALSE and is now conditional on nested refinement; measured 4→5 facets at −5.0% (8192 dirs) and −13.1% (20000).**

*reported confidence 0.85*

**Claim.** The documented monotonicity in cone facets is false: more facets does NOT enlarge the inscribed polyhedral cone, so Q1 genuinely decreases at several facet increments. The claim survives only because the test sweeps [4, 8, 16, 32, 64] — nested doublings, the one family for which containment does hold.

**Trigger.** The module's own fixture `three_coplanar(0.6)` with facets stepped 4 -> 5 (also 8 -> 9, 10 -> 11, 14 -> 16, 20 -> 24, 24 -> 25 is fine but 32 -> 33 drops), at the module's own direction counts.

**Reported wrong output.** facets 4 -> 5 at 8192 directions: Q1 0.330879794 -> 0.314446042, a 5.0% DECREASE; at 20000 directions 0.330879794 -> 0.287560554, a 13.1% decrease; with 2e6 random directions (a much tighter estimate of the exact polyhedral value) 0.285307653 -> 0.276632626, a 3.0% decrease — so it is the wrench polytope shrinking, not a sampling artifact. The doc says `More cone facets enlarge the inscribed polyhedral cone, so Q1 rises toward the smooth-cone value from below` and the test asserts `q >= prev - 1e-9` with the message `more facets cannot shrink the inscribed cone`; a caller who refines facets 32 -> 33 to tighten the estimate gets a smaller number (0.341815 -> 0.340984 at 20000 dirs).

## 4. `AugmentedVbd::step` — crates/ferromotion-cloth/src/vbd.rs:311

**CLOSED 2026-08-15 — confirmed by measurement. See the Closed section at the end of this file.**

*reported confidence 0.85*

**Claim.** The multiplier update `lambda <- lambda + penalty*C` is the augmented Lagrangian for the HARD constraint C = 0, and `Spring::stiffness` appears nowhere in the step except in the clamp bound — so `AugmentedVbd` converges to an inextensible-chain solution, not to the elastic implicit-Euler solution the module says VBD/AVBD both target. A deliberately soft spring is silently made nearly rigid.

**Trigger.** Uniform hanging chain, `hanging_chain(10, 0.1, 0.05, 1e2)`, `VbdSolver::new(1.0/60.0, 4)` with damping 0.05, `AugmentedVbd::new(solver, 10, 1e4)`, run to rest (settled by 2000 steps, KE ~ 1e-29).

**Reported wrong output.** AVBD settles at total length 1.0226 m; the exact static elastic equilibrium sum_j (0.1 + (10-j)*m*g/k) is 1.2698 m, and plain `VbdSolver` at 64 sweeps reproduces 1.2698 m to 4 decimals. AVBD is 19.5% short, with per-link extension ~10x too small (link 0: +0.00496 m vs correct +0.04905 m). 8 of 10 multipliers sit pegged at the clamp +/-stiffness*rest = 10.000 N while the true tension in that link is at most 4.905 N — the clamp, not the physics, sets the answer, and it lets the multiplier pull 2x the force the spring could. This also falsifies the in-file justification at line 248 ("the mixed fixture ... is fine ... because the soft links leave the chain compliant enough"): under this update the soft links are not compliant. The existing test `avbd_handles_a_stiffness_ratio_that_slows_plain_vbd` measures only the STIFF links (`filter(|(i,_)| i % 2 == 0)`), so the regime where AVBD is wrong is the half of the fixture never asserted on.

## 5. `P2Mesh::build (consumed by solve_oseen / solve_stokes / solve_navier_stokes)` — crates/ferromotion-fluid/src/ufvm_stokes.rs:46

**CLOSED 2026-08-15 — fixed — a boundary edge is one with exactly ONE incident triangle, not one whose endpoints are both boundary vertices.**

*reported confidence 0.85*

**Claim.** A P2 mid-edge velocity node is flagged boundary iff BOTH endpoint vertices are boundary vertices, which is not the same test as the edge lying on the domain boundary; on every TriMesh::unit_square mesh exactly two strictly interior velocity nodes are flagged and then hard Dirichlet-pinned to u_bc evaluated at an interior point.

**Trigger.** Any n. The triangulation's corner-cutting diagonals (0, 1-h)-(h, 1) and (1-h, 0)-(1, h) have both endpoints on the boundary but lie in the interior; their midpoints (h/2, 1-h/2) and (1-h/2, h/2) get node_boundary = true. For unit_square(17, ·), h = 1/16, that is the nodes at (0.03125, 0.96875) and (0.96875, 0.03125). Drive it with any BC defined per-wall — e.g. a Stokes driven cavity, u_bc = |x,y| if y > 1-1e-9 {(1.0,0.0)} else {(0.0,0.0)}.

**Reported wrong output.** Those two interior velocity dofs are set to fixed_val = u_bc(interior point) — (0,0) for the cavity BC — and dropped from the solve instead of being computed, so the returned velocity has an O(1) error at the two nodes nearest the top-left and bottom-right corners, a spurious element divergence there, and a locally wrong pressure. It is silent in both MMS gates because their u_bc IS the exact solution everywhere: the two nodes are handed the exact answer, so the reported "order > 2.3" and "worst element divergence < 5e-2" are measured with 2 interior dofs pre-solved.

## 6. `Vof::step / module doc "Boundedness"` — crates/ferromotion-fluid/src/vof.rs:9

**CLOSED 2026-08-14 — confirmed and fixed. See the Closed section at the end of this file for the measured numbers.**

*reported confidence 0.85*

**Claim.** The doc asserts boundedness unconditionally — "The minmod slope limiter is TVD, so C never overshoots [0,1] — no spurious negative or super-unity volume fractions" — but the scheme is MUSCL-minmod with FORWARD EULER, whose TVD property holds only under a CFL bound that is nowhere documented and nowhere enforced; step(dt, vel) accepts any dt from the caller.

**Trigger.** The module's own solid-body-rotation benchmark with the dt scaling raised from 0.4 to 0.8: n = 48, omega = 2.0, dt = 0.8 * h / (omega * 0.71) (max cell CFL |u|max*dt/h ~ 0.80). The shipped tests use factor 0.4, i.e. under 2x of margin from a failure the doc calls impossible.

**Reported wrong output.** C reaches -1.21e-5 within 120 steps — a negative volume fraction, the exact thing the doc rules out. At factor 1.0 the field reaches +/-7.2e6 in 120 steps (total blow-up); at 1.2, +/-2.3e21. In 1D pure advection the same flux/limiter/time-stepping gives min -0.548 / max 1.551 at CFL 0.70 and min -3145 / max 3250 at CFL 0.90, while staying exactly in [0,1] up to ~0.6.

## 7. `Admittance::step` — crates/ferromotion-control/src/admittance.rs:35

**CLOSED 2026-08-14 — confirmed and fixed. See the Closed section at the end of this file for the measured numbers.**

*reported confidence 0.85*

**Claim.** The doc comment claims the integrator gives "unconditional stability", but the scheme is symplectic (semi-implicit) Euler with the damping term evaluated at the OLD velocity, which is only conditionally stable: the exact bound is dt²·k/m + 2·dt·d/m ≤ 4. Past it the compliant reference diverges geometrically instead of settling on the spring law.

**Trigger.** Admittance::new(1.0, 8.0, 50.0) — the crate's own test gains from admittance_relaxes_to_spring_law — with f_ext = 5.0, x_ref = 0.0, and dt = 0.17 s or larger (the bound gives dt < 0.1650 s for these gains; a 6 Hz outer compliance loop is inside the failing range).

**Reported wrong output.** After 20 s the commanded position is 5.11e4 at dt = 0.170, −9.01e26 at dt = 0.200, and −3.25e41 at dt = 0.250, against the documented equilibrium x_ref + F/k = 0.1000 (which it does reach, to 4 decimals, at dt ≤ 0.160). No error, no NaN, no diagnostic — just a finite exploding command.

## 8. `ChunkClock::deliver` — crates/ferromotion-policy/src/clock.rs:165

**CLOSED 2026-08-15 — fixed — the latency prediction is now capped by what the fast loop actually consumed.**

*reported confidence 0.82*

**Claim.** `frozen` is computed purely from `inference_time` and is never reconciled with `consumed_in_chunk`; when it exceeds the actions actually consumed, `saturating_sub` silently clamps the realignment offset to 0, which both drops unexecuted actions from the stream and restores the unshifted freeze target the comment above it says it fixed.

**Trigger.** `ChunkClock::new(20, 1, 0.01, 0, 6)`, field `|a| a.map(|x| 0.5 + 0.3*x)`, `a0[i] = 0.05*i`, steps = 4 Heun: deliver(inference = 0.0), tick() twice (2 actions executed), then deliver(inference = 0.05) — a measured 50 ms inference is 5 ticks at 100 Hz, so frozen = 5 while consumed_in_chunk = 2. Any caller whose fast loop is behind its nominal `control_period`, or that re-plans sooner than one chunk, lands here.

**Reported wrong output.** offset = 2.saturating_sub(5) = 0, so `queue` restarts at chunk index 5 while playback had reached slot 2: three trajectory slots are skipped and the commanded stream goes 0.5825, 0.6500, 0.9199 where the next action should have been 0.7174 — a boundary jump of 0.2699 against a 0.0675 median interior step, i.e. 4.0x, the same class of discontinuity the comment at clock.rs:157-163 reports having removed (16.6x). `state()` still returns Nominal and `last_frozen` reports 5; ClockHealth has no variant for the disagreement. The module doc asserts "frozen matches what executed … exactly" and "it cannot silently disagree with what actually executed".

## 9. `FemSim::forces / FemSim::psi` — crates/ferromotion-fem/src/lib.rs:191

**CLOSED 2026-08-15 — fixed — the barrier has a real gradient via ∂J/∂F = cof(F); it does NOT fully rescue a degenerate element, and that limit is documented and tested.**

*reported confidence 0.8*

**Claim.** `psi` returns +INFINITY for an inverted element and calls it an "infinite energy barrier", but `forces` skips that element on the identical `j <= 0.0` test — so the barrier has exactly zero gradient. An inverted tet contributes no force in either direction and can never recover, while `energy()` is poisoned to +inf forever; the GPU path mirrors the skip and never computes energy at all, so there the inversion is entirely invisible. The header also attributes Psi = 0.5*mu*(I_C-3) - mu*lnJ + 0.5*lambda*(lnJ)^2 to Smith et al. 2018 "stable Neo-Hookean", whose defining contribution is replacing log J precisely so the energy stays finite and differentiable at J <= 0; this is the classical log-barrier Neo-Hookean, which has the property the citation claims it fixed.

**Trigger.** Static, minimal: `single_tet(8.0, 4.0)` with `x[3] = Vector3::new(0.0, 0.0, -0.2)` (J = -0.2). Dynamic and reachable with plausible parameters: `FemSim::box_grid(2,2,2, 0.3, mass 0.4, mu 4.0e2, lambda 2.0e2, dt 3e-4)`, `damping_rate = 10.03` (per-second; was `damping = 0.003` per step, equivalent at this dt), `floor = Some(0.0)`, `k_contact = 1.0e5`, lifted 0.2 m and given v_z = -30 m/s.

**Reported wrong output.** Static case: `energy()` = inf and `forces()` = [0,0,0,0] exactly, where the true gradient of the implemented Psi is unbounded. Dynamic case (measured): 20 of 40 tets have J < 0 at step 53; the body then keeps running with all-finite positions, bounces, and "settles" back onto the floor by step 3999 with KE 1.75 J and a plausible-looking shape — while still carrying 20 permanently everted, force-free tets and `energy() == inf` at every one of steps 53, 60, 100, 200, 500, 1000, 2000, 3999. Half the mesh's stiffness silently vanishes and the only report is an inf from a method a caller may never call (and which the GPU path does not expose).

## 10. `ComplementaryFilter::update_rp (and ::update, line 59)` — crates/ferromotion-control/src/complementary_filter.rs:73

**CLOSED 2026-08-15 — confirmed by measurement. See the Closed section at the end of this file.**

*reported confidence 0.8*

**Claim.** The gyro-integrated angle and the accelerometer angle are blended by plain linear interpolation with no angle wrapping, so when the true roll crosses ±π the accel reference jumps by 2π while the integrated prediction does not, and the fused estimate is dragged toward the midpoint of two angles that are physically adjacent but numerically 2π apart.

**Trigger.** A body rolling continuously about x at ωx = 1.0 rad/s, alpha = 0.95, dt = 0.01, accel supplied as the true gravity direction (so roll_acc = ay.atan2(az) ∈ (−π, π] — exactly what the crate's own test at line ~205 computes). At t = 3.1416 s the true roll passes π.

**Reported wrong output.** At t = 3.20 s the true roll is −3.0832 rad (wrapped) but the filter returns +1.5355 rad. Worst error after the crossing is 3.0642 rad = 175.6° at t = 3.28 s — the filter reports the body as essentially level while it is inverted — and it takes ≈ 2.9 s of continued rolling to re-converge. Every crossing of ±π repeats it. pitch is unaffected (atan2(−ax, √(ay²+az²)) is confined to ±π/2), so only the roll channel and the 1-axis `update` are hit.



---

# Closed

## `Admittance::step` — crates/ferromotion-control/src/admittance.rs — CONFIRMED, fixed 2026-08-14

The doc claimed "unconditional stability". The scheme is symplectic in the spring but evaluates damping at the
OLD velocity, so the exact bound is `dt²·k/m + 2·dt·d/m ≤ 4`. Measured on the module's own gains
(`m=1, d=8, k=50`, limit `dt ≤ 0.164962`), settling from rest under a constant force against the documented
equilibrium `x_ref + F/k = 0.1`:

| dt | result | bound LHS |
|---|---|---|
| 0.160 | 0.1000 settled | 3.84 |
| 0.164 | 0.1028 | 3.969 |
| 0.165 | 0.1578 drifting | 4.001 |
| 0.170 | **5.11e4** | 4.165 |
| 0.200 | **−9.01e26** | 5.20 |

The analytic bound is sharp to three decimals. Doc corrected; added `Admittance::stability_limit()`. A 6 Hz outer
compliance loop sits at `dt ≈ 0.167`, just past the limit for those gains, so this is not an exotic regime.

## `Vof` boundedness — crates/ferromotion-fluid/src/vof.rs — CONFIRMED, fixed 2026-08-14

The doc claimed `C` "never overshoots `[0,1]`" unconditionally. It is MUSCL-minmod with FORWARD EULER, so TVD is
conditional, and `step` accepts any `dt`. Measured over 200 steps of the module's own rotation benchmark:

| `factor` | per-component CFL | worst `C` |
|---|---|---|
| 0.4 | 0.28 | exactly `[0,1]` |
| 0.6 | 0.41 | exactly `[0,1]` |
| 0.8 | 0.55 | **−1.34e-2** |
| 0.9 | 0.62 | −3.1e11 |

The onset straddles the classical 0.5 MUSCL/forward-Euler bound. Doc corrected with the measured table; added
`Vof::max_cfl()`.

**Two things this one taught, both worth carrying to the remaining items.**

1. **`factor` is NOT the CFL.** The shipped `dt = factor·h/(ω·0.71)` divides by the speed *magnitude*, while the
   Courant number governing a dimension-summed flux update is per-component and smaller by ≈√2. Asserting
   `factor == CFL` is how the first version of the new test failed. Measured, `CFL ≈ 0.69·factor`.
2. **`Vof::bounds` — the boundedness monitor itself — swallowed `NaN`**, via the same `f64::min`/`f64::max`
   family fixed elsewhere. It now propagates. The sharp case is a field with *some* `NaN` cells among finite
   ones, which reported a clean, plausible sub-interval of `[0,1]`.
3. **Conservation outlives boundedness.** At `factor` 0.8 the volume drift was still exactly 0 while `C` was
   already negative, so **a conservation check cannot stand in for a boundedness check.**

## `ComplementaryFilter::update_rp` — CONFIRMED, fixed 2026-08-15

The accel reference `atan2(ay, az)` is confined to `(−π, π]`; the gyro-integrated prediction accumulates unbounded.
A plain linear blend across ±π is therefore pulled toward the midpoint of two angles that are physically adjacent
but numerically `2π` apart. Reproduced exactly (`ωx = 1`, `α = 0.95`, `dt = 0.01`, noiseless accel): at `t = 3.20 s`
true roll `−3.0832`, filter returned **`+1.5355`**; worst error **3.064152263457804 rad = 175.563°** — an attitude
filter reporting a body as level while inverted — re-converging only after ≈2.9 s, and repeating every crossing.

Fixed by blending the **wrapped innovation**,
`angle ← wrap(pred + (1−α)·wrap(acc − pred))`, identical to the classic form whenever the two agree within ±π.
Worst error becomes `0.00°`. Mutation-verified in Rust.

**Two notes.** (1) `update` now returns a **wrapped** angle — a behaviour change; a caller wanting a continuous
angle must track its own turn count. (2) An existing test fed `acc = 0.5·k` up to **9.5 rad** as an "accel angle",
which `atan2` cannot return; once the blend is circular, 9.5 and 9.5 − 2π are the same attitude, so the assertion
had two right answers. Inputs now stay inside the contract; the assertion is unchanged.

## `AugmentedVbd::step` — CONFIRMED, and DELIBERATELY NOT FIXED 2026-08-15

Confirmed exactly. On `hanging_chain(10, 0.1, 0.05, 1e2)` run to rest, against the analytic static equilibrium
`nL + (mg/k)·n(n+1)/2 = 1.269775 m`: `VbdSolver` at 64 sweeps gives **1.269775 m (0.0000%)**, `AugmentedVbd` gives
**1.022610 m (−19.47%)**. Link 0 extends `0.004959` where the spring law gives `0.049050` (**10× too little**), and
the clamp bound `k·rest = 10.0 N` is **2× the true maximum tension** `n·m·g = 4.905 N`, so the clamp is not what
holds the answer together. `Spring::stiffness` enters the step nowhere but that clamp.

**A compliance term does NOT repair it, measured.** Dividing the multiplier update by `1 + penalty/k` does drive
`λ → k·C`, the elastic force — the algebra is right — but the applied force is `λ + penalty·C`, so the total becomes
`(k + penalty)·C`, **101× too stiff** at `penalty = 1e4, k = 1e2`. It moved the chain only `1.0226 → 1.0311` while
making the mixed fixture's stiff-link violation **3× worse** (`1.11e-4 → 3.37e-4 m`). Reverted.

**The conclusion is structural**: for a compliant spring the correct total force is just `k·C`, so the augmented
Lagrangian contributes nothing. It is a **hard-constraint** tool. `AugmentedVbd` solves the *inextensible* problem
and `VbdSolver` the elastic one — different fixed points, which the module doc previously denied. Documented with
the measurements. Making AVBD handle finite stiffness needs the adaptive-penalty formulation from the AVBD
literature; **that is the open work, not a patch to this update.**

Why it hid: `avbd_handles_a_stiffness_ratio_that_slows_plain_vbd` filters to `i % 2 == 0` — the **stiff links only**
— so the regime where the solver is wrong was the half of its own fixture never asserted on.
`the_soft_links_are_driven_nearly_rigid` now pins it, and fails deliberately if AVBD is later made elastic without
the doc being updated.


---

# Closed 2026-08-15 — the final four

## `P2Mesh::build` boundary flag — FIXED

`node_boundary` was `base.boundary[a] && base.boundary[b]`: a mid-edge node counted as boundary iff both
*endpoint vertices* were. That is strictly weaker than the edge lying on the boundary — a corner-cutting diagonal
satisfies it while lying in the interior — so on every `TriMesh::unit_square` exactly **two** strictly interior
velocity dofs were hard Dirichlet-pinned to `u_bc` evaluated at an interior point, i.e. *set* rather than solved.

Replaced with the topological definition: an interior edge is shared by two triangles, a boundary edge by one.
No geometry, no tolerance, valid for any triangulation. The test recounts incidence independently of `build` and
asserts the old rule over-counted by **exactly 2**; mutating it back names the two nodes at `(0.125, 0.875)` and
`(0.875, 0.125)`, exactly as predicted.

## `FemSim` inversion barrier — FIXED, with a documented limit

`psi` returned `+INFINITY` for `J ≤ 0` and called it an "infinite energy barrier" while `forces` **skipped** the
element on the identical test — a barrier with exactly zero gradient. An inverted tet exerted no force in either
direction and could never recover, while `energy()` was poisoned to `+inf` and the simulation carried on with
finite positions.

Now a finite `½k(J − J_recover)²` whose gradient goes through `∂J/∂F = cof(F)`, written from column
cross-products so it is defined even at `det F = 0` — the configuration a flattening tet passes through.

**It does not fully rescue an inverted element, and the test says so.** `cof(F)` vanishes as the element
flattens, so the restoring force dies exactly where it is most needed; measured with gravity off, `J` rises
monotonically `−0.2 → −0.0346` over 200k steps and does not cross zero. Escaping a degenerate element needs an
energy whose gradient does not route through `∂J/∂F` — the purpose of the *stable* Neo-Hookean formulations. A
separate test asserts healthy elements are **bit-identical** under the new branch.

## `ChunkClock::deliver` realignment offset — FIXED

`frozen_for` is a *prediction* from the measured inference time; `consumed_in_chunk` is what the fast loop
actually ticked. When the prediction was larger, `offset = consumed_in_chunk.saturating_sub(frozen)` clamped to
`0`, which restored the **unshifted** freeze target the shift exists to fix *and* restarted `queue` past where
playback had reached, dropping the slots between. Measured with two ticks executed and a 50 ms inference
(`frozen = 5`, `consumed = 2`): three slots skipped, boundary jump 0.2699 against a 0.0675 median interior step.

Capped by reality: `frozen_for(t).min(consumed_in_chunk)`. You can only freeze against actions that ran.
Reachable whenever the fast loop ticks slower than its nominal period — the normal condition under load.

## Facet monotonicity — the CLAIM was false, now conditional

The module doc stated flatly that more cone facets enlarge the inscribed polyhedral cone so `Q1` rises **from
below**. That holds only under **nested** refinement `k → 2k`, where every coarse generator is retained. For any
other increment the generators sit at `2πj/k` and *move*, so the finer polytope is not a superset. Measured on
`three_coplanar(0.6)`, `4 → 5` facets: **0.330879794 → 0.314446042** at 8192 directions (**−5.0%**) and
**→ 0.287560554** at 20000 (**−13.1%**). It is the wrench polytope rather than the sampler — at 2e6 random
directions the drop persists, 0.285308 → 0.276633.

The old test survived only because it swept `[4, 8, 16, 32, 64]` — nested doublings, the one family where
containment holds. `facet_monotonicity_needs_nesting` now checks both halves and asserts the decrease as **real**,
so the false claim cannot be restored by widening a tolerance.

## The pattern across all ten

Three of the ten hid behind a test that measured the **wrong half of its own fixture**: AVBD's test filters to
`i % 2 == 0` (the stiff links, where it is right); the facet sweep used only nested doublings; the complementary
filter's degenerate-blend test fed `9.5 rad` as an `atan2` output. None was a missing test. Each was a test whose
scope excluded the failing regime — which is why a green suite of 1,290 said nothing about any of them.
