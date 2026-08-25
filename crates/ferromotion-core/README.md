# ferromotion-core

[![crates.io](https://img.shields.io/crates/v/ferromotion-core.svg)](https://crates.io/crates/ferromotion-core)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-core)](https://docs.rs/ferromotion-core)

Kinematics, dynamics, and optimization for physical AI, in pure Rust (native + `wasm32`).

- **Kinematics:** SE(3) forward kinematics, analytic geometric + multi-frame Jacobians, URDF loading
  (from a string — browser-friendly), robust IK (Levenberg–Marquardt + random-restart).
- **Dynamics:** Recursive Newton-Euler inverse dynamics, forward dynamics, joint-space mass matrix,
  gravity compensation, and a symplectic free-rigid-body integrator.
- **Optimization:** composable costs, block-tridiagonal trajectory optimization, sparse factor-graph
  solve, augmented-Lagrangian hard constraints, motion retargeting (position / vector / DexPilot).
- **Contact:** convex contact-implicit dynamics, SOC Coulomb friction, interior-point differentiable
  contact (Dojo-style, smooth through stick↔slip), DCOL differentiable collision, a planar
  rigid-body-with-friction simulator, and **articulated-multibody contact** — differentiable
  frictional floor contact on a full robot chain, with the control gradient `∂q̇⁺/∂τ`.

## A URDF is not an actuator model

`Joint` carries four things a robot description states about its actuators, all of which were being parsed and
discarded: `effort`, `max_velocity`, `armature` and `damping`. Each is `Option<f64>`, unstated by default, and
contributes exactly nothing when absent — so a model that says nothing behaves as it always did.

`armature` is the one that changes whether a problem is solvable. A geared servo's rotor accelerates with the
joint, and its apparent inertia scales as the **square** of the gear ratio, so on a light distal link it does
not merely contribute to the joint-space inertia — it is the larger term. Measured on the SO-101, whose URDF is
the LeRobot original:

| | wrist joint inertia |
|---|---|
| link, from the URDF's `<inertial>` | `3.45e-5` kg·m² |
| reflected rotor, `345²·1e-7` | `1.19e-2` kg·m² |

A factor of **345**. With the link term alone, 10 N·m is 289,728 rad/s² and one 5 ms step adds 1,449 rad/s.
What that costs, measured over a 4×4 grid of PD gains and substep counts (`so101_reach_rl --sweep`):

| | reaches a 1 cm target | best settle | electrical | control rate |
|---|---|---|---|---|
| link inertia only | **1 of 16** configurations | 0.0177 m | 13.8 J | 10 kHz |
| plus reflected rotor | **16 of 16** | 0.0001 m | 4.0 J | 200 Hz |

At *identical* gains and substeps the term is worth 4.2x the settling accuracy and 3.5x the energy; the whole
grid is what shows the other half, that without it only one corner works at all.

The plant is not unsolvable without the term. It is **stiff**, and a working region that collapses to one
corner of the grid. `actuator_plausibility` reports this straight off the model — for the SO-101 it flags
joints 3 and 4 at 12,049 and 289,728 rad/s² implied acceleration, with no simulation run. An earlier draft of
this
section claimed it was unsolvable outright, from a sweep that still contained a hard velocity clamp injecting
energy at every clamp event and never tried a low gain at a high substep count. The `--sweep` mode exists so
the number is reproducible from committed code, which is what would have caught the wrong claim sooner.

The bench also re-measures the inertia sensitivity every run, because the conclusion should rest on the term
being present rather than on one estimate of a rotor inertia.

`to_mjcf` writes a `Robot` back out, which is what makes the term usable end to end: **load a URDF, attach the
servo model, save MJCF, and the rotor inertia survives.** Nothing could express it before, so a corrected plant
existed only in memory. `so101_servo.mjcf` in `examples/` is that file for the SO-101 — the same arm carrying
`N²·J_rotor` and the back-EMF droop — and a test regenerates it from the URDF on every run, so it cannot drift
from the code that produced it. With a realistic 3 N·m limit attached, `actuator_plausibility` flags nothing on
it; on the URDF it was derived from, it still flags two joints, which is what keeps that assertion honest.

The export round-trips through `from_mjcf_full` and the test compares the **dynamics** — mass matrix, gravity
vector, forward kinematics — not the text, because matching bytes is not the property that matters. Floats are
written with Rust's shortest round-trip `Display`; a first version used `{:.17}` with trailing zeros trimmed and
silently rounded `0.012900000000000002` to `0.0129`, breaking the round trip by one ulp.

`identify_actuator` measures the term rather than looking it up, which matters because `J_rotor` is rarely
published for a given servo and a gear ratio **squared** multiplies whatever error an estimate carries. No
optimizer is involved: RNEA is linear in both terms, so subtracting the rigid-body torque leaves a two-column
regression per joint and the joints decouple into independent exact 2-parameter fits.

The practical obstacle is that hardware does not measure `q̈`. It reads a quantized encoder and differentiates
twice, and at 200 Hz that scales quantization by `1/dt²` = 40,000. Measured on the SO-101 with exact torques and
only the kinematics quantized (`so101_reach_rl --identify`):

| rate reconstruction, 12-bit encoder | armature error | damping error |
|---|---|---|
| ideal derivatives | 0.0% | 0.0% |
| central differences | **88–217%**, often negative | <1% |
| `SavGol` 50 ms, order 3 | **1.5–2.7%** | <0.1% |
| `SavGol` 125 ms, order 3 | 11.9–17.7% | 0.1–0.6% |

Damping survives quantization and armature does not, because damping multiplies `q̇` (one differentiation) and
armature multiplies `q̈` (two). So the encoder was never the limit — the differencing was, and `SavGol` fixes it
on the arm exactly as it ships. Note the window length is **not monotone**: an order-3 polynomial cannot follow a
9 rad/s excitation across 125 ms, and over-smoothing biases the second derivative. Match the window to the
excitation bandwidth rather than making it large.

Two things the fit reports rather than hides. Excitation that cannot separate the terms — `q̈` proportional to
`q̇`, which any purely exponential motion satisfies exactly — comes back with `conditioning` at zero and the fit
as `NaN`, instead of a readable-looking number. And a negative rotor inertia is returned as-is with
`physical: false`, since a clamped zero is indistinguishable from a measured one. `conditioning` is about the
trajectory and not the data quality, though: it reads `1.000` on the 217%-wrong rows above, and `residual` is
what catches those.

MuJoCo has carried `armature` for exactly this reason and MJCF states it per joint; URDF has no field for it at
all, so a URDF-only pipeline has no way to express the dominant term. Which format carries what, and which of
it this crate reads:

| | position limits | effort | velocity | armature | damping |
|---|---|---|---|---|---|
| URDF | read | read | read | absent from format | read |
| MJCF | read | on `<actuator>`, not read | on `<actuator>`, not read | read | read |
| SDF | read | read | read | absent from format | read |
| USD | read | in `PhysicsDriveAPI`, not read | not read | absent from format | not read |

URDF's `<dynamics friction=...>` (Coulomb) is parsed by `urdf_rs` and deliberately not read: the RNEA term is
discontinuous at zero velocity and needs a smoothing choice that belongs to the caller. That row of the table
was wrong in a draft of this file, claiming URDF had no damping at all; checking it before publishing is what
turned up the third loader with the same defect.

Both terms are applied inside `inverse_dynamics`, which is what keeps
`M(q)·q̈ + bias(q,q̇) == inverse_dynamics(q,q̇,q̈)` exactly true: `mass_matrix` is built from that function, so
armature lands on its diagonal automatically and `forward_dynamics` inherits both. Putting armature into
`mass_matrix` directly is the obvious shortcut and it breaks the identity wherever `q̇ ≠ 0`.

For a DC servo the dominant part of `damping` is not friction but back-EMF speed droop: torque falls linearly
to zero at no-load speed, so `b = τ_stall / ω_0` follows from two catalogue numbers instead of being fitted.
Modelling it that way also removes any need for a hard velocity clamp, which injects energy at every clamp
event.

```rust
use ferromotion_core::{from_urdf_str, solve_ik, IkOptions};
use nalgebra::{Isometry3, Translation3};

let robot = from_urdf_str(urdf, "base_link", "tool").unwrap();
let target = Isometry3::from_parts(Translation3::new(0.4, 0.1, 0.3), Default::default());
let res = solve_ik(&robot, &target, &vec![0.0; robot.dof()], &IkOptions::default());
```

Part of [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion). Dual-licensed MIT OR Apache-2.0.
