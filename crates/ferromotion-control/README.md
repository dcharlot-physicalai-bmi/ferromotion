# ferromotion-control

[![crates.io](https://img.shields.io/crates/v/ferromotion-control.svg)](https://crates.io/crates/ferromotion-control)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-control)](https://docs.rs/ferromotion-control)

A control corpus for physical AI, in pure Rust (native + `wasm32`), built on
[`ferromotion-core`](https://crates.io/crates/ferromotion-core) dynamics.

- **Feedback / optimal:** PID, LQR, computed-torque, Cartesian impedance, admittance + hybrid force/position.
- **Predictive:** linear MPC, TinyMPC (embedded ADMM), iLQR/DDP, MPPI, CEM, SRBD & centroidal MPC.
- **Whole-body / task-space:** operational-space control (OSC), whole-body QP (WBC), placo-style QP-IK.
- **Safety / robust:** CBF-QP safety filter, sliding-mode control.
- **Legged:** capture point, ZMP preview.
- **Multi-agent:** ALGAMES game-theoretic (generalized-Nash) trajectory optimization.
- **Estimation:** Kalman / EKF / UKF, complementary filter, generalized-momentum observer.
- **Solvers:** ReLUQP (unrolled-ADMM QP), TrajectoryBundles (gradient-free), plus a `clarabel` QP backend.

## The actuator layer

A commanded torque is not a torque. Between the two sit a motor drive, a winding that heats, a gearbox with
clearance, and a battery whose voltage sags under the current it is asked for. These model that gap:

- **`foc`** — field-oriented control for a brushless machine: Clarke/Park, the `dq` current dynamics, MTPA,
  field weakening, base speed, a tuned PI current loop, and space-vector modulation.
- **`motor_thermal`** — two-node winding/housing network. Copper's `+0.393%/K` makes `I²R` a positive feedback
  loop, so the equilibrium temperature rise returns `None` on thermal runaway rather than a fixed point that
  does not exist.
- **`friction`** — Stribeck curve and the LuGre bristle model, the latter's steady state reproducing the former.
- **`backlash`** — lost motion as a deadband on relative position. A reversal costs the full width.
- **`battery`** — Thévenin pack. Past `OCV/2R`, drawing more current delivers *less* power.
- **`rotordynamics`** — the gyroscopic reaction a spinning rotor imposes on its housing, and the Jeffcott
  whirl model with critical speeds. The rotor's spin is factored out of every multibody model because it is
  orders of magnitude faster than any body rate, and that factoring-out silently discards both effects.
- **`fatigue`** — rainflow counting, S-N curves, mean-stress corrections, Miner accumulation.
- **`actuator`** — series-elastic joints, a brushed DC motor with cogging, transport delay.

Beyond the rotating motor, four actuator families with nothing in common but the job:

- **`piezo`** — microns of stroke, kilonewtons of force, and a capacitive load. The useful work sits on the
  straight load line between free displacement and blocked force, maximised exactly at a matched load.
- **`sma`** — nitinol, at roughly **13 MJ/m³** against an electric motor's 0.1. Path-dependent: martensite
  fraction depends on where you have been, not just on temperature and stress.
- **`mckibben`** — braided pneumatic muscle. Everything follows from the braid angle and one application of
  virtual work, `F = P(−dV/dL)`.
- **`muscle`** — Hill's biological force-velocity model, which shares only the name with `mckibben`.

Three results from those, each measured rather than quoted:

**A piezo's coupling `k²` is measurable two entirely independent ways** — from the constitutive coefficients
`d²/(sᴱεᵀ)`, and from the resonance/antiresonance gap `1 − (f_r/f_a)²`. Those share no arithmetic, so their
agreement is real evidence. `k² > 1` is rejected at construction: it describes a transducer returning more work
than it receives, and it is what mixing coefficients measured under different boundary conditions produces.

**An SMA wire's bandwidth is set by cooling, and cooling is passive.** `cycle_time` returns heating and cooling
separately because they are not symmetric: ten times the drive power heats ten times faster and cools not at
all faster. Forced convection is the only lever. Quoting a cycle rate from the heating figure alone is the
standard way these get oversold.

**A McKibben muscle wound past 54.7356° extends instead of contracting.** The force term is
`3cos²θ − 1`, which vanishes at `arccos(1/√3)`, so free contraction is `1 − 1/(√3 cos θ₀)` — negative for a
braid past that angle. Same hardware, opposite sign, decided by the winding. `free_contraction` returns the
signed value unclamped, because an extensor is a real device rather than an error.

Two results from that layer worth stating, because both are measured rather than assumed:

**Space-vector modulation reaches a peak phase voltage of `V_dc/√3`; sine-triangle modulation reaches
`V_dc/2`.** That is 15.47% more voltage from the same hardware, and therefore 15.47% more speed before field
weakening is needed, out of a change in arithmetic. The test bisects on where each modulator's duties first
leave `[0, 1]` rather than taking the formula's word for it.

**A disc-shaped rotor has no conical critical speed at any speed whatever.** Gyroscopic stiffening raises the
forward whirl branch faster than the synchronous line rises, so for a polar-to-diametral inertia ratio
`γ = I_p/I_d ≥ 1` the two never cross and `Ω_cr = ω_n/√(1−γ)` has no real solution. A thin disc is `γ = 2`
exactly, so the common case is the exempt one, while a slender armature (`γ = 6r²/L² ≪ 1`) has a conical
critical speed and must be designed around it. Relatedly: whirl amplitude peaks near `r = 1` at `e/(2ζ)` and
then falls back to the eccentricity `e` **exactly** as `r → ∞`, because above resonance the rotor turns about
its mass centre. The design question is not "stay below resonance" but "pick a side".

**The stability bound that binds a current loop is the regulator's, not the machine's.** `Pmsm::max_stable_dt`
is the open-loop `2L/R`; a PI regulator at bandwidth `ω_bw` imposes roughly `2/ω_bw`, which for a 500 Hz loop
on a 5 ms machine is 15.7 times tighter. Stepping at a tenth of the plant bound and closing that loop produces
`NaN`. Measured critical `dt·ω_bw` across four machines and three bandwidths: 1.22 to 1.94, always below 2.

Every controller is verified in closed loop (against `ferromotion-core::forward_dynamics`).

Part of [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion). Dual-licensed MIT OR Apache-2.0.
