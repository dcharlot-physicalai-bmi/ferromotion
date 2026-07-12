# ferromotion-control — the physical-AI control corpus, ported to Rust

Ingest **every control scheme, method, and technique used in physical AI** into `ferromotion-control` —
pure Rust, native + universal WASM. The same 3-tier honesty as the rest of the program: almost all
control is **algorithmic/weights-free → PORT-shaped** (runs everywhere, including the browser);
**learned policies** run as exported weights (interop via `burn`/`candle`), not rewritten.

Most serious controllers need dynamics — and ferromotion-core already has it: mass matrix `M(q)`, RNEA bias
`C·q̇+G`, Jacobians, forward dynamics (simulation), a QP backend (`clarabel`), and a block-tridiagonal
trajectory solver. That's the substrate the whole corpus builds on.

## Status
| Family | Method | Status | ferromotion leverage |
|---|---|---|---|
| Feedback | **PID** (n-dim) | ✅ shipped | — |
| Model-based manip. | **Computed-torque / inverse-dynamics** | ✅ shipped | `M(q)`, RNEA bias |
| Model-based manip. | **Cartesian impedance** | ✅ shipped | Jacobian, gravity comp |
| Simulation | **forward dynamics** (for closed-loop test) | ✅ shipped (ferromotion-core) | `M⁻¹`, RNEA |
| Feedback/optimal | **LQR** (discrete DARE) | ✅ shipped | linearize about a setpoint |
| Predictive | **Linear MPC** (condensed QP over a horizon) | ✅ shipped | `clarabel` QP |
| Model-based manip. | **Operational-space control (OSC)** | ✅ shipped | `M`, Jacobian, `Λ = (J M⁻¹ Jᵀ)⁻¹`, DC null-space |
| Model-based manip. | **Admittance + hybrid force/position** | ✅ shipped | Cartesian spring-damper + selection matrix |
| Whole-body | **WBC** — acceleration-QP task stack + accel limits → inverse dynamics | ✅ shipped | diffik idea + `clarabel` + RNEA |
| Nonlinear optimal | **iLQR / DDP** | ✅ shipped | finite-diff linearization of `forward_dynamics` + backward Riccati |
| Sampling | **MPPI** (sampling-based MPC, deterministic LCG) | ✅ shipped | `forward_dynamics` rollouts — batchable on the GPU fleet path |
| Safety | **CBF-QP** (rel-deg-1 + HOCBF) safety filter | ✅ shipped | min‖u−u_nom‖² s.t. barrier, `clarabel` |
| Legged / balance | **capture point + ZMP preview** (cart-table) | ✅ shipped | LIPM; capture-point also live on the site |
| Robust | **sliding-mode** (boundary layer) | ✅ shipped | `M`·(equiv) + reaching law; H∞ later |
| Estimation | **Kalman + EKF + UKF** | ✅ shipped | generic nalgebra; numerical Jacobians |
| Learning-based | RL/VLA **policy runner** (exported weights), residual control | interop | `burn`/`candle` on WebGPU; ties to our checkpoint + in-browser VLA |

## Status: the corpus backbone is shipped
Shipped + verified (31 tests, WASM-clean): PID · computed-torque · Cartesian impedance · LQR · linear MPC ·
OSC · WBC · **iLQR/DDP** · **MPPI** · **CBF-QP** · **sliding-mode** · **admittance + hybrid force/position** ·
**capture-point + ZMP preview** · **Kalman + EKF + UKF**. (The last 7 modules authored in parallel via a
multi-agent workflow, then integrated + verified in the main loop.)

Honest note on MPPI: it's an approximate sampling controller — verified as *drives each joint to its goal* +
deterministic; tight settling is tuning-specific (and it's the natural first heavy GPU kernel — batched
`forward_dynamics` rollouts, the fleet path that earns the GPU speedup).

## Depth shipped (2nd workflow, integrated + verified)
**H∞** state-feedback (ARE + γ-bisection; closed-loop attenuates disturbance below γ) · **centroidal MPC**
(LIPM cart-table walking; ZMP stays in the support polygon) · **generalized-momentum observer** (recovers an
unknown external joint torque) · **complementary filter** (fuses biased gyro + noisy accel, beats both) ·
**CEM** (cross-entropy sampling MPC; elite cost monotonically decreases). Plus ports: **Pink/mink** parity
(`pink`, 6-DoF pose + posture + limit tasks), **dex-retargeting** (position/vector/DexPilot), **placo** (hard-limit
task-space QP). (TRAC-IK robust-restart IK + KDL resolved-rate were already shipped.)

**GPU-MPPI shipped:** batched rollouts as a WebGPU kernel — **26× over CPU at 16k rollouts (steady state)**,
matches CPU to 7e-7 (`gpu/mppi.html`). The first heavy kernel that earns the GPU.

## Learned control (interop) — shipped: `ferromotion-policy`
On-device runner for exported learned policies: a pure-Rust MLP inference engine (obs normalization +
tanh-squash + action scaling) with a JSON checkpoint loader. We **run** trained weights; we don't rewrite
training. Verified: forward pass matches hand-computation, JSON round-trips, and a policy executed by the runner
regulates both a double integrator and the ARM2 arm via `forward_dynamics` (plugs into the ecosystem). Small
enough to run in the browser — the same on-device path as our released checkpoint + in-browser VLA. Large
transformer VLAs stay an ONNX/`candle` concern; this covers the MLP-policy case that spans most RL control.

## What's left
- Niche extensions: iSAM2-style incremental estimation, whole-body centroidal MPC on the full robot (vs LIPM),
  contact-implicit trajectory optimization; ONNX/`candle` path for large transformer VLAs.
- Non-code: wire onto the site / into ElRobot; publish the crates.
