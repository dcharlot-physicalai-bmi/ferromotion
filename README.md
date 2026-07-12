# Ferromotion — the Rust library for the kinematics, dynamics & control of physical AI

A native-Rust + universal-WASM port of the ideas in **PyRoki** (Kim, Yi, et al., IROS 2025,
MIT) — a modular toolkit for **robot kinematic optimization**: inverse kinematics, trajectory
optimization, and motion retargeting expressed as one composable nonlinear-least-squares problem.

Working name `ferromotion`; final naming is the Dean's call.

## Why Rust + WASM
PyRoki is Python + JAX (CPU/GPU/TPU, server-side). The Institute runs **in-browser, on-device**
(WebGPU/WASM — Forge, the labs, the Rust course). A Rust core with a `wasm32` build gives us the
same kinematics **everywhere**: native (fast, GPU later), and in every browser with zero install —
the universal variant no one else in this niche has.

## PyRoki → Rust mapping
| PyRoki (Python/JAX) | ferromotion (Rust) | notes |
|---|---|---|
| `jaxlie` (SE3/SO3) | `ferromotion-lie` | quaternion SO3 + SE3, exp/log, tangent residuals |
| `yourdfpy` (URDF) | `ferromotion-urdf` (wrap `urdf-rs`) | URDF → kinematic tree |
| `Robot` FK + autodiff Jac | `ferromotion-robot` | FK; **analytic geometric Jacobian** (fast, exact) + optional forward-mode autodiff (`num-dual`) for arbitrary costs |
| collision primitives (sphere/capsule/half-space) | `ferromotion-collide` | closed-form signed distance + gradients |
| costs (EE pose, collision, manipulability, limits, smoothness) | `ferromotion-costs` | composable `Cost` trait, decoupled from variables |
| `jaxls` LM / augmented-Lagrangian | `ferromotion-solve` | LM nonlinear-least-squares; dense v1 → **block-sparse** (temporal) + Aug-Lagrangian for hard constraints |
| `viser` viz | (n/a) | our labs render in WebGPU already |
| — | `ferromotion-wasm` | `wasm-bindgen` JS/TS API for the browser labs + Forge |

Linear algebra: **`nalgebra`** (pure Rust, WASM-clean, no BLAS). Autodiff for arbitrary costs:
forward-mode dual numbers (low-DoF problems → dense is fine); analytic Jacobians for the hot paths.

## Roadmap
- **M0 — vertical slice (this pass):** revolute chain → FK → analytic Jacobian → LM IK → converges;
  compiles to `wasm32-unknown-unknown`. Proves the whole spine.
- **M1 — real robots:** `urdf-rs` loader; all joint types; Panda/SO-101/humanoid URDFs; pose (6-DoF) cost.
- **M2 — cost library + collision:** capsule/sphere self- & world-collision, manipulability, joint limits,
  smoothness; composable `Cost` trait; augmented-Lagrangian hard constraints.
- **M3 — trajectories & retargeting:** time-series variables, block-sparse LM; human→robot motion retargeting.
- **M4 — WASM API + labs:** `ferromotion-wasm` bindings, a TS wrapper, Web-Worker solving; wire into the arm/pilot
  labs and Forge (target-pose IK, live retargeting on-device).
- **M5 — parity + benches:** benchmark vs PyRoki/cuRobo; GPU (wgpu) path for batched solves.

## Layout
```
ferromotion/
  crates/
    ferromotion-core/     # M0: lie + robot + solve + ik (single crate; splits into ferromotion-lie/-robot/-solve later)
    ferromotion-urdf/     # M1
    ferromotion-costs/    # M2
    ferromotion-collide/  # M2
    ferromotion-wasm/     # M4
  examples/
```
Graduates to its own repo (`dcharlot-physicalai-bmi/ferromotion`) once M1 lands.
