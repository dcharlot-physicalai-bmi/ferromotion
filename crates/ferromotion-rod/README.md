# ferromotion-rod

[![crates.io](https://img.shields.io/crates/v/ferromotion-rod.svg)](https://crates.io/crates/ferromotion-rod)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-rod)](https://docs.rs/ferromotion-rod)

**Discrete Elastic Rods** (Bergou et al., SIGGRAPH 2008) for cables, tendons, and continuum/soft
robots — the 1D-deformable companion in [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion),
alongside the cloth (2D) and MPM (volumetric) solvers. Pure Rust (native + `wasm32`).

A rod is a polyline centerline. Two elastic energies act on it: **stretch** (edge springs to rest
length) and **isotropic bending** on the discrete curvature binormal
`κb_i = 2·e_{i-1}×e_i / (‖e_{i-1}‖‖e_i‖ + e_{i-1}·e_i)`, with energy `E_i = (EI / 2ℓ_i)‖κb_i‖²` — the
discretization that reproduces the continuum `½∫EI κ² ds`. Nodal forces are the **exact analytic
gradient** of the energy (verified against finite differences), so the rod is differentiable.

Validated against the analytic **Euler-Bernoulli cantilever** (tip deflection within ~4 %), with
energy conserved under free vibration.

```rust
use ferromotion_rod::Rod;
use nalgebra::Vector3;

let mut rod = Rod::straight(20, 1.0, 0.001, 200.0, 0.5, Vector3::new(0.0, 0.0, -9.81));
let residual = rod.relax(200_000, 1e-4); // sag to static equilibrium
let tip_deflection = -rod.x.last().unwrap().z;
```

Twist via material frames is a further extension. Dual-licensed MIT OR Apache-2.0.
